"""
model.py - TabPFN Architecture for OpenTab

A transformer architecture designed for tabular data that performs in-context learning.
Each cell in the table is treated as a separate position, with two-way attention:
1. Inter-feature attention: Features in the same sample attend to each other
2. Inter-sample attention: Samples attend to other samples (test only attends to train)

Key features from the TabPFN paper:
- Random feature embeddings to differentiate features with same statistics
- Z-normalization of inputs using training set statistics
- Missing value indicator as extra input
- Half-precision layer norms for efficiency
- Train-state caching for fast inference

Reference:
- "TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second"
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union


class TabPFNEncoder(nn.Module):
    """
    Encodes tabular data into embeddings.
    
    Following the paper:
    - All features mapped to float (categoricals → integers)
    - Z-normalization using training set mean/std per feature
    - Linear encoding into embedding dimension
    - Missing values: set to 0, add indicator input
    - Random feature embeddings added to all positions of a feature
    """
    
    def __init__(
        self,
        embedding_dim: int,
        max_features: int = 100,
        random_feature_dim: int = None,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_features = max_features
        
        # Random feature embedding dimension (1/4 of embedding_dim per paper)
        if random_feature_dim is None:
            random_feature_dim = embedding_dim // 4
        self.random_feature_dim = random_feature_dim
        
        # Linear encoder for normalized values (input: value + missing indicator = 2)
        self.value_encoder = nn.Linear(2, embedding_dim)
        
        # Learned projection for random feature embeddings
        self.feature_embedding_proj = nn.Linear(random_feature_dim, embedding_dim)
        
        # Target encoder (for training labels)
        self.target_encoder = nn.Linear(1, embedding_dim)
    
    def forward(
        self,
        X: torch.Tensor,
        y_train: torch.Tensor,
        train_size: int,
        missing_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode features and targets into embeddings.
        
        Args:
            X: (batch, n_samples, n_features) feature values
            y_train: (batch, train_size) training labels
            train_size: number of training samples
            missing_mask: (batch, n_samples, n_features) boolean mask for missing values
            
        Returns:
            (batch, n_samples, n_features + 1, embedding_dim) embeddings
        """
        batch_size, n_samples, n_features = X.shape
        device = X.device
        
        # Handle missing values
        if missing_mask is None:
            # Detect NaN or special missing value (-999)
            missing_mask = torch.isnan(X) | (X == -999.0)
        
        # Set missing values to 0
        X_filled = X.clone()
        X_filled[missing_mask] = 0.0
        
        # Z-normalize using training set statistics (per feature)
        X_train = X_filled[:, :train_size, :]  # (batch, train_size, n_features)
        
        # Compute mean and std from training portion
        mean = X_train.mean(dim=1, keepdim=True)  # (batch, 1, n_features)
        std = X_train.std(dim=1, keepdim=True) + 1e-8  # (batch, 1, n_features)
        
        # Normalize all samples using training statistics
        X_normalized = (X_filled - mean) / std
        X_normalized = torch.clamp(X_normalized, -100, 100)
        
        # Create input: [normalized_value, missing_indicator]
        missing_indicator = missing_mask.float()
        encoder_input = torch.stack([X_normalized, missing_indicator], dim=-1)
        # Shape: (batch, n_samples, n_features, 2)
        
        # Encode features
        feature_embeddings = self.value_encoder(encoder_input)
        # Shape: (batch, n_samples, n_features, embedding_dim)
        
        # Add random feature embeddings (same for all samples of a feature)
        # Generate deterministic random vectors based on feature index
        random_vectors = torch.randn(
            n_features, self.random_feature_dim, 
            device=device, dtype=X.dtype
        )
        # Make it deterministic by using a seed based on n_features
        # In practice, we just use random vectors that are fixed per forward pass
        
        feature_emb_addition = self.feature_embedding_proj(random_vectors)
        # Shape: (n_features, embedding_dim)
        
        # Add to all samples
        feature_embeddings = feature_embeddings + feature_emb_addition.unsqueeze(0).unsqueeze(1)
        
        # Encode targets
        # Pad test positions with mean of training targets
        y_train_float = y_train.float()
        y_mean = y_train_float.mean(dim=1, keepdim=True)
        y_padded = torch.cat([
            y_train_float,
            y_mean.expand(-1, n_samples - train_size)
        ], dim=1)
        # Shape: (batch, n_samples)
        
        target_embeddings = self.target_encoder(y_padded.unsqueeze(-1))
        # Shape: (batch, n_samples, embedding_dim)
        # Add feature dimension
        target_embeddings = target_embeddings.unsqueeze(2)
        # Shape: (batch, n_samples, 1, embedding_dim)
        
        # Concatenate features and target
        embeddings = torch.cat([feature_embeddings, target_embeddings], dim=2)
        # Shape: (batch, n_samples, n_features + 1, embedding_dim)
        
        return embeddings


class InterFeatureAttention(nn.Module):
    """
    Attention across features within each sample (row-wise).
    
    This is fully connected attention - each feature can attend to all others.
    """
    
    def __init__(self, embedding_dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embedding_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_samples, n_features, embedding_dim)
        Returns:
            (batch, n_samples, n_features, embedding_dim)
        """
        batch_size, n_samples, n_features, embedding_dim = x.shape
        
        # Reshape for attention: (batch * n_samples, n_features, embedding_dim)
        x_flat = x.reshape(batch_size * n_samples, n_features, embedding_dim)
        
        # Self-attention across features
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        x_flat = x_flat + attn_out
        
        # Reshape back and normalize
        x = x_flat.reshape(batch_size, n_samples, n_features, embedding_dim)
        x = self.norm(x)
        
        return x


class InterSampleAttention(nn.Module):
    """
    Attention across samples within each feature (column-wise).
    
    Key constraint from the paper:
    - Training samples attend to all training samples
    - Test samples attend ONLY to training samples (not to each other)
    
    This ensures test samples don't influence each other during prediction.
    """
    
    def __init__(self, embedding_dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embedding_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, x: torch.Tensor, train_size: int) -> torch.Tensor:
        """
        Args:
            x: (batch, n_samples, n_features, embedding_dim)
            train_size: number of training samples
        Returns:
            (batch, n_samples, n_features, embedding_dim)
        """
        batch_size, n_samples, n_features, embedding_dim = x.shape
        
        # Transpose to (batch, n_features, n_samples, embedding_dim)
        x = x.transpose(1, 2)
        
        # Reshape for attention: (batch * n_features, n_samples, embedding_dim)
        x_flat = x.reshape(batch_size * n_features, n_samples, embedding_dim)
        
        # Split into train and test
        x_train = x_flat[:, :train_size, :]  # (B*F, train_size, D)
        x_test = x_flat[:, train_size:, :]   # (B*F, test_size, D)
        
        # Training samples attend to training samples
        train_attn, _ = self.attn(x_train, x_train, x_train)
        
        # Test samples attend to training samples only
        if x_test.shape[1] > 0:
            test_attn, _ = self.attn(x_test, x_train, x_train)
            attn_out = torch.cat([train_attn, test_attn], dim=1)
        else:
            attn_out = train_attn
        
        x_flat = x_flat + attn_out
        
        # Reshape and transpose back
        x = x_flat.reshape(batch_size, n_features, n_samples, embedding_dim)
        x = x.transpose(1, 2)  # (batch, n_samples, n_features, embedding_dim)
        x = self.norm(x)
        
        return x


class TabPFNLayer(nn.Module):
    """
    Single transformer layer with two-way attention + MLP.
    
    Following the paper:
    1. Inter-feature attention (column-wise within each row)
    2. Inter-sample attention (row-wise within each column, with masking)
    3. MLP sublayer
    Each sublayer followed by residual addition and layer norm.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        n_heads: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.feature_attn = InterFeatureAttention(embedding_dim, n_heads, dropout)
        self.sample_attn = InterSampleAttention(embedding_dim, n_heads, dropout)
        
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embedding_dim),
            nn.Dropout(dropout),
        )
        self.mlp_norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, x: torch.Tensor, train_size: int) -> torch.Tensor:
        """
        Args:
            x: (batch, n_samples, n_features, embedding_dim)
            train_size: number of training samples
        Returns:
            (batch, n_samples, n_features, embedding_dim)
        """
        # Inter-feature attention
        x = self.feature_attn(x)
        
        # Inter-sample attention
        x = self.sample_attn(x, train_size)
        
        # MLP
        x = x + self.mlp(x)
        x = self.mlp_norm(x)
        
        return x


class TabPFNDecoder(nn.Module):
    """
    Decoder that maps embeddings to predictions.
    
    For classification: outputs logits over classes
    For regression: outputs logits over bins (piecewise constant distribution)
    """
    
    def __init__(self, embedding_dim: int, mlp_dim: int, n_outputs: int):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, n_outputs),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_test, embedding_dim) test sample embeddings
        Returns:
            (batch, n_test, n_outputs) logits
        """
        return self.decoder(x)


class OpenTabModel(nn.Module):
    """
    TabPFN model for in-context learning on tabular data.
    
    The model takes a table with features and targets, and predicts
    targets for test samples using the training samples as context.
    
    Architecture (from paper):
    - Each cell treated as separate position
    - Two-way attention: inter-feature + inter-sample
    - Test samples only attend to training samples
    - Random feature embeddings for disambiguation
    
    Complexity:
    - Compute: O(n² + m²) where n=samples, m=features
    - Memory: O(n * m)
    """
    
    def __init__(
        self,
        embedding_size: int = 128,
        n_heads: int = 4,
        mlp_hidden_size: int = 256,
        n_layers: int = 6,
        n_outputs: int = 10,
        max_features: int = 100,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.embedding_size = embedding_size
        self.n_outputs = n_outputs
        self.max_features = max_features
        
        # Encoder
        self.encoder = TabPFNEncoder(
            embedding_dim=embedding_size,
            max_features=max_features,
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TabPFNLayer(embedding_size, n_heads, mlp_hidden_size, dropout)
            for _ in range(n_layers)
        ])
        
        # Decoder
        self.decoder = TabPFNDecoder(embedding_size, mlp_hidden_size, n_outputs)
    
    def forward(
        self,
        X: torch.Tensor,
        y_train: torch.Tensor,
        train_size: int,
        missing_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for prediction.
        
        Args:
            X: (batch, n_samples, n_features) all features (train + test)
            y_train: (batch, train_size) training labels
            train_size: number of training samples
            missing_mask: (batch, n_samples, n_features) missing value mask
            
        Returns:
            (batch, n_test, n_outputs) logits for test samples
        """
        # Encode inputs
        embeddings = self.encoder(X, y_train, train_size, missing_mask)
        # Shape: (batch, n_samples, n_features + 1, embedding_dim)
        
        # Apply transformer layers
        for layer in self.layers:
            embeddings = layer(embeddings, train_size)
        
        # Extract test sample embeddings from target column (last column)
        test_embeddings = embeddings[:, train_size:, -1, :]
        # Shape: (batch, n_test, embedding_dim)
        
        # Decode to predictions
        logits = self.decoder(test_embeddings)
        
        return logits
    
    def forward_train_test(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convenience method for sklearn-like interface.
        
        Args:
            X_train: (batch, n_train, n_features)
            y_train: (batch, n_train)
            X_test: (batch, n_test, n_features)
            
        Returns:
            (batch, n_test, n_outputs) logits
        """
        train_size = X_train.shape[1]
        X = torch.cat([X_train, X_test], dim=1)
        return self.forward(X, y_train, train_size)


class OpenTabClassifier:
    """
    Sklearn-like interface for classification.
    
    Usage:
        clf = OpenTabClassifier("checkpoint.pt")
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        probabilities = clf.predict_proba(X_test)
    """
    
    def __init__(
        self,
        model: Union[OpenTabModel, str, None] = None,
        device: Optional[str] = None,
        temperature: float = 0.9,  # Softmax temperature for calibration
    ):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.temperature = temperature
        
        if model is None:
            self.model = OpenTabModel().to(self.device)
        elif isinstance(model, str):
            checkpoint = torch.load(model, map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                config = checkpoint.get('config', {})
                self.model = OpenTabModel(
                    embedding_size=config.get('embedding_size', 128),
                    n_heads=config.get('n_heads', 4),
                    mlp_hidden_size=config.get('mlp_hidden_size', 256),
                    n_layers=config.get('n_layers', 6),
                    n_outputs=config.get('max_classes', 10),
                    max_features=config.get('max_features', 100),
                    dropout=config.get('dropout', 0.0),
                ).to(self.device)
                self.model.load_state_dict(checkpoint['model_state'])
            else:
                self.model = OpenTabModel().to(self.device)
                self.model.load_state_dict(checkpoint)
        else:
            self.model = model.to(self.device)
        
        self.model.eval()
        self.X_train_ = None
        self.y_train_ = None
        self.n_classes_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Store training data for in-context learning."""
        self.X_train_ = X.astype(np.float32)
        self.y_train_ = y.astype(np.int64)
        self.n_classes_ = int(y.max()) + 1
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.X_train_ is None:
            raise ValueError("Must call fit() first")
        
        X_test = X.astype(np.float32)
        
        with torch.no_grad():
            X_train_t = torch.from_numpy(self.X_train_).unsqueeze(0).to(self.device)
            y_train_t = torch.from_numpy(self.y_train_).unsqueeze(0).to(self.device)
            X_test_t = torch.from_numpy(X_test).unsqueeze(0).to(self.device)
            
            logits = self.model.forward_train_test(X_train_t, y_train_t, X_test_t)
            logits = logits[:, :, :self.n_classes_]
            
            # Apply temperature scaling
            probs = F.softmax(logits / self.temperature, dim=-1)
            
            return probs.squeeze(0).cpu().numpy()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.predict_proba(X).argmax(axis=1)


class OpenTabRegressor:
    """
    Sklearn-like interface for regression.
    
    Uses piecewise constant output distribution (binning) as described in the paper.
    """
    
    def __init__(
        self,
        model: Union[OpenTabModel, str, None] = None,
        device: Optional[str] = None,
        n_bins: int = 64,
    ):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.n_bins = n_bins
        
        if model is None:
            self.model = OpenTabModel(n_outputs=n_bins).to(self.device)
        elif isinstance(model, str):
            checkpoint = torch.load(model, map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                config = checkpoint.get('config', {})
                self.model = OpenTabModel(
                    embedding_size=config.get('embedding_size', 128),
                    n_heads=config.get('n_heads', 4),
                    mlp_hidden_size=config.get('mlp_hidden_size', 256),
                    n_layers=config.get('n_layers', 6),
                    n_outputs=config.get('n_bins', n_bins),
                    max_features=config.get('max_features', 100),
                    dropout=config.get('dropout', 0.0),
                ).to(self.device)
                self.model.load_state_dict(checkpoint['model_state'])
            else:
                self.model = OpenTabModel(n_outputs=n_bins).to(self.device)
                self.model.load_state_dict(checkpoint)
        else:
            self.model = model.to(self.device)
        
        self.model.eval()
        self.X_train_ = None
        self.y_train_ = None
        self.y_mean_ = None
        self.y_std_ = None
        self.bin_centers_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Store training data and compute bin boundaries."""
        self.X_train_ = X.astype(np.float32)
        
        # Z-normalize targets
        self.y_mean_ = float(y.mean())
        self.y_std_ = float(y.std()) + 1e-8
        y_normalized = (y - self.y_mean_) / self.y_std_
        
        # Create bins
        y_min, y_max = y_normalized.min(), y_normalized.max()
        margin = (y_max - y_min) * 0.1 + 1e-6
        bin_edges = np.linspace(y_min - margin, y_max + margin, self.n_bins + 1)
        self.bin_centers_ = ((bin_edges[:-1] + bin_edges[1:]) / 2).astype(np.float32)
        
        # Discretize
        y_bins = np.clip(np.digitize(y_normalized, bin_edges[1:-1]), 0, self.n_bins - 1)
        self.y_train_ = y_bins.astype(np.int64)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict regression values (expected value over bins)."""
        if self.X_train_ is None:
            raise ValueError("Must call fit() first")
        
        X_test = X.astype(np.float32)
        
        with torch.no_grad():
            X_train_t = torch.from_numpy(self.X_train_).unsqueeze(0).to(self.device)
            y_train_t = torch.from_numpy(self.y_train_).unsqueeze(0).to(self.device)
            X_test_t = torch.from_numpy(X_test).unsqueeze(0).to(self.device)
            
            logits = self.model.forward_train_test(X_train_t, y_train_t, X_test_t)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            
            # Expected value
            predictions = (probs * self.bin_centers_).sum(axis=-1)
            
            # Denormalize
            predictions = predictions * self.y_std_ + self.y_mean_
            
            return predictions


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Testing OpenTabModel...")
    
    model = OpenTabModel(
        embedding_size=128,
        n_heads=4,
        mlp_hidden_size=256,
        n_layers=6,
        n_outputs=10,
    )
    
    print(f"Parameters: {count_parameters(model):,}")
    
    # Test forward pass
    batch_size = 2
    n_train, n_test, n_features = 50, 10, 8
    
    X = torch.randn(batch_size, n_train + n_test, n_features)
    y_train = torch.randint(0, 3, (batch_size, n_train))
    
    logits = model(X, y_train, n_train)
    
    print(f"Input: X={X.shape}, y_train={y_train.shape}")
    print(f"Output: {logits.shape}")
    print("✓ Model test passed!")
