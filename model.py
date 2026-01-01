"""
model.py - OpenTab Transformer Architecture

A implementation of the TabPFN (Prior-Data Fitted Network) architecture.
The model processes tabular data as a 3D tensor: (batch, rows, features+target)
and applies attention across both dimensions.

Architecture Overview:
1. FeatureEncoder: Embed each feature value into a dense vector
2. TargetEncoder: Embed target values (with padding for test samples)
3. TransformerEncoder: Apply attention across features and samples
4. Decoder: Map embeddings to class predictions

References:
- TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second
- nanoTabPFN: A Lightweight and Educational Reimplementation of TabPFN
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union


# Special value used to indicate missing during training/inference
MISSING_INDICATOR = -999.0


class FeatureEncoder(nn.Module):
    """Encodes scalar feature values into dense embeddings.
    
    Each feature value is normalized based on training statistics,
    then projected to an embedding space.
    
    Following TabPFN, this encoder handles missing values by:
    1. Detecting missing indicator values (-999.0)
    2. Using a learned embedding for missing values
    3. Adding a missing indicator embedding to the feature embedding
    """
    
    def __init__(self, embedding_size: int):
        super().__init__()
        self.embedding_size = embedding_size
        self.linear = nn.Linear(1, embedding_size)
        
        # Learned embedding for missing values (replaces the feature embedding)
        self.missing_embedding = nn.Parameter(torch.randn(embedding_size) * 0.02)
        
        # Learned indicator that a value is missing (added to embedding)
        self.missing_indicator_embedding = nn.Parameter(torch.randn(embedding_size) * 0.02)
    
    def forward(
        self, 
        x: torch.Tensor, 
        train_size: int,
        missing_indicator: float = MISSING_INDICATOR,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, rows, features) tensor of feature values
            train_size: number of training samples (for normalization)
            missing_indicator: value used to mark missing data
            
        Returns:
            (batch, rows, features, embedding_size) tensor of embeddings
        """
        batch, rows, features = x.shape
        
        # Detect missing values (using approximate comparison for float)
        missing_mask = (x < missing_indicator + 1) & (x > missing_indicator - 1)
        
        # Replace missing values with 0 for normalization computation
        x_filled = x.clone()
        x_filled[missing_mask] = 0.0
        
        x_filled = x_filled.unsqueeze(-1)  # (batch, rows, features, 1)
        
        # Compute mean and std from training portion only, excluding missing
        train_x = x[:, :train_size]
        train_missing_mask = missing_mask[:, :train_size]
        
        # Mask out missing values for statistics computation
        train_x_masked = train_x.clone()
        train_x_masked[train_missing_mask] = float('nan')
        
        # Compute stats ignoring NaN (missing values)
        mean = torch.nanmean(train_x_masked, dim=1, keepdim=True)
        
        # Handle case where all values are missing
        mean = torch.where(torch.isnan(mean), torch.zeros_like(mean), mean)
        
        # Compute std (handling missing values)
        train_x_centered = train_x_masked - mean
        train_x_centered[train_missing_mask] = 0.0
        n_valid = (~train_missing_mask).float().sum(dim=1, keepdim=True).clamp(min=1)
        var = (train_x_centered ** 2).sum(dim=1, keepdim=True) / n_valid
        std = torch.sqrt(var + 1e-8)
        
        # Normalize all data (train + test) using training statistics
        x_normalized = (x_filled - mean.unsqueeze(-1)) / std.unsqueeze(-1)
        x_normalized = torch.clamp(x_normalized, -100, 100)
        
        # Get base embeddings
        embeddings = self.linear(x_normalized)  # (batch, rows, features, embedding_size)
        
        # Apply missing value handling
        # For missing values, replace with learned missing embedding
        missing_mask_expanded = missing_mask.unsqueeze(-1).expand(-1, -1, -1, self.embedding_size)
        embeddings = torch.where(
            missing_mask_expanded,
            self.missing_embedding.view(1, 1, 1, -1).expand(batch, rows, features, -1),
            embeddings
        )
        
        # Add missing indicator to all positions that were missing
        # This helps the model learn that missingness itself is informative
        embeddings = embeddings + missing_mask_expanded.float() * self.missing_indicator_embedding.view(1, 1, 1, -1)
        
        return embeddings


class TargetEncoder(nn.Module):
    """Encodes target values into dense embeddings.
    
    Training targets are embedded directly. Test positions are padded
    with the mean target value (since we don't know the true labels).
    """
    
    def __init__(self, embedding_size: int):
        super().__init__()
        self.linear = nn.Linear(1, embedding_size)
    
    def forward(self, y_train: torch.Tensor, total_rows: int) -> torch.Tensor:
        """
        Args:
            y_train: (batch, train_size) tensor of training labels
            total_rows: total number of rows (train + test)
            
        Returns:
            (batch, total_rows, 1, embedding_size) tensor of embeddings
        """
        # Convert to float for embedding
        y_train = y_train.float()
        
        # Add feature dimension
        if y_train.dim() == 2:
            y_train = y_train.unsqueeze(-1)
        
        # Pad test positions with mean of training targets
        train_size = y_train.shape[1]
        test_size = total_rows - train_size
        
        if test_size > 0:
            mean_y = y_train.mean(dim=1, keepdim=True)
            padding = mean_y.expand(-1, test_size, -1)
            y_full = torch.cat([y_train, padding], dim=1)
        else:
            y_full = y_train
        
        # Embed: (batch, rows, 1) -> (batch, rows, 1, embedding_size)
        return self.linear(y_full.unsqueeze(-1))


class TransformerEncoderLayer(nn.Module):
    """A single transformer layer with attention over both features and samples.
    
    Unlike standard transformers, TabPFN applies attention in two directions:
    1. Across features (column-wise): features can interact
    2. Across samples (row-wise): with causal masking for train→test
    
    The attention pattern ensures test samples can attend to training samples
    but not to other test samples (to prevent data leakage).
    """
    
    def __init__(
        self,
        embedding_size: int,
        n_heads: int,
        mlp_hidden_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        # Attention across features (columns)
        self.attn_features = nn.MultiheadAttention(
            embedding_size, n_heads, batch_first=True, dropout=dropout
        )
        
        # Attention across samples (rows)
        self.attn_samples = nn.MultiheadAttention(
            embedding_size, n_heads, batch_first=True, dropout=dropout
        )
        
        # MLP block
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, mlp_hidden_size),
            nn.GELU(),
            nn.Linear(mlp_hidden_size, embedding_size),
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.norm3 = nn.LayerNorm(embedding_size)
    
    def forward(self, x: torch.Tensor, train_size: int) -> torch.Tensor:
        """
        Args:
            x: (batch, rows, cols, embedding_size) tensor
            train_size: number of training samples (for attention masking)
            
        Returns:
            (batch, rows, cols, embedding_size) tensor
        """
        batch_size, n_rows, n_cols, emb_size = x.shape
        
        # 1. Attention across features (for each row)
        # Reshape: (batch * rows, cols, emb)
        x_flat = x.reshape(batch_size * n_rows, n_cols, emb_size)
        attn_out, _ = self.attn_features(x_flat, x_flat, x_flat)
        x_flat = x_flat + attn_out
        x = x_flat.reshape(batch_size, n_rows, n_cols, emb_size)
        x = self.norm1(x)
        
        # 2. Attention across samples (for each feature)
        # Reshape: (batch * cols, rows, emb)
        x = x.transpose(1, 2)  # (batch, cols, rows, emb)
        x_flat = x.reshape(batch_size * n_cols, n_rows, emb_size)
        
        # Split into train and test portions
        x_train = x_flat[:, :train_size]
        x_test = x_flat[:, train_size:]
        
        # Training samples attend to themselves
        train_attn, _ = self.attn_samples(x_train, x_train, x_train)
        
        # Test samples attend to training samples only
        if x_test.shape[1] > 0:
            test_attn, _ = self.attn_samples(x_test, x_train, x_train)
            attn_out = torch.cat([train_attn, test_attn], dim=1)
        else:
            attn_out = train_attn
        
        x_flat = x_flat + attn_out
        x = x_flat.reshape(batch_size, n_cols, n_rows, emb_size)
        x = x.transpose(1, 2)  # Back to (batch, rows, cols, emb)
        x = self.norm2(x)
        
        # 3. MLP
        x = x + self.mlp(x)
        x = self.norm3(x)
        
        return x


class TransformerEncoder(nn.Module):
    """Stack of transformer encoder layers."""
    
    def __init__(
        self,
        embedding_size: int,
        n_heads: int,
        mlp_hidden_size: int,
        n_layers: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embedding_size, n_heads, mlp_hidden_size, dropout)
            for _ in range(n_layers)
        ])
    
    def forward(self, x: torch.Tensor, train_size: int) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, train_size)
        return x


class Decoder(nn.Module):
    """Maps embeddings to output predictions."""
    
    def __init__(self, embedding_size: int, mlp_hidden_size: int, n_outputs: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, mlp_hidden_size),
            nn.GELU(),
            nn.Linear(mlp_hidden_size, n_outputs),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_test, embedding_size) tensor
            
        Returns:
            (batch, n_test, n_outputs) tensor of logits
        """
        return self.mlp(x)


class OpenTabModel(nn.Module):
    """
    OpenTab: A implementation of TabPFN for tabular classification.
    
    The model takes a table of (features, target) and predicts targets for
    test samples using in-context learning.
    
    Args:
        embedding_size: Dimension of embeddings
        n_heads: Number of attention heads
        mlp_hidden_size: Hidden size of MLP blocks
        n_layers: Number of transformer layers
        n_outputs: Number of output classes
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        embedding_size: int = 96,
        n_heads: int = 4,
        mlp_hidden_size: int = 192,
        n_layers: int = 3,
        n_outputs: int = 10,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.embedding_size = embedding_size
        self.n_outputs = n_outputs
        
        self.feature_encoder = FeatureEncoder(embedding_size)
        self.target_encoder = TargetEncoder(embedding_size)
        self.transformer = TransformerEncoder(
            embedding_size, n_heads, mlp_hidden_size, n_layers, dropout
        )
        self.decoder = Decoder(embedding_size, mlp_hidden_size, n_outputs)
    
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        train_size: int,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, n_rows, n_features) tensor of features
            y: (batch, train_size) tensor of training labels
            train_size: number of training samples
            
        Returns:
            (batch, n_test, n_outputs) tensor of logits for test samples
        """
        n_rows = x.shape[1]
        
        # Encode features: (batch, rows, features, emb)
        x_emb = self.feature_encoder(x, train_size)
        
        # Encode targets: (batch, rows, 1, emb)
        y_emb = self.target_encoder(y, n_rows)
        
        # Concatenate features and target: (batch, rows, features+1, emb)
        combined = torch.cat([x_emb, y_emb], dim=2)
        
        # Apply transformer
        transformed = self.transformer(combined, train_size)
        
        # Extract test embeddings from target column
        # Shape: (batch, n_test, emb)
        test_emb = transformed[:, train_size:, -1, :]
        
        # Decode to predictions
        logits = self.decoder(test_emb)
        
        return logits
    
    def forward_train_test(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
    ) -> torch.Tensor:
        """Convenience method matching sklearn interface.
        
        Args:
            X_train: (batch, n_train, n_features) training features
            y_train: (batch, n_train) training labels
            X_test: (batch, n_test, n_features) test features
            
        Returns:
            (batch, n_test, n_outputs) tensor of logits
        """
        train_size = X_train.shape[1]
        x = torch.cat([X_train, X_test], dim=1)
        return self.forward(x, y_train, train_size)


class OpenTabClassifier:
    """Sklearn-like interface for OpenTab classification.
    
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
    ):
        """
        Args:
            model: Either a OpenTabModel instance or path to checkpoint
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        
        if model is None:
            # Create default model
            self.model = OpenTabModel().to(self.device)
        elif isinstance(model, str):
            # Load from checkpoint (weights_only=False for numpy arrays in config)
            checkpoint = torch.load(model, map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                # Full checkpoint from train.py
                config = checkpoint.get('config', {})
                self.model = OpenTabModel(
                    embedding_size=config.get('embedding_size', 96),
                    n_heads=config.get('n_heads', 4),
                    mlp_hidden_size=config.get('mlp_hidden', 192),
                    n_layers=config.get('n_layers', 3),
                    n_outputs=config.get('max_classes', 10),
                    dropout=config.get('dropout', 0.0),
                ).to(self.device)
                self.model.load_state_dict(checkpoint['model_state'])
            elif isinstance(checkpoint, dict) and 'model' in checkpoint:
                # Alternative checkpoint format
                self.model = OpenTabModel(
                    **checkpoint.get('architecture', {})
                ).to(self.device)
                self.model.load_state_dict(checkpoint['model'])
            else:
                # Just state dict
                self.model = OpenTabModel().to(self.device)
                self.model.load_state_dict(checkpoint)
        else:
            self.model = model.to(self.device)
        
        self.model.eval()
        
        # Storage for fit data
        self.X_train_ = None
        self.y_train_ = None
        self.n_classes_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Store training data for later prediction.
        
        Note: TabPFN doesn't actually "train" during fit - it just stores
        the data for use during prediction (in-context learning).
        """
        self.X_train_ = X.astype(np.float32)
        self.y_train_ = y.astype(np.int64)
        self.n_classes_ = int(y.max()) + 1
        return self
    
    def predict_proba(self, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """Predict class probabilities for test samples.
        
        Args:
            X: Test samples array
            batch_size: Number of test samples to process at once (to avoid OOM)
        """
        if self.X_train_ is None:
            raise ValueError("Must call fit() before predict()")
        
        X_test = X.astype(np.float32)
        n_test = X_test.shape[0]
        
        # Process in batches to avoid memory issues
        if n_test <= batch_size:
            # Small enough to process at once
            with torch.no_grad():
                X_train_t = torch.from_numpy(self.X_train_).unsqueeze(0).to(self.device)
                y_train_t = torch.from_numpy(self.y_train_).float().unsqueeze(0).to(self.device)
                X_test_t = torch.from_numpy(X_test).unsqueeze(0).to(self.device)
                
                logits = self.model.forward_train_test(X_train_t, y_train_t, X_test_t)
                logits = logits[:, :, :self.n_classes_]
                probs = F.softmax(logits, dim=-1)
                
                return probs.squeeze(0).cpu().numpy()
        else:
            # Process in batches
            all_probs = []
            with torch.no_grad():
                X_train_t = torch.from_numpy(self.X_train_).unsqueeze(0).to(self.device)
                y_train_t = torch.from_numpy(self.y_train_).float().unsqueeze(0).to(self.device)
                
                for i in range(0, n_test, batch_size):
                    batch_end = min(i + batch_size, n_test)
                    X_batch = X_test[i:batch_end]
                    X_test_t = torch.from_numpy(X_batch).unsqueeze(0).to(self.device)
                    
                    logits = self.model.forward_train_test(X_train_t, y_train_t, X_test_t)
                    logits = logits[:, :, :self.n_classes_]
                    probs = F.softmax(logits, dim=-1)
                    all_probs.append(probs.squeeze(0).cpu().numpy())
            
            return np.concatenate(all_probs, axis=0)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for test samples."""
        probs = self.predict_proba(X)
        return probs.argmax(axis=1)


class OpenTabRegressor:
    """Sklearn-like interface for OpenTab regression.
    
    For regression, the model outputs a distribution over target values
    using a binning approach (Bar Distribution). The model predicts
    probabilities over bins, and we can extract mean, median, or quantiles.
    
    Usage:
        reg = OpenTabRegressor("checkpoint.pt")
        reg.fit(X_train, y_train)
        predictions = reg.predict(X_test)  # Returns mean predictions
    """
    
    # Number of bins for target discretization (like TabPFN)
    N_BINS = 64
    
    def __init__(
        self,
        model: Union[OpenTabModel, str, None] = None,
        device: Optional[str] = None,
        n_bins: int = 64,
    ):
        """
        Args:
            model: Either a OpenTabModel instance or path to checkpoint
            device: Device to use ('cuda', 'cpu', or None for auto)
            n_bins: Number of bins for target discretization
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.n_bins = n_bins
        
        if model is None:
            # Create default model with n_bins outputs for regression
            self.model = OpenTabModel(n_outputs=n_bins).to(self.device)
        elif isinstance(model, str):
            # Load from checkpoint
            checkpoint = torch.load(model, map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                config = checkpoint.get('config', {})
                # For regression, n_outputs should be n_bins
                self.model = OpenTabModel(
                    embedding_size=config.get('embedding_size', 96),
                    n_heads=config.get('n_heads', 4),
                    mlp_hidden_size=config.get('mlp_hidden', 192),
                    n_layers=config.get('n_layers', 3),
                    n_outputs=config.get('n_bins', config.get('max_classes', n_bins)),
                    dropout=config.get('dropout', 0.0),
                ).to(self.device)
                self.model.load_state_dict(checkpoint['model_state'])
            elif isinstance(checkpoint, dict) and 'model' in checkpoint:
                self.model = OpenTabModel(
                    **checkpoint.get('architecture', {})
                ).to(self.device)
                self.model.load_state_dict(checkpoint['model'])
            else:
                self.model = OpenTabModel(n_outputs=n_bins).to(self.device)
                self.model.load_state_dict(checkpoint)
        else:
            self.model = model.to(self.device)
        
        self.model.eval()
        
        # Storage for fit data
        self.X_train_ = None
        self.y_train_ = None
        self.y_train_raw_ = None  # Store raw values before normalization
        
        # Target normalization parameters
        self.y_mean_ = None
        self.y_std_ = None
        
        # Bin boundaries (computed during fit)
        self.bin_edges_ = None
        self.bin_centers_ = None
    
    def _create_bins(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create bin edges and centers for target discretization.
        
        Uses a mixture of uniform and quantile-based binning for robustness.
        """
        y_min, y_max = y.min(), y.max()
        
        # Add margin to handle extrapolation
        margin = (y_max - y_min) * 0.1 + 1e-6
        y_min -= margin
        y_max += margin
        
        # Create bin edges (uniform spacing in normalized space)
        bin_edges = np.linspace(y_min, y_max, self.n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        return bin_edges.astype(np.float32), bin_centers.astype(np.float32)
    
    def _discretize_targets(self, y: np.ndarray) -> np.ndarray:
        """Convert continuous targets to bin indices."""
        # Clip to bin range
        y_clipped = np.clip(y, self.bin_edges_[0], self.bin_edges_[-1])
        # Find bin indices
        bin_indices = np.digitize(y_clipped, self.bin_edges_[1:-1])
        return bin_indices.astype(np.int64)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Store training data for later prediction.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training targets (n_samples,) - continuous values
        """
        self.X_train_ = X.astype(np.float32)
        self.y_train_raw_ = y.astype(np.float32)
        
        # Normalize targets
        self.y_mean_ = float(y.mean())
        self.y_std_ = float(y.std()) + 1e-8
        y_normalized = (y - self.y_mean_) / self.y_std_
        
        # Create bins in normalized space
        self.bin_edges_, self.bin_centers_ = self._create_bins(y_normalized)
        
        # Discretize targets for the model
        self.y_train_ = self._discretize_targets(y_normalized)
        
        return self
    
    def predict(self, X: np.ndarray, output_type: str = 'mean') -> np.ndarray:
        """Predict target values for test samples.
        
        Args:
            X: Test features (n_samples, n_features)
            output_type: Type of prediction ('mean', 'median', 'mode')
            
        Returns:
            Predicted target values (n_samples,)
        """
        if self.X_train_ is None:
            raise ValueError("Must call fit() before predict()")
        
        X_test = X.astype(np.float32)
        
        with torch.no_grad():
            # Add batch dimension
            X_train_t = torch.from_numpy(self.X_train_).unsqueeze(0).to(self.device)
            y_train_t = torch.from_numpy(self.y_train_).float().unsqueeze(0).to(self.device)
            X_test_t = torch.from_numpy(X_test).unsqueeze(0).to(self.device)
            
            # Forward pass
            logits = self.model.forward_train_test(X_train_t, y_train_t, X_test_t)
            
            # Get probabilities over bins
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            
            # Convert to predictions based on output_type
            bin_centers = self.bin_centers_
            
            if output_type == 'mean':
                # Expected value: sum of bin_center * probability
                predictions = (probs * bin_centers).sum(axis=-1)
            elif output_type == 'median':
                # Find bin where cumulative probability crosses 0.5
                cumprobs = np.cumsum(probs, axis=-1)
                median_bins = (cumprobs >= 0.5).argmax(axis=-1)
                predictions = bin_centers[median_bins]
            elif output_type == 'mode':
                # Most likely bin
                mode_bins = probs.argmax(axis=-1)
                predictions = bin_centers[mode_bins]
            else:
                raise ValueError(f"Unknown output_type: {output_type}")
            
            # Denormalize predictions
            predictions = predictions * self.y_std_ + self.y_mean_
            
            return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability distribution over bins.
        
        Args:
            X: Test features (n_samples, n_features)
            
        Returns:
            Probabilities over bins (n_samples, n_bins)
        """
        if self.X_train_ is None:
            raise ValueError("Must call fit() before predict()")
        
        X_test = X.astype(np.float32)
        
        with torch.no_grad():
            X_train_t = torch.from_numpy(self.X_train_).unsqueeze(0).to(self.device)
            y_train_t = torch.from_numpy(self.y_train_).float().unsqueeze(0).to(self.device)
            X_test_t = torch.from_numpy(X_test).unsqueeze(0).to(self.device)
            
            logits = self.model.forward_train_test(X_train_t, y_train_t, X_test_t)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            
            return probs
    
    def predict_quantiles(self, X: np.ndarray, quantiles: list = None) -> np.ndarray:
        """Predict quantiles of the target distribution.
        
        Args:
            X: Test features (n_samples, n_features)
            quantiles: List of quantiles to predict (default: [0.1, 0.5, 0.9])
            
        Returns:
            Quantile predictions (n_samples, n_quantiles)
        """
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]
        
        probs = self.predict_proba(X)
        cumprobs = np.cumsum(probs, axis=-1)
        
        results = []
        for q in quantiles:
            q_bins = (cumprobs >= q).argmax(axis=-1)
            q_values = self.bin_centers_[q_bins]
            # Denormalize
            q_values = q_values * self.y_std_ + self.y_mean_
            results.append(q_values)
        
        return np.stack(results, axis=-1)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test
    print("Testing OpenTab model...")
    
    model = OpenTabModel(
        embedding_size=96,
        n_heads=4,
        mlp_hidden_size=192,
        n_layers=3,
        n_outputs=10,
    )
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Create dummy data
    batch_size = 2
    n_train = 50
    n_test = 10
    n_features = 5
    
    X_train = torch.randn(batch_size, n_train, n_features)
    y_train = torch.randint(0, 3, (batch_size, n_train)).float()
    X_test = torch.randn(batch_size, n_test, n_features)
    
    # Forward pass
    logits = model.forward_train_test(X_train, y_train, X_test)
    
    print(f"Input shapes: X_train={X_train.shape}, y_train={y_train.shape}, X_test={X_test.shape}")
    print(f"Output shape: {logits.shape}")
    print("✓ Model test passed!")
