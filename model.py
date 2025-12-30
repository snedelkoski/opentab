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
    
    Following TabPFN exactly:
    1. NaN handling: Replace NaN/missing with 0, create indicator tensor
    2. Normalization: Per-feature normalization using training statistics
    3. Encoding: Linear projection of [value, nan_indicator] to embedding space
    4. Feature grouping: If features_per_group > 1, group features before encoding
    
    The encoder takes (value, nan_indicator) pairs and projects to embedding_size.
    With feature grouping, it takes features_per_group * 2 inputs (values + indicators).
    """
    
    def __init__(self, embedding_size: int, features_per_group: int = 1):
        super().__init__()
        self.embedding_size = embedding_size
        self.features_per_group = features_per_group
        
        # Linear encoder following TabPFN:
        # Input: [value, nan_indicator] per feature (or per group of features)
        # For features_per_group=1: input is 2 (value + indicator)
        # For features_per_group=N: input is N*2 (N values + N indicators)
        input_size = features_per_group * 2  # value + nan_indicator for each feature in group
        self.encoder = nn.Linear(input_size, embedding_size, bias=True)
    
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
            (batch, rows, n_groups, embedding_size) tensor of embeddings
            where n_groups = ceil(features / features_per_group)
        """
        batch, rows, features = x.shape
        
        # Step 1: Detect missing values (using approximate comparison for float)
        nan_mask = (x < missing_indicator + 1) & (x > missing_indicator - 1)
        
        # Step 2: Replace missing values with 0 for normalization
        x_filled = x.clone()
        x_filled[nan_mask] = 0.0
        
        # Step 3: Compute mean and std from training portion only, excluding missing
        train_x = x[:, :train_size]
        train_nan_mask = nan_mask[:, :train_size]
        
        # Mask out missing values for statistics computation
        train_x_masked = train_x.clone()
        train_x_masked[train_nan_mask] = float('nan')
        
        # Compute stats ignoring NaN (missing values)
        mean = torch.nanmean(train_x_masked, dim=1, keepdim=True)
        
        # Handle case where all values are missing
        mean = torch.where(torch.isnan(mean), torch.zeros_like(mean), mean)
        
        # Compute std (handling missing values)
        train_x_centered = train_x_masked - mean
        train_x_centered[train_nan_mask] = 0.0
        n_valid = (~train_nan_mask).float().sum(dim=1, keepdim=True).clamp(min=1)
        var = (train_x_centered ** 2).sum(dim=1, keepdim=True) / n_valid
        std = torch.sqrt(var + 1e-8)
        
        # Step 4: Normalize all data (train + test) using training statistics
        x_normalized = (x_filled - mean) / std
        x_normalized = torch.clamp(x_normalized, -100, 100)
        
        # Create nan indicator tensor (1 where nan, 0 where not)
        nan_indicators = nan_mask.float()
        
        # Step 5: Handle feature grouping and encode
        g = self.features_per_group
        
        # Pad features to be divisible by group size
        n_groups = (features + g - 1) // g
        padded_features = n_groups * g
        
        if padded_features > features:
            # Pad with zeros (and mark as nan)
            pad_size = padded_features - features
            x_pad = torch.zeros(batch, rows, pad_size, device=x_normalized.device, dtype=x_normalized.dtype)
            x_normalized = torch.cat([x_normalized, x_pad], dim=2)
            nan_pad = torch.ones(batch, rows, pad_size, device=nan_indicators.device, dtype=nan_indicators.dtype)
            nan_indicators = torch.cat([nan_indicators, nan_pad], dim=2)
        
        # Reshape to groups: (batch, rows, n_groups, features_per_group)
        x_grouped = x_normalized.reshape(batch, rows, n_groups, g)
        nan_grouped = nan_indicators.reshape(batch, rows, n_groups, g)
        
        # Concatenate values and nan indicators: (batch, rows, n_groups, features_per_group * 2)
        # This follows TabPFN's approach of [value, nan_indicator] input to the encoder
        encoder_input = torch.cat([x_grouped, nan_grouped], dim=-1)
        
        # Step 6: Linear projection to embedding space
        embeddings = self.encoder(encoder_input)  # (batch, rows, n_groups, embedding_size)
        
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
    
    Uses PyTorch's scaled_dot_product_attention with Flash Attention (when available)
    for significant speedups on supported hardware.
    """
    
    def __init__(
        self,
        embedding_size: int,
        n_heads: int,
        mlp_hidden_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.embedding_size = embedding_size
        self.n_heads = n_heads
        self.head_dim = embedding_size // n_heads
        self.dropout = dropout
        
        assert embedding_size % n_heads == 0, "embedding_size must be divisible by n_heads"
        
        # QKV projections for feature attention
        self.qkv_features = nn.Linear(embedding_size, 3 * embedding_size, bias=True)
        self.out_proj_features = nn.Linear(embedding_size, embedding_size, bias=True)
        
        # QKV projections for sample attention
        self.qkv_samples = nn.Linear(embedding_size, 3 * embedding_size, bias=True)
        self.out_proj_samples = nn.Linear(embedding_size, embedding_size, bias=True)
        
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
    
    def _attention(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor,
        dropout_p: float = 0.0,
    ) -> torch.Tensor:
        """Apply scaled dot-product attention using Flash Attention when available.
        
        Args:
            q, k, v: (batch, n_heads, seq_len, head_dim) tensors
            dropout_p: dropout probability (only used during training)
            
        Returns:
            (batch, n_heads, seq_len, head_dim) attention output
        """
        # PyTorch's SDPA automatically uses Flash Attention when:
        # - CUDA device with compute capability >= 8.0 (Ampere+)
        # - Inputs are float16 or bfloat16
        # - No attention mask or causal mask is used
        # Falls back to efficient C++ implementation otherwise
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None,
            dropout_p=dropout_p if self.training else 0.0,
            is_causal=False,
        )
    
    def _reshape_for_attention(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Reshape (batch, seq, emb) -> (batch, n_heads, seq, head_dim)"""
        seq_len = x.shape[1]
        return x.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
    
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
        
        # Compute Q, K, V
        qkv = self.qkv_features(x_flat)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        q = self._reshape_for_attention(q, batch_size * n_rows)
        k = self._reshape_for_attention(k, batch_size * n_rows)
        v = self._reshape_for_attention(v, batch_size * n_rows)
        
        # Apply Flash Attention
        attn_out = self._attention(q, k, v, self.dropout)
        
        # Reshape back: (batch*rows, n_heads, cols, head_dim) -> (batch*rows, cols, emb)
        attn_out = attn_out.transpose(1, 2).reshape(batch_size * n_rows, n_cols, emb_size)
        attn_out = self.out_proj_features(attn_out)
        
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
        
        # Compute Q, K, V for training samples (self-attention)
        qkv_train = self.qkv_samples(x_train)
        q_train, k_train, v_train = qkv_train.chunk(3, dim=-1)
        
        q_train = self._reshape_for_attention(q_train, batch_size * n_cols)
        k_train = self._reshape_for_attention(k_train, batch_size * n_cols)
        v_train = self._reshape_for_attention(v_train, batch_size * n_cols)
        
        # Training samples self-attention
        train_attn = self._attention(q_train, k_train, v_train, self.dropout)
        train_attn = train_attn.transpose(1, 2).reshape(batch_size * n_cols, train_size, emb_size)
        train_attn = self.out_proj_samples(train_attn)
        
        # Test samples attend to training samples only
        if x_test.shape[1] > 0:
            test_size = x_test.shape[1]
            
            # Query from test, Key/Value from train
            q_test = self.qkv_samples(x_test).chunk(3, dim=-1)[0]  # Only need Q
            q_test = self._reshape_for_attention(q_test, batch_size * n_cols)
            
            # Reuse k_train and v_train from above
            test_attn = self._attention(q_test, k_train, v_train, self.dropout)
            test_attn = test_attn.transpose(1, 2).reshape(batch_size * n_cols, test_size, emb_size)
            test_attn = self.out_proj_samples(test_attn)
            
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


class FeaturePositionalEmbedding(nn.Module):
    """Feature positional embeddings using the 'subspace' approach from TabPFN.
    
    This helps the model distinguish between different features during attention.
    Each feature gets a unique positional identity by:
    1. Generating a random vector in a lower-dimensional subspace
    2. Projecting it to the full embedding dimension via a learned linear layer
    
    The random vectors are deterministically generated from a seed, ensuring
    consistency across forward passes while allowing different features to
    have different embeddings.
    """
    
    def __init__(self, embedding_size: int, subspace_dim: int = None, max_features: int = 1000, seed: int = 0):
        super().__init__()
        self.embedding_size = embedding_size
        self.subspace_dim = subspace_dim or embedding_size // 4
        self.max_features = max_features
        self.seed = seed
        
        # Learned projection from subspace to full embedding dimension
        self.projection = nn.Linear(self.subspace_dim, embedding_size)
        
        # Pre-generate random subspace vectors for efficiency
        # These are fixed random vectors, not learned
        generator = torch.Generator().manual_seed(seed)
        self.register_buffer(
            'subspace_vectors',
            torch.randn(max_features, self.subspace_dim, generator=generator)
        )
    
    def forward(self, n_features: int) -> torch.Tensor:
        """
        Generate positional embeddings for n_features.
        
        Args:
            n_features: Number of features (columns) to generate embeddings for
            
        Returns:
            (n_features, embedding_size) tensor of positional embeddings
        """
        # Get the pre-generated random vectors for these features
        subspace_vecs = self.subspace_vectors[:n_features]  # (n_features, subspace_dim)
        
        # Project to full embedding dimension
        pos_emb = self.projection(subspace_vecs)  # (n_features, embedding_size)
        
        return pos_emb


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
        use_feature_pos_emb: Whether to use feature positional embeddings
        max_features: Maximum number of features (for positional embeddings)
        features_per_group: Number of features to group together before attention.
            Higher values reduce attention cost but may lose fine-grained feature info.
            TabPFN uses 2. Set to 1 to disable grouping.
    """
    
    def __init__(
        self,
        embedding_size: int = 96,
        n_heads: int = 4,
        mlp_hidden_size: int = 192,
        n_layers: int = 3,
        n_outputs: int = 10,
        dropout: float = 0.0,
        use_feature_pos_emb: bool = True,
        max_features: int = 1000,
        features_per_group: int = 1,
    ):
        super().__init__()
        
        self.embedding_size = embedding_size
        self.n_outputs = n_outputs
        self.use_feature_pos_emb = use_feature_pos_emb
        self.features_per_group = features_per_group
        
        self.feature_encoder = FeatureEncoder(embedding_size, features_per_group=features_per_group)
        self.target_encoder = TargetEncoder(embedding_size)
        self.transformer = TransformerEncoder(
            embedding_size, n_heads, mlp_hidden_size, n_layers, dropout
        )
        self.decoder = Decoder(embedding_size, mlp_hidden_size, n_outputs)
        
        # Feature positional embeddings (following TabPFN's "subspace" approach)
        # Note: with grouping, we have fewer columns so we need fewer positional embeddings
        if use_feature_pos_emb:
            max_groups = (max_features + features_per_group - 1) // features_per_group
            self.feature_pos_emb = FeaturePositionalEmbedding(
                embedding_size, 
                subspace_dim=embedding_size // 4,
                max_features=max_groups + 1,  # +1 for target column
            )
        else:
            self.feature_pos_emb = None
    
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
        n_features = x.shape[2]
        
        # Encode features: (batch, rows, features, emb)
        x_emb = self.feature_encoder(x, train_size)
        
        # Encode targets: (batch, rows, 1, emb)
        y_emb = self.target_encoder(y, n_rows)
        
        # Concatenate features (or feature groups) and target: (batch, rows, n_groups+1, emb)
        combined = torch.cat([x_emb, y_emb], dim=2)
        
        # Add feature positional embeddings
        # This helps the model distinguish between different feature groups
        if self.feature_pos_emb is not None:
            n_cols = combined.shape[2]  # n_groups + 1 (for target)
            pos_emb = self.feature_pos_emb(n_cols)  # (n_cols, emb)
            # Broadcast across batch and rows: (1, 1, n_cols, emb)
            combined = combined + pos_emb.unsqueeze(0).unsqueeze(0)
        
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
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for test samples."""
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
            
            # Get probabilities for actual classes
            logits = logits[:, :, :self.n_classes_]
            probs = F.softmax(logits, dim=-1)
            
            return probs.squeeze(0).cpu().numpy()
    
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
