"""
train.py - Training Loop for OpenTab

This module implements the training loop for OpenTab, including:
- Data loading from HDF5 files or online generation
- Gradient accumulation for large effective batch sizes
- Learning rate scheduling
- Checkpointing and logging
- Evaluation callbacks

The training process:
1. Sample/load a batch of synthetic datasets
2. For each dataset: pass (X, y) through model, compute loss on held-out test set
3. Aggregate losses and update model weights
4. Periodically evaluate on validation data

Usage:
    python train.py --config config.yaml
    python train.py --data data/synthetic.h5 --epochs 100
    python train.py --online --n_steps 10000  # Generate data on-the-fly
"""

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, Iterator, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Try to import optional dependencies
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

try:
    from schedulefree import AdamWScheduleFree
    HAS_SCHEDULEFREE = True
except ImportError:
    HAS_SCHEDULEFREE = False

from model import OpenTabModel
from generate_data import get_prior, SyntheticDataGenerator

# sklearn datasets for real-world evaluation
try:
    from sklearn.datasets import (
        load_iris, load_wine, load_breast_cancer,
        load_diabetes, fetch_california_housing
    )
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@dataclass
class TrainConfig:
    """Training configuration.
    
    Note: Following TabPFN, classification and regression are trained as separate models.
    Set task_type='classification' for classification or task_type='regression' for regression.
    The prior_type will be automatically set based on task_type if not explicitly provided.
    """
    # Data
    data_path: Optional[str] = None  # Path to HDF5 data file
    online_generation: bool = False  # Generate data on-the-fly
    prior_type: str = 'mixed'  # Prior for online generation ('mixed' for classification, 'mixed_regression' for regression)
    task_type: str = 'classification'  # 'classification' or 'regression'
    
    # Model architecture
    embedding_size: int = 96
    n_heads: int = 4
    n_layers: int = 3
    mlp_hidden: int = 192
    dropout: float = 0.0
    max_features: int = 20
    max_samples: int = 100
    max_classes: int = 10
    use_feature_pos_emb: bool = True  # Feature positional embeddings (TabPFN "subspace" style)
    features_per_group: int = 2  # Group features before attention (TabPFN uses 2). Set to 1 to disable.
    
    # Training
    n_epochs: int = 100
    n_steps: Optional[int] = None  # If set, train for this many steps instead of epochs
    batch_size: int = 32
    grad_accumulation_steps: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # Optimization
    optimizer: str = 'adamw'  # 'adamw' or 'schedulefree'
    scheduler: str = 'cosine'  # 'cosine', 'linear', or 'constant'
    
    # Logging
    log_interval: int = 100
    eval_interval: int = 1000
    real_eval_interval: int = 2000  # Evaluate on real sklearn datasets
    save_interval: int = 5000
    output_dir: str = 'checkpoints'
    
    # Misc
    seed: int = 42
    device: str = 'auto'
    num_workers: int = 0
    compile_model: bool = False  # torch.compile (disabled by default due to dynamic train_size)
    
    def __post_init__(self):
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Auto-select prior based on task_type if using default
        if self.prior_type == 'mixed' and self.task_type == 'regression':
            self.prior_type = 'mixed_regression'
        
        # Validate task_type
        if self.task_type not in ('classification', 'regression'):
            raise ValueError(f"task_type must be 'classification' or 'regression', got '{self.task_type}'")


class HDF5Dataset(Dataset):
    """Dataset that reads from HDF5 file."""
    
    def __init__(self, path: str, max_classes: int = 10):
        if not HAS_H5PY:
            raise ImportError("h5py required for HDF5 datasets: pip install h5py")
        
        self.path = path
        self.max_classes = max_classes
        
        # Open file to get length
        with h5py.File(path, 'r') as f:
            self.length = f['X'].shape[0]
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        with h5py.File(self.path, 'r') as f:
            X = torch.tensor(f['X'][idx], dtype=torch.float32)
            y = torch.tensor(f['y'][idx], dtype=torch.long)
            train_size = int(f['single_eval_pos'][idx])
            n_features = int(f['num_features'][idx])
            n_samples = int(f['num_datapoints'][idx])
        
        return {
            'X': X,
            'y': y,
            'train_size': train_size,
            'n_features': n_features,
            'n_samples': n_samples,
        }


class OnlineDataset(Dataset):
    """Dataset that generates data on-the-fly."""
    
    def __init__(
        self,
        prior_type: str = 'mixed',
        length: int = 100000,
        max_samples: int = 100,
        max_features: int = 20,
        max_classes: int = 10,
    ):
        self.prior = get_prior(prior_type)
        self.length = length
        self.max_samples = max_samples
        self.max_features = max_features
        self.max_classes = max_classes
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Random dataset parameters
        n_samples = random.randint(10, self.max_samples)
        n_features = random.randint(2, self.max_features)
        n_classes = random.randint(2, self.max_classes)
        train_ratio = random.uniform(0.5, 0.9)
        
        # Generate
        dataset = self.prior.generate(n_samples, n_features, n_classes, train_ratio)
        
        # Pad to max size
        X_padded = np.zeros((self.max_samples, self.max_features), dtype=np.float32)
        y_padded = np.zeros(self.max_samples, dtype=np.int64)
        
        X_padded[:n_samples, :n_features] = dataset.X
        y_padded[:n_samples] = dataset.y
        
        return {
            'X': torch.tensor(X_padded),
            'y': torch.tensor(y_padded),
            'train_size': dataset.train_size,
            'n_features': n_features,
            'n_samples': n_samples,
        }


def collate_fn(batch):
    """Custom collate function for variable-length sequences."""
    return {
        'X': torch.stack([b['X'] for b in batch]),
        'y': torch.stack([b['y'] for b in batch]),
        'train_size': torch.tensor([b['train_size'] for b in batch]),
        'n_features': torch.tensor([b['n_features'] for b in batch]),
        'n_samples': torch.tensor([b['n_samples'] for b in batch]),
    }


class Trainer:
    """Training loop with logging and checkpointing."""
    
    def __init__(self, config: TrainConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.device_type = 'cuda' if 'cuda' in str(self.device) else 'cpu'
        # Set seeds
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        
        # Create model
        self.model = OpenTabModel(
            embedding_size=config.embedding_size,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            mlp_hidden_size=config.mlp_hidden,
            n_outputs=config.max_classes,
            dropout=config.dropout,
            use_feature_pos_emb=config.use_feature_pos_emb,
            max_features=config.max_features,
            features_per_group=config.features_per_group,
        ).to(self.device)
        
        # Create optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        self.best_real_accuracy = 0.0
        
        # Logging
        self.log_history = []
        self.real_eval_history = []
        
        # Create output dir
        os.makedirs(config.output_dir, exist_ok=True)
    
    def _create_optimizer(self):
        """Create optimizer based on config."""
        params = self.model.parameters()
        
        if self.config.optimizer == 'schedulefree':
            if not HAS_SCHEDULEFREE:
                print("Warning: schedulefree not available, using AdamW")
                return torch.optim.AdamW(
                    params,
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                )
            return AdamWScheduleFree(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        else:
            return torch.optim.AdamW(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
    
    def _create_scheduler(self, num_training_steps: int):
        """Create learning rate scheduler."""
        if self.config.optimizer == 'schedulefree':
            return None  # ScheduleFree doesn't need a scheduler
        
        warmup_steps = min(self.config.warmup_steps, num_training_steps // 10)
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            
            if self.config.scheduler == 'constant':
                return 1.0
            
            progress = (step - warmup_steps) / (num_training_steps - warmup_steps)
            
            if self.config.scheduler == 'linear':
                return max(0.1, 1 - progress)
            else:  # cosine
                return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def _create_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        if self.config.data_path:
            dataset = HDF5Dataset(
                self.config.data_path,
                max_classes=self.config.max_classes,
            )
        else:
            dataset = OnlineDataset(
                prior_type=self.config.prior_type,
                length=100000,  # Virtual length
                max_samples=self.config.max_samples,
                max_features=self.config.max_features,
                max_classes=self.config.max_classes,
            )
        
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
        )
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute cross-entropy loss on test positions."""
        X = batch['X'].to(self.device)
        y = batch['y'].to(self.device)
        train_size = batch['train_size']
        n_samples = batch['n_samples']
        
        # Use bfloat16 autocast for faster training on CUDA
        autocast_ctx = torch.amp.autocast(device_type=self.device_type, dtype=torch.bfloat16) if self.device_type == "cuda" else nullcontext()
        
        # Process each batch item individually since train_size varies
        batch_size = X.shape[0]
        losses = []
        
        for i in range(batch_size):
            ts = train_size[i].item()
            ns = n_samples[i].item()
            
            if ts < ns:  # Have test samples
                # Get single sample: add batch dim
                X_i = X[i:i+1]  # (1, samples, features)
                y_train_i = y[i:i+1, :ts]  # (1, train_size) - only train labels!
                
                # Skip if data contains NaN/Inf
                if torch.isnan(X_i).any() or torch.isinf(X_i).any():
                    continue
                
                # Forward pass for this sample with autocast
                with autocast_ctx:
                    logits = self.model(X_i, y_train_i, ts)  # (1, n_test, classes)
                
                # Skip if model output is NaN
                if torch.isnan(logits).any():
                    continue
                
                # Get predictions on test positions
                test_logits = logits[0, :ns-ts]  # (n_test, classes)
                test_targets = y[i, ts:ns]  # (n_test,)
                
                loss = F.cross_entropy(test_logits, test_targets)
                
                # Only append valid losses
                if not torch.isnan(loss):
                    losses.append(loss)
        
        if losses:
            return torch.stack(losses).mean()
        else:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step."""
        self.model.train()
        
        loss = self.compute_loss(batch)
        loss = loss / self.config.grad_accumulation_steps
        loss.backward()
        
        return loss.item() * self.config.grad_accumulation_steps
    
    def train(self):
        """Main training loop."""
        dataloader = self._create_dataloader()
        
        # Calculate total steps
        if self.config.n_steps:
            total_steps = self.config.n_steps
        else:
            steps_per_epoch = len(dataloader) // self.config.grad_accumulation_steps
            total_steps = steps_per_epoch * self.config.n_epochs
        
        # Create scheduler
        self.scheduler = self._create_scheduler(total_steps)
        
        print(f"Starting training:")
        print(f"  Device: {self.device}")
        print(f"  Total steps: {total_steps}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Grad accumulation: {self.config.grad_accumulation_steps}")
        print(f"  Effective batch size: {self.config.batch_size * self.config.grad_accumulation_steps}")
        print(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print()
        
        # Training loop
        data_iter = iter(dataloader)
        accumulated_loss = 0.0
        step_losses = []
        start_time = time.time()
        
        for step in range(total_steps):
            # Accumulate gradients
            for accum_step in range(self.config.grad_accumulation_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    self.epoch += 1
                    data_iter = iter(dataloader)
                    batch = next(data_iter)
                
                loss = self.train_step(batch)
                accumulated_loss += loss
            
            # Update weights
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.max_grad_norm
                )
            
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            if self.scheduler:
                self.scheduler.step()
            
            # Track loss
            step_loss = accumulated_loss / self.config.grad_accumulation_steps
            step_losses.append(step_loss)
            accumulated_loss = 0.0
            
            self.global_step = step + 1
            
            # Logging
            if self.global_step % self.config.log_interval == 0:
                avg_loss = np.mean(step_losses[-self.config.log_interval:])
                lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.config.learning_rate
                elapsed = time.time() - start_time
                steps_per_sec = self.global_step / elapsed
                
                log_msg = (
                    f"Step {self.global_step}/{total_steps} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"LR: {lr:.2e} | "
                    f"Speed: {steps_per_sec:.1f} steps/s"
                )
                print(log_msg)
                
                self.log_history.append({
                    'step': self.global_step,
                    'loss': float(avg_loss),
                    'lr': float(lr),
                })
            
            # Evaluation
            if self.config.eval_interval and self.global_step % self.config.eval_interval == 0:
                self.evaluate()
            
            # Real-world evaluation on sklearn datasets
            if self.config.real_eval_interval and self.global_step % self.config.real_eval_interval == 0:
                self.evaluate_real_datasets()
            
            # Save checkpoint
            if self.global_step % self.config.save_interval == 0:
                self.save_checkpoint()
        
        # Final save
        self.save_checkpoint(final=True)
        print("Training complete!")
    
    def evaluate(self):
        """Evaluate model on fresh synthetic data."""
        self.model.eval()
        
        # Generate eval data
        eval_dataset = OnlineDataset(
            prior_type=self.config.prior_type,
            length=100,
            max_samples=self.config.max_samples,
            max_features=self.config.max_features,
            max_classes=self.config.max_classes,
        )
        
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        # Use bfloat16 autocast for faster inference on CUDA
        autocast_ctx = torch.amp.autocast(device_type=self.device_type, dtype=torch.bfloat16) if self.device_type == "cuda" else nullcontext()
        
        with torch.no_grad():
            for batch in eval_loader:
                X = batch['X'].to(self.device)
                y = batch['y'].to(self.device)
                train_size = batch['train_size']
                n_samples = batch['n_samples']
                
                # Process each batch item individually since train_size varies
                batch_size = X.shape[0]
                for i in range(batch_size):
                    ts = train_size[i].item()
                    ns = n_samples[i].item()
                    
                    if ts < ns:
                        X_i = X[i:i+1]
                        y_train_i = y[i:i+1, :ts]
                        
                        # Skip if data contains NaN/Inf
                        if torch.isnan(X_i).any() or torch.isinf(X_i).any():
                            continue
                        
                        with autocast_ctx:
                            logits = self.model(X_i, y_train_i, ts)
                        
                        # Skip if model output is NaN
                        if torch.isnan(logits).any():
                            continue
                        
                        test_logits = logits[0, :ns-ts]
                        test_targets = y[i, ts:ns]
                        
                        loss = F.cross_entropy(test_logits, test_targets)
                        
                        # Skip NaN losses
                        if not torch.isnan(loss):
                            total_loss += loss.item() * (ns - ts)
                            preds = test_logits.argmax(dim=-1)
                            total_correct += (preds == test_targets).sum().item()
                            total_samples += (ns - ts)
        
        if total_samples > 0:
            avg_loss = total_loss / total_samples
            accuracy = total_correct / total_samples
            
            print(f"\n  Eval | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}\n")
            
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.save_checkpoint(best=True)
        
        self.model.train()
    
    def evaluate_real_datasets(self):
        """Evaluate model on real sklearn datasets for tracking progress.
        
        This provides a quick sanity check on real data during training.
        Only evaluates datasets matching the current task_type (classification or regression).
        
        Note: Following TabPFN, classification and regression are trained as separate models.
        """
        if not HAS_SKLEARN:
            return
        
        self.model.eval()
        
        results = {}
        
        if self.config.task_type == 'classification':
            # Classification datasets
            classification_datasets = [
                ('Iris', load_iris()),
                ('Wine', load_wine()),
                ('Breast Cancer', load_breast_cancer()),
            ]
            
            total_acc = 0.0
            n_classification = 0
            
            # Use bfloat16 autocast for faster inference on CUDA
            autocast_ctx = torch.amp.autocast(device_type=self.device_type, dtype=torch.bfloat16) if self.device_type == "cuda" else nullcontext()
            
            with torch.no_grad():
                for name, data in classification_datasets:
                    X, y = data.data, data.target
                    
                    # Use small subset for speed
                    n_samples = min(100, len(X))
                    indices = np.random.choice(len(X), n_samples, replace=False)
                    X, y = X[indices], y[indices]
                    
                    # Split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.3, random_state=42, stratify=y
                    )
                    
                    # Scale
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)
                    
                    # Pad to model dimensions
                    n_train, n_features = X_train.shape
                    n_test = X_test.shape[0]
                    n_classes = len(np.unique(y))
                    
                    # Combine train and test
                    X_all = np.zeros((1, n_train + n_test, self.config.max_features), dtype=np.float32)
                    X_all[0, :n_train, :n_features] = X_train
                    X_all[0, n_train:n_train+n_test, :n_features] = X_test
                    
                    y_all = np.zeros((1, n_train + n_test), dtype=np.int64)
                    y_all[0, :n_train] = y_train
                    y_all[0, n_train:n_train+n_test] = y_test
                    
                    # Convert to tensors
                    X_tensor = torch.tensor(X_all, device=self.device)
                    y_train_tensor = torch.tensor(y_all[:, :n_train], device=self.device)
                    
                    # Predict - pass only train labels and train_size as int
                    with autocast_ctx:
                        logits = self.model(X_tensor, y_train_tensor, n_train)
                    
                    # Get predictions on test set
                    test_logits = logits[0, :n_test, :n_classes]
                    preds = test_logits.argmax(dim=-1).cpu().numpy()
                    
                    # Compute accuracy
                    accuracy = (preds == y_test).mean()
                    results[name] = accuracy
                    total_acc += accuracy
                    n_classification += 1
            
            avg_acc = total_acc / n_classification if n_classification > 0 else 0.0
            
            # Log results
            clf_str = ' | '.join([f"{k}: {v:.3f}" for k, v in results.items()])
            print(f"\n  Real Eval (Classification) | {clf_str} | Avg: {avg_acc:.3f}\n")
            
            self.real_eval_history.append({
                'step': self.global_step,
                'results': results,
                'avg_accuracy': avg_acc,
            })
            
            # Save best model based on real accuracy
            if avg_acc > self.best_real_accuracy:
                self.best_real_accuracy = avg_acc
                self.save_checkpoint(best=True)
        
        else:  # regression
            # Regression datasets
            regression_datasets = [
                ('Diabetes', load_diabetes()),
            ]
            
            # Try to add California Housing (may not be available offline)
            try:
                regression_datasets.append(('California', fetch_california_housing()))
            except Exception:
                pass
            
            total_r2 = 0.0
            n_regression = 0
            
            # Use bfloat16 autocast for faster inference on CUDA
            autocast_ctx = torch.amp.autocast(device_type=self.device_type, dtype=torch.bfloat16) if self.device_type == "cuda" else nullcontext()
            
            with torch.no_grad():
                for name, data in regression_datasets:
                    X, y = data.data, data.target
                    
                    # Use small subset for speed
                    n_samples = min(100, len(X))
                    indices = np.random.choice(len(X), n_samples, replace=False)
                    X, y = X[indices], y[indices]
                    
                    # Split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.3, random_state=42
                    )
                    
                    # Scale features
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)
                    
                    # Scale targets for the model (will use bin indices)
                    y_scaler = StandardScaler()
                    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
                    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).flatten()
                    
                    # Discretize targets into bins
                    n_bins = min(self.config.max_classes, 32)
                    y_min, y_max = y_train_scaled.min(), y_train_scaled.max()
                    margin = (y_max - y_min) * 0.1 + 1e-6
                    bin_edges = np.linspace(y_min - margin, y_max + margin, n_bins + 1)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    
                    y_train_bins = np.clip(np.digitize(y_train_scaled, bin_edges[1:-1]), 0, n_bins - 1)
                    
                    # Pad to model dimensions
                    n_train, n_features = X_train.shape
                    n_test = X_test.shape[0]
                    
                    X_all = np.zeros((1, n_train + n_test, self.config.max_features), dtype=np.float32)
                    X_all[0, :n_train, :n_features] = X_train
                    X_all[0, n_train:n_train+n_test, :n_features] = X_test
                    
                    y_all = np.zeros((1, n_train + n_test), dtype=np.int64)
                    y_all[0, :n_train] = y_train_bins
                    
                    # Convert to tensors
                    X_tensor = torch.tensor(X_all, device=self.device)
                    y_train_tensor = torch.tensor(y_all[:, :n_train], device=self.device)
                    
                    # Predict
                    with autocast_ctx:
                        logits = self.model(X_tensor, y_train_tensor, n_train)
                    
                    # Get predictions - use expected value over bin probabilities
                    test_logits = logits[0, :n_test, :n_bins]
                    probs = torch.softmax(test_logits, dim=-1).cpu().numpy()
                    
                    # Compute mean prediction
                    bin_centers_t = bin_centers[:n_bins]
                    preds_scaled = (probs * bin_centers_t).sum(axis=-1)
                    
                    # Inverse transform to original scale
                    preds = y_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
                    
                    # Compute R2 score
                    r2 = r2_score(y_test, preds)
                    r2 = max(-1.0, min(1.0, r2))  # Clip to valid range
                    
                    results[f"{name}_R2"] = r2
                    total_r2 += r2
                    n_regression += 1
            
            avg_r2 = total_r2 / n_regression if n_regression > 0 else 0.0
            
            # Log results
            reg_str = ' | '.join([f"{k}: {v:.3f}" for k, v in results.items()])
            print(f"\n  Real Eval (Regression) | {reg_str} | Avg R2: {avg_r2:.3f}\n")
            
            self.real_eval_history.append({
                'step': self.global_step,
                'results': results,
                'avg_r2': avg_r2,
            })
            
            # Save best model based on R2 score
            if avg_r2 > self.best_real_accuracy:  # Reusing this variable for R2
                self.best_real_accuracy = avg_r2
            self.save_checkpoint(best=True)
        
        self.model.train()
    
    def save_checkpoint(self, best: bool = False, final: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict() if self.scheduler else None,
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_loss': self.best_loss,
            'best_real_accuracy': self.best_real_accuracy,
            'config': vars(self.config),
            'log_history': self.log_history,
            'real_eval_history': self.real_eval_history,
        }
        
        if best:
            path = os.path.join(self.config.output_dir, 'best_model.pt')
        elif final:
            path = os.path.join(self.config.output_dir, 'final_model.pt')
        else:
            path = os.path.join(self.config.output_dir, f'checkpoint_{self.global_step}.pt')
        
        torch.save(checkpoint, path)
        print(f"  Saved checkpoint: {path}")
        
        # Save log history as JSON
        log_path = os.path.join(self.config.output_dir, 'training_log.json')
        with open(log_path, 'w') as f:
            json.dump(self.log_history, f, indent=2)
    
    def load_checkpoint(self, path: str):
        """Load checkpoint and resume training."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        if checkpoint['scheduler_state'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        self.best_real_accuracy = checkpoint.get('best_real_accuracy', 0.0)
        self.log_history = checkpoint.get('log_history', [])
        self.real_eval_history = checkpoint.get('real_eval_history', [])
        
        print(f"Loaded checkpoint from {path}")
        print(f"  Resuming from step {self.global_step}, epoch {self.epoch}")
        print(f"  Best real accuracy: {self.best_real_accuracy:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train OpenTab')
    
    # Data arguments
    parser.add_argument('--data', type=str, default=None,
                       help='Path to HDF5 training data')
    parser.add_argument('--online', action='store_true',
                       help='Generate training data on-the-fly')
    parser.add_argument('--prior', type=str, default='mixed',
                       choices=['mlp', 'gp', 'tree', 'scm', 'mixed',
                                'mlp_regression', 'gp_regression', 'linear_regression', 'mixed_regression'],
                       help='Prior type for data generation (use *_regression for regression task)')
    parser.add_argument('--task', type=str, default='classification',
                       choices=['classification', 'regression'],
                       help='Task type: classification or regression (trains separate models like TabPFN)')
    
    # Model arguments
    parser.add_argument('--embedding_size', type=int, default=96)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--mlp_hidden', type=int, default=192)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--max_features', type=int, default=20)
    parser.add_argument('--max_samples', type=int, default=100)
    parser.add_argument('--max_classes', type=int, default=10)
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--steps', type=int, default=None,
                       help='Number of training steps (overrides epochs)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--grad_accumulation', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    
    # Optimization arguments
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['adamw', 'schedulefree'])
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'linear', 'constant'])
    
    # Logging arguments
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--eval_interval', type=int, default=5)
    parser.add_argument('--real_eval_interval', type=int, default=20,
                       help='Evaluate on real sklearn datasets every N steps')
    parser.add_argument('--save_interval', type=int, default=100)
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    
    # Misc arguments
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.data and not args.online:
        print("Note: No data path specified, using online generation")
        args.online = True
    
    # Create config
    config = TrainConfig(
        data_path=args.data,
        online_generation=args.online,
        prior_type=args.prior,
        embedding_size=args.embedding_size,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        mlp_hidden=args.mlp_hidden,
        dropout=args.dropout,
        max_features=args.max_features,
        max_samples=args.max_samples,
        max_classes=args.max_classes,
        n_epochs=args.epochs,
        n_steps=args.steps,
        batch_size=args.batch_size,
        grad_accumulation_steps=args.grad_accumulation,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        real_eval_interval=args.real_eval_interval,
        save_interval=args.save_interval,
        output_dir=args.output_dir,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        task_type=args.task,
    )
    
    # Create trainer
    trainer = Trainer(config)
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()
