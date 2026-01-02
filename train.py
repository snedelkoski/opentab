"""
train.py - Training Loop for OpenTab

Implements the training procedure from the TabPFN paper:
- Training on synthetic datasets generated via SCMs
- Cross-entropy loss on held-out test samples
- Adam optimizer with linear warmup and cosine annealing
- Samples up to 2048 training samples, fixed 128 validation samples
- Beta distribution for sampling number of features
- Table size restriction to avoid memory peaks

Training details from paper:
- ~2,000,000 steps with batch size 64
- ~130M synthetic datasets total
- Features sampled from Beta(0.95, 8.0) scaled to [1, 160]
- Max table size: 75,000 cells

Usage:
    python train.py --online --steps 100000
    python train.py --data data/synthetic.h5 --epochs 100
"""

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Optional, Dict, Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset

from model import OpenTabModel
from generate_data import SCMDataGenerator

# Optional imports
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

try:
    from sklearn.datasets import load_iris, load_wine, load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@dataclass
class TrainConfig:
    """Training configuration following TabPFN paper."""
    
    # Data generation
    max_train_samples: int = 2048  # Max training samples per dataset
    eval_samples: int = 128  # Fixed validation set size
    max_features: int = 160  # Max features
    max_classes: int = 10  # Max classes for classification
    max_table_cells: int = 75000  # Max cells to avoid memory peaks
    
    # Feature sampling: Beta(k, b) scaled to [1, max_features]
    feature_beta_k: float = 0.95
    feature_beta_b: float = 8.0
    
    # Model architecture
    embedding_size: int = 128
    n_heads: int = 4
    n_layers: int = 6
    mlp_hidden: int = 256
    dropout: float = 0.0
    
    # Training
    n_steps: int = 100000  # Paper uses ~2M steps
    batch_size: int = 64  # Paper uses 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # Task type
    is_regression: bool = False
    n_bins: int = 64  # For regression
    
    # Logging
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 10000
    output_dir: str = 'checkpoints'
    
    # Hardware
    seed: int = 42
    device: str = 'auto'
    use_amp: bool = True  # Automatic mixed precision
    
    def __post_init__(self):
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


class OnlineDataGenerator(IterableDataset):
    """
    Generates synthetic datasets on-the-fly following TabPFN paper.
    
    Key aspects:
    - Samples up to 2048 training samples
    - Fixed 128 validation samples
    - Features sampled from Beta distribution
    - Table size limited to 75,000 cells
    """
    
    def __init__(self, config: TrainConfig):
        self.config = config
        self.generator = SCMDataGenerator(
            n_samples_range=(20, config.max_train_samples + config.eval_samples),
            n_features_range=(2, config.max_features),
            n_classes_range=(2, config.max_classes),
            is_regression=config.is_regression,
        )
    
    def _sample_n_features(self) -> int:
        """Sample number of features from Beta distribution."""
        # Beta(k, b) scaled to [1, max_features]
        beta_sample = np.random.beta(
            self.config.feature_beta_k,
            self.config.feature_beta_b
        )
        n_features = int(1 + beta_sample * (self.config.max_features - 1))
        return max(2, min(n_features, self.config.max_features))
    
    def _sample_n_samples(self, n_features: int) -> int:
        """Sample number of samples, respecting max table size."""
        # Max samples based on cell limit
        max_samples_by_cells = self.config.max_table_cells // n_features
        max_samples = min(
            self.config.max_train_samples + self.config.eval_samples,
            max_samples_by_cells
        )
        
        # Sample uniformly
        n_samples = random.randint(
            self.config.eval_samples + 10,  # At least 10 training samples
            max(self.config.eval_samples + 10, max_samples)
        )
        return n_samples
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        while True:
            # Sample dataset dimensions
            n_features = self._sample_n_features()
            n_samples = self._sample_n_samples(n_features)
            
            # Train/test split: fixed eval_samples for test
            train_size = n_samples - self.config.eval_samples
            train_size = max(10, train_size)
            n_samples = train_size + self.config.eval_samples
            
            # Generate dataset
            try:
                dataset = self.generator.generate(
                    n_samples=n_samples,
                    n_features=n_features,
                    train_ratio=train_size / n_samples,
                )
            except Exception:
                continue  # Skip failed generations
            
            # Convert to tensors
            X = torch.tensor(dataset.X, dtype=torch.float32)
            if self.config.is_regression:
                y = torch.tensor(dataset.y, dtype=torch.float32)
            else:
                y = torch.tensor(dataset.y, dtype=torch.long)
            
            yield {
                'X': X,
                'y': y,
                'train_size': train_size,
                'n_features': n_features,
                'n_samples': n_samples,
            }


class HDF5Dataset(Dataset):
    """Dataset from pre-generated HDF5 file."""
    
    def __init__(self, path: str):
        if not HAS_H5PY:
            raise ImportError("h5py required: pip install h5py")
        self.path = path
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


def collate_variable_size(batch):
    """Collate batches with variable sizes by padding."""
    max_samples = max(b['n_samples'] for b in batch)
    max_features = max(b['n_features'] for b in batch)
    batch_size = len(batch)
    
    X_padded = torch.zeros(batch_size, max_samples, max_features)
    y_padded = torch.zeros(batch_size, max_samples, dtype=batch[0]['y'].dtype)
    train_sizes = torch.zeros(batch_size, dtype=torch.long)
    n_features = torch.zeros(batch_size, dtype=torch.long)
    n_samples = torch.zeros(batch_size, dtype=torch.long)
    
    for i, b in enumerate(batch):
        ns, nf = b['n_samples'], b['n_features']
        X_padded[i, :ns, :nf] = b['X'][:ns, :nf]
        y_padded[i, :ns] = b['y'][:ns]
        train_sizes[i] = b['train_size']
        n_features[i] = nf
        n_samples[i] = ns
    
    return {
        'X': X_padded,
        'y': y_padded,
        'train_size': train_sizes,
        'n_features': n_features,
        'n_samples': n_samples,
    }


class Trainer:
    """Training loop for OpenTab."""
    
    def __init__(self, config: TrainConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Set seeds
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        
        # Create model
        n_outputs = config.n_bins if config.is_regression else config.max_classes
        self.model = OpenTabModel(
            embedding_size=config.embedding_size,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            mlp_hidden_size=config.mlp_hidden,
            n_outputs=n_outputs,
            dropout=config.dropout,
        ).to(self.device)
        
        # Optimizer: Adam with weight decay
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Scheduler: linear warmup + cosine annealing
        self.scheduler = self._create_scheduler()
        
        # Mixed precision
        self.scaler = torch.amp.GradScaler('cuda') if config.use_amp and self.device.type == 'cuda' else None
        
        # Training state
        self.global_step = 0
        self.best_loss = float('inf')
        self.log_history = []
        
        # Output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _create_scheduler(self):
        """Linear warmup + cosine annealing."""
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / self.config.warmup_steps
            
            progress = (step - self.config.warmup_steps) / (
                self.config.n_steps - self.config.warmup_steps
            )
            return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute cross-entropy loss on held-out test samples.
        
        Following paper: loss = -log p(y_test | X, y_train)
        """
        X = batch['X'].to(self.device)
        y = batch['y'].to(self.device)
        train_sizes = batch['train_size']
        n_samples = batch['n_samples']
        
        batch_size = X.shape[0]
        losses = []
        
        for i in range(batch_size):
            ts = train_sizes[i].item()
            ns = n_samples[i].item()
            
            if ts >= ns:
                continue  # No test samples
            
            # Get single item
            X_i = X[i:i+1, :ns]
            y_train_i = y[i:i+1, :ts]
            
            # Skip if data is invalid
            if torch.isnan(X_i).any() or torch.isinf(X_i).any():
                continue
            
            # Forward pass
            logits = self.model(X_i, y_train_i, ts)
            
            if torch.isnan(logits).any():
                continue
            
            # Compute loss on test samples
            test_logits = logits[0, :ns - ts]
            test_targets = y[i, ts:ns]
            
            if self.config.is_regression:
                # For regression, use cross-entropy over bins
                loss = F.cross_entropy(test_logits, test_targets.long())
            else:
                loss = F.cross_entropy(test_logits, test_targets)
            
            if not torch.isnan(loss):
                losses.append(loss)
        
        if losses:
            return torch.stack(losses).mean()
        return torch.tensor(0.0, device=self.device, requires_grad=True)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Mixed precision forward pass
        if self.scaler is not None:
            with torch.amp.autocast('cuda'):
                loss = self.compute_loss(batch)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss = self.compute_loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
        
        self.scheduler.step()
        
        return loss.item()
    
    def train(self, data_path: Optional[str] = None):
        """Main training loop."""
        # Create data loader
        if data_path:
            dataset = HDF5Dataset(data_path)
            loader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=4,
                collate_fn=collate_variable_size,
                pin_memory=True,
            )
            data_iter = iter(loader)
        else:
            # Online generation
            dataset = OnlineDataGenerator(self.config)
            loader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                num_workers=0,  # Must be 0 for IterableDataset
                collate_fn=collate_variable_size,
            )
            data_iter = iter(loader)
        
        print(f"Training for {self.config.n_steps} steps")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Device: {self.device}")
        print()
        
        # Training loop
        losses = []
        start_time = time.time()
        
        for step in range(self.config.n_steps):
            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)
            
            # Train step
            loss = self.train_step(batch)
            losses.append(loss)
            self.global_step = step + 1
            
            # Logging
            if self.global_step % self.config.log_interval == 0:
                avg_loss = np.mean(losses[-self.config.log_interval:])
                lr = self.scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                speed = self.global_step / elapsed
                
                print(f"Step {self.global_step}/{self.config.n_steps} | "
                      f"Loss: {avg_loss:.4f} | LR: {lr:.2e} | "
                      f"Speed: {speed:.1f} steps/s")
                
                self.log_history.append({
                    'step': self.global_step,
                    'loss': float(avg_loss),
                    'lr': float(lr),
                })
            
            # Evaluation
            if self.global_step % self.config.eval_interval == 0:
                self.evaluate()
            
            # Save checkpoint
            if self.global_step % self.config.save_interval == 0:
                self.save_checkpoint()
        
        # Final save
        self.save_checkpoint(final=True)
        print("Training complete!")
    
    def evaluate(self):
        """Evaluate on fresh synthetic data and sklearn datasets."""
        self.model.eval()
        
        # Generate eval data
        eval_gen = OnlineDataGenerator(self.config)
        eval_loader = DataLoader(
            eval_gen,
            batch_size=self.config.batch_size,
            collate_fn=collate_variable_size,
        )
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for i, batch in enumerate(eval_loader):
                if i >= 10:  # Evaluate on 10 batches
                    break
                
                X = batch['X'].to(self.device)
                y = batch['y'].to(self.device)
                train_sizes = batch['train_size']
                n_samples_batch = batch['n_samples']
                
                for j in range(X.shape[0]):
                    ts = train_sizes[j].item()
                    ns = n_samples_batch[j].item()
                    
                    if ts >= ns:
                        continue
                    
                    X_j = X[j:j+1, :ns]
                    y_train_j = y[j:j+1, :ts]
                    
                    if torch.isnan(X_j).any():
                        continue
                    
                    logits = self.model(X_j, y_train_j, ts)
                    
                    if torch.isnan(logits).any():
                        continue
                    
                    test_logits = logits[0, :ns - ts]
                    test_targets = y[j, ts:ns]
                    
                    if not self.config.is_regression:
                        n_classes = int(test_targets.max().item()) + 1
                        test_logits = test_logits[:, :n_classes]
                        loss = F.cross_entropy(test_logits, test_targets)
                        preds = test_logits.argmax(dim=-1)
                        total_correct += (preds == test_targets).sum().item()
                    else:
                        loss = F.cross_entropy(test_logits, test_targets.long())
                    
                    total_loss += loss.item() * (ns - ts)
                    total_samples += (ns - ts)
        
        if total_samples > 0:
            avg_loss = total_loss / total_samples
            acc = total_correct / total_samples if not self.config.is_regression else 0
            
            print(f"\n  Eval | Loss: {avg_loss:.4f} | Acc: {acc:.4f}\n")
            
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.save_checkpoint(best=True)
        
        # Evaluate on sklearn datasets
        if HAS_SKLEARN and not self.config.is_regression:
            self._eval_sklearn()
        
        self.model.train()
    
    def _eval_sklearn(self):
        """Quick eval on sklearn datasets."""
        from model import OpenTabClassifier
        
        classifier = OpenTabClassifier(model=self.model, device=str(self.device))
        
        datasets = [
            ('Iris', load_iris()),
            ('Wine', load_wine()),
            ('Breast Cancer', load_breast_cancer()),
        ]
        
        results = []
        for name, data in datasets:
            X, y = data.data, data.target
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            classifier.fit(X_train.astype(np.float32), y_train.astype(np.int64))
            y_pred = classifier.predict(X_test.astype(np.float32))
            acc = (y_pred == y_test).mean()
            results.append(f"{name}: {acc:.3f}")
        
        print(f"  sklearn | {' | '.join(results)}\n")
    
    def save_checkpoint(self, best: bool = False, final: bool = False):
        """Save checkpoint."""
        checkpoint = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'config': {
                'embedding_size': self.config.embedding_size,
                'n_heads': self.config.n_heads,
                'n_layers': self.config.n_layers,
                'mlp_hidden_size': self.config.mlp_hidden,
                'max_classes': self.config.max_classes,
                'max_features': self.config.max_features,
                'n_bins': self.config.n_bins,
                'is_regression': self.config.is_regression,
                'dropout': self.config.dropout,
            },
            'log_history': self.log_history,
        }
        
        if best:
            path = os.path.join(self.config.output_dir, 'best_model.pt')
        elif final:
            path = os.path.join(self.config.output_dir, 'final_model.pt')
        else:
            path = os.path.join(self.config.output_dir, f'checkpoint_{self.global_step}.pt')
        
        torch.save(checkpoint, path)
        print(f"  Saved: {path}")
        
        # Save log
        with open(os.path.join(self.config.output_dir, 'log.json'), 'w') as f:
            json.dump(self.log_history, f, indent=2)
    
    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        self.log_history = checkpoint.get('log_history', [])
        print(f"Loaded checkpoint from {path}, step {self.global_step}")


def main():
    parser = argparse.ArgumentParser(description='Train OpenTab')
    
    # Data
    parser.add_argument('--data', type=str, default=None, help='HDF5 data path')
    parser.add_argument('--online', action='store_true', help='Generate data on-the-fly')
    
    # Model
    parser.add_argument('--embedding_size', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--mlp_hidden', type=int, default=256)
    
    # Training
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    
    # Task
    parser.add_argument('--regression', action='store_true', help='Train for regression')
    parser.add_argument('--max_classes', type=int, default=10)
    parser.add_argument('--n_bins', type=int, default=64, help='Bins for regression')
    
    # Logging
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--eval_interval', type=int, default=1000)
    parser.add_argument('--save_interval', type=int, default=10000)
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--resume', type=str, default=None)
    
    args = parser.parse_args()
    
    if not args.data and not args.online:
        print("Using online data generation")
        args.online = True
    
    config = TrainConfig(
        embedding_size=args.embedding_size,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        mlp_hidden=args.mlp_hidden,
        n_steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        is_regression=args.regression,
        max_classes=args.max_classes,
        n_bins=args.n_bins,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        output_dir=args.output_dir,
        seed=args.seed,
        device=args.device,
    )
    
    trainer = Trainer(config)
    
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    trainer.train(args.data)


if __name__ == '__main__':
    main()
