"""
evaluate.py - Evaluation for OpenTab

This module provides tools to evaluate OpenTab on eval data and the TabArena benchmark.

TabArena is a living benchmark for tabular ML with 51 curated datasets and
standardized evaluation protocols.

Note: Following TabPFN, OpenTab trains separate models for classification and regression.
Use OpenTabClassifier for classification tasks and OpenTabRegressor for regression tasks.

Usage:
    # Quick classification evaluation on sklearn datasets
    python evaluate.py --checkpoint checkpoints/classifier.pt --mode quick
    
    # Quick regression evaluation on sklearn datasets  
    python evaluate.py --checkpoint checkpoints/regressor.pt --mode quick-regression
    
    # Full TabArena-Lite evaluation (classification only, 51 datasets, 1 fold)
    python evaluate.py --checkpoint checkpoints/classifier.pt --mode lite
    
    # Full TabArena evaluation (all datasets, all folds)
    python evaluate.py --checkpoint checkpoints/classifier.pt --mode full

Requirements:
    pip install tabarena autogluon openml

References:
    - TabArena: https://github.com/autogluon/tabarena
    - Leaderboard: https://huggingface.co/spaces/TabArena/leaderboard
"""

import argparse
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Import our model
from model import OpenTabModel, OpenTabClassifier, OpenTabRegressor

# Try importing TabArena dependencies
try:
    import openml
    from tabarena.benchmark.models.wrapper.abstract_class import AbstractExecModel
    HAS_TABARENA = True
except ImportError:
    HAS_TABARENA = False
    AbstractExecModel = object  # Fallback for type hints
    print("Warning: TabArena dependencies not installed.")
    print("Install with: pip install tabarena autogluon openml")


# Global variable to store checkpoint path for model initialization
_GLOBAL_CHECKPOINT_PATH: Optional[str] = None


def set_checkpoint_path(path: str):
    """Set the global checkpoint path for OpenTab model."""
    global _GLOBAL_CHECKPOINT_PATH
    _GLOBAL_CHECKPOINT_PATH = path


class OpenTabWrapper(AbstractExecModel):
    """
    TabArena-compatible wrapper for OpenTab (Classification).
    
    This wrapper inherits from AbstractExecModel to be compatible with
    TabArena's Experiment class for benchmarking.
    
    Supports classification (binary/multiclass) tasks only.
    For regression, use OpenTabRegressor directly.
    
    Note: Following TabPFN, classification and regression are separate models.
    This wrapper is for classification only.
    
    Note: OpenTab is designed for small datasets (up to ~1000 samples).
    For larger datasets, we subsample the training data.
    """
    
    # Maximum training samples to use (TabPFN-style limit)
    MAX_TRAIN_SAMPLES = 512  # Further reduced for memory
    MAX_FEATURES = 32  # Limit features more strictly for memory
    MAX_TEST_BATCH = 256  # Maximum test samples per batch
    
    def __init__(self, checkpoint_path: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.checkpoint_path = checkpoint_path or _GLOBAL_CHECKPOINT_PATH
        self._model = None
        self._n_classes = None
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._feature_columns = None
    
    def _fit(self, X: pd.DataFrame, y: pd.Series, X_val=None, y_val=None,
             num_cpus=1, num_gpus=0, time_limit=None, **kwargs):
        """Fit the OpenTab model on training data."""
        # Set device based on available resources
        self._device = 'cuda' if num_gpus > 0 and torch.cuda.is_available() else 'cpu'
        
        # Limit number of features first
        if X.shape[1] > self.MAX_FEATURES:
            # Keep only the first MAX_FEATURES columns
            self._feature_columns = X.columns[:self.MAX_FEATURES].tolist()
            X = X[self._feature_columns]
        else:
            self._feature_columns = X.columns.tolist()
        
        # Subsample if dataset is too large
        if len(X) > self.MAX_TRAIN_SAMPLES:
            # Stratified subsampling
            from sklearn.model_selection import train_test_split
            try:
                _, X, _, y = train_test_split(
                    X, y, 
                    test_size=self.MAX_TRAIN_SAMPLES / len(X),
                    stratify=y,
                    random_state=42
                )
            except ValueError:
                # If stratification fails, do random sampling
                idx = np.random.RandomState(42).choice(len(X), self.MAX_TRAIN_SAMPLES, replace=False)
                X = X.iloc[idx]
                y = y.iloc[idx]
        
        # Preprocess data
        X_processed = self._preprocess_X(X)
        
        # Determine number of classes
        self._n_classes = len(np.unique(y))
        
        # Load model from checkpoint if provided
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, map_location=self._device, weights_only=False)
            config = checkpoint.get('config', {})
            
            model = OpenTabModel(
                embedding_size=config.get('embedding_size', 96),
                n_heads=config.get('n_heads', 4),
                n_layers=config.get('n_layers', 3),
                mlp_hidden_size=config.get('mlp_hidden', 192),
                n_outputs=max(config.get('max_classes', 10), self._n_classes),
                dropout=config.get('dropout', 0.0),
            )
            model.load_state_dict(checkpoint['model_state'])
            self._model = OpenTabClassifier(model=model, device=self._device)
        else:
            # Create a new model with default architecture
            model = OpenTabModel(
                n_outputs=max(10, self._n_classes),
                embedding_size=96,
                n_heads=4,
                n_layers=3,
            )
            self._model = OpenTabClassifier(model=model, device=self._device)
        
        # Fit (store training data for in-context learning)
        self._model.fit(X_processed, y.values)
        
        return self
    
    # Special value used to indicate missing during training
    MISSING_INDICATOR = -999.0
    
    def _preprocess_X(self, X: pd.DataFrame) -> np.ndarray:
        """
        Preprocess features for OpenTab following TabPFN conventions.
        
        Handles:
        1. Categorical features: ordinal encoding (matches training prior)
        2. Missing values: replaced with special indicator value
        3. Feature subsetting: only use features seen during training
        """
        X = X.copy()
        
        # Limit features if we did so during training
        if self._feature_columns is not None:
            # Only keep the features we used during training
            available_cols = [c for c in self._feature_columns if c in X.columns]
            missing_cols = [c for c in self._feature_columns if c not in X.columns]
            X = X[available_cols]
            # Add missing columns with indicator value
            for col in missing_cols:
                X[col] = self.MISSING_INDICATOR
            X = X[self._feature_columns]  # Ensure correct order
        
        # Track missing values BEFORE filling
        missing_mask = X.isna()
        
        # Convert categorical columns to ordinal encoding
        # This matches the ordinal encoding used in FeatureAugmenter
        for col in X.select_dtypes(include=['category', 'object']).columns:
            # Handle NaN in categorical: assign -1 then convert to indicator
            codes = pd.Categorical(X[col]).codes.astype(np.float32)
            codes[codes == -1] = np.nan  # Mark as missing
            X[col] = codes
        
        # Update missing mask after categorical conversion
        missing_mask = missing_mask | X.isna()
        
        # Fill missing values with indicator (matches training augmentation)
        X = X.fillna(self.MISSING_INDICATOR)
        
        return X.astype(np.float32).values
    
    def _predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict class labels."""
        X_processed = self._preprocess_X(X)
        y_pred = self._model.predict(X_processed)
        return pd.Series(y_pred, index=X.index)
    
    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict class probabilities with batching for memory efficiency."""
        X_processed = self._preprocess_X(X)
        
        # Use small batch size to prevent OOM
        # The model uses in-context learning, so memory scales with train + test size
        BATCH_SIZE = 64  # Very small batch for test predictions
        
        # Clear cache before prediction
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if len(X_processed) <= BATCH_SIZE:
            probs = self._model.predict_proba(X_processed)
        else:
            # Batched prediction
            all_probs = []
            for i in range(0, len(X_processed), BATCH_SIZE):
                batch = X_processed[i:i + BATCH_SIZE]
                batch_probs = self._model.predict_proba(batch)
                all_probs.append(batch_probs)
                # Clear CUDA cache between batches
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            probs = np.vstack(all_probs)
        
        # Ensure we return probabilities for all classes
        if probs.shape[1] < self._n_classes:
            # Pad with zeros if needed
            padded = np.zeros((probs.shape[0], self._n_classes), dtype=np.float32)
            padded[:, :probs.shape[1]] = probs
            probs = padded
        elif probs.shape[1] > self._n_classes:
            # Trim if needed
            probs = probs[:, :self._n_classes]
        
        return pd.DataFrame(probs, index=X.index)
    
    @classmethod
    def supported_problem_types(cls) -> List[str]:
        """Return supported problem types."""
        return ['binary', 'multiclass']
    
    def _get_default_resources(self) -> Tuple[int, int]:
        """Return default CPU/GPU resources."""
        return (1, 1)  # 1 CPU, 1 GPU
    
    def get_minimum_resources(self, is_gpu_available: bool = False) -> Dict[str, int]:
        """Return minimum required resources."""
        return {'num_cpus': 1, 'num_gpus': 1 if is_gpu_available else 0}
    
    def cleanup(self):
        """Release resources after evaluation. Called by TabArena after each task."""
        import gc
        self._model = None
        self._feature_columns = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class OpenTabRegressionWrapper(AbstractExecModel):
    """
    TabArena-compatible wrapper for OpenTab (Regression).
    
    This wrapper inherits from AbstractExecModel to be compatible with
    TabArena's Experiment class for benchmarking.
    
    Supports regression tasks only.
    For classification, use OpenTabWrapper.
    
    Note: Following TabPFN, classification and regression are separate models.
    This wrapper is for regression only.
    """
    
    # Maximum training samples to use (TabPFN-style limit)
    MAX_TRAIN_SAMPLES = 512
    MAX_FEATURES = 32
    MAX_TEST_BATCH = 256
    
    # Special value used to indicate missing during training
    MISSING_INDICATOR = -999.0
    
    def __init__(self, checkpoint_path: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.checkpoint_path = checkpoint_path or _GLOBAL_CHECKPOINT_PATH
        self._model = None
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._feature_columns = None
    
    def _fit(self, X: pd.DataFrame, y: pd.Series, X_val=None, y_val=None, 
             num_cpus=1, num_gpus=0, time_limit=None, **kwargs):
        """Fit the OpenTab regression model on training data."""
        # Set device based on available resources
        self._device = 'cuda' if num_gpus > 0 and torch.cuda.is_available() else 'cpu'
        
        # Limit number of features first
        if X.shape[1] > self.MAX_FEATURES:
            self._feature_columns = X.columns[:self.MAX_FEATURES].tolist()
            X = X[self._feature_columns]
        else:
            self._feature_columns = X.columns.tolist()
        
        # Subsample if dataset is too large
        if len(X) > self.MAX_TRAIN_SAMPLES:
            from sklearn.model_selection import train_test_split
            try:
                _, X, _, y = train_test_split(
                    X, y, 
                    test_size=self.MAX_TRAIN_SAMPLES / len(X),
                    random_state=42
                )
            except ValueError:
                idx = np.random.RandomState(42).choice(len(X), self.MAX_TRAIN_SAMPLES, replace=False)
                X = X.iloc[idx]
                y = y.iloc[idx]
        
        # Preprocess data
        X_processed = self._preprocess_X(X)
        
        # Load model from checkpoint if provided
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, map_location=self._device, weights_only=False)
            config = checkpoint.get('config', {})
            
            model = OpenTabModel(
                embedding_size=config.get('embedding_size', 96),
                n_heads=config.get('n_heads', 4),
                n_layers=config.get('n_layers', 3),
                mlp_hidden_size=config.get('mlp_hidden', 192),
                n_outputs=1,  # Regression has single output
                dropout=config.get('dropout', 0.0),
            )
            model.load_state_dict(checkpoint['model_state'])
            self._model = OpenTabRegressor(model=model, device=self._device)
        else:
            # Create a new model with default architecture
            model = OpenTabModel(
                n_outputs=1,
                embedding_size=96,
                n_heads=4,
                n_layers=3,
            )
            self._model = OpenTabRegressor(model=model, device=self._device)
        
        # Fit (store training data for in-context learning)
        self._model.fit(X_processed, y.values)
        
        return self
    
    def _preprocess_X(self, X: pd.DataFrame) -> np.ndarray:
        """
        Preprocess features for OpenTab following TabPFN conventions.
        
        Handles:
        1. Categorical features: ordinal encoding
        2. Missing values: replaced with special indicator value
        3. Feature subsetting: only use features seen during training
        """
        X = X.copy()
        
        # Limit features if we did so during training
        if self._feature_columns is not None:
            available_cols = [c for c in self._feature_columns if c in X.columns]
            missing_cols = [c for c in self._feature_columns if c not in X.columns]
            X = X[available_cols]
            for col in missing_cols:
                X[col] = self.MISSING_INDICATOR
            X = X[self._feature_columns]
        
        # Track missing values BEFORE filling
        missing_mask = X.isna()
        
        # Convert categorical columns to ordinal encoding
        for col in X.select_dtypes(include=['category', 'object']).columns:
            codes = pd.Categorical(X[col]).codes.astype(np.float32)
            codes[codes == -1] = np.nan
            X[col] = codes
        
        # Update missing mask after categorical conversion
        missing_mask = missing_mask | X.isna()
        
        # Fill missing values with indicator
        X = X.fillna(self.MISSING_INDICATOR)
        
        return X.astype(np.float32).values
    
    def _predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict regression values."""
        X_processed = self._preprocess_X(X)
        
        # Use batching for large test sets
        BATCH_SIZE = 64
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if len(X_processed) <= BATCH_SIZE:
            y_pred = self._model.predict(X_processed)
        else:
            all_preds = []
            for i in range(0, len(X_processed), BATCH_SIZE):
                batch = X_processed[i:i + BATCH_SIZE]
                batch_pred = self._model.predict(batch)
                all_preds.append(batch_pred)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            y_pred = np.concatenate(all_preds)
        
        return pd.Series(y_pred, index=X.index)
    
    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Regression does not support predict_proba."""
        raise NotImplementedError("Regression tasks do not support predict_proba. Use _predict instead.")
    
    @classmethod
    def supported_problem_types(cls) -> List[str]:
        """Return supported problem types."""
        return ['regression']
    
    def _get_default_resources(self) -> Tuple[int, int]:
        """Return default CPU/GPU resources."""
        return (1, 1)
    
    def get_minimum_resources(self, is_gpu_available: bool = False) -> Dict[str, int]:
        """Return minimum required resources."""
        return {'num_cpus': 1, 'num_gpus': 1 if is_gpu_available else 0}
    
    def cleanup(self):
        """Release resources after evaluation."""
        import gc
        self._model = None
        self._feature_columns = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def get_tabarena_configs(checkpoint_path: str, num_configs: int = 1) -> List[Dict[str, Any]]:
    """
    Generate hyperparameter configurations for TabArena benchmarking.
    
    Args:
        checkpoint_path: Path to model checkpoint
        num_configs: Number of configurations to generate
    
    Returns:
        List of configuration dictionaries
    """
    configs = []
    
    # Default configuration (from checkpoint)
    configs.append({
        'checkpoint_path': checkpoint_path,
        'device': 'auto',
    })
    
    return configs


def compute_elo_ratings(
    output_dir: str,
    method_name: str = "OpenTab",
    eval_dir: Optional[Path] = None,
) -> Optional[pd.DataFrame]:
    """
    Compute Elo ratings by comparing with TabArena baseline methods.
    
    This downloads baseline results from TabArena and computes head-to-head
    Elo ratings to show how the model compares.
    
    Args:
        output_dir: Directory containing the evaluation results
        method_name: Name of the method for display
        eval_dir: Directory to save Elo results
    
    Returns:
        DataFrame with Elo leaderboard or None if computation fails
    """
    try:
        from tabarena.nips2025_utils.end_to_end_single import EndToEndSingle, EndToEndResultsSingle
        from bencheval.website_format import format_leaderboard
    except ImportError:
        print("\nNote: Could not compute Elo ratings (tabarena end_to_end not available)")
        return None
    
    print("\n" + "=" * 60)
    print("COMPUTING ELO RATINGS")
    print("=" * 60)
    print("Comparing with TabArena baseline methods...")
    print("(This may take a moment to download baseline results)")
    
    try:
        # First, process our results into TabArena format
        path_raw = Path(output_dir) / "data" / method_name
        
        if not path_raw.exists():
            print(f"Warning: Results path not found: {path_raw}")
            return None
        
        # Process results and cache them
        # Following the official TabArena pattern from run_evaluate_model.py
        print("Processing raw results...")
        end_to_end = EndToEndSingle.from_path_raw(
            path_raw=path_raw,
            name=method_name,  # Explicitly set method name
            cache=True,
        )
        
        # Convert to results object
        end_to_end_results = end_to_end.to_results()
        
        # Compare on TabArena tasks
        # Using only_valid_tasks=True to compare only on tasks we have results for
        elo_output_dir = eval_dir if eval_dir else Path(output_dir) / "elo"
        elo_output_dir = Path(elo_output_dir)
        elo_output_dir.mkdir(parents=True, exist_ok=True)
        
        print("Computing Elo ratings against TabArena baselines...")
        leaderboard = end_to_end_results.compare_on_tabarena(
            output_dir=elo_output_dir,
            only_valid_tasks=True,  # Only compare on tasks we ran
            subset='classification',  # Classification subset
        )
        
        # Format leaderboard for display
        try:
            leaderboard_formatted = format_leaderboard(leaderboard)
            print("\nFormatted Leaderboard:")
            print(leaderboard_formatted.to_markdown(index=False))
        except Exception:
            pass  # format_leaderboard may not be available
        
        # Find our method in the leaderboard
        our_row = leaderboard[leaderboard['method'] == method_name]
        
        if len(our_row) > 0:
            our_elo = our_row['elo'].values[0]
            our_elo_plus = our_row['elo+'].values[0]
            our_elo_minus = our_row['elo-'].values[0]
            our_rank = our_row['rank'].values[0]
            our_winrate = our_row['winrate'].values[0]
            total_methods = len(leaderboard)
            
            print("\n" + "=" * 60)
            print("ELO RATINGS (vs TabArena Baselines)")
            print("=" * 60)
            print(f"\n{'Method':<40} {'Elo':>8} {'95% CI':>15} {'Rank':>8} {'Win Rate':>10}")
            print("-" * 85)
            
            # Show top 5 methods
            for idx, row in leaderboard.head(5).iterrows():
                elo_ci = f"+{row['elo+']:.0f}/-{row['elo-']:.0f}"
                marker = " <--" if row['method'] == method_name else ""
                print(f"{row['method']:<40} {row['elo']:>8.1f} {elo_ci:>15} {row['rank']:>8.1f} {row['winrate']*100:>9.1f}%{marker}")
            
            # Show separator and our method if not in top 5
            if our_rank > 5:
                print("..." + " " * 80)
                elo_ci = f"+{our_elo_plus:.0f}/-{our_elo_minus:.0f}"
                print(f"{method_name:<40} {our_elo:>8.1f} {elo_ci:>15} {our_rank:>8.1f} {our_winrate*100:>9.1f}% <--")
            
            # Show bottom reference
            last_row = leaderboard.iloc[-1]
            if last_row['method'] != method_name:
                print("..." + " " * 80)
                elo_ci = f"+{last_row['elo+']:.0f}/-{last_row['elo-']:.0f}"
                print(f"{last_row['method']:<40} {last_row['elo']:>8.1f} {elo_ci:>15} {last_row['rank']:>8.1f} {last_row['winrate']*100:>9.1f}%")
            
            print("\n" + "-" * 60)
            print(f"OpenTab Summary:")
            print(f"  Elo Rating: {our_elo:.1f} (95% CI: +{our_elo_plus:.1f}/-{our_elo_minus:.1f})")
            print(f"  Rank: {our_rank:.0f}/{total_methods}")
            print(f"  Win Rate: {our_winrate*100:.1f}%")
            print("-" * 60)
            
            # Save Elo leaderboard
            if eval_dir:
                leaderboard.to_csv(eval_dir / "elo_leaderboard.csv", index=False)
                print(f"\nElo leaderboard saved to {eval_dir / 'elo_leaderboard.csv'}")
        
        return leaderboard
        
    except Exception as e:
        print(f"\nWarning: Could not compute Elo ratings: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_quick_evaluation(
    checkpoint_path: str,
    datasets: Optional[List[str]] = None,
    output_dir: str = 'eval_results',
):
    """
    Run quick evaluation on a few datasets without full TabArena setup.
    
    This is useful for testing before running the full benchmark.
    """
    from sklearn.datasets import load_iris, load_wine, load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    print("=" * 60)
    print("Quick Evaluation (without TabArena)")
    print("=" * 60)
    
    # Load model
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        config = checkpoint.get('config', {})
        
        model = OpenTabModel(
            embedding_size=config.get('embedding_size', 96),
            n_heads=config.get('n_heads', 4),
            n_layers=config.get('n_layers', 3),
            mlp_hidden_size=config.get('mlp_hidden', 192),
            n_outputs=config.get('max_classes', 10),
            dropout=config.get('dropout', 0.0),
        )
        model.load_state_dict(checkpoint['model_state'])
        classifier = OpenTabClassifier(model=model)
        print(f"Loaded model from {checkpoint_path}")
    else:
        # Create untrained model
        model = OpenTabModel()
        classifier = OpenTabClassifier(model=model)
        print("Warning: Using untrained model")
    
    # Test datasets
    test_datasets = [
        ('Iris', load_iris()),
        ('Wine', load_wine()),
        ('Breast Cancer', load_breast_cancer()),
    ]
    
    results = []
    
    for name, data in test_datasets:
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Fit and predict
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        accuracy = (y_pred == y_test).mean()
        results.append({'dataset': name, 'accuracy': accuracy})
        print(f"  {name}: {accuracy:.4f}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'quick_eval_results.csv'), index=False)
    
    print(f"\nResults saved to {output_dir}/quick_eval_results.csv")
    return results_df


def quick_eval_regression(
    checkpoint_path: Optional[str] = None,
    output_dir: str = 'eval_results',
):
    """
    Quick regression evaluation on sklearn datasets.
    
    Uses Diabetes and California Housing for fast regression testing.
    """
    from sklearn.datasets import load_diabetes, fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    
    print("=" * 60)
    print("Quick Regression Evaluation")
    print("=" * 60)
    
    # Load model
    if checkpoint_path and os.path.exists(checkpoint_path):
        regressor = OpenTabRegressor(checkpoint_path)
        print(f"Loaded model from {checkpoint_path}")
    else:
        # Create untrained model
        model = OpenTabModel(n_outputs=64)  # 64 bins for regression
        regressor = OpenTabRegressor(model=model)
        print("Warning: Using untrained model")
    
    # Test datasets
    test_datasets = [
        ('Diabetes', load_diabetes()),
    ]
    
    # Try to add California Housing
    try:
        test_datasets.append(('California', fetch_california_housing()))
    except Exception:
        print("Note: California Housing dataset not available offline")
    
    results = []
    
    for name, data in test_datasets:
        X, y = data.data, data.target
        
        # Use subset for speed
        n_samples = min(500, len(X))
        indices = np.random.RandomState(42).choice(len(X), n_samples, replace=False)
        X, y = X[indices], y[indices]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Fit and predict
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        
        # Compute metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            'dataset': name,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
        })
        print(f"  {name}:")
        print(f"    RMSE: {rmse:.4f}")
        print(f"    MAE:  {mae:.4f}")
        print(f"    R2:   {r2:.4f}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'quick_eval_regression_results.csv'), index=False)
    
    print(f"\nResults saved to {output_dir}/quick_eval_regression_results.csv")
    return results_df


def quick_eval_all(
    checkpoint_path: Optional[str] = None,
    output_dir: str = 'eval_results',
):
    """
    Run both classification and regression quick evaluation.
    
    Note: Following TabPFN, classification and regression require separate models.
    This function uses the SAME checkpoint for both, which may not produce good
    results unless you're testing a model trained on mixed data (not recommended).
    
    For proper evaluation:
    - Use `--mode quick` with classification checkpoint for classification
    - Use `--mode quick-regression` with regression checkpoint for regression
    """
    print("\n" + "=" * 60)
    print("CLASSIFICATION EVALUATION")
    print("=" * 60 + "\n")
    clf_results = quick_eval(checkpoint_path, output_dir)
    
    print("\n" + "=" * 60)
    print("REGRESSION EVALUATION")
    print("=" * 60 + "\n")
    reg_results = quick_eval_regression(checkpoint_path, output_dir)
    
    return clf_results, reg_results


def run_tabarena_lite(
    checkpoint_path: str,
    output_dir: str = 'tabarena_results',
    cache: bool = True,
):
    """
    Run TabArena-Lite evaluation (classification datasets only, 1 fold per dataset).
    
    This is the recommended evaluation for quick benchmarking.
    """
    if not HAS_TABARENA:
        print("Error: TabArena not installed. Run:")
        print("  pip install tabarena autogluon openml")
        return None
    
    from tabarena.benchmark.experiment import Experiment, ExperimentBatchRunner
    from tabarena.benchmark.result import ExperimentResults
    from tabarena.nips2025_utils.tabarena_context import TabArenaContext
    
    # Initialize Ray with runtime environment that excludes large files
    try:
        import ray
        if not ray.is_initialized():
            ray.init(
                runtime_env={
                    'excludes': [
                        'data/',
                        'checkpoints/',
                        'tabarena_results/',
                        'tabarena_elo_results/',
                        'eval_results/',
                        '*.h5',
                        '*.pt',
                        '*.pth',
                        '__pycache__/',
                        '.ipynb_checkpoints/',
                        '.git/',
                    ]
                }
            )
    except Exception as e:
        print(f"Warning: Could not initialize Ray: {e}")
    
    print("=" * 60)
    print("TabArena-Lite Evaluation")
    print("=" * 60)
    
    # Set global checkpoint path
    set_checkpoint_path(checkpoint_path)
    
    # Get TabArena context and metadata
    tabarena_context = TabArenaContext()
    task_metadata = tabarena_context.task_metadata
    
    # Filter to classification tasks only (binary and multiclass)
    classification_tasks = task_metadata[
        task_metadata['problem_type'].isin(['binary', 'multiclass'])
    ]
    
    datasets = list(classification_tasks['name'])
    folds = [0]  # TabArena-Lite uses 1 fold
    
    print(f"Running on {len(datasets)} classification datasets")
    
    # Create experiment configuration using Experiment with our wrapper
    methods = [
        Experiment(
            name="OpenTab",
            method_cls=OpenTabWrapper,
            method_kwargs={
                'checkpoint_path': checkpoint_path,
            },
        ),
    ]
    
    # Run experiments
    exp_batch_runner = ExperimentBatchRunner(
        expname=output_dir,
        task_metadata=task_metadata,
    )
    
    results_lst = exp_batch_runner.run(
        datasets=datasets,
        folds=folds,
        methods=methods,
        ignore_cache=not cache,
    )
    
    # Extract results from cached files
    import pickle
    results_data = []
    base_path = Path(output_dir) / "data" / "OpenTab"
    
    for tid_dir in base_path.iterdir():
        if tid_dir.is_dir():
            result_file = tid_dir / "0" / "results.pkl"
            if result_file.exists():
                with open(result_file, 'rb') as f:
                    result = pickle.load(f)
                
                task_meta = result.get('task_metadata', {})
                results_data.append({
                    'dataset': task_meta.get('name', tid_dir.name),
                    'tid': tid_dir.name,
                    'metric': result.get('metric', 'unknown'),
                    'metric_error': result.get('metric_error', float('nan')),
                    'metric_score': 1.0 - result.get('metric_error', float('nan')),  # Convert error to score
                    'time_train_s': result.get('time_train_s', 0),
                    'time_infer_s': result.get('time_infer_s', 0),
                    'problem_type': result.get('problem_type', 'unknown'),
                })
    
    df_results = pd.DataFrame(results_data)
    df_results = df_results.sort_values('dataset')
    
    # Save results
    eval_dir = Path(output_dir) / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(eval_dir / "results.csv", index=False)
    
    print("\n" + "=" * 60)
    print("TABARENA-LITE RESULTS: OpenTab")
    print("=" * 60)
    
    # Display results by problem type
    for ptype in ['binary', 'multiclass']:
        subset = df_results[df_results['problem_type'] == ptype]
        if len(subset) > 0:
            print(f"\n{ptype.upper()} CLASSIFICATION ({len(subset)} datasets):")
            print("-" * 50)
            display_cols = ['dataset', 'metric', 'metric_score', 'time_train_s']
            print(subset[display_cols].to_string(index=False))
            print(f"\nMean score: {subset['metric_score'].mean():.4f}")
    
    # Overall summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total datasets evaluated: {len(df_results)}")
    print(f"Binary classification: {len(df_results[df_results['problem_type'] == 'binary'])} datasets")
    print(f"Multiclass classification: {len(df_results[df_results['problem_type'] == 'multiclass'])} datasets")
    print(f"\nOverall mean metric score: {df_results['metric_score'].mean():.4f}")
    print(f"Overall mean metric error: {df_results['metric_error'].mean():.4f}")
    print(f"Total training time: {df_results['time_train_s'].sum():.2f}s")
    print(f"Total inference time: {df_results['time_infer_s'].sum():.2f}s")
    
    print(f"\nResults saved to {eval_dir}")
    
    # Compute Elo ratings by comparing with TabArena baselines
    elo_results = compute_elo_ratings(
        output_dir=output_dir,
        method_name="OpenTab",
        eval_dir=eval_dir,
    )
    
    return df_results


def run_tabarena_full(
    checkpoint_path: str,
    output_dir: str = 'tabarena_results_full',
    cache: bool = True,
):
    """
    Run full TabArena evaluation (classification datasets, multiple folds).
    
    This provides the most thorough evaluation but takes longer.
    """
    if not HAS_TABARENA:
        print("Error: TabArena not installed.")
        return None
    
    from tabarena.benchmark.experiment import Experiment, ExperimentBatchRunner
    from tabarena.benchmark.result import ExperimentResults
    from tabarena.nips2025_utils.tabarena_context import TabArenaContext
    
    # Initialize Ray with runtime environment that excludes large files
    try:
        import ray
        if not ray.is_initialized():
            ray.init(
                runtime_env={
                    'excludes': [
                        'data/',
                        'checkpoints/',
                        'tabarena_results/',
                        'tabarena_elo_results/',
                        'eval_results/'
                    ]
                }
            )
    except Exception as e:
        print(f"Warning: Could not initialize Ray: {e}")
    
    print("=" * 60)
    print("TabArena Full Evaluation")
    print("=" * 60)
    
    # Set global checkpoint path
    set_checkpoint_path(checkpoint_path)
    
    # Get TabArena context and metadata
    tabarena_context = TabArenaContext()
    task_metadata = tabarena_context.task_metadata
    
    # Filter to classification tasks only (binary and multiclass)
    classification_tasks = task_metadata[
        task_metadata['problem_type'].isin(['binary', 'multiclass'])
    ]
    
    datasets = list(classification_tasks['name'])
    folds = [0, 1, 2]  # Full uses 3 folds
    
    print(f"Running on {len(datasets)} classification datasets with {len(folds)} folds")
    
    # Create experiment configuration using Experiment with our wrapper
    methods = [
        Experiment(
            name="OpenTab",
            method_cls=OpenTabWrapper,
            method_kwargs={
                'checkpoint_path': checkpoint_path,
            },
        ),
    ]
    
    # Run experiments
    exp_batch_runner = ExperimentBatchRunner(
        expname=output_dir,
        task_metadata=task_metadata,
    )
    
    results_lst = exp_batch_runner.run(
        datasets=datasets,
        folds=folds,
        methods=methods,
        ignore_cache=not cache,
    )
    
    # Process results
    experiment_results = ExperimentResults(task_metadata=task_metadata)
    repo = experiment_results.repo_from_results(results_lst=results_lst)
    
    # Generate leaderboard
    eval_dir = Path(output_dir) / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    df_results = repo.results()
    
    # Save results
    df_results.to_csv(eval_dir / "results_full.csv", index=False)
    
    print("\n" + "=" * 60)
    print("RESULTS (Full Evaluation)")
    print("=" * 60)
    print(df_results.to_string())
    
    print(f"\nResults saved to {eval_dir}")
    
    return df_results


def run_on_custom_datasets(
    checkpoint_path: str,
    datasets: List[Tuple[np.ndarray, np.ndarray, str]],
    output_dir: str = 'custom_eval_results',
):
    """
    Evaluate OpenTab on custom datasets.
    
    Args:
        checkpoint_path: Path to model checkpoint
        datasets: List of (X, y, name) tuples
        output_dir: Directory to save results
    """
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    
    print("=" * 60)
    print("Custom Dataset Evaluation")
    print("=" * 60)
    
    # Load model
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        config = checkpoint.get('config', {})
        model = OpenTabModel(
            embedding_size=config.get('embedding_size', 96),
            n_heads=config.get('n_heads', 4),
            n_layers=config.get('n_layers', 3),
            mlp_hidden_size=config.get('mlp_hidden', 192),
            n_outputs=config.get('max_classes', 10),
            dropout=config.get('dropout', 0.0),
        )
        model.load_state_dict(checkpoint['model_state'])
        classifier = OpenTabClassifier(model=model)
    else:
        model = OpenTabModel()
        classifier = OpenTabClassifier(model=model)
    
    results = []
    
    for X, y, name in datasets:
        print(f"\nEvaluating on {name}...")
        
        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, test_idx in cv.split(X_scaled, y_encoded):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
            
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            acc = (y_pred == y_test).mean()
            scores.append(acc)
        
        mean_acc = np.mean(scores)
        std_acc = np.std(scores)
        
        results.append({
            'dataset': name,
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'n_samples': len(X),
            'n_features': X.shape[1],
            'n_classes': len(np.unique(y)),
        })
        
        print(f"  Accuracy: {mean_acc:.4f} (+/- {std_acc:.4f})")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'custom_eval_results.csv'), index=False)
    
    print(f"\n{'=' * 60}")
    print("Summary:")
    print(results_df.to_string(index=False))
    print(f"\nResults saved to {output_dir}/custom_eval_results.csv")
    
    return results_df


def submit_to_leaderboard(
    results_dir: str,
    method_name: str = "OpenTab",
):
    """
    Generate submission files for the official TabArena leaderboard.
    
    Note: Official submission requires running the full TabArena evaluation
    and following the submission guidelines at:
    https://huggingface.co/spaces/TabArena/leaderboard
    """
    print("=" * 60)
    print("Leaderboard Submission")
    print("=" * 60)
    
    if not HAS_TABARENA:
        print("Error: TabArena not installed.")
        return
    
    from tabarena.nips2025_utils.end_to_end_single import EndToEndSingle, EndToEndResultsSingle
    from bencheval.website_format import format_leaderboard
    
    # Load cached results
    try:
        end_to_end_results = EndToEndResultsSingle.from_cache(method=method_name)
        leaderboard = end_to_end_results.compare_on_tabarena(
            only_valid_tasks=False,
            output_dir=Path(results_dir) / "submission",
        )
        
        leaderboard_formatted = format_leaderboard(leaderboard)
        
        print("\nSubmission Leaderboard:")
        print(leaderboard_formatted.to_markdown(index=False))
        
        submission_dir = Path(results_dir) / "submission"
        submission_dir.mkdir(parents=True, exist_ok=True)
        leaderboard_formatted.to_csv(submission_dir / "submission_leaderboard.csv", index=False)
        
        print(f"\n✓ Submission files saved to {submission_dir}")
        print("\nTo submit to the official leaderboard:")
        print("1. Verify results in the submission directory")
        print("2. Visit: https://huggingface.co/spaces/TabArena/leaderboard")
        print("3. Follow the submission instructions")
        
    except Exception as e:
        print(f"Error generating submission: {e}")
        print("\nMake sure you have run the full TabArena evaluation first:")
        print("  python evaluate.py --checkpoint your_model.pt --mode full")


def main():
    parser = argparse.ArgumentParser(description='Evaluate OpenTab on TabArena')
    
    parser.add_argument('--checkpoint', '-c', type=str, default=None,
                       help='Path to model checkpoint')
    parser.add_argument('--mode', '-m', type=str, default='quick',
                       choices=['quick', 'quick-regression', 'quick-all', 'lite', 'full', 'submit'],
                       help='Evaluation mode')
    parser.add_argument('--output', '-o', type=str, default='eval_results',
                       help='Output directory for results')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable result caching')
    
    args = parser.parse_args()
    
    if args.mode == 'quick':
        run_quick_evaluation(
            checkpoint_path=args.checkpoint,
            output_dir=args.output,
        )
    elif args.mode == 'quick-regression':
        quick_eval_regression(
            checkpoint_path=args.checkpoint,
            output_dir=args.output,
        )
    elif args.mode == 'quick-all':
        quick_eval_all(
            checkpoint_path=args.checkpoint,
            output_dir=args.output,
        )
    elif args.mode == 'lite':
        run_tabarena_lite(
            checkpoint_path=args.checkpoint,
            output_dir=args.output,
            cache=not args.no_cache,
        )
    elif args.mode == 'full':
        run_tabarena_full(
            checkpoint_path=args.checkpoint,
            output_dir=args.output,
            cache=not args.no_cache,
        )
    elif args.mode == 'submit':
        submit_to_leaderboard(
            results_dir=args.output,
        )


if __name__ == '__main__':
    main()
