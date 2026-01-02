"""
evaluate.py - Evaluation for OpenTab

This module provides tools to evaluate OpenTab on the TabArena benchmark.

TabArena is a living benchmark for tabular ML with 51 curated datasets and
standardized evaluation protocols.

Usage:
    # Quick classification evaluation on sklearn datasets
    python evaluate.py --checkpoint checkpoints/classifier.pt --mode quick
    
    # Quick regression evaluation on sklearn datasets  
    python evaluate.py --checkpoint checkpoints/regressor.pt --mode quick-regression
    
    # Full TabArena-Lite evaluation (classification only, 51 datasets, 1 fold)
    python evaluate.py --checkpoint checkpoints/classifier.pt --mode lite
    
    # Full TabArena evaluation (all datasets, all folds)
    python evaluate.py --checkpoint checkpoints/classifier.pt --mode full
    
    # Generate leaderboard from results directory (official TabArena approach)
    python evaluate.py --mode leaderboard --results eval_results --method OpenTab
    
    # Generate leaderboard from cached results
    python evaluate.py --mode leaderboard-cache --method OpenTab

Requirements:
    # For benchmark evaluation:
    pip install tabarena autogluon openml
    
    # Full TabArena installation (recommended):
    git clone https://github.com/autogluon/tabarena.git
    cd tabarena
    uv pip install --prerelease=allow -e ./tabarena[benchmark]

References:
    - TabArena: https://github.com/autogluon/tabarena
    - Leaderboard: https://huggingface.co/spaces/TabArena/leaderboard
"""

import argparse
import os
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import torch

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


# =============================================================================
# Simple TabArena Wrappers (following SimpleLightGBM pattern)
# =============================================================================

class OpenTabWrapper(AbstractExecModel):
    """
    Simple TabArena-compatible wrapper for OpenTab (Classification).
    
    Following the SimpleLightGBM pattern from TabArena:
    - AbstractExecModel handles preprocessing (preprocess_data=True by default)
    - We just implement _fit, _predict, _predict_proba
    """
    
    # TabPFN-style limits
    MAX_TRAIN_SAMPLES = 1024
    MAX_FEATURES = 100
    
    def __init__(self, checkpoint_path: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.checkpoint_path = checkpoint_path
        self._model = None
        self._n_classes = None
        self._max_features = None  # Will be set from checkpoint
    
    def _fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """Fit the OpenTab model on training data."""
        # Use CPU to avoid OOM on small GPUs
        device = 'cpu'  # torch.cuda.is_available() can use 'cuda' if you have enough GPU memory
        
        # Subsample if needed (TabPFN-style)
        if len(X) > self.MAX_TRAIN_SAMPLES:
            idx = np.random.RandomState(42).choice(len(X), self.MAX_TRAIN_SAMPLES, replace=False)
            X = X.iloc[idx]
            y = y.iloc[idx]
        
        # Convert to numpy
        X_np = X.fillna(0).values.astype(np.float32)
        y_np = y.values
        
        self._n_classes = len(np.unique(y_np))
        
        # Load model from checkpoint
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, map_location=device, weights_only=False)
            config = checkpoint.get('config', {})
            
            # Get max_features from config to ensure we truncate properly
            self._max_features = config.get('max_features', 100)
            
            model = OpenTabModel(
                embedding_size=config.get('embedding_size', 128),
                n_heads=config.get('n_heads', 4),
                n_layers=config.get('n_layers', 6),
                mlp_hidden_size=config.get('mlp_hidden_size', 256),
                n_outputs=config.get('max_classes', 10),
                max_features=self._max_features,
                dropout=config.get('dropout', 0.0),
            )
            model.load_state_dict(checkpoint['model_state'])
            self._model = OpenTabClassifier(model=model, device=device)
        else:
            self._max_features = self.MAX_FEATURES
            model = OpenTabModel(n_outputs=max(10, self._n_classes), max_features=self._max_features)
            self._model = OpenTabClassifier(model=model, device=device)
        
        # Limit features to what model was trained on
        if X_np.shape[1] > self._max_features:
            X_np = X_np[:, :self._max_features]
        
        self._model.fit(X_np, y_np)
        return self
    
    def _predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict class labels."""
        X_np = X.fillna(0).values.astype(np.float32)
        if X_np.shape[1] > self._max_features:
            X_np = X_np[:, :self._max_features]
        y_pred = self._model.predict(X_np)
        return pd.Series(y_pred, index=X.index)
    
    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict class probabilities."""
        X_np = X.fillna(0).values.astype(np.float32)
        if X_np.shape[1] > self._max_features:
            X_np = X_np[:, :self._max_features]
        probs = self._model.predict_proba(X_np)
        
        # Ensure correct number of columns
        if probs.shape[1] < self._n_classes:
            padded = np.zeros((probs.shape[0], self._n_classes), dtype=np.float32)
            padded[:, :probs.shape[1]] = probs
            probs = padded
        elif probs.shape[1] > self._n_classes:
            probs = probs[:, :self._n_classes]
        
        return pd.DataFrame(probs, index=X.index)
    
    def cleanup(self):
        """Release resources."""
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class OpenTabRegressionWrapper(AbstractExecModel):
    """
    Simple TabArena-compatible wrapper for OpenTab (Regression).
    """
    
    MAX_TRAIN_SAMPLES = 1024
    MAX_FEATURES = 100
    
    def __init__(self, checkpoint_path: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.checkpoint_path = checkpoint_path
        self._model = None
    
    def _fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """Fit the OpenTab regression model."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Subsample if needed
        if len(X) > self.MAX_TRAIN_SAMPLES:
            idx = np.random.RandomState(42).choice(len(X), self.MAX_TRAIN_SAMPLES, replace=False)
            X = X.iloc[idx]
            y = y.iloc[idx]
        
        # Limit features
        if X.shape[1] > self.MAX_FEATURES:
            X = X.iloc[:, :self.MAX_FEATURES]
        
        X_np = X.fillna(0).values.astype(np.float32)
        y_np = y.values.astype(np.float32)
        
        # Load model
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            self._model = OpenTabRegressor(self.checkpoint_path, device=device)
        else:
            model = OpenTabModel(n_outputs=64)  # 64 bins for regression
            self._model = OpenTabRegressor(model=model, device=device)
        
        self._model.fit(X_np, y_np)
        return self
    
    def _predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict regression values."""
        X_np = X.fillna(0).values.astype(np.float32)
        if X_np.shape[1] > self.MAX_FEATURES:
            X_np = X_np[:, :self.MAX_FEATURES]
        y_pred = self._model.predict(X_np)
        return pd.Series(y_pred, index=X.index)
    
    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Regression does not support predict_proba."""
        raise NotImplementedError("Regression does not support predict_proba")
    
    def cleanup(self):
        """Release resources."""
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# =============================================================================
# Config generator for TabArena (following custom_random_forest_model.py pattern)
# =============================================================================

def get_opentab_configs(checkpoint_path: str, num_random_configs: int = 0):
    """
    Generate TabArena experiment configurations for OpenTab.
    
    Following the pattern from examples/benchmarking/custom_tabarena_model/
    """
    if not HAS_TABARENA:
        raise ImportError("TabArena not installed")
    
    from tabarena.utils.config_utils import ConfigGenerator
    from tabarena.benchmark.experiment import Experiment
    
    # Manual configs (just the default for now)
    manual_configs = [
        {'checkpoint_path': checkpoint_path},
    ]
    
    # No search space for now (TabPFN-style models don't need HPO)
    search_space = {}
    
    gen = ConfigGenerator(
        model_cls=OpenTabWrapper,
        manual_configs=manual_configs,
        search_space=search_space,
    )
    
    return gen.generate_all_bag_experiments(
        num_random_configs=num_random_configs,
        fold_fitting_strategy="sequential_local",
    )


# =============================================================================
# Quick evaluation (without full TabArena)
# =============================================================================

def run_quick_evaluation(checkpoint_path: Optional[str], output_dir: str = 'eval_results'):
    """Quick evaluation on sklearn datasets."""
    from sklearn.datasets import load_iris, load_wine, load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    print("=" * 60)
    print("Quick Evaluation")
    print("=" * 60)
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
        classifier = OpenTabClassifier(model=model, device=device)
        print(f"Loaded model from {checkpoint_path}")
    else:
        classifier = OpenTabClassifier(device=device)
        print("Using untrained model")
    
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
        
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        
        results.append({'dataset': name, 'accuracy': accuracy})
        print(f"  {name}: {accuracy:.4f}")
    
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(results).to_csv(f"{output_dir}/quick_eval.csv", index=False)
    print(f"\nResults saved to {output_dir}/quick_eval.csv")
    return results


def quick_eval_regression(checkpoint_path: Optional[str], output_dir: str = 'eval_results'):
    """Quick regression evaluation."""
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score
    
    print("=" * 60)
    print("Quick Regression Evaluation")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if checkpoint_path and os.path.exists(checkpoint_path):
        regressor = OpenTabRegressor(checkpoint_path, device=device)
        print(f"Loaded model from {checkpoint_path}")
    else:
        regressor = OpenTabRegressor(device=device)
        print("Using untrained model")
    
    data = load_diabetes()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"  Diabetes: RMSE={rmse:.4f}, R2={r2:.4f}")
    
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame([{'dataset': 'Diabetes', 'rmse': rmse, 'r2': r2}]).to_csv(
        f"{output_dir}/quick_eval_regression.csv", index=False
    )
    return {'rmse': rmse, 'r2': r2}


# =============================================================================
# TabArena evaluation
# =============================================================================

def run_tabarena_lite(checkpoint_path: str, output_dir: str = 'tabarena_results'):
    """
    Run TabArena-Lite evaluation following the official pattern.
    
    This uses the pattern from examples/benchmarking/custom_tabarena_model/
    """
    if not HAS_TABARENA:
        print("Error: TabArena not installed")
        return None
    
    from tabarena.benchmark.experiment import Experiment, run_experiments_new
    
    print("=" * 60)
    print("TabArena-Lite Evaluation")
    print("=" * 60)
    
    # Get all TabArena-v0.1 tasks
    task_ids = openml.study.get_suite("tabarena-v0.1").tasks
    
    # Create experiment with our wrapper
    experiments = [
        Experiment(
            name="OpenTab",
            method_cls=OpenTabWrapper,
            method_kwargs={'checkpoint_path': checkpoint_path},
        ),
    ]
    
    # Run experiments (TabArena-Lite = 1 fold per dataset)
    run_experiments_new(
        output_dir=output_dir,
        model_experiments=experiments,
        tasks=task_ids,
        repetitions_mode="TabArena-Lite",
    )
    
    print(f"\nResults saved to {output_dir}")
    return output_dir


def run_tabarena_full(checkpoint_path: str, output_dir: str = 'tabarena_results_full'):
    """Run full TabArena evaluation (multiple folds)."""
    if not HAS_TABARENA:
        print("Error: TabArena not installed")
        return None
    
    from tabarena.benchmark.experiment import Experiment, run_experiments_new
    
    print("=" * 60)
    print("TabArena Full Evaluation")
    print("=" * 60)
    
    task_ids = openml.study.get_suite("tabarena-v0.1").tasks
    
    experiments = [
        Experiment(
            name="OpenTab",
            method_cls=OpenTabWrapper,
            method_kwargs={'checkpoint_path': checkpoint_path},
        ),
    ]
    
    run_experiments_new(
        output_dir=output_dir,
        model_experiments=experiments,
        tasks=task_ids,
        repetitions_mode="TabArena",  # Full mode
    )
    
    print(f"\nResults saved to {output_dir}")
    return output_dir


# =============================================================================
# Leaderboard Generation with TabArena (Official Approach)
# =============================================================================

def generate_leaderboard_from_results(
    results_dir: str,
    output_dir: str = 'leaderboard',
    method_name: str = 'OpenTab',
    only_valid_tasks: bool = True,
):
    """
    Generate leaderboard by comparing OpenTab results against TabArena baselines.
    
    This uses the official TabArena EndToEndSingle approach which:
    1. Loads raw artifacts from results.pkl files
    2. Processes and caches results
    3. Compares against TabArena leaderboard methods
    4. Generates figures and formatted leaderboard
    
    Args:
        results_dir: Directory containing results.pkl files from TabArena evaluation
        output_dir: Directory to save leaderboard outputs and figures
        method_name: Name of the method (default: 'OpenTab')
        only_valid_tasks: If True, only compare on tasks with results (no imputation)
    
    Returns:
        pd.DataFrame: Leaderboard with ELO ratings and other metrics
    """
    try:
        from tabarena.nips2025_utils.end_to_end_single import EndToEndResultsSingle, EndToEndSingle
        from bencheval.website_format import format_leaderboard
    except ImportError:
        print("Error: TabArena not properly installed")
        print("Install with:")
        print("  pip install -e ./tabarena")
        print("  pip install -e ./bencheval")
        print("Or: uv pip install --prerelease=allow -e ./tabarena")
        return None
    
    print("=" * 80)
    print("Generating Leaderboard with TabArena (Official Approach)")
    print("=" * 80)
    
    path_raw = Path(results_dir)
    fig_output_dir = Path(output_dir)
    
    print(f"\nLoading results from: {path_raw}")
    print(f"Output directory: {fig_output_dir}")
    print(f"Method: {method_name}")
    
    # Step 1: Process raw results and cache them
    # This loads results.pkl files, infers metadata, and generates processed results
    print("\n[1/3] Processing raw results...")
    try:
        end_to_end = EndToEndSingle.from_path_raw(path_raw=path_raw)
        end_to_end_results = end_to_end.to_results()
        print("✓ Results processed and cached")
    except Exception as e:
        print(f"Error processing results: {e}")
        print("\nTrying to load from cache...")
        try:
            end_to_end_results = EndToEndResultsSingle.from_cache(method=method_name)
            print("✓ Loaded results from cache")
        except Exception as e2:
            print(f"Error loading from cache: {e2}")
            return None
    
    # Step 2: Compare against TabArena leaderboard
    print("\n[2/3] Comparing against TabArena leaderboard...")
    os.makedirs(fig_output_dir, exist_ok=True)
    
    leaderboard = end_to_end_results.compare_on_tabarena(
        only_valid_tasks=only_valid_tasks,
        output_dir=fig_output_dir,
    )
    
    # Step 3: Format and display leaderboard
    print("\n[3/3] Formatting leaderboard...")
    leaderboard_website = format_leaderboard(leaderboard)
    
    # Save leaderboard
    leaderboard_path = fig_output_dir / 'leaderboard.csv'
    leaderboard.to_csv(leaderboard_path)
    
    leaderboard_formatted_path = fig_output_dir / 'leaderboard_formatted.csv'
    leaderboard_website.to_csv(leaderboard_formatted_path, index=False)
    
    # Print results
    print("\n" + "=" * 80)
    print("TABARENA LEADERBOARD")
    print("=" * 80)
    print(leaderboard_website.to_markdown(index=False))
    
    print(f"\n✓ Leaderboard saved to: {leaderboard_path}")
    print(f"✓ Formatted leaderboard saved to: {leaderboard_formatted_path}")
    print(f"✓ Figures saved to: {fig_output_dir}")
    
    return leaderboard


def generate_leaderboard_from_cache(
    method_name: str = 'OpenTab',
    output_dir: str = 'leaderboard',
    only_valid_tasks: bool = True,
):
    """
    Generate leaderboard from cached results (after running lite/full evaluation).
    
    Use this after running `--mode lite` or `--mode full` to generate the leaderboard
    without re-processing raw results.
    
    Args:
        method_name: Name of the method to load from cache
        output_dir: Directory to save leaderboard outputs
        only_valid_tasks: If True, only compare on tasks with results
    
    Returns:
        pd.DataFrame: Leaderboard
    """
    try:
        from tabarena.nips2025_utils.end_to_end_single import EndToEndResultsSingle
        from bencheval.website_format import format_leaderboard
    except ImportError:
        print("Error: TabArena not properly installed")
        return None
    
    print("=" * 80)
    print("Generating Leaderboard from Cached Results")
    print("=" * 80)
    
    fig_output_dir = Path(output_dir)
    
    print(f"\nLoading cached results for method: {method_name}")
    
    # Load from cache
    end_to_end_results = EndToEndResultsSingle.from_cache(method=method_name)
    
    # Compare against TabArena
    os.makedirs(fig_output_dir, exist_ok=True)
    leaderboard = end_to_end_results.compare_on_tabarena(
        only_valid_tasks=only_valid_tasks,
        output_dir=fig_output_dir,
    )
    
    # Format and display
    leaderboard_website = format_leaderboard(leaderboard)
    
    # Save
    leaderboard.to_csv(fig_output_dir / 'leaderboard.csv')
    leaderboard_website.to_csv(fig_output_dir / 'leaderboard_formatted.csv', index=False)
    
    print("\n" + "=" * 80)
    print("TABARENA LEADERBOARD")
    print("=" * 80)
    print(leaderboard_website.to_markdown(index=False))
    
    return leaderboard


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate OpenTab on TabArena')
    parser.add_argument('--checkpoint', '-c', type=str, default=None,
                       help='Path to model checkpoint')
    parser.add_argument('--mode', '-m', type=str, default='quick',
                       choices=['quick', 'quick-regression', 'lite', 'full', 
                               'leaderboard', 'leaderboard-cache'],
                       help='Evaluation mode')
    parser.add_argument('--output', '-o', type=str, default='eval_results',
                       help='Output directory for results')
    parser.add_argument('--results', '-r', type=str, default=None,
                       help='Path to results directory (for leaderboard mode)')
    parser.add_argument('--method', type=str, default='OpenTab',
                       help='Method name for leaderboard')
    
    args = parser.parse_args()
    
    if args.mode == 'quick':
        run_quick_evaluation(args.checkpoint, args.output)
    elif args.mode == 'quick-regression':
        quick_eval_regression(args.checkpoint, args.output)
    elif args.mode == 'lite':
        run_tabarena_lite(args.checkpoint, args.output)
    elif args.mode == 'full':
        run_tabarena_full(args.checkpoint, args.output)
    elif args.mode == 'leaderboard':
        # Use results directory from --results or default to eval_results
        results_dir = args.results if args.results else args.output
        generate_leaderboard_from_results(
            results_dir=results_dir, 
            output_dir=args.output + '/leaderboard',
            method_name=args.method,
        )
    elif args.mode == 'leaderboard-cache':
        # Load from cache (after running lite/full)
        generate_leaderboard_from_cache(
            method_name=args.method,
            output_dir=args.output + '/leaderboard',
        )


if __name__ == '__main__':
    main()
