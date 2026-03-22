# OpenTab — Agent Guidelines

OpenTab is a minimal Python implementation of a TabPFN-style Prior-Data Fitted Network for tabular data. It uses in-context learning: a Transformer trained on synthetic SCM-generated datasets makes predictions at inference time without gradient updates.

## Repository Layout

```
opentab/
├── model.py           # Transformer architecture + sklearn-compatible wrappers
├── generate_data.py   # SCM-based synthetic tabular data generation
├── train.py           # Training loop (offline HDF5 or online generation)
├── evaluate.py        # Evaluation: sklearn quick-eval + TabArena benchmark
├── __init__.py        # Empty package marker
├── pyproject.toml     # Project metadata, deps, tool config
├── data/              # Gitignored — pre-generated HDF5 datasets
├── checkpoints/       # Gitignored — .pt model checkpoints
└── eval_results/      # Gitignored — evaluation output CSVs
```

All source lives in four flat `.py` files at the repo root. There is no `src/` layout and no subdirectories for source code.

## Environment Setup

```bash
# Recommended (uses uv lockfile for reproducibility)
uv sync

# Alternative
pip install -e ".[dev]"

# With optional extras
pip install -e ".[dev,training,benchmark]"
```

Requires Python ≥ 3.11.

## Commands

### Data Generation
```bash
python generate_data.py --n_datasets 1000000 --output data/synthetic.h5
# or after install:
opentab-generate --n_datasets 1000000 --output data/synthetic.h5
```

### Training
```bash
python train.py --online --steps 100000
python train.py --dataset data/synthetic.h5 --steps 50000  # offline HDF5
opentab-train --online --steps 100000
```

### Evaluation
```bash
python evaluate.py --checkpoint checkpoints/final_model.pt --mode lite
python evaluate.py --mode leaderboard --results eval_results --method OpenTab
opentab-evaluate --checkpoint checkpoints/final_model.pt --mode lite
```

### Linting and Formatting
```bash
ruff check .          # lint
ruff check --fix .    # lint with auto-fix
black .               # format
ruff format .         # format (alternative)
```

### Testing
```bash
pytest                                          # all tests
pytest tests/test_foo.py                        # single file
pytest tests/test_foo.py::test_function_name    # single test
pytest -x                                       # stop on first failure
pytest -k "keyword"                             # filter by name
```

Tests go in `tests/` with filenames matching `test_*.py`. There are currently no tests; add them as `tests/test_<module>.py`.

## Code Style

### Formatter: Black
- Line length: **100**
- Configured in `pyproject.toml` under `[tool.black]`

### Linter: Ruff
- Line length: **100**
- Rules enabled: `E` (pycodestyle errors), `F` (pyflakes), `W` (pycodestyle warnings), `I` (isort)
- `E501` (line-too-long) is ignored — Black handles line length
- Configured in `pyproject.toml` under `[tool.ruff]`

Always run `ruff check --fix . && black .` before committing.

## Naming Conventions

| Entity | Style | Examples |
|---|---|---|
| Files | `snake_case.py` | `model.py`, `generate_data.py` |
| Classes | `PascalCase` | `OpenTabModel`, `SCMDataGenerator`, `TrainConfig` |
| Functions / methods | `snake_case` | `sample_hyperparameters()`, `get_random_activation()` |
| Variables / parameters | `snake_case` | `embedding_dim`, `batch_size`, `missing_mask` |
| Constants / module-level flags | `UPPER_SNAKE_CASE` | `HAS_H5PY`, `ACTIVATION_FUNCTIONS` |
| Private / internal | leading `_` | `_build_tree()`, `_evaluate()`, `_fit()` |
| Dataclass fields | `snake_case` | `train_size`, `n_features`, `checkpoint_dir` |

## Type Annotations

Use type hints on all function signatures and dataclass fields. Use `typing` module types for Python 3.9 compatibility:

```python
from typing import Optional, Tuple, Dict, List, Union, Callable, Iterator

def foo(x: np.ndarray, n: int = 10) -> Optional[torch.Tensor]:
    ...
```

Use `@dataclass` for all configuration/hyperparameter objects rather than plain dicts or `argparse.Namespace`.

## Import Conventions

- **Absolute imports only** — no relative imports.
- **Import order (enforced by ruff/isort):** standard library → third-party → local modules.
- **Optional dependencies** use a guarded `try/except ImportError` pattern with a boolean flag:

```python
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
```

- Check the flag before use and raise `ImportError` with an install hint:

```python
if not HAS_H5PY:
    raise ImportError("h5py required: pip install h5py")
```

- Cross-module imports use direct module names: `from model import OpenTabModel`.
- `__init__.py` is intentionally empty — do not add re-exports.

## Error Handling

1. **Precondition violations** → raise `ValueError` (sklearn convention):
   ```python
   if self.X_train_ is None:
       raise ValueError("Must call fit() first")
   ```

2. **Hot-path exceptions** (data generation, training loop) → `except Exception: continue` to keep long runs alive. Log or count skips where useful.

3. **NaN/Inf guards** before loss steps:
   ```python
   if torch.isnan(X_i).any() or torch.isinf(X_i).any():
       continue
   if torch.isnan(loss):
       continue
   ```

4. **Numerical stability** → clip values and add epsilon where needed:
   ```python
   np.clip(x, -500, 500)
   p = np.clip(p, 1e-8, 1 - 1e-8)
   ```

5. Do not define custom exception classes unless a new public API genuinely requires distinguishable exception types.

## Key Architecture Patterns

### sklearn-Compatible Wrappers
`OpenTabClassifier` and `OpenTabRegressor` expose `.fit(X, y)` / `.predict(X)` / `.predict_proba(X)`. `fit()` only stores training data — no weight updates occur. All learning is in-context at inference time.

### Callable Edge Mappings
`NeuralNetworkMapping`, `DecisionTreeMapping`, and `CategoricalDiscretization` implement `__call__`, making them interchangeable as `Callable[[np.ndarray], np.ndarray]` in the SCM graph.

### Dual Attention
`InterFeatureAttention` attends across features within a row; `InterSampleAttention` attends across samples within a feature column. Test samples attend only to training samples, never to each other.

### Missing Values
Represented as `float('nan')` in numpy and `-999.0` as a sentinel in torch tensors. The encoder zeros them out and sets a binary `missing_indicator` channel.

### Checkpoint Format
```python
{
    "model_state": ...,
    "optimizer_state": ...,
    "scheduler_state": ...,
    "global_step": int,
    "best_loss": float,
    "config": dict,       # nested hyperparameter dict
    "log_history": list,
}
```
Load with `torch.load(path, weights_only=False)`.

### Gradient Accumulation
Manual accumulation with an `is_accumulating` flag: `loss / grad_accum_steps → .backward() → conditional optimizer.step() + zero_grad()`.

## Dependencies

Core: `torch>=2.0.0`, `numpy>=1.21.0`, `scikit-learn>=1.0.0`, `h5py>=3.7.0`, `pandas>=2.3.3`, `tqdm>=4.60.0`  
Dev: `pytest>=7.0.0`, `jupyter`, `ipykernel`  
Training extras: `schedulefree`, `wandb`  
Benchmark extras: `autogluon`, `openml`, `tabarena` (from git), `ray`
