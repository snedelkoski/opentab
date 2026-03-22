# OpenTab — Repository Analysis

## What does this project do?

OpenTab is a minimal Python implementation of a TabPFN-style Prior-Data Fitted Network for tabular data. It uses in-context learning: a Transformer trained on synthetic SCM-generated datasets makes predictions at inference time without gradient updates. The model treats each cell in a table as a separate position and uses dual attention (inter-feature and inter-sample) to learn from training examples provided as context. It supports both classification and regression, and includes sklearn-compatible wrappers (`OpenTabClassifier`, `OpenTabRegressor`) and a TabArena benchmark integration.

## Primary success metric

**Mean classification accuracy on small tabular datasets (Iris, Wine, Breast Cancer) using an untrained model, normalized to [0.0, 1.0].**

Since the autoimprove loop cannot train a full model within 300 seconds (the training loop is designed for ~2M steps at batch size 64), the primary metric evaluates the model architecture and data generation pipeline quality by running a short training session (as many steps as fit in ~240s) and then measuring classification accuracy on 3 sklearn datasets (Iris, Wine, Breast Cancer). The score is the mean accuracy across these datasets — already naturally in [0.0, 1.0].

This metric captures improvements to:
- Model architecture (embedding, attention, decoder)
- Training loop efficiency (optimizer, scheduler, loss computation)
- Synthetic data generation quality (SCM structure, edge mappings, post-processing)
- sklearn wrapper correctness (fit/predict/predict_proba)

## File map

```
model.py           — Transformer architecture (encoder, dual attention, decoder) + sklearn-compatible wrappers
train.py           — Training loop with online/offline data, gradient accumulation, mixed precision, evaluation
generate_data.py   — SCM-based synthetic tabular data generation (DAG sampling, edge mappings, post-processing)
evaluate.py        — Evaluation: sklearn quick-eval + TabArena benchmark integration
__init__.py        — Empty package marker
pyproject.toml     — Project metadata, dependencies, tool config (ruff, black, pytest)
tests/test_smoke.py — 4 smoke tests: generator shapes, classifier contract, collation, config resolution
tests/conftest.py  — Adds repo root to sys.path for test imports
```

## Tech stack

- **Language**: Python 3.11+
- **Framework**: PyTorch (model + training), scikit-learn (wrappers + eval datasets)
- **Package manager**: uv (with uv.lock)
- **Test runner**: pytest
- **Linter**: ruff
- **Formatter**: black (line length 100)

## How to build

N/A (pure Python, installed via `uv sync` or `pip install -e .`)

## How to test

```bash
uv run pytest tests/
```

## How to run

```bash
# Train
python train.py --online --steps 100000

# Evaluate
python evaluate.py --checkpoint checkpoints/final_model.pt --mode quick
```

## How to evaluate the primary metric

Run a short training session (bounded to ~240s) then evaluate classification accuracy on Iris, Wine, and Breast Cancer. The evaluator script handles this end-to-end and outputs a JSON score.

```bash
uv run .autoimprove/evaluators/train_and_eval.py
```

## Editable files

- `model.py` — Transformer architecture, encoder, attention layers, decoder, sklearn wrappers
- `train.py` — Training loop, data loading, optimizer setup, loss computation, evaluation
- `generate_data.py` — SCM-based synthetic data generation pipeline
- `evaluate.py` — Evaluation scripts and TabArena wrappers

## Fixed files

- `tests/` — Test suite (ground truth for correctness)
- `.autoimprove/` — Evaluation system (must not be modified after baseline)
- `pyproject.toml` — Build config, dependencies
- `uv.lock` — Dependency lockfile
- `__init__.py` — Package marker
- `.gitignore` — Git config

## Current quality issues

1. **No comprehensive test suite** — Only 4 smoke tests covering basic shapes and contracts. No tests for training loop, data generation edge cases, regression wrapper, or missing value handling.
2. **Whitespace lint violations** — `evaluate.py` has trailing whitespace and blank-line-with-whitespace warnings from ruff.
3. **Debug print statement** — `evaluate.py:185` has `print("All predictions", all_predictions)` which is a debug artifact that should not be in production code.
4. **Unused imports** — `evaluate.py` imports `scipy.stats` at line 59 and again inside `_predict` at line 184.
5. **Type annotation gaps** — Some functions (e.g., `_build_tree`, `_evaluate` in `generate_data.py`) lack return type annotations.
6. **Hardcoded constants** — Magic numbers scattered throughout (e.g., `-999.0` sentinel, `75000` max cells, temperature `0.9`). Some are documented, but consistency could improve.
7. **Training loop processes one sample at a time** — The `compute_loss` method in `train.py` iterates over batch items one by one (`for i in range(batch_size)`), which is inherently slow and doesn't leverage batched GPU computation.
8. **No gradient checkpointing** — For memory-constrained training, gradient checkpointing could help.
9. **DecisionTreeMapping is O(n_samples)** — The decision tree edge mapping evaluates samples one by one in a Python loop, which is very slow for large datasets.
