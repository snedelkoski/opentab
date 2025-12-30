# OpenTab

An open implementation of TabPFN-like models (Prior-Data Fitted Network for Tabular Data) with **full synthetic data generation** from scratch.

## Quick Start

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync

# Install TabArena for benchmarking (optional)
uv pip install "tabarena @ git+https://github.com/autogluon/tabarena.git#subdirectory=tabarena"

# Generate synthetic data for classification
python generate_data.py --prior mixed --n_datasets 100000 --output data/synthetic_clf.h5

# Generate synthetic data for regression
python generate_data.py --prior mixed_regression --n_datasets 100000 --output data/synthetic_reg.h5

# Train classification model
python train.py --data data/synthetic_clf.h5 --task classification --epochs 100 --output checkpoints/classifier.pt

# Train regression model (separate model, following TabPFN approach)
python train.py --data data/synthetic_reg.h5 --task regression --epochs 100 --output checkpoints/regressor.pt

# Evaluate classification (quick test on sklearn datasets)
python evaluate.py --checkpoint checkpoints/classifier.pt --mode quick

# Evaluate regression
python evaluate.py --checkpoint checkpoints/regressor.pt --mode quick-regression

# Evaluate on TabArena benchmark (51 datasets)
python evaluate.py --checkpoint checkpoints/classifier.pt --mode lite

# Generate leaderboard comparing against SOTA methods
python evaluate.py --mode leaderboard --results eval_results --method OpenTab
```

## Two-Model Approach

Following [TabPFN](https://github.com/PriorLabs/TabPFN), OpenTab trains **separate models** for classification and regression:

| Task | Prior Type | Model Class | Checkpoint |
|------|------------|-------------|------------|
| Classification | `augmented_mixed` (MLP, GP, Tree, SCM) | `OpenTabClassifier` | `classifier.pt` |
| Regression | `augmented_mixed_regression` (MLP, GP, Linear) | `OpenTabRegressor` | `regressor.pt` |


## Use as a Classifier

```python
from model import OpenTabClassifier

clf = OpenTabClassifier("checkpoints/classifier.pt")
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)
```

## Use as a Regressor

```python
from model import OpenTabRegressor

reg = OpenTabRegressor("checkpoints/regressor.pt")
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)  # Returns mean predictions
predictions = reg.predict(X_test, output_type='median')  # Or median
quantiles = reg.predict_quantiles(X_test, quantiles=[0.1, 0.5, 0.9])
```

## What is TabPFN?

TabPFN learns to **approximate Bayesian inference** on tabular data. It's trained on millions of synthetic datasets, then at inference time performs **in-context learning** - a single forward pass, no gradient updates.

```
Training:   Generate synthetic data → Train Transformer → Save checkpoint
Inference:  clf.fit(X_train, y_train) → clf.predict(X_test)  # One forward pass!
```

## Repository Structure

```
model.py          # Transformer architecture + OpenTabClassifier/Regressor
generate_data.py  # Synthetic data generation (classification & regression priors)
train.py          # Training loop with task-specific evaluation
evaluate.py       # TabArena benchmark + quick evaluation + TabArena wrappers
```

## TabArena Benchmark Integration

OpenTab includes full [TabArena](https://github.com/autogluon/tabarena) compatibility for standardized benchmarking against state-of-the-art tabular ML methods.

### Installation

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Install TabArena
uv pip install "tabarena @ git+https://github.com/autogluon/tabarena.git#subdirectory=tabarena"
```

### Evaluation Modes

| Mode | Description | Command |
|------|-------------|---------|
| `quick` | Test on sklearn classification datasets | `python evaluate.py --mode quick` |
| `quick-regression` | Test on sklearn regression datasets | `python evaluate.py --mode quick-regression` |
| `lite` | TabArena-Lite (51 datasets, 1 fold) | `python evaluate.py --mode lite` |
| `full` | Full TabArena (all datasets, all folds) | `python evaluate.py --mode full` |
| `leaderboard` | Generate leaderboard with ELO ratings | `python evaluate.py --mode leaderboard --results eval_results` |
| `leaderboard-cache` | Load leaderboard from cache | `python evaluate.py --mode leaderboard-cache --method OpenTab` |

### Complete Evaluation Workflow

```bash
# Step 1: Run TabArena-Lite evaluation (51 datasets, 1 fold each)
python evaluate.py --checkpoint checkpoints/classifier.pt --mode lite

# Step 2: Generate leaderboard comparing against all TabArena baselines
python evaluate.py --mode leaderboard --results eval_results --method OpenTab
```

This generates:
- **ELO ratings** with 95% confidence intervals
- **Win rates** against all methods
- **Average ranks** across datasets
- **Improvability scores**
- **Comparison figures** saved to `eval_results/leaderboard/`

### TabArena Wrappers

Two wrapper classes implement the `AbstractExecModel` interface for TabArena compatibility:

```python
from evaluate import OpenTabWrapper, OpenTabRegressionWrapper

# Classification (binary/multiclass)
clf = OpenTabWrapper(problem_type='multiclass', eval_metric=metric)
clf._fit(X_train, y_train)
predictions = clf._predict(X_test)
probabilities = clf._predict_proba(X_test)
clf.cleanup()  # Release GPU memory

# Regression
reg = OpenTabRegressionWrapper(problem_type='regression', eval_metric=metric)
reg._fit(X_train, y_train)
predictions = reg._predict(X_test)
reg.cleanup()
```

### Wrapper Features

| Feature | OpenTabWrapper | OpenTabRegressionWrapper |
|---------|----------------|--------------------------|
| Problem types | `binary`, `multiclass` | `regression` |
| `_fit()` with resources | ✅ `num_cpus`, `num_gpus`, `time_limit` | ✅ |
| `_predict()` | ✅ Returns `pd.Series` | ✅ Returns `pd.Series` |
| `_predict_proba()` | ✅ Returns `pd.DataFrame` | ❌ (raises `NotImplementedError`) |
| `cleanup()` | ✅ Releases GPU memory | ✅ |
| Batched inference | ✅ (64 samples/batch) | ✅ |
| Auto subsampling | ✅ (512 train samples max) | ✅ |

## Key Commands

| Task | Command |
|------|---------|
| Generate classification data | `python generate_data.py --prior mixed --n_datasets 100000 --output data/clf.h5` |
| Generate regression data | `python generate_data.py --prior mixed_regression --n_datasets 100000 --output data/reg.h5` |
| Generate with augmentation | `python generate_data.py --prior augmented_mixed --n_datasets 100000` |
| Train classification | `python train.py --online --task classification --epochs 100` |
| Train regression | `python train.py --online --task regression --epochs 100` |
| Quick eval (classification) | `python evaluate.py --checkpoint model.pt --mode quick` |
| Quick eval (regression) | `python evaluate.py --checkpoint model.pt --mode quick-regression` |
| TabArena-Lite | `python evaluate.py --checkpoint model.pt --mode lite` |
| TabArena-Full | `python evaluate.py --checkpoint model.pt --mode full` |
| Generate Leaderboard | `python evaluate.py --mode leaderboard --results eval_results` |

## Configuration

**Data Generation:**
- `--n_datasets`: Number of synthetic datasets (default: 100000)
- `--max_samples`: Max samples per dataset (default: 100)
- `--max_features`: Max features (default: 20)
- `--prior`: Prior type - `mlp`, `gp`, `scm`, `tree`, `mixed` for classification; `mlp_regression`, `gp_regression`, `linear_regression`, `mixed_regression` for regression
- **Augmented priors**: Use `augmented_*` prefix (e.g., `augmented_mixed`) to add categorical features and missing values

**Training:**
- `--task`: Task type - `classification` or `regression`
- `--epochs`: Training epochs (default: 100)
- `--batch_size`: Batch size (default: 32)
- `--embedding_size`: Embedding dimension (default: 96)
- `--n_layers`: Transformer layers (default: 3)

**Evaluation:**
- `--mode quick`: Test classification on sklearn datasets (Iris, Wine, Breast Cancer)
- `--mode quick-regression`: Test regression on sklearn datasets (Diabetes, California Housing)
- `--mode lite`: TabArena-Lite (51 classification datasets, 1 fold)
- `--mode full`: Full TabArena (all folds)

## Handling Categorical Features and Missing Values

Following the [TabPFN Nature paper](https://www.nature.com/articles/s41586-024-08328-6), OpenTab handles real-world data challenges through **training-time augmentation**:

### Categorical Features
- **During training**: Random features are converted to categorical via percentile binning
- **During inference**: Categorical columns are ordinal-encoded (matching the training format)
- Controlled by `categorical_prob` parameter (default: 0.3 = 30% of features)

### Missing Values
- **During training**: Random values are masked with a special indicator (-999.0)
- **Four missingness patterns** (matching real-world scenarios):
  - `random`: Completely random missingness
  - `column`: Some columns have high missingness rates
  - `row`: Some rows have many missing values
  - `block`: Correlated block missingness
- **Model learns**: Dedicated `missing_embedding` and `missing_indicator_embedding` parameters
- Controlled by `missing_prob` parameter (default: 0.1 = 10% of values)

### Usage

```python
from generate_data import get_prior

# Standard prior (no augmentation)
prior = get_prior('mixed')

# Augmented prior with categorical features and missing values
prior = get_prior('augmented_mixed', 
                  categorical_prob=0.3,  # 30% of features become categorical
                  missing_prob=0.1,      # 10% of values are missing
                  augment_prob=0.7)      # Apply augmentation 70% of the time

# Generate training data
dataset = prior.generate(n_samples=100, n_features=10, n_classes=3)

# Metadata available
print(f"Categorical features: {dataset.categorical_mask}")
print(f"Missing values: {dataset.missing_mask.sum()}")
```

## References

```bibtex
@article{hollmann2023tabpfn,
  title={TabPFN: A transformer that solves small tabular classification problems in a second},
  author={Hollmann, Noah and Müller, Samuel and Eggensperger, Katharina and Hutter, Frank},
  booktitle={ICLR 2023},
  year={2023}
}

@article{erickson2025tabarena,
  title={TabArena: A Living Benchmark for Machine Learning on Tabular Data}, 
  author={Nick Erickson and Lennart Purucker and Andrej Tschalzev and David Holzmüller and
          Prateek Mutalik Desai and David Salinas and Frank Hutter},
  year={2025},
  journal={arXiv preprint arXiv:2506.16791},
  url={https://arxiv.org/abs/2506.16791}, 
}
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to submit pull requests, report issues, and contribute to the project.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
