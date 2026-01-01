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

## TabArena Benchmark Results

Here's what you can achieve with a model trained on **1M synthetic datasets**, **200 max samples**, **50 max features**, **10 classes** for just **1 epoch** on a modest **NVIDIA GeForce RTX 3050 4GB** laptop GPU:

<details>
<summary>Click to expand full leaderboard</summary>

```
================================================================================
TABARENA LEADERBOARD
================================================================================
|   # | Model                                             |   Elo [⬆️] | Elo 95% CI   |   Score [⬆️] |   Rank [⬇️] |   Harmonic Rank [⬇️] |   Improvability (%) [⬇️] |   Median Train Time (s/1K) [⬇️] |   Median Predict Time (s/1K) [⬇️] | Verified   |   Imputed (%) [⬇️] | Imputed   | Hardware   |
|----:|:--------------------------------------------------|-----------:|:-------------|-------------:|------------:|---------------------:|-------------------------:|--------------------------------:|----------------------------------:|:-----------|-------------------:|:----------|:-----------|
|   0 | RealTabPFN-v2.5 (tuned + ensembled)               |       1632 | +101/-70     |        0.688 |        8.22 |                 2.63 |                    3.759 |                         2059.94 |                             9.785 | Unknown    |               0    | False     | Unknown    |
|   1 | AutoGluon 1.4 (extreme, 4h)                       |       1612 | +90/-77      |        0.661 |        8.89 |                 3.98 |                    5.868 |                          556.15 |                             6.31  | Unknown    |               0    | False     | Unknown    |
|   2 | RealTabPFN-v2.5 (tuned)                           |       1582 | +86/-66      |        0.614 |       10.06 |                 4.34 |                    5.875 |                         2059.94 |                             1.03  | Unknown    |               0    | False     | Unknown    |
|   3 | AutoGluon 1.4 (best, 4h)                          |       1548 | +67/-63      |        0.564 |       11.43 |                 4.87 |                    7.644 |                         1754.94 |                             1.767 | Unknown    |               0    | False     | Unknown    |
|   4 | RealTabPFN-v2.5 (default)                         |       1541 | +72/-51      |        0.565 |       11.75 |                 6.17 |                    6.522 |                            5.71 |                             0.611 | Unknown    |               0    | False     | Unknown    |
|   5 | RealMLP_GPU (tuned + ensembled)                   |       1513 | +58/-50      |        0.509 |       13    |                 7.6  |                    8.756 |                         2791.97 |                            13.886 | Unknown    |               0    | False     | Unknown    |
|   6 | TabDPT_GPU (tuned + ensembled)                    |       1439 | +68/-59      |        0.452 |       16.78 |                 5.02 |                    9.172 |                         6154.73 |                           386.167 | Unknown    |               0    | False     | Unknown    |
|   7 | RealMLP_GPU (tuned)                               |       1437 | +57/-56      |        0.415 |       16.91 |                 8.43 |                   10.281 |                         2791.97 |                             0.373 | Unknown    |               0    | False     | Unknown    |
|   8 | LightGBM (tuned + ensembled)                      |       1416 | +42/-41      |        0.323 |       18.05 |                13.35 |                   11.697 |                          416.56 |                             2.236 | Unknown    |               0    | False     | Unknown    |
|   9 | TabM_GPU (tuned + ensembled)                      |       1414 | +70/-45      |        0.373 |       18.22 |                 9.07 |                   10.747 |                         3133.91 |                             1.273 | Unknown    |               0    | False     | Unknown    |
|  10 | TabDPT_GPU (tuned)                                |       1396 | +78/-60      |        0.394 |       19.23 |                 6.66 |                   10.815 |                         6154.73 |                            39.452 | Unknown    |               0    | False     | Unknown    |
|  11 | CatBoost (tuned + ensembled)                      |       1395 | +60/-46      |        0.33  |       19.28 |                12.45 |                   11.285 |                         1665.53 |                             0.559 | Unknown    |               0    | False     | Unknown    |
|  12 | ModernNCA_GPU (tuned + ensembled)                 |       1387 | +83/-63      |        0.382 |       19.76 |                 7.9  |                   11.588 |                         4618.5  |                             7.737 | Unknown    |               0    | False     | Unknown    |
|  13 | CatBoost (tuned)                                  |       1370 | +53/-48      |        0.302 |       20.77 |                13.28 |                   11.713 |                         1665.53 |                             0.065 | Unknown    |               0    | False     | Unknown    |
|  14 | XGBoost (tuned + ensembled)                       |       1368 | +44/-48      |        0.279 |       20.85 |                13.51 |                   12.324 |                          700.96 |                             1.439 | Unknown    |               0    | False     | Unknown    |
|  15 | LightGBM (tuned)                                  |       1353 | +49/-47      |        0.262 |       21.76 |                16.33 |                   12.602 |                          416.56 |                             0.381 | Unknown    |               0    | False     | Unknown    |
|  16 | ModernNCA_GPU (tuned)                             |       1350 | +61/-56      |        0.301 |       21.96 |                11.58 |                   12.202 |                         4618.5  |                             0.47  | Unknown    |               0    | False     | Unknown    |
|  17 | CatBoost (default)                                |       1349 | +44/-45      |        0.267 |       22.04 |                12.16 |                   12.296 |                            6.7  |                             0.088 | Unknown    |               0    | False     | Unknown    |
|  18 | TabM_GPU (tuned)                                  |       1347 | +66/-54      |        0.296 |       22.11 |                13.28 |                   11.681 |                         3133.91 |                             0.13  | Unknown    |               0    | False     | Unknown    |
|  19 | XGBoost (tuned)                                   |       1345 | +48/-45      |        0.254 |       22.24 |                12.85 |                   12.508 |                          700.96 |                             0.213 | Unknown    |               0    | False     | Unknown    |
|  20 | xRFM_GPU (tuned + ensembled)                      |       1344 | +61/-45      |        0.286 |       22.33 |                13.51 |                   12.482 |                          866.11 |                             2.007 | Unknown    |               0    | False     | Unknown    |
|  21 | TabPFNv2_GPU (tuned + ensembled) [35.29% IMPUTED] |       1338 | +78/-72      |        0.344 |       22.68 |                 7.38 |                   12.653 |                         2942.08 |                            17.372 | Unknown    |              35.29 | True      | Unknown    |
|  22 | Mitra_GPU (default) [35.29% IMPUTED]              |       1313 | +57/-68      |        0.285 |       24.2  |                10.18 |                   13.462 |                           87.34 |                             2.433 | Unknown    |              35.29 | True      | Unknown    |
|  23 | xRFM_GPU (tuned)                                  |       1294 | +61/-46      |        0.207 |       25.42 |                13.72 |                   13.858 |                          866.11 |                             0.097 | Unknown    |               0    | False     | Unknown    |
|  24 | TabDPT_GPU (default)                              |       1290 | +71/-69      |        0.284 |       25.64 |                 9.22 |                   13.508 |                           45.42 |                            39.406 | Unknown    |               0    | False     | Unknown    |
|  25 | TabICL_GPU (default) [29.41% IMPUTED]             |       1286 | +61/-56      |        0.254 |       25.92 |                 9.07 |                   13.311 |                            6.86 |                             1.52  | Unknown    |              29.41 | True      | Unknown    |
|  26 | TabM_GPU (default)                                |       1284 | +54/-51      |        0.226 |       26.04 |                18.25 |                   14.086 |                           11.56 |                             0.127 | Unknown    |               0    | False     | Unknown    |
|  27 | TabPFNv2_GPU (tuned) [35.29% IMPUTED]             |       1271 | +70/-59      |        0.236 |       26.81 |                13.63 |                   14.323 |                         2942.08 |                             0.262 | Unknown    |              35.29 | True      | Unknown    |
|  28 | EBM (tuned + ensembled)                           |       1270 | +52/-56      |        0.196 |       26.88 |                16.5  |                   15.046 |                         2961.52 |                             0.482 | Unknown    |               0    | False     | Unknown    |
|  29 | RealMLP_GPU (default)                             |       1259 | +43/-50      |        0.147 |       27.61 |                19.24 |                   14.457 |                           10.44 |                             1.714 | Unknown    |               0    | False     | Unknown    |
|  30 | TorchMLP (tuned + ensembled)                      |       1259 | +47/-54      |        0.151 |       27.61 |                22.18 |                   13.83  |                         2832.8  |                             1.801 | Unknown    |               0    | False     | Unknown    |
|  31 | TabPFNv2_GPU (default) [35.29% IMPUTED]           |       1234 | +69/-77      |        0.213 |       29.11 |                11.75 |                   15.18  |                            3.27 |                             0.315 | Unknown    |              35.29 | True      | Unknown    |
|  32 | EBM (tuned)                                       |       1221 | +62/-58      |        0.147 |       29.94 |                15.65 |                   15.879 |                         2961.52 |                             0.048 | Unknown    |               0    | False     | Unknown    |
|  33 | ModernNCA_GPU (default)                           |       1218 | +59/-48      |        0.133 |       30.11 |                16.17 |                   16.702 |                           13.74 |                             0.316 | Unknown    |               0    | False     | Unknown    |
|  34 | ExtraTrees (tuned + ensembled)                    |       1206 | +55/-58      |        0.116 |       30.88 |                22.68 |                   17.12  |                          191.44 |                             0.76  | Unknown    |               0    | False     | Unknown    |
|  35 | EBM (default)                                     |       1196 | +58/-58      |        0.135 |       31.47 |                17    |                   16.826 |                            7.66 |                             0.046 | Unknown    |               0    | False     | Unknown    |
|  36 | TorchMLP (tuned)                                  |       1191 | +46/-61      |        0.116 |       31.75 |                24.88 |                   15.694 |                         2832.8  |                             0.112 | Unknown    |               0    | False     | Unknown    |
|  37 | XGBoost (default)                                 |       1186 | +61/-53      |        0.117 |       32.06 |                18.06 |                   15.734 |                            2.06 |                             0.122 | Unknown    |               0    | False     | Unknown    |
|  38 | FastaiMLP (tuned + ensembled)                     |       1177 | +65/-68      |        0.111 |       32.62 |                23.17 |                   17.478 |                          594.95 |                             4.651 | Unknown    |               0    | False     | Unknown    |
|  39 | ExtraTrees (tuned)                                |       1170 | +66/-61      |        0.117 |       32.99 |                19.92 |                   18.168 |                          191.44 |                             0.101 | Unknown    |               0    | False     | Unknown    |
|  40 | RandomForest (tuned + ensembled)                  |       1165 | +58/-64      |        0.087 |       33.29 |                24.79 |                   18.052 |                          377.08 |                             0.747 | Unknown    |               0    | False     | Unknown    |
|  41 | LightGBM (default)                                |       1154 | +55/-48      |        0.083 |       33.99 |                29.68 |                   16.533 |                            2.2  |                             0.171 | Unknown    |               0    | False     | Unknown    |
|  42 | RandomForest (tuned)                              |       1122 | +48/-47      |        0.05  |       35.82 |                29.96 |                   18.866 |                          377.08 |                             0.091 | Unknown    |               0    | False     | Unknown    |
|  43 | FastaiMLP (tuned)                                 |       1104 | +58/-77      |        0.072 |       36.78 |                25.54 |                   19.105 |                          594.95 |                             0.337 | Unknown    |               0    | False     | Unknown    |
|  44 | TorchMLP (default)                                |       1031 | +55/-68      |        0.023 |       40.57 |                36.51 |                   20.938 |                            8.96 |                             0.129 | Unknown    |               0    | False     | Unknown    |
|  45 | xRFM_GPU (default)                                |       1031 | +71/-72      |        0.057 |       40.59 |                26.84 |                   23.722 |                            3.14 |                             0.741 | Unknown    |               0    | False     | Unknown    |
|  46 | RandomForest (default)                            |       1000 | +62/-56      |        0.014 |       42.05 |                34.8  |                   23.738 |                            0.43 |                             0.053 | Unknown    |               0    | False     | Unknown    |
|  47 | KNN (tuned + ensembled)                           |        972 | +63/-84      |        0.024 |       43.27 |                38.44 |                   25.579 |                          129.1  |                             1.627 | Unknown    |               0    | False     | Unknown    |
|  48 | FastaiMLP (default)                               |        969 | +69/-75      |        0.022 |       43.43 |                39.91 |                   23.045 |                            3.12 |                             0.312 | Unknown    |               0    | False     | Unknown    |
|  49 | ExtraTrees (default)                              |        968 | +76/-82      |        0.019 |       43.47 |                39.66 |                   25.337 |                            0.26 |                             0.054 | Unknown    |               0    | False     | Unknown    |
|  50 | Linear (tuned + ensembled)                        |        905 | +79/-108     |        0.024 |       45.96 |                24.12 |                   32.211 |                          240.73 |                             0.308 | Unknown    |               0    | False     | Unknown    |
|  51 | Linear (tuned)                                    |        873 | +77/-108     |        0.017 |       47.07 |                32.36 |                   32.797 |                          240.73 |                             0.068 | Unknown    |               0    | False     | Unknown    |
|  52 | KNN (tuned)                                       |        813 | +68/-91      |        0.012 |       48.92 |                46.89 |                   31.715 |                          129.1  |                             0.103 | Unknown    |               0    | False     | Unknown    |
|  53 | Linear (default)                                  |        812 | +87/-129     |        0.008 |       48.96 |                46.1  |                   35.354 |                            1.23 |                             0.115 | Unknown    |               0    | False     | Unknown    |
|  54 | OpenTab                                           |        620 | +150/-265    |        0.01  |       53.08 |                50.02 |                   53.371 |                            0.02 |                             3.331 | Unknown    |               0    | False     | Unknown    |
|  55 | KNN (default)                                     |        602 | +78/-126     |        0     |       53.35 |                52.99 |                   44.314 |                            0.19 |                             0.037 | Unknown    |               0    | False     | Unknown    |
```

</details>

Note: OpenTab is still in early development. With more training data, larger models, and more epochs, performance is expected to improve significantly. The current results demonstrate a working baseline trained with minimal compute resources.
