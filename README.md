# OpenTab

An open implementation of [TabPFN](https://github.com/PriorLabs/TabPFN) - Prior-Data Fitted Networks for tabular data. This implementation follows the approach described in the [TabPFN Nature paper](https://www.nature.com/articles/s41586-024-08328-6), using **Structural Causal Models (SCMs)** for synthetic data generation.

## Quick Start

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync

# Install TabArena for benchmarking (optional)
uv pip install "tabarena @ git+https://github.com/autogluon/tabarena.git#subdirectory=tabarena"

# Train classification model (online data generation)
python train.py --online --steps 100000 --output_dir checkpoints

# Train regression model
python train.py --online --regression --steps 100000 --output_dir checkpoints

# Evaluate on sklearn datasets (quick test)
python evaluate.py --checkpoint checkpoints/model_100000.pt --mode quick

# Evaluate on TabArena benchmark (51 datasets)
python evaluate.py --checkpoint checkpoints/model_100000.pt --mode lite

# Generate leaderboard comparing against SOTA methods
python evaluate.py --mode leaderboard --results eval_results --method OpenTab
```

## How It Works

TabPFN learns to **approximate Bayesian inference** on tabular data. It's trained on millions of synthetic datasets generated from Structural Causal Models (SCMs), then at inference time performs **in-context learning** - a single forward pass, no gradient updates.

```
Training:   Generate SCM-based data → Train Transformer → Save checkpoint
Inference:  clf.fit(X_train, y_train) → clf.predict(X_test)  # One forward pass!
```

## Use as a Classifier

```python
from model import OpenTabClassifier

clf = OpenTabClassifier("checkpoints/model.pt")
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)
```

## Use as a Regressor

```python
from model import OpenTabRegressor

reg = OpenTabRegressor("checkpoints/model.pt")
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)  # Returns expected value over bins
```

## Repository Structure

```
model.py          # Transformer architecture (TabPFN encoder/decoder) + OpenTabClassifier/Regressor
generate_data.py  # SCM-based synthetic data generation
train.py          # Training loop with online data generation
evaluate.py       # TabArena benchmark + quick evaluation
```

## Training

Training uses **online data generation** - each batch is a freshly generated synthetic dataset from an SCM prior.

```bash
# Basic training (classification)
python train.py --online --steps 100000

# Training with custom architecture
python train.py --online \
    --embedding_size 128 \
    --n_heads 4 \
    --n_layers 6 \
    --mlp_hidden 256 \
    --batch_size 64 \
    --lr 3e-4

# Training for regression
python train.py --online --regression --n_bins 64

# Resume from checkpoint
python train.py --online --resume checkpoints/model_50000.pt
```

### Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--online` | - | Enable online data generation (recommended) |
| `--data` | None | HDF5 file with pre-generated data (alternative to --online) |
| `--steps` | 100000 | Number of training steps |
| `--batch_size` | 64 | Batch size |
| `--lr` | 3e-4 | Learning rate |
| `--warmup_steps` | 1000 | Linear warmup steps |
| `--embedding_size` | 128 | Transformer embedding dimension |
| `--n_heads` | 4 | Number of attention heads |
| `--n_layers` | 6 | Number of transformer layers |
| `--mlp_hidden` | 256 | MLP hidden layer size |
| `--regression` | False | Train for regression task |
| `--max_classes` | 10 | Maximum number of classes (classification) |
| `--n_bins` | 64 | Number of bins (regression) |
| `--output_dir` | checkpoints | Output directory for checkpoints |
| `--compile` | False | Use torch.compile() for ~10-20% faster training |

## Data Generation

Synthetic data is generated using **Structural Causal Models (SCMs)** following the TabPFN v2 Nature paper:

- **DAG Structure**: Growing network with redirection (preferential attachment for scale-free networks)
- **Node Representations**: Vector embeddings in ℝᵈ
- **Edge Functions**: Neural networks (with various activations), decision trees, or categorical mappings
- **Activations**: identity, log, sigmoid, abs, sin, tanh, rank, square, power (2-5), smooth ReLU, step, modulo
- **Post-processing**: Kumaraswamy warping (20% of datasets), quantization, missing values (MCAR with NaN)
- **Feature Sampling**: Beta(0.95, 8.0) distribution scaled to [1, 160] features (paper-aligned)
- **Cell Budget**: Tables capped at 75,000 cells (samples reduced for high feature counts)

```bash
# Generate and save synthetic datasets to HDF5
python generate_data.py --n_datasets 100000 --output data/synthetic.h5

# Generate with visualization
python generate_data.py --n_datasets 10 --visualize

# Generate for regression
python generate_data.py --n_datasets 100000 --regression --output data/synthetic_reg.h5
```

### Data Generation Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n_datasets` | 100000 | Number of datasets to generate |
| `--output` | data/synthetic.h5 | Output HDF5 file path |
| `--max_samples` | 512 | Maximum samples per dataset (paper uses up to 2048) |
| `--max_features` | 160 | Maximum features per dataset (Beta distribution, paper-aligned) |
| `--max_classes` | 10 | Maximum classes (classification) |
| `--regression` | False | Generate regression data |
| `--visualize` | False | Show visualization of generated data |
| `--seed` | 42 | Random seed |

## Evaluation

### Quick Evaluation (sklearn datasets)

```bash
# Classification: Iris, Wine, Breast Cancer
python evaluate.py --checkpoint checkpoints/model.pt --mode quick

# Regression: Diabetes, California Housing
python evaluate.py --checkpoint checkpoints/model.pt --mode quick-regression
```

### TabArena Benchmark

[TabArena](https://github.com/autogluon/tabarena) provides standardized benchmarking against state-of-the-art tabular ML methods.

```bash
# TabArena-Lite (51 datasets, 1 fold each)
python evaluate.py --checkpoint checkpoints/model.pt --mode lite

# Full TabArena (all datasets, all folds)
python evaluate.py --checkpoint checkpoints/model.pt --mode full

# Generate leaderboard with ELO ratings
python evaluate.py --mode leaderboard --results eval_results --method OpenTab

# Load cached leaderboard
python evaluate.py --mode leaderboard-cache --method OpenTab
```

### Evaluation Modes

| Mode | Description |
|------|-------------|
| `quick` | Test on sklearn classification datasets |
| `quick-regression` | Test on sklearn regression datasets |
| `lite` | TabArena-Lite (51 datasets, 1 fold) |
| `full` | Full TabArena (all datasets, all folds) |
| `leaderboard` | Generate leaderboard with ELO ratings |
| `leaderboard-cache` | Load leaderboard from cache |

## Model Architecture

The model follows the TabPFN architecture with two-way attention:

1. **Input Processing**:
   - Z-normalization per feature using training statistics
   - Random feature embeddings for disambiguation
   - Missing value indicators (zero-fill + indicator embedding)

2. **TabPFN Encoder** (N layers):
   - **Inter-feature attention**: Fully connected across features
   - **Inter-sample attention**: Test samples attend to train samples only

3. **Decoder**:
   - MLP decoder for classification (softmax over classes)
   - Piecewise constant output for regression (binning approach)

## TabArena Leaderboard

Results on [TabArena](https://github.com/autogluon/tabarena) benchmark comparing OpenTab against state-of-the-art tabular ML methods:

| # | Model | Elo ⬆️ | Elo 95% CI | Score ⬆️ | Rank ⬇️ | Harmonic Rank ⬇️ | Improvability (%) ⬇️ |
|---:|:---|---:|:---|---:|---:|---:|---:|
| 0 | RealTabPFN-v2.5 (tuned + ensembled) | 1631 | +100/-69 | 0.689 | 8.22 | 2.63 | 3.759 |
| 1 | AutoGluon 1.4 (extreme, 4h) | 1611 | +92/-77 | 0.662 | 8.91 | 3.98 | 5.868 |
| 2 | RealTabPFN-v2.5 (tuned) | 1580 | +87/-67 | 0.615 | 10.08 | 4.34 | 5.875 |
| 3 | AutoGluon 1.4 (best, 4h) | 1547 | +67/-64 | 0.565 | 11.45 | 4.87 | 7.644 |
| 4 | RealTabPFN-v2.5 (default) | 1540 | +72/-50 | 0.565 | 11.75 | 6.17 | 6.522 |
| 5 | RealMLP_GPU (tuned + ensembled) | 1512 | +58/-51 | 0.51 | 13.02 | 7.61 | 8.756 |
| 6 | TabDPT_GPU (tuned + ensembled) | 1438 | +69/-59 | 0.453 | 16.8 | 5.02 | 9.172 |
| 7 | RealMLP_GPU (tuned) | 1435 | +57/-56 | 0.416 | 16.95 | 8.43 | 10.281 |
| 8 | LightGBM (tuned + ensembled) | 1416 | +40/-40 | 0.323 | 18.05 | 13.36 | 11.697 |
| 9 | TabM_GPU (tuned + ensembled) | 1412 | +70/-45 | 0.373 | 18.24 | 9.1 | 10.747 |
| 10 | TabDPT_GPU (tuned) | 1395 | +77/-60 | 0.395 | 19.25 | 6.66 | 10.815 |
| 11 | CatBoost (tuned + ensembled) | 1394 | +59/-46 | 0.331 | 19.3 | 12.45 | 11.285 |
| 12 | ModernNCA_GPU (tuned + ensembled) | 1386 | +84/-63 | 0.382 | 19.76 | 7.9 | 11.588 |
| 13 | CatBoost (tuned) | 1368 | +54/-46 | 0.303 | 20.79 | 13.29 | 11.713 |
| 14 | XGBoost (tuned + ensembled) | 1367 | +46/-49 | 0.279 | 20.87 | 13.52 | 12.324 |
| 15 | LightGBM (tuned) | 1352 | +50/-48 | 0.262 | 21.78 | 16.33 | 12.602 |
| 16 | ModernNCA_GPU (tuned) | 1349 | +62/-57 | 0.301 | 21.96 | 11.58 | 12.202 |
| 17 | CatBoost (default) | 1348 | +44/-44 | 0.267 | 22.04 | 12.16 | 12.296 |
| 18 | TabM_GPU (tuned) | 1346 | +66/-54 | 0.296 | 22.13 | 13.29 | 11.681 |
| 19 | XGBoost (tuned) | 1344 | +47/-45 | 0.254 | 22.25 | 12.86 | 12.508 |
| 20 | xRFM_GPU (tuned + ensembled) | 1342 | +62/-45 | 0.286 | 22.35 | 13.53 | 12.482 |
| 21 | TabPFNv2_GPU (tuned + ensembled) [35.29% IMPUTED] | 1337 | +76/-73 | 0.345 | 22.7 | 7.4 | 12.653 |
| 22 | Mitra_GPU (default) [35.29% IMPUTED] | 1313 | +56/-68 | 0.286 | 24.2 | 10.18 | 13.462 |
| 23 | xRFM_GPU (tuned) | 1292 | +60/-44 | 0.208 | 25.46 | 13.73 | 13.858 |
| 24 | TabDPT_GPU (default) | 1289 | +70/-71 | 0.285 | 25.66 | 9.24 | 13.508 |
| 25 | TabICL_GPU (default) [29.41% IMPUTED] | 1284 | +61/-56 | 0.255 | 25.94 | 9.08 | 13.311 |
| 26 | TabM_GPU (default) | 1283 | +55/-51 | 0.226 | 26.06 | 18.29 | 14.086 |
| 27 | TabPFNv2_GPU (tuned) [35.29% IMPUTED] | 1270 | +70/-60 | 0.237 | 26.83 | 13.71 | 14.323 |
| 28 | EBM (tuned + ensembled) | 1269 | +51/-56 | 0.196 | 26.9 | 16.52 | 15.046 |
| 29 | RealMLP_GPU (default) | 1258 | +44/-51 | 0.147 | 27.63 | 19.25 | 14.457 |
| 30 | TorchMLP (tuned + ensembled) | 1258 | +46/-53 | 0.151 | 27.63 | 22.19 | 13.83 |
| 31 | TabPFNv2_GPU (default) [35.29% IMPUTED] | 1234 | +69/-78 | 0.214 | 29.11 | 11.75 | 15.18 |
| 32 | EBM (tuned) | 1220 | +62/-57 | 0.147 | 29.96 | 15.65 | 15.879 |
| 33 | ModernNCA_GPU (default) | 1217 | +59/-48 | 0.135 | 30.13 | 16.17 | 16.702 |
| 34 | ExtraTrees (tuned + ensembled) | 1204 | +56/-59 | 0.117 | 30.92 | 22.7 | 17.12 |
| 35 | EBM (default) | 1195 | +58/-59 | 0.135 | 31.49 | 17.01 | 16.826 |
| 36 | TorchMLP (tuned) | 1191 | +46/-60 | 0.116 | 31.73 | 24.87 | 15.694 |
| 37 | XGBoost (default) | 1186 | +61/-53 | 0.117 | 32.06 | 18.06 | 15.734 |
| 38 | FastaiMLP (tuned + ensembled) | 1176 | +64/-68 | 0.112 | 32.66 | 23.19 | 17.478 |
| 39 | ExtraTrees (tuned) | 1169 | +65/-60 | 0.118 | 33.03 | 19.98 | 18.168 |
| 40 | RandomForest (tuned + ensembled) | 1165 | +60/-65 | 0.087 | 33.31 | 24.79 | 18.052 |
| 41 | LightGBM (default) | 1153 | +54/-48 | 0.083 | 34.01 | 29.69 | 16.533 |
| 42 | RandomForest (tuned) | 1121 | +48/-47 | 0.05 | 35.84 | 29.97 | 18.866 |
| 43 | FastaiMLP (tuned) | 1103 | +58/-76 | 0.073 | 36.84 | 25.56 | 19.105 |
| 44 | TorchMLP (default) | 1031 | +55/-67 | 0.023 | 40.59 | 36.52 | 20.938 |
| 45 | xRFM_GPU (default) | 1030 | +71/-72 | 0.057 | 40.63 | 26.85 | 23.722 |
| 46 | RandomForest (default) | 1000 | +62/-55 | 0.014 | 42.07 | 34.81 | 23.738 |
| 47 | KNN (tuned + ensembled) | 973 | +63/-82 | 0.024 | 43.27 | 38.46 | 25.579 |
| 48 | FastaiMLP (default) | 968 | +69/-76 | 0.022 | 43.47 | 39.93 | 23.045 |
| 49 | ExtraTrees (default) | 968 | +73/-83 | 0.019 | 43.49 | 39.67 | 25.337 |
| 50 | Linear (tuned + ensembled) | 906 | +78/-108 | 0.024 | 45.98 | 24.13 | 32.211 |
| 51 | Linear (tuned) | 874 | +76/-108 | 0.017 | 47.09 | 32.38 | 32.797 |
| 52 | KNN (tuned) | 817 | +67/-89 | 0.012 | 48.9 | 46.88 | 31.715 |
| 53 | Linear (default) | 814 | +86/-128 | 0.008 | 48.98 | 46.14 | 35.354 |
| 54 | **OpenTab** | **679** | +133/-257 | 0.024 | 52.16 | 43.75 | 52.806 |
| 55 | KNN (default) | 613 | +76/-113 | 0 | 53.31 | 52.96 | 44.314 |

**Training configuration**: 1M synthetic datasets, 512 max samples, 50 max features, 10 classes, ~15,500 steps (1 epoch) on a modest NVIDIA GeForce RTX 3050 4GB laptop GPU.

## References

```bibtex
@article{hollmann2025tabpfn,
  title={Accurate predictions on small data with a tabular foundation model},
  author={Hollmann, Noah and Müller, Samuel and Purucker, Lennart and others},
  journal={Nature},
  year={2025},
  publisher={Nature Publishing Group}
}

@article{hollmann2023tabpfn,
  title={TabPFN: A transformer that solves small tabular classification problems in a second},
  author={Hollmann, Noah and Müller, Samuel and Eggensperger, Katharina and Hutter, Frank},
  booktitle={ICLR 2023},
  year={2023}
}

@article{erickson2025tabarena,
  title={TabArena: A Living Benchmark for Machine Learning on Tabular Data}, 
  author={Nick Erickson and Lennart Purucker and others},
  year={2025},
  journal={arXiv preprint arXiv:2506.16791},
}
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
