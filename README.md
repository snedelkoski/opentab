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
