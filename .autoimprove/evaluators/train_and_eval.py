#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.0.0", "numpy>=1.21.0", "scikit-learn>=1.0.0", "h5py>=3.7.0", "tqdm>=4.60.0"]
# ///
"""
Primary evaluator: Train OpenTab for up to 240 seconds, then measure
classification accuracy on Iris, Wine, and Breast Cancer.

Score = mean accuracy across the 3 datasets (float in [0.0, 1.0]).
"""

import json
import os
import sys
import time

import numpy as np
import torch

# Add repo root to path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

TRAIN_TIME_BUDGET = 240  # seconds for training
TOTAL_TIME_BUDGET = 290  # leave 10s headroom from 300s limit


def train_model(time_budget: int = TRAIN_TIME_BUDGET):
    """Train OpenTab for a fixed time budget and return the model."""
    from torch.utils.data import DataLoader

    from train import OnlineDataGenerator, TrainConfig, Trainer, collate_variable_size

    print(f"Starting training with {time_budget}s budget...", file=sys.stderr)

    config = TrainConfig(
        # Smaller model for faster iteration within time budget
        embedding_size=64,
        n_heads=4,
        n_layers=3,
        mlp_hidden=128,
        max_features=100,
        max_classes=10,
        # Training params
        n_steps=999999,  # will stop by time
        batch_size=8,
        learning_rate=3e-4,
        warmup_steps=100,
        max_grad_norm=1.0,
        # Data generation
        max_train_samples=256,
        eval_samples=64,
        max_table_cells=20000,
        # Hardware
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_amp=True,
        seed=42,
        # Logging
        log_interval=50,
        eval_interval=999999,  # skip internal eval
        save_interval=999999,  # skip saving
        output_dir="/tmp/autoimprove_opentab",
    )

    os.makedirs(config.output_dir, exist_ok=True)

    trainer = Trainer(config)

    # Create data loader
    dataset = OnlineDataGenerator(config)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=0,
        collate_fn=collate_variable_size,
    )
    data_iter = iter(loader)

    # Training loop with time budget
    start_time = time.time()
    losses = []
    step = 0
    trainer.optimizer.zero_grad()

    while True:
        elapsed = time.time() - start_time
        if elapsed >= time_budget:
            break

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        try:
            loss = trainer.train_step(batch, is_accumulating=False)
            losses.append(loss)
            step += 1
        except Exception:
            continue

        if step % 50 == 0:
            avg_loss = np.mean(losses[-50:]) if losses else 0
            elapsed = time.time() - start_time
            print(
                f"Step {step} | Loss: {avg_loss:.4f} | " f"Time: {elapsed:.0f}s/{time_budget}s",
                file=sys.stderr,
            )

    elapsed = time.time() - start_time
    avg_loss = np.mean(losses[-50:]) if losses else float("inf")
    print(
        f"Training complete: {step} steps in {elapsed:.1f}s | " f"Final avg loss: {avg_loss:.4f}",
        file=sys.stderr,
    )

    return trainer.model, config, step, float(avg_loss)


def evaluate_sklearn(model, device: str):
    """Evaluate on sklearn classification datasets. Returns mean accuracy."""
    from sklearn.datasets import load_breast_cancer, load_iris, load_wine
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    from model import OpenTabClassifier

    classifier = OpenTabClassifier(model=model, device=device)

    datasets = [
        ("Iris", load_iris()),
        ("Wine", load_wine()),
        ("Breast Cancer", load_breast_cancer()),
    ]

    accuracies = []
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
        acc = float((y_pred == y_test).mean())
        accuracies.append(acc)
        print(f"  {name}: {acc:.4f}", file=sys.stderr)

    mean_acc = float(np.mean(accuracies))
    print(f"  Mean accuracy: {mean_acc:.4f}", file=sys.stderr)
    return mean_acc, {name: acc for (name, _), acc in zip(datasets, accuracies)}


def main():
    t0 = time.time()

    # Train
    model, config, steps, final_loss = train_model()

    # Evaluate
    print("Evaluating on sklearn datasets...", file=sys.stderr)
    model.eval()
    mean_acc, per_dataset = evaluate_sklearn(model, config.device)

    elapsed = time.time() - t0
    print(f"Total time: {elapsed:.1f}s", file=sys.stderr)

    # Output result
    print(
        json.dumps(
            {
                "name": "train_and_eval",
                "score": round(mean_acc, 6),
                "details": {
                    "mean_accuracy": round(mean_acc, 6),
                    "per_dataset": {k: round(v, 4) for k, v in per_dataset.items()},
                    "training_steps": steps,
                    "final_loss": round(final_loss, 4),
                    "elapsed_seconds": round(elapsed, 1),
                    "device": config.device,
                },
            }
        )
    )


if __name__ == "__main__":
    main()
