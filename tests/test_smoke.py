import numpy as np

from generate_data import SCMDataGenerator
from model import OpenTabClassifier, OpenTabModel
from train import TrainConfig, collate_variable_size


def test_scm_generator_produces_valid_shapes() -> None:
    generator = SCMDataGenerator(
        n_samples_range=(32, 32),
        n_features_range=(4, 4),
        n_classes_range=(2, 3),
        is_regression=False,
    )
    dataset = generator.generate(train_ratio=0.75)

    assert dataset.X.shape == (32, 4)
    assert dataset.y.shape == (32,)
    assert 1 <= dataset.train_size < 32
    assert dataset.n_classes >= 1
    assert np.isfinite(dataset.X[~np.isnan(dataset.X)]).all()
    assert np.issubdtype(dataset.y.dtype, np.integer)


def test_classifier_fit_predict_contract() -> None:
    rng = np.random.RandomState(0)
    X_train = rng.randn(24, 4).astype(np.float32)
    y_train = rng.randint(0, 3, size=24).astype(np.int64)
    X_test = rng.randn(8, 4).astype(np.float32)

    model = OpenTabModel(
        embedding_size=16,
        n_heads=1,
        n_layers=1,
        mlp_hidden_size=32,
        n_outputs=3,
        max_features=8,
        dropout=0.0,
    )
    clf = OpenTabClassifier(model=model, device="cpu")

    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)

    assert preds.shape == (8,)
    assert probs.shape == (8, 3)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert np.logical_and(preds >= 0, preds < 3).all()


def test_collate_variable_size_pads_batch() -> None:
    sample_a = {
        "X": np.ones((5, 3), dtype=np.float32),
        "y": np.array([0, 1, 0, 1, 0], dtype=np.int64),
        "train_size": 3,
        "n_features": 3,
        "n_samples": 5,
    }
    sample_b = {
        "X": np.ones((3, 2), dtype=np.float32) * 2,
        "y": np.array([1, 0, 1], dtype=np.int64),
        "train_size": 2,
        "n_features": 2,
        "n_samples": 3,
    }

    import torch

    batch = [
        {
            "X": torch.tensor(sample_a["X"]),
            "y": torch.tensor(sample_a["y"]),
            "train_size": sample_a["train_size"],
            "n_features": sample_a["n_features"],
            "n_samples": sample_a["n_samples"],
        },
        {
            "X": torch.tensor(sample_b["X"]),
            "y": torch.tensor(sample_b["y"]),
            "train_size": sample_b["train_size"],
            "n_features": sample_b["n_features"],
            "n_samples": sample_b["n_samples"],
        },
    ]

    collated = collate_variable_size(batch)

    assert tuple(collated["X"].shape) == (2, 5, 3)
    assert tuple(collated["y"].shape) == (2, 5)
    assert collated["train_size"].tolist() == [3, 2]
    assert collated["n_features"].tolist() == [3, 2]
    assert collated["n_samples"].tolist() == [5, 3]


def test_train_config_auto_device_resolves() -> None:
    config = TrainConfig(device="auto")
    assert config.device in {"cpu", "cuda"}
