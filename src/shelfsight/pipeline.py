"""Controlled CNN ablations and frozen-backbone transfer benchmark."""

import hashlib
import importlib.metadata
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import prepare, write_json
from .evaluation import bootstrap, metrics
from .models import CatalogCNN, extract_features
from .training import fit, predict, seed_everything
from .visualization import render


def configurations():
    changes = {
        "baseline": {},
        "kernel5": {"kernel": 5},
        "stride2": {"stride": 2},
        "more_filters": {"filters": 32},
        "average_pool": {"pooling": "average"},
        "pool3": {"pool_size": 3},
        "shallow": {"depth": 2},
        "deep": {"depth": 4},
        "augmentation": {"augment": True},
    }
    return [{"name": "cnn_" + name, **change} for name, change in changes.items()]


def run(root):
    root = Path(root)
    report, models = root / "reports", root / "models"
    report.mkdir(exist_ok=True)
    models.mkdir(exist_ok=True)
    seed_everything()
    parts, audit = prepare(root)
    write_json(report / "data_audit.json", audit)
    write_json(report / "experiment_configs.json", configurations())
    pd.concat(
        [
            pd.DataFrame(
                {"image_id": p[2], "pixel_sha256": p[3], "label": p[1], "split": k}
            )
            for k, p in parts.items()
        ]
    ).to_csv(report / "split_manifest.csv", index=False)
    validation, histories, candidates = [], [], {}
    for config in configurations():
        seed_everything()
        model = CatalogCNN(**config)
        history, timing = fit(model, parts, audit, config)
        score = metrics(
            parts["validation"][1], predict(model, parts["validation"], audit)
        )
        validation.append({"id": config["name"], **score, **timing})
        histories.extend({"id": config["name"], **row} for row in history)
        torch.save(model.state_dict(), models / (config["name"] + ".pt"))
        candidates[config["name"]] = model
        print(config["name"], score, timing, flush=True)
    experiments = pd.DataFrame(validation)
    experiments.to_csv(report / "validation_experiments.csv", index=False)
    history = pd.DataFrame(histories)
    history.to_csv(report / "training_history.csv", index=False)
    selected = experiments.sort_values("macro_f1", ascending=False).iloc[0].id
    write_json(
        report / "selection.json",
        {"custom_cnn": selected, "criterion": "validation_macro_f1"},
    )
    output, predictions = [], []

    def add(name, probabilities, training_seconds=None):
        score = metrics(parts["test"][1], probabilities)
        interval = bootstrap(parts["test"][1], probabilities, parts["test"][3])
        output.append(
            {
                "id": name,
                **score,
                "ci_low": interval[0],
                "ci_high": interval[1],
                "training_seconds": training_seconds,
            }
        )
        frame = pd.DataFrame(probabilities, columns=["p" + str(i) for i in range(10)])
        frame = frame.assign(
            id=name, image_id=parts["test"][2], target=parts["test"][1]
        )
        predictions.append(frame)

    # Evaluate only the locked winner and predeclared reference configurations.
    for name in dict.fromkeys([selected, "cnn_baseline", "cnn_augmentation"]):
        add(
            name,
            predict(candidates[name], parts["test"], audit),
            float(experiments.set_index("id").loc[name, "training_seconds"]),
        )
    flat = {
        k: p[0].reshape(len(p[0]), -1).astype(np.float32) / 255
        for k, p in parts.items()
    }
    for name, model in [
        ("prior", DummyClassifier(strategy="prior")),
        (
            "linear_pixels",
            make_pipeline(
                StandardScaler(),
                SGDClassifier(
                    loss="log_loss",
                    alpha=0.001,
                    max_iter=100,
                    tol=0.001,
                    random_state=42,
                ),
            ),
        ),
    ]:
        start = time.perf_counter()
        model.fit(flat["train"], parts["train"][1])
        add(name, model.predict_proba(flat["test"]), time.perf_counter() - start)
    # Extract using the official frozen backbone. No test labels enter this step.
    start = time.perf_counter()
    features = {}
    for split, part in parts.items():
        features[split], url = extract_features(part[0])
        print("MobileNet feature extraction", split, features[split].shape, flush=True)
    extraction_seconds = time.perf_counter() - start
    start = time.perf_counter()
    head = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))
    head.fit(features["train"], parts["train"][1])
    head_seconds = time.perf_counter() - start
    val_transfer = metrics(
        parts["validation"][1], head.predict_proba(features["validation"])
    )
    write_json(
        report / "transfer_metadata.json",
        {
            "backbone": "MobileNetV3-Small",
            "weights": "IMAGENET1K_V1",
            "weights_url": url,
            "frozen_layers": (
                "All pretrained features and pooling; original classifier discarded"
            ),
            "feature_dimensions": int(features["train"].shape[1]),
            "head": "Train-only StandardScaler + multinomial LogisticRegression(C=0.1)",
            "preprocessing": (
                "Official resize256/center-crop224/ImageNet normalization; "
                "replicated grayscale"
            ),
            "validation": val_transfer,
            "feature_extraction_seconds": extraction_seconds,
            "head_training_seconds": head_seconds,
            "training_seconds_scope": (
                "CNN training excludes download; transfer includes all split "
                "feature extraction plus head fit"
            ),
        },
    )
    add(
        "mobilenet_frozen",
        head.predict_proba(features["test"]),
        extraction_seconds + head_seconds,
    )
    results = pd.DataFrame(output)
    prediction_table = pd.concat(predictions, ignore_index=True)
    results.to_csv(report / "test_metrics.csv", index=False)
    prediction_table.to_csv(report / "test_predictions.csv", index=False)
    write_json(
        report / "run_metadata.json",
        {
            "seed": 42,
            "device": "cpu",
            "threads": 2,
            "versions": {
                p: importlib.metadata.version(p)
                for p in ["torch", "torchvision", "numpy", "pandas", "scikit-learn"]
            },
            "source_hashes": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (root / "src/shelfsight").glob("*.py")
            },
        },
    )
    render(root, parts, experiments, results, prediction_table, history)
    return results
