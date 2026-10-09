"""Classification, calibration and uncertainty metrics."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss


def metrics(target, probabilities):
    if (
        probabilities.shape != (len(target), 10)
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or not np.allclose(probabilities.sum(1), 1, atol=1e-5)
    ):
        raise ValueError("Expected ten valid class probabilities per image")
    probabilities = probabilities.astype(float)
    probabilities /= probabilities.sum(1, keepdims=True)
    prediction = probabilities.argmax(1)
    confidence = probabilities.max(1)
    ece = 0.0
    for lower in np.linspace(0, 0.9, 10):
        mask = (confidence > lower) & (confidence <= lower + 0.10000001)
        if mask.any():
            ece += mask.mean() * abs(
                (prediction[mask] == target[mask]).mean() - confidence[mask].mean()
            )
    return {
        "accuracy": float(accuracy_score(target, prediction)),
        "macro_f1": float(
            f1_score(target, prediction, average="macro", labels=range(10))
        ),
        "log_loss": float(log_loss(target, probabilities, labels=range(10))),
        "ece_10_bins": float(ece),
    }


def bootstrap(target, probabilities, hashes, repeats=500):
    rng = np.random.default_rng(42)
    unique = np.unique(hashes)
    index = {h: np.flatnonzero(hashes == h) for h in unique}
    values = []
    for _ in range(repeats):
        sampled = np.concatenate([index[h] for h in rng.choice(unique, len(unique))])
        values.append(
            f1_score(
                target[sampled],
                probabilities[sampled].argmax(1),
                labels=range(10),
                average="macro",
                zero_division=0,
            )
        )
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]
