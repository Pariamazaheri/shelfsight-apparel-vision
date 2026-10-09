"""Recompute all results and audit image isolation without downloading data."""

import hashlib
import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

from shelfsight.evaluation import metrics


def main():
    root = Path(__file__).resolve().parents[1]
    report = root / "reports"
    manifest = pd.read_csv(report / "split_manifest.csv")
    assert manifest.groupby("pixel_sha256").split.nunique().max() == 1
    test_ids = set(manifest[manifest.split == "test"].image_id)
    validation = pd.read_csv(report / "validation_experiments.csv")
    selected = json.loads((report / "selection.json").read_text())["custom_cnn"]
    assert selected == validation.sort_values("macro_f1", ascending=False).iloc[0].id
    predictions = pd.read_csv(report / "test_predictions.csv")
    table = pd.read_csv(report / "test_metrics.csv")
    for row in table.itertuples():
        p = predictions[predictions.id == row.id]
        assert len(p) == len(test_ids) and set(p.image_id) == test_ids
        probability = p[["p" + str(i) for i in range(10)]].to_numpy()
        actual = metrics(p.target.to_numpy(), probability)
        for column, value in actual.items():
            np.testing.assert_allclose(
                value, getattr(row, column), rtol=2e-5, atol=1e-6
            )
    raw = root / "data/raw/fashion-mnist_test.csv"
    if raw.exists():
        source = pd.read_csv(raw)
        for row in table.itertuples():
            p = predictions[predictions.id == row.id]
            np.testing.assert_array_equal(
                source.iloc[p.image_id.to_numpy()].label, p.target.to_numpy()
            )
    metadata = json.loads((report / "run_metadata.json").read_text())
    for filename, digest in metadata["source_hashes"].items():
        assert (
            hashlib.sha256(
                (root / "src/shelfsight" / filename).read_bytes()
            ).hexdigest()
            == digest
        )
    notebook = nbformat.read(root / "notebooks/apparel_vision.ipynb", as_version=4)
    cells = [c for c in notebook.cells if c.cell_type == "code"]
    assert cells and all(c.execution_count is not None for c in cells)
    assert not any(o.output_type == "error" for c in cells for o in c.outputs)
    assert len(list((report / "figures").glob("*.png"))) == 8
    print(f"Verified {len(table)} held-out models and {len(validation)} CNN designs")


if __name__ == "__main__":
    main()
