"""Local checkpoint reconstruction, input schema and frozen transfer contract."""

import json

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.models import mobilenet_v3_small

import shelfsight.models as model_module
from shelfsight.inference import predict_image
from shelfsight.models import CatalogCNN
from shelfsight.training import fit, predict, seed_everything


def test_frozen_backbone_is_eval_and_classifier_removed(monkeypatch):
    monkeypatch.setattr(
        model_module,
        "mobilenet_v3_small",
        lambda weights: mobilenet_v3_small(weights=None),
    )
    model, _, url = model_module.frozen_mobilenet()
    assert not model.training
    assert not any(p.requires_grad for p in model.parameters())
    assert "mobilenet_v3_small" in url
    with torch.no_grad():
        assert model(torch.randn(2, 3, 64, 64)).shape == (2, 576)


def test_checkpoint_inference_and_invalid_image(tmp_path):
    models, reports = tmp_path / "models", tmp_path / "reports"
    models.mkdir()
    reports.mkdir()
    config = {"name": "fixture", "filters": 4}
    torch.save(CatalogCNN(**config).state_dict(), models / "fixture.pt")
    (reports / "selection.json").write_text(json.dumps({"custom_cnn": "fixture"}))
    (reports / "experiment_configs.json").write_text(json.dumps([config]))
    (reports / "data_audit.json").write_text(
        json.dumps({"normalization_mean": 0.3, "normalization_std": 0.4})
    )
    source = tmp_path / "image.png"
    Image.fromarray(np.zeros((28, 28), dtype=np.uint8)).save(source)
    result = predict_image(tmp_path, source, tmp_path / "prediction.json")
    assert abs(sum(result["probabilities"].values()) - 1) < 1e-5
    Image.fromarray(np.zeros((30, 30), dtype=np.uint8)).save(source)
    with pytest.raises(ValueError, match="28-by-28"):
        predict_image(tmp_path, source, tmp_path / "prediction.json")


def test_small_training_checkpoint_and_predictions():
    seed_everything()
    images = np.random.default_rng(1).integers(
        0, 256, size=(12, 28, 28), dtype=np.uint8
    )
    labels = np.arange(12) % 3
    parts = {
        "train": (images[:8], labels[:8], None, None),
        "validation": (images[8:], labels[8:], None, None),
    }
    audit = {"normalization_mean": 0.3, "normalization_std": 0.4}
    model = CatalogCNN(filters=4)
    history, timing = fit(model, parts, audit, {"epochs": 1})
    probability = predict(model, parts["validation"], audit)
    assert len(history) == timing["best_epoch"] == 1
    assert np.isfinite(probability).all() and probability.shape == (4, 10)
