"""Inference for the benchmark's 28-by-28 grayscale apparel images."""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .data import CLASSES, write_json
from .models import CatalogCNN
from .training import predict


def predict_image(root, source, destination):
    root = Path(root)
    with Image.open(source) as image:
        if image.mode != "L" or image.size != (28, 28):
            raise ValueError("Expected a 28-by-28 grayscale benchmark image")
        pixels = np.asarray(image, dtype=np.uint8)[None]
    report = root / "reports"
    name = json.loads((report / "selection.json").read_text())["custom_cnn"]
    configs = json.loads((report / "experiment_configs.json").read_text())
    config = next(c for c in configs if c["name"] == name)
    model = CatalogCNN(**config)
    model.load_state_dict(
        torch.load(root / "models" / (name + ".pt"), weights_only=True)
    )
    audit = json.loads((report / "data_audit.json").read_text())
    probability = predict(model, (pixels, np.array([0]), None, None), audit)[0]
    result = {
        "model": name,
        "predicted_category": CLASSES[int(probability.argmax())],
        "confidence": float(probability.max()),
        "probabilities": dict(zip(CLASSES, map(float, probability))),
    }
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, result)
    return result
