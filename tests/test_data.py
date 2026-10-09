"""Data contracts and augmentation boundaries."""

import numpy as np
import pandas as pd
import pytest
import torch

from shelfsight.data import ApparelDataset, image_hashes, validate


def frame():
    pixels = np.zeros((3, 784), dtype=int)
    return pd.DataFrame(
        np.column_stack([[0, 1, 2], pixels]),
        columns=["label"] + [f"p{i}" for i in range(784)],
    )


def test_source_schema_shapes():
    images, labels = validate(frame())
    assert images.shape == (3, 28, 28)
    assert labels.tolist() == [0, 1, 2]


@pytest.mark.parametrize("invalid", [-1, 256, np.nan, 0.5])
def test_bad_pixels_rejected(invalid):
    data = frame().astype(float)
    data.iloc[0, 1] = invalid
    with pytest.raises(ValueError):
        validate(data)


def test_bad_label_rejected():
    data = frame()
    data.iloc[0, 0] = 10
    with pytest.raises(ValueError):
        validate(data)


def test_identical_pixels_share_hash_regardless_of_label():
    images = np.zeros((3, 28, 28), dtype=np.uint8)
    images[2, 1, 1] = 1
    hashes = image_hashes(images)
    assert hashes[0] == hashes[1] != hashes[2]


def test_evaluation_transform_deterministic_and_augmentation_finite():
    images = np.random.default_rng(2).integers(0, 256, size=(2, 28, 28), dtype=np.uint8)
    normal = ApparelDataset(images, [1, 2], 0.3, 0.4)
    torch.testing.assert_close(normal[0][0], normal[0][0])
    augmented = ApparelDataset(images, [1, 2], 0.3, 0.4, augment=True)
    output, label = augmented[0]
    assert output.shape == (1, 28, 28) and label == 1
    assert torch.isfinite(output).all()
