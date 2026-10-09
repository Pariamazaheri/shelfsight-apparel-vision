"""CNN design variants and probability validation."""

import numpy as np
import pytest
import torch

from shelfsight.evaluation import metrics
from shelfsight.models import CatalogCNN
from shelfsight.pipeline import configurations


@pytest.mark.parametrize("config", configurations(), ids=lambda c: c["name"])
def test_every_cnn_variant_forward_backward(config):
    torch.set_num_threads(2)
    model = CatalogCNN(**config)
    output = model(torch.randn(3, 1, 28, 28))
    loss = torch.nn.functional.cross_entropy(output, torch.tensor([1, 2, 3]))
    loss.backward()
    assert output.shape == (3, 10)
    assert torch.isfinite(loss)
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


def test_perfect_predictions_and_invalid_probabilities():
    y = np.arange(10)
    p = np.eye(10)
    result = metrics(y, p)
    assert result["macro_f1"] == result["accuracy"] == 1
    assert result["ece_10_bins"] == 0
    with pytest.raises(ValueError):
        metrics(y, p * 2)
    with pytest.raises(ValueError):
        metrics(y, np.full((10, 10), np.nan))
