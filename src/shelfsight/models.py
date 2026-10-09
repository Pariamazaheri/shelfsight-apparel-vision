"""A compact custom CNN and frozen ImageNet feature extractor."""

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class CatalogCNN(nn.Module):
    def __init__(
        self,
        filters=16,
        kernel=3,
        stride=1,
        pooling="max",
        pool_size=2,
        depth=3,
        dropout=0.2,
        **kwargs,
    ):
        super().__init__()
        layers, incoming = [], 1
        for i in range(depth):
            outgoing = filters * min(2**i, 4)
            layers.extend(
                [
                    nn.Conv2d(
                        incoming,
                        outgoing,
                        kernel,
                        stride=stride if i == 0 else 1,
                        padding=kernel // 2,
                    ),
                    nn.ReLU(),
                ]
            )
            if i < 2:
                pool = nn.MaxPool2d if pooling == "max" else nn.AvgPool2d
                layers.append(pool(pool_size))
            incoming = outgoing
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d((3, 3)))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(incoming * 9, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def frozen_mobilenet():
    """Freeze every pretrained layer; use official ImageNet preprocessing."""
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = mobilenet_v3_small(weights=weights)
    extractor = nn.Sequential(model.features, model.avgpool, nn.Flatten()).eval()
    for parameter in extractor.parameters():
        parameter.requires_grad_(False)
    return extractor, weights.transforms(), weights.url


def extract_features(images, batch_size=64):
    model, transform, url = frozen_mobilenet()
    output = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            x = (
                torch.tensor(images[i : i + batch_size], dtype=torch.float32).unsqueeze(
                    1
                )
                / 255
            )
            # Benchmark is grayscale. Replication adds no color information.
            output.append(model(transform(x.repeat(1, 3, 1, 1))).numpy())
    import numpy as np

    return np.concatenate(output), url
