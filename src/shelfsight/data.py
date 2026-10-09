"""Kaggle Fashion-MNIST data, exact-duplicate isolation and image transforms."""

import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
SOURCE = "https://www.kaggle.com/datasets/zalando-research/fashionmnist"


def download(root):
    folder = Path(root) / "data/raw"
    folder.mkdir(parents=True, exist_ok=True)
    if not all(
        (folder / f"fashion-mnist_{part}.csv").exists() for part in ["train", "test"]
    ):
        archive = folder / "source.zip"
        urllib.request.urlretrieve(
            "https://www.kaggle.com/api/v1/datasets/download/zalando-research/fashionmnist",
            archive,
        )
        with zipfile.ZipFile(archive) as stream:
            for name in ["fashion-mnist_train.csv", "fashion-mnist_test.csv"]:
                (folder / name).write_bytes(stream.read(name))
    return folder


def image_hashes(images):
    return np.array([hashlib.sha256(image.tobytes()).hexdigest() for image in images])


def validate(frame):
    if frame.shape[1] != 785 or not frame.label.isin(range(10)).all():
        raise ValueError("Unexpected Fashion-MNIST schema")
    pixels = frame.iloc[:, 1:].to_numpy()
    if not np.isfinite(pixels).all() or not ((pixels >= 0) & (pixels <= 255)).all():
        raise ValueError("Invalid pixels")
    if not np.equal(pixels, np.floor(pixels)).all():
        raise ValueError("Pixels must be integer-valued")
    return pixels.astype(np.uint8).reshape(-1, 28, 28), frame.label.to_numpy(np.int64)


def prepare(root, train_size=12000, validation_size=3000):
    folder = download(root)
    original = {
        k: pd.read_csv(folder / f"fashion-mnist_{k}.csv") for k in ["train", "test"]
    }
    train_images, train_labels = validate(original["train"])
    test_images, test_labels = validate(original["test"])
    hashes = image_hashes(train_images)
    # Keep official test unchanged. Remove exact test matches from development data.
    test_hashes = image_hashes(test_images)
    excluded = np.isin(hashes, test_hashes)
    duplicate = pd.Series(hashes).duplicated().to_numpy()
    eligible = np.flatnonzero(~excluded & ~duplicate)
    # Sample only development data; no test labels are used to define this sample.
    selected, _ = train_test_split(
        eligible,
        train_size=train_size + validation_size,
        stratify=train_labels[eligible],
        random_state=42,
    )
    training, validation = train_test_split(
        selected,
        train_size=train_size,
        stratify=train_labels[selected],
        random_state=43,
    )
    scale_mean = float(train_images[training].mean() / 255)
    scale_std = float(train_images[training].std() / 255)
    parts = {
        "train": (
            train_images[training],
            train_labels[training],
            training,
            hashes[training],
        ),
        "validation": (
            train_images[validation],
            train_labels[validation],
            validation,
            hashes[validation],
        ),
        "test": (test_images, test_labels, np.arange(len(test_images)), test_hashes),
    }
    audit = {
        "official_train_rows": len(train_images),
        "official_test_rows": len(test_images),
        "exact_test_matches_excluded": int(excluded.sum()),
        "development_duplicate_images": int(duplicate.sum()),
        "eligible_development_rows": len(eligible),
        "train_rows": len(training),
        "validation_rows": len(validation),
        "test_rows": len(test_images),
        "normalization_mean": scale_mean,
        "normalization_std": scale_std,
        "sha256": {
            k: hashlib.sha256(
                (folder / f"fashion-mnist_{k}.csv").read_bytes()
            ).hexdigest()
            for k in ["train", "test"]
        },
    }
    return parts, audit


class ApparelDataset(Dataset):
    def __init__(self, images, labels, mean, std, augment=False):
        self.images, self.labels = images, labels
        operations = [transforms.ToPILImage()]
        if augment:
            operations.extend(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(8),
                    transforms.RandomCrop(28, padding=2),
                ]
            )
        operations.extend([transforms.ToTensor(), transforms.Normalize([mean], [std])])
        self.transform = transforms.Compose(operations)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.transform(self.images[index]), int(self.labels[index])


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
