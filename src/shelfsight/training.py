"""Seeded training with augmentation limited to training images."""

import random
import time
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import ApparelDataset


def seed_everything():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)


def fit(model, parts, audit, config):
    start = time.perf_counter()
    loaders = {}
    for split in ["train", "validation"]:
        images, labels, _, _ = parts[split]
        dataset = ApparelDataset(
            images,
            labels,
            audit["normalization_mean"],
            audit["normalization_std"],
            augment=split == "train" and config.get("augment", False),
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=128,
            shuffle=split == "train",
            generator=torch.Generator().manual_seed(42),
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()
    best, stale, best_epoch = float("inf"), 0, 0
    checkpoint = deepcopy(model.state_dict())
    history = []
    for epoch in range(1, config.get("epochs", 8) + 1):
        row = {"epoch": epoch}
        for split, loader in loaders.items():
            model.train(split == "train")
            total, correct, loss_sum = 0, 0, 0.0
            with torch.set_grad_enabled(split == "train"):
                for x, y in loader:
                    if split == "train":
                        optimizer.zero_grad()
                    logits = model(x)
                    loss = criterion(logits, y)
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite training loss")
                    if split == "train":
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 5)
                        optimizer.step()
                    loss_sum += loss.item() * len(y)
                    correct += int((logits.argmax(1) == y).sum())
                    total += len(y)
            row[split + "_loss"] = loss_sum / total
            row[split + "_accuracy"] = correct / total
        history.append(row)
        if row["validation_loss"] < best - 1e-5:
            best, stale, best_epoch = row["validation_loss"], 0, epoch
            checkpoint = deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= 3:
            break
    model.load_state_dict(checkpoint)
    model.eval()
    return history, {
        "training_seconds": time.perf_counter() - start,
        "parameters": sum(p.numel() for p in model.parameters()),
        "best_epoch": best_epoch,
        "epochs_run": len(history),
    }


def predict(model, part, audit):
    dataset = ApparelDataset(
        part[0], part[1], audit["normalization_mean"], audit["normalization_std"]
    )
    model.eval()
    output = []
    with torch.no_grad():
        for x, _ in DataLoader(dataset, batch_size=256):
            output.append(torch.softmax(model(x), dim=1).numpy())
    return np.concatenate(output)
