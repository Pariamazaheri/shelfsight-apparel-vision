"""Visual error analysis for apparel categories and controlled experiments."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from .data import CLASSES


def render(root, parts, validation, results, predictions, history):
    folder = Path(root) / "reports/figures"
    folder.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 140, "axes.titleweight": "bold"})

    def save(name):
        plt.tight_layout()
        plt.savefig(folder / (name + ".png"), bbox_inches="tight")
        plt.close()

    fig, axes = plt.subplots(2, 5, figsize=(11, 5))
    for label, ax in enumerate(axes.flat):
        i = np.flatnonzero(parts["train"][1] == label)[0]
        ax.imshow(parts["train"][0][i], cmap="gray")
        ax.set_title(CLASSES[label])
        ax.axis("off")
    fig.suptitle("Fashion-MNIST: benchmark apparel categories")
    save("catalog_samples")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name in ["cnn_baseline", "cnn_augmentation"]:
        h = history[history.id == name]
        axes[0].plot(h.epoch, h.train_loss, label=name.replace("cnn_", "") + " train")
        axes[0].plot(
            h.epoch,
            h.validation_loss,
            ls="--",
            label=name.replace("cnn_", "") + " validation",
        )
        axes[1].plot(h.epoch, h.validation_accuracy, label=name.replace("cnn_", ""))
    axes[0].set(
        title="Cross-entropy: training versus validation", xlabel="Epoch", ylabel="Loss"
    )
    axes[1].set(
        title="Augmentation comparison", xlabel="Epoch", ylabel="Validation accuracy"
    )
    for ax in axes:
        ax.legend(fontsize=8)
    save("training_curves")
    fig, ax = plt.subplots(figsize=(9, 5))
    v = validation.sort_values("macro_f1")
    ax.barh(v.id.str.replace("cnn_", ""), v.macro_f1, color="#287f8e")
    ax.set(
        title="CNN design comparisons on validation images",
        xlabel="Macro F1",
        xlim=(0, 1),
    )
    save("cnn_ablation")
    fig, ax = plt.subplots(figsize=(9, 4))
    r = results.sort_values("macro_f1")
    ax.barh(r.id, r.macro_f1, color="#287f8e")
    ax.set(title="Held-out apparel classification", xlabel="Macro F1", xlim=(0, 1))
    save("model_comparison")
    selected = validation.sort_values("macro_f1", ascending=False).iloc[0].id
    p = predictions[predictions.id == selected]
    probabilities = p[["p" + str(i) for i in range(10)]].to_numpy()
    predicted = probabilities.argmax(1)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        confusion_matrix(p.target, predicted, normalize="true"),
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        cbar_kws={"label": "Recall share"},
        ax=ax,
    )
    ax.set(
        title="Selected CNN: category confusion",
        xlabel="Predicted category",
        ylabel="Actual category",
    )
    plt.xticks(rotation=45, ha="right")
    save("category_confusion")
    from pandas import DataFrame

    DataFrame(
        classification_report(
            p.target,
            predicted,
            labels=range(10),
            target_names=CLASSES,
            output_dict=True,
        )
    ).transpose().to_csv(Path(root) / "reports/category_metrics.csv")
    wrong = np.flatnonzero(predicted != p.target.to_numpy())
    ranked = wrong[np.argsort(probabilities[wrong].max(1))[::-1]][:12]
    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    for idx, ax in zip(ranked, axes.flat):
        image_id = int(p.iloc[idx].image_id)
        ax.imshow(parts["test"][0][image_id], cmap="gray")
        ax.set_title(
            f"Actual: {CLASSES[int(p.iloc[idx].target)]}\n"
            f"Pred: {CLASSES[predicted[idx]]} ({probabilities[idx].max():.0%})",
            fontsize=9,
        )
        ax.axis("off")
    fig.suptitle("Confident errors: cases requiring catalog review")
    save("confident_errors")
    fig, ax = plt.subplots(figsize=(7, 5))
    for name in [selected, "mobilenet_frozen", "linear_pixels"]:
        f = predictions[predictions.id == name]
        prob = f[["p" + str(i) for i in range(10)]].to_numpy()
        confidence, correct = prob.max(1), prob.argmax(1) == f.target.to_numpy()
        xs, ys = [], []
        for low in np.linspace(0, 0.9, 10):
            mask = (confidence > low) & (confidence <= low + 0.10000001)
            if mask.any():
                xs.append(confidence[mask].mean())
                ys.append(correct[mask].mean())
        ax.plot(xs, ys, marker="o", label=name)
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(
        title="Confidence reliability on held-out images",
        xlabel="Mean confidence",
        ylabel="Observed accuracy",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend(fontsize=8)
    save("confidence_reliability")
    fig, ax = plt.subplots(figsize=(8, 5))
    for row in validation.itertuples():
        ax.scatter(row.training_seconds, row.macro_f1, s=row.parameters / 1800)
        ax.annotate(
            row.id.replace("cnn_", ""),
            (row.training_seconds, row.macro_f1),
            xytext=(3, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set(
        title="CNN capacity, quality and measured CPU time",
        xlabel="Training time (seconds)",
        ylabel="Validation macro F1",
    )
    save("capacity_tradeoff")
