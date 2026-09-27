"""วัด accuracy และสร้างตาราง/กราฟ."""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


def save_table(rows, path):
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_model(y_test, predictions, classes, save_path):
    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions, labels=range(len(classes)))
    print(f"Test accuracy: {accuracy:.1%}")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set(xlabel="Predicted", ylabel="True", title="Confusion matrix")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return accuracy


def plot_history(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = range(1, len(history.history["loss"]) + 1)
    for ax, metric in zip(axes, ["accuracy", "loss"]):
        ax.plot(epochs, history.history[metric], label="Train")
        ax.plot(epochs, history.history[f"val_{metric}"], label="Validation")
        ax.set(xlabel="Epoch", ylabel=metric.capitalize())
        ax.legend()
    axes[0].set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_comparison(results, save_path):
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(results))
    for offset, key, label in [(-0.2, "validation_accuracy", "Validation"),
                              (0.2, "test_accuracy", "Test")]:
        bars = ax.bar(x + offset, [r[key] * 100 for r in results], 0.4, label=label)
        ax.bar_label(bars, fmt="%.0f%%")
    ax.set_xticks(x, [f"{r['model']}\n{r['epochs']} epochs" for r in results])
    ax.set(ylabel="Accuracy (%)", ylim=(0, 115))
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_predictions(images, labels, predictions, confidence, classes, save_path):
    indices = np.linspace(0, len(images) - 1, min(4, len(images)), dtype=int)
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i >= len(indices):
            continue
        index = indices[i]
        ax.imshow(images[index])
        correct = labels[index] == predictions[index]
        ax.set_title(f"True: {classes[labels[index]]}\n"
                     f"Pred: {classes[predictions[index]]} ({confidence[index]:.0%})",
                     color="green" if correct else "red")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
