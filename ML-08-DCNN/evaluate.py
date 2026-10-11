import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def plot_history(history, output_dir):
    values = history.history
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = range(1, len(values["loss"]) + 1)
    for ax, metric in zip(axes, ["accuracy", "loss"]):
        ax.plot(epochs, values[metric], label="train")
        ax.plot(epochs, values["val_" + metric], label="validation")
        ax.set(xlabel="Epoch", ylabel=metric.capitalize(), title=metric.capitalize())
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_history.png", dpi=150)
    plt.close(fig)


def plot_predictions(images, y, probabilities, classes, save_path, count=6):
    predictions = (probabilities >= 0.5).astype(int)
    count = min(count, len(images))
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for i, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if i >= count:
            continue
        pred = predictions[i]
        confidence = probabilities[i] if pred == 1 else 1 - probabilities[i]
        ax.imshow(images[i])
        ax.set_title(f"Pred: {classes[pred]} ({confidence:.1%})\nTrue: {classes[y[i]]}",
                     color="green" if pred == y[i] else "red")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def evaluate_model(y_test, probabilities, classes, images, output_dir):
    predictions = (probabilities >= 0.5).astype(int)
    accuracy = float(accuracy_score(y_test, predictions))
    print(f"Test accuracy: {accuracy:.2%}")
    print(classification_report(y_test, predictions, labels=[0, 1],
                                target_names=classes, zero_division=0))
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], classes)
    ax.set_yticks([0, 1], classes)
    ax.set(xlabel="Predicted", ylabel="True", title="Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, matrix[i, j], ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    plot_predictions(images, y_test, probabilities, classes, output_dir / "prediction_sample.png")
    return accuracy
