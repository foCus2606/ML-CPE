import csv
import os
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


def evaluate_model(y_test, predictions, classes):
    labels = list(range(len(classes)))

    accuracy = accuracy_score(y_test, predictions)

    print("\n------------ Evaluation ------------------")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=labels,
            target_names=classes,
            zero_division=0
        )
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels
    )

    print("Confusion Matrix:")
    print(matrix)

    return accuracy, matrix


def plot_confusion_matrix(matrix, classes, title, save_path):
    fig, ax = plt.subplots(figsize=(7, 6))

    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set_xticks(np.arange(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(classes)), classes)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    threshold = matrix.max() / 2 if matrix.max() > 0 else 0

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(
                j,
                i,
                matrix[i, j],
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black"
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_history(history, title, save_path):
    # Same idea as teacher example: training/validation accuracy and loss
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss")
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def save_results_csv(results, save_path):
    with open(save_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Configuration",
            "Hidden Layers",
            "Epochs",
            "Accuracy",
            "Accuracy (%)",
            "Epochs Actually Trained"
        ])

        for item in results:
            writer.writerow([
                item["name"],
                item["hidden_layers"],
                item["epochs"],
                f'{item["accuracy"]:.4f}',
                f'{item["accuracy"] * 100:.2f}',
                item["trained_epochs"]
            ])


def plot_comparison(results, save_path):
    labels = [
        f'{item["name"]}\n{item["epochs"]} epochs'
        for item in results
    ]
    values = [
        item["accuracy"] * 100
        for item in results
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values)

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Neural Network Configuration / Epoch Comparison")
    ax.set_ylim(0, 100)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1,
            f"{value:.2f}%",
            ha="center"
        )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
