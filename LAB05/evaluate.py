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


def evaluate_model(y_test, predictions, classes, kernel_name, output_dir):
    labels = list(range(len(classes)))

    accuracy = accuracy_score(y_test, predictions)

    print("\n" + "-" * 55)
    print(f"Kernel: {kernel_name}")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        target_names=classes,
        zero_division=0
    )
    print(report)

    print("Confusion Matrix:")
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    print(matrix)

    save_path = os.path.join(
        output_dir,
        f"confusion_matrix_{kernel_name.lower()}.png"
    )
    plot_confusion_matrix(matrix, classes, kernel_name, save_path)
    print(f"Saved: {save_path}")

    return accuracy


def plot_confusion_matrix(matrix, classes, kernel_name, save_path):
    fig, ax = plt.subplots(figsize=(7, 6))

    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set_xticks(np.arange(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(classes)), classes)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix - {kernel_name}")

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


def save_accuracy_results(results, output_dir):
    csv_path = os.path.join(output_dir, "kernel_accuracy.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Kernel", "Accuracy", "Accuracy (%)"])

        for kernel_name, accuracy in results.items():
            writer.writerow([
                kernel_name,
                f"{accuracy:.4f}",
                f"{accuracy * 100:.2f}"
            ])

    print(f"Saved: {csv_path}")


def plot_accuracy_results(results, output_dir):
    names = list(results.keys())
    values = [results[name] * 100 for name in names]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(names, values)

    ax.set_xlabel("SVM Kernel")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("SVM Kernel Accuracy Comparison")
    ax.set_ylim(0, 100)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1,
            f"{value:.2f}%",
            ha="center"
        )

    fig.tight_layout()

    save_path = os.path.join(output_dir, "kernel_accuracy_comparison.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"Saved: {save_path}")
