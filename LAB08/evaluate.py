import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

def save_history_plot(histories, output_dir):
    """Save combined train/validation accuracy and loss curves for all experiments."""
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    for name, history in histories.items():
        plt.plot(history["accuracy"], label=f"{name} train")
        plt.plot(history["val_accuracy"], linestyle="--", label=f"{name} validation")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_comparison.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    for name, history in histories.items():
        plt.plot(history["loss"], label=f"{name} train")
        plt.plot(history["val_loss"], linestyle="--", label=f"{name} validation")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_comparison.png"), dpi=150)
    plt.close()

def evaluate_model(model, x_test, y_test, class_names, output_dir, experiment_name):
    os.makedirs(output_dir, exist_ok=True)
    probabilities = model.predict(x_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    accuracy = accuracy_score(y_test, predictions)

    matrix = confusion_matrix(y_test, predictions, labels=list(range(len(class_names))))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 7))
    display.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    ax.set_title(f"Confusion Matrix - {experiment_name}")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"confusion_matrix_{experiment_name}.png"), dpi=150)
    plt.close(fig)

    report = classification_report(
        y_test, predictions, labels=list(range(len(class_names))),
        target_names=class_names, zero_division=0
    )
    with open(os.path.join(output_dir, f"classification_report_{experiment_name}.txt"), "w", encoding="utf-8") as f:
        f.write(f"Experiment: {experiment_name}\nTest accuracy: {accuracy:.4f}\n\n{report}")

    return accuracy, predictions, probabilities

def save_accuracy_chart(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "accuracy_results.csv"), index=False, encoding="utf-8-sig")

    plt.figure(figsize=(10, 6))
    labels = df["experiment"].tolist()
    values = df["test_accuracy"].tolist()
    bars = plt.bar(labels, values)
    plt.ylim(0, 1)
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy by DCNN Configuration and Epochs")
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.3)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, value + 0.015, f"{value:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "test_accuracy_comparison.png"), dpi=150)
    plt.close()
