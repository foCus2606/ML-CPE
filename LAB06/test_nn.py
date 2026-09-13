import csv
import json
import os
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

N_SAMPLES = 5


def get_best_model_name():
    csv_path = os.path.join(
        OUTPUT_DIR,
        "nn_comparison.csv"
    )

    with open(csv_path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    best = max(
        rows,
        key=lambda row: float(row["Accuracy"])
    )

    return best["Configuration"]


def test_nn(n_samples=N_SAMPLES):
    best_model_name = get_best_model_name()

    model = keras.models.load_model(
        os.path.join(
            OUTPUT_DIR,
            f"{best_model_name}.keras"
        )
    )

    X_test = np.load(
        os.path.join(OUTPUT_DIR, "X_test.npy")
    )

    y_test = np.load(
        os.path.join(OUTPUT_DIR, "y_test.npy")
    )

    with open(
        os.path.join(OUTPUT_DIR, "classes.json"),
        encoding="utf-8"
    ) as file:
        classes = json.load(file)

    n_samples = min(
        n_samples,
        len(X_test)
    )

    index = np.random.choice(
        len(X_test),
        n_samples,
        replace=False
    )

    X_sample = X_test[index]
    y_sample = y_test[index]

    probabilities = model.predict(
        X_sample,
        verbose=0
    )

    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)

    cols = min(5, n_samples)
    rows = int(np.ceil(n_samples / cols))

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(3.2 * cols, 4.0 * rows)
    )

    axes = np.atleast_1d(axes).ravel()

    for i, ax in enumerate(axes):
        if i >= n_samples:
            ax.axis("off")
            continue

        pred = classes[predictions[i]]
        true = classes[y_sample[i]]
        correct = predictions[i] == y_sample[i]

        ax.imshow(X_sample[i])
        ax.set_xticks([])
        ax.set_yticks([])

        status = "Correct" if correct else "Wrong"

        ax.set_title(
            f"Pred: {pred}\n"
            f"True: {true}\n"
            f"Conf: {confidence[i] * 100:.1f}%\n"
            f"{status}"
        )

        print(
            f"[{i + 1}] "
            f"Pred: {pred:<12} "
            f"True: {true:<12} "
            f"Confidence: {confidence[i] * 100:5.1f}% "
            f"{status}"
        )

    correct_total = int(
        (predictions == y_sample).sum()
    )

    fig.suptitle(
        f"Best Model: {best_model_name} | "
        f"{correct_total}/{n_samples} correct"
    )

    fig.tight_layout()

    save_path = os.path.join(
        OUTPUT_DIR,
        "prediction_sample.png"
    )

    fig.savefig(
        save_path,
        dpi=150
    )
    plt.close(fig)

    print(f"\nBest model: {best_model_name}")
    print(f"Correct: {correct_total}/{n_samples}")
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    test_nn()
