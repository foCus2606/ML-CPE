import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = "outputs"
IMG_SIZE = 100
N_SAMPLES = 5


def load_best_kernel():
    accuracy_file = os.path.join(OUTPUT_DIR, "kernel_accuracy.csv")

    data = np.genfromtxt(
        accuracy_file,
        delimiter=",",
        dtype=str,
        skip_header=1
    )

    if data.ndim == 1:
        data = np.array([data])

    accuracies = data[:, 1].astype(float)
    best_index = np.argmax(accuracies)

    return data[best_index, 0]


def test_svm(n_samples=N_SAMPLES):
    best_kernel = load_best_kernel()

    model = joblib.load(
        os.path.join(
            OUTPUT_DIR,
            f"svm_{best_kernel.lower()}.pkl"
        )
    )

    transformer = joblib.load(
        os.path.join(OUTPUT_DIR, "transformer.pkl")
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

    n_samples = min(n_samples, len(X_test))

    index = np.random.choice(
        len(X_test),
        n_samples,
        replace=False
    )

    X_sample = X_test[index]
    y_sample = y_test[index]

    predictions = model.predict(
        transformer.transform(X_sample)
    )

    cols = min(5, n_samples)
    rows = int(np.ceil(n_samples / cols))

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(3.2 * cols, 3.8 * rows)
    )

    axes = np.atleast_1d(axes).ravel()

    for i, ax in enumerate(axes):
        if i >= n_samples:
            ax.axis("off")
            continue

        pred = classes[predictions[i]]
        true = classes[y_sample[i]]
        correct = predictions[i] == y_sample[i]

        ax.imshow(
            X_sample[i].reshape(IMG_SIZE, IMG_SIZE),
            cmap="gray"
        )

        ax.set_xticks([])
        ax.set_yticks([])

        status = "Correct" if correct else "Wrong"

        ax.set_title(
            f"Pred: {pred}\nTrue: {true}\n{status}"
        )

        print(
            f"[{i + 1}] Pred: {pred:<20} "
            f"True: {true:<20} {status}"
        )

    correct_total = int(
        (predictions == y_sample).sum()
    )

    fig.suptitle(
        f"Best Kernel: {best_kernel} | "
        f"{correct_total}/{n_samples} correct"
    )

    fig.tight_layout()

    save_path = os.path.join(
        OUTPUT_DIR,
        "prediction_sample.png"
    )

    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"\nBest Kernel: {best_kernel}")
    print(f"Correct: {correct_total}/{n_samples}")
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    test_svm()
