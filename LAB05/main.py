import json
import os

import joblib
import numpy as np

from data_load import load_data
from preprocess import to_features
from split_data import split_dataset
from svm_model import train_svm_models, predict_svm
from evaluate import (
    evaluate_model,
    save_accuracy_results,
    plot_accuracy_results
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "D:\ML\LAB05\data-animal\cat-breeds\cat-breeds")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

IMG_SIZE = 100
TEST_SIZE = 0.2
MAX_PER_CLASS = 200


def main():
    print("=" * 60)
    print("LAB 5: Support Vector Machine - Cat Breed Classification")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n[Step 1] Loading dataset...")
    images, labels, classes = load_data(
        DATA_PATH,
        IMG_SIZE,
        MAX_PER_CLASS
    )

    print("\nDataset loaded successfully.")
    print(f"Total images : {len(images)}")
    print(f"Classes      : {classes}")

    if len(classes) < 2:
        raise ValueError("Dataset must contain at least 2 class folders.")

    with open(
        os.path.join(OUTPUT_DIR, "classes.json"),
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(classes, file, ensure_ascii=False, indent=2)

    print("\n[Step 2] Preprocessing images...")
    X = to_features(images)
    y = labels

    print(f"Feature shape: {X.shape}")

    print("\n[Step 3] Splitting dataset...")
    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y,
        TEST_SIZE
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")

    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

    print("\n[Step 4] Training SVM models...")
    models, transformer = train_svm_models(X_train, y_train)

    joblib.dump(
        transformer,
        os.path.join(OUTPUT_DIR, "transformer.pkl")
    )

    print("\n[Step 5-6] Prediction and Evaluation...")

    results = {}

    for kernel_name, model in models.items():
        predictions = predict_svm(
            model,
            transformer,
            X_test
        )

        accuracy = evaluate_model(
            y_test,
            predictions,
            classes,
            kernel_name,
            OUTPUT_DIR
        )

        results[kernel_name] = accuracy

        model_path = os.path.join(
            OUTPUT_DIR,
            f"svm_{kernel_name.lower()}.pkl"
        )
        joblib.dump(model, model_path)

    print("\n" + "=" * 60)
    print("SVM KERNEL ACCURACY COMPARISON")
    print("=" * 60)

    for kernel_name, accuracy in results.items():
        print(f"{kernel_name:<12}: {accuracy * 100:.2f}%")

    best_kernel = max(results, key=results.get)

    print("-" * 60)
    print(
        f"Best Kernel : {best_kernel} "
        f"({results[best_kernel] * 100:.2f}%)"
    )

    save_accuracy_results(results, OUTPUT_DIR)
    plot_accuracy_results(results, OUTPUT_DIR)

    with open(
        os.path.join(OUTPUT_DIR, "best_kernel.txt"),
        "w",
        encoding="utf-8"
    ) as file:
        file.write(
            f"Best Kernel: {best_kernel}\n"
            f"Accuracy: {results[best_kernel] * 100:.2f}%\n"
        )

    print("\nFinished.")


if __name__ == "__main__":
    main()
