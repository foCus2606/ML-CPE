import json
import os
import numpy as np

from data_loader import load_data
from preprocessing import to_features
from split_data import split_dataset
from nn_model import train_model, predict_model
from evaluate import (
    evaluate_model,
    plot_confusion_matrix,
    plot_history,
    save_results_csv,
    plot_comparison
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "D:\ML\LAB06\data-animal\cat-breeds\cat-breeds")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

IMG_SIZE = 100
TEST_SIZE = 0.2
VAL_SIZE = 0.1
MAX_PER_CLASS = 200
BATCH_SIZE = 32

EXPERIMENTS = [
    {
        "name": "NN_A",
        "hidden_layers": [128],
        "epochs": 10
    },
    {
        "name": "NN_B",
        "hidden_layers": [256, 128],
        "epochs": 20
    },
    {
        "name": "NN_C",
        "hidden_layers": [256, 128, 64],
        "epochs": 30
    }
]


def main():
    print("=" * 65)
    print("LAB 6: Neural Network - 5 Cat Breed Classification")
    print("=" * 65)

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

    with open(
        os.path.join(OUTPUT_DIR, "classes.json"),
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            classes,
            file,
            ensure_ascii=False,
            indent=2
        )

    print("\n[Step 2] Preprocessing images...")
    X = to_features(images)
    y = labels

    print(f"Feature shape: {X.shape}")

    print("\n[Step 3] Splitting dataset...")

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        X,
        y,
        TEST_SIZE,
        VAL_SIZE
    )

    print(f"Training samples  : {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Testing samples   : {len(X_test)}")

    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

    results = []

    for exp in EXPERIMENTS:
        print("\n" + "=" * 65)
        print(
            f'Experiment: {exp["name"]} | '
            f'Hidden layers: {exp["hidden_layers"]} | '
            f'Epochs: {exp["epochs"]}'
        )
        print("=" * 65)

        model_path = os.path.join(
            OUTPUT_DIR,
            f'{exp["name"]}.keras'
        )

        model, history = train_model(
            X_train,
            y_train,
            X_val,
            y_val,
            len(classes),
            exp["hidden_layers"],
            exp["epochs"],
            BATCH_SIZE,
            model_path
        )

        predictions = predict_model(
            model,
            X_test
        )

        accuracy, matrix = evaluate_model(
            y_test,
            predictions,
            classes
        )

        trained_epochs = len(history.history["loss"])

        results.append({
            "name": exp["name"],
            "hidden_layers": str(exp["hidden_layers"]),
            "epochs": exp["epochs"],
            "accuracy": accuracy,
            "trained_epochs": trained_epochs
        })

        plot_history(
            history,
            (
                f'{exp["name"]} - '
                f'Hidden Layers {exp["hidden_layers"]} - '
                f'{exp["epochs"]} Epochs'
            ),
            os.path.join(
                OUTPUT_DIR,
                f'{exp["name"]}_history.png'
            )
        )

        plot_confusion_matrix(
            matrix,
            classes,
            f'Confusion Matrix - {exp["name"]}',
            os.path.join(
                OUTPUT_DIR,
                f'{exp["name"]}_confusion_matrix.png'
            )
        )

    print("\n" + "=" * 65)
    print("FINAL COMPARISON")
    print("=" * 65)

    for item in results:
        print(
            f'{item["name"]:<8} '
            f'Hidden={item["hidden_layers"]:<18} '
            f'Epochs={item["epochs"]:<3} '
            f'Accuracy={item["accuracy"] * 100:.2f}%'
        )

    best = max(
        results,
        key=lambda item: item["accuracy"]
    )

    print("-" * 65)
    print(
        f'Best model: {best["name"]} | '
        f'Hidden Layers: {best["hidden_layers"]} | '
        f'Epochs: {best["epochs"]} | '
        f'Accuracy: {best["accuracy"] * 100:.2f}%'
    )

    save_results_csv(
        results,
        os.path.join(
            OUTPUT_DIR,
            "nn_comparison.csv"
        )
    )

    plot_comparison(
        results,
        os.path.join(
            OUTPUT_DIR,
            "nn_accuracy_comparison.png"
        )
    )

    with open(
        os.path.join(OUTPUT_DIR, "best_model.txt"),
        "w",
        encoding="utf-8"
    ) as file:
        file.write(
            f'Best model: {best["name"]}\n'
            f'Hidden Layers: {best["hidden_layers"]}\n'
            f'Epochs configured: {best["epochs"]}\n'
            f'Epochs actually trained: {best["trained_epochs"]}\n'
            f'Accuracy: {best["accuracy"] * 100:.2f}%\n'
        )

    print("\nFinished.")


if __name__ == "__main__":
    main()
