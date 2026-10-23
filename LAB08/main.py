import os
import json
import random
import numpy as np
import tensorflow as tf
import pandas as pd

from data_loader import load_images, CLASS_NAMES
from preprocessing import preprocess_images
from split_data import split_dataset
from cnn_model import build_dcnn, CONFIGURATIONS
from evaluate import save_history_plot, evaluate_model, save_accuracy_chart

SEED = 42
EPOCHS_TO_TEST = [10, 20]
BATCH_SIZE = 32
OUTPUT_DIR = "outputs"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    print("=" * 60)
    print("LAB 8: Deep Convolutional Neural Network (DCNN)")
    print("Dataset: Cat Breeds - 5 classes")
    print("=" * 60)

    images, labels, paths = load_images()
    images = preprocess_images(images)
    (x_train, y_train, train_paths), (x_val, y_val, val_paths), (x_test, y_test, test_paths) = split_dataset(
        images, labels, paths, test_size=0.20, val_size=0.10, random_state=SEED
    )

    print(f"\nTotal images: {len(images)}")
    print(f"Training: {len(x_train)}")
    print(f"Validation: {len(x_val)}")
    print(f"Testing: {len(x_test)}")

    histories = {}
    results = []
    best_accuracy = -1.0
    best_model_path = os.path.join(OUTPUT_DIR, "best_model.keras")
    prediction_rows = []

    # Test 2 DCNN structures x 2 epoch values = 4 experiments.
    for config_name in CONFIGURATIONS:
        for epochs in EPOCHS_TO_TEST:
            experiment_name = f"{config_name}_{epochs}epochs"
            print("\n" + "-" * 60)
            print(f"Experiment: {experiment_name}")
            model = build_dcnn(
                input_shape=x_train.shape[1:],
                num_classes=len(CLASS_NAMES),
                config_name=config_name
            )
            model.summary()

            history_object = model.fit(
                x_train, y_train,
                validation_data=(x_val, y_val),
                epochs=epochs,
                batch_size=BATCH_SIZE,
                verbose=1
            )
            histories[experiment_name] = history_object.history

            model_path = os.path.join(OUTPUT_DIR, f"{experiment_name}.keras")
            model.save(model_path)

            test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
            accuracy, predictions, probabilities = evaluate_model(
                model, x_test, y_test, CLASS_NAMES, OUTPUT_DIR, experiment_name
            )
            results.append({
                "experiment": experiment_name,
                "configuration": config_name,
                "convolutional_layers": len(CONFIGURATIONS[config_name]["filters"]),
                "epochs": epochs,
                "test_loss": float(test_loss),
                "test_accuracy": float(accuracy),
                "final_train_accuracy": float(history_object.history["accuracy"][-1]),
                "final_validation_accuracy": float(history_object.history["val_accuracy"][-1]),
                "final_train_loss": float(history_object.history["loss"][-1]),
                "final_validation_loss": float(history_object.history["val_loss"][-1]),
                "model_file": model_path
            })

            for path, true_label, pred_label, prob in zip(test_paths, y_test, predictions, probabilities):
                prediction_rows.append({
                    "experiment": experiment_name,
                    "image_path": path,
                    "actual_class": CLASS_NAMES[int(true_label)],
                    "predicted_class": CLASS_NAMES[int(pred_label)],
                    "correct": bool(true_label == pred_label),
                    "confidence": float(np.max(prob))
                })

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                model.save(best_model_path)
                print(f"New best model saved: {best_model_path} (accuracy={best_accuracy:.4f})")

    save_history_plot(histories, OUTPUT_DIR)
    save_accuracy_chart(results, OUTPUT_DIR)
    pd.DataFrame(prediction_rows).to_csv(
        os.path.join(OUTPUT_DIR, "predictions.csv"), index=False, encoding="utf-8-sig"
    )
    with open(os.path.join(OUTPUT_DIR, "training_histories.json"), "w", encoding="utf-8") as f:
        json.dump(histories, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    results_df = pd.DataFrame(results).sort_values("test_accuracy", ascending=False)
    print(results_df[["experiment", "convolutional_layers", "epochs", "test_accuracy"]].to_string(index=False))
    print(f"\nBest experiment: {results_df.iloc[0]['experiment']}")
    print(f"Best test accuracy: {results_df.iloc[0]['test_accuracy']:.4f}")
    print(f"Best model: {best_model_path}")


if __name__ == "__main__":
    main()
