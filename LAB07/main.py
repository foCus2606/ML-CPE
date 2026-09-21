import json
import os
import numpy as np

from data_loader import load_data
from preprocessing import to_features
from split_data import split_dataset
from cnn_model import train_model, predict_model
from evaluate import evaluate_model, plot_history, save_results

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = r'E:\LAB07_CNN\cat-breeds\cat-breeds'
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

IMG_SIZE = 100
TEST_SIZE = 0.2
VAL_SIZE = 0.1
MAX_PER_CLASS = 200
BATCH_SIZE = 32

EXPERIMENTS = [
    ('CNN_1', 10),
    ('CNN_1', 20),
    ('CNN_2', 10),
    ('CNN_2', 20)
]


def main():
    print('=' * 60)
    print('LAB 7: CNN - Cat Breed Classification')
    print('=' * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('\n[Step 1] Loading dataset...')
    images, labels, classes = load_data(DATA_PATH, IMG_SIZE, MAX_PER_CLASS)
    print(f'Total images : {len(images)}')
    print(f'Classes      : {classes}')

    with open(os.path.join(OUTPUT_DIR, 'classes.json'), 'w', encoding='utf-8') as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)

    print('\n[Step 2] Preprocessing images...')
    X = to_features(images)
    y = labels
    print(f'Feature shape: {X.shape}')

    print('\n[Step 3] Splitting dataset...')
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        X, y, TEST_SIZE, VAL_SIZE
    )
    print(f'Training samples  : {len(X_train)}')
    print(f'Validation samples: {len(X_val)}')
    print(f'Testing samples   : {len(X_test)}')

    np.save(os.path.join(OUTPUT_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(OUTPUT_DIR, 'y_test.npy'), y_test)

    print('\n[Step 4-6] Training, Prediction and Evaluation...')
    results = []

    for config_name, epochs in EXPERIMENTS:
        name = f'{config_name}_epoch{epochs}'

        model, history = train_model(
            X_train, y_train, X_val, y_val,
            len(classes), config_name, epochs,
            OUTPUT_DIR, BATCH_SIZE
        )

        predictions = predict_model(model, X_test)
        accuracy = evaluate_model(
            y_test, predictions, classes, name, OUTPUT_DIR
        )
        plot_history(history, name, OUTPUT_DIR)
        results.append((config_name, epochs, accuracy))

    save_results(results, OUTPUT_DIR)

    print('\n' + '=' * 60)
    print('CNN ACCURACY COMPARISON')
    print('=' * 60)
    for config_name, epochs, accuracy in results:
        print(f'{config_name:<8} | {epochs:>2} epochs | {accuracy * 100:.2f}%')

    best = max(results, key=lambda x: x[2])
    print('-' * 60)
    print(f'Best: {best[0]} | {best[1]} epochs | {best[2] * 100:.2f}%')
    print('\nFinished. Results are in the outputs folder.')


if __name__ == '__main__':
    main()
