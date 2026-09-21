import json
import os
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
N_SAMPLES = 5


def find_best_model():
    path = os.path.join(OUTPUT_DIR, 'cnn_accuracy_results.csv')
    with open(path, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    best = max(rows, key=lambda r: float(r['Accuracy']))
    return best['CNN Configuration'], int(best['Epochs'])


def test_cnn(n_samples=N_SAMPLES):
    config, epochs = find_best_model()
    model = keras.models.load_model(
        os.path.join(OUTPUT_DIR, f'{config}_epoch{epochs}.keras')
    )
    X_test = np.load(os.path.join(OUTPUT_DIR, 'X_test.npy'))
    y_test = np.load(os.path.join(OUTPUT_DIR, 'y_test.npy'))

    with open(os.path.join(OUTPUT_DIR, 'classes.json'), encoding='utf-8') as f:
        classes = json.load(f)

    n_samples = min(n_samples, len(X_test))
    index = np.random.choice(len(X_test), n_samples, replace=False)
    X_sample = X_test[index]
    y_sample = y_test[index]

    probabilities = model.predict(X_sample, verbose=0)
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)

    cols = min(5, n_samples)
    rows = int(np.ceil(n_samples / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.8 * rows))
    axes = np.atleast_1d(axes).ravel()

    for i, ax in enumerate(axes):
        if i >= n_samples:
            ax.axis('off')
            continue

        pred = classes[predictions[i]]
        true = classes[y_sample[i]]
        status = 'Correct' if predictions[i] == y_sample[i] else 'Wrong'

        ax.imshow(X_sample[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f'Pred: {pred} ({confidence[i] * 100:.0f}%)\n'
            f'True: {true}\n{status}'
        )
        print(f'[{i+1}] Pred: {pred:<12} True: {true:<12} {status}')

    correct = int((predictions == y_sample).sum())
    fig.suptitle(f'{config}, {epochs} epochs | {correct}/{n_samples} correct')
    fig.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'prediction_sample.png')
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f'\nBest model: {config}, {epochs} epochs')
    print(f'Correct: {correct}/{n_samples}')
    print(f'Saved: {save_path}')


if __name__ == '__main__':
    test_cnn()
