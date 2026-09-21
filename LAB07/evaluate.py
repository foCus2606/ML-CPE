import csv
import os
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

def smooth_curve(values, weight=0.6):

    values = np.array(values, dtype=float)

    if len(values) == 0:
        return values

    smooth = [values[0]]

    for value in values[1:]:

        new_value = (
            weight * smooth[-1]
            + (1 - weight) * value
        )

        smooth.append(new_value)

    return np.array(smooth)

def evaluate_model(y_test, predictions, classes, name, output_dir):

    labels = list(range(len(classes)))

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print('\n' + '-' * 55)

    print(name)

    print(
        f'Accuracy: {accuracy * 100:.2f}%'
    )

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

    print(
        'Confusion Matrix (count):'
    )

    print(matrix)

    row_sum = matrix.sum(
        axis=1,
        keepdims=True
    )

    matrix_pct = np.divide(
        matrix * 100.0,
        row_sum,
        out=np.zeros_like(
            matrix,
            dtype=float
        ),
        where=row_sum != 0
    )

    fig, ax = plt.subplots(
        figsize=(8.5, 7)
    )

    image = ax.imshow(
        matrix_pct,
        cmap='Blues',
        vmin=0,
        vmax=100
    )

 
    cbar = fig.colorbar(
        image,
        ax=ax
    )

    cbar.set_label(
        'Percentage (%)'
    )

    ax.set_xticks(
        np.arange(len(classes)),
        classes,
        rotation=35,
        ha='right'
    )

    ax.set_yticks(
        np.arange(len(classes)),
        classes
    )

    ax.set_xlabel(
        'Predicted class',
        fontsize=11
    )

    ax.set_ylabel(
        'True class',
        fontsize=11
    )

    ax.set_title(
        f'Confusion Matrix (%) - {name}',
        fontsize=14,
        pad=14
    )


    for i in range(len(classes)):

        for j in range(len(classes)):

            value = matrix_pct[i, j]

            ax.text(
                j,
                i,
                f'{value:.1f}%',
                ha='center',
                va='center',

                color=(
                    'white'
                    if value >= 50
                    else 'black'
                ),

                fontsize=10
            )

    ax.set_ylim(
        len(classes) - 0.5,
        -0.5
    )

    fig.tight_layout()

    save_path = os.path.join(
        output_dir,
        f'confusion_{name}.png'
    )

    fig.savefig(
        save_path,
        dpi=180,
        bbox_inches='tight'
    )

    plt.close(fig)

    return accuracy

def plot_history(history, name, output_dir):

    epochs = np.arange(
        1,
        len(history.history['accuracy']) + 1
    )
    train_acc = (
        np.array(
            history.history['accuracy']
        )
        * 100
    )

    val_acc = (
        np.array(
            history.history['val_accuracy']
        )
        * 100
    )
    train_acc_smooth = smooth_curve(
        train_acc,
        weight=0.6
    )

    val_acc_smooth = smooth_curve(
        val_acc,
        weight=0.6
    )

    fig, ax = plt.subplots(
        figsize=(8.5, 5.5)
    )

    ax.plot(
        epochs,
        train_acc_smooth,
        linewidth=2.5,
        label='Train'
    )

    ax.plot(
        epochs,
        val_acc_smooth,
        linewidth=2.5,
        label='Validation'
    )

    ax.set_xlabel(
        'Epoch'
    )

    ax.set_ylabel(
        'Accuracy (%)'
    )

    ax.set_title(
        f'Training & Validation Accuracy - {name}',
        fontsize=14,
        pad=12
    )

    ax.set_ylim(
        0,
        100
    )

    ax.set_xticks(
        epochs
    )

    ax.grid(
        True,
        alpha=0.20
    )

    ax.legend(
        frameon=True
    )

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            output_dir,
            f'accuracy_{name}.png'
        ),
        dpi=180,
        bbox_inches='tight'
    )

    plt.close(fig)

    train_loss = np.array(
        history.history['loss']
    )

    val_loss = np.array(
        history.history['val_loss']
    )

    train_loss_smooth = smooth_curve(
        train_loss,
        weight=0.6
    )

    val_loss_smooth = smooth_curve(
        val_loss,
        weight=0.6
    )

    fig, ax = plt.subplots(
        figsize=(8.5, 5.5)
    )

    ax.plot(
        epochs,
        train_loss_smooth,
        linewidth=2.5,
        label='Train'
    )

    ax.plot(
        epochs,
        val_loss_smooth,
        linewidth=2.5,
        label='Validation'
    )

    ax.set_xlabel(
        'Epoch'
    )

    ax.set_ylabel(
        'Loss'
    )

    ax.set_title(
        f'Training & Validation Loss - {name}',
        fontsize=14,
        pad=12
    )

    ax.set_xticks(
        epochs
    )

    ax.grid(
        True,
        alpha=0.20
    )

    ax.legend(
        frameon=True
    )

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            output_dir,
            f'loss_{name}.png'
        ),
        dpi=180,
        bbox_inches='tight'
    )

    plt.close(fig)

def save_results(results, output_dir):

    path = os.path.join(
        output_dir,
        'cnn_accuracy_results.csv'
    )

    with open(
        path,
        'w',
        newline='',
        encoding='utf-8'
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            'CNN Configuration',
            'Epochs',
            'Accuracy',
            'Accuracy (%)'
        ])

        for config, epochs, accuracy in results:

            writer.writerow([
                config,
                epochs,
                f'{accuracy:.4f}',
                f'{accuracy * 100:.2f}'
            ])

    names = [
        f'{config}\n{epochs} epochs'
        for config, epochs, _ in results
    ]

    values = [
        accuracy * 100
        for _, _, accuracy in results
    ]

    fig, ax = plt.subplots(
        figsize=(9, 5.8)
    )

    bars = ax.bar(
        names,
        values,
        width=0.62
    )

    ax.set_ylabel(
        'Test Accuracy (%)'
    )

    ax.set_xlabel(
        'CNN configuration'
    )

    ax.set_title(
        'CNN Accuracy Comparison',
        fontsize=14,
        pad=12
    )

    ax.set_ylim(
        0,
        100
    )

    ax.grid(
        axis='y',
        alpha=0.25
    )

    ax.set_axisbelow(
        True
    )

    for bar, value in zip(
        bars,
        values
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,

            value + 1.5,

            f'{value:.2f}%',

            ha='center',
            va='bottom',
            fontsize=10
        )

    fig.tight_layout()

    bar_path = os.path.join(
        output_dir,
        'cnn_accuracy_comparison.png'
    )

    fig.savefig(
        bar_path,
        dpi=180,
        bbox_inches='tight'
    )

    plt.close(fig)

    print(
        f'Saved: {path}'
    )

    print(
        f'Saved: {bar_path}'
    )