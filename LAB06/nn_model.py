import os
from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_shape, num_classes, hidden_layers):
    """
    Fully-connected Neural Network (MLP).

    hidden_layers example:
        [128]
        [256, 128]
        [256, 128, 64]
    """

    model = keras.Sequential()
    model.add(keras.Input(shape=input_shape))

    # Standardize pixel values from 0-255 to 0-1
    model.add(layers.Rescaling(1.0 / 255))
    model.add(layers.Flatten())

    for neurons in hidden_layers:
        model.add(layers.Dense(neurons, activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))

    model.add(
        layers.Dense(
            num_classes,
            activation="softmax"
        )
    )

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    num_classes,
    hidden_layers,
    epochs,
    batch_size=32,
    output_path=None
):
    model = build_model(
        X_train.shape[1:],
        num_classes,
        hidden_layers
    )

    print("\nModel configuration:", hidden_layers)
    print("Epochs:", epochs)
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5
        )
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        model.save(output_path)

    return model, history


def predict_model(model, X_test):
    probabilities = model.predict(X_test, verbose=0)
    return probabilities.argmax(axis=1)
