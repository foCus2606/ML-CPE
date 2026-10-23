from tensorflow.keras import layers, models

CONFIGURATIONS = {
    "CNN_1": {
        "filters": [32, 64],
        "dense_units": 128,
        "description": "2 Convolutional layers"
    },
    "CNN_2": {
        "filters": [32, 64, 128],
        "dense_units": 128,
        "description": "3 Convolutional layers"
    },
}

def build_dcnn(input_shape=(100, 100, 3), num_classes=5, config_name="CNN_1"):
    """Build a DCNN using the selected number of convolutional layers."""
    if config_name not in CONFIGURATIONS:
        raise ValueError(f"Unknown configuration: {config_name}")

    config = CONFIGURATIONS[config_name]
    model = models.Sequential(name=config_name)
    model.add(layers.Input(shape=input_shape))

    for filters in config["filters"]:
        model.add(layers.Conv2D(filters, (3, 3), padding="same", activation="relu"))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(config["dense_units"], activation="relu"))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
