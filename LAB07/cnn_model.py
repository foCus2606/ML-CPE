import os
from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_shape, num_classes, config_name='CNN_1'):
    """สร้าง CNN 2 แบบ เพื่อใช้เปรียบเทียบตามโจทย์ LAB 7"""
    model = keras.Sequential()
    model.add(keras.Input(shape=input_shape))
    model.add(layers.Rescaling(1.0 / 255))

    model.add(layers.RandomFlip('horizontal'))
    model.add(layers.RandomRotation(0.1))

    if config_name == 'CNN_1':
      
        model.add(layers.Conv2D(32, 3, padding='same', activation='relu'))
        model.add(layers.MaxPooling2D())
        model.add(layers.Conv2D(64, 3, padding='same', activation='relu'))
        model.add(layers.MaxPooling2D())
        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.3))

    elif config_name == 'CNN_2':
     
        model.add(layers.Conv2D(32, 3, padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D())
        model.add(layers.Conv2D(64, 3, padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D())
        model.add(layers.Conv2D(128, 3, padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D())
        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.3))
    else:
        raise ValueError('config_name must be CNN_1 or CNN_2')

    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_model(X_train, y_train, X_val, y_val, num_classes,
                config_name, epochs, output_dir, batch_size=32):
    model = build_model(X_train.shape[1:], num_classes, config_name)

    print(f'\nTraining {config_name} | Epochs = {epochs}')
    model.summary()

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    os.makedirs(output_dir, exist_ok=True)
    model.save(os.path.join(output_dir, f'{config_name}_epoch{epochs}.keras'))
    return model, history


def predict_model(model, X_test):
    probabilities = model.predict(X_test, verbose=0)
    return probabilities.argmax(axis=1)
