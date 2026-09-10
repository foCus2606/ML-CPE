import os
import cv2
import numpy as np

from preprocess import preprocess_image

VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp")

SELECTED_CLASSES = [
    "abyssinian",
    "cyprus",
    "lykoi",
    "donskoy",
    "chausie"
]


def load_data(data_path, img_size=100, max_per_class=None):

    images = []
    labels = []

    classes = SELECTED_CLASSES

    print("Selected classes:", classes)

    for label, class_name in enumerate(classes):

        class_path = os.path.join(data_path, class_name)

        filenames = sorted(
            f for f in os.listdir(class_path)
            if f.lower().endswith(VALID_EXT)
        )

        loaded = 0
        skipped = 0

        for filename in filenames:

            if max_per_class is not None and loaded >= max_per_class:
                break

            image_path = os.path.join(class_path, filename)
            image = cv2.imread(image_path)

            image = preprocess_image(image, img_size)

            if image is None:
                skipped += 1
                continue

            images.append(image)
            labels.append(label)

            loaded += 1

        print(
            f"Loaded class {class_name}: "
            f"{loaded} images ({skipped} skipped)"
        )

    return np.stack(images), np.array(labels), classes