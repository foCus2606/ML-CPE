import cv2
import numpy as np


def preprocess_image(image, img_size=100):
    """Resize one image and convert it to grayscale."""

    if image is None or image.size == 0:
        return None
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize image
    image = cv2.resize(
        image,
        (img_size, img_size),
        interpolation=cv2.INTER_AREA
    )

    return image


def to_features(images):
    """(n, h, w) -> (n, h*w), normalize pixels to 0-1."""

    features = images.reshape(len(images), -1).astype(np.float32)
    features /= 255.0

    return features
