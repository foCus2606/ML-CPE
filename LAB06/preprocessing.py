import cv2
import numpy as np


def preprocess_image(image, img_size=100):
    """Resize one image to img_size x img_size RGB."""

    if image is None or image.size == 0:
        return None

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(
        image,
        (img_size, img_size),
        interpolation=cv2.INTER_AREA
    )

    return image


def to_features(images):
    """
    Keep image shape (n, h, w, 3).
    Scaling 0-255 -> 0-1 is done inside the model.
    """
    return np.ascontiguousarray(images, dtype=np.uint8)
