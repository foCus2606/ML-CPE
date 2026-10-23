import numpy as np

def preprocess_images(images):
    """Convert pixel values from 0-255 to float values from 0-1."""
    return images.astype("float32") / 255.0
