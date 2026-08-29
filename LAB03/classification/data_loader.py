import os
import numpy as np
from PIL import Image

def load_image_folder(base_folder, image_size=(64, 64)):
    images = []
    labels = []
    
    classes = sorted([d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))])
    class_map = {cls_name: i for i, cls_name in enumerate(classes)}
    
    for cls_name in classes:
        cls_folder = os.path.join(base_folder, cls_name)
        for filename in os.listdir(cls_folder):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(cls_folder, filename)
                try:
                    with Image.open(img_path) as img:
                        img = img.convert('L').resize(image_size)
                        images.append(np.array(img))
                        labels.append(class_map[cls_name])
                except Exception:
                    pass
                    
    return np.array(images), np.array(labels), classes

def to_features(images):
    return images.reshape(len(images), -1)

def as_images(pixels):
    return pixels