from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DATASET_PATH = (
    Path(__file__).resolve().parent.parent
    / "data-animal"
    / "cat-breeds"
    / "cat-breeds"
)

CLASS_NAMES = [
    "abyssinian",
    "cyprus",
    "lykoi",
    "donskoy",
    "chausie",
]

IMG_SIZE = 16
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _image_to_feature(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0

    return img.reshape(-1)


def load_data():
    X_raw = []
    breed_labels = []
    image_names = []

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset folder not found:\n{DATASET_PATH}\n"
            "Please check data-animal/cat-breeds/cat-breeds."
        )

    for class_name in CLASS_NAMES:
        folder = DATASET_PATH / class_name

        if not folder.exists():
            raise FileNotFoundError(f"Class folder not found: {folder}")

        for img_path in sorted(folder.iterdir()):
            if img_path.suffix.lower() not in VALID_EXTENSIONS:
                continue

            feature = _image_to_feature(img_path)
            if feature is None:
                print(f"[skip] cannot read: {img_path.name}")
                continue

            X_raw.append(feature)
            breed_labels.append(class_name)
            image_names.append(img_path.name)

    X_raw = np.asarray(X_raw, dtype="float32")

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw).astype("float32")

    df = pd.DataFrame({
        "image_name": image_names,
        "breed": breed_labels,
    })

    return {
        "X": X,
        "X_raw": X_raw,
        "df": df,
        "features": [f"pixel_{i}" for i in range(X.shape[1])],
        "class_names": CLASS_NAMES,
        "img_size": IMG_SIZE,
    }


if __name__ == "__main__":
    data = load_data()
    print("size data :", data["X"].shape)
    print("mean after scale :", data["X"].mean(axis=0).mean().round(3))
