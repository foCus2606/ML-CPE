from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
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
    """
    Convert one image to a numeric feature vector.

    We use a small RGB image (16x16x3 = 768 features) so that the
    TensorFlow KNN distance calculation does not consume too much memory.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype("float32") / 255.0
    return img.reshape(-1)


def load_images():
    X = []
    y = []
    loaded_per_class = {}

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset folder not found:\n{DATASET_PATH}\n"
            "Please check that data-animal/cat-breeds/cat-breeds exists."
        )

    for label, class_name in enumerate(CLASS_NAMES):
        folder = DATASET_PATH / class_name

        if not folder.exists():
            raise FileNotFoundError(f"Class folder not found: {folder}")

        count = 0
        for img_path in sorted(folder.iterdir()):
            if img_path.suffix.lower() not in VALID_EXTENSIONS:
                continue

            feature = _image_to_feature(img_path)
            if feature is None:
                print(f"[skip] cannot read: {img_path.name}")
                continue

            X.append(feature)
            y.append(label)
            count += 1

        loaded_per_class[class_name] = count

    X = np.asarray(X, dtype="float32")
    y = np.asarray(y, dtype="int32")

    return X, y, loaded_per_class


def load_data(test_size=0.2, seed=42):
    X, y, loaded_per_class = load_images()

    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.25,
        random_state=seed,
        stratify=y_temp,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype("float32")
    X_val = scaler.transform(X_val).astype("float32")
    X_test = scaler.transform(X_test).astype("float32")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "class_names": CLASS_NAMES,
        "n_rows": len(X),
        "n_features": X.shape[1],
        "loaded_per_class": loaded_per_class,
        "img_size": IMG_SIZE,
    }


if __name__ == "__main__":
    data = load_data()
    print("train :", data["X_train"].shape)
    print("val   :", data["X_val"].shape)
    print("test  :", data["X_test"].shape)
    print("classes:", data["class_names"])
    print("images per class:", data["loaded_per_class"])
