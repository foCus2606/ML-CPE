import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.insert(0, ROOT_DIR)

from data_loader import load_image_folder, to_features, as_images
from model import train_model, predict_model
from evaluate import evaluate_model, plot_samples

# อ้างอิง Path ตามโฟลเดอร์ dataset จริงของคุณที่ซ้อนกันอยู่
DATASET_DIR = os.path.join(ROOT_DIR, "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "training_set", "training_set")
TEST_DIR = os.path.join(DATASET_DIR, "test_set", "test_set")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

PCA_COMPONENTS = 100
C = 1.0
IMG_SIZE = (64, 64)


def main():
    print("--" * 30)
    print("Classification: Dog vs Cat Prediction")
    print("--" * 30)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n[Step 1] Loading Training Set from folders...")
    train_images, y_train, CLASSES = load_image_folder(TRAIN_DIR, image_size=IMG_SIZE)
    if len(train_images) == 0:
        print("Error: ไม่พบรูปภาพใน training_set กรุณาตรวจสอบโฟลเดอร์")
        return
    X_train = to_features(train_images)
    print(f"Classes found : {CLASSES}")
    print(f"Train samples : {len(X_train)}")

    print("\n[Step 2] Loading Test Set from folders...")
    test_images, y_test, _ = load_image_folder(TEST_DIR, image_size=IMG_SIZE)
    X_test = to_features(test_images)
    print(f"Test samples  : {len(X_test)}")

    print("\n[Step 3] Training logistic regression...")
    model = train_model(X_train, y_train, PCA_COMPONENTS, C)
    print("Training completed.")

    print("\n[Step 4] Testing model...")
    predictions = predict_model(model, X_test)

    print("\n[Step 5] Evaluating model...")
    evaluate_model(y_test, predictions, CLASSES,
                   os.path.join(OUTPUT_DIR, "confusion_matrix.png"))

    plot_samples(as_images(test_images), y_test, predictions, CLASSES,
                 os.path.join(OUTPUT_DIR, "dog_cat_samples.png"))


if __name__ == "__main__":
    main()