import os
import sys
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))

# ชี้ไปที่โฟลเดอร์ dataset หมา-แมวของคุณ
DATASET_DIR = os.path.join(ROOT_DIR, "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "training_set", "training_set")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

TEST_SIZE = 0.2
PCA_COMPONENTS = 50
ALPHA = 1.0
IMG_SIZE = (64, 64)


def load_image_folder(base_folder, image_size=(64, 64)):
    images = []
    labels = []
    if not os.path.exists(base_folder):
        return np.array(images), np.array(labels)
    
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
    return np.array(images), np.array(labels)


def to_features(images):
    return images.astype(np.float32) / 255.0


def as_images(pixels):
    return pixels


def train_model(X_train, y_train, pca_components=50, alpha=1.0):
    model = make_pipeline(
        StandardScaler(),
        PCA(n_components=pca_components, random_state=42),
        Ridge(alpha=alpha),
    )
    model.fit(X_train, y_train)
    return model


def predict_model(model, X_test):
    return model.predict(X_test)


def evaluate_model(y_test, predictions, save_path):
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("\n------------ Evaluation (Regression) ------------------")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R2   : {r2:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(y_test, predictions, s=6, alpha=0.25)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    axes[0].set_xlabel("Actual value")
    axes[0].set_ylabel("Predicted value")
    axes[0].set_title("Predicted vs Actual")

    axes[1].hist(predictions - y_test, bins=60)
    axes[1].axvline(0, color="r", linestyle="--")
    axes[1].set_xlabel("Residuals")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residuals Distribution")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_samples(images, y_test, predictions, save_path, n_samples=8):
    n_samples = min(n_samples, len(images))
    if n_samples == 0:
        return
    index = np.random.choice(len(images), n_samples, replace=False)

    fig, axes = plt.subplots(2, 4, figsize=(10, 6))
    axes = axes.ravel()

    for ax, i in zip(axes, index):
        ax.imshow(images[i], cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Pred: {predictions[i]:.1f}\nTrue: {y_test[i]:.1f}")

    fig.suptitle("Regression Sample Predictions")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    print("--" * 30)
    print("Regression: Using Dog/Cat images for continuous prediction")
    print("--" * 30)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n[Step 1] Loading images from training_set...")
    images, labels = load_image_folder(TRAIN_DIR, image_size=IMG_SIZE)
    if len(images) == 0:
        print("Error: ไม่พบรูปภาพ กรุณาตรวจสอบโฟลเดอร์ dataset ใน D:\\ML\\dataset")
        return

    X = to_features(images.reshape(len(images), -1))
    # แปลง label ให้เป็นค่าต่อเนื่อง (Continuous value) เพื่อทำ Regression
    y = labels.astype(np.float32) * 10.0 + 5.0 

    print(f"Total samples : {len(X)}")

    print("\n[Step 2] Splitting dataset...")
    X_train, X_test, y_train, y_test, _, test_index = train_test_split(
        X, y, np.arange(len(X)), test_size=TEST_SIZE, random_state=42
    )

    print("\n[Step 3] Training Ridge regression...")
    model = train_model(X_train, y_train, PCA_COMPONENTS, ALPHA)
    print("Training completed.")

    print("\n[Step 4] Testing and Evaluating model...")
    predictions = predict_model(model, X_test)

    evaluate_model(y_test, predictions, os.path.join(OUTPUT_DIR, "regression_results.png"))
    plot_samples(images[test_index], y_test, predictions, os.path.join(OUTPUT_DIR, "regression_samples.png"))


if __name__ == "__main__":
    main()