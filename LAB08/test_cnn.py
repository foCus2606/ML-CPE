import argparse
import numpy as np
from tensorflow.keras.models import load_model
from data_loader import CLASS_NAMES, IMAGE_SIZE
import cv2

def predict_image(model_path, image_path):
    model = load_model(model_path)
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"อ่านรูปภาพไม่ได้: {image_path}")
    image = cv2.resize(image, IMAGE_SIZE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)

    probabilities = model.predict(image, verbose=0)[0]
    index = int(np.argmax(probabilities))
    print(f"Predicted class: {CLASS_NAMES[index]}")
    print(f"Confidence: {probabilities[index] * 100:.2f}%")
    print("\nAll class probabilities:")
    for class_name, probability in zip(CLASS_NAMES, probabilities):
        print(f"{class_name}: {probability * 100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict cat breed using a trained DCNN model")
    parser.add_argument("image", help="Path to an image to classify")
    parser.add_argument("--model", default="outputs/best_model.keras", help="Path to a saved Keras model")
    args = parser.parse_args()
    predict_image(args.model, args.image)
