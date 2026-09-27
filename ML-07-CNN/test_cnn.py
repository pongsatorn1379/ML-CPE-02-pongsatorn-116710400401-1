"""โหลดโมเดลที่ฝึกไว้ แล้วดูผลทำนาย 4 ภาพ."""
import json
import os
import numpy as np
from tensorflow import keras
from cnn_model import predict_model
from evaluate import plot_predictions

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "saved_model")


def test_cnn():
    model = keras.models.load_model(os.path.join(MODEL_DIR, "cnn_model.keras"))
    images = np.load(os.path.join(MODEL_DIR, "X_test.npy"))
    labels = np.load(os.path.join(MODEL_DIR, "y_test.npy"))
    with open(os.path.join(MODEL_DIR, "classes.json"), encoding="utf-8") as file:
        classes = json.load(file)

    # เลือกภาพกระจายทั่ว test เพื่อให้เห็นทั้ง BMW และ Honda
    indices = np.linspace(0, len(images) - 1, min(4, len(images)), dtype=int)
    images, labels = images[indices], labels[indices]
    predictions, confidence = predict_model(model, images)
    for i in range(len(images)):
        print(f"Image {i + 1}: true={classes[labels[i]]}, "
              f"predicted={classes[predictions[i]]}, confidence={confidence[i]:.0%}")
    path = os.path.join(OUTPUT_DIR, "prediction_sample.png")
    plot_predictions(images, labels, predictions, confidence, classes, path)
    print(f"Saved: {path}")


if __name__ == "__main__":
    test_cnn()
