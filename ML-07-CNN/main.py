"""ไฟล์หลัก: อ่านตามขั้นตอนที่ 1–6."""
import json
import os
import numpy as np
from tensorflow import keras
from data_loader import load_data
from preprocessing import to_features
from split_data import split_dataset
from cnn_model import build_model, train_model, predict_model
from evaluate import (save_table, evaluate_model, plot_history,
                      plot_comparison, plot_predictions)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "saved_model")
IMG_SIZE = 64
EPOCHS = [10, 20, 30]
CONFIGURATIONS = [
    {"name": "CNN_1_layer", "conv_layers": 1, "neurons": 32},
    {"name": "CNN_2_layers", "conv_layers": 2, "neurons": 64},
]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ขั้นที่ 1: อ่านภาพและ label จากชื่อโฟลเดอร์
    print("[1] Load dataset")
    images, labels, classes = load_data(DATA_PATH, IMG_SIZE)

    # ขั้นที่ 2: ปรับค่าพิกเซลจาก 0–255 เป็น 0–1
    print("[2] Preprocess images")
    X = to_features(images)
    y = labels

    # ขั้นที่ 3: แบ่ง train 60%, validation 20%, test 20%
    print("[3] Split dataset")
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)
    print(f"Train: {len(X_train)}, validation: {len(X_val)}, test: {len(X_test)}")
    np.save(os.path.join(MODEL_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(MODEL_DIR, "y_test.npy"), y_test)
    with open(os.path.join(MODEL_DIR, "classes.json"), "w", encoding="utf-8") as file:
        json.dump(classes, file)

    # ขั้นที่ 4: ฝึก CNN สองแบบ แต่ละแบบลอง 10, 20, 30 epochs
    print("[4] Train and compare")
    results = []
    best_score = None
    for config in CONFIGURATIONS:
        for epochs in EPOCHS:
            keras.backend.clear_session()
            keras.utils.set_random_seed(42)
            model = build_model(X_train.shape[1:], len(classes),
                                config["conv_layers"], config["neurons"])
            print(f"\n{config['name']}, {epochs} epochs")
            history = train_model(model, X_train, y_train, X_val, y_val, epochs)

            train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
            val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
            test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
            row = {"model": config["name"], "epochs": epochs,
                   "train_accuracy": train_acc, "validation_accuracy": val_acc,
                   "test_accuracy": test_acc, "train_loss": train_loss,
                   "validation_loss": val_loss, "test_loss": test_loss}
            results.append(row)
            # เลือกด้วย validation เท่านั้น; ถ้า accuracy เท่ากันดู loss ต่ำกว่า
            score = (val_acc, -val_loss)
            if best_score is None or score > best_score:
                best_score = score
                best_result = row
                model.save(os.path.join(MODEL_DIR, "cnn_model.keras"))
                # เก็บกราฟเฉพาะโมเดลที่เลือก ไม่สร้างกราฟ/CSV แยกทุกการทดลอง
                plot_history(history, os.path.join(OUTPUT_DIR, "training_history.png"))

    # ขั้นที่ 5: โหลดโมเดลที่เลือกมาทำนาย test
    print("[5] Predict test images")
    best_model = keras.models.load_model(os.path.join(MODEL_DIR, "cnn_model.keras"))
    predictions, confidence = predict_model(best_model, X_test)

    # ขั้นที่ 6: บันทึกตาราง กราฟ และภาพทำนาย
    print("[6] Save results")
    save_table(results, os.path.join(OUTPUT_DIR, "results.csv"))
    plot_comparison(results, os.path.join(OUTPUT_DIR, "accuracy_comparison.png"))
    evaluate_model(y_test, predictions, classes,
                   os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plot_predictions(X_test, y_test, predictions, confidence, classes,
                     os.path.join(OUTPUT_DIR, "prediction_sample.png"))
    print(f"Selected: {best_result['model']}, {best_result['epochs']} epochs")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
