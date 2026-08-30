# PART 1: Import Libraries

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import set_random_seed


# PART 2: กำหนดค่าพื้นฐาน

DATASET_PATH = "dataset"
OUTPUT_PATH = "outputs"

IMG_SIZE = 64

class_names = ["BMW", "Honda"]

os.makedirs(OUTPUT_PATH, exist_ok=True)

set_random_seed(42)


# PART 3: อ่าน Dataset รูปภาพ

images = []
labels = []

for label, class_name in enumerate(class_names):

    folder_path = os.path.join(DATASET_PATH, class_name)

    for file_name in os.listdir(folder_path):

        file_path = os.path.join(folder_path, file_name)

        img = cv2.imread(file_path)

        # ข้ามไฟล์ที่อ่านไม่ได้
        if img is None:
            continue

        # OpenCV อ่านรูปเป็น BGR
        # แปลงเป็น RGB สำหรับแสดงผลให้สีถูกต้อง
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ทำให้รูปทุกภาพมีขนาดเท่ากัน
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        images.append(img)
        labels.append(label)


# แปลง List เป็น NumPy Array
X_images = np.array(images)
y = np.array(labels)


print("จำนวนรูปทั้งหมด:", len(X_images))
print("Shape ของรูป:", X_images.shape)

for i, class_name in enumerate(class_names):
    print(class_name, ":", np.sum(y == i), "รูป")


# PART 4: แปลงรูปภาพเป็น Feature

# รูป 64 x 64 x 3
# จะถูก Flatten เป็นข้อมูล 1 มิติ
#
# 64 x 64 x 3 = 12,288 Features

X = X_images.reshape(len(X_images), -1)

print("Shape หลัง Flatten:", X.shape)


# PART 5: แบ่ง Training และ Testing Dataset

# แยก index เพื่อให้สามารถนำรูปต้นฉบับ
# กลับมาใช้แสดง Prediction ได้

indices = np.arange(len(X))

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train = X[train_idx]
X_test = X[test_idx]

y_train = y[train_idx]
y_test = y[test_idx]

X_test_images = X_images[test_idx]

print("Training:", len(X_train))
print("Testing:", len(X_test))


# PART 6: Standardize Input Features

# StandardScaler จะเรียนรู้ค่า mean และ standard deviation
# จาก Training Dataset เท่านั้น

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

# Testing Dataset ต้องใช้ scaler ตัวเดิม
X_test = scaler.transform(X_test)


# PART 7: Function สำหรับสร้าง Neural Network

def create_model(hidden_layers):

    model = Sequential()

    # Hidden Layer แรก
    model.add(
        Dense(
            hidden_layers[0],
            activation="relu",
            input_shape=(X_train.shape[1],)
        )
    )

    # Hidden Layer ที่เหลือ
    for neurons in hidden_layers[1:]:

        model.add(
            Dense(
                neurons,
                activation="relu"
            )
        )

    # Output Layer
    #
    # มี 2 Classes
    # BMW และ Honda
    #
    # ใช้ Softmax สำหรับ Classification

    model.add(
        Dense(
            len(class_names),
            activation="softmax"
        )
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# PART 8: กำหนด Neural Network Configurations

# เปรียบเทียบจำนวน Hidden Layers
# และจำนวน Neurons ตามโจทย์ LAB

model_configs = {

    "Model 1": [64],

    "Model 2": [128, 64],

    "Model 3": [256, 128, 64]
}


# เปรียบเทียบจำนวน Epochs

epoch_list = [
    10,
    30,
    50
]


# เก็บผลการทดลองทั้งหมด
results = []

best_model = None
best_history = None
best_accuracy = 0
best_name = ""
best_epoch = 0


# PART 9: Train Neural Network

for model_name, hidden_layers in model_configs.items():

    for epochs in epoch_list:

        print()
        print("Training:", model_name)
        print("Hidden Layers:", hidden_layers)
        print("Epochs:", epochs)

        # สร้าง Model ใหม่ทุกครั้ง
        # เพื่อให้การเปรียบเทียบ Epoch ยุติธรรม

        model = create_model(hidden_layers)

        history = model.fit(
            X_train,
            y_train,

            epochs=epochs,

            batch_size=16,

            # แบ่ง Training 20%
            # ไปใช้เป็น Validation Dataset
            validation_split=0.20,

            verbose=1
        )


        # PART 10: Evaluate Model

        test_loss, test_accuracy = model.evaluate(
            X_test,
            y_test,
            verbose=0
        )

        print("Test Accuracy:", test_accuracy)


        # บันทึกผลแต่ละการทดลอง

        results.append({

            "Model": model_name,

            "Hidden Layers": str(hidden_layers),

            "Epochs": epochs,

            "Test Accuracy": test_accuracy,

            "Test Loss": test_loss,

            "Final Train Accuracy":
                history.history["accuracy"][-1],

            "Final Validation Accuracy":
                history.history["val_accuracy"][-1]
        })


        # เก็บ Model ที่ Accuracy สูงที่สุด

        if test_accuracy > best_accuracy:

            best_accuracy = test_accuracy

            best_model = model

            best_history = history

            best_name = model_name

            best_epoch = epochs


# PART 11: บันทึก Results เป็น CSV

results_df = pd.DataFrame(results)

results_df.to_csv(
    os.path.join(
        OUTPUT_PATH,
        "results.csv"
    ),
    index=False
)

print()
print(results_df)

print()
print("Best Model:", best_name)
print("Best Epoch:", best_epoch)
print("Best Accuracy:", best_accuracy)


# PART 12: Training History

# แสดง Training Accuracy
# และ Validation Accuracy

plt.figure(figsize=(10, 5))

plt.plot(
    best_history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    best_history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    f"Training and Validation Accuracy\n"
    f"{best_name} - {best_epoch} Epochs"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_PATH,
        "training_history.png"
    )
)

plt.close()


# PART 13: Confusion Matrix

prediction_probability = best_model.predict(
    X_test,
    verbose=0
)

y_pred = np.argmax(
    prediction_probability,
    axis=1
)

cm = confusion_matrix(
    y_test,
    y_pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot()

plt.title(
    f"Confusion Matrix\n"
    f"Accuracy: {best_accuracy * 100:.2f}%"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_PATH,
        "confusion_matrix.png"
    )
)

plt.close()


# PART 14: Prediction Sample

# สุ่มรูปจาก Testing Dataset จำนวน 4 รูป

num_samples = min(
    4,
    len(X_test_images)
)

random_indices = np.random.choice(
    len(X_test_images),
    num_samples,
    replace=False
)

correct = 0

for idx in random_indices:

    if y_pred[idx] == y_test[idx]:
        correct += 1


plt.figure(figsize=(9, 9))

plt.suptitle(
    f"Prediction: {correct}/{num_samples} correct",
    fontsize=16
)


for i, idx in enumerate(random_indices):

    plt.subplot(2, 2, i + 1)

    plt.imshow(
        X_test_images[idx]
    )

    predicted_class = y_pred[idx]

    true_class = y_test[idx]

    confidence = (
        prediction_probability[idx][predicted_class]
        * 100
    )

    predicted_name = class_names[predicted_class]

    true_name = class_names[true_class]


    # ถ้าทายถูกให้ใช้สีเขียว
    # ถ้าทายผิดให้ใช้สีแดง

    if predicted_class == true_class:
        text_color = "green"

    else:
        text_color = "red"


    plt.title(
        f"Pred: {predicted_name} "
        f"({confidence:.0f}%)\n"
        f"True: {true_name}",
        color=text_color
    )

    plt.axis("off")


plt.tight_layout(
    rect=[0, 0, 1, 0.95]
)

plt.savefig(
    os.path.join(
        OUTPUT_PATH,
        "prediction_sample.png"
    )
)

plt.close()


# PART 15: แสดงผลลัพธ์สุดท้าย

print()
print("Finished")
print()

print(
    "Best Model:",
    best_name
)

print(
    "Best Epoch:",
    best_epoch
)

print(
    f"Best Accuracy: "
    f"{best_accuracy * 100:.2f}%"
)

print()

print(
    "Output files:"
)

print(
    "outputs/training_history.png"
)

print(
    "outputs/confusion_matrix.png"
)

print(
    "outputs/prediction_sample.png"
)

print(
    "outputs/results.csv"
)