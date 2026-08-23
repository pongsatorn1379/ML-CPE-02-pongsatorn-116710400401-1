# LAB 05 : Support Vector Machine
# Image Classification : BMW vs Honda

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# PART 1: กำหนดค่าพื้นฐานของโปรแกรม

# ตำแหน่ง Folder ที่เก็บ Dataset
DATASET_PATH = "dataset"

# Folder สำหรับเก็บผลลัพธ์
OUTPUT_PATH = "outputs"

# กำหนดขนาดรูปภาพให้ทุกภาพมีขนาดเท่ากัน
# 64 x 64 จะได้ทั้งหมด 4096 Feature ต่อรูป
IMG_SIZE = 64

# ใช้ข้อมูล 20% สำหรับ Test
TEST_SIZE = 0.2

# ทำให้การแบ่งข้อมูลได้ผลเหมือนเดิมทุกครั้งที่ Run
RANDOM_STATE = 42

# สร้าง Folder outputs ถ้ายังไม่มี
os.makedirs(OUTPUT_PATH, exist_ok=True)


# PART 2: สร้าง Function สำหรับอ่านรูปภาพจาก Dataset

def load_images(dataset_path, img_size):

    # X ใช้เก็บรูปภาพ
    images = []

    # y ใช้เก็บ Class ของแต่ละรูป
    labels = []

    # เก็บชื่อ Class เช่น BMW และ Honda
    class_names = []

    # อ่านชื่อ Folder ที่อยู่ภายใน dataset
    # sorted() ทำให้เรียงชื่อ Folder ให้เหมือนเดิมทุกครั้ง
    for class_index, class_name in enumerate(
        sorted(os.listdir(dataset_path))
    ):

        # สร้าง Path เช่น dataset/BMW
        class_folder = os.path.join(
            dataset_path,
            class_name
        )

        # ถ้าไม่ใช่ Folder ให้ข้าม
        if not os.path.isdir(class_folder):
            continue

        # เก็บชื่อ Class
        class_names.append(class_name)

        # แสดง Class ที่กำลังอ่าน
        print(
            f"Loading Class: {class_name}"
        )

        # อ่านชื่อไฟล์ทั้งหมดใน Folder ของ Class
        for file_name in os.listdir(class_folder):

            # Path เต็มของรูป
            file_path = os.path.join(
                class_folder,
                file_name
            )

            # อ่านรูปเป็น Grayscale
            # รูปจะเหลือเพียงค่าความสว่าง ไม่ใช้สี RGB
            img = cv2.imread(
                file_path,
                cv2.IMREAD_GRAYSCALE
            )

            # ถ้า OpenCV อ่านรูปไม่ได้ ให้ข้ามรูปนั้น
            if img is None:
                continue

            # ปรับรูปทุกภาพให้มีขนาดเท่ากัน
            img = cv2.resize(
                img,
                (img_size, img_size)
            )

            # เก็บรูปภาพ
            images.append(img)

            # เก็บหมายเลข Class ของรูป
            # ตัวอย่าง BMW = 0 และ Honda = 1
            labels.append(class_index)

    # แปลง List เป็น NumPy Array
    images = np.array(images)
    labels = np.array(labels)

    return images, labels, class_names


# PART 3: โหลด Dataset

X_images, y, class_names = load_images(
    DATASET_PATH,
    IMG_SIZE
)

# แสดงชื่อ Class ที่พบ
print("\nClasses:", class_names)

# แสดงจำนวนรูปทั้งหมด
print("Total Images:", len(X_images))

# แสดงขนาดข้อมูลรูปภาพ
print("Image Shape:", X_images.shape)


# PART 4: แปลงรูปภาพให้เป็น Feature

# SVM ไม่สามารถรับรูปภาพ 2 มิติโดยตรง
# จึงต้องแปลงรูป 64 x 64 ให้เป็นข้อมูล 1 มิติ
#
# เช่น
# 64 x 64
# จะกลายเป็น
# 4096 Feature

X = X_images.reshape(
    len(X_images),
    -1
)

# แสดงขนาด Feature หลังจาก Flatten
print("Feature Shape:", X.shape)


# PART 5: แบ่งข้อมูล Train และ Test

# Train ใช้สำหรับให้ Model เรียนรู้
# Test ใช้สำหรับทดสอบ Model หลังจาก Train เสร็จแล้ว
#
# TEST_SIZE = 0.2
# หมายถึง
# Train 80%
# Test 20%

X_train, X_test, y_train, y_test, X_train_images, X_test_images = train_test_split(
    X,
    y,
    X_images,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

# stratify=y
# ช่วยให้สัดส่วน BMW และ Honda
# ใน Train และ Test ใกล้เคียงกับ Dataset เดิม

print("\nTrain Data:", X_train.shape)
print("Test Data:", X_test.shape)


# PART 6: Standardize ข้อมูล

# SVM มีความไวต่อ Scale ของ Feature
# จึงใช้ StandardScaler ปรับค่าของ Feature
# ให้อยู่ในระดับใกล้เคียงกัน

scaler = StandardScaler()

# fit_transform()
# เรียนรู้ค่า Mean และ Standard Deviation จาก Train
# แล้วนำมาแปลง Train Data
X_train_scaled = scaler.fit_transform(
    X_train
)

# transform()
# ใช้ค่าที่เรียนรู้จาก Train
# มาแปลง Test Data
#
# ไม่ใช้ fit_transform กับ Test
# เพราะไม่ต้องการให้ Model ได้ข้อมูลจาก Test ล่วงหน้า
X_test_scaled = scaler.transform(
    X_test
)


# PART 7: สร้าง SVM ทั้ง 3 Kernel

# LAB กำหนดให้เปรียบเทียบ
# Linear
# Polynomial
# RBF

models = {

    # Linear เหมาะกับข้อมูลที่สามารถแบ่งได้ค่อนข้างเป็นเส้นตรง
    "Linear": SVC(
        kernel="linear"
    ),

    # Polynomial สามารถสร้าง Decision Boundary แบบโค้ง
    "Polynomial": SVC(
        kernel="poly"
    ),

    # RBF เหมาะกับข้อมูลที่มีความสัมพันธ์ซับซ้อน
    # และไม่สามารถแบ่งด้วยเส้นตรงได้ง่าย
    "RBF": SVC(
        kernel="rbf"
    )
}


# Dictionary สำหรับเก็บ Accuracy ของแต่ละ Model
accuracy_results = {}

# Dictionary สำหรับเก็บ Prediction ของแต่ละ Model
prediction_results = {}


# PART 8: Function สำหรับสร้างรูปแสดงผล Prediction

def save_prediction_image(
    images,
    y_true,
    y_pred,
    class_names,
    model_name,
    save_path,
    num_images=4
):

    # ป้องกันกรณี Test Data มีน้อยกว่า 4 รูป
    num_images = min(
        num_images,
        len(images)
    )

    # นับจำนวนรูปที่ Model ทำนายถูก
    correct_count = np.sum(
        y_true[:num_images] == y_pred[:num_images]
    )

    # สร้างพื้นที่แสดงรูปแบบ 2 x 2
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(8, 8)
    )

    # ทำให้ axes กลายเป็น Array 1 มิติ
    # เพื่อเรียก axes[0], axes[1] ได้ง่าย
    axes = axes.ravel()

    # แสดงชื่อ Model และจำนวนรูปที่ทายถูก
    fig.suptitle(
        f"{model_name} SVM Prediction: "
        f"{correct_count}/{num_images} correct",
        fontsize=16
    )

    # วนแสดงรูปแต่ละรูป
    for i in range(num_images):

        # แสดงรูปภาพแบบ Grayscale
        axes[i].imshow(
            images[i],
            cmap="gray"
        )

        # ไม่แสดงแกน X และ Y
        axes[i].axis(
            "off"
        )

        # Class ที่ Model ทำนาย
        predicted_class = class_names[
            y_pred[i]
        ]

        # Class จริงของรูป
        true_class = class_names[
            y_true[i]
        ]

        # ถ้าทำนายถูกให้ข้อความเป็นสีเขียว
        # ถ้าทำนายผิดให้เป็นสีแดง
        if y_pred[i] == y_true[i]:
            text_color = "green"
        else:
            text_color = "red"

        # แสดง Prediction และค่าจริงบนรูป
        axes[i].set_title(
            f"Pred: {predicted_class}\n"
            f"True: {true_class}",
            color=text_color,
            fontsize=12
        )

    # จัดตำแหน่งองค์ประกอบของรูป
    plt.tight_layout(
        rect=[0, 0, 1, 0.95]
    )

    # บันทึกเป็นไฟล์ PNG
    plt.savefig(
        save_path,
        dpi=300
    )

    # ปิด Figure หลังจากบันทึก
    plt.close()


# PART 9: Train และ Predict ด้วย SVM

# วนทำทีละ Model
# Linear -> Polynomial -> RBF

for model_name, model in models.items():

    # Train Model ด้วย Train Data
    model.fit(
        X_train_scaled,
        y_train
    )

    # ให้ Model ทำนาย Test Data
    y_pred = model.predict(
        X_test_scaled
    )

    # คำนวณ Accuracy
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    # เก็บ Accuracy ไว้ใช้เปรียบเทียบภายหลัง
    accuracy_results[
        model_name
    ] = accuracy

    # เก็บผล Prediction
    prediction_results[
        model_name
    ] = y_pred

    # แสดง Accuracy
    print(
        f"{model_name} Accuracy: "
        f"{accuracy:.4f}"
    )

    # สร้างชื่อไฟล์ของรูป Prediction
    image_file = os.path.join(
        OUTPUT_PATH,
        f"prediction_{model_name.lower()}.png"
    )

    # สร้างรูปเปรียบเทียบ Prediction
    save_prediction_image(
        X_test_images,
        y_test,
        y_pred,
        class_names,
        model_name,
        image_file,
        num_images=4
    )


# PART 10: บันทึก Accuracy เป็น CSV

# เปลี่ยน Dictionary ให้เป็น DataFrame
accuracy_df = pd.DataFrame({
    "Kernel": list(
        accuracy_results.keys()
    ),

    "Accuracy": list(
        accuracy_results.values()
    )
})

# บันทึกเป็น CSV
accuracy_df.to_csv(
    os.path.join(
        OUTPUT_PATH,
        "accuracy_scores.csv"
    ),
    index=False
)

# แสดง Accuracy ทั้งหมด
print("\nAccuracy Results")
print(accuracy_df)


# PART 11: สร้างกราฟเปรียบเทียบ Accuracy

plt.figure(
    figsize=(8, 5)
)

# สร้าง Bar Graph
plt.bar(
    accuracy_results.keys(),
    accuracy_results.values()
)

# ชื่อแกน X
plt.xlabel(
    "SVM Kernel"
)

# ชื่อแกน Y
plt.ylabel(
    "Accuracy"
)

# ชื่อกราฟ
plt.title(
    "SVM Kernel Accuracy Comparison"
)

# กำหนดช่วง Accuracy ตั้งแต่ 0 ถึง 1
plt.ylim(
    0,
    1.1
)

# แสดงค่า Accuracy ด้านบนของแต่ละแท่ง
for index, accuracy in enumerate(
    accuracy_results.values()
):

    plt.text(
        index,
        accuracy + 0.02,
        f"{accuracy:.2f}",
        ha="center"
    )

# จัดตำแหน่งกราฟ
plt.tight_layout()

# บันทึกกราฟ
plt.savefig(
    os.path.join(
        OUTPUT_PATH,
        "accuracy_comparison.png"
    ),
    dpi=300
)

# ปิดกราฟ
plt.close()


# PART 12: บันทึก Prediction ลง CSV

# สร้าง DataFrame สำหรับเก็บผลจริง
# และผลทำนายของแต่ละ Kernel

prediction_df = pd.DataFrame({
    "True": y_test
})

# เพิ่ม Prediction ของแต่ละ Model
for model_name in models.keys():

    prediction_df[
        model_name
    ] = prediction_results[
        model_name
    ]


# เปลี่ยนเลข Class เป็นชื่อ BMW / Honda

prediction_df["True Name"] = [
    class_names[value]
    for value in prediction_df["True"]
]

prediction_df["Linear Name"] = [
    class_names[value]
    for value in prediction_df["Linear"]
]

prediction_df["Polynomial Name"] = [
    class_names[value]
    for value in prediction_df["Polynomial"]
]

prediction_df["RBF Name"] = [
    class_names[value]
    for value in prediction_df["RBF"]
]


# บันทึกผล Prediction เป็น CSV
prediction_df.to_csv(
    os.path.join(
        OUTPUT_PATH,
        "predictions.csv"
    ),
    index=False
)


# PART 13: แสดง Prediction ตัวอย่าง

print("\nPrediction Example")

print(
    prediction_df[
        [
            "True Name",
            "Linear Name",
            "Polynomial Name",
            "RBF Name"
        ]
    ].head(10)
)


# แสดงข้อความเมื่อโปรแกรมทำงานเสร็จ
print("\nFinished")
print("Results saved in outputs folder.")