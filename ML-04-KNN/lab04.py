# LAB 04 KNN
# PART 1: Import Libraries

import os
# ใช้อ่านและจัดการข้อมูล CSV
import pandas as pd

# ใช้แบ่งข้อมูลเป็น Train และ Test
from sklearn.model_selection import train_test_split

# ใช้ Standardize Feature
from sklearn.preprocessing import StandardScaler

# KNN สำหรับ Classification
from sklearn.neighbors import KNeighborsClassifier

# ใช้วัด Accuracy
from sklearn.metrics import accuracy_score



# PART 2: Load Dataset

# อ่าน Dataset จากไฟล์ fruit.csv
df = pd.read_csv(
    "dataset/fruit.csv"
)


# ดูข้อมูล 5 แถวแรก
print("First 5 Rows")
print(df.head())



# PART 3: Explore Dataset

# ดูจำนวนแถวและ Column
print("\nDataset Shape")
print(df.shape)


# ดูชื่อ Column
print("\nColumns")
print(df.columns)


# ตรวจสอบ Missing Value
print("\nMissing Values")
print(df.isnull().sum())


# ตรวจสอบข้อมูลซ้ำ
print("\nDuplicated Rows")
print(df.duplicated().sum())



# PART 4: Preprocess Dataset

# ลบแถวที่มีข้อมูลว่าง
df = df.dropna()


# ลบข้อมูลที่ซ้ำกัน
df = df.drop_duplicates()


print("\nData After Preprocessing")
print(df.shape)



# PART 5: Select Features & Target

# X = Feature ที่ใช้ทำนาย
X = df[
    [
        "weight",
        "sweetness",
        "size"
    ]
]


# y = Class ที่ต้องการทำนาย
y = df["class"]


print("\nFeatures")
print(X.head())


print("\nTarget")
print(y.head())



# PART 6: Train / Test Split

# แบ่งข้อมูลเป็น
# 80% Training
# 20% Testing

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\nTrain / Test")

print(
    "Training Data:",
    X_train.shape
)

print(
    "Testing Data:",
    X_test.shape
)



# PART 7: Standardization

# สร้าง StandardScaler
scaler = StandardScaler()


# เรียนรู้ Scale จาก Train
# และปรับ Training Data
X_train_scaled = scaler.fit_transform(
    X_train
)


# ใช้ Scale เดิมกับ Testing Data
X_test_scaled = scaler.transform(
    X_test
)



# PART 8: Train KNN Models

# ค่า K ที่ใบงานให้ทดลอง
k_values = [3, 5, 7]


# ใช้เก็บ Accuracy
results = {}


for k in k_values:

    # สร้าง KNN
    # K คือจำนวนเพื่อนบ้านที่นำมาพิจารณา
    model = KNeighborsClassifier(
        n_neighbors=k
    )


    # Train Model
    model.fit(
        X_train_scaled,
        y_train
    )


    # ทำนาย Testing Data
    y_pred = model.predict(
        X_test_scaled
    )


    # คำนวณ Accuracy
    accuracy = accuracy_score(
        y_test,
        y_pred
    )


    # เก็บ Accuracy ของแต่ละ K
    results[k] = accuracy


    print(
        f"K = {k} | Accuracy = {accuracy:.4f}"
    )



# PART 9: Find Best K

# หา K ที่มี Accuracy สูงที่สุด
best_k = max(
    results,
    key=results.get
)


# Accuracy ของ K ที่ดีที่สุด
best_accuracy = results[
    best_k
]


print("\nBest Result")

print(
    "Best K =",
    best_k
)

print(
    "Best Accuracy =",
    round(best_accuracy, 4)
)

# สร้าง Folder outputs ถ้ายังไม่มี
os.makedirs(
    "outputs",
    exist_ok=True
)


# นำผล Accuracy ของแต่ละ K มาสร้างเป็น DataFrame
results_df = pd.DataFrame({
    "K": list(results.keys()),
    "Accuracy": list(results.values())
})


# บันทึกผลลงไฟล์ CSV
results_df.to_csv(
    "outputs/results.csv",
    index=False
)


print("\nResults saved to outputs/results.csv")