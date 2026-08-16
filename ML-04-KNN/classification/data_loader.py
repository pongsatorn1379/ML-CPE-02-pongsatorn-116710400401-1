from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Path ชี้ไปยังไฟล์ cars.csv
CSV_PATH = Path(__file__).resolve().parent.parent / "cars" / "cars.csv"

# 2. ตั้งค่า TARGET (ใช้ 'cylinders' เป็นตัวจำแนกประเภท)
TARGET = "cylinders"

# 3. ชื่อคอลัมน์ตัวเลขที่มีอยู่จริงใน cars.csv
NUMERIC_FEATURES = [
    "economy (mpg)",
    "displacement (cc)",
    "power (hp)",
    "weight (lb)",
    "0-60 mph (s)",
    "year",
]

# 4. คอลัมน์ข้อความ (ปล่อยเป็น dict ว่าง หากไม่ต้องแปลง)
TEXT_FEATURES = {}


def load_data(test_size=0.2, seed=42):
    df = pd.read_csv(CSV_PATH)

    # ทำความสะอาดคอลัมน์ข้อความ (ถ้ามี)
    for col, mapping in TEXT_FEATURES.items():
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().map(mapping)

    # แมปปิ้ง Target คลาส
    df[TARGET] = df[TARGET].astype(str).str.strip()
    class_names = sorted(df[TARGET].dropna().unique())
    target_mapping = {name: i for i, name in enumerate(class_names)}
    df[TARGET] = df[TARGET].map(target_mapping)

    # ดึงเฉพาะคอลัมน์ที่ใช้งานและลบค่าว่าง (NaN)
    used_columns = NUMERIC_FEATURES + list(TEXT_FEATURES.keys()) + [TARGET]
    df_clean = df[used_columns].dropna()

    X = df_clean[NUMERIC_FEATURES + list(TEXT_FEATURES.keys())].to_numpy(
        dtype="float32"
    )
    y = df_clean[TARGET].to_numpy(dtype="int32")

    # Split data (Train 60% / Validation 20% / Test 20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=seed, stratify=y_temp
    )

    # Scaling
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
        "class_names": class_names,
        "feature_names": NUMERIC_FEATURES + list(TEXT_FEATURES.keys()),
        "n_rows": len(df_clean),
    }


if __name__ == "__main__":
    data = load_data()
    print("train :", data["X_train"].shape)
    print("val   :", data["X_val"].shape)
    print("test  :", data["X_test"].shape)
    print("คลาส   :", data["class_names"])