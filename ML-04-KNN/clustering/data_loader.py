from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Path ชี้ไปยังไฟล์ cars.csv
CSV_PATH = Path(__file__).resolve().parent.parent / "cars" / "cars.csv"

# 2. เปลี่ยน Features เป็นคอลัมน์ตัวเลขของข้อมูลรถยนต์
FEATURES = [
    "economy (mpg)",
    "cylinders",
    "displacement (cc)",
    "power (hp)",
    "weight (lb)",
    "0-60 mph (s)",
    "year"
]

# ---------------------------------------------------------------------------
def load_data():
    """
    คืนค่าเป็น dict ที่มี X, X_raw, df, features
    """
    df = pd.read_csv(CSV_PATH)
    df = df.dropna()  # ลบแถวที่มีค่าว่าง (เช่น ใน power หรือ economy) ออก

    X_raw = df[FEATURES].to_numpy(dtype="float32")  
    X = StandardScaler().fit_transform(X_raw).astype("float32") 

    return {"X": X, "X_raw": X_raw, "df": df, "features": FEATURES}

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    data = load_data()
    print("size data :", data["X"].shape)
    print("mean after scale (should be close to 0) :", data["X"].mean(axis=0).round(3))