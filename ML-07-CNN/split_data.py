"""แบ่งข้อมูลเป็น train, validation และ test."""
from sklearn.model_selection import train_test_split


def split_dataset(X, y, test_size=0.2, val_size=0.2):
    # แบ่ง test ออกก่อน; stratify รักษาสัดส่วนของแต่ละคลาส
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    # val_size เป็นสัดส่วนของข้อมูลทั้งหมด
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_ratio,
        random_state=42, stratify=y_train
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
