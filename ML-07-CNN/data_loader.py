"""อ่านภาพจากโฟลเดอร์ BMW และ Honda."""
import hashlib
import os
import cv2
import numpy as np
from preprocessing import preprocess_image


def load_data(data_path, img_size=64):
    images, labels = [], []
    seen = {}  # ใช้ข้ามภาพซ้ำก่อนแบ่ง train/test
    classes = sorted(folder for folder in os.listdir(data_path)
                     if os.path.isdir(os.path.join(data_path, folder)))

    for label, class_name in enumerate(classes):
        class_path = os.path.join(data_path, class_name)
        count = 0
        for filename in sorted(os.listdir(class_path)):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue
            image = cv2.imread(os.path.join(class_path, filename))
            if image is None:
                print(f"Skip unreadable image: {filename}")
                continue
            # เทียบภาพเต็มก่อน resize เพื่อไม่ให้ภาพซ้ำข้ามชุดข้อมูล
            signature = hashlib.sha256(
                str(image.shape).encode() + image.tobytes()
            ).hexdigest()
            if signature in seen:
                if seen[signature] != label:
                    raise ValueError(f"Same image has different labels: {filename}")
                print(f"Skip duplicate: {filename}")
                continue
            seen[signature] = label
            images.append(preprocess_image(image, img_size))
            labels.append(label)
            count += 1
        if count < 5:
            raise ValueError(f"Class {class_name} needs at least 5 images")
        print(f"{class_name}: {count} images")

    if len(classes) < 2:
        raise ValueError("Dataset needs at least two classes")
    return np.array(images), np.array(labels), classes
