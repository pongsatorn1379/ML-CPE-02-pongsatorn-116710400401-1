"""ปรับขนาด สี และค่าพิกเซลก่อนฝึก."""
import cv2
import numpy as np


def preprocess_image(image, img_size=64):
    if image is None:
        return None
    # OpenCV อ่านสีแบบ BGR จึงเปลี่ยนเป็น RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.resize(image, (img_size, img_size))


def to_features(images):
    # ปรับพิกเซลจาก 0–255 เป็น 0–1
    return np.asarray(images, dtype=np.float32) / 255.0
