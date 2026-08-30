นายพงศธร รอดดี 116710400401-1 Sec2 ที่มา Dataset >> https://www.kaggle.com/datasets/ahmedelsany/car-brand-classification-dataset

อ้างอิงจากอาจารย์และนำมาปรับใช้
# LAB 06: Neural Network (NN)
## Dataset

Dataset เป็นรูปภาพรถ 2 Class ได้แก่

- BMW
- Honda

ชื่อ Folder จะถูกใช้เป็นชื่อ Class ของข้อมูลโดยอัตโนมัติ

ตัวอย่าง:

```text
dataset/BMW/   -> Class BMW
dataset/Honda/ -> Class Honda
```

ควรมีจำนวนรูปของแต่ละ Class ใกล้เคียงกัน เพื่อไม่ให้โมเดลเอนเอียงไปยัง Class ใด Class หนึ่งมากเกินไป

## หลักการทำงาน

โปรแกรมมีขั้นตอนหลักดังนี้

1. อ่านรูปภาพจาก Folder `dataset`
2. อ่านชื่อ Folder เพื่อใช้เป็นชื่อ Class
3. Resize รูปภาพให้มีขนาดเท่ากัน เช่น 64 x 64 Pixel
4. แปลงรูปภาพเป็น RGB
5. แปลงรูปภาพให้เป็น Feature แบบ 1 มิติด้วยการ Flatten
6. แบ่งข้อมูลออกเป็น Train และ Test
7. ใช้ `StandardScaler` ปรับ Scale ของ Feature
8. สร้าง Neural Network หลาย Configuration
9. Train Neural Network ด้วยจำนวน Epochs ที่แตกต่างกัน
10. คำนวณ Accuracy ของแต่ละ Model
11. เลือก Model ที่มี Accuracy สูงที่สุด
12. แสดง Training / Validation Accuracy และ Loss
13. สร้าง Confusion Matrix
14. แสดงตัวอย่าง Prediction ของข้อมูล Test
15. บันทึกผลลัพธ์ลงใน Folder `outputs`

## การแปลงรูปภาพเป็น Feature

Neural Network แบบ Dense ที่ใช้ใน LAB นี้จะรับข้อมูลเป็น Feature แบบ 1 มิติ จึงต้อง Resize และ Flatten รูปภาพก่อนนำเข้า Model

ตัวอย่าง ถ้ากำหนดขนาดรูปเป็น 64 x 64 Pixel และใช้ภาพ RGB:

```text
64 x 64 x 3 = 12,288 Features
```

รูปภาพแต่ละรูปจะถูก Flatten จาก Matrix 3 มิติให้กลายเป็นข้อมูล 1 มิติที่มี 12,288 ค่า เพื่อนำไปใช้เป็น Input ของ Neural Network

## Train และ Test Data

Dataset ถูกแบ่งเป็น

```text
Train = 80%
Test  = 20%
```

- Train Data ใช้สำหรับให้ Neural Network เรียนรู้
- Test Data ใช้สำหรับตรวจสอบความสามารถของ Model กับข้อมูลที่ไม่เคยเห็นมาก่อน

มีการใช้ `stratify=y` เพื่อช่วยรักษาสัดส่วนของ BMW และ Honda ใน Train และ Test ให้ใกล้เคียงกับ Dataset เดิม

## StandardScaler

ก่อนนำข้อมูลเข้า Neural Network จะใช้ `StandardScaler` เพื่อปรับ Feature ให้อยู่ใน Scale ที่เหมาะสม

```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

`fit_transform()` ใช้กับ Train Data เพื่อเรียนรู้ค่า Mean และ Standard Deviation

`transform()` ใช้กับ Test Data โดยใช้ค่าที่เรียนรู้จาก Train Data เพื่อป้องกันไม่ให้ข้อมูล Test รั่วไหลเข้าสู่ขั้นตอน Training

## Neural Network ที่ใช้

LAB นี้เปรียบเทียบ Neural Network หลาย Configuration โดยเปลี่ยนจำนวน Hidden Layers และ Neurons

### Model 1

```text
Input
  ↓
64 Neurons
  ↓
Output
```

### Model 2

```text
Input
  ↓
128 Neurons
  ↓
64 Neurons
  ↓
Output
```

### Model 3

```text
Input
  ↓
256 Neurons
  ↓
128 Neurons
  ↓
64 Neurons
  ↓
Output
```

Hidden Layer ใช้ Activation Function แบบ `ReLU`

Output Layer ใช้ `Softmax` สำหรับจำแนก Class BMW และ Honda

## Epochs ที่ใช้

แต่ละ Neural Network Configuration จะถูก Train ด้วยจำนวน Epochs ที่แตกต่างกัน เช่น

```text
10 Epochs
30 Epochs
50 Epochs
```

เพื่อเปรียบเทียบว่าจำนวน Epochs ส่งผลต่อ Accuracy ของ Model อย่างไร

## การประเมินผล

ใช้ Accuracy เพื่อวัดจำนวนข้อมูลที่ Model ทำนายถูกต้อง

```python
test_loss, test_accuracy = model.evaluate(X_test, y_test)
```

ผลของแต่ละ Model และจำนวน Epochs จะถูกเก็บไว้ใน `results.csv`

ตัวอย่าง:

```text
Model 1 | 10 Epochs | Accuracy 0.8200
Model 1 | 30 Epochs | Accuracy 0.8500
Model 2 | 30 Epochs | Accuracy 0.9000
Model 3 | 50 Epochs | Accuracy 0.8800
```

ค่าที่ได้จริงจะขึ้นอยู่กับ Dataset และรูปภาพที่นำมาใช้

## Training History

โปรแกรมจะแสดงผลการ Train ของ Model ที่มี Accuracy สูงที่สุด โดยเปรียบเทียบ

- Training Accuracy
- Validation Accuracy
- Training Loss
- Validation Loss

ผลลัพธ์จะถูกบันทึกเป็น

```text
outputs/training_history.png
```

## Confusion Matrix

Confusion Matrix ใช้แสดงจำนวนข้อมูลที่ Model ทำนาย BMW และ Honda ได้ถูกหรือผิด

ผลลัพธ์จะถูกบันทึกเป็น

```text
outputs/confusion_matrix.png
```

## การแสดงผล Prediction

โปรแกรมจะสุ่มตัวอย่างจาก Test Data มาแสดงเป็นรูปภาพ พร้อมข้อความ

```text
Pred: BMW (92%)
True: BMW
```

ความหมายคือ

- `Pred` คือ Class ที่ Model ทำนาย
- เปอร์เซ็นต์คือค่าความมั่นใจของ Model
- `True` คือ Class จริงของรูป

ถ้าทำนายถูก ข้อความจะแสดงเป็นสีเขียว

ถ้าทำนายผิด ข้อความจะแสดงเป็นสีแดง

ด้านบนของรูปจะแสดงจำนวนที่ทำนายถูก เช่น

```text
Prediction: 4/4 correct
```

## ผลลัพธ์ที่ได้

เมื่อรันโปรแกรมแล้ว จะมีการสร้าง Folder `outputs` และบันทึกไฟล์ เช่น

```text
outputs/
├── confusion_matrix.png
├── prediction_sample.png
├── training_history.png
└── results.csv
```

### confusion_matrix.png

แสดงผล Confusion Matrix ของ Model ที่มี Accuracy สูงที่สุด

### prediction_sample.png

แสดงตัวอย่างรูปภาพจาก Test Data พร้อมค่า Prediction, True Class และ Confidence

### training_history.png

แสดง Training / Validation Accuracy และ Loss ของ Model ที่เลือก

### results.csv

เก็บผลการเปรียบเทียบ Neural Network แต่ละ Configuration และจำนวน Epochs ที่แตกต่างกัน

## สรุป

LAB นี้นำ Neural Network มาใช้จำแนกรูปภาพรถ BMW และ Honda โดยรูปภาพจะถูก Resize และ Flatten ให้เป็น Feature ก่อนแบ่งข้อมูลเป็น Train/Test และปรับ Scale ด้วย StandardScaler จากนั้นสร้าง Neural Network หลาย Configuration และ Train ด้วยจำนวน Epochs ที่แตกต่างกัน แล้วเปรียบเทียบผลด้วย Accuracy สุดท้ายแสดง Training History, Confusion Matrix และตัวอย่าง Prediction เป็นรูปภาพ เพื่อให้เห็นผลการทำงานของ Model อย่างชัดเจน
