นายพงศธร รอดดี 116710400401-1 Sec2 ที่มา Dataset >> https://www.kaggle.com/datasets/ahmedelsany/car-brand-classification-dataset

#ผมดูของอจ.ไม่เข้าใจเลยกลับมาทำเอง 

# LAB 05: Support Vector Machine (SVM)

โปรเจกต์นี้เป็นส่วนหนึ่งของ LAB 05 เรื่อง Support Vector Machine (SVM) โดยใช้รูปภาพรถ BMW และ Honda เป็น Dataset สำหรับทำ Image Classification และเปรียบเทียบประสิทธิภาพของ SVM ทั้ง 3 Kernel ได้แก่ Linear, Polynomial และ RBF


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
3. แปลงรูปภาพเป็น Grayscale
4. Resize รูปภาพให้มีขนาดเท่ากัน เช่น 64 x 64 Pixel
5. แปลงรูปภาพ 2 มิติให้เป็น Feature แบบ 1 มิติ
6. แบ่งข้อมูลออกเป็น Train และ Test
7. ใช้ `StandardScaler` ปรับ Scale ของ Feature
8. Train SVM ทั้ง 3 Kernel
   - Linear
   - Polynomial
   - RBF
9. ให้โมเดลทำนายข้อมูล Test
10. คำนวณ Accuracy ของแต่ละ Kernel
11. บันทึกผล Prediction และกราฟลงใน Folder `outputs`

## การแปลงรูปภาพเป็น Feature

SVM ไม่สามารถรับรูปภาพโดยตรงได้ จึงต้องแปลงรูปภาพให้เป็นข้อมูลตัวเลขก่อน

ตัวอย่าง ถ้ากำหนดขนาดรูปเป็น 64 x 64 Pixel:

```text
64 x 64 = 4096 Features
```

รูปภาพแต่ละรูปจะถูก Flatten จาก Matrix 2 มิติ ให้กลายเป็นข้อมูล 1 มิติที่มี 4096 ค่า เพื่อนำไปใช้ Train SVM

## Train และ Test Data

Dataset ถูกแบ่งเป็น

```text
Train = 80%
Test  = 20%
```

- Train Data ใช้สำหรับให้โมเดลเรียนรู้
- Test Data ใช้สำหรับตรวจสอบความสามารถของโมเดลกับข้อมูลที่ไม่เคยเห็นมาก่อน

มีการใช้ `stratify=y` เพื่อช่วยรักษาสัดส่วนของ BMW และ Honda ใน Train และ Test ให้ใกล้เคียงกับ Dataset เดิม

## StandardScaler

ก่อนนำข้อมูลเข้า SVM จะใช้ `StandardScaler` เพื่อปรับ Feature ให้อยู่ใน Scale ที่เหมาะสม

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

`fit_transform()` ใช้กับ Train Data เพื่อเรียนรู้ค่า Mean และ Standard Deviation

`transform()` ใช้กับ Test Data โดยใช้ค่าที่เรียนรู้จาก Train Data เพื่อป้องกันไม่ให้ข้อมูล Test รั่วไหลเข้าสู่ขั้นตอน Training

## SVM Kernel ที่ใช้

### Linear

```python
SVC(kernel="linear")
```

เหมาะกับข้อมูลที่สามารถแบ่ง Class ได้ค่อนข้างเป็นเส้นตรง

### Polynomial

```python
SVC(kernel="poly")
```

เหมาะกับข้อมูลที่ต้องใช้เส้นแบ่งแบบ Polynomial หรือเส้นโค้ง

### RBF

```python
SVC(kernel="rbf")
```

เหมาะกับข้อมูลที่มีรูปแบบซับซ้อนและไม่สามารถแบ่งด้วยเส้นตรงได้ง่าย

## การประเมินผล

ใช้ Accuracy เพื่อวัดจำนวนข้อมูลที่โมเดลทำนายถูกต้อง

```python
accuracy = accuracy_score(y_test, y_pred)
```

ตัวอย่างผลลัพธ์:

```text
Linear Accuracy: 0.8500
Polynomial Accuracy: 0.8000
RBF Accuracy: 0.9000
```

ค่าที่ได้จริงจะขึ้นอยู่กับ Dataset และรูปภาพที่นำมาใช้

## การแสดงผล Prediction

โปรแกรมจะสุ่ม/เลือกตัวอย่างจาก Test Data มาแสดงเป็นรูปภาพ พร้อมข้อความ

```text
Pred: BMW
True: BMW
```

ความหมายคือ

- `Pred` คือ Class ที่ Model ทำนาย
- `True` คือ Class จริงของรูป

ถ้าทำนายถูก ข้อความจะแสดงเป็นสีเขียว

ถ้าทำนายผิด ข้อความจะแสดงเป็นสีแดง

ด้านบนของรูปจะแสดงจำนวนที่ทำนายถูก เช่น

```text
Prediction: 3/4 correct
```

## ผลลัพธ์ที่ได้

เมื่อรันโปรแกรมแล้ว จะมีการสร้าง Folder `outputs` และบันทึกไฟล์ เช่น

```text
outputs/
├── accuracy_scores.csv
├── predictions.csv
├── accuracy_comparison.png
├── prediction_linear.png
├── prediction_polynomial.png
└── prediction_rbf.png
```

### accuracy_scores.csv

เก็บค่า Accuracy ของแต่ละ Kernel

### predictions.csv

เก็บค่าจริงและค่าที่แต่ละ Model ทำนาย

### accuracy_comparison.png

กราฟเปรียบเทียบ Accuracy ของ Linear, Polynomial และ RBF

### prediction_linear.png

แสดงตัวอย่าง Prediction ของ Linear Kernel

### prediction_polynomial.png

แสดงตัวอย่าง Prediction ของ Polynomial Kernel

### prediction_rbf.png

แสดงตัวอย่าง Prediction ของ RBF Kernel



## สรุป

LAB นี้นำ Support Vector Machine มาใช้จำแนกรูปภาพรถ BMW และ Honda โดยรูปภาพจะถูกปรับขนาดและแปลงเป็น Feature ก่อนนำเข้าโมเดล จากนั้นข้อมูลจะถูก Standardize และแบ่งเป็น Train/Test ก่อน Train SVM จำนวน 3 Kernel ได้แก่ Linear, Polynomial และ RBF สุดท้ายเปรียบเทียบผลด้วย Accuracy และแสดงผล Prediction เป็นรูปภาพ เพื่อให้เห็นทั้งค่าที่โมเดลทำนายและค่าจริงของข้อมูล
