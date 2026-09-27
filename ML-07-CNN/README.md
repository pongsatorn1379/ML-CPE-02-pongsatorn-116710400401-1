นายพงศธร รอดดี 116710400401-1 Sec2

ที่มา Dataset: https://www.kaggle.com/datasets/ahmedelsany/car-brand-classification-dataset

อ้างอิงจากอาจารย์และนำมาปรับใช้

# LAB 07: Convolutional Neural Network (CNN)

## Dataset

ใช้รูปภาพรถ 2 Class ได้แก่ BMW และ Honda โดยใช้ชื่อ Folder เป็นชื่อ Class

```text
dataset/
├── BMW/
└── Honda/
```

โปรแกรมข้ามภาพซ้ำก่อนแบ่งข้อมูล เพื่อไม่ให้ภาพเดียวกันอยู่ทั้ง Train และ Test โดยไม่ลบภาพต้นฉบับ

## หลักการทำงาน

1. อ่านรูปภาพจาก Folder `dataset`
2. Resize ภาพเป็น 64 × 64 Pixel และแปลงเป็น RGB
3. ปรับค่าพิกเซลจาก 0–255 เป็น 0–1
4. แบ่งข้อมูลเป็น Train, Validation และ Test
5. สร้าง CNN 2 Configuration
6. Train แต่ละแบบด้วย 10, 20 และ 30 Epochs
7. เปรียบเทียบ Accuracy ของแต่ละการทดลอง
8. เลือก Model จาก Validation Accuracy สูงสุด ถ้าเท่ากันใช้ Validation Loss ต่ำสุด
9. แสดง Training History, Confusion Matrix และตัวอย่าง Prediction
10. บันทึกผลลงใน Folder `outputs`

## การเตรียมภาพ

CNN รับภาพขนาด 64 × 64 × 3 โดย 3 คือช่องสี RGB จึงเก็บภาพเป็นข้อมูล 3 มิติก่อนเข้า Convolution Layer

ปรับค่าพิกเซลด้วยการหาร 255:

```python
X = images.astype("float32") / 255.0
```

จากนั้นใช้ Flatten ภายใน Model หลัง Convolution และ Pooling เพื่อเตรียม Features ให้ Dense Layer

## Train, Validation และ Test

แบ่งข้อมูลโดยประมาณ:

```text
Train      = 60%
Validation = 20%
Test       = 20%
```

- Train ใช้ให้ Model เรียนรู้
- Validation ใช้เปรียบเทียบและเลือก Model
- Test ใช้ประเมินกับข้อมูลที่ไม่ได้ใช้ฝึก

ใช้ `stratify=y` รักษาสัดส่วน BMW/Honda และ `random_state=42` เพื่อให้แบ่งได้ชุดเดิม จำนวนภาพจริงอาจไม่ตรงเปอร์เซ็นต์พอดีเพราะปัดเป็นจำนวนเต็ม

## CNN ที่ใช้

| Configuration | Convolution Layers | Filters | Dense Neurons |
|---|---:|---|---:|
| Model 1 | 1 | 16 | 32 |
| Model 2 | 2 | 16, 32 | 64 |

ลำดับการทำงาน:

```text
Input Image
  ↓
Convolution + ReLU
  ↓
Max Pooling
  ↓
Flatten
  ↓
Dense + ReLU
  ↓
Output: BMW / Honda
```

Model 2 ใช้ Convolution + ReLU และ Max Pooling สองชุด ก่อนเข้า Flatten

Output ใช้ Softmax สำหรับจำแนก BMW และ Honda ใช้ Adam ในการปรับน้ำหนัก

## Epochs ที่ใช้

แต่ละ Configuration ทดลอง 10, 20 และ 30 Epochs รวม 6 การทดลอง โดยเริ่มฝึกใหม่ทุกครั้งและใช้ข้อมูลแบ่งชุดเดียวกัน

Epoch คือการฝึกผ่านข้อมูล Train ครบหนึ่งรอบ เพิ่ม Epochs ไม่รับประกันว่า Accuracy จะสูงขึ้นเสมอ

## การประเมินผล

Accuracy คือสัดส่วนภาพที่ Model ทำนายถูก ส่วน Loss คือค่าความผิดพลาด โดยทั่วไปยิ่งต่ำยิ่งดี

```python
test_loss, test_accuracy = model.evaluate(X_test, y_test)
```

บันทึก Accuracy และ Loss ของทุกการทดลองใน `results.csv` และเปรียบเทียบด้วย `accuracy_comparison.png`

กราฟ Training History แสดง Training/Validation Accuracy และ Loss ของ Model ที่เลือก หาก Train ดีขึ้นแต่ Validation แย่ลง อาจเกิด Overfitting หรือการจำภาพฝึกมากเกินไป

## Prediction

แสดงตัวอย่าง 4 ภาพจาก Test พร้อมข้อความ:

```text
True: BMW
Pred: BMW (92%)
```

`True` คือ Class จริง `Pred` คือ Class ที่ทำนาย เปอร์เซ็นต์คือค่า Confidence จาก Model ไม่ใช่การรับประกันว่าตอบถูก ข้อความสีเขียวคือถูก สีแดงคือผิด

## ผลลัพธ์ที่ได้

```text
outputs/
├── results.csv
├── accuracy_comparison.png
├── training_history.png
├── confusion_matrix.png
└── prediction_sample.png
```

| ไฟล์ | หน้าที่ |
|---|---|
| results.csv | เปรียบเทียบ Accuracy/Loss ของ 6 การทดลอง |
| accuracy_comparison.png | กราฟเปรียบเทียบ Configuration และ Epochs |
| training_history.png | Training/Validation Accuracy และ Loss ของ Model ที่เลือก |
| confusion_matrix.png | จำนวนที่ทำนายถูกและผิด แยกตาม Class |
| prediction_sample.png | ตัวอย่างภาพพร้อมคำตอบจริงและผลทำนาย |

Folder `saved_model` เก็บ Model และข้อมูลที่ `test_cnn.py` ต้องใช้ ไม่ใช่ผลสำหรับใส่รายงาน

## วิธีรัน

ติดตั้ง Library:

```powershell
python -m pip install -r requirements.txt
```

รันทั้งหมด:

```powershell
python Runall.py
```

หรือรันแยก:

```powershell
python main.py
python test_cnn.py
```

`main.py` ฝึกและประเมิน Model ส่วน `test_cnn.py` โหลด Model มาแสดงตัวอย่าง Prediction ต้องฝึกก่อนครั้งแรก

## ข้อจำกัด

Dataset มีภาพน้อย ผลจาก Test ชุดเดียวจึงยังไม่บอกความแม่นยำกับภาพรถทั่วไป ควรใช้ผลจริงจากการรันในการเขียนรายงาน และเพิ่มข้อมูลหากต้องการพัฒนาต่อ
