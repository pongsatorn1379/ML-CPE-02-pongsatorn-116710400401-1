นายพงศธร รอดดี 116710400401-1 ที่มา Dataset >> ChatGPT
#ผมดูของอจ.ไม่เข้าใจเลยกลับมาทำเอง 

# LAB 1: K-Nearest Neighbors (KNN)

โปรเจกต์นี้เป็นการทดลองใช้ K-Nearest Neighbors (KNN) สำหรับจำแนกชนิดของผลไม้จากข้อมูลในไฟล์ fruit.csv

ชนิดของผลไม้ใน Dataset มี 3 Class

- Apple
- Orange
- Banana

## Dataset

ไฟล์ข้อมูลอยู่ที่

dataset/fruit.csv

Column ที่ใช้มีดังนี้

| Column | ความหมาย |
|---|---|
| weight | น้ำหนักของผลไม้ |
| sweetness | ระดับความหวาน |
| size | ขนาดของผลไม้ |
| class | ชนิดของผลไม้ |

## หลักการทำงาน

KNN เป็นอัลกอริทึม Machine Learning แบบ Supervised Learning

ในการทำ Classification จะใช้หลักการดังนี้

1. รับข้อมูลใหม่
2. คำนวณระยะห่างระหว่างข้อมูลใหม่กับข้อมูล Training
3. เลือกข้อมูลที่อยู่ใกล้ที่สุดจำนวน K ตัว
4. ใช้ Majority Voting เพื่อหาว่า Class ใดมีจำนวนมากที่สุด
5. ทำนายข้อมูลใหม่เป็น Class นั้น

## ขั้นตอนการทำ LAB

### PART 1: Import Libraries

นำเข้า Library สำหรับอ่านข้อมูล แบ่งข้อมูล Standardize Feature สร้าง KNN Model และวัด Accuracy

### PART 2: Load Dataset

อ่านข้อมูลจากไฟล์ fruit.csv

### PART 3: Explore Dataset

ตรวจสอบข้อมูลเบื้องต้น ได้แก่

- จำนวนแถวและ Column
- ชื่อ Column
- Missing Value
- ข้อมูลซ้ำ

### PART 4: Preprocess Dataset

เตรียมข้อมูลก่อนนำไป Train โดยลบข้อมูลที่ว่างและข้อมูลที่ซ้ำกัน

### PART 5: Select Features and Target

กำหนด Feature ที่ใช้ทำนาย

- weight
- sweetness
- size

กำหนด Target

- class

Class ที่ต้องการทำนายมี Apple, Orange และ Banana

### PART 6: Train / Test Split

แบ่ง Dataset เป็น

- 80% Training Data
- 20% Testing Data

Training Data ใช้สำหรับ Train Model

Testing Data ใช้สำหรับทดสอบความสามารถของ Model

### PART 7: Standardization

ใช้ StandardScaler เพื่อปรับ Scale ของ Feature ก่อนนำไปใช้กับ KNN

ขั้นตอนนี้ช่วยให้ Feature มี Scale ที่เหมาะสมกับการคำนวณระยะห่างของ KNN

### PART 8: Train KNN Models

ทดลองค่า K ตามใบงาน

- K = 3
- K = 5
- K = 7

ค่า K หมายถึงจำนวนเพื่อนบ้านที่ใกล้ที่สุดที่นำมาใช้ในการตัดสิน Class

จากนั้นให้ Model ทำนายข้อมูล Test และคำนวณ Accuracy ของแต่ละค่า K

### PART 9: Find Best K

เปรียบเทียบ Accuracy ของแต่ละค่า K

ค่า K ที่ให้ Accuracy สูงที่สุดจะถูกเลือกเป็น Best K

## การวัดผล Model

ใช้ Accuracy ในการวัดว่า Model ทำนายข้อมูล Test ได้ถูกต้องมากแค่ไหน

Accuracy คำนวณจากจำนวนข้อมูลที่ทำนายถูกเทียบกับจำนวนข้อมูลทั้งหมด

ค่า Accuracy ยิ่งสูง แสดงว่า Model ทำนายข้อมูล Test ได้ถูกต้องมากขึ้น

## Output ที่ต้องดู

- Accuracy ของ K = 3
- Accuracy ของ K = 5
- Accuracy ของ K = 7
- Best K
- Best Accuracy

## สรุป

LAB นี้ใช้ KNN เพื่อจำแนกชนิดของผลไม้จาก weight, sweetness และ size

Model จะทดลองค่า K เท่ากับ 3, 5 และ 7 แล้วเปรียบเทียบ Accuracy จากข้อมูล Test เพื่อหาค่า K ที่เหมาะสมที่สุดสำหรับ Dataset นี้
