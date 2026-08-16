#นายพงศธร รอดดี 116710400401-1 Sec 2 Credit : Github ของคุณ mbostock
 Machine Learning: KNN และ K-Means กับข้อมูลรถยนต์
 1. ภาพรวม

ใช้ข้อมูลรถยนต์จากไฟล์ `cars/cars.csv` เพื่อสาธิตการเรียนรู้ของเครื่อง 2 รูปแบบ

1. **Classification ด้วย K-Nearest Neighbors (KNN)**: ทำนายค่า `cylinders` ของรถยนต์จากคุณลักษณะทางตัวเลข
2. **Clustering ด้วย K-Means**: จัดรถยนต์ออกเป็นกลุ่มตามความคล้ายกันโดยไม่ใช้ป้ายกำกับล่วงหน้า จากนั้นใช้ KNN เพื่อจัดรถใหม่เข้ากลุ่มที่เหมาะสม

โค้ด KNN และ K-Means เขียนด้วย TensorFlow ในไฟล์ `knn_tf.py`, `kmeans_tf.py` และ `knn_tools.py` ส่วนการแบ่งข้อมูล การปรับสเกล และการประเมินผลใช้ pandas, NumPy, scikit-learn และ matplotlib

3. ข้อมูลและคุณลักษณะที่ใช้

ไฟล์ `cars/cars.csv` มีข้อมูลเกี่ยวกับรถยนต์ เช่น `economy (mpg)`, `cylinders`, `displacement (cc)`, `power (hp)`, `weight (lb)`, `0-60 mph (s)` และ `year`

ในส่วน classification ใช้ `cylinders` เป็นตัวแปรเป้าหมาย จึงไม่ใช้คอลัมน์นี้เป็น feature เพื่อป้องกันการนำคำตอบมาใช้ทำนายตัวเอง ส่วน clustering ใช้ feature ที่กำหนดไว้ทั้งหมด รวมทั้ง `cylinders` เพราะเป็นการค้นหากลุ่มจากข้อมูลทั้งหมด

4. การเตรียมข้อมูล

Classification: `classification/data_loader.py`

ฟังก์ชัน `load_data()` ทำงานดังนี้

1. อ่าน CSV ด้วย pandas
2. ทำความสะอาดค่า target `cylinders`
3. เรียงชื่อคลาสและแปลงคลาสเป็นตัวเลข เช่น 0, 1, 2
4. เลือกเฉพาะ feature และ target แล้วลบแถวที่มีค่า NaN
5. แบ่งข้อมูลแบบ stratified เป็น train 60%, validation 20% และ test 20%
6. ใช้ `StandardScaler` โดย fit เฉพาะชุด train แล้ว transform validation และ test ด้วย scaler เดียวกัน
7. คืนค่า train, validation, test, ชื่อคลาส และจำนวนแถว

การ standardize คือ

```text
ค่าหลังปรับสเกล = (ค่าจริง - ค่าเฉลี่ยของ train) / ส่วนเบี่ยงเบนมาตรฐานของ train
```

ขั้นตอนนี้สำคัญกับ KNN เพราะ KNN ใช้ระยะทาง หาก feature มีช่วงค่าต่างกันมาก feature ที่มีตัวเลขใหญ่จะมีอิทธิพลมากเกินไป

Clustering: `clustering/data_loader.py`

โค้ดอ่านข้อมูล ลบแถวที่มีค่าว่าง และสร้างข้อมูล 2 ชุด

- `X_raw`: ข้อมูลหน่วยจริง ใช้อธิบายค่าเฉลี่ยของแต่ละกลุ่ม
- `X`: ข้อมูลที่ผ่าน StandardScaler ใช้คำนวณระยะทาง

ส่วนนี้ไม่แบ่ง train/validation/test เพราะ K-Means เป็น unsupervised learning และไม่มี target สำหรับแบ่งคำตอบตั้งต้น

5. Classification ด้วย KNN
	หลักการเมื่อมีข้อมูลใหม่ KNN คำนวณระยะห่างจากข้อมูล train ทุกแถวด้วย Euclidean distance

```text
d(x,y) = sqrt(ผลรวมของ (x_i - y_i)^2 ทุก feature)
```

จากนั้นเลือกข้อมูลที่ใกล้ที่สุดจำนวน `k` แถว แล้วให้ข้อมูลใหม่เป็นคลาสที่มีเสียงโหวตมากที่สุด

การทำงานใน `knn_tf.py`

คลาส `TFKNNClassifier` มีขั้นตอนสำคัญดังนี้

- `fit(X, y)`: เก็บข้อมูล train เป็น TensorFlow tensor และหาจำนวนคลาส
- `_distance(X_new)`: ใช้ broadcasting คำนวณระยะห่างระหว่างข้อมูลใหม่ทุกแถวกับ train ทุกแถวพร้อมกัน
- `predict(X)`: ใช้ `tf.math.top_k(-dist)` เพื่อเลือกค่าระยะทางที่น้อยที่สุด ดึง label ของเพื่อนบ้าน แปลงเป็น one-hot และรวมคะแนนโหวต
- `score(X, y)`: คำนวณ accuracy หรือสัดส่วนจำนวนคำตอบที่ทำนายถูก

KNN ไม่มีการสร้างสมการหรือปรับน้ำหนักตอน fit แต่เก็บข้อมูล train ไว้และคำนวณตอน predict

การเลือกค่า k ใน `classification/main.py`

โปรแกรมทดสอบค่า `k` คือ `1, 2, 3, 5, 10, 11, 15, 21, 30` โดยฝึกกับ train และวัด accuracy กับ validation ทุกค่า แล้วเลือกค่าที่มี validation accuracy สูงสุดด้วย `np.argmax()`

การใช้ validation เพื่อเลือก k และเก็บ test ไว้จนจบทำให้การประเมิน test เป็นธรรม เพราะโมเดลไม่เคยใช้ test ในการตัดสินใจเลือก hyperparameter

หลังเลือก `best_k` โปรแกรมฝึกโมเดลใหม่ด้วย train แล้วทำนาย test และแสดงผลดังนี้

- accuracy ของ test
- classification report: precision, recall และ F1-score รายคลาส
- confusion matrix: แถวคือคลาสจริง คอลัมน์คือคลาสที่ทำนาย
- เปรียบเทียบกับ `sklearn.neighbors.KNeighborsClassifier`
- เปรียบเทียบกับ baseline ที่ทำนายคลาสที่พบบ่อยที่สุดทุกแถว

ไฟล์ผลลัพธ์

- `outputs/01_k_curve.png`: กราฟ k กับ validation accuracy
- `outputs/02_confusion_matrix.png`: กราฟ confusion matrix
- `outputs/predictions.csv`: true label, predicted label และ correct

6. Clustering ด้วย K-Means

	หลักการK-Means แบ่งข้อมูลเป็น `k` กลุ่มโดยวนซ้ำ 2 ขั้นตอน

1. **Assign**: ให้แต่ละจุดอยู่กลุ่มของ centroid ที่ใกล้ที่สุด
2. **Update**: หาค่าเฉลี่ยสมาชิกในแต่ละกลุ่ม แล้วใช้ค่าเฉลี่ยเป็น centroid ใหม่

หยุดเมื่อ centroid เคลื่อนที่น้อยกว่า `1e-4` หรือวนครบ `max_iter=100` รอบ
การทำงานใน `kmeans_tf.py`
คลาส `TFKMeans` สุ่มเลือกข้อมูลจำนวน k จุดเป็น centroid เริ่มต้น โดยใช้ seed 42 เพื่อให้ทำซ้ำได้ จากนั้นคำนวณ Euclidean distance ด้วย TensorFlow และใช้ `tf.argmin()` เลือก centroid ที่ใกล้ที่สุด
ถ้ากลุ่มใดไม่มีสมาชิก โค้ดจะคง centroid เดิมไว้ เพื่อป้องกันปัญหา centroid หายระหว่างการวนซ้ำ หลังจบการเรียนรู้จะเก็บ `labels_`, `centroids_`, จำนวนรอบ `n_iter_` และ `inertia_`
`inertia` คือผลรวมกำลังสองของระยะห่างจากแต่ละจุดไปยัง centroid ของกลุ่มตัวเอง ยิ่งน้อยแปลว่าจุดในกลุ่มกระชับขึ้น แต่ค่า inertia จะลดลงเมื่อจำนวนกลุ่มเพิ่ม จึงต้องดูร่วมกับ elbow และ silhouette

การเลือกจำนวนกลุ่มใน `clustering/main.py` ทดลองจำนวนกลุ่ม 2 ถึง 8 และบันทึก

- **Inertia**: ใช้สร้าง Elbow Method
- **Silhouette score**: วัดว่าจุดใกล้กลุ่มของตัวเองและห่างจากกลุ่มอื่นเพียงใด ค่าสูงมักหมายถึงกลุ่มแยกกันชัด

จากนั้นกำหนด `N_CLUSTERS = 4` และรัน K-Means อีกครั้งด้วย k เท่ากับ 4 การเลือกนี้อ้างอิงจากกราฟและค่า silhouette ไม่ใช่ label จริง

โปรแกรมสร้างตาราง profile โดยใช้ `X_raw` หาค่าเฉลี่ยแยกตาม cluster และนับจำนวนสมาชิก เพื่ออธิบายลักษณะของรถในแต่ละกลุ่ม

ใช้ KNN จัดรถใหม่เข้ากลุ่ม

หลัง K-Means สร้าง label ให้รถทุกคัน โปรแกรมจำลองรถใหม่ดังนี้

- ใช้ 80% แรกเป็นข้อมูลที่รู้กลุ่มแล้ว
- ใช้ 20% ที่เหลือเป็นรถใหม่
- ฝึก `KNNClusterAssigner(k=5)` ด้วยข้อมูล 80%
- ให้ KNN ทำนายกลุ่มรถใหม่ แล้วเปรียบเทียบกับ label ที่ K-Means คำนวณไว้

จุดประสงค์คือแสดงว่า เมื่อสร้างกลุ่มด้วย K-Means แล้ว สามารถใช้ KNN จัดข้อมูลใหม่เข้ากลุ่มได้โดยไม่ต้องรัน K-Means ใหม่ทุกครั้ง หากกลุ่มแยกจากกันชัด KNN มักจัดรถใหม่ได้ดีขึ้น

ไฟล์ผลลัพธ์

- `outputs/01_elbow.png`: กราฟจำนวนกลุ่มกับ inertia
- `outputs/02_clusters.png`: scatter plot ผลการจัดกลุ่ม
- `outputs/cluster_summary.csv`: ค่าเฉลี่ย feature และจำนวนสมาชิก
- `outputs/clustered_cars.csv`: ข้อมูลรถพร้อมคอลัมน์ cluster

การประเมินผลและข้อสังเกต

Classification มีคำตอบจริงคือ `cylinders` จึงประเมินด้วย accuracy, precision, recall, F1-score และ confusion matrix ได้โดยตรง ส่วน clustering ไม่มีคำตอบจริง จึงใช้ inertia, silhouette score และ profile ของแต่ละกลุ่มแทน
ถ้า accuracy ของ KNN ใกล้เคียงหรือต่ำกว่า baseline แปลว่า feature ที่เลือกอาจแยกค่า `cylinders` ได้ไม่ดี หรือข้อมูลมีความสัมพันธ์กับ target ไม่มาก ไม่ได้แปลว่าโค้ดผิดเสมอไป การเปรียบเทียบกับ scikit-learn ช่วยตรวจสอบว่า KNN ที่เขียนเองให้ผลสอดคล้องกับไลบรารีมาตรฐานหรือไม่
ข้อจำกัดของ KNN คือใช้การโหวตแบบธรรมดา ไม่ให้น้ำหนักตามระยะทาง และคำนวณระยะห่างกับ train ทุกแถว จึงอาจใช้หน่วยความจำมากเมื่อข้อมูลขนาดใหญ่ ส่วน K-Means ขึ้นกับ centroid เริ่มต้นและเหมาะกับกลุ่มที่มีรูปทรงค่อนข้างกลม จึงควรดู silhouette และกราฟประกอบ ไม่ควรสรุปจากจำนวนกลุ่มเพียงอย่างเดียว
ไฟล์ `runall.py` เรียกไฟล์ Python ภายในโฟลเดอร์ทีละไฟล์ แต่สำหรับสร้างผลลัพธ์หลักควรเรียก `main.py` โดยตรง เพราะเป็นตัวควบคุมลำดับขั้นตอนทั้งหมด


