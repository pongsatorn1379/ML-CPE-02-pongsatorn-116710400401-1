นายพงศธร รอดดี 116710400401-1 Sec 2    ML02

Version ภาษาไทย

ทำไมต้องทำ Data Preprocessing ก่อนที่จะไปสอนให้ AI หรือ ML เรียนรู้เราต้องทำให้ข้อมูลดิบที่ได้มานั้นพร้อมใช้งานก่อน เช่น อาจจะมีพิมพ์ เว้นว่าง หรือรูปซ้ำๆ ถ้าเราเอาข้อมูลที่ผิดพลาดไปสอน โมเดลก็จะจำอะไรผิดๆไปใช้


Part 1 Dataset Exploration : เราต้องรู้ก่อนว่าข้อมูลเราหน้าตาเป็นยังไง มีจุดบกพร่องตรงไหนบ้าง เพื่อที่จะได้เลือกวิธีทำความสะอาดได้ถูกจุด
Load Dataset : เป็นการโหลดไฟล์ที่ต้องการให้นำมาตรวจสอบ
Display Shape : เพื่อให้รู้สเกลของงาน ว่าเรากำลังจัดการกับข้อมูลกี่แถว กี่คอลัมน์
Display Data Types : เพื่อเช็คว่าคอลัมน์ไหนเป็นตัวเลข และคอลัมน์ไหนเป็นข้อความ
Display Summary Statistics : เป็นการเช็คความผิดปกติแบบไวที่สุด เมื่อผมเช็คค่า Min/Max ผมเจอทันทีว่าคอลัมน์อายุมีคนอายุ -5 ปี และ 99 ปี ซึ่งเป็นไปไม่ได้แสดงว่าต้องมีคนกรอกข้อมูลผิดแน่นอน
Display Missing Values : โมเดล ML ส่วนใหญ่ถ้ารันเจอค่าว่างโปรแกรมจะพังและ Error ทันทีต้องหาให้เจอก่อนว่ามีช่องโหว่ตรงไหน
Display Duplicate Records : ถ้าเราปล่อยให้มีข้อมูลซ้ำ โมเดลจะให้ความสำคัญกับข้อมูลนั้นมากเกินไปจนเกิดความลำเอียง
Display Class Distribution : เพื่อดูว่าสัดส่วนของคนที่ 'สอบผ่าน' และ 'สอบตก' มีความสมดุลกันหรือไม่


Part 2 Data Visualization การวาดกราฟออกมาจะช่วยยืนยันสมมติฐานและทำให้เราเข้าใจข้อมูลได้ง่ายขึ้น
Histogram : การเป็นกราฟจะทำให้มองได้ง่ายกว่าว่าข้อมูลไปเกาะกลุ่มอยู่ส่วนไหน
Correlation Heatmap : เพื่อหาความสัมพันธ์ของตัวแปรครับ เช่น กราฟสีจะช่วยคอนเฟิร์มว่า 'ชั่วโมงเรียน' มีผลเชิงบวกกับ 'คะแนนสอบ' หรือไม่


Part 3 Data Cleaning เมื่อเราเห็นปัญหาจาก LAB 1 และ 2 แล้ว ขั้นตอนนี้คือการทำความสะอาดข้อมูล
Duplicate Removal : สั่งลบแถวที่ซ้ำซ้อนออก เพื่อไม่ให้โมเดลเกิดความลำเอียง
Missing Value Handling : ผมเลือกใช้วิธี เติมค่าเฉลี่ย ลงไปในช่องที่ว่าง แทนที่จะลบแถวนั้นทิ้ง เพราะข้อมูลเรามีน้อย การลบทิ้งจะทำให้เสียข้อมูลที่มีค่าอื่นๆ ของนักเรียนคนนั้นไปด้วย
Incorrect Data Correction : เพื่อจัดการคนที่อายุ -5 และ 99 ปี เขียนโค้ดดักจับว่าถ้าอายุไม่อยู่ในเกณฑ์ที่สมเหตุสมผล ให้เปลี่ยนเป็นค่า 'อายุเฉลี่ย' แทน เพื่อดึงข้อมูลให้กลับมาสู่ความเป็นจริง
Data Type Conversion : พอเราคำนวณค่าเฉลี่ยอายุ ตัวเลขมันกลายเป็นทศนิยม ต้องแปลงกลับให้เป็นจำนวนเต็มเพื่อให้ข้อมูลมีความสมจริงและถูกต้องตามตรรกะ
Compare Mean & Median : ถ้าค่า Mean และ Median มีค่าใกล้เคียงกัน แปลว่าข้อมูลเราค่อนข้างสมดุล


Part 4 Feature Engineering ขั้นตอนนี้คือหัวใจสำคัญก่อนส่งข้อมูลเข้าโมเดล เพราะ Machine Learning เป็นสมการคณิตศาสตร์ มันอ่านตัวหนังสือภาษาอังกฤษ (เช่น Male, Female, Yes, No) ไม่ออก เราจึงต้อง แปลงข้อความพวกนี้ให้เป็นตัวเลข 0 และ 1 เพื่อให้ AI นำไปคำนวณต่อได้
Label Encoding : ใช้สำหรับข้อมูลที่มีแค่ 2 สถานะ เช่น Passed ผมแปลงให้ No กลายเป็น 0 และ Yes กลายเป็น 1
One-Hot Encoding : ใช้กับข้อมูลประเภทหมวดหมู่ เช่น Gender ใช้คำสั่งแปลงให้คอลัมน์เพศแตกออกเป็นคอลัมน์ใหม่ที่มีค่าเป็น 1 กับ 0 เพื่อป้องกันไม่ให้โมเดลเข้าใจผิดว่า เพศที่มีค่าตัวเลขสูงกว่า จะมีค่ามากกว่าหรือดีกว่าครับ



Version English

Student Name: Pongsatorn Roddee

Student ID: 116710400401-1 | Section: 2

Introduction: Why Data Preprocessing?
Before feeding data into an AI or Machine Learning model, we must ensure it is ready for use. Raw data often contains errors such as typos, missing values, or duplicates. Since models learn from the data provided—a principle known as "Garbage In, Garbage Out"—if we train a model with flawed data, it will learn incorrect patterns and produce inaccurate results.

Part 1: Dataset Exploration
This step acts as a "health check" for the data. We need to understand the data's structure and identify potential issues to apply the correct cleaning techniques.

Load Dataset: Loading the file into the environment to begin analysis.

Display Shape: Understanding the scale of the work by identifying the number of rows and columns.

Display Data Types: Determining which columns are numerical (ready for calculation) and which are categorical (needing transformation).

Display Summary Statistics: Quickly detecting anomalies; for instance, identifying ages like -5 or 99, which are physically impossible and indicate data entry errors.

Display Missing Values: Most ML models will throw an error if they encounter missing values (NaN); therefore, we must locate these gaps first.

Display Duplicate Records: Removing duplicates is crucial because repeated data can bias the model, causing it to over-weight certain records.

Display Class Distribution: Assessing whether the target classes (e.g., 'Passed' vs. 'Failed') are balanced.

Part 2: Data Visualization
Visualizing data confirms our hypotheses and makes the overall data patterns easier to comprehend than looking at raw tables.

Histogram: Displays the distribution of data, making it easy to see where most values cluster.

Correlation Heatmap: Identifies relationships between variables; for example, confirming whether 'Study Hours' has a positive impact on 'Exam Scores'.

Part 3: Data Cleaning
After identifying issues in the exploration and visualization phases, this step involves "treating" the data to ensure 100% integrity.

Duplicate Removal: Deleting redundant rows to prevent model bias.

Missing Value Handling: Filling missing values with the Mean rather than deleting rows, as our dataset is small and we want to preserve other valuable information from those records.

Incorrect Data Correction: Setting conditions to catch unrealistic ages (e.g., -5 or 99) and replacing them with the Average Age to bring the data back to a logical range.

Data Type Conversion: Converting decimal-based mean calculations back into Integers for ages, ensuring the data aligns with real-world logic.

Compare Mean & Median: If the Mean and Median are close, it indicates the data is well-balanced and free from extreme outliers that could distort the analysis.

Part 4: Feature Engineering
This is the most critical step before training. Machine Learning models are mathematical equations that cannot process text (e.g., 'Male', 'Female', 'Yes', 'No'). We must act as a translator, converting these strings into 0s and 1s so the AI can perform calculations.

Label Encoding: Used for binary states; for instance, converting 'Passed' status where 'No' becomes 0 and 'Yes' becomes 1.

One-Hot Encoding: Used for categorical data like 'Gender'. We split the gender column into a new format consisting of 1s and 0s. This prevents the model from mistakenly assuming that a higher number (e.g., 2) implies a higher rank or "better" value than a lower number (e.g., 1).







