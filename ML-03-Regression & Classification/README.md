นายพงศธร รอดดี 116710400401-1 Sec 2
ML-03-Regression & Classification
Dataset Credit : Kaggle


LAB 1: Regression

Simple Linear Regression (สมการถดถอยเชิงเส้นอย่างง่าย)สิ่งที่ทำในโค้ด: สร้างโมเดลโดยใช้ตัวแปรต้น (Features) เพียงตัวเดียว ในที่นี้คือการหยิบเอาแค่ "พื้นที่ใบหน้า (face_area)" มาลองทำนายอายุ  
เหตุผลที่ต้องทำ: เพื่อใช้เป็นโมเดลพื้นฐาน (Baseline Model) สำหรับการเรียนรู้และเปรียบเทียบ ผลลัพธ์จากการทดลองนี้จะแสดงให้เห็นถึงข้อจำกัดอย่างชัดเจนว่า การใช้ข้อมูลปัจจัยเดียวไม่เพียงพอต่อการอธิบายความซับซ้อนของโครงหน้าได้ ทำให้โมเดลทำนายได้ไม่แม่นยำ (ค่า R-Squared ติดลบ)

Multiple Linear Regression (สมการถดถอยเชิงเส้นพหุคูณ)สิ่งที่ทำในโค้ด: สร้างโมเดลโดยนำคุณลักษณะของใบหน้าทุกมิติที่คำนวณได้ (ความกว้าง, ความสูง, พื้นที่, สัดส่วน ฯลฯ รวม 6 ตัวแปร) มาใช้ทำนายพร้อมกัน โดยมีการใช้ StandardScaler ปรับสเกลข้อมูลก่อนนำไปสอนโมเดล  
เหตุผลที่ต้องทำ: เพื่อให้โมเดลสามารถเรียนรู้ความสัมพันธ์ของใบหน้าแบบองค์รวม การนำตัวแปรหลายมิติมาทำงานร่วมกันจะช่วยยกระดับประสิทธิภาพของโมเดลได้อย่างก้าวกระโดด (ค่า R-Squared เพิ่มขึ้นเป็น 0.6483) ส่วนการปรับสเกลนั้นจำเป็นเพื่อไม่ให้ตัวแปรที่มีค่าตัวเลขสูงๆ อย่างพื้นที่ใบหน้า ไปลดทอนความสำคัญของตัวแปรที่มีค่าน้อยอย่างสัดส่วนใบหน้า

Age Prediction S (การทำนายอายุ)สิ่งที่ทำในโค้ด: เป็นเป้าหมายหลักของการทดลองใน Lab 1 โดยมีการเขียนโค้ดเพื่อจำลองตัวแปรเป้าหมายคือ "อายุ (Age)" ขึ้นมาให้อยู่ในช่วง 1 ถึง 85 ปี โดยอิงจากสมการคณิตศาสตร์ที่นำสัดส่วนใบหน้ามาผสมกับค่าความคลาดเคลื่อนแบบสุ่ม  เหตุผลที่จัดอยู่ในหมวดนี้: เนื่องจากตารางข้อมูลต้นฉบับไม่มีป้ายกำกับอายุอยู่จริงจึงต้องจำลองขึ้นมา และสาเหตุที่การทำนายอายุต้องอยู่ในหมวด Regression เป็นเพราะ "อายุ" คือข้อมูลชนิด "ตัวเลขต่อเนื่อง (Continuous Data)" โมเดลในกลุ่ม Regression จึงเป็นเครื่องมือที่ถูกต้องและเหมาะสมที่สุดในการสร้างสมการเพื่อทำนายผลลัพธ์ออกมาเป็นค่าตัวเลขครับ   


LAB 2: Classification

Preparing Classification Data (การเตรียมข้อมูลสำหรับการจำแนกประเภท)สิ่งที่ทำในโค้ด: เขียนโค้ดสุ่มสร้างตัวแปรเป้าหมาย (Target) คือ "เพศ (Gender)" โดยกำหนดให้เป็นค่า 0 และ 1 ลงไปในตารางข้อมูล จากนั้นทำการเลือกตัวแปรต้น (Features) มาเพียง 2 ตัว ได้แก่ ความกว้างใบหน้า (face_width) และความสูงใบหน้า (face_height) เพื่อนำไปแบ่งเป็นชุดฝึกและชุดทดสอบ  
เหตุผลที่ต้องทำ: ข้อมูลดิบจากไฟล์ต้นฉบับมีเพียงพิกัดมุมของกรอบใบหน้า และไม่มีป้ายกำกับเรื่องเพศอยู่จริง จึงจำเป็นต้องจำลองขึ้นมาเพื่อใช้ทดสอบการเรียนรู้ ส่วนสาเหตุที่เลือกฟีเจอร์มาเพียง 2 ตัวแปรนั้น เป็นเทคนิคเพื่อให้สามารถนำไปวาดกราฟแบบ 2 มิติเพื่อดูขอบเขตการตัดสินใจในขั้นตอนต่อไปได้ง่ายขึ้น

Decision Boundary Visualization (การวาดภาพเส้นแบ่งการตัดสินใจ)สิ่งที่ทำในโค้ด: ดึงค่าน้ำหนักสัมประสิทธิ์ (Weights) และจุดตัดแกน (Intercept) ที่ได้จากการคำนวณของโมเดลมาสร้างเป็นสมการเส้นตรง จากนั้นพล็อตกราฟกระจาย (Scatter plot) ของข้อมูลชุดทดสอบ แล้ววาดเส้นแบ่งนี้ทับลงไป  
เหตุผลที่ต้องทำ: การทำงานของอัลกอริทึมทางคณิตศาสตร์มักเป็นกล่องดำที่มองเห็นได้ยาก การนำมาพล็อตกราฟจะช่วยให้เราเห็นภาพชัดเจนว่าโมเดลขีดเส้นแบ่งอาณาเขตอย่างไรเพื่อแยกเพศชายและหญิงออกจากกัน และทำให้เห็นว่ามีจุดข้อมูลใดบ้างที่กระจายตัวทับซ้อนกันจนโมเดลอาจเกิดความสับสน

Logistic Regression (การใช้อัลกอริทึมการถดถอยโลจิสติก)สิ่งที่ทำในโค้ด: เรียกใช้ฟังก์ชัน LogisticRegression() และสั่งให้โมเดลฝึกสอน (Fit) เรียนรู้รูปแบบจากข้อมูลใบหน้าและเพศที่เตรียมไว้  
เหตุผลที่ต้องทำ: แม้ชื่อจะมีคำว่า Regression แต่อัลกอริทึมนี้ถูกออกแบบมาเพื่องาน Classification โดยเฉพาะ มันทำงานโดยการเปลี่ยนผลลัพธ์จากสมการเชิงเส้นให้กลายเป็น "ค่าความน่าจะเป็น (Probability)" ตั้งแต่ 0 ถึง 1 จึงเป็นโมเดลพื้นฐานที่เหมาะสมและทรงประสิทธิภาพมากสำหรับงานจำแนกประเภทที่มีคำตอบเพียง 2 ทางเลือก (Binary Classification) เช่น ชายหรือหญิง

Gender Prediction (การทำนายเพศ)สิ่งที่ทำในโค้ด: ใช้คำสั่ง .predict() เพื่อให้โมเดลประมวลผลข้อมูลคุณลักษณะใบหน้าชุดทดสอบ (X_test_class) ที่โมเดลไม่เคยเห็นมาก่อน แล้วทำนายผลลัพธ์ออกมาเป็นค่า 0 หรือ 1  เหตุผลที่จัดอยู่ในหมวดนี้: นี่คือเป้าหมายหลักของการทดลองใน Lab 2 สาเหตุที่การทำนายเพศถูกจัดอยู่ในหมวด Classification เป็นเพราะเป้าหมายคือข้อมูลชนิด "หมวดหมู่ (Categorical Data)" อัลกอริทึมจึงต้องตัดสินใจฟันธงเพื่อจัดกลุ่มข้อมูล ไม่ใช่การประมาณค่าตัวเลขต่อเนื่องแบบการทายอายุใน Lab 1

Confusion Matrix (เมทริกซ์ความสับสน)สิ่งที่ทำในโค้ด: นำผลการทำนายที่ได้จากโมเดลมาเปรียบเทียบกับคำตอบจริง แล้วนำไปสร้างเป็นตารางเมทริกซ์ (Confusion Matrix) พร้อมพล็อตเป็นรูปภาพ Heatmap ให้ดูง่ายขึ้น  
เหตุผลที่ต้องทำ: การประเมินผลโมเดล Classification ไม่สามารถใช้ค่าความคลาดเคลื่อน (Error) แบบ Regression ได้ การใช้ตาราง Confusion Matrix จึงจำเป็นอย่างยิ่ง เพื่อให้เห็นรายละเอียดเชิงลึกว่าโมเดลทายถูกตรงกลุ่มกี่คน และทายผิดสลับกลุ่ม (ทายชายเป็นหญิง หรือหญิงเป็นชาย) ไปอย่างละกี่คน ซึ่งจะนำไปสู่การหาค่า Precision, Recall ในการวัดผลขั้นสูงต่อไป


LAB 3: Model Comparison

Simple vs Multiple Linear Regression (การเปรียบเทียบสมการถดถอยเชิงเส้นอย่างง่ายและพหุคูณ)สิ่งที่ทำในโค้ด: นำผลการทำนายที่ได้จากโมเดลใน Lab 1 มาคำนวณและเปรียบเทียบค่าความแม่นยำ (R-Squared) ระหว่างโมเดลที่ใช้ตัวแปรเดียวกับโมเดลที่ใช้หลายตัวแปร  
เหตุผลที่ต้องทำ: เพื่อแสดงให้เห็นผลลัพธ์เชิงประจักษ์ทางสถิติว่า โมเดล Multiple Regression มีความแม่นยำสูงกว่าอย่างชัดเจน (R-Squared = 0.6483) เมื่อเทียบกับ Simple Regression (R-Squared ติดลบ) เป็นการพิสูจน์ว่าข้อมูลคุณลักษณะใบหน้าต้องใช้หลายมิติประกอบกันจึงจะอธิบาย "อายุ" ได้ดี  

Training vs Testing Performance (การเปรียบเทียบประสิทธิภาพบนชุดฝึกสอนและชุดทดสอบ)สิ่งที่ทำในโค้ด: นำโมเดลจำแนกประเภท (Classification) จาก Lab 2 มาสั่งให้ทำนายข้อมูล แล้ววัดค่าความแม่นยำ (Accuracy) โดยแยกเทียบระหว่างคะแนนที่ได้จากชุดข้อมูลที่ใช้สอน (Training Accuracy) และชุดข้อมูลที่ใช้สอบ (Testing Accuracy)  
เหตุผลที่ต้องทำ: นี่คือขั้นตอนสำคัญในการตรวจสอบภาวะ "การจำข้อสอบ (Overfitting)" หากความแม่นยำตอนฝึกสอนสูงกว่าตอนสอบมาก ๆ (ในโค้ดคือ 52.8% เทียบกับ 42.5%) จะเป็นสัญญาณเตือนว่าโมเดลอาจจะแค่ท่องจำข้อมูลชุดฝึก แต่ไม่สามารถนำไปประยุกต์ใช้หรือทายผลกับข้อมูลใหม่ที่ไม่เคยเห็นได้ดีนัก 

Regression vs Classification (การเปรียบเทียบการทำนายแบบถดถอยและการจำแนกประเภท)สิ่งที่ทำในโค้ด: ดึงตัวอย่างผลลัพธ์ที่ได้จากการทำนายของโมเดลทั้งสองประเภทมาพิมพ์แสดงผลเปรียบเทียบกันให้เห็นชัด ๆ  
เหตุผลที่ต้องทำ: เพื่อเน้นย้ำให้เห็นความแตกต่างของผลลัพธ์อย่างเป็นรูปธรรม ว่าแม้จะเริ่มต้นจากชุดข้อมูลพิกัดกรอบใบหน้าชุดเดียวกัน แต่เมื่อเป้าหมายการแก้ปัญหาต่างกัน โมเดล Regression (ทายอายุ) จะให้ผลลัพธ์เป็นตัวเลขที่มีความต่อเนื่อง (เช่น 25.8, 35.6) ในขณะที่โมเดล Classification (ทายเพศ) จะให้คำตอบแบบฟันธงเป็นหมวดหมู่ (เช่น 0 หรือ 1)  

Model Performance Metrics (การดูตัวชี้วัดประสิทธิภาพโมเดล)สิ่งที่ทำในโค้ด: เรียกใช้ฟังก์ชัน classification_report เพื่อคำนวณและแสดงผลค่าทางสถิติขั้นสูงสำหรับโมเดลจำแนกเพศ ซึ่งประกอบไปด้วยค่า Precision, Recall และ F1-score แยกตามกลุ่ม (Male และ Female)  
เหตุผลที่ต้องทำ: ในงาน Classification การดูแค่ความแม่นยำรวม (Accuracy) อาจไม่เพียงพอและหลอกตาได้ (โดยเฉพาะถ้าข้อมูลทั้งสองกลุ่มมีจำนวนไม่เท่ากัน) การใช้ตัวชี้วัดเหล่านี้จะช่วยเจาะลึกให้เราเห็นประสิทธิภาพที่แท้จริงว่า โมเดลเก่งในการทายกลุ่มไหนมากกว่ากัน หรือมีจุดอ่อนในการทายพลาดกลุ่มไหนเป็นพิเศษ  




LAB 1: Regression

Simple Linear Regression

What it does in the code: Creates a model using only one feature (independent variable), which in this case is the "face area" (face_area), to try and predict age.

Reason for doing this: To serve as a baseline model for learning and comparison. The results from this experiment clearly demonstrate the limitation that using a single factor is insufficient to explain the complexity of facial structures, resulting in an inaccurate model (with a negative R-Squared value).

Multiple Linear Regression

What it does in the code: Creates a model by utilizing all calculated facial feature dimensions simultaneously (width, height, area, ratios, etc., totaling 6 variables). A StandardScaler is used to scale the data before training the model.

Reason for doing this: To allow the model to learn the holistic relationships of the face. Using multiple dimensions together significantly elevates the model's performance (R-Squared increases to 0.6483). Data scaling is necessary so that variables with high numerical values, such as face area, do not diminish the importance of variables with lower values, such as facial proportions.

Age Prediction S

What it does in the code: This is the primary goal of the experiment in Lab 1. Code is written to simulate the target variable, "Age," generating values ranging from 1 to 85 years based on a mathematical equation that combines facial proportions with random noise.

Reason for being in this category: Since the original dataset does not contain actual age labels, they must be simulated. Predicting age falls under the Regression category because "Age" is continuous data. Regression models are the correct and most appropriate tools for creating equations to predict continuous numerical outcomes.

LAB 2: Classification

Preparing Classification Data

What it does in the code: Writes code to randomly generate the target variable, "Gender," assigning values of 0 and 1 into the data table. Then, it selects only 2 features—face width (face_width) and face height (face_height)—to split into training and testing sets.

Reason for doing this: The raw data from the original file only provides the coordinates of the facial bounding boxes and lacks actual gender labels, so it needs to be simulated for testing and learning. Selecting only 2 features is a technique that makes it easier to plot a 2D graph to visualize the decision boundary in the subsequent step.

Decision Boundary Visualization

What it does in the code: Extracts the weights and intercept calculated by the model to formulate a linear equation. It then creates a scatter plot of the testing data and overlays this boundary line on it.

Reason for doing this: The inner workings of mathematical algorithms are often opaque "black boxes." Plotting a graph clearly illustrates how the model draws a boundary line to separate males from females, and it reveals which data points overlap to the extent that they might confuse the model.

Logistic Regression

What it does in the code: Calls the LogisticRegression() function and instructs the model to fit (learn) the patterns from the prepared face and gender data.

Reason for doing this: Despite having "Regression" in its name, this algorithm is specifically designed for Classification tasks. It works by converting the output of a linear equation into a "Probability" ranging from 0 to 1. Therefore, it is a highly appropriate and powerful foundational model for classification tasks that have only 2 choices (Binary Classification), such as male or female.

Gender Prediction

What it does in the code: Uses the .predict() command to have the model process the testing facial feature data (X_test_class) that it has never seen before, outputting a predicted value of 0 or 1.

Reason for being in this category: This is the main objective of the Lab 2 experiment. Gender prediction is categorized under Classification because the target is categorical data. The algorithm must make a definitive decision to group the data, rather than estimating a continuous numerical value like the age prediction in Lab 1.

Confusion Matrix

What it does in the code: Compares the model's predictions with the actual answers to create a Confusion Matrix, plotting it as a Heatmap for easier visual interpretation.

Reason for doing this: Evaluating a Classification model cannot rely on error metrics used in Regression. A Confusion Matrix is essential to reveal in-depth details about how many instances the model predicted correctly per group, and how many it misclassified (e.g., predicting a male as a female or vice versa). This ultimately leads to the calculation of advanced evaluation metrics like Precision and Recall.

LAB 3: Model Comparison

Simple vs Multiple Linear Regression

What it does in the code: Takes the prediction results from the models in Lab 1 to calculate and compare the accuracy metric (R-Squared) between the single-variable model and the multi-variable model.

Reason for doing this: To provide empirical statistical evidence that the Multiple Regression model possesses significantly higher accuracy (R-Squared = 0.6483) compared to Simple Regression (negative R-Squared). This proves that facial feature data requires a combination of multiple dimensions to effectively explain "Age."

Training vs Testing Performance

What it does in the code: Uses the classification model from Lab 2 to predict data, then measures the Accuracy by separately comparing the scores obtained from the training dataset (Training Accuracy) and the testing dataset (Testing Accuracy).

Reason for doing this: This is a crucial step for detecting "Overfitting." If the training accuracy is substantially higher than the testing accuracy (in the code, 52.8% versus 42.5%), it serves as a warning sign that the model might merely be memorizing the training data and will struggle to apply its learning to predict new, unseen data accurately.

Regression vs Classification

What it does in the code: Extracts sample prediction outputs from both types of models and prints them side-by-side for a clear comparison.

Reason for doing this: To concretely emphasize the difference in outcomes. Even though both start from the exact same facial bounding box dataset, because their problem-solving goals differ, the Regression model (predicting age) outputs continuous numerical values (e.g., 25.8, 35.6), whereas the Classification model (predicting gender) provides definitive categorical answers (e.g., 0 or 1).

Model Performance Metrics

What it does in the code: Calls the classification_report function to calculate and display advanced statistical metrics for the gender classification model, consisting of Precision, Recall, and F1-score, broken down by group (Male and Female).

Reason for doing this: In Classification tasks, looking solely at overall Accuracy might be insufficient and misleading (especially if the dataset is imbalanced between the two groups). Utilizing these metrics helps drill down into the true performance, showing whether the model is better at predicting a specific group or if it has a particular weakness in misclassifying one group over the other.