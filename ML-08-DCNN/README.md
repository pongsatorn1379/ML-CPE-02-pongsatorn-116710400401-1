# LAB 08: Deep Convolutional Neural Network (DCNN)
## ภาษาไทย

นายพงศธร รอดดี 116710400401-1 Sec2

ที่มา Dataset: https://www.kaggle.com/datasets/ahmedelsany/car-brand-classification-dataset

อ้างอิงตัวอย่างจากอาจารย์ใน `Machine-Learning-Course-main/ML-08-DCNN/mini-proj` และนำมาปรับใช้

## Dataset

ใช้รูปภาพรถ 2 Class ได้แก่ **BMW** และ **Honda** โดยใช้ชื่อ Folder เป็นชื่อ Class

```text
dataset/
├── BMW/
└── Honda/
```

โปรแกรมข้ามไฟล์ภาพที่มีข้อมูลเหมือนกันทุกไบต์ก่อนแบ่งชุดข้อมูล เพื่อป้องกันภาพซ้ำอยู่คนละชุด โดยไม่ลบภาพต้นฉบับ ภาพที่ดูคล้ายกันแต่ไฟล์มีข้อมูลต่างกันยังไม่ถูกตรวจจับเป็นภาพซ้ำโดยอัตโนมัติ

## หลักการทำงาน

1. อ่านรูปภาพจาก Folder `dataset`
2. แปลงเป็น RGB และ Resize เป็น 224 × 224 Pixel
3. แบ่งข้อมูลเป็น Train, Validation และ Test
4. สร้าง DCNN 4 แบบ โดยใช้ Weight ที่ฝึกจาก ImageNet มาแล้ว
5. Freeze ส่วน Backbone และฝึกส่วน Classifier ใหม่สำหรับ BMW/Honda
6. ทดลองแต่ละโมเดลด้วย 5, 10 และ 20 Epochs
7. ประเมินผลและสร้าง Training History, Confusion Matrix และตัวอย่าง Prediction
8. เปรียบเทียบการทดลองและเลือกแบบที่มี Validation Accuracy สูงสุด หากเท่ากันใช้ Validation Loss ต่ำสุด
9. บันทึกผลสำหรับรายงานเป็นไฟล์ PNG ใน Folder `outputs`

โค้ดนี้ใช้ **Transfer Learning** ต่างจากตัวอย่างที่สร้างเครือข่ายแบบ VGG ขึ้นมาฝึกใหม่ โดยฝึกเฉพาะ Classifier ที่เพิ่มเข้าไป ส่วน Weight ของ Backbone คงเดิม

## การเตรียมภาพ

โมเดลรับภาพขนาด **224 × 224 × 3** โดย 3 คือช่องสี RGB ตัวโหลดภาพเก็บค่าพิกเซลในช่วง 0–255 แล้วปรับค่าภายในโมเดลให้เหมาะกับแต่ละ Backbone

| Model | การปรับค่าภาพ |
|---|---|
| VGG16 | เปลี่ยน RGB เป็น BGR และลบค่าเฉลี่ยแต่ละช่องสีของ ImageNet |
| ResNet101 | เปลี่ยน RGB เป็น BGR และลบค่าเฉลี่ยแต่ละช่องสีของ ImageNet |
| MobileNetV2 | ปรับค่าพิกเซลให้อยู่ประมาณ −1 ถึง 1 |
| EfficientNetB0 | ใช้การปรับค่าที่รวมอยู่ใน Backbone |

ระหว่างฝึกใช้ Random Horizontal Flip และ Random Rotation เล็กน้อยเพื่อเพิ่มความหลากหลายของภาพ ขั้นตอนเหล่านี้ไม่ทำงานตอนประเมินและทำนาย

## Train, Validation และ Test

แบ่งข้อมูลโดยประมาณ:

```text
Train      = 60%
Validation = 20%
Test       = 20%
```

- **Train:** ใช้เรียนรู้ Weight ของ Classifier ใหม่
- **Validation:** ใช้เปรียบเทียบและเลือก Configuration ที่ดีที่สุด
- **Test:** ใช้ประเมินกับภาพที่ไม่ได้ใช้ฝึกหรือเลือกโมเดล

ใช้ Stratification เพื่อรักษาสัดส่วน Class และ `random_state=42` เพื่อแบ่งได้ชุดเดิม จำนวนภาพจริงอาจคลาดเคลื่อนจากเปอร์เซ็นต์เล็กน้อยเพราะต้องปัดเป็นจำนวนเต็ม ทุกการทดลองในการรันเดียวกันใช้ข้อมูลแบ่งชุดเดียวกัน

## DCNN ที่ใช้

| Model | แนวคิดหลัก |
|---|---|
| VGG16 | เรียง Convolution หลายชั้นต่อกัน โครงสร้างเข้าใจง่าย |
| ResNet101 | เครือข่ายลึกที่มีทางลัดแบบ Residual Connection |
| MobileNetV2 | ลดการคำนวณด้วยโครงสร้าง Convolution และ Inverted Residual |
| EfficientNetB0 | ออกแบบให้ความลึก ความกว้าง และความละเอียดภาพสมดุลกัน |

ทุก Backbone ใช้ Weight จาก ImageNet และถอด Classifier เดิมด้วย `include_top=False` แล้วเพิ่ม Classifier รูปแบบเดียวกัน:

```text
Input Image: 224 × 224 × 3
  ↓
Random Flip + Random Rotation
  ↓
Model-Specific Preprocessing
  ↓
Frozen Pretrained DCNN Backbone
  ↓
Global Average Pooling
  ↓
Dense: 128 neurons + ReLU
  ↓
Dropout: 0.3
  ↓
Dense: 1 neuron + Sigmoid
  ↓
Prediction: BMW / Honda
```

ใช้ Adam, Learning Rate `0.001`, Binary Cross-Entropy และ Batch Size `8` ตัวโหลดเรียงชื่อ Class ทำให้ BMW เป็น Label 0 และ Honda เป็น Label 1 ค่า Sigmoid ตั้งแต่ `0.5` ขึ้นไปทำนายเป็น Honda ค่าต่ำกว่านั้นทำนายเป็น BMW

## Epochs ที่ใช้

แต่ละโมเดลทดลอง **5, 10 และ 20 Epochs** รวม **12 การทดลอง** เมื่อรันทั้งหมด

ทุกการทดลองเริ่มจาก Backbone ที่ฝึกมาแล้วและ Classifier ที่เริ่ม Weight ใหม่ การทดลอง 20 Epochs ไม่ได้ฝึกต่อจากการทดลอง 10 Epochs และไม่มี Early Stopping จึงฝึกครบจำนวนรอบที่กำหนด

Epoch คือการผ่านข้อมูล Train ครบหนึ่งรอบ เพิ่ม Epochs ไม่รับประกันว่า Validation/Test Accuracy จะสูงขึ้นเสมอ

## การประเมินผลและเลือก Best Model

Accuracy คือสัดส่วนภาพที่ทำนายถูก ส่วน Loss คือค่าความผิดพลาด โดยทั่วไปยิ่งต่ำยิ่งดี

แสดง Test Accuracy และ Classification Report ซึ่งมี Precision, Recall และ F1-score ในหน้าต่างที่รัน พร้อมตาราง Validation Accuracy/Loss, Test Accuracy/Loss และเวลาฝึก เวลาที่แสดงนับเฉพาะ `model.fit()` ไม่รวมการดาวน์โหลด Weight สร้างโมเดล หรือประเมินผล

เลือก Best Model จาก Validation Accuracy สูงสุด หากเท่ากันเลือก Validation Loss ต่ำสุด โดยไม่ใช้ Test Accuracy ในการตัดสิน

`best_model.png` เป็น **ภาพสรุปการทดลองที่ดีที่สุด** ไม่ใช่ไฟล์โมเดลที่โหลดกลับมาทำนายได้ โค้ดบันทึกผลเป็น PNG เท่านั้น ไม่บันทึก Weight, JSON หรือ CSV

Comparison และ Best Model รวมเฉพาะการทดลองที่เสร็จใน **การรันครั้งปัจจุบัน** ไม่รวมผลจากการรันคนละครั้ง เมื่อรันใหม่ ภาพสรุปและภาพการทดลองที่มี Path เดิมจะถูกเขียนทับ

## Prediction

แสดงภาพจาก Test สูงสุด 6 ภาพ พร้อม Class จริง ผลทำนาย และ Confidence:

```text
Pred: BMW (92%)
True: BMW
```

ข้อความสีเขียวคือทำนายถูก สีแดงคือทำนายผิด Confidence เป็นค่าจากโมเดล ไม่ใช่การรับประกันว่าคำตอบถูก

## ผลลัพธ์ที่ได้

```text
outputs/
├── VGG16/
│   ├── epochs_5/
│   │   ├── training_history.png
│   │   ├── confusion_matrix.png
│   │   └── prediction_sample.png
│   ├── epochs_10/    (ไฟล์ PNG 3 แบบเดียวกัน)
│   └── epochs_20/    (ไฟล์ PNG 3 แบบเดียวกัน)
├── ResNet101/        (Folder Epochs และไฟล์ PNG แบบเดียวกัน)
├── MobileNetV2/      (Folder Epochs และไฟล์ PNG แบบเดียวกัน)
├── EfficientNetB0/   (Folder Epochs และไฟล์ PNG แบบเดียวกัน)
├── comparison/
│   └── comparison.png
└── best_model/
    └── best_model.png
```

| ไฟล์ | หน้าที่ |
|---|---|
| training_history.png | Training/Validation Accuracy และ Loss ในแต่ละ Epoch |
| confusion_matrix.png | จำนวนที่ทำนายถูกและผิด แยกตาม Class |
| prediction_sample.png | ตัวอย่างภาพพร้อม Class จริง ผลทำนาย และ Confidence |
| comparison.png | กราฟ Test Accuracy ตาม Epochs และตาราง Validation Accuracy, Test Accuracy และเวลาฝึก |
| best_model.png | สรุปโมเดลและ Epochs ที่เลือก พร้อมผล Validation/Test และเวลาฝึก |

เมื่อรันค่าเริ่มต้นครบทั้งหมดจะได้ **38 PNG** ได้แก่ ภาพของ 12 การทดลองรวม 36 ภาพ และภาพสรุป 2 ภาพ

## หน้าที่ของไฟล์โค้ด

| ไฟล์ | หน้าที่ |
|---|---|
| data_loader.py | อ่าน Folder Class โหลดภาพ และข้ามไฟล์ภาพซ้ำ |
| preprocessing.py | แปลงภาพเป็น RGB และ Resize |
| split_data.py | เตรียมและแบ่ง Train, Validation และ Test |
| model_builder.py | เพิ่มการปรับค่าภาพ Augmentation และ Classifier ที่ใช้ร่วมกัน |
| vgg16_model.py | สร้างและรัน VGG16 |
| resnet101_model.py | สร้างและรัน ResNet101 |
| mobilenetv2_model.py | สร้างและรัน MobileNetV2 |
| efficientnetb0_model.py | สร้างและรัน EfficientNetB0 |
| main.py | ฝึกและประเมินโมเดลที่เลือก |
| evaluate.py | สร้างภาพ PNG 3 แบบของแต่ละการทดลอง |
| compare_results.py | สร้างภาพ Comparison และ Best Model |
| runall.py | รันหลายโมเดลและหลายจำนวน Epochs |

ตั้งค่า Image Size, Batch Size และ Epochs เริ่มต้นใน `main.py` โดยไม่ใช้ `config.py`

## วิธีรัน

ใช้ Anaconda Environment ที่มี TensorFlow, NumPy, scikit-learn, Matplotlib และ Pillow โดย Environment สำหรับ GPU บน Windows ที่ตั้งค่าไว้ใช้ Python 3.10 และ TensorFlow 2.10.1

เปิด Anaconda Prompt แล้วเข้า Environment และ Folder:

```bat
conda activate mrmai
cd /d C:\Users\ponga\Desktop\ML\ML_for_me\ML-08-DCNN
```

รันทั้งหมดด้วย Epochs เริ่มต้น:

```bat
python runall.py
```

หรือเลือกคำสั่งตามโมเดลที่ต้องการ แต่ละคำสั่งจะทดลอง 5, 10 และ 20 Epochs:

```bat
python vgg16_model.py
python resnet101_model.py
python mobilenetv2_model.py
python efficientnetb0_model.py
```

ระบุโมเดลและจำนวน Epochs เอง:

```bat
python main.py --model MobileNetV2 --epochs 5
```

เลือกหลายโมเดลและหลายจำนวน Epochs:

```bat
python runall.py --models MobileNetV2 EfficientNetB0 --epochs 5 10
```

การใช้ Backbone ครั้งแรกอาจต้องเชื่อมต่ออินเทอร์เน็ตเพื่อดาวน์โหลด ImageNet Weight ครั้งต่อไปใช้ Weight ที่ Cache ไว้ TensorFlow จะใช้ GPU ที่รองรับและตรวจพบโดยอัตโนมัติ โค้ดแสดงรายการ GPU และใช้ CPU หากไม่พบ GPU

## ข้อจำกัด

Dataset มีภาพน้อย ผลจาก Test ชุดเดียวจึงยังไม่บอกความแม่นยำกับภาพรถทั่วไป ควรใช้ผลจริงจากการรันในการเขียนรายงาน หากต้องการข้อสรุปที่น่าเชื่อถือขึ้นควรเพิ่มภาพที่หลากหลายและประเมินซ้ำ

การทดลองนี้เปรียบเทียบ Backbone ที่ Freeze ไว้และฝึก Classifier ใหม่ ไม่ได้ฝึก DCNN ทั้งหมดจากศูนย์หรือ Fine-tune Backbone

---

## English

Adapted from the instructor's Lab 08 examples in `Machine-Learning-Course-main/ML-08-DCNN/mini-proj`.
## Dataset

This project classifies car images into two classes: **BMW** and **Honda**. Folder names are used as class names.

```text
dataset/
├── BMW/
└── Honda/
```

The loader skips byte-identical duplicate images before splitting the dataset, without deleting the original files. This prevents identical copies from appearing in different subsets. Visually similar images with different file contents are not automatically detected.

## Workflow

1. Load images from the `dataset` folder.
2. Convert images to RGB and resize them to 224 × 224 pixels.
3. Split the data into training, validation, and test sets.
4. Create four DCNN architectures using pretrained ImageNet weights.
5. Freeze the pretrained backbone and train a new classifier for BMW/Honda.
6. Train each architecture for 5, 10, and 20 epochs.
7. Evaluate each experiment and save its training history, confusion matrix, and sample predictions.
8. Compare the experiments and select the best result using validation accuracy, with validation loss as the tie-breaker.
9. Save all report outputs as PNG images in `outputs`.

Unlike the course example, which constructs a VGG-style network from scratch, this project uses **transfer learning** with pretrained backbones. Only the new classifier is trained; the backbone weights remain frozen.

## Image Preparation

Each model receives images with shape **224 × 224 × 3**, where 3 represents the RGB channels. The loader keeps pixel values in the 0–255 range. Model-specific preprocessing takes place inside the model:

| Model | Preprocessing |
|---|---|
| VGG16 | Convert RGB to BGR and subtract ImageNet channel means |
| ResNet101 | Convert RGB to BGR and subtract ImageNet channel means |
| MobileNetV2 | Scale pixel values to approximately −1 to 1 |
| EfficientNetB0 | Use the preprocessing included in the pretrained backbone |

Training augmentation applies random horizontal flips and small rotations. Augmentation is disabled during evaluation and prediction.

## Training, Validation, and Test Sets

The approximate split is:

```text
Training   = 60%
Validation = 20%
Test       = 20%
```

- **Training:** learn the new classifier weights.
- **Validation:** compare experiments and select the best model configuration.
- **Test:** measure performance on images not used for training or model selection.

The split uses stratification to preserve class proportions and `random_state=42` for reproducibility. Actual counts may differ slightly from these percentages because image counts must be integers. All experiments within a run use the same split.

## DCNN Models

| Model | Main Architectural Idea |
|---|---|
| VGG16 | Stacked convolution layers with a simple sequential structure |
| ResNet101 | A deep network with residual shortcut connections |
| MobileNetV2 | Efficient convolutions with inverted residual blocks |
| EfficientNetB0 | A compact architecture designed to balance depth, width, and image resolution |

Every backbone uses ImageNet weights and removes its original classifier with `include_top=False`. A shared classifier design is added:

```text
Input Image: 224 × 224 × 3
  ↓
Random Flip + Random Rotation
  ↓
Model-Specific Preprocessing
  ↓
Frozen Pretrained DCNN Backbone
  ↓
Global Average Pooling
  ↓
Dense: 128 neurons + ReLU
  ↓
Dropout: 0.3
  ↓
Dense: 1 neuron + Sigmoid
  ↓
Prediction: BMW / Honda
```

The classifier uses Adam with a learning rate of `0.001`, binary cross-entropy loss, and a batch size of `8`. Class names are sorted by the loader: BMW is label 0 and Honda is label 1. A sigmoid output of at least `0.5` predicts Honda; otherwise, it predicts BMW.

## Epoch Experiments

Each model is trained for **5, 10, and 20 epochs**, giving **12 experiments** in a complete run.

Each experiment starts with the pretrained backbone and a newly initialized classifier. A 20-epoch experiment does not continue from the 10-epoch experiment. The code trains for the requested number of epochs without early stopping.

An epoch is one pass through the training set. Increasing the number of epochs does not guarantee higher validation or test accuracy.

## Evaluation and Best Model Selection

Accuracy is the proportion of correctly classified images. Loss measures prediction error; lower values are generally better.

The console displays test accuracy and a classification report containing precision, recall, and F1-score. It also prints a comparison of validation accuracy/loss, test accuracy/loss, and training time. Training time measures the `model.fit()` call only; it excludes weight downloads, model creation, and evaluation.

The best experiment is selected by:

1. Highest validation accuracy.
2. Lowest validation loss if validation accuracy is tied.

Test accuracy is reported for the selected experiment but is not used to select it.

`best_model.png` is a summary image of the selected experiment. It is **not a saved model** and cannot be loaded for later predictions. The code does not save model weights, JSON, or CSV files.

Comparison and best-model summaries include only experiments completed in the **current invocation**. Separate runs are not merged. A later run overwrites these summary images and any experiment images with the same output paths.

## Prediction Samples

Each experiment displays up to six test images with their true labels, predicted labels, and confidence values:

```text
Pred: BMW (92%)
True: BMW
```

Green text indicates a correct prediction; red text indicates an incorrect prediction. Confidence is a model output, not a guarantee that the prediction is correct.

## Output Files

```text
outputs/
├── VGG16/
│   ├── epochs_5/
│   │   ├── training_history.png
│   │   ├── confusion_matrix.png
│   │   └── prediction_sample.png
│   ├── epochs_10/    (same three PNG files)
│   └── epochs_20/    (same three PNG files)
├── ResNet101/        (same epoch folders and PNG files)
├── MobileNetV2/      (same epoch folders and PNG files)
├── EfficientNetB0/   (same epoch folders and PNG files)
├── comparison/
│   └── comparison.png
└── best_model/
    └── best_model.png
```

| File | Purpose |
|---|---|
| training_history.png | Training and validation accuracy/loss across epochs |
| confusion_matrix.png | Counts of correct and incorrect predictions for each class |
| prediction_sample.png | Test images with true labels, predictions, and confidence |
| comparison.png | Test accuracy by epoch count and a table of validation accuracy, test accuracy, and training time |
| best_model.png | Selected architecture, epoch count, validation/test scores, and training time |

A complete default run creates **38 PNG files**: 36 experiment images and 2 summary images.

## Source Files

| File | Responsibility |
|---|---|
| data_loader.py | Load class folders, read images, and skip identical duplicates |
| preprocessing.py | Convert images to RGB and resize them |
| split_data.py | Prepare the training, validation, and test subsets |
| model_builder.py | Add preprocessing, augmentation, and the shared classifier |
| vgg16_model.py | Build and run the VGG16 experiment |
| resnet101_model.py | Build and run the ResNet101 experiment |
| mobilenetv2_model.py | Build and run the MobileNetV2 experiment |
| efficientnetb0_model.py | Build and run the EfficientNetB0 experiment |
| main.py | Train and evaluate one selected architecture |
| evaluate.py | Generate the three PNG outputs for each experiment |
| compare_results.py | Generate comparison and best-model summary images |
| runall.py | Run multiple architectures and epoch counts |

Settings such as image size, batch size, and default epoch counts are defined in `main.py`. No separate `config.py` is required.

## How to Run

Use the configured Anaconda environment containing TensorFlow, NumPy, scikit-learn, Matplotlib, and Pillow. The current Windows GPU environment uses Python 3.10 and TensorFlow 2.10.1.

Open Anaconda Prompt and enter:

```bat
conda activate mrmai
cd /d C:\Users\ponga\Desktop\ML\ML_for_me\ML-08-DCNN
```

Run all four models with the default epoch counts:

```bat
python runall.py
```

Run one model with all three default epoch counts:

```bat
python vgg16_model.py
python resnet101_model.py
python mobilenetv2_model.py
python efficientnetb0_model.py
```

Each line above is a separate command for its corresponding model. Alternatively, specify a model and epoch count:

```bat
python main.py --model MobileNetV2 --epochs 5
```

Run a selected group of models and epoch counts:

```bat
python runall.py --models MobileNetV2 EfficientNetB0 --epochs 5 10
```

The first use of a backbone may download its ImageNet weights, requiring internet access. Later runs reuse the cached weights. TensorFlow uses an available supported GPU automatically; the code prints detected GPU devices and falls back to the CPU when none are detected.

## Limitations

The dataset is small, so results from one test split may vary substantially and do not establish accuracy on car images in general. Use actual results from your own run in the report. More diverse images and repeated evaluation would be needed for stronger conclusions.

This project compares frozen pretrained feature extractors with a newly trained classifier. It does not train the full DCNN from scratch or fine-tune the backbone.
