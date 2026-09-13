# LAB 6 - Neural Network (NN)

เวอร์ชันนี้ปรับจากตัวอย่างอาจารย์ให้ใช้ Dataset แมว 5 สายพันธุ์เดิม

- abyssinian
- cyprus
- lykoi
- donskoy
- chausie

แม้ใน `data-animal` จะมีสายพันธุ์อื่น โปรแกรมจะอ่านเฉพาะ 5 คลาสนี้

## โครงสร้างโฟลเดอร์

LAB06/
├── data-animal/
│   ├── abyssinian/
│   ├── cyprus/
│   ├── lykoi/
│   ├── donskoy/
│   ├── chausie/
│   └── ...โฟลเดอร์อื่นมีได้ แต่จะไม่ถูกใช้
├── main.py
├── data_loader.py
├── preprocessing.py
├── split_data.py
├── nn_model.py
├── evaluate.py
└── test_nn.py

## การทดลองที่ทำ

เพื่อให้ตรงใบงานที่กำหนดให้เปรียบเทียบทั้งจำนวน epochs
และโครงสร้าง Neural Network โปรแกรมจะทดลอง 3 แบบ

1. NN_A
   - Hidden layer: [128]
   - Epochs: 10

2. NN_B
   - Hidden layers: [256, 128]
   - Epochs: 20

3. NN_C
   - Hidden layers: [256, 128, 64]
   - Epochs: 30

EarlyStopping อาจทำให้จำนวน epoch ที่ train จริงน้อยกว่าค่าที่กำหนด
ถ้า validation loss ไม่ดีขึ้นต่อเนื่อง

## Dataset split

จาก 1,000 รูปโดยประมาณ

- Train = 700
- Validation = 100
- Test = 200

ใช้ stratify เพื่อรักษาสัดส่วนแต่ละสายพันธุ์

## ติดตั้ง Library

pip install -r requirements.txt

## เริ่ม Train

python main.py

## Output

จะสร้างในโฟลเดอร์ `outputs`

- NN_A.keras
- NN_B.keras
- NN_C.keras
- NN_A_history.png
- NN_B_history.png
- NN_C_history.png
- NN_A_confusion_matrix.png
- NN_B_confusion_matrix.png
- NN_C_confusion_matrix.png
- nn_comparison.csv
- nn_accuracy_comparison.png
- best_model.txt
- X_test.npy
- y_test.npy
- classes.json

## ทดสอบ Prediction

หลังจากรัน main.py เสร็จ

python test_nn.py

โปรแกรมจะเลือก model ที่ Accuracy สูงที่สุด และสุ่ม 5 รูปจาก Test set
เพื่อแสดง Predicted class, True class และ Confidence
