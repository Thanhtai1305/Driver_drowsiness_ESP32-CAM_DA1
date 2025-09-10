# 🚗 Driver Drowsiness Detection System using ESP32-CAM  

## 📖 Introduction  
This project implements a **driver drowsiness detection system** using only the **ESP32-CAM** module.  
The system is able to:  
- Detect **eye state (open/closed)** using a trained deep learning model (TensorFlow → TensorFlow Lite → deployed on ESP32-CAM).  
- Detect **face orientation (looking forward / turned away / tilted)** using classical image processing techniques (Gaussian blur, Sobel filter, angle calculation, etc.).  

Captured images are processed on the ESP32-CAM, results are displayed in the Serial Monitor, and images are saved to an SD card in `.pgm` format for further analysis.  

---

## ⚙️ System Functions  
1. **Eye Detection (Open/Closed)**  
   - A CNN model is trained on a Kaggle dataset of open/closed eyes.  
   - The trained model is optimized and deployed to ESP32-CAM.  
   - The ESP32-CAM runs inference in real-time to classify whether the driver’s eyes are open or closed.  

2. **Face Orientation Detection**  
   - Implemented with image processing techniques:  
     - Gaussian blur → noise reduction  
     - Sobel filter → edge detection  
     - Angle estimation → calculate face tilt and detect if the driver is looking forward or turning away  
   - Implemented in `nhan_dien_buon_ngu.ino`.  

3. **Data Output**  
   - Results shown in Serial Monitor.  
   - Cropped eye/face images are saved to SD card in `.pgm` format.  
   - Images can later be visualized using Python and OpenCV.  

---

## 🔀 Implemented Cases  ((case-1, case-2 and case-3) branches in this project)

### 🟢 Case-1: Eye detection without AI + save images to SD () 
- Implemented a **non-AI method** for eye detection using image processing techniques.  
- Captured frames are stored on the **SD card in `.pgm` format**.  
- Added `Binary images/` folder for OpenCV visualization in Python.  

### 🟢 Case-2: AI-based eye & face recognition with fixed eye region  
- Integrated **AI model (`eye_model.h`)** built with TensorFlow Lite.  
- Eye detection is performed on a **fixed eye region** of the face.  
- Combined with face detection logic for improved accuracy.  

### 🟢 Case-3: AI + face orientation with tilted head handling  
- Used **AI eye detection** together with **image processing filters** (Gaussian, Sobel).  
- Added **face tilt angle calculation** to handle cases where the driver’s head is not perfectly straight.  
- Extended functionality in `nhan_dien_buon_ngu.ino ` and test in file `nhan_dien_buon_ngu_SD.ino`.  

---

## 🧠 AI Model Training Workflow  
The AI workflow for **eye state recognition** is as follows:  

1. **Dataset**  
   - Kaggle dataset of open/closed eyes.  

2. **Preprocessing** (`preprocessed.py`)  
   - Image resizing, grayscale conversion, normalization.  

3. **Model Training** (`train.py`)  
   - Convolutional Neural Network (CNN) built with TensorFlow.  
   - Output: `eye_model.h5`.  

4. **Model Conversion**  
   - `convert_model.py`: Convert `.h5` → `.tflite`.  
   - `convert_to_c_array.py`: Convert `.tflite` → `.h` C header file.  

5. **Deployment on ESP32-CAM**  
   - The final `eye_model.h` is included in the ESP32-CAM sketch.  
   - Inference runs with **TensorFlow Lite for Microcontrollers**.  

---

## 📂 Project Structure  
```
Driver_drowsiness_ESP32-CAM_DA1/
│-- CodeC/                  
│   ├── nhan_dien_mat.ino          # v1: basic eye + face detection
│   ├── nhan_dien_buon_ngu.ino     # v2: improved, handles face tilt
│   └── eye_model.h                # trained AI model in C array format
│
│-- TRAIN/                
│   ├── preprocessed.py             # dataset preprocessing
│   ├── train.py                    # CNN model training
│   ├── convert_model.py            # .h5 → .tflite
│   ├── convert_to_c_array.py       # .tflite → .h
│   └── dataset/                    # Kaggle dataset (open/closed eyes)
│
│-- nhan_dien_buon_ngu_SD/          # saved images from ESP32-CAM
│   └── *.pgm
│
│-- Binary images/                  # OpenCV visualization of .pgm
│
└── README.md                       # project documentation
```

---

## 🛠 Hardware & Software Requirements  
- **Hardware**  
  - ESP32-CAM (AI Thinker, OV2640 camera)  
  - MicroSD card  
  - Power supply (5V)  

- **Software**  
  - Arduino IDE or PlatformIO  
  - TensorFlow & TensorFlow Lite  
  - OpenCV (for post-analysis of `.pgm` images)  

- **Arduino Libraries**  
  - TensorFlow Lite for Microcontrollers  
  - ESP32 camera library  
  - SD card library  

---

## 📜 Usage  
1. Upload `nhan_dien_buon_ngu.ino` to ESP32-CAM.  
2. Open Serial Monitor to observe detection results.  
3. Captured `.pgm` images are automatically stored on SD card.  
4. Use Python + OpenCV to visualize saved images for verification.  

---

## 📌 Future Work  
- Add buzzer or alarm system for real-time driver alerts.  
- Optimize CNN model for higher accuracy with larger datasets.  
- Improve memory usage for smoother inference on ESP32-CAM.  
- Extend system with IoT integration for remote monitoring.  

---

## 📺 Demo
[Youtube Demo](https://www.youtube.com/watch?v=XHIIgUVYBc8&list=PLRiJxzEnUSjPVM7qvBsmr5u5ssI-_G2mH&index=2)

You can watch the demo video of the system by clicking on the link above.


---

## 👤 Author  
- **Trần Nguyễn Thành Tài**  
- Major: Computer Engineering  
- Ho Chi Minh City University of Technology and Education (HCMUTE)  
- Email: thanhtait4025@gmail.com  
