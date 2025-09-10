import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load mô hình đã train (lát nữa ta sẽ train riêng)
MODEL_PATH = "eye_model.h5"  # Tên file mô hình sau khi train
model = load_model(MODEL_PATH)

# Load Haar Cascade để nhận diện khuôn mặt & mắt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Mở webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        for (ex, ey, ew, eh) in eyes:
            eye_roi = roi_gray[ey:ey+eh, ex:ex+ew]
            eye_roi = cv2.resize(eye_roi, (24, 24))  # Resize về đúng input của model
            eye_roi = eye_roi / 255.0  # Chuẩn hóa ảnh
            eye_roi = eye_roi.reshape(1, 24, 24, 1)  # Thêm batch dimension

            # Dự đoán mắt mở hay nhắm
            prediction = model.predict(eye_roi)
            label = "Open" if prediction > 0.5 else "Closed"
            color = (0, 255, 0) if label == "Open" else (0, 0, 255)

            # Hiển thị lên frame
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Eye Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
