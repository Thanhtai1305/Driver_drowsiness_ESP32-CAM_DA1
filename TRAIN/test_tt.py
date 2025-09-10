import cv2
import dlib
import numpy as np
from tensorflow.keras.models import load_model

# Load mô hình nhận diện mắt
MODEL_PATH = "eye_model.h5"  # F ile mô hình bạn đã train
model = load_model(MODEL_PATH)

# Load bộ nhận diện khuôn mặt của dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("d:/DO AN 1/TRAIN/shape_predictor_68_face_landmarks.dat")

# Hàm tính toán góc mặt
def get_head_tilt_ratio(landmarks):
    left_eye = landmarks[36]  # Góc ngoài mắt trái
    right_eye = landmarks[45] # Góc ngoài mắt phải
    nose = landmarks[30]      # Chóp mũi

    # Tính toán độ nghiêng bằng khoảng cách Euclidean
    eye_dist = np.linalg.norm(left_eye - right_eye)
    nose_eye_dist = np.linalg.norm(nose - ((left_eye + right_eye) / 2))

    return nose_eye_dist / eye_dist  # Tỉ lệ đầu nghiêng

# Biến đếm số lần mắt nhắm
closed_eye_counter = 0
closed_eye_threshold = 10  # Nếu mắt nhắm liên tục 10 frame sẽ cảnh báo

# Khởi động webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

        ratio = get_head_tilt_ratio(landmarks)

        # Xác định vùng mắt trái & mắt phải
        left_eye_region = gray[landmarks[37][1]:landmarks[41][1], landmarks[36][0]:landmarks[39][0]]
        right_eye_region = gray[landmarks[43][1]:landmarks[47][1], landmarks[42][0]:landmarks[45][0]]

        # Kiểm tra nếu vùng mắt đủ lớn
        if left_eye_region.shape[0] > 0 and left_eye_region.shape[1] > 0 and \
           right_eye_region.shape[0] > 0 and right_eye_region.shape[1] > 0:

            # Resize về (24,24) để đưa vào model
            left_eye_resized = cv2.resize(left_eye_region, (24, 24)) / 255.0
            right_eye_resized = cv2.resize(right_eye_region, (24, 24)) / 255.0

            # Dự đoán mắt mở/nhắm
            left_eye_pred = model.predict(left_eye_resized.reshape(1, 24, 24, 1))
            right_eye_pred = model.predict(right_eye_resized.reshape(1, 24, 24, 1))

            # Xác định trạng thái mắt
            is_left_closed = left_eye_pred < 0.5
            is_right_closed = right_eye_pred < 0.5

            if is_left_closed and is_right_closed:
                closed_eye_counter += 1
                eye_label = "Closed"
                eye_color = (0, 0, 255)  # Đỏ
            else:
                closed_eye_counter = 0
                eye_label = "Open"
                eye_color = (0, 255, 0)  # Xanh

            cv2.putText(frame, eye_label, (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, eye_color, 2)

        # Nếu mắt nhắm liên tục hoặc đầu nghiêng quá nhiều -> Cảnh báo
        if closed_eye_counter > closed_eye_threshold or ratio > 0.5:
            alert_text = "DROWSY / DISTRACTED!"
            alert_color = (0, 0, 255)  # Đỏ
        else:
            alert_text = "FOCUSED"
            alert_color = (0, 255, 0)  # Xanh

        cv2.putText(frame, alert_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, alert_color, 2)

    cv2.imshow("Driver Monitoring", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
