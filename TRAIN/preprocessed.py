import cv2
import os
import numpy as np

def process_images(input_folder, output_folder, size=(24, 24)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # Tạo thư mục nếu chưa tồn tại

    for img_name in os.listdir(input_folder):
        img_path = os.path.join(input_folder, img_name)
        
        # Đọc ảnh, chuyển sang ảnh xám và resize
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Lỗi khi đọc ảnh: {img_name}")
            continue  # Bỏ qua ảnh lỗi
        
        img = cv2.resize(img, size)

        # Lưu ảnh đã xử lý
        save_path = os.path.join(output_folder, img_name)
        cv2.imwrite(save_path, img)

# Xử lý dữ liệu cho cả hai thư mục "open" và "closed"
process_images('dataset/open', 'processed_dataset/open')
process_images('dataset/closed', 'processed_dataset/closed')

print("✔ Xử lý dữ liệu hoàn tất! Ảnh đã được resize và lưu trong 'processed_dataset/'")
