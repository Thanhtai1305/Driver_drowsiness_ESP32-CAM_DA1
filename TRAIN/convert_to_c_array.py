import numpy as np

# Đọc mô hình đã chuyển đổi
with open("eye_model.tflite", "rb") as f:
    tflite_model = f.read()

# Chuyển mô hình sang dạng mảng C
c_model = ", ".join(str(x) for x in tflite_model)

# Tạo file C header
with open("eye_model.h", "w") as f:
    f.write(f"#ifndef EYE_MODEL_H\n#define EYE_MODEL_H\n\n")
    f.write(f"const unsigned char eye_model[] = {{ {c_model} }};\n")
    f.write(f"const unsigned int eye_model_len = {len(tflite_model)};\n\n")
    f.write(f"#endif // EYE_MODEL_H\n")

print("✔ Chuyển đổi mô hình sang C array hoàn tất!")
