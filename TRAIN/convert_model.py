import tensorflow as tf

# Load mô hình đã huấn luyện
model = tf.keras.models.load_model('eye_model.h5')

# Chuyển sang định dạng TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Lưu mô hình dưới dạng .tflite
with open('eye_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("✔ Chuyển đổi mô hình sang TensorFlow Lite hoàn tất!")
