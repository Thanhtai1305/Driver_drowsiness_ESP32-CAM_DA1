#define CAMERA_MODEL_AI_THINKER
#include "eye_model.h"
#include <TensorFlowLite_ESP32.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include "esp_camera.h"

// Cấu hình ESP32-CAM
#define PWDN_GPIO   32
#define RESET_GPIO  15
#define XCLK_GPIO    0
#define SIOD_GPIO   26
#define SIOC_GPIO   27
#define Y9_GPIO     35
#define Y8_GPIO     34
#define Y7_GPIO     39
#define Y6_GPIO     36
#define Y5_GPIO     21
#define Y4_GPIO     19
#define Y3_GPIO     18
#define Y2_GPIO     5
#define VSYNC_GPIO  25
#define HREF_GPIO   23
#define PCLK_GPIO   22
#define FLASH_GPIO  4


// Tensor Arena
const int tensor_arena_size = 300 * 1024;
uint8_t* tensor_arena;

// Khởi tạo TensorFlow Lite
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter; // Sửa lỗi cú pháp
tflite::MicroMutableOpResolver<10> resolver;

// Khởi tạo mô hình AI
const tflite::Model* model = tflite::GetModel(eye_model);
tflite::MicroInterpreter* interpreter;

// Lưu trữ lịch sử xác suất
static float prob_history[3] = {0};
static int history_index = 0;

// Buffer tạm để xử lý ảnh
uint8_t processedImage[24 * 24];

// Hàm tính độ sáng trung bình của một vùng
float calculateRegionBrightness(uint8_t *src, int width, int height, int start_x, int end_x, int start_y, int end_y) {
    float sum = 0;
    int count = 0;
    for (int y = start_y; y < end_y; y++) {
        for (int x = start_x; x < end_x; x++) {
            sum += src[y * width + x];
            count++;
        }
    }
    return sum / count;
}

// Hàm phát hiện hướng mặt
void detectFaceDirection(camera_fb_t *fb) {
    float left_brightness = calculateRegionBrightness(fb->buf, 96, 96, 0, 48, 0, 96);
    float right_brightness = calculateRegionBrightness(fb->buf, 96, 96, 48, 96, 0, 96);

    float brightness_diff = left_brightness - right_brightness;
    float threshold = 20.0f;

    Serial.print("Độ sáng nửa trái: ");
    Serial.println(left_brightness);
    Serial.print("Độ sáng nửa phải: ");
    Serial.println(right_brightness);

    if (brightness_diff > threshold || brightness_diff < -threshold) {
        Serial.println("⚠️ Mặt quay sang chỗ khác!");
        
    } else {
        Serial.println("✅ Mặt hướng về phía trước.");
        
    }
}

// Hàm làm mịn ảnh bằng bộ lọc Gaussian 3x3
void smoothImage(uint8_t *src, uint8_t *dst, int width, int height) {
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int sum = 0;
            sum += src[(y - 1) * width + (x - 1)] * 1;
            sum += src[(y - 1) * width + x] * 2;
            sum += src[(y - 1) * width + (x + 1)] * 1;
            sum += src[y * width + (x - 1)] * 2;
            sum += src[y * width + x] * 4;
            sum += src[y * width + (x + 1)] * 2;
            sum += src[(y + 1) * width + (x - 1)] * 1;
            sum += src[(y + 1) * width + x] * 2;
            sum += src[(y + 1) * width + (x + 1)] * 1;
            dst[y * width + x] = sum / 16;
        }
    }
    for (int x = 0; x < width; x++) {
        dst[x] = src[x];
        dst[(height - 1) * width + x] = src[(height - 1) * width + x];
    }
    for (int y = 0; y < height; y++) {
        dst[y * width] = src[y * width];
        dst[y * width + (width - 1)] = src[y * width + (width - 1)];
    }
}

// Hàm cắt và resize vùng mắt
void cropAndResize(uint8_t *src, uint8_t *dst, int src_width, int src_height, int crop_x, int crop_y, int crop_size, int dst_size) {
    uint8_t cropped[crop_size * crop_size];
    for (int y = 0; y < crop_size; y++) {
        for (int x = 0; x < crop_size; x++) {
            int src_y = crop_y + y;
            int src_x = crop_x + x;
            if (src_x >= 0 && src_x < src_width && src_y >= 0 && src_y < src_height) {
                cropped[y * crop_size + x] = src[src_y * src_width + src_x];
            } else {
                cropped[y * crop_size + x] = 0;
            }
        }
    }
    for (int y = 0; y < dst_size; y++) {
        for (int x = 0; x < dst_size; x++) {
            int src_x = x * crop_size / dst_size;
            int src_y = y * crop_size / dst_size;
            dst[y * dst_size + x] = cropped[src_y * crop_size + src_x];
        }
    }
}

// Hàm xử lý ảnh: Cắt vùng mắt, làm mịn, và chuẩn hóa
void processImage(camera_fb_t *fb, TfLiteTensor* input) {
    int crop_size = 24;
    int dst_size = 24;

    uint8_t left_eye[24 * 24];
    cropAndResize(fb->buf, left_eye, 96, 96, 24, 24, crop_size, dst_size);

    uint8_t right_eye[24 * 24];
    cropAndResize(fb->buf, right_eye, 96, 96, 48, 24, crop_size, dst_size);

    uint8_t tempImage[24 * 24];
    memcpy(tempImage, left_eye, 24 * 24);
    smoothImage(tempImage, left_eye, 24, 24);
    memcpy(tempImage, right_eye, 24 * 24);
    smoothImage(tempImage, right_eye, 24, 24);

    for (int i = 0; i < 24 * 24; i++) {
        processedImage[i] = (left_eye[i] + right_eye[i]) / 2;
    }

    Serial.println("Một số giá trị pixel sau xử lý:");
    for (int i = 0; i < 10; i++) {
        Serial.print(processedImage[i]);
        Serial.print(" ");
    }
    Serial.println();

    for (int i = 0; i < 24 * 24; i++) {
        input->data.f[i] = (float)processedImage[i] / 255.0f;
    }
}

// Hàm chạy mô hình TensorFlow Lite
float runModel() {
    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("❌ Lỗi khi chạy mô hình!");
        return -1.0f;
    }
    TfLiteTensor* output = interpreter->output(0);
    return output->data.f[0];
}

void setup() {
    Serial.begin(115200);
    Serial.println("Khởi động ESP32-CAM...");

    pinMode(FLASH_GPIO, OUTPUT);
    digitalWrite(FLASH_GPIO, LOW);

    if (psramFound()) {
        Serial.println("PSRAM found! Total PSRAM: " + String(ESP.getPsramSize()) + " bytes");
        Serial.printf("PSRAM còn lại trước khi cấp phát: %d bytes\n", ESP.getFreePsram());
    } else {
        Serial.println("No PSRAM found.");
        delay(1000);
        ESP.restart();
    }

    tensor_arena = (uint8_t*)ps_malloc(tensor_arena_size);
    if (tensor_arena == nullptr) {
        Serial.println("❌ Failed to allocate tensor arena in PSRAM!");
        delay(1000);
        ESP.restart();
    }
    Serial.printf("PSRAM còn lại sau khi cấp phát tensor arena: %d bytes\n", ESP.getFreePsram());

    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddAveragePool2D();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddRelu();
    resolver.AddReshape();
    resolver.AddLogistic();

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, tensor_arena_size, error_reporter);
    interpreter = &static_interpreter;

    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = Y2_GPIO;
    config.pin_d1 = Y3_GPIO;
    config.pin_d2 = Y4_GPIO;
    config.pin_d3 = Y5_GPIO;
    config.pin_d4 = Y6_GPIO;
    config.pin_d5 = Y7_GPIO;
    config.pin_d6 = Y8_GPIO;
    config.pin_d7 = Y9_GPIO;
    config.pin_xclk = XCLK_GPIO;
    config.pin_pclk = PCLK_GPIO;
    config.pin_vsync = VSYNC_GPIO;
    config.pin_href = HREF_GPIO;
    config.pin_sscb_sda = SIOD_GPIO;
    config.pin_sscb_scl = SIOC_GPIO;
    config.pin_pwdn = PWDN_GPIO;
    config.pin_reset = RESET_GPIO;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_GRAYSCALE;
    config.frame_size = FRAMESIZE_96X96;
    config.fb_count = 1;

    Serial.println("Kiểm tra trạng thái camera...");
    if (digitalRead(PWDN_GPIO) == HIGH) {
        Serial.println("Camera đang ở trạng thái tắt nguồn (power down).");
    } else {
        Serial.println("Camera đang ở trạng thái bật nguồn.");
    }
    Serial.printf("Tần số XCLK: %d Hz\n", config.xclk_freq_hz);

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("❌ Lỗi khởi động camera: 0x%x\n", err);
        delay(1000);
        ESP.restart();
    }
    Serial.println("✅ Camera khởi động thành công!");

    sensor_t *s = esp_camera_sensor_get();
    if (s == NULL) {
        Serial.println("❌ Không tìm thấy cảm biến camera!");
        delay(1000);
        ESP.restart();
    }
    Serial.printf("Cảm biến camera: %s\n", s->id.PID == OV2640_PID ? "OV2640" : "Unknown");

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("❌ Lỗi cấp phát bộ nhớ cho mô hình AI!");
        delay(1000);
        ESP.restart();
    }
    Serial.println("✅ TensorFlow Lite sẵn sàng!");

    TfLiteTensor* input = interpreter->input(0);
    Serial.print("Input shape: ");
    Serial.print(input->dims->data[0]);
    for (int i = 1; i < input->dims->size; i++) {
        Serial.print(" x ");
        Serial.print(input->dims->data[i]);
    }
    Serial.println();
}

void loop() {
    digitalWrite(FLASH_GPIO, HIGH);
    delay(100);
    camera_fb_t *fb = esp_camera_fb_get();
    digitalWrite(FLASH_GPIO, LOW);

    if (!fb) {
        Serial.println("❌ Không thể chụp ảnh!");
        return;
    }
    Serial.println("📷 Ảnh đã chụp, đang xử lý...");

    // Phát hiện hướng mặt 
    detectFaceDirection(fb);

    TfLiteTensor* input = interpreter->input(0);
    processImage(fb, input);

    float eye_open_prob = runModel();
    if (eye_open_prob < 0) {
        esp_camera_fb_return(fb);
        return;
    }
    Serial.print("Xác suất thô: ");
    Serial.println(eye_open_prob);

    prob_history[history_index] = eye_open_prob;
    history_index = (history_index + 1) % 3;
    float avg_prob = 0;
    for (int i = 0; i < 3; i++) {
        avg_prob += prob_history[i];
    }
    avg_prob /= 3;
    Serial.print("Xác suất trung bình: ");
    Serial.println(avg_prob);

    //  Trạng thái mắt
    if (avg_prob > 0.5) {
        Serial.println("👀 Mắt đang mở");
        
    } else {
        Serial.println("😴 Mắt đang nhắm");
        
    }

    esp_camera_fb_return(fb);
    delay(500);
}