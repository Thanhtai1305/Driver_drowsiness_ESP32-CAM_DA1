#define CAMERA_MODEL_AI_THINKER
#include "eye_model.h"
#include <TensorFlowLite_ESP32.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include "esp_camera.h"
#include "FS.h"
#include "SD_MMC.h"
#include <math.h>

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
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
tflite::MicroMutableOpResolver<10> resolver;

// Khởi tạo mô hình AI
const tflite::Model* model = tflite::GetModel(eye_model);
tflite::MicroInterpreter* interpreter;

// Lưu trữ lịch sử xác suất
static float prob_history[3] = {0};
static int history_index = 0;

// Biến đếm để tạo tên file duy nhất
static unsigned long file_counter = 0;

// Buffer toàn cục để xử lý ảnh
uint8_t processedLeftEye[24 * 24];  // Buffer cho ảnh mắt trái đã xử lý
uint8_t processedRightEye[24 * 24]; // Buffer cho ảnh mắt phải đã xử lý
uint8_t workingImage[96 * 96];
uint8_t tempImage[96 * 96];
uint8_t croppedEye[24 * 24];
uint8_t edgeImage[96 * 96];
uint8_t equalizedImage[96 * 96]; // Buffer cho ảnh sau khi cân bằng histogram

// Cấu trúc để lưu ứng viên mắt
struct EyeCandidate {
    int x;
    int y;
    float edge_value;
};

// Khai báo prototype của các hàm
float runModel();
bool initSDCard();
void saveImageAsPGM(const char* filename, uint8_t *image_data, int width, int height);
String generateUniqueFilename(const char* prefix);

// Hàm tạo tên file duy nhất dựa trên biến đếm
String generateUniqueFilename(const char* prefix) {
    file_counter++;
    return String(prefix) + "_" + String(file_counter) + ".pgm";
}

// Hàm tính độ sáng trung bình của một vùng
float calculateRegionBrightness(uint8_t *src, int width, int height, int start_x, int end_x, int start_y, int end_y) {
    float sum = 0;
    int count = 0;
    for (int y = start_y; y < end_y; y++) {
        for (int x = start_x; x < end_x; x++) {
            if (y < height && x < width) {
                sum += src[y * width + x];
                count++;
            }
        }
    }
    return (count > 0) ? sum / count : 0;
}

// Hàm phát hiện hướng mặt (trả về bool)
bool detectFaceDirection(camera_fb_t *fb) {
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
        return false; // Mặt quay đi
    } else {
        Serial.println("✅ Mặt hướng về phía trước.");
        return true; // Mặt nhìn thẳng
    }
}

// Hàm cân bằng histogram để tăng độ tương phản
void equalizeHistogram(uint8_t *src, uint8_t *dst, int width, int height) {
    int histogram[256] = {0};
    for (int i = 0; i < width * height; i++) {
        histogram[src[i]]++;
    }

    int cdf[256] = {0};
    cdf[0] = histogram[0];
    for (int i = 1; i < 256; i++) {
        cdf[i] = cdf[i - 1] + histogram[i];
    }

    int cdf_min = cdf[0];
    for (int i = 1; i < 256; i++) {
        if (cdf[i] != 0) {
            cdf_min = cdf[i];
            break;
        }
    }
    float scale = 255.0f / (width * height - cdf_min);
    for (int i = 0; i < width * height; i++) {
        if (src[i] == 0) {
            dst[i] = 0;
        } else {
            dst[i] = (uint8_t)((cdf[src[i]] - cdf_min) * scale);
        }
    }
}

// Hàm tính độ tương phản của một vùng
float calculateRegionContrast(uint8_t *src, int width, int height, int start_x, int end_x, int start_y, int end_y) {
    float sum = 0;
    float sum_sq = 0;
    int count = 0;
    for (int y = start_y; y < end_y; y++) {
        for (int x = start_x; x < end_x; x++) {
            if (y < height && x < width) {
                float pixel = src[y * width + x];
                sum += pixel;
                sum_sq += pixel * pixel;
                count++;
            }
        }
    }
    if (count <= 0) return 0;
    float mean = sum / count;
    float variance = (sum_sq / count) - (mean * mean);
    return sqrt(variance);
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

// Hàm phát hiện vùng mặt và ước lượng khu vực mắt
bool detectFaceAndEyes(camera_fb_t *fb, int *face_x, int *face_y, int *face_size) {
    int width = 96;
    int height = 96;
    int face_size_min = 40;
    int face_size_max = 80;
    float max_contrast = 0;

    memcpy(workingImage, fb->buf, 96 * 96);
    memcpy(tempImage, workingImage, 96 * 96);
    smoothImage(tempImage, workingImage, width, height);

    for (int size = face_size_min; size <= face_size_max; size += 10) {
        for (int y = 0; y <= height - size; y += 10) {
            for (int x = 0; x <= width - size; x += 10) {
                float contrast = calculateRegionContrast(workingImage, width, height, x, x + size, y, y + size);
                if (contrast > max_contrast) {
                    max_contrast = contrast;
                    *face_x = x;
                    *face_y = y;
                    *face_size = size;
                }
            }
        }
    }

    if (max_contrast < 10.0f) {
        Serial.println("Không tìm thấy mặt!");
        return false;
    }

    Serial.print("Vùng mặt tại: (");
    Serial.print(*face_x);
    Serial.print(", ");
    Serial.print(*face_y);
    Serial.print(") với kích thước: ");
    Serial.println(*face_size);

    return true;
}

// Hàm áp dụng bộ lọc Sobel để tìm cạnh
void applySobelFilter(uint8_t *src, uint8_t *dst, int width, int height) {
    equalizeHistogram(src, equalizedImage, width, height);
    memcpy(tempImage, equalizedImage, width * height);
    smoothImage(tempImage, equalizedImage, width, height);

    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int gx = (-1 * equalizedImage[(y - 1) * width + (x - 1)]) + 
                     (1 * equalizedImage[(y - 1) * width + (x + 1)]) + 
                     (-2 * equalizedImage[y * width + (x - 1)]) + 
                     (2 * equalizedImage[y * width + (x + 1)]) + 
                     (-1 * equalizedImage[(y + 1) * width + (x - 1)]) + 
                     (1 * equalizedImage[(y + 1) * width + (x + 1)]);

            int gy = (-1 * equalizedImage[(y - 1) * width + (x - 1)]) + 
                     (-2 * equalizedImage[(y - 1) * width + x]) + 
                     (-1 * equalizedImage[(y - 1) * width + (x + 1)]) + 
                     (1 * equalizedImage[(y + 1) * width + (x - 1)]) + 
                     (2 * equalizedImage[(y + 1) * width + x]) + 
                     (1 * equalizedImage[(y + 1) * width + (x + 1)]);

            int magnitude = sqrt(gx * gx + gy * gy);
            if (magnitude > 255) magnitude = 255;
            dst[y * width + x] = (uint8_t)magnitude;
        }
    }

    for (int x = 0; x < width; x++) {
        dst[x] = 0;
        dst[(height - 1) * width + x] = 0;
    }
    for (int y = 0; y < height; y++) {
        dst[y * width] = 0;
        dst[y * width + (width - 1)] = 0;
    }
}

// Hàm tìm mắt với hỗ trợ mặt nghiêng
bool detectEyesInFace(uint8_t *src, int width, int height, int face_x, int face_y, int face_size, int *left_eye_x, int *left_eye_y, int *right_eye_x, int *right_eye_y) {
    applySobelFilter(src, edgeImage, width, height);

    int eye_region_y_start = max(0, face_y - face_size / 4);
    int eye_region_y_end = min(height, face_y + face_size);
    int eye_region_x_start = max(0, face_x - face_size / 4);
    int eye_region_x_end = min(width, face_x + face_size + face_size / 4);

    const int max_candidates = 5;
    EyeCandidate left_candidates[max_candidates] = {0};
    EyeCandidate right_candidates[max_candidates] = {0};

    for (int y = eye_region_y_start; y < eye_region_y_end; y += 2) {
        for (int x = eye_region_x_start; x < eye_region_x_end; x += 2) {
            float edge_value = edgeImage[y * width + x];
            if (edge_value < 30.0f) continue;

            if (x < face_x + face_size / 2) {
                for (int i = 0; i < max_candidates; i++) {
                    if (edge_value > left_candidates[i].edge_value) {
                        for (int j = max_candidates - 1; j > i; j--) {
                            left_candidates[j] = left_candidates[j-1];
                        }
                        left_candidates[i].x = x;
                        left_candidates[i].y = y;
                        left_candidates[i].edge_value = edge_value;
                        break;
                    }
                }
            } else {
                for (int i = 0; i < max_candidates; i++) {
                    if (edge_value > right_candidates[i].edge_value) {
                        for (int j = max_candidates - 1; j > i; j--) {
                            right_candidates[j] = right_candidates[j-1];
                        }
                        right_candidates[i].x = x;
                        right_candidates[i].y = y;
                        right_candidates[i].edge_value = edge_value;
                        break;
                    }
                }
            }
        }
    }

    if (left_candidates[0].edge_value < 30.0f || right_candidates[0].edge_value < 30.0f) {
        Serial.println("Cạnh quá yếu, không tìm thấy mắt!");
        return false;
    }

    *left_eye_x = left_candidates[0].x;
    *left_eye_y = left_candidates[0].y;
    *right_eye_x = right_candidates[0].x;
    *right_eye_y = right_candidates[0].y;

    float eye_distance_x = abs(*right_eye_x - *left_eye_x);
    float eye_distance_y = abs(*right_eye_y - *left_eye_y);
    float eye_distance = sqrt(eye_distance_x * eye_distance_x + eye_distance_y * eye_distance_y);
    float min_eye_distance = face_size / 8;
    float max_eye_distance = face_size;

    float tilt_angle = atan2(*right_eye_y - *left_eye_y, *right_eye_x - *left_eye_x) * 180 / PI;
    Serial.print("Góc nghiêng của mắt: ");
    Serial.print(tilt_angle);
    Serial.println(" độ");

    if (eye_distance < min_eye_distance) {
        Serial.println("Khoảng cách giữa hai mắt quá gần!");
        float adjusted_distance = min_eye_distance;
        float mid_x = (*left_eye_x + *right_eye_x) / 2;
        float mid_y = (*left_eye_y + *right_eye_y) / 2;
        float angle_rad = tilt_angle * PI / 180;
        *left_eye_x = mid_x - (adjusted_distance / 2) * cos(angle_rad);
        *left_eye_y = mid_y - (adjusted_distance / 2) * sin(angle_rad);
        *right_eye_x = mid_x + (adjusted_distance / 2) * cos(angle_rad);
        *right_eye_y = mid_y + (adjusted_distance / 2) * sin(angle_rad);
    } else if (eye_distance > max_eye_distance) {
        Serial.println("Khoảng cách giữa hai mắt quá xa!");
        float adjusted_distance = max_eye_distance;
        float mid_x = (*left_eye_x + *right_eye_x) / 2;
        float mid_y = (*left_eye_y + *right_eye_y) / 2;
        float angle_rad = tilt_angle * PI / 180;
        *left_eye_x = mid_x - (adjusted_distance / 2) * cos(angle_rad);
        *left_eye_y = mid_y - (adjusted_distance / 2) * sin(angle_rad);
        *right_eye_x = mid_x + (adjusted_distance / 2) * cos(angle_rad);
        *right_eye_y = mid_y + (adjusted_distance / 2) * sin(angle_rad);
    }

    if (*left_eye_x < 0) *left_eye_x = 0;
    if (*left_eye_x >= width) *left_eye_x = width - 1;
    if (*left_eye_y < 0) *left_eye_y = 0;
    if (*left_eye_y >= height) *left_eye_y = height - 1;
    if (*right_eye_x < 0) *right_eye_x = 0;
    if (*right_eye_x >= width) *right_eye_x = width - 1;
    if (*right_eye_y < 0) *right_eye_y = 0;
    if (*right_eye_y >= height) *right_eye_y = height - 1;

    Serial.print("Mắt trái tại: (");
    Serial.print(*left_eye_x);
    Serial.print(", ");
    Serial.print(*left_eye_y);
    Serial.println(")");
    Serial.print("Mắt phải tại: (");
    Serial.print(*right_eye_x);
    Serial.print(", ");
    Serial.print(*right_eye_y);
    Serial.println(")");

    return true;
}

// Hàm xoay vùng ảnh
void rotateImage(uint8_t *src, uint8_t *dst, int size, float angle_rad) {
    int center = size / 2;
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            int src_x = (int)((x - center) * cos(angle_rad) + (y - center) * sin(angle_rad) + center);
            int src_y = (int)(-(x - center) * sin(angle_rad) + (y - center) * cos(angle_rad) + center);
            if (src_x >= 0 && src_x < size && src_y >= 0 && src_y < size) {
                dst[y * size + x] = src[src_y * size + src_x];
            } else {
                dst[y * size + x] = 0;
            }
        }
    }
}

// Hàm cắt và resize vùng mắt với xử lý nghiêng
void cropAndResize(uint8_t *src, uint8_t *dst, int src_width, int src_height, int crop_x, int crop_y, int crop_size, int dst_size, float tilt_angle) {
    for (int y = 0; y < crop_size; y++) {
        for (int x = 0; x < crop_size; x++) {
            int src_y = crop_y + y - crop_size / 2;
            int src_x = crop_x + x - crop_size / 2;
            if (src_x >= 0 && src_x < src_width && src_y >= 0 && src_y < src_height) {
                croppedEye[y * crop_size + x] = src[src_y * src_width + src_x];
            } else {
                croppedEye[y * crop_size + x] = 0;
            }
        }
    }

    uint8_t rotatedEye[24 * 24];
    float angle_rad = -tilt_angle * PI / 180;
    rotateImage(croppedEye, rotatedEye, crop_size, angle_rad);

    for (int y = 0; y < dst_size; y++) {
        for (int x = 0; x < dst_size; x++) {
            int src_x = x * crop_size / dst_size;
            int src_y = y * crop_size / dst_size;
            dst[y * dst_size + x] = rotatedEye[src_y * crop_size + src_x];
        }
    }
}

// Hàm xử lý ảnh với hỗ trợ mặt nghiêng
void processImage(camera_fb_t *fb, TfLiteTensor* input, int left_eye_x, int left_eye_y, int right_eye_x, int right_eye_y, float& left_eye_prob, float& right_eye_prob) {
    int crop_size = 24;
    int dst_size = 24;

    float tilt_angle = atan2(right_eye_y - left_eye_y, right_eye_x - left_eye_x) * 180 / PI;

    // Xử lý mắt trái
    cropAndResize(workingImage, processedLeftEye, 96, 96, left_eye_x, left_eye_y, crop_size, dst_size, tilt_angle);
    uint8_t temp[24 * 24];
    memcpy(temp, processedLeftEye, 24 * 24);
    smoothImage(temp, processedLeftEye, 24, 24);

    for (int i = 0; i < 24 * 24; i++) {
        input->data.f[i] = (float)processedLeftEye[i] / 255.0f;
    }
    left_eye_prob = runModel();
    if (left_eye_prob < 0) left_eye_prob = 0.5f;

    // Xử lý mắt phải
    cropAndResize(workingImage, processedRightEye, 96, 96, right_eye_x, right_eye_y, crop_size, dst_size, tilt_angle);
    memcpy(temp, processedRightEye, 24 * 24);
    smoothImage(temp, processedRightEye, 24, 24);

    for (int i = 0; i < 24 * 24; i++) {
        input->data.f[i] = (float)processedRightEye[i] / 255.0f;
    }
    right_eye_prob = runModel();
    if (right_eye_prob < 0) right_eye_prob = 0.5f;

    Serial.println("Một số giá trị pixel sau xử lý (mắt trái):");
    for (int i = 0; i < 10; i++) {
        Serial.print(processedLeftEye[i]);
        Serial.print(" ");
    }
    Serial.println();

    Serial.println("Một số giá trị pixel sau xử lý (mắt phải):");
    for (int i = 0; i < 10; i++) {
        Serial.print(processedRightEye[i]);
        Serial.print(" ");
    }
    Serial.println();
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

// Hàm khởi tạo SD card
bool initSDCard() {
    Serial.println("Đang khởi động SD card...");
    
    if(!SD_MMC.begin("/sdcard", true)) { // Chế độ 1-line mode
        Serial.println("❌ Không thể khởi động SD card");
        return false;
    }
    
    uint8_t cardType = SD_MMC.cardType();
    if(cardType == CARD_NONE) {
        Serial.println("❌ Không tìm thấy SD card");
        return false;
    }
    
    Serial.print("Loại SD card: ");
    if(cardType == CARD_MMC) {
        Serial.println("MMC");
    } else if(cardType == CARD_SD) {
        Serial.println("SDSC");
    } else if(cardType == CARD_SDHC) {
        Serial.println("SDHC");
    } else {
        Serial.println("Không xác định");
    }
    
    uint64_t cardSize = SD_MMC.cardSize() / (1024 * 1024);
    Serial.printf("Dung lượng SD card: %lluMB\n", cardSize);
    
    return true;
}

// Hàm lưu ảnh dạng PGM
void saveImageAsPGM(const char* filename, uint8_t *image_data, int width, int height) {
    Serial.printf("Đang lưu ảnh vào %s...\n", filename);
    
    File file = SD_MMC.open(filename, FILE_WRITE);
    if(!file) {
        Serial.println("❌ Không thể mở file để ghi");
        return;
    }
    
    // Ghi header PGM
    file.print("P5\n"); // Magic number cho binary PGM
    file.print("# Saved from ESP32-CAM\n"); // Comment
    file.printf("%d %d\n", width, height); // Kích thước ảnh
    file.print("255\n"); // Giá trị pixel tối đa
    
    // Ghi dữ liệu ảnh
    size_t bytes_written = file.write(image_data, width * height);
    file.close();
    
    if(bytes_written == width * height) {
        Serial.printf("✅ Đã lưu ảnh %s (%d bytes)\n", filename, bytes_written);
    } else {
        Serial.printf("❌ Lỗi khi lưu ảnh (chỉ ghi được %d/%d bytes)\n", 
                     bytes_written, width * height);
    }
}

void setup() {
    Serial.begin(115200);
    Serial.println("Khởi động ESP32-CAM...");

    // Khởi tạo prob_history
    for (int i = 0; i < 3; i++) {
        prob_history[i] = 0.0f;
    }

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

    // Khởi tạo SD card
    if (!initSDCard()) {
        Serial.println("Tiếp tục mà không có SD card, ảnh sẽ không được lưu");
    } else {
        Serial.println("✅ SD card sẵn sàng");
    }

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
        delay(500);
        return;
    }
    Serial.println("📷 Ảnh đã chụp, đang xử lý...");

    // Kiểm tra hướng mặt
    bool is_facing_forward = detectFaceDirection(fb);

    // Kiểm tra trạng thái mắt
    bool eyes_open = false; // Mặc định mắt nhắm
    int face_x = 0, face_y = 0, face_size = 0;
    if (detectFaceAndEyes(fb, &face_x, &face_y, &face_size)) {
        int left_eye_x = 0, left_eye_y = 0, right_eye_x = 0, right_eye_y = 0;
        if (detectEyesInFace(workingImage, 96, 96, face_x, face_y, face_size, &left_eye_x, &left_eye_y, &right_eye_x, &right_eye_y)) {
            TfLiteTensor* input = interpreter->input(0);
            float left_eye_prob, right_eye_prob;
            processImage(fb, input, left_eye_x, left_eye_y, right_eye_x, right_eye_y, left_eye_prob, right_eye_prob);

            Serial.print("Xác suất thở (mắt trái): ");
            Serial.println(left_eye_prob);
            Serial.print("Xác suất thở (mắt phải): ");
            Serial.println(right_eye_prob);

            // Cập nhật lịch sử xác suất
            prob_history[history_index] = (left_eye_prob + right_eye_prob) / 2.0f;
            history_index = (history_index + 1) % 3;
            float avg_prob = 0;
            for (int i = 0; i < 3; i++) {
                avg_prob += prob_history[i];
            }
            avg_prob /= 3;
            Serial.print("Xác suất trung bình: ");
            Serial.println(avg_prob);

            // In giá trị prob_history để debug
            Serial.print("Lịch sử xác suất: ");
            for (int i = 0; i < 3; i++) {
                Serial.print(prob_history[i]);
                Serial.print(" ");
            }
            Serial.println();

            // Quyết định trạng thái mắt và lưu ảnh dựa trên avg_prob
            if (avg_prob > 0.5) {
                Serial.println("👀 Mắt đang mở");
            } else {
                Serial.println("😴 Mắt đang nhắm");
                eyes_open = false;
                if (SD_MMC.cardType() != CARD_NONE) {
                    // Lưu ảnh gốc
                    String filename = generateUniqueFilename("/original");
                    saveImageAsPGM(filename.c_str(), fb->buf, fb->width, fb->height);
                    
                    // Lưu ảnh đã xử lý
                    String processed_filename = generateUniqueFilename("/processed");
                    saveImageAsPGM(processed_filename.c_str(), workingImage, 96, 96);
                    
                    // Lưu ảnh mắt
                    String eye_filename = generateUniqueFilename("/eyes_closed");
                    saveImageAsPGM(eye_filename.c_str(), processedLeftEye, 24, 24);
                    saveImageAsPGM((eye_filename + "_right").c_str(), processedRightEye, 24, 24);
                    Serial.println("✅ Lưu ảnh cả hai mắt nhắm");
                }
            }
        } else {
            Serial.println("Không tìm thấy mắt, coi như mắt nhắm.");
            eyes_open = false;
        }
    } else {
        Serial.println("Không tìm thấy mặt, coi như mắt nhắm.");
        eyes_open = false;
    }

    esp_camera_fb_return(fb);
    delay(500);
}