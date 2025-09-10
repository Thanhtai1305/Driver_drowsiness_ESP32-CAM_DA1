#include "esp_camera.h"
#include "SD_MMC.h"
#include "FS.h"
#include <math.h>

/* CẤU HÌNH CAMERA */
#define PWDN_GPIO_NUM  32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM  0
#define SIOD_GPIO_NUM  26
#define SIOC_GPIO_NUM  27
#define Y9_GPIO_NUM    35
#define Y8_GPIO_NUM    34
#define Y7_GPIO_NUM    39
#define Y6_GPIO_NUM    36
#define Y5_GPIO_NUM    21
#define Y4_GPIO_NUM    19
#define Y3_GPIO_NUM    18
#define Y2_GPIO_NUM    5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM  23
#define PCLK_GPIO_NUM  22

#define EYE_AR_THRESHOLD  0.2

void initCamera() {
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;
    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;
    config.pin_sscb_sda = SIOD_GPIO_NUM;
    config.pin_sscb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_GRAYSCALE;  // Dữ liệu ảnh xám
    config.frame_size = FRAMESIZE_QVGA;         // 320x240
    config.jpeg_quality = 10;
    config.fb_count = 1;

    if (esp_camera_init(&config) != ESP_OK) {
        Serial.println("Camera init failed");
        return;
    }
    Serial.println("Camera Ready!");
}

// Lưu ảnh xám (.pgm) để có thể xem được
void saveGrayscalePGMToSD(camera_fb_t *fb) {
    String path = "/image_" + String(millis()) + ".pgm";
    File file = SD_MMC.open(path, FILE_WRITE);
    if (!file) {
        Serial.println("Failed to open file for writing");
        return;
    }

    // Header ảnh PGM kiểu nhị phân
    file.print("P5\n");
    file.printf("%d %d\n", fb->width, fb->height);
    file.print("255\n");

    // Dữ liệu ảnh nhị phân grayscale
    file.write(fb->buf, fb->len);
    file.close();

    Serial.println("PGM grayscale image saved to: " + path);
}

// Tính chỉ số EAR đơn giản (minh họa)
float calculateEAR(int p1, int p2, int p3, int p4, int p5, int p6) {
    float A = sqrt(pow(p2 - p6, 2) + pow(p3 - p5, 2));
    float B = sqrt(pow(p1 - p4, 2));
    return A / (2.0 * B);
}

void captureAndProcessImage() {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        Serial.println("Camera capture failed");
        return;
    }

    // Lưu ảnh vào SD dưới dạng PGM
    saveGrayscalePGMToSD(fb);

    Serial.println("Image captured, processing EAR...");

    // Vùng mắt mẫu (tùy chỉnh theo vị trí thực tế)
    int eye_x = 80, eye_y = 50, eye_width = 100, eye_height = 40;
    int p1 = fb->buf[eye_y * fb->width + eye_x];
    int p4 = fb->buf[eye_y * fb->width + eye_x + eye_width];
    int p2 = fb->buf[(eye_y + 10) * fb->width + (eye_x + eye_width / 2)];
    int p3 = fb->buf[(eye_y + 20) * fb->width + (eye_x + eye_width / 2)];
    int p5 = fb->buf[(eye_y + 30) * fb->width + (eye_x + eye_width / 2)];
    int p6 = fb->buf[(eye_y + 40) * fb->width + (eye_x + eye_width / 2)];

    float EAR = calculateEAR(p1, p2, p3, p4, p5, p6);
    Serial.print("EAR: ");
    Serial.println(EAR);

    if (EAR < EYE_AR_THRESHOLD) {
        Serial.println("Eyes Closed (Drowsiness Detected!)");
    } else {
        Serial.println("Eyes Open");
    }

    esp_camera_fb_return(fb);
}

void setup() {
    Serial.begin(115200);
    delay(1000);

    Serial.println("Initializing camera...");
    initCamera();

    Serial.println("Mounting SD card...");
    if (!SD_MMC.begin()) {
        Serial.println("SD Card Mount Failed");
        return;
    }
    uint8_t cardType = SD_MMC.cardType();
    if (cardType == CARD_NONE) {
        Serial.println("No SD card attached");
        return;
    }

    Serial.println("SD Card Ready!");
}

void loop() {
    captureAndProcessImage();
    delay(3000);  // Chụp mỗi 3 giây
}
