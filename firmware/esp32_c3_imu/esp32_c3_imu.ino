/*
 * Digital Spotter — ESP32-C3 IMU Sensor Node Firmware
 * ─────────────────────────────────────────────────────────────────────────────
 * Main sketch. Clean entry point — all logic is in the .h modules.
 *
 * Wiring:
 *   MPU9250 VCC -> 3.3V   | MPU9250 GND -> GND
 *   MPU9250 SDA -> GPIO 5 | MPU9250 SCL -> GPIO 6
 *   Buzzer  +   -> GPIO 10| Buzzer  -   -> GND
 *
 * Required Arduino Libraries (Install via Library Manager):
 *   - "Adafruit NeoPixel" by Adafruit
 *   - "PubSubClient"      by Nick O'Leary
 *
 * Before flashing, fill in your credentials in secrets.h
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include <Arduino.h>
#include <Adafruit_NeoPixel.h>
#include "secrets.h"
#include "DigitalSpotterIMU.h"
#include "DigitalSpotterComms.h"

// ─── Pin Definitions ─────────────────────────────────────────────────────────
#define I2C_SDA_PIN      5
#define I2C_SCL_PIN      6
#define BUZZER_PIN       10
#define RGB_LED_PIN      8

// ─── NeoPixel ────────────────────────────────────────────────────────────────
#define NUM_LEDS         1
Adafruit_NeoPixel pixels(NUM_LEDS, RGB_LED_PIN, NEO_GRB + NEO_KHZ800);

// ─── Module Instances ────────────────────────────────────────────────────────
DigitalSpotterIMU   imu;
DigitalSpotterComms comms;

// ─── Timing ──────────────────────────────────────────────────────────────────
unsigned long lastPublishMs = 0;
const int     PUBLISH_INTERVAL_MS = 20;  // 50Hz

// ─── LED State ───────────────────────────────────────────────────────────────
enum LedState {
    LED_OFF,
    LED_SOLID,
    LED_PULSE,
    LED_FLASH,
};

struct LedAnim {
    LedState  state   = LED_SOLID;
    uint8_t   r = 0, g = 0, b = 0;
    uint16_t  periodMs = 1000; // for PULSE / FLASH
    unsigned long nextToggleMs = 0;
    bool      flashOn = false;
} ledAnim;

// ─── LED Helpers ─────────────────────────────────────────────────────────────
void setLED(uint8_t r, uint8_t g, uint8_t b) {
    pixels.setPixelColor(0, pixels.Color(r, g, b));
    pixels.show();
}

void startPulse(uint8_t r, uint8_t g, uint8_t b, uint16_t periodMs = 1000) {
    ledAnim = {LED_PULSE, r, g, b, periodMs, millis(), false};
}

void startFlash(uint8_t r, uint8_t g, uint8_t b, uint16_t periodMs = 300) {
    ledAnim = {LED_FLASH, r, g, b, periodMs, millis(), true};
    setLED(r, g, b);
}

void startSolid(uint8_t r, uint8_t g, uint8_t b) {
    ledAnim = {LED_SOLID, r, g, b, 0, 0, false};
    setLED(r, g, b);
}

void updateLED() {
    unsigned long now = millis();
    if (ledAnim.state == LED_PULSE) {
        // Smooth breathing using a sine wave
        float phase = (now % ledAnim.periodMs) / (float)ledAnim.periodMs;
        float brightness = 0.5f * (1.0f + sin(2.0f * PI * phase - PI / 2.0f));
        pixels.setPixelColor(0, pixels.Color(
            (uint8_t)(ledAnim.r * brightness),
            (uint8_t)(ledAnim.g * brightness),
            (uint8_t)(ledAnim.b * brightness)));
        pixels.show();
    } else if (ledAnim.state == LED_FLASH) {
        if (now >= ledAnim.nextToggleMs) {
            ledAnim.flashOn = !ledAnim.flashOn;
            ledAnim.nextToggleMs = now + ledAnim.periodMs;
            if (ledAnim.flashOn) setLED(ledAnim.r, ledAnim.g, ledAnim.b);
            else                 setLED(0, 0, 0);
        }
    }
}

// ─── Buzzer Helpers ───────────────────────────────────────────────────────────
void beep(unsigned int freq, unsigned long durationMs) {
    tone(BUZZER_PIN, freq, durationMs);
    delay(durationMs);
    noTone(BUZZER_PIN);
}

void beepWifiOk()  { beep(2500, 150); }
void beepMqttOk()  { beep(2500, 100); delay(80); beep(2500, 100); }
void beepBadForm() {
    for (int i = 0; i < 5; i++) {
        beep(2500, 80);
        if (i < 4) delay(60);
    }
}
void beepGoodForm() { beep(2500, 120); }

// ─── Result callback (called when Pi sends inference result) ──────────────────
void onResult(bool is_bad_form, float confidence) {
    if (is_bad_form) {
        Serial.printf("[RESULT] BAD FORM  — confidence: %.1f%%\n", confidence * 100);
        beepBadForm();
    } else {
        Serial.printf("[RESULT] GOOD FORM — confidence: %.1f%%\n", confidence * 100);
        beepGoodForm();
    }
}

// ─── Setup ───────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    delay(1500);

    Serial.println();
    Serial.println("╔══════════════════════════════════╗");
    Serial.println("║   Digital Spotter — ESP32-C3     ║");
    Serial.println("║   IMU Sensor Node v2.0           ║");
    Serial.println("╚══════════════════════════════════╝");

    // ── 1. Buzzer init ──────────────────────────────────────────────────────
    pinMode(BUZZER_PIN, OUTPUT);

    // ── 2. RGB LED init ─────────────────────────────────────────────────────
    pixels.begin();
    pixels.setBrightness(60);
    startSolid(0, 0, 0);
    Serial.println("[LED]  RGB LED ready");

    // ── 3. IMU init ─────────────────────────────────────────────────────────
    Serial.println("[IMU]  Initializing MPU-9250...");
    startPulse(0, 0, 255, 800); // Blue pulse = IMU init
    if (!imu.begin(I2C_SDA_PIN, I2C_SCL_PIN)) {
        Serial.println("[IMU]  ❌ FAILED — halting");
        startFlash(255, 0, 0, 300);
        while (1) { updateLED(); delay(10); }
    }
    Serial.println("[IMU]  ✓ Ready");

    // ── 4. WiFi ─────────────────────────────────────────────────────────────
    Serial.printf("[WiFi] Connecting to: %s\n", WIFI_SSID);
    startPulse(0, 0, 255, 600); // Blue pulse = WiFi connecting
    if (!comms.connectWiFi()) {
        Serial.println("[WiFi] ❌ FAILED — halting");
        startFlash(255, 0, 0, 300);
        while (1) { updateLED(); delay(10); }
    }
    Serial.println("[WiFi] ✓ Connected");
    startSolid(0, 255, 0); // Green flash = WiFi OK
    beepWifiOk();
    delay(400);

    // ── 5. MQTT ─────────────────────────────────────────────────────────────
    Serial.printf("[MQTT] Connecting to HiveMQ: %s:%d\n", MQTT_BROKER, MQTT_PORT);
    startPulse(255, 140, 0, 600); // Amber pulse = MQTT connecting
    if (!comms.connectMQTT(onResult)) {
        Serial.println("[MQTT] ❌ FAILED — halting");
        startFlash(255, 0, 0, 300);
        while (1) { updateLED(); delay(10); }
    }
    Serial.println("[MQTT] ✓ Connected to HiveMQ Cloud");
    startSolid(0, 255, 0); // Green flash = MQTT OK
    beepMqttOk();
    delay(400);

    // ── All good → breathing green ──────────────────────────────────────────
    startPulse(0, 255, 0, 1200); // Slow green breathe = streaming
    Serial.println();
    Serial.println("════════════════════════════════════");
    Serial.println("  Streaming IMU data at 50Hz");
    Serial.println("  Topic: " TOPIC_IMU_PUB);
    Serial.println("════════════════════════════════════");
}

// ─── Loop ────────────────────────────────────────────────────────────────────
void loop() {
    unsigned long now = millis();

    // Maintain MQTT + process incoming messages
    comms.loop();

    // Animate LED (non-blocking)
    updateLED();

    // Publish IMU data at 50Hz
    if (now - lastPublishMs >= PUBLISH_INTERVAL_MS) {
        lastPublishMs = now;

        float ax, ay, az, gx, gy, gz;
        if (imu.read(ax, ay, az, gx, gy, gz)) {
            bool published = comms.publishIMU(ax, ay, az, gx, gy, gz);

            // Print to Serial Plotter (also useful for debugging)
            Serial.printf("AccX:%.3f,AccY:%.3f,AccZ:%.3f,GyroX:%.2f,GyroY:%.2f,GyroZ:%.2f%s\n",
                ax, ay, az, gx, gy, gz,
                published ? "" : " [TX_FAIL]");
        }
    }
}
