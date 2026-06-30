#pragma once
/*
 * DigitalSpotterComms.h
 * 
 * Handles WiFi connection and bidirectional HiveMQ Cloud MQTT for the ESP32-C3.
 * 
 * Publishes:  digitalspotter/imu    — JSON IMU data at 50Hz
 * Subscribes: digitalspotter/result — JSON inference result from Raspberry Pi
 * 
 * Required Arduino Libraries (Install via Library Manager):
 * - "PubSubClient" by Nick O'Leary
 * - "WiFiClientSecure" (built-in with ESP32 board package)
 */

#include <Arduino.h>
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <PubSubClient.h>
#include "secrets.h"

// MQTT Topics
#define TOPIC_IMU_PUB    "digitalspotter/imu"
#define TOPIC_RESULT_SUB "digitalspotter/result"

// ─── Callback function type for when a result arrives from the Pi ────────────
typedef void (*ResultCallback)(bool is_bad_form, float confidence);

// ─── DigitalSpotterComms class ───────────────────────────────────────────────
class DigitalSpotterComms {
private:
    WiFiClientSecure _wifiClient;
    PubSubClient _mqttClient;
    ResultCallback _onResult = nullptr;
    bool _resultEnabled = true; // Can be set to false to disable bi-directional

    // Static trampoline for PubSubClient callback (cannot use lambda with member function)
    static DigitalSpotterComms* _instance;
    static void _mqttCallbackTrampoline(char* topic, byte* payload, unsigned int length) {
        if (_instance) _instance->_onMqttMessage(topic, payload, length);
    }

    void _onMqttMessage(char* topic, byte* payload, unsigned int length) {
        if (!_resultEnabled || !_onResult) return;
        if (String(topic) != TOPIC_RESULT_SUB) return;

        // Parse: {"bad_form": true, "confidence": 0.87}
        String msg = "";
        for (unsigned int i = 0; i < length; i++) msg += (char)payload[i];

        bool is_bad = msg.indexOf("\"bad_form\":true") != -1 || 
                      msg.indexOf("\"bad_form\": true") != -1;
        
        float confidence = 0.0f;
        int ci = msg.indexOf("\"confidence\":");
        if (ci != -1) {
            confidence = msg.substring(ci + 13).toFloat();
        }

        _onResult(is_bad, confidence);
    }

    bool _reconnectMqtt() {
        if (_mqttClient.connected()) return true;

        Serial.print("[MQTT] Connecting to ");
        Serial.print(MQTT_BROKER);
        Serial.print("...");

        if (_mqttClient.connect(MQTT_CLIENT_ID, MQTT_USERNAME, MQTT_PASSWORD)) {
            Serial.println(" OK");
            if (_resultEnabled) {
                _mqttClient.subscribe(TOPIC_RESULT_SUB);
                Serial.println("[MQTT] Subscribed to: " TOPIC_RESULT_SUB);
            }
            return true;
        }

        Serial.print(" FAILED, rc=");
        Serial.println(_mqttClient.state());
        return false;
    }

public:
    DigitalSpotterComms() : _mqttClient(_wifiClient) {
        _instance = this;
    }

    // Connect to WiFi. Returns true on success.
    bool connectWiFi() {
        Serial.print("[WiFi] Connecting to ");
        Serial.print(WIFI_SSID);

        WiFi.mode(WIFI_STA);
        WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

        unsigned long startMs = millis();
        while (WiFi.status() != WL_CONNECTED) {
            if (millis() - startMs > 15000) { // 15 second timeout
                Serial.println(" TIMEOUT");
                return false;
            }
            delay(250);
            Serial.print(".");
        }

        Serial.println(" OK");
        Serial.print("[WiFi] IP Address: ");
        Serial.println(WiFi.localIP());
        return true;
    }

    // Connect to HiveMQ MQTT broker. Returns true on success.
    bool connectMQTT(ResultCallback callback = nullptr) {
        _onResult = callback;
        // Use setInsecure() to skip CA cert verification.
        // HiveMQ Cloud uses TLS encryption regardless — the connection is still
        // encrypted end-to-end. Only the server identity check is skipped.
        _wifiClient.setInsecure();
        _mqttClient.setServer(MQTT_BROKER, MQTT_PORT);
        _mqttClient.setCallback(_mqttCallbackTrampoline);
        _mqttClient.setBufferSize(512);
        _mqttClient.setKeepAlive(30);

        Serial.println("[MQTT] Using TLS (insecure mode — encrypted but no cert check)");
        return _reconnectMqtt();
    }

    // Call this every loop iteration to maintain connection and process messages.
    void loop() {
        if (!_mqttClient.connected()) {
            Serial.println("[MQTT] Disconnected. Reconnecting...");
            _reconnectMqtt();
        }
        _mqttClient.loop();
    }

    // Publish a 6-DOF IMU reading as JSON.
    bool publishIMU(float ax, float ay, float az, float gx, float gy, float gz) {
        if (!_mqttClient.connected()) return false;

        char buf[160];
        snprintf(buf, sizeof(buf),
            "{\"ts\":%lu,\"ax\":%.3f,\"ay\":%.3f,\"az\":%.3f,\"gx\":%.2f,\"gy\":%.2f,\"gz\":%.2f}",
            millis(), ax, ay, az, gx, gy, gz);

        return _mqttClient.publish(TOPIC_IMU_PUB, buf);
    }

    bool isWiFiConnected() { return WiFi.status() == WL_CONNECTED; }
    bool isMQTTConnected() { return _mqttClient.connected(); }

    // Disable result subscription (makes this publish-only)
    void disableResultSubscription() { _resultEnabled = false; }
};

// Static member definition
DigitalSpotterComms* DigitalSpotterComms::_instance = nullptr;
