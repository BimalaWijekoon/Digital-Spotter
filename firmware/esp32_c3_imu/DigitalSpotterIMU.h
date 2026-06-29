#pragma once
#include <Arduino.h>
#include <Wire.h>

class DigitalSpotterIMU {
private:
    uint8_t mpu_address = 0x00;

public:
    bool begin(int sda_pin, int scl_pin) {
        Wire.begin(sda_pin, scl_pin);
        
        Serial.println("\nScanning I2C bus...");
        byte error;
        for(byte address = 1; address < 127; address++ ) {
            Wire.beginTransmission(address);
            error = Wire.endTransmission();
            if (error == 0) {
                Serial.print("I2C device found at address 0x");
                Serial.println(address, HEX);
                // If it's 0x68 or 0x69, it's our IMU!
                if (address == 0x68 || address == 0x69) {
                    mpu_address = address;
                }
            }
        }

        if (mpu_address == 0x00) {
            Serial.println("❌ No IMU found! Check wiring (SDA, SCL, VCC, GND).");
            return false;
        }

        Serial.print("✓ IMU selected at address 0x");
        Serial.println(mpu_address, HEX);

        // --- Determine the actual chip (WHO_AM_I test) ---
        Wire.beginTransmission(mpu_address);
        Wire.write(0x75);  // WHO_AM_I register
        Wire.endTransmission(false);
        Wire.requestFrom((uint16_t)mpu_address, (uint8_t)1, (bool)true);

        if (Wire.available()) {
            uint8_t id = Wire.read();
            Serial.print("WHO_AM_I = 0x");
            Serial.println(id, HEX);
            if (id == 0x68) Serial.println("Chip identified as: MPU6050");
            else if (id == 0x70) Serial.println("Chip identified as: MPU6500");
            else if (id == 0x71) Serial.println("Chip identified as: MPU9250");
            else if (id == 0x73) Serial.println("Chip identified as: MPU9255");
            else Serial.println("Chip identified as: Unknown clone");
        } else {
            Serial.println("Failed to read WHO_AM_I register");
        }
        // ------------------------------------------------

        // Wake up the IMU (Raw I2C)
        // Write 0x00 to PWR_MGMT_1 (register 0x6B)
        Wire.beginTransmission(mpu_address);
        Wire.write(0x6B); 
        Wire.write(0x00);
        error = Wire.endTransmission();
        
        if (error != 0) {
            Serial.println("❌ Failed to wake up IMU!");
            return false;
        }

        return true;
    }

    bool read(float &ax, float &ay, float &az, float &gx, float &gy, float &gz) {
        // Request 14 bytes starting from register 0x3B (ACCEL_XOUT_H)
        Wire.beginTransmission(mpu_address);
        Wire.write(0x3B);
        Wire.endTransmission(false);
        
        Wire.requestFrom((uint16_t)mpu_address, (uint8_t)14, (bool)true);
        
        if (Wire.available() == 14) {
            // Read raw 16-bit values (High byte first, then Low byte)
            int16_t axRaw = Wire.read() << 8 | Wire.read();
            int16_t ayRaw = Wire.read() << 8 | Wire.read();
            int16_t azRaw = Wire.read() << 8 | Wire.read();
            
            // Skip temperature
            Wire.read(); 
            Wire.read();
            
            int16_t gxRaw = Wire.read() << 8 | Wire.read();
            int16_t gyRaw = Wire.read() << 8 | Wire.read();
            int16_t gzRaw = Wire.read() << 8 | Wire.read();

            // Convert raw values (Default range is +/- 2g and +/- 250 deg/s)
            ax = axRaw / 16384.0;
            ay = ayRaw / 16384.0;
            az = azRaw / 16384.0;
            
            gx = gxRaw / 131.0;
            gy = gyRaw / 131.0;
            gz = gzRaw / 131.0;
            
            return true;
        }
        return false;
    }
};
