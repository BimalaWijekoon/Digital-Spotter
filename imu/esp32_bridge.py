"""
imu/esp32_bridge.py
Purpose: BLE/WiFi bridge to ESP32-S3 IMU sensor.
         Placeholder — returns mock data until hardware available.
Author: bimalawijekoon
Version: 1.0.0
Last Modified: 2026-06-15
"""

import logging
import time
import math
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import Config

logger = logging.getLogger(__name__)

# TODO: PHASE_8 - Import BLE library (bleak) when ESP32 ready


class ESP32Bridge:
    """Bridge to ESP32-S3 IMU sensor.

    Currently a mock implementation that generates synthetic
    accelerometer and gyroscope data. Will be replaced with
    BLE/WiFi communication when hardware is available.
    """

    def __init__(self):
        """Initialize ESP32Bridge in mock mode."""
        self._connected = False
        self._is_mock = True
        self._ble_address = Config.IMU.BLE_ADDRESS
        self._sample_rate = Config.IMU.SAMPLE_RATE_HZ
        self._last_sample_time = 0
        self._frame_count = 0

    def connect(self):
        """Connect to ESP32-S3 via BLE or WiFi.

        Returns:
            bool: True if connected (always True in mock mode).

        Side Effects:
            Sets internal connected state.
        """
        # TODO: PHASE_8 - Implement BLE connection via bleak:
        # async with BleakClient(self._ble_address) as client:
        #     self._client = client
        #     self._connected = client.is_connected
        self._connected = True
        self._is_mock = True
        logger.info("ESP32Bridge connected (MOCK mode)")
        return True

    def disconnect(self):
        """Disconnect from ESP32-S3.

        Side Effects:
            Clears connection state.
        """
        self._connected = False
        logger.info("ESP32Bridge disconnected")

    def read_sample(self):
        """Read one IMU sample.

        Returns:
            dict: IMU data with keys:
                acc_x, acc_y, acc_z (m/s²),
                gyro_x, gyro_y, gyro_z (deg/s),
                v_bar_velocity (m/s),
                p_bar_power (W),
                smoothness_jerk (m/s³),
                timestamp_ms (int).

        Side Effects:
            None (pure mock data generation).
        """
        if not self._connected:
            return self._empty_sample()

        if self._is_mock:
            return self._mock_sample()

        # TODO: PHASE_8 - Read actual BLE characteristic
        # data = await self._client.read_gatt_char(IMU_CHAR_UUID)
        # return self._parse_imu_packet(data)
        return self._mock_sample()

    def is_connected(self):
        """Check connection status.
        Returns:
            bool: True if connected.
        """
        return self._connected

    @property
    def is_mock(self):
        """Check if using mock data.
        Returns:
            bool: True if no real hardware.
        """
        return self._is_mock

    def _mock_sample(self):
        """Generate synthetic IMU data simulating a barbell squat.

        Returns:
            dict: Mock IMU sample with plausible squat motion data.
        """
        self._frame_count += 1
        t = self._frame_count / self._sample_rate

        # Simulate squat cycle (2Hz oscillation)
        phase = math.sin(2 * math.pi * 0.33 * t)

        # Accelerometer (gravity + motion)
        acc_x = np.random.normal(0, 0.1)
        acc_y = -9.81 + phase * 2.0 + np.random.normal(0, 0.2)
        acc_z = np.random.normal(0, 0.1)

        # Gyroscope
        gyro_x = np.random.normal(0, 5.0)
        gyro_y = np.random.normal(0, 5.0)
        gyro_z = np.random.normal(0, 2.0)

        # Derived bar metrics
        velocity = abs(phase) * 0.6 + np.random.normal(0, 0.02)
        power = velocity * 80.0 * 9.81
        jerk = np.random.normal(0, 0.5)

        return {
            "acc_x": float(acc_x),
            "acc_y": float(acc_y),
            "acc_z": float(acc_z),
            "gyro_x": float(gyro_x),
            "gyro_y": float(gyro_y),
            "gyro_z": float(gyro_z),
            "v_bar_velocity": float(velocity),
            "p_bar_power": float(power),
            "smoothness_jerk": float(jerk),
            "timestamp_ms": int(time.time() * 1000),
        }

    def _empty_sample(self):
        """Return zeroed IMU sample.
        Returns:
            dict: All-zeros IMU sample.
        """
        return {
            "acc_x": 0.0, "acc_y": 0.0, "acc_z": 0.0,
            "gyro_x": 0.0, "gyro_y": 0.0, "gyro_z": 0.0,
            "v_bar_velocity": 0.0, "p_bar_power": 0.0,
            "smoothness_jerk": 0.0,
            "timestamp_ms": int(time.time() * 1000),
        }

    def _parse_imu_packet(self, data):
        """Parse raw BLE packet into IMU dict.

        Args:
            data: bytes from BLE characteristic read.

        Returns:
            dict: Parsed IMU sample.
        """
        # TODO: PHASE_8 - Implement struct.unpack for ESP32 packet format
        # Format: 6 x float32 (acc + gyro) = 24 bytes
        import struct
        if len(data) >= 24:
            values = struct.unpack('<6f', data[:24])
            return {
                "acc_x": values[0], "acc_y": values[1], "acc_z": values[2],
                "gyro_x": values[3], "gyro_y": values[4], "gyro_z": values[5],
                "v_bar_velocity": 0.0, "p_bar_power": 0.0,
                "smoothness_jerk": 0.0,
                "timestamp_ms": int(time.time() * 1000),
            }
        return self._empty_sample()
