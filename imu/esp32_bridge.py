"""
imu/esp32_bridge.py
Purpose: MQTT bridge to ESP32-C3 IMU sensor over HiveMQ Cloud.
         Falls back to mock data if MQTT is disabled or broker is unavailable.
Author: bimalawijekoon
Version: 2.0.0
Last Modified: 2026-06-29
"""

import json
import logging
import threading
import time
import math
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import Config

logger = logging.getLogger(__name__)

# Attempt paho-mqtt import
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    logger.warning("paho-mqtt not installed — run: pip install paho-mqtt")


class ESP32Bridge:
    """MQTT bridge to ESP32-C3 IMU sensor over HiveMQ Cloud.

    Subscribes to 'digitalspotter/imu' and stores the latest sample.
    Publishes inference results to 'digitalspotter/result' for ESP32 buzzer feedback.
    Falls back to mock mode if MQTT is disabled or unavailable.

    Usage:
        bridge = ESP32Bridge()
        bridge.connect()
        sample = bridge.read_sample()  # Returns latest IMU dict
    """

    TOPIC_IMU    = "digitalspotter/imu"
    TOPIC_RESULT = "digitalspotter/result"

    def __init__(self):
        """Initialize ESP32Bridge."""
        self._connected   = False
        self._is_mock     = not MQTT_AVAILABLE or not Config.MQTT.ENABLED
        self._sample_rate = Config.IMU.SAMPLE_RATE_HZ
        self._frame_count = 0
        self._lock        = threading.Lock()
        self._latest_sample = None
        self._last_data_time = 0.0
        self._client = None

        if self._is_mock:
            logger.info("ESP32Bridge starting in MOCK mode (MQTT disabled or unavailable)")

    # ─── Public API ──────────────────────────────────────────────────────────

    def connect(self) -> bool:
        """Connect to HiveMQ Cloud over TLS MQTT.

        Returns:
            bool: True if connected (or mock mode).
        """
        if self._is_mock:
            self._connected = True
            logger.info("ESP32Bridge connected (MOCK mode)")
            return True

        try:
            self._client = mqtt.Client(
                client_id="digital-spotter-pi",
                protocol=mqtt.MQTTv311,
            )
            self._client.username_pw_set(Config.MQTT.USERNAME, Config.MQTT.PASSWORD)
            self._client.tls_set()  # Uses system default CA certs (works with HiveMQ)

            self._client.on_connect    = self._on_connect
            self._client.on_disconnect = self._on_disconnect
            self._client.on_message    = self._on_message

            self._client.connect(
                Config.MQTT.BROKER,
                Config.MQTT.PORT,
                keepalive=30,
            )
            self._client.loop_start()  # Background thread for MQTT network loop

            # Wait up to 8 seconds for connection
            deadline = time.time() + 8.0
            while not self._connected and time.time() < deadline:
                time.sleep(0.1)

            if self._connected:
                logger.info("ESP32Bridge connected to HiveMQ Cloud (%s:%d)",
                            Config.MQTT.BROKER, Config.MQTT.PORT)
                return True
            else:
                logger.error("ESP32Bridge MQTT connection timed out")
                self._is_mock = True
                return False

        except Exception as e:
            logger.error("ESP32Bridge MQTT connect error: %s — falling back to mock", e)
            self._is_mock = True
            self._connected = True
            return False

    def disconnect(self):
        """Disconnect from HiveMQ Cloud."""
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
        self._connected = False
        logger.info("ESP32Bridge disconnected")

    def read_sample(self) -> dict:
        """Return the latest IMU sample.

        Returns:
            dict: IMU data with keys acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z,
                  v_bar_velocity, p_bar_power, smoothness_jerk, timestamp_ms.
        """
        if not self._connected:
            return self._empty_sample()

        if self._is_mock:
            return self._mock_sample()

        with self._lock:
            if self._latest_sample is not None:
                return self._latest_sample.copy()

        # Real mode but no data received yet
        return self._empty_sample()

    def publish_result(self, is_bad_form: bool, confidence: float):
        """Publish inference result to ESP32 for buzzer feedback.

        This can be disabled if bidirectional MQTT introduces latency.
        Call bridge.disable_result_publishing() to turn off.

        Args:
            is_bad_form: True if bad form detected.
            confidence:  Model confidence 0.0–1.0.
        """
        if self._is_mock or not self._client or not self._connected:
            return
        if not self._result_publishing_enabled:
            return

        payload = json.dumps({
            "bad_form":   is_bad_form,
            "confidence": round(float(confidence), 4),
            "ts":         int(time.time() * 1000),
        })
        self._client.publish(self.TOPIC_RESULT, payload, qos=0)

    def disable_result_publishing(self):
        """Disable sending inference results back to ESP32 (makes bridge publish-only)."""
        self._result_publishing_enabled = False
        logger.info("Result publishing disabled — ESP32 buzzer feedback is OFF")

    def is_connected(self) -> bool:
        """Check connection status."""
        return self._connected

    @property
    def data_is_flowing(self) -> bool:
        """True if real IMU packets arrived in the last 3 seconds (or mock is active)."""
        if self._is_mock:
            return self._connected
        return self._connected and (time.time() - self._last_data_time) < 3.0

    @property
    def is_mock(self) -> bool:
        """True if using mock data."""
        return self._is_mock

    # ─── MQTT Callbacks ──────────────────────────────────────────────────────

    def _on_connect(self, client, userdata, flags, rc):
        """Called when MQTT connection is established."""
        if rc == 0:
            self._connected = True
            client.subscribe(self.TOPIC_IMU, qos=0)
            logger.info("[MQTT] Connected — subscribed to %s", self.TOPIC_IMU)
        else:
            logger.error("[MQTT] Connection refused, rc=%d", rc)

    def _on_disconnect(self, client, userdata, rc):
        """Called when MQTT connection is lost."""
        self._connected = False
        if rc != 0:
            logger.warning("[MQTT] Unexpected disconnect (rc=%d) — auto-reconnecting", rc)

    def _on_message(self, client, userdata, msg):
        """Called for each incoming MQTT message."""
        try:
            data = json.loads(msg.payload.decode("utf-8"))

            sample = {
                "acc_x":            float(data.get("ax", 0.0)),
                "acc_y":            float(data.get("ay", 0.0)),
                "acc_z":            float(data.get("az", 0.0)),
                "gyro_x":           float(data.get("gx", 0.0)),
                "gyro_y":           float(data.get("gy", 0.0)),
                "gyro_z":           float(data.get("gz", 0.0)),
                "v_bar_velocity":   0.0,
                "p_bar_power":      0.0,
                "smoothness_jerk":  0.0,
                "timestamp_ms":     int(data.get("ts", time.time() * 1000)),
            }

            with self._lock:
                self._latest_sample  = sample
                self._last_data_time = time.time()

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning("[MQTT] Malformed IMU packet: %s", e)

    # ─── Mock / Empty Samples ─────────────────────────────────────────────────

    def _mock_sample(self) -> dict:
        """Generate synthetic IMU data simulating a barbell squat."""
        self._frame_count += 1
        t = self._frame_count / self._sample_rate
        phase = math.sin(2 * math.pi * 0.33 * t)

        acc_x = np.random.normal(0, 0.1)
        acc_y = -9.81 + phase * 2.0 + np.random.normal(0, 0.2)
        acc_z = np.random.normal(0, 0.1)
        gyro_x = np.random.normal(0, 5.0)
        gyro_y = np.random.normal(0, 5.0)
        gyro_z = np.random.normal(0, 2.0)
        velocity = abs(phase) * 0.6 + np.random.normal(0, 0.02)
        power = velocity * 80.0 * 9.81
        jerk = np.random.normal(0, 0.5)

        return {
            "acc_x": float(acc_x),   "acc_y": float(acc_y),   "acc_z": float(acc_z),
            "gyro_x": float(gyro_x), "gyro_y": float(gyro_y), "gyro_z": float(gyro_z),
            "v_bar_velocity": float(velocity),
            "p_bar_power":    float(power),
            "smoothness_jerk": float(jerk),
            "timestamp_ms":   int(time.time() * 1000),
        }

    def _empty_sample(self) -> dict:
        """Return zeroed IMU sample."""
        return {
            "acc_x": 0.0, "acc_y": 0.0, "acc_z": 0.0,
            "gyro_x": 0.0, "gyro_y": 0.0, "gyro_z": 0.0,
            "v_bar_velocity": 0.0, "p_bar_power": 0.0,
            "smoothness_jerk": 0.0,
            "timestamp_ms": int(time.time() * 1000),
        }

    # ─── Internal state ───────────────────────────────────────────────────────
    _result_publishing_enabled = True
