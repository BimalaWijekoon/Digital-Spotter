"""
config/config.py
Purpose: Master configuration — all settings in one place.
         All other modules import from here. Zero hardcoded values anywhere else.
Author: bimalawijekoon
Version: 1.0.0
Last Modified: 2026-06-15
"""

import os
from pathlib import Path


class _Camera:
    """Camera hardware configuration for IMX219 module."""
    WIDTH = 1280
    HEIGHT = 720
    FPS = 30
    SENSOR = "imx219"


class _Stream:
    """mediamtx WebRTC/RTSP streaming configuration."""
    MEDIAMTX_HOST = os.getenv("DS_STREAM_HOST", "192.168.1.8")
    WEBRTC_PORT = int(os.getenv("DS_WEBRTC_PORT", "8889"))
    RTSP_PORT = int(os.getenv("DS_RTSP_PORT", "8554"))
    STREAM_PATH = "picam"


class _API:
    """Flask API server configuration."""
    HOST = os.getenv("DS_API_HOST", "0.0.0.0")
    PORT = int(os.getenv("DS_API_PORT", "5000"))
    DEBUG = os.getenv("DS_DEBUG", "false").lower() == "true"
    SECRET_KEY = os.getenv(
        "DS_SECRET_KEY", "digital-spotter-dev-key-change-in-prod"
    )


class _Frontend:
    """Frontend static file server configuration."""
    HOST = os.getenv("DS_FRONTEND_HOST", "0.0.0.0")
    PORT = int(os.getenv("DS_FRONTEND_PORT", "8080"))
    STATIC_DIR = "frontend"


class _Paths:
    """All filesystem paths used by the application.
    Uses pathlib.Path for cross-platform compatibility.
    """
    ROOT = Path(__file__).parent.parent
    BASE_DIR = ROOT
    MODELS_DIR = ROOT / "models"
    LOGS_DIR = ROOT / "logs"
    DATA_DIR = ROOT / "data"
    FRONTEND_DIR = ROOT / "frontend"
    MODEL_TFLITE = MODELS_DIR / "model_quantized.tflite"
    MODEL_FULLPREC = MODELS_DIR / "model_fullprec.tflite"
    SCALER_PKL = MODELS_DIR / "scaler.pkl"
    DATABASE = DATA_DIR / "sessions.db"
    SCHEMA_SQL = ROOT / "database" / "schema.sql"


class _MediaPipe:
    """MediaPipe BlazePose model configuration."""
    MODEL_COMPLEXITY = 2  # Heavy model
    MIN_DETECTION_CONFIDENCE = 0.7
    MIN_TRACKING_CONFIDENCE = 0.7
    VISIBILITY_THRESHOLD = 0.15


class _Inference:
    """LSTM inference pipeline configuration."""
    SEQUENCE_LENGTH = 3  # 3 phases per rep
    NUM_FEATURES = 38  # Feature vector dimensionality
    DECISION_THRESHOLD = 0.5  # Sigmoid cutoff
    BAD_FORM_LABEL = "Bad Form"
    GOOD_FORM_LABEL = "Good Form"


class _RepSegmentation:
    """Rep segmentation algorithm parameters."""
    SAVITZKY_GOLAY_WINDOW_RATIO = 0.4
    POLYNOMIAL_ORDER = 2
    PROMINENCE_THRESHOLD = 15
    MIN_INTER_PEAK_RATIO = 1.2
    PHASE_BOUNDARIES = [0.35, 0.45]


class _Exercises:
    """Supported exercise types."""
    SQUAT = {"id": 0, "name": "Barbell Back Squat"}
    DEADLIFT = {"id": 1, "name": "Conventional Deadlift"}

    @classmethod
    def get_by_id(cls, exercise_id):
        """Get exercise dict by its numeric ID.

        Args:
            exercise_id: Integer exercise identifier (0=Squat, 1=Deadlift).

        Returns:
            dict with 'id' and 'name' keys, or None if not found.
        """
        for attr in [cls.SQUAT, cls.DEADLIFT]:
            if attr["id"] == exercise_id:
                return attr
        return None

    @classmethod
    def get_name(cls, exercise_id):
        """Get exercise name by its numeric ID.

        Args:
            exercise_id: Integer exercise identifier.

        Returns:
            Exercise name string, or 'Unknown' if not found.
        """
        ex = cls.get_by_id(exercise_id)
        return ex["name"] if ex else "Unknown"


class _IMU:
    """IMU configuration."""
    BLE_ADDRESS = os.getenv("DS_BLE_ADDRESS", "00:11:22:33:44:55")
    SAMPLE_RATE_HZ = int(os.getenv("DS_IMU_SAMPLE_RATE", "50"))


class _MQTT:
    """MQTT configuration for ESP32-S3 IMU data."""
    BROKER = os.getenv("DS_MQTT_BROKER", "localhost")
    PORT = int(os.getenv("DS_MQTT_PORT", "1883"))
    TOPIC = "digitalspotter/imu"
    ENABLED = os.getenv("DS_MQTT_ENABLED", "false").lower() == "true"


class _Logging:
    """Application logging configuration."""
    LEVEL = os.getenv("DS_LOG_LEVEL", "INFO")
    FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    FILE = "logs/app.log"
    MAX_BYTES = 10485760  # 10 MB
    BACKUP_COUNT = 5


class Config:
    """Master configuration class.

    Usage:
        from config.config import Config
        port = Config.API.PORT
        model_path = Config.PATHS.MODEL_TFLITE
    """
    VERSION = "1.0.0"
    CAMERA = _Camera()
    STREAM = _Stream()
    API = _API()
    FRONTEND = _Frontend()
    PATHS = _Paths()
    MEDIAPIPE = _MediaPipe()
    INFERENCE = _Inference()
    REP_SEGMENTATION = _RepSegmentation()
    EXERCISES = _Exercises()
    IMU = _IMU()
    MQTT = _MQTT()
    LOGGING = _Logging()
