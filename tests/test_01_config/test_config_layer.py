"""
tests/test_01_config/test_config_layer.py
Purpose: Test Config and Constants immutability and values.
Author: bimalawijekoon
Version: 1.0.0
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import Config
from config.constants import (
    JOINT_NAMES, MEDIAPIPE_LANDMARKS, REP_PHASES,
    WEBSOCKET_EVENTS, TOTAL_FEATURES,
)


class TestConfig:
    def test_version_exists(self):
        assert hasattr(Config, 'VERSION')
        assert Config.VERSION == "1.0.0"

    def test_paths_exist(self):
        assert hasattr(Config.PATHS, 'BASE_DIR')
        assert hasattr(Config.PATHS, 'DATABASE')

    def test_api_defaults(self):
        assert Config.API.PORT == 5000
        assert Config.API.HOST == "0.0.0.0"

    def test_camera_settings(self):
        assert Config.CAMERA.FPS == 30
        assert Config.CAMERA.WIDTH == 1280
        assert Config.CAMERA.HEIGHT == 720

    def test_inference_settings(self):
        assert Config.INFERENCE.NUM_FEATURES == 38
        assert Config.INFERENCE.SEQUENCE_LENGTH == 3

    def test_exercise_name_lookup(self):
        assert Config.EXERCISES.get_name(0) == "Barbell Back Squat"
        assert Config.EXERCISES.get_name(1) == "Conventional Deadlift"


class TestConstants:
    def test_joint_names_7(self):
        assert len(JOINT_NAMES) == 7

    def test_mediapipe_landmarks(self):
        assert "LEFT_HIP" in MEDIAPIPE_LANDMARKS
        assert MEDIAPIPE_LANDMARKS["LEFT_HIP"] == 23

    def test_rep_phases(self):
        assert len(REP_PHASES) == 3
        assert REP_PHASES[1] == "Eccentric/Descent"

    def test_websocket_events(self):
        assert "POSE_DATA" in WEBSOCKET_EVENTS
        assert WEBSOCKET_EVENTS["POSE_DATA"] == "pose_data"

    def test_total_features(self):
        assert TOTAL_FEATURES == 38
