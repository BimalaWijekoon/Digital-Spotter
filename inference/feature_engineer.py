"""
inference/feature_engineer.py
Purpose: Transforms raw angles + IMU data into the exact 38 features
         used in training. Feature order MUST match training pipeline.
Author: bimalawijekoon
Version: 1.0.0
Last Modified: 2026-06-15
"""

import logging
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.constants import (
    NUM_LANDMARK_XY_FEATURES,
    NUM_ANGLE_FEATURES,
    NUM_IMU_RAW_FEATURES,
    NUM_BAR_FEATURES,
    NUM_CONTEXT_FEATURES,
    TOTAL_FEATURES,
)

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Transform raw pose/IMU data into the 38-feature training vector.

    Feature groups (in order):
        1. Vision XY landmarks (16): L/R Hip/Knee/Shoulder/Ankle X and Y
        2. Computed angles (7): bilateral mean + L-R asymmetry + trunk
        3. IMU raw (6): Acc XYZ + Gyro XYZ (zeros until ESP32 connected)
        4. Bar performance (3): Velocity, Power, Jerk (zeros until IMU)
        5. Context (6): Phase, ExID, Load, Height, Weight, FTR
    """

    # Landmarks used for XY extraction (order matters!)
    _XY_LANDMARKS = [
        "LEFT_HIP", "RIGHT_HIP",
        "LEFT_KNEE", "RIGHT_KNEE",
        "LEFT_SHOULDER", "RIGHT_SHOULDER",
        "LEFT_ANKLE", "RIGHT_ANKLE",
    ]

    def engineer(self, landmarks, angles, imu_data=None, context=None):
        """Build the complete 38-feature vector.

        Args:
            landmarks: Dict mapping landmark name to
                (x, y, z, visibility).
            angles: Dict mapping joint name to angle in degrees.
            imu_data: Optional dict with keys 'acc_x', 'acc_y', 'acc_z',
                'gyro_x', 'gyro_y', 'gyro_z'. None = zeros.
            context: Optional dict with keys 'rep_phase', 'exercise_id',
                'load_kg', 'height_cm', 'weight_kg', 'ftr'.
                None = defaults.

        Returns:
            numpy.ndarray: Shape (38,) feature vector, float32.
        """
        features = np.zeros(TOTAL_FEATURES, dtype=np.float32)
        idx = 0

        # Group 1: Vision XY landmarks (16 features)
        xy_features = self._extract_landmark_xy(landmarks)
        features[idx:idx + NUM_LANDMARK_XY_FEATURES] = xy_features
        idx += NUM_LANDMARK_XY_FEATURES

        # Group 2: Computed angles (7 features)
        angle_features = self._compute_angle_features(angles)
        features[idx:idx + NUM_ANGLE_FEATURES] = angle_features
        idx += NUM_ANGLE_FEATURES

        # Group 3: IMU raw (6 features)
        imu_features = self._extract_imu_features(imu_data)
        features[idx:idx + NUM_IMU_RAW_FEATURES] = imu_features
        idx += NUM_IMU_RAW_FEATURES

        # Group 4: Bar performance (3 features)
        bar_features = self._extract_bar_features(imu_data)
        features[idx:idx + NUM_BAR_FEATURES] = bar_features
        idx += NUM_BAR_FEATURES

        # Group 5: Context (6 features)
        ctx_features = self._extract_context_features(context)
        features[idx:idx + NUM_CONTEXT_FEATURES] = ctx_features
        idx += NUM_CONTEXT_FEATURES

        assert idx == TOTAL_FEATURES, (
            f"Feature count mismatch: expected {TOTAL_FEATURES}, got {idx}"
        )

        return features

    def _extract_landmark_xy(self, landmarks):
        """Extract 16 XY coordinates from 8 landmarks.

        Args:
            landmarks: Dict of {name: (x, y, z, visibility)}.

        Returns:
            numpy.ndarray: Shape (16,) — [x1, y1, x2, y2, ...].
        """
        xy = np.zeros(NUM_LANDMARK_XY_FEATURES, dtype=np.float32)
        for i, name in enumerate(self._XY_LANDMARKS):
            if name in landmarks:
                x, y = landmarks[name][0], landmarks[name][1]
                xy[i * 2] = x
                xy[i * 2 + 1] = y
        return xy

    def _compute_angle_features(self, angles):
        """Compute the 7 angle features.

        For bilateral joints: uses the mean of L and R.
        Includes L-R asymmetry encoded in the raw angles.

        Args:
            angles: Dict of {joint_name: angle_degrees}.

        Returns:
            numpy.ndarray: Shape (7,) angle features.
        """
        return np.array([
            angles.get("LEFT_HIP", 0.0),
            angles.get("RIGHT_HIP", 0.0),
            angles.get("LEFT_KNEE", 0.0),
            angles.get("RIGHT_KNEE", 0.0),
            angles.get("LEFT_ANKLE", 0.0),
            angles.get("RIGHT_ANKLE", 0.0),
            angles.get("TRUNK", 0.0),
        ], dtype=np.float32)

    def _compute_asymmetry_features(self, angles):
        """Compute left-right asymmetry features.

        Args:
            angles: Dict of {joint_name: angle_degrees}.

        Returns:
            numpy.ndarray: Shape (3,) — hip, knee, ankle asymmetry.
        """
        hip_asym = abs(
            angles.get("LEFT_HIP", 0) - angles.get("RIGHT_HIP", 0)
        )
        knee_asym = abs(
            angles.get("LEFT_KNEE", 0) - angles.get("RIGHT_KNEE", 0)
        )
        ankle_asym = abs(
            angles.get("LEFT_ANKLE", 0) - angles.get("RIGHT_ANKLE", 0)
        )
        return np.array(
            [hip_asym, knee_asym, ankle_asym], dtype=np.float32
        )

    def _compute_bilateral_mean(self, angles):
        """Compute bilateral mean for hip, knee, ankle.

        Args:
            angles: Dict of {joint_name: angle_degrees}.

        Returns:
            numpy.ndarray: Shape (3,) — mean hip, knee, ankle.
        """
        hip_mean = (
            angles.get("LEFT_HIP", 0) + angles.get("RIGHT_HIP", 0)
        ) / 2.0
        knee_mean = (
            angles.get("LEFT_KNEE", 0) + angles.get("RIGHT_KNEE", 0)
        ) / 2.0
        ankle_mean = (
            angles.get("LEFT_ANKLE", 0) + angles.get("RIGHT_ANKLE", 0)
        ) / 2.0
        return np.array(
            [hip_mean, knee_mean, ankle_mean], dtype=np.float32
        )

    def _extract_imu_features(self, imu_data):
        """Extract 6 raw IMU features.

        Args:
            imu_data: Optional dict with acc/gyro values.
                None returns zeros (placeholder).

        Returns:
            numpy.ndarray: Shape (6,) — [acc_x, acc_y, acc_z,
                gyro_x, gyro_y, gyro_z].
        """
        # TODO: PHASE_8 - Replace with actual IMU data from ESP32-S3
        if imu_data is None:
            return np.zeros(NUM_IMU_RAW_FEATURES, dtype=np.float32)

        return np.array([
            imu_data.get("acc_x", 0.0),
            imu_data.get("acc_y", 0.0),
            imu_data.get("acc_z", 0.0),
            imu_data.get("gyro_x", 0.0),
            imu_data.get("gyro_y", 0.0),
            imu_data.get("gyro_z", 0.0),
        ], dtype=np.float32)

    def _extract_bar_features(self, imu_data):
        """Extract 3 bar performance features.

        Args:
            imu_data: Optional dict with bar velocity/power/jerk.
                None returns zeros (placeholder).

        Returns:
            numpy.ndarray: Shape (3,) — [velocity, power, jerk].
        """
        # TODO: PHASE_8 - Compute from IMU when available
        if imu_data is None:
            return np.zeros(NUM_BAR_FEATURES, dtype=np.float32)

        return np.array([
            imu_data.get("v_bar_velocity", 0.0),
            imu_data.get("p_bar_power", 0.0),
            imu_data.get("smoothness_jerk", 0.0),
        ], dtype=np.float32)

    def _extract_context_features(self, context):
        """Extract 6 context features.

        Args:
            context: Optional dict with session context.
                None returns defaults.

        Returns:
            numpy.ndarray: Shape (6,) — [phase, exercise_id, load_kg,
                height_cm, weight_kg, ftr].
        """
        if context is None:
            context = {}

        return np.array([
            context.get("rep_phase", 0),
            context.get("exercise_id", 0),
            context.get("load_kg", 0.0),
            context.get("height_cm", 175.0),
            context.get("weight_kg", 75.0),
            context.get("ftr", 0.0),
        ], dtype=np.float32)
