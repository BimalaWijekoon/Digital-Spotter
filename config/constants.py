"""
config/constants.py
Purpose: Immutable constants — joint names, angle indices, error categories,
         rep phases, MediaPipe landmark IDs, and WebSocket event names.
         These never change at runtime.
Author: bimalawijekoon
Version: 1.0.0
Last Modified: 2026-06-15
"""

# ──────────────────────────────────────────────────────────────────────
# Joint names used throughout the system
# ──────────────────────────────────────────────────────────────────────
JOINT_NAMES = [
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
    "TRUNK",
]

# ──────────────────────────────────────────────────────────────────────
# Index mapping for joint angles in feature vectors
# ──────────────────────────────────────────────────────────────────────
ANGLE_INDICES = {
    "LEFT_HIP": 0,
    "RIGHT_HIP": 1,
    "LEFT_KNEE": 2,
    "RIGHT_KNEE": 3,
    "LEFT_ANKLE": 4,
    "RIGHT_ANKLE": 5,
    "TRUNK": 6,
}

# ──────────────────────────────────────────────────────────────────────
# Error category labels from the LSTM classifier
# ──────────────────────────────────────────────────────────────────────
ERROR_CATEGORIES = {
    0: "Good Form",
    1: "Knee Valgus",
    2: "Spinal Rounding",
    3: "Excessive Forward Lean",
    4: "Asymmetric Load",
    5: "Hip Shift",
}

# ──────────────────────────────────────────────────────────────────────
# Repetition temporal phases
# ──────────────────────────────────────────────────────────────────────
REP_PHASES = {
    1: "Eccentric/Descent",
    2: "Isometric/Bottom",
    3: "Concentric/Lockout",
}

# ──────────────────────────────────────────────────────────────────────
# MediaPipe Pose landmark indices (subset used for angle computation)
# Full list: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
# ──────────────────────────────────────────────────────────────────────
MEDIAPIPE_LANDMARKS = {
    "LEFT_HIP": 23,
    "RIGHT_HIP": 24,
    "LEFT_KNEE": 25,
    "RIGHT_KNEE": 26,
    "LEFT_ANKLE": 27,
    "RIGHT_ANKLE": 28,
    "LEFT_SHOULDER": 11,
    "RIGHT_SHOULDER": 12,
    "LEFT_HEEL": 29,
    "RIGHT_HEEL": 30,
    "LEFT_FOOT_INDEX": 31,
    "RIGHT_FOOT_INDEX": 32,
    "NOSE": 0,
}

# ──────────────────────────────────────────────────────────────────────
# WebSocket event name constants
# ──────────────────────────────────────────────────────────────────────
WEBSOCKET_EVENTS = {
    "POSE_DATA": "pose_data",
    "INFERENCE_RESULT": "inference_result",
    "REP_COMPLETE": "rep_complete",
    "SESSION_UPDATE": "session_update",
    "SYSTEM_STATUS": "system_status",
    "ERROR": "error",
}

# ──────────────────────────────────────────────────────────────────────
# Feature engineering constants
# ──────────────────────────────────────────────────────────────────────
NUM_LANDMARK_XY_FEATURES = 16  # 8 landmarks x 2 (X, Y)
NUM_ANGLE_FEATURES = 7         # 7 joint angles
NUM_IMU_RAW_FEATURES = 6      # Acc XYZ + Gyro XYZ
NUM_BAR_FEATURES = 3          # Velocity, Power, Jerk
NUM_CONTEXT_FEATURES = 6      # Phase, ExID, Load, Height, Weight, FTR
TOTAL_FEATURES = (
    NUM_LANDMARK_XY_FEATURES
    + NUM_ANGLE_FEATURES
    + NUM_IMU_RAW_FEATURES
    + NUM_BAR_FEATURES
    + NUM_CONTEXT_FEATURES
)  # = 38

# ──────────────────────────────────────────────────────────────────────
# Skeleton connections for drawing overlay
# Pairs of (landmark_a, landmark_b) for line drawing
# ──────────────────────────────────────────────────────────────────────
SKELETON_CONNECTIONS = [
    (11, 12),  # LEFT_SHOULDER → RIGHT_SHOULDER
    (11, 23),  # LEFT_SHOULDER → LEFT_HIP
    (12, 24),  # RIGHT_SHOULDER → RIGHT_HIP
    (23, 24),  # LEFT_HIP → RIGHT_HIP
    (23, 25),  # LEFT_HIP → LEFT_KNEE
    (24, 26),  # RIGHT_HIP → RIGHT_KNEE
    (25, 27),  # LEFT_KNEE → LEFT_ANKLE
    (26, 28),  # RIGHT_KNEE → RIGHT_ANKLE
    (27, 29),  # LEFT_ANKLE → LEFT_HEEL
    (28, 30),  # RIGHT_ANKLE → RIGHT_HEEL
    (29, 31),  # LEFT_HEEL → LEFT_FOOT_INDEX
    (30, 32),  # RIGHT_HEEL → RIGHT_FOOT_INDEX
    (27, 31),  # LEFT_ANKLE → LEFT_FOOT_INDEX
    (28, 32),  # RIGHT_ANKLE → RIGHT_FOOT_INDEX
]
