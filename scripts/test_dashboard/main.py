"""
scripts/test_dashboard/main.py
Purpose: Pi-targeted test dashboard backend — FastAPI server for manual
         inference testing with the v4 BiLSTM+MHA model and GPIO buzzer.
         Replaces the old PC-only TEST DASBOARD/main.py.
Author: bimalawijekoon
Version: 2.0.0
Last Modified: 2026-06-25
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import numpy as np
import logging
import time
import os
from pathlib import Path
from typing import List, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import Config
from config.constants import FEATURE_ORDER, TOTAL_FEATURES, SEQUENCE_LENGTH_MODEL
from inference.lstm_runner import LSTMRunner
from inference.preprocessor import Preprocessor
from hardware.buzzer import Buzzer
from hardware.rgb_led import RgbLed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Pydantic request models — matches v4 40-feature contract
# ─────────────────────────────────────────────────────────────────

class SubjectInfo(BaseModel):
    height_cm: float = Field(175.0, description="Subject height in cm")
    weight_kg: float = Field(75.0, description="Subject weight in kg")
    femur_tibia_ratio: float = Field(1.20, description="Femur-to-tibia ratio")


class SessionInfo(BaseModel):
    exercise_id: int = Field(0, description="0=Squat, 1=Deadlift")
    load_kg: float = Field(60.0, description="Barbell load in kg")


class PhaseData(BaseModel):
    """One phase of sensor data — 32 raw features (context + vision + angles
    + IMU + bar performance). The remaining 8 engineered features are
    computed server-side from these inputs."""
    rep_phase: int = Field(..., ge=1, le=3)

    # Vision joints (16)
    vis_l_shoulder_x: float = 0.0
    vis_l_shoulder_y: float = 0.0
    vis_r_shoulder_x: float = 0.0
    vis_r_shoulder_y: float = 0.0
    vis_l_hip_x: float = 0.0
    vis_l_hip_y: float = 0.0
    vis_r_hip_x: float = 0.0
    vis_r_hip_y: float = 0.0
    vis_l_knee_x: float = 0.0
    vis_l_knee_y: float = 0.0
    vis_r_knee_x: float = 0.0
    vis_r_knee_y: float = 0.0
    vis_l_ankle_x: float = 0.0
    vis_l_ankle_y: float = 0.0
    vis_r_ankle_x: float = 0.0
    vis_r_ankle_y: float = 0.0

    # Trunk angle (1)
    angle_trunk_inclination: float = 0.0

    # IMU (6)
    imu_acc_x: float = 0.0
    imu_acc_y: float = 0.0
    imu_acc_z: float = 0.0
    imu_gyro_x: float = 0.0
    imu_gyro_y: float = 0.0
    imu_gyro_z: float = 0.0

    # Bar performance (3)
    v_bar_velocity_vertical: float = 0.0
    p_bar_power_watts: float = 0.0
    smoothness_jerk: float = 0.0

    # Angle asymmetries (3)
    hip_flexion_asymmetry: float = 0.0
    knee_flexion_asymmetry: float = 0.0
    ankle_dorsiflexion_asymmetry: float = 0.0

    # Engineered features (4) — can be overridden, otherwise computed
    imu_acc_magnitude: Optional[float] = None
    knee_hip_coupling_l: Optional[float] = None
    knee_hip_coupling_r: Optional[float] = None
    velocity_decel_ratio: Optional[float] = None


class PredictionRequest(BaseModel):
    subject: SubjectInfo
    session: SessionInfo
    phases: List[PhaseData] = Field(..., min_length=3, max_length=3)


# ─────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Digital Spotter — Test Dashboard API",
    description="Pi-targeted inference testing with v4 BiLSTM+MHA model and GPIO buzzer",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
_runner: LSTMRunner = None
_preprocessor: Preprocessor = None
_buzzer: Buzzer = None
_rgb: RgbLed = None


@app.on_event("startup")
async def startup():
    """Load model, preprocessor, and initialize buzzer at startup."""
    global _runner, _preprocessor, _buzzer, _rgb

    logger.info("Loading v4 inference pipeline...")

    _runner = LSTMRunner()
    _runner.load()
    logger.info("✓ LSTMRunner loaded (mock=%s)", _runner.is_mock)

    _preprocessor = Preprocessor()
    _preprocessor.load()
    logger.info("✓ Preprocessor loaded (mock=%s)", _preprocessor.is_mock)

    _buzzer = Buzzer()
    logger.info("✓ Buzzer initialized (mock=%s, pin=%d)", _buzzer.is_mock, Config.BUZZER.GPIO_PIN)

    _rgb = RgbLed()
    _rgb.idle()
    logger.info("✓ RGB LED initialized (mock=%s, pins=%d,%d,%d)", 
                _rgb.is_mock, Config.RGB_LED.PIN_R, Config.RGB_LED.PIN_G, Config.RGB_LED.PIN_B)

    logger.info("Dashboard ready — v4 model, %d features, threshold=%.3f",
                TOTAL_FEATURES, Config.INFERENCE.DECISION_THRESHOLD)


@app.on_event("shutdown")
async def shutdown():
    """Clean up GPIO on shutdown."""
    if _buzzer:
        _buzzer.cleanup()
    if _rgb:
        _rgb.cleanup()


# ─────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────

@app.get("/")
async def serve_dashboard():
    """Serve the dashboard HTML."""
    index_path = Path(__file__).parent / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path), media_type="text/html")
    return {"error": "index.html not found", "expected": str(index_path)}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ready" if _runner and _runner.is_loaded() else "degraded",
        "model_loaded": _runner.is_loaded() if _runner else False,
        "model_mock": _runner.is_mock if _runner else True,
        "preprocessor_loaded": _preprocessor.is_loaded() if _preprocessor else False,
        "preprocessor_mock": _preprocessor.is_mock if _preprocessor else True,
        "buzzer_enabled": _buzzer.is_enabled if _buzzer else False,
        "buzzer_mock": _buzzer.is_mock if _buzzer else True,
        "buzzer_pin": Config.BUZZER.GPIO_PIN,
        "rgb_enabled": _rgb.is_enabled if _rgb else False,
        "rgb_mock": _rgb.is_mock if _rgb else True,
        "threshold": Config.INFERENCE.DECISION_THRESHOLD,
        "features": TOTAL_FEATURES,
        "sequence_length": SEQUENCE_LENGTH_MODEL,
    }


@app.post("/predict")
async def predict(request: PredictionRequest):
    """Run v4 inference on 3 phases of input data.

    Assembles the 40-feature vectors, computes the delta row (phase3 - phase1),
    runs winsorize → impute → scale preprocessing, then feeds the (4, 40)
    sequence to LSTMRunner. Fires buzzer on bad form detection.

    Returns:
        JSON with label, confidence, probability, latency, buzzer_fired flag.
    """
    if not _runner or not _runner.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    if _rgb and _rgb.is_enabled:
        _rgb.processing()

    start_time = time.perf_counter()

    # Build (3, 40) raw feature matrix from the 3 phases
    raw_phases = []
    for phase in request.phases:
        bmi = request.subject.weight_kg / ((request.subject.height_cm / 100.0) ** 2)

        # Compute engineered features if not provided
        acc_mag = phase.imu_acc_magnitude
        if acc_mag is None:
            acc_mag = float(np.sqrt(
                phase.imu_acc_x**2 + phase.imu_acc_y**2 + phase.imu_acc_z**2
            ))

        # Knee-hip coupling: knee_angle * hip_y (proxy for depth-load coupling)
        khc_l = phase.knee_hip_coupling_l
        if khc_l is None:
            khc_l = phase.vis_l_knee_y * phase.vis_l_hip_y

        khc_r = phase.knee_hip_coupling_r
        if khc_r is None:
            khc_r = phase.vis_r_knee_y * phase.vis_r_hip_y

        vdr = phase.velocity_decel_ratio
        if vdr is None:
            vdr = 1.0  # placeholder — real value computed after all phases

        # Assemble in EXACT FEATURE_ORDER
        feature_vec = [
            request.subject.height_cm,               # Subject_Height_cm
            request.subject.weight_kg,                # Subject_Weight_kg
            float(request.session.exercise_id),       # Exercise_ID
            float(phase.rep_phase),                   # Rep_Phase
            request.session.load_kg,                  # Load_kg
            phase.vis_l_shoulder_x,                   # Vis_L_Shoulder_X
            phase.vis_l_shoulder_y,                   # Vis_L_Shoulder_Y
            phase.vis_r_shoulder_x,                   # Vis_R_Shoulder_X
            phase.vis_r_shoulder_y,                   # Vis_R_Shoulder_Y
            phase.vis_l_hip_x,                        # Vis_L_Hip_X
            phase.vis_l_hip_y,                        # Vis_L_Hip_Y
            phase.vis_r_hip_x,                        # Vis_R_Hip_X
            phase.vis_r_hip_y,                        # Vis_R_Hip_Y
            phase.vis_l_knee_x,                       # Vis_L_Knee_X
            phase.vis_l_knee_y,                       # Vis_L_Knee_Y
            phase.vis_r_knee_x,                       # Vis_R_Knee_X
            phase.vis_r_knee_y,                       # Vis_R_Knee_Y
            phase.vis_l_ankle_x,                      # Vis_L_Ankle_X
            phase.vis_l_ankle_y,                      # Vis_L_Ankle_Y
            phase.vis_r_ankle_x,                      # Vis_R_Ankle_X
            phase.vis_r_ankle_y,                      # Vis_R_Ankle_Y
            phase.angle_trunk_inclination,            # Angle_Trunk_Inclination
            phase.imu_acc_x,                          # IMU_Acc_X
            phase.imu_acc_y,                          # IMU_Acc_Y
            phase.imu_acc_z,                          # IMU_Acc_Z
            phase.imu_gyro_x,                         # IMU_Gyro_X
            phase.imu_gyro_y,                         # IMU_Gyro_Y
            phase.imu_gyro_z,                         # IMU_Gyro_Z
            phase.v_bar_velocity_vertical,            # V_Bar_Velocity_Vertical
            phase.p_bar_power_watts,                  # P_Bar_Power_Watts
            phase.smoothness_jerk,                    # Smoothness_Jerk
            phase.hip_flexion_asymmetry,              # Hip_Flexion_Asymmetry
            phase.knee_flexion_asymmetry,             # Knee_Flexion_Asymmetry
            phase.ankle_dorsiflexion_asymmetry,       # Ankle_Dorsiflexion_Asymmetry
            acc_mag,                                  # IMU_Acc_Magnitude
            khc_l,                                    # Knee_Hip_Coupling_L
            khc_r,                                    # Knee_Hip_Coupling_R
            vdr,                                      # Velocity_Decel_Ratio
            request.subject.femur_tibia_ratio,        # Femur_Tibia_Ratio
            bmi,                                      # BMI
        ]
        raw_phases.append(feature_vec)

    # Stack to (3, 40)
    X_raw = np.array(raw_phases, dtype=np.float32)

    # Compute real Velocity_Decel_Ratio now that all phases are available
    vdr_idx = FEATURE_ORDER.index("Velocity_Decel_Ratio")
    vel_idx = FEATURE_ORDER.index("V_Bar_Velocity_Vertical")
    phase1_vel = X_raw[0, vel_idx]
    phase3_vel = X_raw[2, vel_idx]
    real_vdr = phase3_vel / (abs(phase1_vel) + 1e-6)
    X_raw[:, vdr_idx] = real_vdr

    # Compute delta row (phase3 - phase1) — this is the 4th timestep
    delta = X_raw[2] - X_raw[0]
    X_model = np.vstack([X_raw, delta.reshape(1, -1)])  # (4, 40)

    # Run inference (LSTMRunner handles preprocessing internally)
    result = _runner.predict(X_model)

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Fire buzzer and LED based on outcome
    buzzer_fired = False
    if result.is_bad_form:
        if _buzzer and _buzzer.is_enabled:
            _buzzer.beep_bad_form()
            buzzer_fired = True
        if _rgb and _rgb.is_enabled:
            _rgb.bad_form()
        logger.info("🔴 BAD FORM detected (%.1f%%)", result.confidence * 100)
    else:
        if _rgb and _rgb.is_enabled:
            _rgb.good_form()
        logger.info("🟢 %s (%.1f%%)", result.label, result.confidence * 100)

    return {
        "label": result.label,
        "confidence": round(result.confidence * 100, 1),
        "probability": {
            "good_form": round((1.0 - result.confidence) * 100, 1),
            "bad_form": round(result.confidence * 100, 1),
        },
        "is_bad_form": result.is_bad_form,
        "threshold": Config.INFERENCE.DECISION_THRESHOLD,
        "latency_ms": round(elapsed_ms, 2),
        "is_mock": result.is_mock,
        "buzzer_fired": buzzer_fired,
    }


@app.post("/buzzer/test")
async def buzzer_test():
    """Fire a single test beep to verify buzzer hardware."""
    if not _buzzer:
        raise HTTPException(status_code=503, detail="Buzzer not initialized")
    _buzzer.beep_test()
    return {
        "status": "beeped",
        "pin": Config.BUZZER.GPIO_PIN,
        "is_mock": _buzzer.is_mock,
    }


@app.post("/buzzer/bad-form")
async def buzzer_bad_form():
    """Fire the full bad-form triple beep pattern."""
    if not _buzzer:
        raise HTTPException(status_code=503, detail="Buzzer not initialized")
    _buzzer.beep_bad_form()
    return {
        "status": "triple_beep",
        "pattern": Config.BUZZER.BAD_FORM_PATTERN,
        "is_mock": _buzzer.is_mock,
    }


@app.post("/rgb/test")
async def rgb_test(color: str = "white"):
    """Test RGB LED colors. Valid colors: red, green, blue, amber, cyan, purple, white, off, idle, processing, bad_form, good_form"""
    if not _rgb:
        raise HTTPException(status_code=503, detail="RGB not initialized")
    
    color = color.lower()
    if color == "red": _rgb.set_color(_rgb.RED)
    elif color == "green": _rgb.set_color(_rgb.GREEN)
    elif color == "blue": _rgb.set_color(_rgb.BLUE)
    elif color == "amber": _rgb.set_color(_rgb.AMBER)
    elif color == "cyan": _rgb.set_color(_rgb.CYAN)
    elif color == "purple": _rgb.set_color(_rgb.PURPLE)
    elif color == "white": _rgb.set_color(_rgb.WHITE)
    elif color == "off": _rgb.off()
    elif color == "idle": _rgb.idle()
    elif color == "processing": _rgb.processing()
    elif color == "bad_form": _rgb.bad_form()
    elif color == "good_form": _rgb.good_form()
    else: raise HTTPException(status_code=400, detail="Invalid color")
    
    return {
        "status": "color_set",
        "color": color,
        "is_mock": _rgb.is_mock,
    }


@app.get("/feature-order")
async def feature_order():
    """Return the exact feature order for debugging."""
    return {
        "feature_order": FEATURE_ORDER,
        "total_features": TOTAL_FEATURES,
        "sequence_length": SEQUENCE_LENGTH_MODEL,
    }


if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("Digital Spotter — Test Dashboard (v4 BiLSTM+MHA)")
    print(f"Features: {TOTAL_FEATURES} × {SEQUENCE_LENGTH_MODEL} timesteps")
    print(f"Threshold: {Config.INFERENCE.DECISION_THRESHOLD}")
    print(f"Buzzer: GPIO {Config.BUZZER.GPIO_PIN} ({'enabled' if Config.BUZZER.ENABLED else 'disabled'})")
    print(f"RGB LED: GPIO {Config.RGB_LED.PIN_R},{Config.RGB_LED.PIN_G},{Config.RGB_LED.PIN_B} ({'enabled' if Config.RGB_LED.ENABLED else 'disabled'})")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)
