"""
api/routes.py
Purpose: All REST API endpoints as Flask Blueprint.
Author: bimalawijekoon
Version: 1.0.0
Last Modified: 2026-06-15
"""

import logging
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Blueprint, jsonify, request

from config.config import Config
from config.constants import WEBSOCKET_EVENTS
from database.queries import (
    create_session, end_session, get_session,
    get_session_reps, get_recent_sessions, get_session_stats,
)
from inference.lstm_runner import LSTMRunner, InferenceResult

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__, url_prefix="/api")

# Shared state — will be set by integration module
_frame_processor = None
_lstm_runner = LSTMRunner()


def set_frame_processor(fp):
    """Set the shared FrameProcessor instance.
    Args:
        fp: FrameProcessor instance from the integration layer.
    """
    global _frame_processor
    _frame_processor = fp


@api_bp.route("/health", methods=["GET"])
def health():
    """System health check endpoint.
    Returns:
        JSON with status, camera, model, and uptime info.
    """
    from api.app import get_uptime
    model_loaded = _lstm_runner.is_loaded()
    camera_ok = _frame_processor is not None

    if model_loaded and camera_ok:
        status = "ok"
    elif model_loaded or camera_ok:
        status = "degraded"
    else:
        status = "degraded"

    return jsonify({
        "status": status,
        "camera": camera_ok,
        "model": model_loaded,
        "uptime_s": get_uptime(),
    })


@api_bp.route("/status", methods=["GET"])
def status():
    """Full system status endpoint.
    Returns:
        JSON with streaming state, session info, fps, latency.
    """
    fps = 0.0
    latency = 0.0
    if _frame_processor:
        latest = _frame_processor.get_latest()
        if latest:
            fps = latest.fps
            latency = latest.processing_time_ms

    return jsonify({
        "streaming": _frame_processor is not None,
        "session_active": False,
        "current_session_id": None,
        "fps": round(fps, 1),
        "latency_ms": round(latency, 2),
    })


@api_bp.route("/session/start", methods=["POST"])
def session_start():
    """Start a new training session.
    Body:
        exercise_id (int): 0=Squat, 1=Deadlift.
        notes (str, optional): Session notes.
    Returns:
        JSON with session_id and started_at.
    """
    data = request.get_json(silent=True) or {}
    exercise_id = data.get("exercise_id", 0)
    notes = data.get("notes")

    exercise_name = Config.EXERCISES.get_name(exercise_id)
    session_id = create_session(exercise_id, exercise_name, notes)

    if _frame_processor:
        _frame_processor.reset_session()

    session = get_session(session_id)
    return jsonify({
        "session_id": session_id,
        "started_at": session.started_at if session else "",
    }), 201


@api_bp.route("/session/stop", methods=["POST"])
def session_stop():
    """End current session and save stats.
    Body:
        session_id (int): Session to end.
    Returns:
        JSON with session_id and final rep counts.
    """
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id")

    if not session_id:
        return jsonify({"error": "session_id required"}), 400

    success = end_session(session_id)
    if not success:
        return jsonify({"error": "Session not found"}), 404

    session = get_session(session_id)
    return jsonify({
        "session_id": session_id,
        "total_reps": session.total_reps,
        "good_reps": session.good_reps,
        "bad_reps": session.bad_reps,
    })


@api_bp.route("/session/<int:session_id>", methods=["GET"])
def session_detail(session_id):
    """Get session details and rep history.
    Args:
        session_id: Session ID from URL path.
    Returns:
        JSON with session object and reps list.
    """
    session = get_session(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    reps = get_session_reps(session_id)
    stats = get_session_stats(session_id)

    return jsonify({
        "session": session.to_dict(),
        "reps": [r.to_dict() for r in reps],
        "stats": stats,
    })


@api_bp.route("/sessions", methods=["GET"])
def sessions_list():
    """List recent sessions.
    Query params:
        limit (int): Max sessions to return (default 10).
    Returns:
        JSON with sessions list.
    """
    limit = request.args.get("limit", 10, type=int)
    sessions = get_recent_sessions(limit=limit)

    return jsonify({
        "sessions": [s.to_dict() for s in sessions],
    })


@api_bp.route("/inference/test", methods=["POST"])
def inference_test():
    """Test inference with dummy data — for development.
    Returns:
        JSON with label, confidence, latency.
    """
    _lstm_runner.load()
    dummy_seq = np.random.randn(3, 38).astype(np.float32)
    result = _lstm_runner.predict(dummy_seq)

    return jsonify(result.to_dict())
