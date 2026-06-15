"""
integration/session_manager.py
Purpose: Orchestrates session lifecycle — ties together database,
         vision pipeline, and inference engine into coherent sessions.
Author: bimalawijekoon
Version: 1.0.0
Last Modified: 2026-06-15
"""

import logging
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import Config
from database.queries import (
    create_session, end_session, get_session, insert_rep,
    get_session_stats,
)
from inference.feature_engineer import FeatureEngineer
from inference.sequence_buffer import SequenceBuffer
from inference.lstm_runner import LSTMRunner
from vision.frame_processor import FrameProcessor

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages a complete training session lifecycle.

    Coordinates:
    - FrameProcessor (pose + angles + rep segmentation)
    - FeatureEngineer (38-dim feature vector)
    - SequenceBuffer (3-phase accumulation)
    - LSTMRunner (inference)
    - Database (session + rep storage)
    """

    def __init__(self, socketio=None):
        """Initialize SessionManager.

        Args:
            socketio: Flask-SocketIO instance for emitting events.
        """
        self._socketio = socketio
        self._frame_processor = FrameProcessor()
        self._feature_engineer = FeatureEngineer()
        self._sequence_buffer = SequenceBuffer()
        self._lstm_runner = LSTMRunner()

        self._session_id = None
        self._exercise_id = 0
        self._exercise_name = ""
        self._is_active = False
        self._rep_count = 0

        # Load model
        self._lstm_runner.load()

        logger.info("SessionManager initialized")

    def start_session(self, exercise_id=0, notes=None):
        """Start a new training session.

        Args:
            exercise_id: Exercise type (0=Squat, 1=Deadlift).
            notes: Optional notes.

        Returns:
            int: New session ID.

        Side Effects:
            Creates DB record, resets pipeline, starts processing.
        """
        self._exercise_id = exercise_id
        self._exercise_name = Config.EXERCISES.get_name(exercise_id)

        self._session_id = create_session(
            exercise_id, self._exercise_name, notes
        )
        self._is_active = True
        self._rep_count = 0

        # Reset pipeline state
        self._frame_processor.reset_session()
        self._sequence_buffer.reset()

        logger.info(
            "Session %d started (exercise=%s)",
            self._session_id, self._exercise_name
        )

        return self._session_id

    def stop_session(self):
        """Stop the active session.

        Returns:
            dict: Final session statistics.

        Side Effects:
            Closes DB record, stops processing.
        """
        if not self._is_active:
            return {}

        self._is_active = False
        end_session(self._session_id)

        stats = get_session_stats(self._session_id)

        logger.info(
            "Session %d ended (reps=%d, good=%d, bad=%d)",
            self._session_id, stats["total_reps"],
            stats["good_reps"], stats["bad_reps"]
        )

        session_id = self._session_id
        self._session_id = None

        return stats

    def process_frame_result(self, frame_result):
        """Process a completed frame through the inference pipeline.

        Called by the frame processing loop for each frame. If a rep
        phase event occurs, engineers features and accumulates in the
        buffer. When all 3 phases are collected, runs LSTM inference.

        Args:
            frame_result: FrameResult from FrameProcessor.

        Returns:
            dict or None: Inference result dict if a full rep
            was classified, None otherwise.

        Side Effects:
            May insert rep into database and emit WebSocket events.
        """
        if not self._is_active or not frame_result.is_valid:
            return None

        # Check for rep event
        rep_event = frame_result.rep_event
        if rep_event is None:
            return None

        # Engineer features for this phase
        context = {
            "rep_phase": rep_event.phase,
            "exercise_id": self._exercise_id,
            "load_kg": 0.0,
            "height_cm": 175.0,
            "weight_kg": 75.0,
            "ftr": 0.0,
        }
        features = self._feature_engineer.engineer(
            frame_result.landmarks,
            frame_result.angles,
            context=context,
        )

        # Push into buffer
        self._sequence_buffer.push(rep_event.phase, features)

        # Emit rep phase event
        if self._socketio:
            from api.socket_events import emit_rep_complete
            emit_rep_complete(self._socketio, rep_event.to_dict())

        # Check if all 3 phases collected
        if not self._sequence_buffer.is_ready():
            return None

        # Run inference
        sequence = self._sequence_buffer.get_sequence()
        result = self._lstm_runner.predict(sequence)
        self._sequence_buffer.reset()

        self._rep_count += 1

        # Save to DB
        rep_data = {
            "rep_number": self._rep_count,
            "phase": 3,
            "timestamp_ms": frame_result.timestamp_ms,
            "angles": frame_result.angles,
            "inference_label": result.label,
            "inference_confidence": result.confidence,
        }
        insert_rep(self._session_id, rep_data)

        # Emit inference result
        result_dict = result.to_dict()
        result_dict["rep_number"] = self._rep_count

        if self._socketio:
            from api.socket_events import (
                emit_inference_result, emit_session_update
            )
            emit_inference_result(self._socketio, result_dict)

            stats = get_session_stats(self._session_id)
            emit_session_update(self._socketio, stats)

        return result_dict

    @property
    def frame_processor(self):
        """Get the FrameProcessor instance.
        Returns:
            FrameProcessor: The internal frame processor.
        """
        return self._frame_processor

    @property
    def session_id(self):
        """Get current session ID.
        Returns:
            int or None: Active session ID.
        """
        return self._session_id

    @property
    def is_active(self):
        """Check if a session is in progress.
        Returns:
            bool: True if session is active.
        """
        return self._is_active

    @property
    def rep_count(self):
        """Get completed rep count.
        Returns:
            int: Reps counted this session.
        """
        return self._rep_count
