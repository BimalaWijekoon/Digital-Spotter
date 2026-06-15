"""
vision/frame_processor.py
Purpose: Orchestrates the full frame processing pipeline:
         capture -> pose -> angles -> segmentation.
         Runs in a background thread, providing thread-safe access
         to the latest results.
Author: bimalawijekoon
Version: 1.0.0
Last Modified: 2026-06-15
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import Config
from vision.pose_engine import PoseEngine, PoseResult
from vision.angle_calculator import AngleCalculator
from vision.rep_segmenter import RepSegmenter, RepEvent

logger = logging.getLogger(__name__)


@dataclass
class FrameResult:
    """Complete result from processing a single frame.

    Attributes:
        landmarks: Dict of landmark positions.
        angles: Dict of 7 joint angles.
        rep_event: RepEvent if a phase transition occurred.
        is_valid: Whether pose was successfully detected.
        fps: Current processing framerate.
        timestamp_ms: Millisecond timestamp of processing.
        processing_time_ms: Pipeline latency in milliseconds.
    """
    landmarks: dict = field(default_factory=dict)
    angles: dict = field(default_factory=dict)
    rep_event: Optional[RepEvent] = None
    is_valid: bool = False
    fps: float = 0.0
    timestamp_ms: int = 0
    processing_time_ms: float = 0.0

    def to_dict(self):
        """Convert to dictionary for WebSocket emission.

        Returns:
            dict: Serializable result dictionary.
        """
        result = {
            "landmarks": self.landmarks,
            "angles": self.angles,
            "is_valid": self.is_valid,
            "fps": round(self.fps, 1),
            "timestamp_ms": self.timestamp_ms,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }
        if self.rep_event:
            result["rep_event"] = self.rep_event.to_dict()
        return result


class FrameProcessor:
    """Orchestrates the vision processing pipeline.

    Manages PoseEngine, AngleCalculator, and RepSegmenter to
    process frames end-to-end. Can run in a background thread
    for continuous processing.
    """

    def __init__(self):
        """Initialize all pipeline components.

        Side Effects:
            Creates PoseEngine, AngleCalculator, RepSegmenter instances.
        """
        self._pose_engine = PoseEngine()
        self._angle_calc = AngleCalculator()
        self._rep_segmenter = RepSegmenter(fps=Config.CAMERA.FPS)

        # Thread-safe result storage
        self._latest_result = None
        self._lock = threading.Lock()

        # Background processing state
        self._running = False
        self._thread = None
        self._frame_times = []
        self._fps = 0.0

        logger.info("FrameProcessor initialized")

    def process(self, frame):
        """Run the full pipeline on a single frame.

        Pipeline: frame -> BlazePose -> angles -> rep segmentation

        Args:
            frame: numpy.ndarray BGR image (H, W, 3).

        Returns:
            FrameResult: Complete processing result.

        Side Effects:
            Updates internal FPS tracking and latest result.
        """
        start_time = time.perf_counter()
        timestamp_ms = int(time.time() * 1000)

        # Step 1: Pose estimation
        pose_result = self._pose_engine.process_frame(frame)

        if not pose_result.is_valid:
            result = FrameResult(
                is_valid=False,
                fps=self._fps,
                timestamp_ms=timestamp_ms,
                processing_time_ms=(
                    (time.perf_counter() - start_time) * 1000
                ),
            )
            self._update_latest(result)
            return result

        # Step 2: Angle calculation
        angles = self._angle_calc.compute_all_angles(pose_result.landmarks)

        # Step 3: Rep segmentation
        rep_event = self._rep_segmenter.update(angles)

        # Step 4: FPS calculation
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._update_fps(elapsed_ms)

        result = FrameResult(
            landmarks=pose_result.landmarks,
            angles=angles,
            rep_event=rep_event,
            is_valid=True,
            fps=self._fps,
            timestamp_ms=timestamp_ms,
            processing_time_ms=elapsed_ms,
        )
        self._update_latest(result)
        return result

    def start(self, frame_source=None):
        """Begin processing loop in background thread.

        Args:
            frame_source: Callable that returns (bool, frame) tuples,
                like cv2.VideoCapture.read(). If None, generates
                synthetic frames for testing.

        Side Effects:
            Starts a daemon thread for continuous processing.
        """
        if self._running:
            logger.warning("FrameProcessor already running")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._processing_loop,
            args=(frame_source,),
            daemon=True,
            name="FrameProcessor",
        )
        self._thread.start()
        logger.info("FrameProcessor background thread started")

    def stop(self):
        """Gracefully shutdown background processing.

        Side Effects:
            Stops the processing thread and releases pose engine.
        """
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._pose_engine.release()
        logger.info("FrameProcessor stopped")

    def get_latest(self):
        """Get the most recent processing result (thread-safe).

        Returns:
            FrameResult or None: Latest result, or None if no
            frames have been processed yet.
        """
        with self._lock:
            return self._latest_result

    def reset_session(self):
        """Reset rep segmenter for a new training session.

        Side Effects:
            Clears rep counter and angle buffers.
        """
        self._rep_segmenter.reset()
        logger.info("FrameProcessor session reset")

    @property
    def rep_count(self):
        """Current rep count from the segmenter.

        Returns:
            int: Number of completed reps.
        """
        return self._rep_segmenter.rep_count

    @property
    def current_phase(self):
        """Current rep phase from the segmenter.

        Returns:
            int: Phase number (0=idle, 1-3 during rep).
        """
        return self._rep_segmenter.current_phase

    def _processing_loop(self, frame_source):
        """Main processing loop for background thread.

        Args:
            frame_source: Callable returning (success, frame) tuples.

        Side Effects:
            Continuously processes frames until stop() is called.
        """
        logger.info("Processing loop started")
        target_interval = 1.0 / Config.CAMERA.FPS

        while self._running:
            loop_start = time.perf_counter()

            try:
                if frame_source is not None:
                    success, frame = frame_source()
                    if not success:
                        time.sleep(0.01)
                        continue
                else:
                    # TODO: PHASE_7 - Replace with direct libcamera
                    # capture instead of synthetic frame for production
                    frame = np.zeros(
                        (Config.CAMERA.HEIGHT, Config.CAMERA.WIDTH, 3),
                        dtype=np.uint8,
                    )

                self.process(frame)

            except Exception as e:
                logger.error("Frame processing error: %s", e)
                time.sleep(0.1)

            # Maintain target FPS
            elapsed = time.perf_counter() - loop_start
            sleep_time = target_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.info("Processing loop stopped")

    def _update_latest(self, result):
        """Thread-safe update of the latest result.

        Args:
            result: FrameResult to store.

        Side Effects:
            Updates self._latest_result under lock.
        """
        with self._lock:
            self._latest_result = result

    def _update_fps(self, elapsed_ms):
        """Update rolling FPS calculation.

        Args:
            elapsed_ms: Processing time for the current frame.

        Side Effects:
            Updates self._fps with rolling average over last 30 frames.
        """
        self._frame_times.append(elapsed_ms)
        if len(self._frame_times) > 30:
            self._frame_times = self._frame_times[-30:]

        if self._frame_times:
            avg_ms = sum(self._frame_times) / len(self._frame_times)
            self._fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
