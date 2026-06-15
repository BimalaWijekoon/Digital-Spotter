"""
inference/sequence_buffer.py
Purpose: Buffer to accumulate 3 phase feature vectors before
         running LSTM inference. Fixed-size sliding window.
Author: bimalawijekoon
Version: 1.0.0
Last Modified: 2026-06-15
"""

import logging
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import Config

logger = logging.getLogger(__name__)


class SequenceBuffer:
    """Fixed-size buffer for collecting phase feature vectors.

    Accumulates feature vectors for each of the 3 rep phases
    (Eccentric, Isometric, Concentric) before triggering LSTM inference.
    """

    def __init__(self, sequence_length=None):
        """Initialize SequenceBuffer.

        Args:
            sequence_length: Number of phases to collect (default 3
                from Config).

        Side Effects:
            Allocates internal buffer array.
        """
        self._seq_len = (
            sequence_length or Config.INFERENCE.SEQUENCE_LENGTH
        )
        self._num_features = Config.INFERENCE.NUM_FEATURES
        self._buffer = np.zeros(
            (self._seq_len, self._num_features),
            dtype=np.float32,
        )
        self._filled = [False] * self._seq_len
        self._count = 0

    def push(self, phase_id, features):
        """Add phase features to the buffer.

        Args:
            phase_id: Phase number (1, 2, or 3). Mapped to
                buffer index (0, 1, 2).
            features: numpy.ndarray of shape (num_features,).

        Side Effects:
            Stores features in the corresponding buffer slot.

        Raises:
            ValueError: If phase_id is not 1, 2, or 3.
        """
        if phase_id < 1 or phase_id > self._seq_len:
            raise ValueError(
                f"Invalid phase_id {phase_id}, "
                f"expected 1-{self._seq_len}"
            )

        idx = phase_id - 1
        features = np.array(features, dtype=np.float32)

        if features.shape[0] != self._num_features:
            logger.warning(
                "Feature size mismatch: expected %d, got %d",
                self._num_features, features.shape[0]
            )
            # Pad or truncate
            padded = np.zeros(self._num_features, dtype=np.float32)
            n = min(features.shape[0], self._num_features)
            padded[:n] = features[:n]
            features = padded

        self._buffer[idx] = features
        if not self._filled[idx]:
            self._filled[idx] = True
            self._count += 1

    def is_ready(self):
        """Check if all phases have been collected.

        Returns:
            bool: True when all phase slots are filled.
        """
        return self._count >= self._seq_len

    def get_sequence(self):
        """Get the complete sequence for LSTM inference.

        Returns:
            numpy.ndarray: Shape (sequence_length, num_features).
                Returns zeros if buffer is not ready.
        """
        return self._buffer.copy()

    def reset(self):
        """Clear buffer for next rep.

        Side Effects:
            Zeros out buffer and resets fill tracking.
        """
        self._buffer = np.zeros(
            (self._seq_len, self._num_features),
            dtype=np.float32,
        )
        self._filled = [False] * self._seq_len
        self._count = 0

    @property
    def count(self):
        """Number of phases currently filled.

        Returns:
            int: Count of filled phase slots (0 to sequence_length).
        """
        return self._count

    @property
    def sequence_length(self):
        """Expected number of phases.

        Returns:
            int: Total phase slots in the buffer.
        """
        return self._seq_len
