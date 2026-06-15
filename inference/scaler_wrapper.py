"""
inference/scaler_wrapper.py
Purpose: Load and apply the joblib StandardScaler from training.
         Falls back to identity transform if scaler.pkl not available.
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


class ScalerWrapper:
    """Wrapper for joblib StandardScaler with graceful fallback.

    Loads the scaler.pkl exported from the training notebook.
    If the file is missing, falls back to identity transform
    (no scaling) so the system can still run with mock inference.
    """

    def __init__(self):
        """Initialize ScalerWrapper with no scaler loaded."""
        self._scaler = None
        self._is_loaded = False
        self._is_mock = False

    def load(self, path=None):
        """Load the StandardScaler from a pickle file.

        Args:
            path: Path to scaler.pkl. Uses Config.PATHS.SCALER_PKL
                if None.

        Returns:
            bool: True if loaded successfully, False if falling back
                to identity transform.

        Side Effects:
            Loads scaler into memory. Logs warning if mock mode.
        """
        scaler_path = Path(path) if path else Config.PATHS.SCALER_PKL

        if not scaler_path.exists():
            logger.warning(
                "Scaler file not found: %s — using identity transform",
                scaler_path
            )
            self._is_mock = True
            self._is_loaded = True
            return False

        try:
            import joblib
            self._scaler = joblib.load(str(scaler_path))
            self._is_loaded = True
            self._is_mock = False
            logger.info("Scaler loaded from %s", scaler_path)
            return True

        except Exception as e:
            logger.error("Failed to load scaler: %s", e)
            self._is_mock = True
            self._is_loaded = True
            return False

    def transform(self, features):
        """Apply the same scaling as training to a feature vector.

        Args:
            features: numpy.ndarray of shape (N,) or (M, N) where
                N is the number of features.

        Returns:
            numpy.ndarray: Scaled features, same shape as input.
                Returns input unchanged if in mock mode.
        """
        if not self._is_loaded:
            self.load()

        features = np.array(features, dtype=np.float32)

        if self._is_mock or self._scaler is None:
            return features

        # Handle both 1D and 2D inputs
        if features.ndim == 1:
            return self._scaler.transform(
                features.reshape(1, -1)
            ).flatten().astype(np.float32)
        else:
            return self._scaler.transform(features).astype(np.float32)

    def is_loaded(self):
        """Check if scaler is loaded (including mock mode).

        Returns:
            bool: True if scaler is ready for use.
        """
        return self._is_loaded

    @property
    def is_mock(self):
        """Check if using identity transform fallback.

        Returns:
            bool: True if no real scaler was loaded.
        """
        return self._is_mock
