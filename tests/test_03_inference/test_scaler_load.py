"""
tests/test_03_inference/test_scaler_load.py
Purpose: Test ScalerWrapper load and transform operations.
Author: bimalawijekoon
Version: 1.0.0
Last Modified: 2026-06-15
"""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from inference.scaler_wrapper import ScalerWrapper


class TestScalerWrapper:
    def test_scaler_loads_mock(self):
        s = ScalerWrapper()
        s.load()
        assert s.is_loaded()
        assert s.is_mock

    def test_mock_transform_identity(self):
        s = ScalerWrapper()
        s.load()
        features = np.ones(38, dtype=np.float32) * 5.0
        result = s.transform(features)
        assert np.allclose(result, features)

    def test_transform_2d(self):
        s = ScalerWrapper()
        s.load()
        features = np.ones((3, 38), dtype=np.float32)
        result = s.transform(features)
        assert result.shape == (3, 38)
