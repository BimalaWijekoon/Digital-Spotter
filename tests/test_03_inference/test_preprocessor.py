"""
tests/test_03_inference/test_preprocessor.py
Purpose: Test Preprocessor load and transform operations.
Author: bimalawijekoon
Version: 1.0.0
Last Modified: 2026-06-21
"""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from inference.preprocessor import Preprocessor


class TestPreprocessor:
    def test_preprocessor_loads_mock(self):
        p = Preprocessor()
        p.load()
        assert p.is_loaded()
        assert p.is_mock

    def test_mock_transform_identity(self):
        p = Preprocessor()
        p.load()
        features = np.ones(40, dtype=np.float32) * 5.0
        result = p.transform(features)
        assert np.allclose(result, features)

    def test_transform_2d(self):
        p = Preprocessor()
        p.load()
        features = np.ones((4, 40), dtype=np.float32)
        result = p.transform(features)
        assert result.shape == (4, 40)
