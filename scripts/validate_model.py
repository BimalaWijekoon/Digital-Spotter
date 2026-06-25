#!/usr/bin/env python3
"""
scripts/validate_model.py
Purpose: Validate TFLite model file — checks input/output shapes,
         runs a sample inference, and reports compatibility.
Author: bimalawijekoon
Version: 1.1.0
"""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import Config


def load_interpreter(model_path):
    """Load a TFLite interpreter with Flex delegate support.

    Tries tf.lite.Interpreter first (full tensorflow package — ships with
    the Flex delegate built in, required for SELECT_TF_OPS models like
    the BiLSTM+MHA architecture used here). Falls back to ai_edge_litert
    only if tensorflow is not installed, but warns that SELECT_TF_OPS
    models will fail in that case.

    Args:
        model_path: Path-like, location of the .tflite file.

    Returns:
        Allocated interpreter instance.

    Raises:
        RuntimeError: If no runtime is available.
    """
    model_path_str = str(model_path)

    # First choice: full tensorflow — Flex delegate included, always works
    # for SELECT_TF_OPS models (Bidirectional LSTM forces this requirement).
    try:
        import tensorflow as tf
        interp = tf.lite.Interpreter(model_path=model_path_str)
        interp.allocate_tensors()
        print("  Runtime: tf.lite.Interpreter (tensorflow package, Flex delegate built-in)")
        return interp
    except ImportError:
        pass
    except Exception as e:
        print(f"  tf.lite.Interpreter failed: {e}")

    # Second choice: ai_edge_litert — works for pure-builtin models only.
    # Will fail at allocate_tensors() for SELECT_TF_OPS models.
    try:
        import ai_edge_litert.interpreter as litert
        interp = litert.Interpreter(model_path=model_path_str)
        interp.allocate_tensors()
        print("  Runtime: ai_edge_litert (no Flex delegate — builtins only)")
        return interp
    except ImportError:
        pass
    except Exception as e:
        print(f"  ai_edge_litert failed: {e}")
        print("  ⚠️  Model requires SELECT_TF_OPS (Flex delegate).")
        print("     ai_edge_litert does not include Flex delegate support.")
        print("     Install full tensorflow: pip install tensorflow==2.16.2")
        raise

    raise RuntimeError(
        "No TFLite runtime available. "
        "Install tensorflow: pip install tensorflow==2.16.2"
    )


def validate():
    model_path = Config.PATHS.MODEL_TFLITE
    scaler_path = Config.PATHS.SCALER_PKL

    print("=" * 50)
    print("Digital Spotter — Model Validator")
    print("=" * 50)

    print(f"\nModel path: {model_path}")
    print(f"  Exists: {model_path.exists()}")

    print(f"\nScaler path: {scaler_path}")
    print(f"  Exists: {scaler_path.exists()}")

    if not model_path.exists():
        print("\n[!] Model file not found — using mock inference.")
        print("    Place model_quantized.tflite in models/ directory.")
        return

    interpreter = load_interpreter(model_path)

    inp = interpreter.get_input_details()
    out = interpreter.get_output_details()

    print(f"\nInput shape:  {inp[0]['shape']}")
    print(f"Input dtype:  {inp[0]['dtype']}")
    print(f"Output shape: {out[0]['shape']}")
    print(f"Output dtype: {out[0]['dtype']}")

    # Run sample inference
    dummy = np.random.randn(*inp[0]['shape']).astype(np.float32)
    interpreter.set_tensor(inp[0]['index'], dummy)

    start = time.perf_counter()
    interpreter.invoke()
    elapsed = (time.perf_counter() - start) * 1000

    result = interpreter.get_tensor(out[0]['index'])
    print(f"\nSample output: {result}")
    print(f"Inference time: {elapsed:.2f} ms")
    print("\n✓ Model validation passed!")


if __name__ == "__main__":
    validate()