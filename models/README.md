# Models Directory

This directory contains ML model artifacts used by the inference pipeline.

## Expected Files

| File | Description | Source |
|------|------------|--------|
| `model_quantized.tflite` | INT8 quantized LSTM model | Training notebook export |
| `model_fullprec.tflite` | Full precision LSTM model | Training notebook export |
| `scaler.pkl` | StandardScaler from training | joblib.dump() from training |

## Notes

- These files are **not tracked in git** (see `.gitignore`)
- Deploy them manually to this directory on the Pi
- The inference pipeline gracefully falls back to mock predictions if files are missing
- See `inference/scaler_wrapper.py` and `inference/lstm_runner.py` for fallback behavior
