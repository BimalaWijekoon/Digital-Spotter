# Models Directory

This directory contains ML model artifacts used by the inference pipeline.

## Expected Files

| File | Description | Source |
|------|------------|--------|
| `model_quantized.tflite` | INT8 quantized LSTM model | Training notebook export |
| `model_fullprec.tflite` | Full precision LSTM model | Training notebook export |
| `scaler.save` | StandardScaler from training | joblib.dump() from training |
| `winsor_bounds.save` | Preprocessing bounds | joblib.dump() from training |
| `feature_medians.save` | Preprocessing imputation values | joblib.dump() from training |

## Notes

- These files are **not tracked in git** (see `.gitignore`)
- Deploy them manually to this directory on the Pi
- The inference pipeline gracefully falls back to mock predictions if files are missing
- See `inference/preprocessor.py` and `inference/lstm_runner.py` for fallback behavior
