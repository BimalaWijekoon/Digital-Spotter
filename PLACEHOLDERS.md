# Placeholders

Tracking all placeholder implementations that need to be replaced with production code.

## Active Placeholders

| Phase | File | Placeholder | Description |
|-------|------|------------|-------------|
| PHASE_3 | `vision/pose_engine.py` | BlazePose weights | Add BlazePose Heavy float16 weights path when model file is deployed |
| PHASE_4 | `inference/preprocessor.py` | preprocessor.save artifacts | Replace placeholder artifacts with actual scaler.save, winsor_bounds.save, feature_medians.save from training notebook |
| PHASE_4 | `inference/lstm_runner.py` | model_quantized.tflite | Replace placeholder model with actual TFLite model from training |
| PHASE_6 | `inference/feature_engineer.py` | IMU features | 6 IMU features are zeros until ESP32-S3 connected |
| PHASE_6 | `inference/feature_engineer.py` | Bar performance | 3 bar features are zeros until IMU connected |
| PHASE_7 | `vision/frame_processor.py` | Frame source | Replace with direct libcamera capture for production |
| PHASE_8 | `imu/mqtt_subscriber.py` | MQTT broker | Set config.MQTT.ENABLED = True when ESP32-S3 hardware ready |
| PHASE_8 | `imu/fusion_engine.py` | Fusion engine | Returns vision-only features until IMU connected |

## Resolved Placeholders

_None yet_
