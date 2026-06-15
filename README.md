# Digital Spotter

Edge-based multimodal AI posture correction system for compound barbell lifts.  
Real-time BlazePose vision + IMU sensor fusion with TFLite LSTM inference on Raspberry Pi 4B.

## Architecture

```
Camera (IMX219) → mediamtx (WebRTC) → BlazePose (MediaPipe) → Angle Calc → Rep Segmentation
                                                                    ↓
                                                              Feature Engineering (38 features)
                                                                    ↓
                                                              LSTM Inference (TFLite)
                                                                    ↓
                                                         Flask API + WebSocket → Dashboard
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Vanilla HTML5 + CSS3 + ES2022 JS |
| Video | WebRTC WHEP via mediamtx |
| Realtime | Flask-SocketIO WebSocket |
| Charts | Chart.js 4.x |
| Backend | Flask 3.x + eventlet |
| Pose | MediaPipe BlazePose Heavy |
| Inference | TFLite LSTM (quantized) |
| IMU | ESP32-S3 + MPU6050 via MQTT |
| Database | SQLite3 |
| Stream | mediamtx v1.9.1 |

## Hardware

- **Edge Device**: Raspberry Pi 4B 8GB RAM
- **Camera**: IMX219 Camera Module v2
- **IMU**: ESP32-S3 + MPU6050 (future phase)
- **Storage**: 32GB SD Card
- **Network**: WiFi, static IP 192.168.1.8

## Quick Start

```bash
# On the Pi
cd ~/digital-spotter
bash scripts/start_all.sh

# Access points
# Dashboard:  http://192.168.1.8:8080
# API:        http://192.168.1.8:5000/api/health
# Stream:     http://192.168.1.8:8889/picam
```

## Project Structure

```
digital-spotter/
├── config/          # Configuration constants
├── api/             # Flask REST API + WebSocket
├── vision/          # MediaPipe BlazePose pipeline
├── inference/       # TFLite LSTM inference
├── imu/             # ESP32-S3 IMU subscriber
├── database/        # SQLite session storage
├── models/          # ML model artifacts
├── frontend/        # Production dashboard
│   ├── css/
│   ├── js/
│   └── assets/
├── tests/           # Isolated test suites
├── scripts/         # DevOps scripts
├── logs/            # Runtime logs
└── data/            # SQLite database files
```

## Running Tests

```bash
# Activate venv
source venv/bin/activate  # Linux/Pi
# or
.\venv\Scripts\Activate.ps1  # Windows

# Run all tests
python -m pytest tests/ -v

# Run specific suite
python -m pytest tests/test_02_pose/ -v
```

## Author

**Bimala Wijekoon** — [@BimalaWijekoon](https://github.com/BimalaWijekoon)

## License

MIT
