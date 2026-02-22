# GestureCam

Android camera streaming app with real-time body pose and hand tracking for gesture-based game control.

## Components

### Android App (`app/`)
Streams camera feed over TCP with full manual control of exposure, ISO, focus, white balance, zoom, and resolution.

### Python Client (`client/`)
Receives the camera stream and runs:
- **Body pose tracking** via MediaPipe (33 landmarks, world coordinates)
- **Hand tracking** via WiLoR-mini (21 keypoints per hand, 3D reconstruction)
- **Gesture classification** via a Transformer model (multi-label, 260-dim features + velocity)

## Usage

### Start the Android app
Install and launch on your phone. It starts a TCP server for video streaming and camera control.

### Connect the client
```bash
cd client
python stream_client.py <phone_ip>
```

### Record gesture training data
```bash
python gesture_recorder.py <phone_ip>
# R = record, BACKSPACE = discard last, Q = quit
```

### Train the gesture model
```bash
python train_gestures.py
```

## Feature Vector

260-dim raw features per frame:
- Pose world coordinates (33 landmarks x 3) = 99
- Left hand 3D (21 keypoints x 3) = 63
- Right hand 3D (21 keypoints x 3) = 63
- Hand presence flags = 2
- Pose visibility scores = 33

Extended to 485 dims with velocity features for the gesture Transformer.

## Controls (stream_client.py)

| Key | Action |
|-----|--------|
| SPACE | Toggle auto/manual exposure |
| A | Toggle auto/manual focus |
| E/D | Exposure compensation +/- |
| I/K | ISO +/- |
| Z/X | Zoom +/- |
| N | Toggle tracking |
| H | Help overlay |
| Q | Quit |

## Requirements

- Android phone with Camera2 API support
- Python 3.10+, `opencv-python`, `numpy`, `mediapipe`, `torch`
- WiLoR-mini models (auto-downloaded on first run)

## Related

- [MCGestureControl](https://github.com/Forkei/MCGestureControl) — Minecraft control policy trained on gesture data from this app
- [MCCTP](https://github.com/lucasoyen/mcctp) — Minecraft Control Transfer Protocol mod
