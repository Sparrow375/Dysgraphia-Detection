# Desktop Stylus & Tablet Kinematics Capture Tool

Real-time high-frequency (120Hz–200Hz) handwriting kinematics recorder for graphic drawing tablets (XP-Pen, Wacom, Huion) built with Python and PyQt6.

## Features
- **Real-Time Kinematic Extraction:**
  - Screen coordinates $(x, y)$ and physical displacement
  - Stylus pressure (4096 / 8192 levels from digitizer)
  - Tilt angles ($x$-tilt, $y$-tilt in degrees)
  - Instantaneous velocity ($v$), acceleration ($a$), and jerk ($j = \Delta a / \Delta t$)
  - Polling frequency / sampling rate monitor (Hz)
- **Stroke Dynamics:**
  - Automatic pen-down and pen-up stroke boundary detection
  - In-air hover hesitation vs on-paper execution tracking
- **Export Formats:**
  - Raw timeseries CSV export (`samples/kinematics_<timestamp>.csv`)
  - Rendered stroke canvas PNG image (`samples/kinematics_<timestamp>.png`)

## Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Connect Tablet & Run
Connect your drawing tablet (XP-Pen, Wacom, or Huion) and launch:
```bash
python stylus_data.py
```
- Write or draw on the tablet surface.
- Press **`S`** to save the recorded timeseries CSV and PNG preview into `samples/`.
- Press **`C`** to clear the canvas.

## Recorded Sample Data
Sample session recordings with real-time pressure, tilt, velocity, acceleration, and jerk are stored in [`samples/`](./samples/).
