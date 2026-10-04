# Samsung Galaxy S26 Ultra S-Pen Dysgraphia Note-Taking & Kinematic Collector

Native Android application built specifically for Samsung Galaxy devices with S-Pen (optimized for Samsung Galaxy S26 Ultra / S24 Ultra / Galaxy Tab S9) to capture unbuffered high-frequency 240Hz+ stylus kinematic and biometric telemetry during pediatric handwriting.

## Project Overview
Part of Workstream A in the Multilingual Dysgraphia Detection Initiative. This app provides a clean, distraction-free digital note-taking canvas for children and students, capturing 24 distinct physical, temporal, spatial, and kinematic telemetry dimensions in the background while providing an instant 1-click export package.

## Telemetry Kinematics Matrix (24 Dimensions)
- **Spatial & Physical Coordinates:**
  - Screen $(x, y)$ pixels
  - Physical coordinates $(x_{\text{mm}}, y_{\text{mm}})$ calibrated via device screen DPI
- **Temporal & Timing:**
  - Relative millisecond/nanosecond timestamps ($t_{\text{rel}}$)
  - UTC wall clock timestamps ($t_{\text{epoch}}$)
  - Inter-sample intervals ($\Delta t$)
- **Pressure & Contact Force:**
  - Continuous pressure levels ($p \in [0.0, 1.0]$) across 4096 levels of Wacom EMR digitizer
  - Dynamic pressure rate ($dp/dt$)
- **Stylus Angles & Spatial Orientation:**
  - Tilt angle $\theta$ (`AXIS_TILT` in radians)
  - Azimuth/Orientation $\phi$ (`AXIS_ORIENTATION` in radians)
  - Angular rate ($d\phi/dt$)
- **In-Air Flight Dynamics:**
  - S-Pen hover tracking via `ACTION_HOVER_MOVE` and `AXIS_DISTANCE`
  - In-air hesitation distance and flight-to-touch transition intervals
- **Neuromotor Kinematics:**
  - Instantaneous stroke velocity ($v$)
  - Instantaneous stroke acceleration ($a$)
  - Neuromotor jerk ($j = \Delta a / \Delta t$) for tremor and motor dysgraphia profiling
- **Hardware Button & Tool Isolation:**
  - Strict palm rejection (`TOOL_TYPE_STYLUS` vs `TOOL_TYPE_FINGER`)
  - S-Pen hardware button tracking (`BUTTON_STYLUS_PRIMARY`)

## Android Architecture & Tech Stack
- **Language & Runtime:** Kotlin, Android SDK (API 34+ target)
- **UI Framework:** Jetpack Compose + Hardware-accelerated custom Canvas View
- **Hardware Integration:** Custom `SPenDrawingView` leveraging `requestUnbufferedDispatch()` and historical event batch unpacking (`getHistoricalX`, `getHistoricalY`, `getHistoricalPressure`, etc.) to guarantee zero dropped touch events even during fast cursive strokes.
- **Export Engine:** `SessionExporter` packages:
  - `timeseries_raw.csv`: High-frequency tabular timeseries
  - `timeseries_raw.jsonl`: Event stream records
  - `strokes_summary.json`: Segmented stroke boundaries & aggregate kinematics
  - `note_render.png`: Clean rendered canvas snapshot
  - `session_metadata.json`: Device info, screen calibration, and subject metadata
  - Compresses into a single `.zip` and invokes the native Android Sharesheet (Quick Share, Google Drive, Gmail, etc.)

## Building and Running
1. Open the project root in **Android Studio Hedgehog / Iguana / Jellyfish** or newer.
2. Select the `spen_note_collector` project or `app` module.
3. Sync Gradle dependencies.
4. Run on a connected physical Samsung Galaxy device (or Android emulator with stylus emulation).
5. Or install the pre-compiled debug APK directly:
   ```bash
   adb install spen_note_collector/app/build/outputs/apk/debug/app-debug.apk
   ```
