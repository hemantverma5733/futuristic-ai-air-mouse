# 🖐 AI Air Mouse v2.0

> **Ultra-smooth, futuristic hand-tracking cursor control**  
> Multi-display · Apple-trackpad feel · 60 FPS · Predictive Kalman motion

---

## ✨ Features

| Feature | Detail |
|---|---|
| **Hand tracking** | MediaPipe Hands – single hand, real-time |
| **Cursor motion** | Kalman filter + ring-buffer smoothing (trackpad feel) |
| **Dead-zone** | Configurable anti-shake radius (default 12 px) |
| **Multi-display** | MacBook + external monitor + iPad (Sidecar/Luna) |
| **Target FPS** | 60 FPS (OpenCV low-latency buffer) |
| **Gestures** | 6 distinct hand gestures (see table below) |

---

## 🖥 Gesture Reference

| Gesture | Hand Shape | Action |
|---|---|---|
| **Move** | ☝️ Index finger extended | Move cursor |
| **Left Click** | 🤏 Quick pinch (index + middle) | Single left click |
| **Drag & Drop** | ✊ Hold pinch (index + middle, 18+ frames) | Mouse down → drag → release |
| **Right Click** | 🤞 Pinch index + ring finger | Right click |
| **Scroll Up** | 🖐 Three fingers up (index+middle+ring, pinky folded) | Scroll up |
| **Scroll Down** | 🤙 Pinky only extended | Scroll down |

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **macOS note:** grant your Terminal / Python Accessibility + Camera permissions in  
> *System Settings → Privacy & Security → Accessibility / Camera*

> **Linux note:** `pyautogui` requires `python3-xlib` or `python3-tk`:
> ```bash
> sudo apt install python3-xlib python3-tk
> ```

### 2. Run

```bash
python air_mouse.py
```

### 3. Options

```
--cam       Camera index (default 0 = built-in webcam)
--width     Capture width in px (default 1280)
--height    Capture height in px (default 720)
--smooth    Smoothing window 3-15 — higher = smoother but more lag (default 7)
--deadzone  Anti-shake dead-zone radius in px (default 12)
--fps       Target FPS (default 60)
--debug     Show multi-display mini-map HUD
```

**Examples:**

```bash
# High precision (less smoothing)
python air_mouse.py --smooth 4 --deadzone 8

# Maximum smoothness (more lag)
python air_mouse.py --smooth 12 --deadzone 15

# External webcam, debug HUD
python air_mouse.py --cam 1 --debug

# Low-end machine
python air_mouse.py --width 640 --height 480 --fps 30
```

### 4. Hot-keys (while running)

| Key | Action |
|---|---|
| `Q` / `Esc` | Quit |
| `M` | Toggle mirror flip |
| `D` | Toggle debug HUD |

---

## 🏗 Architecture

```
Camera Frame
    │
    ▼
MediaPipe Hands  ──────►  21 Landmarks
                               │
                    ┌──────────┼──────────┐
                    │          │          │
               GestureEngine  IndexTip  (other fingers)
                    │          │
                    │     Kalman Filter   ← predictive motion
                    │          │
                    │     Ring Buffer     ← temporal smoothing
                    │          │
                    │     Dead-zone       ← anti-shake
                    │          │
                    └──► DisplayLayout ──► pyautogui.moveTo / click / scroll
```

### Key classes

| Class | File | Purpose |
|---|---|---|
| `DisplayLayout` | `air_mouse.py` | Multi-monitor virtual desktop mapper |
| `KalmanCursor` | `air_mouse.py` | 2-state Kalman filter (position + velocity) |
| `RingBuffer` | `air_mouse.py` | Temporal rolling average |
| `GestureEngine` | `air_mouse.py` | State machine for all 6 gestures |

---

## 🖥 Multi-Display Setup

The system uses **screeninfo** to auto-detect all connected displays and their virtual desktop positions.

Supported configs:
- MacBook only
- MacBook + external monitor (side by side or stacked)
- MacBook + monitor + iPad via Sidecar or [Luna Display](https://lunadisplay.com)
- Any number of monitors (Windows/Linux extended desktop)

Move your hand to the far left/right of the camera frame to reach screens on either side.

---

## ⚙️ Tuning Guide

### Too shaky?
Increase `--deadzone` (try 18–25) and `--smooth` (try 9–12).

### Too laggy?
Decrease `--smooth` (try 4–5) and `--deadzone` (try 6–8).

### Gestures misfiring?
Ensure good lighting — MediaPipe detection confidence drops in dim conditions.  
Keep hand 40–80 cm from camera for best results.

### Low FPS?
- Reduce `--width 640 --height 480`
- Ensure no other apps are using the camera
- On macOS: disable True Tone / camera effects in FaceTime settings

---

## 📋 Requirements

- Python 3.9+
- OpenCV 4.8+
- MediaPipe 0.10+
- PyAutoGUI 0.9.54+
- NumPy 1.24+
- screeninfo 0.8+ *(optional but recommended for multi-display)*

---

## 📄 License

MIT — free to use, modify, and distribute.
