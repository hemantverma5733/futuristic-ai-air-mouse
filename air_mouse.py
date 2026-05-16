"""
╔══════════════════════════════════════════════════════════════════╗
║           🖐  FUTURISTIC AI AIR MOUSE  — v2.0                   ║
║   Ultra-smooth hand-tracking cursor for multi-display setups    ║
╚══════════════════════════════════════════════════════════════════╝

GESTURES
────────
  Index finger           → Move cursor
  Index + Middle pinch   → Left click  (quick) / Drag (hold)
  Index + Ring pinch     → Right click
  Three fingers up       → Scroll up
  Pinky gesture          → Scroll down

REQUIREMENTS
────────────
  pip install opencv-python mediapipe pyautogui numpy screeninfo

USAGE
─────
  python air_mouse.py [--cam 0] [--width 1280] [--height 720]
                      [--smooth 7] [--deadzone 12] [--fps 60]
                      [--debug]
"""

import argparse
import collections
import math
import subprocess
import sys
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pyautogui

# ── Optional: screeninfo for multi-display layout ─────────────────
try:
    from screeninfo import get_monitors
    SCREENINFO_AVAILABLE = True
except ImportError:
    SCREENINFO_AVAILABLE = False
    print("[WARN] screeninfo not found — using primary screen only.")

# ─────────────────────────── CONFIG ──────────────────────────────

pyautogui.FAILSAFE = False   # disable move-to-corner safety stop
pyautogui.PAUSE = 0          # remove built-in delay

# ──────────────────────── ARGUMENT PARSER ────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="AI Air Mouse — futuristic hand-tracking cursor")
    p.add_argument("--cam",      type=int,   default=0,    help="Camera index (default 0)")
    p.add_argument("--width",    type=int,   default=1280, help="Camera capture width")
    p.add_argument("--height",   type=int,   default=720,  help="Camera capture height")
    p.add_argument("--smooth",   type=int,   default=7,    help="Smoothing window 3-15 (higher=smoother)")
    p.add_argument("--deadzone", type=int,   default=12,   help="Dead-zone radius in px (anti-shake)")
    p.add_argument("--fps",      type=int,   default=60,   help="Target FPS")
    p.add_argument("--debug",    action="store_true",      help="Show HUD overlay")
    return p.parse_args()

# ────────────────────── MULTI-DISPLAY HELPER ─────────────────────

class DisplayLayout:
    """Maps [0,1]×[0,1] normalised hand coordinates → global screen space."""

    def __init__(self):
        self.monitors = []
        self.total_width  = 1920
        self.total_height = 1080
        self.origin_x     = 0
        self.origin_y     = 0
        self._build()

    def _build(self):
        if SCREENINFO_AVAILABLE:
            try:
                mons = get_monitors()
                if mons:
                    self.monitors = mons
                    xs = [m.x for m in mons]
                    ys = [m.y for m in mons]
                    xe = [m.x + m.width  for m in mons]
                    ye = [m.y + m.height for m in mons]
                    self.origin_x     = min(xs)
                    self.origin_y     = min(ys)
                    self.total_width  = max(xe) - self.origin_x
                    self.total_height = max(ye) - self.origin_y
                    print(f"[INFO] Detected {len(mons)} display(s) — "
                          f"virtual desktop: {self.total_width}×{self.total_height} "
                          f"origin ({self.origin_x},{self.origin_y})")
                    for m in mons:
                        print(f"       {m.name}: {m.width}×{m.height} @ ({m.x},{m.y})")
                    return
            except Exception as e:
                print(f"[WARN] screeninfo error: {e}")

        # Fallback: ask pyautogui
        w, h = pyautogui.size()
        self.total_width  = w
        self.total_height = h
        print(f"[INFO] Single screen: {w}×{h}")

    def norm_to_global(self, nx: float, ny: float):
        """Convert [0,1] normalised → global pixel coords."""
        gx = int(self.origin_x + nx * self.total_width)
        gy = int(self.origin_y + ny * self.total_height)
        return gx, gy


# ──────────────────────── SMOOTHER / PREDICTOR ───────────────────

class KalmanCursor:
    """
    Lightweight 1-D Kalman filter applied independently on X and Y.
    Gives Apple-trackpad-like feel: tight tracking + predictive smoothing.
    """
    def __init__(self, q=0.05, r=5.0):
        # State: [position, velocity]
        self._x = np.zeros((2, 1), dtype=float)   # x-axis state
        self._y = np.zeros((2, 1), dtype=float)   # y-axis state
        self.P_x = np.eye(2) * 500.0
        self.P_y = np.eye(2) * 500.0
        # Transition: constant-velocity model (dt=1 frame)
        self.F = np.array([[1, 1], [0, 1]], dtype=float)
        self.H = np.array([[1, 0]], dtype=float)
        self.Q = np.array([[q*q, 0], [0, q*q*0.1]], dtype=float)
        self.R = np.array([[r*r]], dtype=float)
        self.initialized = False

    def reset(self, px, py):
        self._x = np.array([[px], [0.0]])
        self._y = np.array([[py], [0.0]])
        self.P_x = np.eye(2) * 500.0
        self.P_y = np.eye(2) * 500.0
        self.initialized = True

    def _update_1d(self, state, P, measurement):
        # Predict
        state = self.F @ state
        P     = self.F @ P @ self.F.T + self.Q
        # Update
        S = self.H @ P @ self.H.T + self.R
        K = P @ self.H.T @ np.linalg.inv(S)
        z = np.array([[measurement]])
        state = state + K @ (z - self.H @ state)
        P     = (np.eye(2) - K @ self.H) @ P
        return state, P

    def update(self, px, py):
        if not self.initialized:
            self.reset(px, py)
            return px, py
        self._x, self.P_x = self._update_1d(self._x, self.P_x, px)
        self._y, self.P_y = self._update_1d(self._y, self.P_y, py)
        return float(self._x[0, 0]), float(self._y[0, 0])


class RingBuffer:
    """Fixed-size FIFO for averaging recent positions."""
    def __init__(self, size=7):
        self.buf = collections.deque(maxlen=size)

    def push(self, val):
        self.buf.append(val)

    def mean(self):
        if not self.buf:
            return 0.0
        return sum(self.buf) / len(self.buf)


# ─────────────────────── GESTURE ENGINE ──────────────────────────

class GestureState:
    IDLE         = "IDLE"
    HOVERING     = "HOVERING"
    LEFT_CLICK   = "LEFT_CLICK"
    DOUBLE_CLICK = "DOUBLE_CLICK"
    DRAGGING     = "DRAGGING"
    RIGHT_CLICK  = "RIGHT_CLICK"
    SCROLL_UP    = "SCROLL_UP"
    SCROLL_DOWN  = "SCROLL_DOWN"
    VOLUME_CTRL  = "VOLUME_CTRL"
    SWIPING      = "SWIPING"
    MISSION_CTRL = "MISSION_CTRL"
    ZOOM         = "ZOOM"


def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


class GestureEngine:
    # Pinch thresholds (normalised landmark distance)
    PINCH_CLOSE        = 0.045
    PINCH_OPEN         = 0.070
    DRAG_HOLD_FRAMES   = 18    # hold pinch N frames → drag
    SCROLL_SPEED       = 3     # scroll units per frame while gesture active
    SCROLL_COOLDOWN    = 6     # frames between scroll ticks

    def __init__(self):
        self.state        = GestureState.IDLE
        self._pinch_hold  = 0
        self._scroll_cd   = 0
        self._drag_active = False
        self._last_click_t = 0.0
        self.click_cooldown = 0.35  # seconds

        # New Feature States
        self._last_vol = -1
        self._last_zoom = -1
        self._history_x = collections.deque(maxlen=15)
        self._history_y = collections.deque(maxlen=15)
        self._temp_state = None
        self._temp_state_t = 0.0
        
        # Hand tracking scroll
        self._last_scroll_y = None
        self._scroll_accumulator = 0.0

    # ── landmark helpers ─────────────────────────────────────────

    @staticmethod
    def _tip(lm, idx): return lm.landmark[idx]

    def _pinch_12(self, lm):
        """Index + Middle finger tips distance."""
        return dist(self._tip(lm, 8), self._tip(lm, 12))

    def _pinch_14(self, lm):
        """Index + Ring finger tips distance."""
        return dist(self._tip(lm, 8), self._tip(lm, 16))

    def _pinch_48(self, lm):
        """Thumb + Index finger tips distance."""
        return dist(self._tip(lm, 4), self._tip(lm, 8))

    def _pinch_416(self, lm):
        """Thumb + Ring finger tips distance."""
        return dist(self._tip(lm, 4), self._tip(lm, 16))

    def _other_fingers_closed(self, lm):
        tips = [12, 16, 20]
        bases = [10, 14, 18]
        return all(lm.landmark[t].y > lm.landmark[b].y for t, b in zip(tips, bases))

    def _open_hand(self, lm):
        tips_4 = [8, 12, 16, 20]
        bases_4 = [6, 10, 14, 18]
        up = all(lm.landmark[t].y < lm.landmark[b].y for t, b in zip(tips_4, bases_4))
        thumb_ext = dist(lm.landmark[4], lm.landmark[17]) > 0.1
        return up and thumb_ext

    def _three_fingers_up(self, lm):
        """Index, Middle, Ring all extended, Pinky folded."""
        tips   = [8, 12, 16]
        bases  = [6, 10, 14]
        up     = all(lm.landmark[t].y < lm.landmark[b].y for t, b in zip(tips, bases))
        pinky_folded = lm.landmark[20].y > lm.landmark[18].y
        return up and pinky_folded

    def _pinky_gesture(self, lm):
        """Only pinky extended, rest folded."""
        pinky_up = lm.landmark[20].y < lm.landmark[18].y
        others_folded = all(
            lm.landmark[t].y > lm.landmark[b].y
            for t, b in [(8,6),(12,10),(16,14)]
        )
        return pinky_up and others_folded

    # ── main tick ────────────────────────────────────────────────

    def tick(self, lm):
        """
        Returns (gesture_label, cursor_norm_xy | None)
        gesture_label: str from GestureState
        cursor_xy: (nx, ny) normalised [0-1] from INDEX TIP, or None to freeze
        """
        p12 = self._pinch_12(lm)
        p14 = self._pinch_14(lm)
        p48 = self._pinch_48(lm)

        index_tip  = self._tip(lm, 8)
        cursor_pos = (index_tip.x, index_tip.y)

        now = time.time()

        # Override for temp states (like DOUBLE CLICK or SWIPE)
        if self._temp_state and now - self._temp_state_t < 0.8:
            return self._temp_state, cursor_pos
        else:
            self._temp_state = None

        self._history_x.append(index_tip.x)
        self._history_y.append(index_tip.y)

        # ── Desktop Swiping & Mission Control (Open Hand) ──────────────
        if self._open_hand(lm) and len(self._history_x) == 15:
            dx = self._history_x[-1] - self._history_x[0]
            dy = self._history_y[-1] - self._history_y[0]
            
            if dx > 0.20:  # Swipe Right
                pyautogui.hotkey('ctrl', 'right')
                self._history_x.clear(); self._history_y.clear()
                self._temp_state = GestureState.SWIPING
                self._temp_state_t = now
                return GestureState.SWIPING, cursor_pos
            elif dx < -0.20: # Swipe Left
                pyautogui.hotkey('ctrl', 'left')
                self._history_x.clear(); self._history_y.clear()
                self._temp_state = GestureState.SWIPING
                self._temp_state_t = now
                return GestureState.SWIPING, cursor_pos
            elif dy < -0.20: # Swipe Up -> Mission Control
                pyautogui.hotkey('ctrl', 'up')
                self._history_x.clear(); self._history_y.clear()
                self._temp_state = GestureState.MISSION_CTRL
                self._temp_state_t = now
                return GestureState.MISSION_CTRL, cursor_pos
            elif dy > 0.20: # Swipe Down -> App Exposé
                pyautogui.hotkey('ctrl', 'down')
                self._history_x.clear(); self._history_y.clear()
                self._temp_state = GestureState.MISSION_CTRL
                self._temp_state_t = now
                return GestureState.MISSION_CTRL, cursor_pos

        # ── Zoom Control (Thumb + Ring Pinch) ───────────────────────
        p416 = self._pinch_416(lm)
        if p416 < self.PINCH_CLOSE and self._tip(lm, 12).y > self._tip(lm, 10).y:
            self._reset_drag()
            # Map Y pos to Zoom: y=0.2 (100%), y=0.8 (0%)
            zoom_level = max(0, min(100, int((0.8 - index_tip.y) / 0.6 * 100)))
            if self._last_zoom != -1:
                if zoom_level - self._last_zoom >= 3:
                    pyautogui.hotkey('cmd', '+')
                    self._last_zoom = zoom_level
                elif self._last_zoom - zoom_level >= 3:
                    pyautogui.hotkey('cmd', '-')
                    self._last_zoom = zoom_level
            else:
                self._last_zoom = zoom_level
            return GestureState.ZOOM, cursor_pos
        else:
            self._last_zoom = -1

        # ── Volume Control (Thumb + Index Pinch) ───────────────────
        if p48 < self.PINCH_CLOSE and self._other_fingers_closed(lm):
            self._reset_drag()
            # Map Y pos to volume: y=0.2 (100%), y=0.8 (0%)
            vol = max(0, min(100, int((0.8 - index_tip.y) / 0.6 * 100)))
            if abs(vol - self._last_vol) >= 2:
                subprocess.Popen(["osascript", "-e", f"set volume output volume {vol}"])
                self._last_vol = vol
            return GestureState.VOLUME_CTRL, cursor_pos

        # ── Scrolling (Based on Hand Movement) ────────────────────────
        is_scroll_up = self._three_fingers_up(lm)
        is_scroll_down = self._pinky_gesture(lm)
        
        if is_scroll_up or is_scroll_down:
            self._reset_drag()
            if self._last_scroll_y is not None:
                delta_y = index_tip.y - self._last_scroll_y
                # macOS: pyautogui.scroll(-10) scrolls down (sees lower content). 
                # Hand down -> y increases -> delta_y positive -> scroll down.
                self._scroll_accumulator += delta_y * -300
                
                scroll_amount = int(self._scroll_accumulator)
                if abs(scroll_amount) >= 1:
                    pyautogui.scroll(scroll_amount)
                    self._scroll_accumulator -= scroll_amount
            
            self._last_scroll_y = index_tip.y
            return GestureState.SCROLL_UP if is_scroll_up else GestureState.SCROLL_DOWN, cursor_pos
        else:
            self._last_scroll_y = None
            self._scroll_accumulator = 0.0

        # ── Right click (index + ring) ────────────────────────────
        if p14 < self.PINCH_CLOSE and p12 > self.PINCH_OPEN:
            self._reset_drag()
            if now - self._last_click_t > self.click_cooldown:
                pyautogui.rightClick()
                self._last_click_t = now
            return GestureState.RIGHT_CLICK, cursor_pos

        # ── Left click / Drag / Double Click (index + middle) ──────
        if p12 < self.PINCH_CLOSE:
            self._pinch_hold += 1
            if self._pinch_hold >= self.DRAG_HOLD_FRAMES:
                # DRAG mode
                if not self._drag_active:
                    pyautogui.mouseDown()
                    self._drag_active = True
                return GestureState.DRAGGING, cursor_pos
            else:
                return GestureState.LEFT_CLICK, cursor_pos
        else:
            # Pinch released
            if self._drag_active:
                pyautogui.mouseUp()
                self._drag_active = False
            if 0 < self._pinch_hold < self.DRAG_HOLD_FRAMES:
                # Quick pinch = click
                if now - self._last_click_t < 0.4:
                    pyautogui.doubleClick()
                    self._last_click_t = 0.0
                    self._temp_state = GestureState.DOUBLE_CLICK
                    self._temp_state_t = now
                    self._pinch_hold = 0
                    return GestureState.DOUBLE_CLICK, cursor_pos
                elif now - self._last_click_t > self.click_cooldown:
                    pyautogui.click()
                    self._last_click_t = now
            self._pinch_hold = 0
            return GestureState.HOVERING, cursor_pos

    def _reset_drag(self):
        if self._drag_active:
            pyautogui.mouseUp()
            self._drag_active = False
        self._pinch_hold = 0


# ─────────────────────────── HUD OVERLAY ─────────────────────────

GESTURE_COLORS = {
    GestureState.HOVERING:    (0, 230, 120),
    GestureState.LEFT_CLICK:  (0, 180, 255),
    GestureState.DOUBLE_CLICK:(0, 255, 255),
    GestureState.DRAGGING:    (0, 100, 255),
    GestureState.RIGHT_CLICK: (200, 80, 255),
    GestureState.SCROLL_UP:   (255, 220, 0),
    GestureState.SCROLL_DOWN: (255, 140, 0),
    GestureState.VOLUME_CTRL: (255, 50, 100),
    GestureState.SWIPING:     (50, 255, 50),
    GestureState.MISSION_CTRL:(200, 255, 100),
    GestureState.ZOOM:        (0, 255, 180),
    GestureState.IDLE:        (120, 120, 120),
}

GESTURE_ICONS = {
    GestureState.HOVERING:    "☝  MOVE",
    GestureState.LEFT_CLICK:  "🤏 L-CLICK",
    GestureState.DOUBLE_CLICK:"✌️ D-CLICK",
    GestureState.DRAGGING:    "✊  DRAG",
    GestureState.RIGHT_CLICK: "🤞 R-CLICK",
    GestureState.SCROLL_UP:   "🖐  SCROLL ▲",
    GestureState.SCROLL_DOWN: "🤙 SCROLL ▼",
    GestureState.VOLUME_CTRL: "🔊 VOLUME",
    GestureState.SWIPING:     "💨 SWIPE",
    GestureState.MISSION_CTRL:"🌐 MISSION",
    GestureState.ZOOM:        "🔍 ZOOM",
    GestureState.IDLE:        "—  IDLE",
}

def draw_hud(frame, gesture, fps, cursor_xy, display: DisplayLayout, debug=True):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    color = GESTURE_COLORS.get(gesture, (200,200,200))
    label = GESTURE_ICONS.get(gesture, gesture)

    # ── Glassmorphism panel ───────────────────────────────────────
    cv2.rectangle(overlay, (8, 8), (340, 170), (20, 20, 30), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # ── Title ─────────────────────────────────────────────────────
    cv2.putText(frame, "AI AIR MOUSE v2.0",
                (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (200, 200, 220), 1, cv2.LINE_AA)

    # ── FPS bar ───────────────────────────────────────────────────
    bar_w = int(fps / 60 * 200)
    cv2.rectangle(frame, (18, 44), (218, 54), (40,40,60), -1)
    cv2.rectangle(frame, (18, 44), (18+bar_w, 54),
                  (0,220,120) if fps > 45 else (255,160,0), -1)
    cv2.putText(frame, f"FPS  {fps:5.1f}",
                (224, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                (180,180,200), 1, cv2.LINE_AA)

    # ── Gesture pill ─────────────────────────────────────────────
    cv2.rectangle(frame, (18, 64), (320, 90), color, -1, cv2.LINE_AA)
    cv2.putText(frame, label,
                (24, 83), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                (10,10,10), 2, cv2.LINE_AA)

    # ── Screen layout mini-map ────────────────────────────────────
    if debug and SCREENINFO_AVAILABLE and display.monitors:
        MAP_X, MAP_Y, MAP_W, MAP_H = 18, 100, 300, 58
        cv2.rectangle(frame, (MAP_X, MAP_Y), (MAP_X+MAP_W, MAP_Y+MAP_H),
                      (40,40,60), -1)
        tw = display.total_width; th = display.total_height
        for m in display.monitors:
            mx = MAP_X + int((m.x - display.origin_x) / tw * MAP_W)
            my = MAP_Y + int((m.y - display.origin_y) / th * MAP_H)
            mw = int(m.width  / tw * MAP_W)
            mh = int(m.height / th * MAP_H)
            cv2.rectangle(frame, (mx, my), (mx+mw, my+mh), (80,120,160), 1)
            cv2.putText(frame, m.name[:6],
                        (mx+3, my+12), cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                        (150,180,210), 1, cv2.LINE_AA)

        # cursor dot on mini-map
        if cursor_xy:
            cx = MAP_X + int(cursor_xy[0] * MAP_W)
            cy = MAP_Y + int(cursor_xy[1] * MAP_H)
            cx = max(MAP_X, min(MAP_X+MAP_W, cx))
            cy = max(MAP_Y, min(MAP_Y+MAP_H, cy))
            cv2.circle(frame, (cx, cy), 4, color, -1)

    # ── Cursor crosshair on camera feed ──────────────────────────
    if cursor_xy:
        px = int(cursor_xy[0] * w)
        py = int(cursor_xy[1] * h)
        cv2.drawMarker(frame, (px, py), color,
                       cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)
        cv2.circle(frame, (px, py), 10, color, 1, cv2.LINE_AA)

    return frame

def draw_skeleton(frame, landmarks, w, h):
    HAND_CONNECTIONS = [
        (0,1), (1,2), (2,3), (3,4),
        (0,5), (5,6), (6,7), (7,8),
        (5,9), (9,10), (10,11), (11,12),
        (9,13), (13,14), (14,15), (15,16),
        (13,17), (0,17), (17,18), (18,19), (19,20)
    ]
    for p1, p2 in HAND_CONNECTIONS:
        x1, y1 = int(landmarks[p1].x * w), int(landmarks[p1].y * h)
        x2, y2 = int(landmarks[p2].x * w), int(landmarks[p2].y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (200, 255, 100), 2, cv2.LINE_AA)
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 4, (120, 255, 0), cv2.FILLED, cv2.LINE_AA)

# ─────────────────────────── MAIN LOOP ───────────────────────────

def main():
    args = parse_args()

    display  = DisplayLayout()
    gesture  = GestureEngine()
    kalman   = KalmanCursor(q=0.08, r=3.0)
    buf_x    = RingBuffer(args.smooth)
    buf_y    = RingBuffer(args.smooth)

    # ── NEW MEDIAPIPE TASKS API ──
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.70,
        min_hand_presence_confidence=0.70,
        min_tracking_confidence=0.70
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS,          args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)          # low-latency buffer

    # Actual camera dimensions after set
    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera: {cam_w}×{cam_h} @ {args.fps} FPS target")
    print(f"[INFO] Smoothing: {args.smooth}  Dead-zone: {args.deadzone}px")
    print("[INFO] Press Q to quit.")

    # FPS counter
    fps_buf = collections.deque(maxlen=30)
    prev_t  = time.perf_counter()

    # Dead-zone: last committed cursor position (global coords)
    last_gx, last_gy = pyautogui.position()

    # Mirror flip flag (webcam is mirrored for natural feel)
    MIRROR = True

    cur_gesture = GestureState.IDLE
    cursor_norm = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Camera read failed — exiting.")
            break

        if MIRROR:
            frame = cv2.flip(frame, 1)

        # ── MediaPipe inference ───────────────────────────────────
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)

        if result.hand_landmarks and len(result.hand_landmarks) > 0:
            landmarks = result.hand_landmarks[0]

            # Draw skeleton
            h, w = frame.shape[:2]
            draw_skeleton(frame, landmarks, w, h)
            
            # Wrapper to keep existing code working
            class DummyLandmarkList:
                def __init__(self, lms):
                    self.landmark = lms
            lm = DummyLandmarkList(landmarks)

            # Gesture + raw cursor
            cur_gesture, raw_pos = gesture.tick(lm)

            if raw_pos is not None:
                nx, ny = raw_pos
                cursor_norm = (nx, ny)

                # 1) Kalman prediction
                kx, ky = kalman.update(nx, ny)

                # 2) Ring-buffer smoothing
                buf_x.push(kx); buf_y.push(ky)
                sx = buf_x.mean(); sy = buf_y.mean()

                # 3) Map to global screen space
                gx, gy = display.norm_to_global(sx, sy)

                # 4) Dead-zone anti-shake
                dx = abs(gx - last_gx); dy = abs(gy - last_gy)
                if dx > args.deadzone or dy > args.deadzone or cur_gesture == GestureState.DRAGGING:
                    pyautogui.moveTo(gx, gy, _pause=False)
                    last_gx, last_gy = gx, gy
            else:
                cursor_norm = None
        else:
            cur_gesture = GestureState.IDLE
            cursor_norm = None
            kalman.initialized = False   # reset predictor when hand lost

        # ── FPS ───────────────────────────────────────────────────
        now   = time.perf_counter()
        fps_buf.append(1.0 / max(now - prev_t, 1e-6))
        prev_t = now
        avg_fps = sum(fps_buf) / len(fps_buf)

        # ── HUD ───────────────────────────────────────────────────
        frame = draw_hud(frame, cur_gesture, avg_fps, cursor_norm, display, args.debug)

        cv2.imshow("AI Air Mouse  [Q = quit]", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('m'):
            MIRROR = not MIRROR
        elif key == ord('d'):
            args.debug = not args.debug

    # Cleanup
    if gesture._drag_active:
        pyautogui.mouseUp()
    cap.release()
    cv2.destroyAllWindows()
    # detector handles its own memory, or we can call .close()
    try:
        detector.close()
    except Exception:
        pass
    print("[INFO] Air Mouse stopped.")


if __name__ == "__main__":
    main()
