import re

with open('air_mouse.py', 'r') as f:
    content = f.read()

# Add subprocess
content = content.replace('import sys\nimport time', 'import subprocess\nimport sys\nimport time')

# Update GestureState
state_block = """class GestureState:
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
"""
content = re.sub(r'class GestureState:[\s\S]*?SCROLL_DOWN\n', state_block, content)

# Update GestureEngine __init__
init_block = """    def __init__(self):
        self.state        = GestureState.IDLE
        self._pinch_hold  = 0
        self._scroll_cd   = 0
        self._drag_active = False
        self._last_click_t = 0.0
        self.click_cooldown = 0.35  # seconds
        
        # New Feature State
        self._last_vol = -1
        self._history_x = collections.deque(maxlen=15)
        self._temp_state = None
        self._temp_state_t = 0.0"""
content = content.replace("""    def __init__(self):
        self.state        = GestureState.IDLE
        self._pinch_hold  = 0
        self._scroll_cd   = 0
        self._drag_active = False
        self._last_click_t = 0.0
        self.click_cooldown = 0.35  # seconds""", init_block)

# Add helper functions to GestureEngine
helpers_block = """    def _pinch_14(self, lm):
        \"\"\"Index + Ring finger tips distance.\"\"\"
        return dist(self._tip(lm, 8), self._tip(lm, 16))

    def _pinch_48(self, lm):
        \"\"\"Thumb + Index finger tips distance.\"\"\"
        return dist(self._tip(lm, 4), self._tip(lm, 8))

    def _other_fingers_closed(self, lm):
        tips = [12, 16, 20]
        bases = [10, 14, 18]
        return all(lm.landmark[t].y > lm.landmark[b].y for t, b in zip(tips, bases))

    def _open_hand(self, lm):
        tips_4 = [8, 12, 16, 20]
        bases_4 = [6, 10, 14, 18]
        up = all(lm.landmark[t].y < lm.landmark[b].y for t, b in zip(tips_4, bases_4))
        thumb_ext = dist(lm.landmark[4], lm.landmark[17]) > 0.1
        return up and thumb_ext"""
content = re.sub(r'    def _pinch_14.*?return dist\(self\._tip\(lm, 8\), self\._tip\(lm, 16\)\)', helpers_block, content, flags=re.DOTALL)

# Update tick method
tick_body = """    def tick(self, lm):
        \"\"\"
        Returns (gesture_label, cursor_norm_xy | None)
        \"\"\"
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

        # ── Desktop Swiping (Open Hand + Velocity) ─────────────────
        if self._open_hand(lm) and len(self._history_x) == 15:
            dx = self._history_x[-1] - self._history_x[0]
            if dx > 0.20:  # Swipe Right
                pyautogui.hotkey('ctrl', 'right')
                self._history_x.clear()
                self._temp_state = GestureState.SWIPING
                self._temp_state_t = now
                return GestureState.SWIPING, cursor_pos
            elif dx < -0.20: # Swipe Left
                pyautogui.hotkey('ctrl', 'left')
                self._history_x.clear()
                self._temp_state = GestureState.SWIPING
                self._temp_state_t = now
                return GestureState.SWIPING, cursor_pos

        # ── Volume Control (Thumb + Index Pinch) ───────────────────
        if p48 < self.PINCH_CLOSE and self._other_fingers_closed(lm):
            self._reset_drag()
            # Map Y pos to volume: y=0.2 (100%), y=0.8 (0%)
            vol = max(0, min(100, int((0.8 - index_tip.y) / 0.6 * 100)))
            if abs(vol - self._last_vol) >= 2:
                subprocess.Popen(["osascript", "-e", f"set volume output volume {vol}"])
                self._last_vol = vol
            return GestureState.VOLUME_CTRL, cursor_pos

        # ── Scroll up (three fingers) ─────────────────────────────
        if self._three_fingers_up(lm):
            self._reset_drag()
            self._scroll_cd -= 1
            if self._scroll_cd <= 0:
                pyautogui.scroll(self.SCROLL_SPEED)
                self._scroll_cd = self.SCROLL_COOLDOWN
            return GestureState.SCROLL_UP, None

        # ── Scroll down (pinky) ───────────────────────────────────
        if self._pinky_gesture(lm):
            self._reset_drag()
            self._scroll_cd -= 1
            if self._scroll_cd <= 0:
                pyautogui.scroll(-self.SCROLL_SPEED)
                self._scroll_cd = self.SCROLL_COOLDOWN
            return GestureState.SCROLL_DOWN, None

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
                if not self._drag_active:
                    pyautogui.mouseDown()
                    self._drag_active = True
                return GestureState.DRAGGING, cursor_pos
            else:
                return GestureState.LEFT_CLICK, cursor_pos
        else:
            if self._drag_active:
                pyautogui.mouseUp()
                self._drag_active = False
            if 0 < self._pinch_hold < self.DRAG_HOLD_FRAMES:
                if now - self._last_click_t < 0.4:
                    pyautogui.doubleClick()
                    self._last_click_t = 0.0
                    self._temp_state = GestureState.DOUBLE_CLICK
                    self._temp_state_t = now
                    self._pinch_hold = 0
                    return GestureState.DOUBLE_CLICK, cursor_pos
                else:
                    pyautogui.click()
                    self._last_click_t = now
            self._pinch_hold = 0
            return GestureState.HOVERING, cursor_pos"""
content = re.sub(r'    def tick.*?return GestureState\.HOVERING, cursor_pos', tick_body, content, flags=re.DOTALL)

# Update GESTURE_COLORS
colors_block = """GESTURE_COLORS = {
    GestureState.HOVERING:    (0, 230, 120),
    GestureState.LEFT_CLICK:  (0, 180, 255),
    GestureState.DOUBLE_CLICK:(0, 255, 255),
    GestureState.DRAGGING:    (0, 100, 255),
    GestureState.RIGHT_CLICK: (200, 80, 255),
    GestureState.SCROLL_UP:   (255, 220, 0),
    GestureState.SCROLL_DOWN: (255, 140, 0),
    GestureState.VOLUME_CTRL: (255, 50, 100),
    GestureState.SWIPING:     (50, 255, 50),
    GestureState.IDLE:        (120, 120, 120),
}"""
content = re.sub(r'GESTURE_COLORS = \{.*?\}', colors_block, content, flags=re.DOTALL)

# Update GESTURE_ICONS
icons_block = """GESTURE_ICONS = {
    GestureState.HOVERING:    "☝  MOVE",
    GestureState.LEFT_CLICK:  "🤏 L-CLICK",
    GestureState.DOUBLE_CLICK:"⚡ D-CLICK",
    GestureState.DRAGGING:    "✊  DRAG",
    GestureState.RIGHT_CLICK: "🤞 R-CLICK",
    GestureState.SCROLL_UP:   "🖐  SCROLL ▲",
    GestureState.SCROLL_DOWN: "🤙 SCROLL ▼",
    GestureState.VOLUME_CTRL: "🔊 VOLUME",
    GestureState.SWIPING:     "💨 SWIPE",
    GestureState.IDLE:        "—  IDLE",
}"""
content = re.sub(r'GESTURE_ICONS = \{.*?\}', icons_block, content, flags=re.DOTALL)

with open('air_mouse.py', 'w') as f:
    f.write(content)
