"""Gesture data recording tool for training a Transformer-based gesture classifier.

Records combined body pose (MediaPipe, 33 landmarks) + hand tracking (WiLoR-mini,
21 landmarks per hand) sequences while the user performs gestures in front of the
camera. Saves raw landmark data as .npz files.

Usage:
    python gesture_recorder.py <phone_ip>
    python gesture_recorder.py <phone_ip> [stream_port]

Controls:
    UP/DOWN   Navigate gesture list
    ENTER     Expand/collapse class, or playback a recording
    RIGHT     Expand class
    LEFT      Collapse class
    SPACE     Start/stop recording
    DELETE    Delete selected recording
    N         Add new gesture class
    H         Toggle help overlay
    Q / ESC   Quit
"""

import os
import sys
import socket
import struct
import time
import math
import json
import shutil
import winsound

# Suppress Samsung's non-standard JPEG warnings from libjpeg
if sys.platform == "win32":
    _stderr_fd = os.dup(2)
    _devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_devnull, 2)
    os.close(_devnull)

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import cv2
import numpy as np

if sys.platform == "win32":
    os.dup2(_stderr_fd, 2)
    os.close(_stderr_fd)

from stream_client import FrameGrabber, recv_exact, CameraControl
from stream_client import DEFAULT_RESOLUTION, DEFAULT_JPEG_QUALITY, DEFAULT_WB
from hand_tracker import HandTracker
from pose_tracker import PoseTracker, POSE_CONNECTIONS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_GESTURE_CLASSES = [
    "idle",           # standing still, random non-gesture movement
    "sword_draw",     # reach behind back, pull forward
    "arrow_pull",     # one arm extended, other pulls back
    "walking",        # walking-in-place motion
    "sprinting",      # faster walking-in-place
    "wave_hello",     # waving hand greeting
    "push_forward",   # pushing motion with both hands
    "dodge_left",     # quick lean/step left
    "dodge_right",    # quick lean/step right
]

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RECORDINGS_DIR = os.path.join(_SCRIPT_DIR, "recordings")
CLASSES_JSON = os.path.join(RECORDINGS_DIR, "classes.json")
MIN_RECORDING_FRAMES = 15
WARN_RECORDING_FRAMES = 0  # disabled — long recordings are expected

# Drawing constants
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_VIS_THRESHOLD = 0.5

# Panel constants
PANEL_WIDTH = 260
PANEL_BG_ALPHA = 0.75
PANEL_ITEM_HEIGHT = 22
PANEL_HEADER_HEIGHT = 36
PANEL_FOOTER_HEIGHT = 60
PANEL_COLOR_BG = (30, 30, 30)
PANEL_COLOR_CURSOR = (70, 50, 20)
PANEL_COLOR_HEADER = (0, 220, 220)
PANEL_COLOR_CLASS = (220, 220, 220)
PANEL_COLOR_CLASS_ACTIVE = (0, 255, 200)
PANEL_COLOR_REC_ITEM = (160, 160, 160)
PANEL_COLOR_COUNT = (120, 120, 120)
PANEL_COLOR_EXPAND = (100, 100, 100)
PANEL_COLOR_FOOTER = (140, 140, 140)
PANEL_COLOR_INPUT_BG = (50, 50, 50)
PANEL_COLOR_INPUT_TEXT = (0, 255, 255)

# Key codes for cv2.waitKeyEx
KEY_UP = 2490368
KEY_DOWN = 2621440
KEY_LEFT = 2424832
KEY_RIGHT = 2555904
KEY_DELETE = 3014656
KEY_ENTER = 13
KEY_BACKSPACE = 8
KEY_SPACE = 32
KEY_ESCAPE = 27

# ---------------------------------------------------------------------------
# Gesture class persistence
# ---------------------------------------------------------------------------


def load_gesture_classes():
    """Load gesture classes from JSON, falling back to defaults."""
    if os.path.isfile(CLASSES_JSON):
        try:
            with open(CLASSES_JSON, "r") as f:
                classes = json.load(f)
            if isinstance(classes, list) and all(isinstance(c, str) for c in classes):
                return classes
        except (json.JSONDecodeError, OSError):
            pass
    return list(DEFAULT_GESTURE_CLASSES)


def save_gesture_classes(classes):
    """Persist gesture classes to JSON."""
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    with open(CLASSES_JSON, "w") as f:
        json.dump(classes, f, indent=2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_sample_count(gesture_name: str) -> int:
    """Count existing .npz recordings for a gesture class."""
    gesture_dir = os.path.join(RECORDINGS_DIR, gesture_name)
    if not os.path.isdir(gesture_dir):
        return 0
    return sum(1 for f in os.listdir(gesture_dir) if f.endswith(".npz"))


def next_sample_path(gesture_name: str) -> str:
    """Return the next auto-incremented sample path, creating dirs as needed."""
    gesture_dir = os.path.join(RECORDINGS_DIR, gesture_name)
    os.makedirs(gesture_dir, exist_ok=True)
    existing = [f for f in os.listdir(gesture_dir) if f.endswith(".npz")]
    if not existing:
        return os.path.join(gesture_dir, "sample_000.npz")
    max_idx = -1
    for f in existing:
        try:
            idx = int(f.replace("sample_", "").replace(".npz", ""))
            max_idx = max(max_idx, idx)
        except ValueError:
            pass
    return os.path.join(gesture_dir, f"sample_{max_idx + 1:03d}.npz")


def load_recordings_info(class_name):
    """Return [(filename, n_frames, duration_sec), ...] for all .npz in a class dir."""
    gesture_dir = os.path.join(RECORDINGS_DIR, class_name)
    if not os.path.isdir(gesture_dir):
        return []
    result = []
    files = sorted(f for f in os.listdir(gesture_dir) if f.endswith(".npz"))
    for fname in files:
        try:
            data = np.load(os.path.join(gesture_dir, fname), allow_pickle=True)
            ts = data["timestamps"]
            n_frames = len(ts)
            duration = float(ts[-1] - ts[0]) if n_frames > 1 else 0.0
            result.append((fname, n_frames, duration))
        except Exception:
            result.append((fname, 0, 0.0))
    return result


def delete_recording(class_name, filename):
    """Delete a specific recording file. Returns True if deleted."""
    path = os.path.join(RECORDINGS_DIR, class_name, filename)
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


# ---------------------------------------------------------------------------
# Frame data extraction
# ---------------------------------------------------------------------------


def extract_frame_data(pose, hands):
    """Build a dict of numpy arrays from current pose + hand detections."""
    # Pose
    if pose is not None:
        pose_norm = np.array(pose.landmarks_norm, dtype=np.float32)
        if pose.world_landmarks:
            pose_world = np.array(pose.world_landmarks, dtype=np.float32)
        else:
            pose_world = np.zeros((33, 3), dtype=np.float32)
        pose_vis = np.array(pose.visibility, dtype=np.float32)
    else:
        pose_norm = np.zeros((33, 3), dtype=np.float32)
        pose_world = np.zeros((33, 3), dtype=np.float32)
        pose_vis = np.zeros(33, dtype=np.float32)

    # Hands — always store left and right separately, zeros for missing
    left_norm = np.zeros((21, 3), dtype=np.float32)
    left_3d = np.zeros((21, 3), dtype=np.float32)
    right_norm = np.zeros((21, 3), dtype=np.float32)
    right_3d = np.zeros((21, 3), dtype=np.float32)
    left_present = 0.0
    right_present = 0.0

    for hand in hands:
        norm = np.array(hand.landmarks_norm, dtype=np.float32)
        kp3d = (np.array(hand.landmarks_3d, dtype=np.float32)
                if hand.landmarks_3d
                else np.zeros((21, 3), dtype=np.float32))
        if hand.handedness == "Left":
            left_norm = norm
            left_3d = kp3d
            left_present = 1.0
        else:
            right_norm = norm
            right_3d = kp3d
            right_present = 1.0

    return {
        "pose_norm": pose_norm,
        "pose_world": pose_world,
        "pose_visibility": pose_vis,
        "left_hand_norm": left_norm,
        "left_hand_3d": left_3d,
        "right_hand_norm": right_norm,
        "right_hand_3d": right_3d,
        "left_hand_present": left_present,
        "right_hand_present": right_present,
        "timestamp": time.time(),
    }


# ---------------------------------------------------------------------------
# Save recording
# ---------------------------------------------------------------------------


def save_recording(frames_data, gesture_name):
    """Stack per-frame dicts and save as a single .npz file. Returns path."""
    path = next_sample_path(gesture_name)
    np.savez(
        path,
        pose_norm=np.stack([f["pose_norm"] for f in frames_data]),
        pose_world=np.stack([f["pose_world"] for f in frames_data]),
        pose_visibility=np.stack([f["pose_visibility"] for f in frames_data]),
        left_hand_norm=np.stack([f["left_hand_norm"] for f in frames_data]),
        left_hand_3d=np.stack([f["left_hand_3d"] for f in frames_data]),
        right_hand_norm=np.stack([f["right_hand_norm"] for f in frames_data]),
        right_hand_3d=np.stack([f["right_hand_3d"] for f in frames_data]),
        left_hand_present=np.array([f["left_hand_present"] for f in frames_data]),
        right_hand_present=np.array([f["right_hand_present"] for f in frames_data]),
        timestamps=np.array([f["timestamp"] for f in frames_data]),
        gesture=gesture_name,
    )
    return path


# ---------------------------------------------------------------------------
# Playback drawing
# ---------------------------------------------------------------------------

# Borrow hand connection topology from HandTracker
_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def draw_playback_skeleton(frame, data, idx, w, h):
    """Draw pose + hand skeletons from recorded arrays onto *frame*."""
    # Pose
    pose_norm = data["pose_norm"][idx]    # (33, 3)
    pose_vis = data["pose_visibility"][idx]  # (33,)

    for i, j in POSE_CONNECTIONS:
        if pose_vis[i] > _VIS_THRESHOLD and pose_vis[j] > _VIS_THRESHOLD:
            pt1 = (int(pose_norm[i, 0] * w), int(pose_norm[i, 1] * h))
            pt2 = (int(pose_norm[j, 0] * w), int(pose_norm[j, 1] * h))
            cv2.line(frame, pt1, pt2, (255, 100, 100), 2)

    for i in range(33):
        if pose_vis[i] > _VIS_THRESHOLD:
            pt = (int(pose_norm[i, 0] * w), int(pose_norm[i, 1] * h))
            cv2.circle(frame, pt, 3, (255, 0, 255), -1)

    # Hands
    hand_specs = [
        ("left_hand_norm", "left_hand_present", (255, 200, 0)),
        ("right_hand_norm", "right_hand_present", (0, 200, 255)),
    ]
    for norm_key, pres_key, color in hand_specs:
        if data[pres_key][idx] > 0.5:
            hn = data[norm_key][idx]  # (21, 3)
            for i, j in _HAND_CONNECTIONS:
                pt1 = (int(hn[i, 0] * w), int(hn[i, 1] * h))
                pt2 = (int(hn[j, 0] * w), int(hn[j, 1] * h))
                cv2.line(frame, pt1, pt2, color, 2)
            for i in range(21):
                pt = (int(hn[i, 0] * w), int(hn[i, 1] * h))
                cv2.circle(frame, pt, 3, (0, 255, 0), -1)


# ---------------------------------------------------------------------------
# Recorder state
# ---------------------------------------------------------------------------


class RecorderState:
    def __init__(self, classes):
        self.classes = classes
        self.active_class = 0       # index into self.classes
        self.expanded = set()       # set of expanded class indices
        self.cursor = 0             # position in visible items list
        self.scroll = 0             # scroll offset for panel

        self.recording = False
        self.frames_data = []
        self.record_start = 0.0
        self.saved_flash_until = 0.0
        self.countdown_until = 0.0  # timestamp when countdown ends and recording starts

        self.input_mode = False
        self.input_text = ""

        self.show_help = False

        self.playback_mode = False
        self.playback_data = None
        self.playback_idx = 0
        self.playback_total = 0
        self.playback_name = ""

        # Caches
        self.rec_cache = {}         # class_name -> [(fname, n_frames, dur)]
        self.sample_counts = {}     # class_name -> int

        # Visible items list (rebuilt each frame)
        self.visible = []

    def stop_playback(self):
        if self.playback_data is not None:
            self.playback_data.close()
            self.playback_data = None
        self.playback_mode = False
        self.playback_idx = 0
        self.playback_total = 0

    def refresh_counts(self):
        for name in self.classes:
            self.sample_counts[name] = get_sample_count(name)

    def refresh_cache(self, class_name):
        self.rec_cache[class_name] = load_recordings_info(class_name)
        self.sample_counts[class_name] = get_sample_count(class_name)


# ---------------------------------------------------------------------------
# Visible items model
# ---------------------------------------------------------------------------


def build_visible(state):
    """Build flat list of visible panel items.

    Each item is a tuple:
      ("class", class_idx, class_name)
      ("rec", class_idx, filename, info_str)
    """
    items = []
    for ci, name in enumerate(state.classes):
        items.append(("class", ci, name))
        if ci in state.expanded:
            recs = state.rec_cache.get(name, [])
            for fname, n_frames, dur in recs:
                info = f"{n_frames}f {dur:.1f}s"
                items.append(("rec", ci, fname, info))
    state.visible = items
    # Clamp cursor
    if state.visible:
        state.cursor = max(0, min(state.cursor, len(state.visible) - 1))
    else:
        state.cursor = 0


def cursor_class_index(state):
    """Return the class index the cursor is currently on or inside."""
    if not state.visible:
        return 0
    item = state.visible[state.cursor]
    return item[1]


# ---------------------------------------------------------------------------
# Panel drawing — left sidebar
# ---------------------------------------------------------------------------


def draw_panel(frame, state):
    """Draw the left sidebar panel onto *frame* (mutates in place)."""
    h, w = frame.shape[:2]
    pw = min(PANEL_WIDTH, w)
    now = time.time()

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (pw, h), PANEL_BG, -1)
    cv2.addWeighted(overlay, PANEL_BG_ALPHA, frame, 1.0 - PANEL_BG_ALPHA, 0, frame)

    # Header
    cv2.putText(frame, "GESTURES", (10, 26), _FONT, 0.65, PANEL_COLOR_HEADER, 2)
    cv2.line(frame, (0, PANEL_HEADER_HEIGHT), (pw, PANEL_HEADER_HEIGHT),
             (80, 80, 80), 1)

    # Available area for items
    items_y_start = PANEL_HEADER_HEIGHT + 4
    footer_y = h - PANEL_FOOTER_HEIGHT
    visible_height = footer_y - items_y_start
    max_visible = max(1, visible_height // PANEL_ITEM_HEIGHT)

    # Auto-scroll to keep cursor visible
    if state.cursor < state.scroll:
        state.scroll = state.cursor
    elif state.cursor >= state.scroll + max_visible:
        state.scroll = state.cursor - max_visible + 1
    state.scroll = max(0, min(state.scroll, max(0, len(state.visible) - max_visible)))

    # Draw visible items
    for vi in range(max_visible):
        idx = state.scroll + vi
        if idx >= len(state.visible):
            break
        item = state.visible[idx]
        y = items_y_start + vi * PANEL_ITEM_HEIGHT
        y_text = y + PANEL_ITEM_HEIGHT - 5

        # Cursor highlight
        if idx == state.cursor:
            cv2.rectangle(frame, (0, y), (pw, y + PANEL_ITEM_HEIGHT),
                          PANEL_COLOR_CURSOR, -1)

        if item[0] == "class":
            ci = item[1]
            name = item[2]
            count = state.sample_counts.get(name, 0)

            # Active indicator dot
            if ci == state.active_class:
                if state.recording:
                    pulse = int(128 + 127 * math.sin(now * 8))
                    dot_color = (0, 0, pulse)
                else:
                    dot_color = (0, 200, 0)
                cv2.circle(frame, (10, y_text - 4), 4, dot_color, -1)

            # Expand indicator
            if ci in state.expanded:
                indicator = "v"
            else:
                indicator = ">"
            cv2.putText(frame, indicator, (20, y_text), _FONT, 0.38,
                        PANEL_COLOR_EXPAND, 1)

            # Class name
            name_color = PANEL_COLOR_CLASS_ACTIVE if ci == state.active_class else PANEL_COLOR_CLASS
            display_name = name if len(name) <= 18 else name[:17] + ".."
            cv2.putText(frame, display_name, (34, y_text), _FONT, 0.40,
                        name_color, 1)

            # Count on right
            count_str = f"({count})"
            cw = cv2.getTextSize(count_str, _FONT, 0.35, 1)[0][0]
            cv2.putText(frame, count_str, (pw - cw - 8, y_text), _FONT, 0.35,
                        PANEL_COLOR_COUNT, 1)

        elif item[0] == "rec":
            fname = item[2]
            info = item[3]

            # Indented filename (without .npz)
            display = fname.replace(".npz", "")
            if len(display) > 14:
                display = display[:13] + ".."
            cv2.putText(frame, display, (38, y_text), _FONT, 0.35,
                        PANEL_COLOR_REC_ITEM, 1)

            # Info on right
            iw = cv2.getTextSize(info, _FONT, 0.30, 1)[0][0]
            cv2.putText(frame, info, (pw - iw - 8, y_text), _FONT, 0.30,
                        PANEL_COLOR_COUNT, 1)

    # Scroll indicators
    if state.scroll > 0:
        cv2.putText(frame, "...", (pw // 2 - 8, items_y_start + 12), _FONT, 0.35,
                    PANEL_COLOR_COUNT, 1)
    if state.scroll + max_visible < len(state.visible):
        cv2.putText(frame, "...", (pw // 2 - 8, footer_y - 4), _FONT, 0.35,
                    PANEL_COLOR_COUNT, 1)

    # Footer separator
    cv2.line(frame, (0, footer_y), (pw, footer_y), (80, 80, 80), 1)

    # Input mode: text box above footer
    if state.input_mode:
        input_y = footer_y + 4
        cv2.rectangle(frame, (4, input_y), (pw - 4, input_y + 22),
                      PANEL_COLOR_INPUT_BG, -1)
        cv2.rectangle(frame, (4, input_y), (pw - 4, input_y + 22),
                      PANEL_COLOR_INPUT_TEXT, 1)
        blink = "_" if int(now * 3) % 2 == 0 else ""
        cv2.putText(frame, f"Name: {state.input_text}{blink}",
                    (8, input_y + 16), _FONT, 0.38, PANEL_COLOR_INPUT_TEXT, 1)
        hint_y = input_y + 38
        cv2.putText(frame, "ENTER=add  ESC=cancel", (8, hint_y), _FONT, 0.30,
                    PANEL_COLOR_FOOTER, 1)
    else:
        # Footer hints
        hints = [
            "SPACE=rec  ENTER=expand",
            "N=new  DEL=delete  H=help",
        ]
        for i, hint in enumerate(hints):
            hy = footer_y + 18 + i * 18
            cv2.putText(frame, hint, (8, hy), _FONT, 0.32, PANEL_COLOR_FOOTER, 1)


# Alias used for the panel background fill
PANEL_BG = PANEL_COLOR_BG


# ---------------------------------------------------------------------------
# Camera overlay drawing
# ---------------------------------------------------------------------------


def draw_overlay(frame, state, display_fps):
    """Draw recording/status overlays on the camera area (right of panel)."""
    h, w = frame.shape[:2]
    now = time.time()
    pw = min(PANEL_WIDTH, w)
    cam_x = pw  # camera area starts after panel
    cam_w = w - pw

    # Countdown overlay
    if state.countdown_until > 0:
        remaining = state.countdown_until - now
        if remaining > 0:
            digit = int(math.ceil(remaining))
            # Large centered number on camera area
            label = str(digit)
            scale = 4.0
            thickness = 8
            tw, th = cv2.getTextSize(label, _FONT, scale, thickness)[0]
            cx = cam_x + (cam_w - tw) // 2
            cy = (h + th) // 2
            # Dark backdrop circle
            ov = frame.copy()
            cv2.circle(ov, (cam_x + cam_w // 2, h // 2), 80, (0, 0, 0), -1)
            cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)
            # Countdown number
            cv2.putText(frame, label, (cx, cy), _FONT, scale, (0, 200, 255), thickness)
            # "Get ready" text
            ready_text = f"Recording {state.classes[state.active_class]}..."
            rtw = cv2.getTextSize(ready_text, _FONT, 0.6, 2)[0][0]
            cv2.putText(frame, ready_text, (cam_x + (cam_w - rtw) // 2, cy + 50),
                        _FONT, 0.6, (180, 180, 180), 2)

    # Red border while recording
    if state.recording:
        cv2.rectangle(frame, (pw + 2, 2), (w - 3, h - 3), (0, 0, 255), 3)

    # Top-right: state indicator
    if state.countdown_until > 0 and state.countdown_until > now:
        state_color = (0, 200, 255)
        state_text = "GET READY"
    elif state.recording:
        pulse = int(128 + 127 * math.sin(now * 6))
        state_color = (0, 0, pulse)
        state_text = "RECORDING"
    elif now < state.saved_flash_until:
        state_color = (255, 200, 0)
        state_text = "SAVED"
    else:
        state_color = (0, 200, 0)
        state_text = "READY"

    text_w = cv2.getTextSize(state_text, _FONT, 0.7, 2)[0][0]
    cv2.putText(frame, state_text, (w - text_w - 10, 28), _FONT, 0.7,
                state_color, 2)

    # FPS below state
    cv2.putText(frame, f"FPS: {display_fps:.1f}", (w - 110, 50), _FONT, 0.45,
                (0, 200, 0), 1)

    # Active class label at top of camera area
    if state.classes:
        name = state.classes[state.active_class]
        label = f"Class: {name}"
        cv2.putText(frame, label, (pw + 10, 28), _FONT, 0.6, (0, 255, 255), 2)

    # Recording info (center-bottom of camera area)
    if state.recording:
        rec_elapsed = now - state.record_start
        n = len(state.frames_data)
        rec_text = f"REC  {n}f  {rec_elapsed:.1f}s"
        tw = cv2.getTextSize(rec_text, _FONT, 0.7, 2)[0][0]
        rx = cam_x + (cam_w - tw) // 2
        cv2.putText(frame, rec_text, (rx, h - 30), _FONT, 0.7, (0, 0, 255), 2)

        if n >= WARN_RECORDING_FRAMES:
            warn = "Long recording!"
            ww = cv2.getTextSize(warn, _FONT, 0.6, 2)[0][0]
            cv2.putText(frame, warn, (cam_x + (cam_w - ww) // 2, h - 55),
                        _FONT, 0.6, (0, 165, 255), 2)

    # Help overlay (centered in camera area)
    if state.show_help:
        help_lines = [
            "UP/DOWN     Navigate list",
            "ENTER       Expand/collapse or playback",
            "RIGHT/LEFT  Expand/collapse class",
            "SPACE       Start/stop recording",
            "DEL/BKSP    Delete recording",
            "N           Add new gesture class",
            "H           Toggle this help",
            "Q / ESC     Quit",
        ]
        box_w = min(320, cam_w - 20)
        box_h = len(help_lines) * 25 + 40
        bx = cam_x + (cam_w - box_w) // 2
        by = (h - box_h) // 2
        ov = frame.copy()
        cv2.rectangle(ov, (bx, by), (bx + box_w, by + box_h), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.85, frame, 0.15, 0, frame)
        cv2.putText(frame, "CONTROLS", (bx + box_w // 2 - 50, by + 25), _FONT,
                    0.6, (0, 255, 255), 2)
        for i, line in enumerate(help_lines):
            cv2.putText(frame, line, (bx + 15, by + 52 + i * 25), _FONT, 0.42,
                        (255, 255, 255), 1)


# ---------------------------------------------------------------------------
# Key handling
# ---------------------------------------------------------------------------


def handle_key(key, state):
    """Process a key press. Returns 'quit' to exit, else None."""
    if key == -1:
        return None

    # --- Input mode (new class name entry) ---
    if state.input_mode:
        if key == KEY_ESCAPE:
            state.input_mode = False
            state.input_text = ""
        elif key == KEY_ENTER:
            name = state.input_text.strip().lower().replace(" ", "_")
            if name and name not in state.classes:
                state.classes.append(name)
                save_gesture_classes(state.classes)
                state.sample_counts[name] = 0
                # Move cursor to the new class
                build_visible(state)
                for i, item in enumerate(state.visible):
                    if item[0] == "class" and item[2] == name:
                        state.cursor = i
                        state.active_class = item[1]
                        break
                print(f"Added class: {name}")
            state.input_mode = False
            state.input_text = ""
        elif key == KEY_BACKSPACE:
            state.input_text = state.input_text[:-1]
        elif 0 < key < 128:
            ch = chr(key)
            if ch.isalnum() or ch == '_':
                state.input_text += ch
        return None

    # --- Playback mode ---
    if state.playback_mode:
        # Any key exits playback
        state.stop_playback()
        return None

    # --- Countdown mode (cancel with SPACE or ESC) ---
    if state.countdown_until > 0:
        if key == KEY_SPACE or key == KEY_ESCAPE:
            state.countdown_until = 0.0
            print("Countdown cancelled.")
        return None

    # --- Recording mode (only SPACE to stop, ESC/Q to quit) ---
    if state.recording:
        if key == KEY_SPACE:
            state.recording = False
            n = len(state.frames_data)
            if n < MIN_RECORDING_FRAMES:
                print(f"Too short ({n} frames, need {MIN_RECORDING_FRAMES}). Discarded.")
            else:
                gname = state.classes[state.active_class]
                path = save_recording(state.frames_data, gname)
                state.refresh_cache(gname)
                state.saved_flash_until = time.time() + 1.5
                print(f"Saved: {path} ({n} frames)")
            state.frames_data = []
        elif key == KEY_ESCAPE or key == ord('q') or key == ord('Q'):
            # Auto-save and quit
            if len(state.frames_data) >= MIN_RECORDING_FRAMES:
                gname = state.classes[state.active_class]
                path = save_recording(state.frames_data, gname)
                print(f"Auto-saved on exit: {path}")
            state.recording = False
            return "quit"
        return None

    # --- Normal mode ---
    if key == KEY_UP:
        if state.visible and state.cursor > 0:
            state.cursor -= 1
            state.active_class = cursor_class_index(state)

    elif key == KEY_DOWN:
        if state.visible and state.cursor < len(state.visible) - 1:
            state.cursor += 1
            state.active_class = cursor_class_index(state)

    elif key == KEY_ENTER:
        if state.visible:
            item = state.visible[state.cursor]
            if item[0] == "class":
                ci = item[1]
                if ci in state.expanded:
                    state.expanded.discard(ci)
                else:
                    # Expand: load recordings
                    name = state.classes[ci]
                    state.refresh_cache(name)
                    state.expanded.add(ci)
                state.active_class = ci
            elif item[0] == "rec":
                # Play back this specific recording
                ci = item[1]
                fname = item[2]
                name = state.classes[ci]
                path = os.path.join(RECORDINGS_DIR, name, fname)
                if os.path.exists(path):
                    state.playback_data = np.load(path, allow_pickle=True)
                    state.playback_total = len(state.playback_data["timestamps"])
                    state.playback_idx = 0
                    state.playback_mode = True
                    state.playback_name = f"{name}/{fname}"
                    print(f"Playing: {path} ({state.playback_total} frames)")

    elif key == KEY_RIGHT:
        if state.visible:
            item = state.visible[state.cursor]
            ci = item[1]
            if ci not in state.expanded:
                name = state.classes[ci]
                state.refresh_cache(name)
                state.expanded.add(ci)
                state.active_class = ci

    elif key == KEY_LEFT:
        if state.visible:
            item = state.visible[state.cursor]
            ci = item[1]
            if ci in state.expanded:
                state.expanded.discard(ci)
                # Move cursor to the class header
                for i, v in enumerate(state.visible):
                    if v[0] == "class" and v[1] == ci:
                        state.cursor = i
                        break
            elif item[0] == "rec":
                # Collapse parent class
                state.expanded.discard(ci)
                for i, v in enumerate(state.visible):
                    if v[0] == "class" and v[1] == ci:
                        state.cursor = i
                        break

    elif key == KEY_SPACE:
        if state.classes:
            state.countdown_until = time.time() + 1.0
            name = state.classes[state.active_class]
            print(f"Recording '{name}' in 1...")

    elif key == KEY_DELETE or key == KEY_BACKSPACE:
        if state.visible:
            item = state.visible[state.cursor]
            if item[0] == "class":
                ci = item[1]
                if len(state.classes) <= 1:
                    print("Can't delete the last class.")
                else:
                    name = state.classes[ci]
                    # Remove recordings directory
                    gesture_dir = os.path.join(RECORDINGS_DIR, name)
                    if os.path.isdir(gesture_dir):
                        shutil.rmtree(gesture_dir)
                    # Remove from state
                    state.classes.pop(ci)
                    save_gesture_classes(state.classes)
                    state.expanded.discard(ci)
                    # Shift expanded indices above the removed one
                    state.expanded = {i - 1 if i > ci else i for i in state.expanded}
                    state.sample_counts.pop(name, None)
                    state.rec_cache.pop(name, None)
                    # Fix active_class
                    if state.active_class >= len(state.classes):
                        state.active_class = len(state.classes) - 1
                    elif state.active_class > ci:
                        state.active_class -= 1
                    # Rebuild and clamp cursor
                    build_visible(state)
                    print(f"Deleted class: {name}")
            elif item[0] == "rec":
                ci = item[1]
                fname = item[2]
                name = state.classes[ci]
                if delete_recording(name, fname):
                    state.refresh_cache(name)
                    print(f"Deleted: {name}/{fname}")

    elif key == ord('n') or key == ord('N'):
        state.input_mode = True
        state.input_text = ""

    elif key == ord('h') or key == ord('H'):
        state.show_help = not state.show_help

    elif key == KEY_ESCAPE or key == ord('q') or key == ord('Q'):
        return "quit"

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    if len(sys.argv) < 2:
        print("Usage: python gesture_recorder.py <phone_ip> [stream_port] [control_port]")
        sys.exit(1)

    host = sys.argv[1]
    stream_port = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    control_port = int(sys.argv[3]) if len(sys.argv) > 3 else 5001

    # --- Connect control ---
    print(f"Connecting control to {host}:{control_port}...")
    ctrl = CameraControl(host, control_port)

    # --- Connect to camera stream ---
    print(f"Connecting stream to {host}:{stream_port}...")
    stream_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    stream_sock.connect((host, stream_port))
    print("Connected!")

    # Auto exposure/focus for recording
    ctrl.auto_exposure()
    ctrl.auto_focus()
    ctrl.set_resolution(*DEFAULT_RESOLUTION)
    ctrl.set_jpeg_quality(DEFAULT_JPEG_QUALITY)
    ctrl.set_wb(DEFAULT_WB)

    # --- Init trackers ---
    print("Initializing hand tracker (WiLoR-mini)...")
    hand_tracker = HandTracker()
    print("Initializing pose tracker (MediaPipe)...")
    pose_tracker = PoseTracker()
    print("Ready.\n")

    # --- State ---
    classes = load_gesture_classes()
    state = RecorderState(classes)
    state.refresh_counts()

    # FPS tracking
    frame_count = 0
    fps_start = time.time()
    display_fps = 0.0

    cv2.namedWindow("Gesture Recorder", cv2.WINDOW_NORMAL)
    grabber = FrameGrabber(stream_sock)

    try:
        while True:
            # Rebuild visible items each iteration
            build_visible(state)

            # ---- Playback mode ----
            if state.playback_mode and state.playback_data is not None:
                if state.playback_idx < state.playback_total:
                    pf = np.zeros((480, 640, 3), dtype=np.uint8)
                    draw_playback_skeleton(pf, state.playback_data,
                                           state.playback_idx, 640, 480)
                    info = (f"PLAYBACK [{state.playback_name}]  "
                            f"{state.playback_idx + 1}/{state.playback_total}")
                    cv2.putText(pf, info, (15, 30), _FONT, 0.6,
                                (0, 255, 255), 1)
                    cv2.putText(pf, "Press any key to exit", (15, 460),
                                _FONT, 0.5, (150, 150, 150), 1)
                    cv2.imshow("Gesture Recorder", pf)
                    # Compute delay from actual recorded timestamps
                    ts = state.playback_data["timestamps"]
                    if state.playback_idx + 1 < state.playback_total:
                        delay_ms = int((ts[state.playback_idx + 1] - ts[state.playback_idx]) * 1000)
                        delay_ms = max(1, min(delay_ms, 200))
                    else:
                        delay_ms = 33
                    state.playback_idx += 1
                    key = cv2.waitKeyEx(delay_ms)
                    if key != -1:
                        state.stop_playback()
                    continue
                else:
                    state.stop_playback()
                    continue

            # ---- Grab frame ----
            jpeg_data = grabber.get()
            if not jpeg_data:
                continue

            frame = cv2.imdecode(
                np.frombuffer(jpeg_data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # FPS
            frame_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 0.5:
                display_fps = frame_count / elapsed
                frame_count = 0
                fps_start = time.time()

            # ---- Process trackers ----
            hands = hand_tracker.process(frame)
            pose = pose_tracker.process(frame)

            # Draw live overlays
            hand_tracker.draw(frame, hands)
            if pose:
                pose_tracker.draw(frame, pose)

            # ---- Countdown → start recording ----
            if state.countdown_until > 0 and time.time() >= state.countdown_until:
                state.countdown_until = 0.0
                state.recording = True
                state.frames_data = []
                state.record_start = time.time()
                name = state.classes[state.active_class]
                winsound.Beep(1000, 100)  # 1kHz for 100ms
                print(f"Recording '{name}'!")

            # ---- Record ----
            if state.recording:
                state.frames_data.append(extract_frame_data(pose, hands))

            # ---- Draw UI ----
            draw_panel(frame, state)
            draw_overlay(frame, state, display_fps)

            cv2.imshow("Gesture Recorder", frame)
            key = cv2.waitKeyEx(1)

            # ---- Key handling ----
            result = handle_key(key, state)
            if result == "quit":
                break

    except (ConnectionError, struct.error) as e:
        print(f"\nStream ended: {e}")
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        grabber.stop()
        hand_tracker.close()
        pose_tracker.close()
        stream_sock.close()
        ctrl.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
