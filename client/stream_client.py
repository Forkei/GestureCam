"""
CV Camera Interactive Stream Client

Connects to the Android camera stream and provides full keyboard control
over camera settings with a live HUD overlay.

On startup, runs a calibration phase:
  - Lets auto-exposure and auto-focus settle for a few seconds
  - Reads the values the camera chose
  - Locks exposure, ISO, focus, and white balance

Usage:
    python stream_client.py <phone_ip>
    python stream_client.py 127.0.0.1      (via ADB port forwarding)

Controls:
    SPACE   Toggle auto/manual exposure (lock exposure)
    A       Toggle auto/manual focus
    E / D   Exposure compensation +/-  (auto mode)
    I / K   ISO up/down                (manual mode)
    T / G   Exposure time up/down      (manual mode)
    F / V   Focus far/near
    Z / X   Zoom in/out
    W       Cycle white balance
    R       Cycle resolution
    J / L   JPEG quality -/+
    N       Toggle hand tracking
    M       Toggle gesture-to-camera control
    0       Re-calibrate (unlock, settle, re-lock)
    H       Toggle help overlay
    S       Print full status to console
    Q/ESC   Quit
"""

import os
import sys
import socket
import struct
import json
import time
import threading

# Suppress Samsung's non-standard JPEG warnings from libjpeg
if sys.platform == "win32":
    _stderr_fd = os.dup(2)
    _devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_devnull, 2)
    os.close(_devnull)

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import cv2
import numpy as np

# Restore stderr now that cv2 is loaded (suppression was only for JPEG warnings)
if sys.platform == "win32":
    os.dup2(_stderr_fd, 2)
    os.close(_stderr_fd)

from config import HandTrackingConfig, GestureConfig, VisualizationConfig
from hand_tracker import HandTracker
from gesture_recognizer import GestureRecognizer, StaticGesture, DynamicGesture
from gesture_event import GestureEventSystem
from gesture_mappings import GestureCameraMapper

# --- Defaults ---
DEFAULT_RESOLUTION = (640, 480)
DEFAULT_JPEG_QUALITY = 70
DEFAULT_WB = "daylight"
CALIBRATION_SECONDS = 3

RESOLUTIONS = [
    (640, 480),
    (1280, 720),
    (1920, 1080),
    (2400, 1080),
    (3840, 2160),
]

WB_MODES = ["auto", "daylight", "cloudy", "shade", "fluorescent", "incandescent", "warm_fluorescent", "twilight"]

EXPOSURE_TIMES = [
    500_000,       # 1/2000s
    1_000_000,     # 1/1000s
    2_000_000,     # 1/500s
    4_000_000,     # 1/250s
    8_000_000,     # 1/125s
    16_666_666,    # 1/60s
    33_333_333,    # 1/30s
    66_666_666,    # 1/15s
    100_000_000,   # 1/10s
]

ISO_VALUES = [50, 100, 200, 400, 800, 1600, 3200]


def recv_exact(sock: socket.socket, n: int) -> bytes:
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("Connection closed")
        data += chunk
    return data


class FrameGrabber:
    """Background thread that reads frames from the stream socket,
    always keeping only the latest frame. Eliminates buffering latency."""

    def __init__(self, sock: socket.socket):
        self._sock = sock
        self._latest: bytes = b""
        self._lock = threading.Lock()
        self._new_frame = threading.Event()
        self._running = True
        self._dropped = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while self._running:
            try:
                length_bytes = recv_exact(self._sock, 4)
                length = struct.unpack(">I", length_bytes)[0]
                jpeg_data = recv_exact(self._sock, length)
                with self._lock:
                    if self._latest:
                        self._dropped += 1
                    self._latest = jpeg_data
                self._new_frame.set()
            except Exception:
                break

    def get(self, timeout: float = 1.0) -> bytes:
        """Block until a new frame is available, return JPEG bytes."""
        self._new_frame.wait(timeout)
        with self._lock:
            data = self._latest
            self._latest = b""
            self._new_frame.clear()
        return data

    @property
    def dropped(self) -> int:
        return self._dropped

    def stop(self):
        self._running = False


def format_exposure(ns: int) -> str:
    if ns <= 0:
        return "auto"
    sec = ns / 1_000_000_000
    if sec >= 0.1:
        return f"{sec:.1f}s"
    return f"1/{int(round(1/sec))}s"


def find_nearest_index(values, target):
    best = 0
    for i, v in enumerate(values):
        if abs(v - target) < abs(values[best] - target):
            best = i
    return best


class CameraControl:
    def __init__(self, host: str, port: int = 5001):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(5)
        self.sock.connect((host, port))
        self.reader = self.sock.makefile("r")

    def send(self, command: dict) -> dict:
        try:
            self.sock.sendall((json.dumps(command) + "\n").encode())
            line = self.reader.readline().strip()
            return json.loads(line) if line else {"status": "error", "message": "empty response"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def set_exposure(self, ns: int): return self.send({"command": "set_exposure_time", "value": ns})
    def set_iso(self, iso: int): return self.send({"command": "set_iso", "value": iso})
    def set_ev(self, ev: int): return self.send({"command": "set_exposure_compensation", "value": ev})
    def auto_exposure(self): return self.send({"command": "set_auto_exposure"})
    def set_focus(self, d: float): return self.send({"command": "set_focus", "value": d})
    def auto_focus(self): return self.send({"command": "set_auto_focus"})
    def set_zoom(self, r: float): return self.send({"command": "set_zoom", "value": r})
    def set_wb(self, mode: str): return self.send({"command": "set_white_balance", "value": mode})
    def set_resolution(self, w: int, h: int): return self.send({"command": "set_resolution", "width": w, "height": h})
    def set_jpeg_quality(self, q: int): return self.send({"command": "set_jpeg_quality", "value": q})
    def set_fps(self, mn: int, mx: int): return self.send({"command": "set_fps", "min": mn, "max": mx})
    def get_status(self): return self.send({"command": "get_status"})
    def get_capabilities(self): return self.send({"command": "get_capabilities"})
    def get_auto_values(self): return self.send({"command": "get_auto_values"})
    def lock_from_auto(self): return self.send({"command": "lock_from_auto"})

    def close(self):
        try: self.sock.close()
        except: pass


def calibrate(ctrl: CameraControl, stream_sock: socket.socket):
    """Run auto for a few seconds, then lock everything."""
    print(f"\n--- CALIBRATION ---")
    print(f"Setting resolution to {DEFAULT_RESOLUTION[0]}x{DEFAULT_RESOLUTION[1]}, "
          f"JPEG Q{DEFAULT_JPEG_QUALITY}, WB {DEFAULT_WB}")

    ctrl.set_resolution(*DEFAULT_RESOLUTION)
    ctrl.set_jpeg_quality(DEFAULT_JPEG_QUALITY)
    ctrl.auto_exposure()
    ctrl.auto_focus()
    ctrl.set_wb(DEFAULT_WB)

    print(f"Auto-exposure/focus settling for {CALIBRATION_SECONDS}s...")

    # Drain frames during calibration
    start = time.time()
    frames = 0
    while time.time() - start < CALIBRATION_SECONDS:
        try:
            length_bytes = recv_exact(stream_sock, 4)
            length = struct.unpack(">I", length_bytes)[0]
            recv_exact(stream_sock, length)
            frames += 1
        except:
            break

    print(f"  ({frames} frames during calibration)")

    # Read what auto settled on
    auto_vals = ctrl.get_auto_values()
    exp_ns = auto_vals.get("exposure_time_ns", 0)
    iso = auto_vals.get("iso", 0)
    focus = auto_vals.get("focus_distance", 0)

    print(f"  Auto chose: shutter={format_exposure(exp_ns)}, ISO={iso}, focus={focus:.2f} dpt")

    # Lock everything
    result = ctrl.lock_from_auto()
    print(f"  Locked! {result.get('message', '')}")
    print(f"--- READY ---\n")

    return {
        "exposure_time_ns": exp_ns,
        "iso": iso,
        "focus_distance": focus,
    }


def draw_hud(frame, state, show_help):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    cv2.rectangle(overlay, (5, 5), (340, 195), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    y = 25
    line_h = 22
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.5
    white = (255, 255, 255)
    green = (0, 255, 0)
    yellow = (0, 255, 255)

    ae = state.get("ae_mode", "auto")
    ae_color = green if ae == "auto" else yellow
    cv2.putText(frame, f"AE: {ae.upper()}", (15, y), font, fs, ae_color, 1); y += line_h

    if ae == "manual":
        exp = state.get("exposure_time_ns", -1)
        iso = state.get("iso", -1)
        cv2.putText(frame, f"Shutter: {format_exposure(exp)}", (15, y), font, fs, white, 1); y += line_h
        cv2.putText(frame, f"ISO: {iso}", (15, y), font, fs, white, 1); y += line_h
    else:
        ev = state.get("exposure_compensation", 0)
        cv2.putText(frame, f"EV: {ev:+d}", (15, y), font, fs, white, 1); y += line_h
        y += line_h

    af = state.get("af_mode", "auto")
    af_color = green if af == "auto" else yellow
    cv2.putText(frame, f"AF: {af.upper()}", (15, y), font, fs, af_color, 1); y += line_h

    if af == "manual":
        fd = state.get("focus_distance", -1)
        cv2.putText(frame, f"Focus: {fd:.1f} dpt", (15, y), font, fs, white, 1); y += line_h
    else:
        y += line_h

    zoom = state.get("zoom", 1.0)
    cv2.putText(frame, f"Zoom: {zoom:.1f}x", (15, y), font, fs, white, 1); y += line_h

    res = state.get("resolution", "?")
    quality = state.get("jpeg_quality", "?")
    cv2.putText(frame, f"Res: {res}  Q:{quality}", (15, y), font, fs, white, 1); y += line_h

    fps = state.get("_fps", 0)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 130, 25), font, 0.6, green, 2)

    # Hand tracking info (top-right, below FPS)
    ht_on = state.get("hand_tracking_enabled", False)
    if ht_on:
        hands_n = state.get("hands_detected", 0)
        ht_color = green if hands_n > 0 else yellow
        cv2.putText(frame, f"Hands: {hands_n}", (w - 130, 50), font, 0.5, ht_color, 1)

        gesture_label = state.get("current_gesture", "")
        if gesture_label:
            cv2.putText(frame, gesture_label, (w - 300, 75), font, 0.5, yellow, 1)

        gc_on = state.get("gesture_control_enabled", False)
        gc_color = green if gc_on else (128, 128, 128)
        gc_text = "Gesture Ctrl: ON" if gc_on else "Gesture Ctrl: OFF"
        cv2.putText(frame, gc_text, (w - 200, 100), font, 0.45, gc_color, 1)
    else:
        cv2.putText(frame, "Hands: OFF", (w - 130, 50), font, 0.5, (128, 128, 128), 1)

    if show_help:
        help_lines = [
            "SPACE  Lock/unlock exposure",
            "A      Lock/unlock focus",
            "E/D    EV comp +/-",
            "I/K    ISO +/-",
            "T/G    Shutter +/-",
            "F/V    Focus far/near",
            "Z/X    Zoom +/-",
            "W      White balance",
            "R      Resolution",
            "J/L    JPEG quality -/+",
            "0      Re-calibrate",
            "N      Toggle hand tracking",
            "M      Toggle gesture control",
            "H      Toggle help",
            "Q/ESC  Quit",
        ]
        bx, by = w - 290, 50
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (bx - 5, by - 20), (bx + 270, by + len(help_lines) * 20 + 5), (0, 0, 0), -1)
        cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)
        for i, line in enumerate(help_lines):
            cv2.putText(frame, line, (bx, by + i * 20), font, 0.45, white, 1)

    return frame


def main():
    if len(sys.argv) < 2:
        print("Usage: python stream_client.py <phone_ip>")
        print("       python stream_client.py 127.0.0.1")
        sys.exit(1)

    host = sys.argv[1]
    stream_port = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    control_port = int(sys.argv[3]) if len(sys.argv) > 3 else 5001

    # Connect control
    print(f"Connecting control to {host}:{control_port}...")
    ctrl = CameraControl(host, control_port)

    caps = ctrl.get_capabilities()
    print(f"Camera: {caps.get('available_resolutions', '?')}")

    # Connect stream
    print(f"Connecting stream to {host}:{stream_port}...")
    stream_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    stream_sock.connect((host, stream_port))
    print("Connected!")

    # --- Calibration ---
    cal = calibrate(ctrl, stream_sock)

    # State tracking
    state = {
        "ae_mode": "manual",
        "exposure_time_ns": cal["exposure_time_ns"],
        "iso": cal["iso"],
        "exposure_compensation": 0,
        "af_mode": "manual",
        "focus_distance": cal["focus_distance"],
        "zoom": 1.0,
        "resolution": f"{DEFAULT_RESOLUTION[0]}x{DEFAULT_RESOLUTION[1]}",
        "jpeg_quality": DEFAULT_JPEG_QUALITY,
        "_fps": 0,
        "hand_tracking_enabled": True,
        "gesture_control_enabled": False,
        "hands_detected": 0,
        "current_gesture": "",
    }

    ev_val = 0
    iso_idx = find_nearest_index(ISO_VALUES, cal["iso"])
    exp_idx = find_nearest_index(EXPOSURE_TIMES, cal["exposure_time_ns"])
    focus_val = cal["focus_distance"]
    zoom_val = 1.0
    wb_idx = WB_MODES.index(DEFAULT_WB)
    res_idx = RESOLUTIONS.index(DEFAULT_RESOLUTION)
    quality_val = DEFAULT_JPEG_QUALITY
    show_help = False

    frame_count = 0
    fps_start = time.time()

    # --- Hand tracking & gesture init ---
    tracker = HandTracker()
    gesture_recognizer = GestureRecognizer()
    event_system = GestureEventSystem()
    gesture_mapper = GestureCameraMapper(ctrl, state, event_system)


    cv2.namedWindow("CV Camera", cv2.WINDOW_NORMAL)
    grabber = FrameGrabber(stream_sock)

    try:
        while True:
            jpeg_data = grabber.get()
            if not jpeg_data:
                continue

            frame = cv2.imdecode(np.frombuffer(jpeg_data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            frame_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 0.5:
                state["_fps"] = frame_count / elapsed
                frame_count = 0
                fps_start = time.time()

            # --- Hand tracking ---
            if state["hand_tracking_enabled"]:
                hands = tracker.process(frame)
                state["hands_detected"] = len(hands)

                if hands:
                    gesture_results = gesture_recognizer.recognize(hands)
                    event_system.process_results(gesture_results)

                    # Build label from first hand's gestures
                    labels = []
                    for gr in gesture_results:
                        if gr.static_gesture != StaticGesture.NONE:
                            labels.append(gr.static_gesture.name)
                        if gr.dynamic_gesture != DynamicGesture.NONE:
                            labels.append(gr.dynamic_gesture.name)
                    state["current_gesture"] = " | ".join(labels) if labels else ""

                    # Draw overlays
                    tracker.draw(frame, hands)

                    # Draw gesture labels above each hand
                    for i, (hand, gr) in enumerate(zip(hands, gesture_results)):
                        bx, by, bw, bh = hand.bounding_box
                        label_parts = []
                        if gr.static_gesture != StaticGesture.NONE:
                            label_parts.append(gr.static_gesture.name.replace("_", " "))
                        if gr.dynamic_gesture != DynamicGesture.NONE:
                            label_parts.append(gr.dynamic_gesture.name.replace("_", " "))
                        if label_parts:
                            label = " | ".join(label_parts)
                            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                            tx = bx + (bw - text_size[0]) // 2
                            ty = by - 25
                            cv2.rectangle(frame, (tx - 4, ty - text_size[1] - 4),
                                          (tx + text_size[0] + 4, ty + 4), (0, 0, 0), -1)
                            cv2.putText(frame, label, (tx, ty),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                else:
                    state["current_gesture"] = ""
            else:
                state["hands_detected"] = 0
                state["current_gesture"] = ""

            frame = draw_hud(frame, state, show_help)
            cv2.imshow("CV Camera", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27 or key == ord('q'):
                break

            elif key == ord('0'):  # Re-calibrate
                ctrl.auto_exposure()
                ctrl.auto_focus()
                state["ae_mode"] = "auto"
                state["af_mode"] = "auto"
                print("Re-calibrating...")
                cal = calibrate(ctrl, stream_sock)
                state["ae_mode"] = "manual"
                state["af_mode"] = "manual"
                state["exposure_time_ns"] = cal["exposure_time_ns"]
                state["iso"] = cal["iso"]
                state["focus_distance"] = cal["focus_distance"]
                iso_idx = find_nearest_index(ISO_VALUES, cal["iso"])
                exp_idx = find_nearest_index(EXPOSURE_TIMES, cal["exposure_time_ns"])
                focus_val = cal["focus_distance"]

            elif key == ord(' '):
                if state["ae_mode"] == "auto":
                    # Lock from current auto
                    auto = ctrl.get_auto_values()
                    ctrl.set_exposure(auto.get("exposure_time_ns", EXPOSURE_TIMES[exp_idx]))
                    ctrl.set_iso(auto.get("iso", ISO_VALUES[iso_idx]))
                    state["ae_mode"] = "manual"
                    state["exposure_time_ns"] = auto.get("exposure_time_ns", 0)
                    state["iso"] = auto.get("iso", 0)
                    iso_idx = find_nearest_index(ISO_VALUES, state["iso"])
                    exp_idx = find_nearest_index(EXPOSURE_TIMES, state["exposure_time_ns"])
                    print(f"[LOCKED] Exposure: {format_exposure(state['exposure_time_ns'])}, ISO: {state['iso']}")
                else:
                    ctrl.auto_exposure()
                    state["ae_mode"] = "auto"
                    ev_val = 0
                    state["exposure_compensation"] = 0
                    print("[UNLOCKED] Auto exposure")

            elif key == ord('a'):
                if state["af_mode"] == "auto":
                    auto = ctrl.get_auto_values()
                    focus_val = auto.get("focus_distance", focus_val)
                    ctrl.set_focus(focus_val)
                    state["af_mode"] = "manual"
                    state["focus_distance"] = focus_val
                    print(f"[LOCKED] Focus: {focus_val:.2f} diopters")
                else:
                    ctrl.auto_focus()
                    state["af_mode"] = "auto"
                    print("[UNLOCKED] Auto focus")

            elif key == ord('e'):
                if state["ae_mode"] == "auto":
                    ev_val = min(ev_val + 2, 20)
                    ctrl.set_ev(ev_val)
                    state["exposure_compensation"] = ev_val

            elif key == ord('d'):
                if state["ae_mode"] == "auto":
                    ev_val = max(ev_val - 2, -20)
                    ctrl.set_ev(ev_val)
                    state["exposure_compensation"] = ev_val

            elif key == ord('i'):
                iso_idx = min(iso_idx + 1, len(ISO_VALUES) - 1)
                ctrl.set_iso(ISO_VALUES[iso_idx])
                state["ae_mode"] = "manual"
                state["iso"] = ISO_VALUES[iso_idx]

            elif key == ord('k'):
                iso_idx = max(iso_idx - 1, 0)
                ctrl.set_iso(ISO_VALUES[iso_idx])
                state["ae_mode"] = "manual"
                state["iso"] = ISO_VALUES[iso_idx]

            elif key == ord('t'):
                exp_idx = max(exp_idx - 1, 0)
                ctrl.set_exposure(EXPOSURE_TIMES[exp_idx])
                state["ae_mode"] = "manual"
                state["exposure_time_ns"] = EXPOSURE_TIMES[exp_idx]

            elif key == ord('g'):
                exp_idx = min(exp_idx + 1, len(EXPOSURE_TIMES) - 1)
                ctrl.set_exposure(EXPOSURE_TIMES[exp_idx])
                state["ae_mode"] = "manual"
                state["exposure_time_ns"] = EXPOSURE_TIMES[exp_idx]

            elif key == ord('f'):
                focus_val = max(focus_val - 0.5, 0.0)
                ctrl.set_focus(focus_val)
                state["af_mode"] = "manual"
                state["focus_distance"] = focus_val

            elif key == ord('v'):
                focus_val = min(focus_val + 0.5, 10.0)
                ctrl.set_focus(focus_val)
                state["af_mode"] = "manual"
                state["focus_distance"] = focus_val

            elif key == ord('z'):
                zoom_val = min(zoom_val + 0.5, 8.0)
                ctrl.set_zoom(zoom_val)
                state["zoom"] = zoom_val
                gesture_mapper.sync_state()

            elif key == ord('x'):
                zoom_val = max(zoom_val - 0.5, 1.0)
                ctrl.set_zoom(zoom_val)
                state["zoom"] = zoom_val
                gesture_mapper.sync_state()

            elif key == ord('w'):
                wb_idx = (wb_idx + 1) % len(WB_MODES)
                ctrl.set_wb(WB_MODES[wb_idx])
                print(f"WB: {WB_MODES[wb_idx]}")

            elif key == ord('r'):
                res_idx = (res_idx + 1) % len(RESOLUTIONS)
                w, h = RESOLUTIONS[res_idx]
                ctrl.set_resolution(w, h)
                state["resolution"] = f"{w}x{h}"
                print(f"Resolution: {w}x{h}")

            elif key == ord('j'):
                quality_val = max(quality_val - 10, 10)
                ctrl.set_jpeg_quality(quality_val)
                state["jpeg_quality"] = quality_val

            elif key == ord('l'):
                quality_val = min(quality_val + 10, 100)
                ctrl.set_jpeg_quality(quality_val)
                state["jpeg_quality"] = quality_val

            elif key == ord('h'):
                show_help = not show_help

            elif key == ord('n'):
                state["hand_tracking_enabled"] = not state["hand_tracking_enabled"]
                status = "ON" if state["hand_tracking_enabled"] else "OFF"
                print(f"[Hand Tracking] {status}")
                if not state["hand_tracking_enabled"]:
                    gesture_recognizer.clear()

            elif key == ord('m'):
                gesture_mapper.sync_state()
                enabled = gesture_mapper.toggle()
                state["gesture_control_enabled"] = enabled

            elif key == ord('s'):
                s = ctrl.get_status()
                print(json.dumps(s, indent=2))

    except (ConnectionError, struct.error) as e:
        print(f"\nStream ended: {e}")
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        grabber.stop()
        tracker.close()
        stream_sock.close()
        ctrl.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
