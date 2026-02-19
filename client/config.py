"""Centralized configuration for hand tracking and gesture recognition."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class StaticGestureDef:
    """Declarative definition for a static hand gesture."""
    name: str  # must match a StaticGesture enum member
    fingers: Tuple[Optional[bool], Optional[bool], Optional[bool],
                   Optional[bool], Optional[bool]]  # thumb, index, middle, ring, pinky; None = don't care
    thumb_index_close: Optional[bool] = None  # True = tips must be close (OK sign)
    thumb_direction: Optional[str] = None     # "up" or "down"
    confidence: float = 0.85
    priority: int = 0  # higher = checked first


DEFAULT_STATIC_GESTURES: List[StaticGestureDef] = [
    StaticGestureDef(
        name="OK_SIGN",
        fingers=(None, None, True, True, True),
        thumb_index_close=True,
        priority=10,
        confidence=0.85,
    ),
    StaticGestureDef(
        name="THUMBS_UP",
        fingers=(True, False, False, False, False),
        thumb_direction="up",
        priority=5,
        confidence=0.85,
    ),
    StaticGestureDef(
        name="THUMBS_DOWN",
        fingers=(True, False, False, False, False),
        thumb_direction="down",
        priority=5,
        confidence=0.85,
    ),
    StaticGestureDef(
        name="FIST",
        fingers=(False, False, False, False, False),
        confidence=0.9,
        priority=0,
    ),
    StaticGestureDef(
        name="OPEN_PALM",
        fingers=(True, True, True, True, True),
        confidence=0.9,
        priority=0,
    ),
    StaticGestureDef(
        name="PEACE",
        fingers=(None, True, True, False, False),
        confidence=0.9,
        priority=0,
    ),
    StaticGestureDef(
        name="POINTING",
        fingers=(None, True, False, False, False),
        confidence=0.9,
        priority=0,
    ),
    StaticGestureDef(
        name="ROCK",
        fingers=(None, True, False, False, True),
        confidence=0.85,
        priority=0,
    ),
    StaticGestureDef(
        name="THREE",
        fingers=(None, True, True, True, False),
        confidence=0.85,
        priority=0,
    ),
    StaticGestureDef(
        name="FOUR",
        fingers=(None, True, True, True, True),
        confidence=0.85,
        priority=0,
    ),
    StaticGestureDef(
        name="PINKY_UP",
        fingers=(None, False, False, False, True),
        confidence=0.85,
        priority=0,
    ),
]


@dataclass
class HandTrackingConfig:
    max_hands: int = 2
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.5
    # Landmark smoothing: EMA alpha (1.0 = raw/no smoothing, 0.0 = full smooth/laggy)
    landmark_smoothing_alpha: float = 0.5
    # Hand persistence: keep last-seen hand for up to N frames before dropping
    hand_persist_frames: int = 5


@dataclass
class GestureConfig:
    # Finger curl/extension thresholds (angles in degrees)
    finger_curl_threshold: float = 90.0    # below = curled
    finger_extend_threshold: float = 160.0  # above = extended

    # Thumb detection (lateral distance ratio relative to palm width)
    thumb_extend_ratio: float = 0.6

    # Dynamic gesture: swipe
    swipe_min_velocity: float = 0.4    # normalized units per second
    swipe_min_distance: float = 0.12   # normalized units
    swipe_history_frames: int = 15

    # Dynamic gesture: pinch/spread
    pinch_close_threshold: float = 0.05   # normalized distance
    pinch_open_threshold: float = 0.15    # normalized distance
    pinch_rate_threshold: float = 0.3     # change per second
    pinch_history_frames: int = 10

    # Dynamic gesture: circle
    circle_min_points: int = 15
    circle_min_angle: float = 5.0         # radians (almost full circle)
    circle_radius_tolerance: float = 0.4  # fraction of mean radius
    circle_history_frames: int = 30

    # Dynamic gesture: wave
    wave_min_reversals: int = 3
    wave_min_amplitude: float = 0.06
    wave_history_frames: int = 25

    # Static gesture hysteresis: require N consecutive identical frames before switching
    gesture_hysteresis_frames: int = 4

    # Config-driven static gesture definitions
    static_gestures: List[StaticGestureDef] = field(default_factory=lambda: list(DEFAULT_STATIC_GESTURES))

    # Event debouncing
    debounce_static_sec: float = 0.3
    debounce_dynamic_sec: float = 0.5

    # History buffer size (max frames to keep)
    max_history_frames: int = 30


@dataclass
class VisualizationConfig:
    landmark_color: Tuple[int, int, int] = (0, 255, 0)
    landmark_radius: int = 3
    connection_color: Tuple[int, int, int] = (0, 200, 0)
    connection_thickness: int = 2
    bbox_color: Tuple[int, int, int] = (0, 255, 255)
    bbox_thickness: int = 2
    label_color: Tuple[int, int, int] = (255, 255, 255)
    label_bg_color: Tuple[int, int, int] = (0, 0, 0)
    label_font_scale: float = 0.6
    label_thickness: int = 2
