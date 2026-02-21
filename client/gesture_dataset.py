"""PyTorch Dataset for gesture classification from recorded pose+hand landmarks.

Loads .npz recordings from client/recordings/, extracts a 260-dim feature vector
per frame, and provides padded/masked sequences for a Transformer model.

Feature vector (260 dims per frame):
    pose_world      (33×3 = 99)  — MediaPipe world coords in meters, hip-centered
    left_hand_3d    (21×3 = 63)  — WiLoR left hand 3D coords
    right_hand_3d   (21×3 = 63)  — WiLoR right hand 3D coords
    left_hand_present   (1)      — 0/1 flag
    right_hand_present  (1)      — 0/1 flag
    pose_visibility     (33)     — per-landmark confidence
"""

import os
import json
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RECORDINGS_DIR = os.path.join(_SCRIPT_DIR, "recordings")
CLASSES_JSON = os.path.join(RECORDINGS_DIR, "classes.json")

INPUT_DIM = 260
DEFAULT_MAX_SEQ_LEN = 30
WINDOW_SIZE = 30
WINDOW_STRIDE = 5


# ---------------------------------------------------------------------------
# Data augmentation helpers
# ---------------------------------------------------------------------------

def _time_warp(features: np.ndarray, rate_range=(0.8, 1.2)) -> np.ndarray:
    """Resample sequence to simulate speed changes. Input shape: (T, D)."""
    T, D = features.shape
    rate = random.uniform(*rate_range)
    new_T = max(1, int(round(T * rate)))
    old_indices = np.linspace(0, T - 1, new_T)
    # Linear interpolation along time axis
    old_grid = np.arange(T)
    warped = np.zeros((new_T, D), dtype=np.float32)
    for d in range(D):
        warped[:, d] = np.interp(old_indices, old_grid, features[:, d])
    return warped


def _add_noise(features: np.ndarray, std: float = 0.005) -> np.ndarray:
    """Add Gaussian noise to landmark coordinates (not flags/visibility)."""
    noisy = features.copy()
    # Landmark coordinates: indices 0..224 (99 + 63 + 63 = 225 values)
    # Flags at 225, 226; visibility at 227..259
    noisy[:, :225] += np.random.normal(0, std, noisy[:, :225].shape).astype(np.float32)
    return noisy


def _random_temporal_crop(features: np.ndarray, max_trim_frac: float = 0.10) -> np.ndarray:
    """Randomly trim 0-max_trim_frac from start and end."""
    T = features.shape[0]
    trim_start = random.randint(0, max(0, int(T * max_trim_frac)))
    trim_end = random.randint(0, max(0, int(T * max_trim_frac)))
    end = max(trim_start + 1, T - trim_end)
    return features[trim_start:end]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(data: dict) -> np.ndarray:
    """Extract 260-dim feature vectors from an .npz data dict.

    Returns: (T, 260) float32 array.
    """
    T = data["pose_world"].shape[0]

    pose_world = data["pose_world"].reshape(T, -1)             # (T, 99)
    left_hand_3d = data["left_hand_3d"].reshape(T, -1)         # (T, 63)
    right_hand_3d = data["right_hand_3d"].reshape(T, -1)       # (T, 63)
    left_present = data["left_hand_present"].reshape(T, 1).astype(np.float32)   # (T, 1)
    right_present = data["right_hand_present"].reshape(T, 1).astype(np.float32) # (T, 1)
    pose_vis = data["pose_visibility"]                          # (T, 33)

    features = np.concatenate([
        pose_world,       # 99
        left_hand_3d,     # 63
        right_hand_3d,    # 63
        left_present,     # 1
        right_present,    # 1
        pose_vis,         # 33
    ], axis=1)  # (T, 260)

    return features.astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GestureDataset(Dataset):
    """PyTorch Dataset for gesture sequences.

    Args:
        samples: List of (features_array, label_index) tuples.
        max_seq_len: Sequences are truncated/padded to this length.
        augment: Whether to apply data augmentation.
    """

    def __init__(self, samples: List[Tuple[np.ndarray, int]],
                 max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
                 augment: bool = False):
        self.samples = samples
        self.max_seq_len = max_seq_len
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        features, label = self.samples[idx]
        features = features.copy()

        # Augmentation (on the numpy array before padding)
        if self.augment:
            features = _random_temporal_crop(features)
            features = _time_warp(features)
            features = _add_noise(features)

        T = features.shape[0]

        # Truncate if too long
        if T > self.max_seq_len:
            features = features[:self.max_seq_len]
            T = self.max_seq_len

        # Pad to max_seq_len
        padded = np.zeros((self.max_seq_len, INPUT_DIM), dtype=np.float32)
        padded[:T] = features

        # Attention mask: 1 = real frame, 0 = padding
        mask = np.zeros(self.max_seq_len, dtype=np.float32)
        mask[:T] = 1.0

        return (
            torch.from_numpy(padded),   # (max_seq_len, 260)
            torch.from_numpy(mask),     # (max_seq_len,)
            label,
        )


# ---------------------------------------------------------------------------
# Loading and splitting
# ---------------------------------------------------------------------------

def scan_recordings(recordings_dir: str = RECORDINGS_DIR
                    ) -> Tuple[List[Tuple[np.ndarray, int]], List[str], Dict[str, int]]:
    """Scan recordings directory and load all samples.

    Only loads classes that have at least one .npz file.

    Returns:
        samples: List of (features_array, label_index) tuples
        class_names: Ordered list of class names (index = label)
        label_map: Dict mapping class_name -> label_index
    """
    # Find classes with data
    classes_with_data = []
    if os.path.isfile(CLASSES_JSON):
        with open(CLASSES_JSON, "r") as f:
            all_classes = json.load(f)
    else:
        all_classes = []

    for cls in all_classes:
        cls_dir = os.path.join(recordings_dir, cls)
        if os.path.isdir(cls_dir):
            npz_files = [f for f in os.listdir(cls_dir) if f.endswith(".npz")]
            if npz_files:
                classes_with_data.append(cls)

    class_names = sorted(classes_with_data)
    label_map = {name: i for i, name in enumerate(class_names)}

    samples = []
    for cls in class_names:
        cls_dir = os.path.join(recordings_dir, cls)
        for fname in sorted(os.listdir(cls_dir)):
            if not fname.endswith(".npz"):
                continue
            path = os.path.join(cls_dir, fname)
            data = np.load(path, allow_pickle=True)
            features = extract_features(data)
            label = label_map[cls]
            samples.append((features, label))

    return samples, class_names, label_map


def stratified_split(samples: List[Tuple[np.ndarray, int]],
                     val_fraction: float = 0.2,
                     seed: int = 42
                     ) -> Tuple[List[Tuple[np.ndarray, int]], List[Tuple[np.ndarray, int]]]:
    """Split samples into train/val sets, stratified by class label.

    Returns:
        (train_samples, val_samples)
    """
    rng = random.Random(seed)

    # Group by label
    by_label: Dict[int, List[Tuple[np.ndarray, int]]] = {}
    for sample in samples:
        label = sample[1]
        by_label.setdefault(label, []).append(sample)

    train, val = [], []
    for label in sorted(by_label.keys()):
        group = by_label[label]
        rng.shuffle(group)
        n_val = max(1, int(round(len(group) * val_fraction)))
        val.extend(group[:n_val])
        train.extend(group[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def create_windows(samples: List[Tuple[np.ndarray, int]],
                   window_size: int = WINDOW_SIZE,
                   window_stride: int = WINDOW_STRIDE,
                   ) -> List[Tuple[np.ndarray, int]]:
    """Slice full-sequence samples into overlapping fixed-size windows.

    Each recording is sliced into windows of `window_size` frames with
    `window_stride` step. Recordings shorter than `window_size` are kept
    as-is (they'll be padded by the Dataset).

    Important: call this AFTER stratified_split to avoid data leakage
    (windows from the same recording landing in both train and val).

    Returns:
        List of (window_array, label) tuples.
    """
    windowed = []
    for features, label in samples:
        T = features.shape[0]
        if T <= window_size:
            windowed.append((features, label))
        else:
            for start in range(0, T - window_size + 1, window_stride):
                windowed.append((features[start:start + window_size], label))
            # Include the tail if the last stride didn't reach it
            if (T - window_size) % window_stride != 0:
                windowed.append((features[T - window_size:], label))
    return windowed
