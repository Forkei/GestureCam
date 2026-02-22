"""Training script for the multi-label gesture Transformer classifier.

Usage:
    cd client
    python train_gestures.py

Trains on recorded .npz data in client/recordings/, saves best model to
client/models/gesture_transformer.pt and config to client/models/gesture_config.json.

Uses BCEWithLogitsLoss for multi-label classification with independent sigmoid
outputs per gesture. Early stopping is based on average validation F1 across
all gestures.

Requirements: torch, numpy
"""

import os
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from gesture_dataset import (
    GestureDataset, scan_recordings, chunk_split_and_window,
    INPUT_DIM, DEFAULT_MAX_SEQ_LEN, WINDOW_SIZE, WINDOW_STRIDE,
    GESTURE_LABELS, NUM_GESTURES,
)
from gesture_model import GestureTransformer

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
MAX_SEQ_LEN = DEFAULT_MAX_SEQ_LEN   # 30
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-4
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 3
DIM_FEEDFORWARD = 256
DROPOUT = 0.35
EARLY_STOP_PATIENCE = 20
VAL_FRACTION = 0.2
SEED = 42
THRESHOLD = 0.5

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(_SCRIPT_DIR, "models")


MAX_POS_WEIGHT = 20.0

def compute_pos_weights(label_vectors: np.ndarray) -> torch.Tensor:
    """Compute per-gesture pos_weight for BCEWithLogitsLoss.

    pos_weight[g] = num_negative / num_positive for gesture g,
    capped at MAX_POS_WEIGHT to prevent extreme overcompensation
    for very rare gestures.

    Args:
        label_vectors: (N, num_gestures) float32 array of multi-hot labels.

    Returns:
        Tensor of shape (num_gestures,).
    """
    num_samples = label_vectors.shape[0]
    pos_counts = label_vectors.sum(axis=0)  # (num_gestures,)
    neg_counts = num_samples - pos_counts

    # Avoid division by zero: if a gesture has no positive samples, use weight 1.0
    pos_counts = np.maximum(pos_counts, 1.0)
    weights = neg_counts / pos_counts

    # Cap to prevent extreme weights for rare classes
    weights = np.minimum(weights, MAX_POS_WEIGHT)

    return torch.from_numpy(weights.astype(np.float32))


def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for features, mask, labels in loader:
        features = features.to(device)
        mask = mask.to(device)
        labels = labels.to(device)  # (B, num_gestures) float32

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(features, mask)  # (B, num_gestures)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)

    return total_loss / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model and collect all predictions, probabilities, and labels.

    Returns:
        avg_loss: Scalar average loss.
        all_probs: (N, num_gestures) sigmoid probabilities.
        all_labels: (N, num_gestures) ground truth labels.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_probs = []
    all_labels = []

    for features, mask, labels in loader:
        features = features.to(device)
        mask = mask.to(device)
        labels = labels.to(device)

        logits = model(features, mask)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)

        probs = torch.sigmoid(logits)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)    # (N, num_gestures)
    all_labels = np.concatenate(all_labels, axis=0)   # (N, num_gestures)

    return total_loss / total_samples, all_probs, all_labels


def optimize_thresholds(all_probs: np.ndarray, all_labels: np.ndarray) -> np.ndarray:
    """Find the optimal threshold per gesture that maximizes F1.

    Sweeps thresholds from 0.3 to 0.9 in steps of 0.05 for each gesture
    independently and picks the one with the highest F1 score.

    Returns:
        Array of shape (num_gestures,) with optimal thresholds.
    """
    num_g = all_labels.shape[1]
    thresholds = np.full(num_g, 0.5)
    candidates = np.arange(0.30, 0.91, 0.05)

    for g in range(num_g):
        best_f1 = -1.0
        best_t = 0.5
        for t in candidates:
            preds = (all_probs[:, g] > t).astype(float)
            tp = ((preds == 1) & (all_labels[:, g] == 1)).sum()
            fp = ((preds == 1) & (all_labels[:, g] == 0)).sum()
            fn = ((preds == 0) & (all_labels[:, g] == 1)).sum()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        thresholds[g] = best_t

    return thresholds


def compute_multilabel_metrics(all_labels: np.ndarray, all_preds: np.ndarray):
    """Compute per-gesture and aggregate multi-label metrics.

    Args:
        all_labels: (N, num_gestures) ground truth binary labels.
        all_preds: (N, num_gestures) binary predictions.

    Returns:
        per_gesture: List of dicts with precision, recall, f1, support per gesture.
        exact_match_acc: Fraction of samples where ALL gestures match exactly.
        hamming_acc: Mean per-label accuracy across all gestures and samples.
        avg_f1: Mean F1 across all gestures.
    """
    N = all_labels.shape[0]
    num_g = all_labels.shape[1]

    per_gesture = []
    f1_sum = 0.0

    for g in range(num_g):
        tp = ((all_preds[:, g] == 1) & (all_labels[:, g] == 1)).sum()
        fp = ((all_preds[:, g] == 1) & (all_labels[:, g] == 0)).sum()
        fn = ((all_preds[:, g] == 0) & (all_labels[:, g] == 1)).sum()
        support = int(all_labels[:, g].sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_gesture.append({
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        })
        f1_sum += f1

    # Exact match: all gestures predicted correctly for a sample
    exact_match = (all_preds == all_labels).all(axis=1).mean()

    # Hamming accuracy: fraction of correct individual predictions
    hamming_acc = (all_preds == all_labels).mean()

    avg_f1 = f1_sum / num_g

    return per_gesture, exact_match, hamming_acc, avg_f1


def print_metrics(all_labels: np.ndarray, all_preds: np.ndarray):
    """Print a clean per-gesture metrics table and aggregate scores."""
    per_gesture, exact_match, hamming_acc, avg_f1 = compute_multilabel_metrics(
        all_labels, all_preds
    )

    print(f"\n{'Gesture':<16s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'Support':>8s}")
    print("-" * 44)
    for i, name in enumerate(GESTURE_LABELS):
        m = per_gesture[i]
        print(f"{name:<16s} {m['precision']:>6.3f} {m['recall']:>6.3f} "
              f"{m['f1']:>6.3f} {m['support']:>8d}")

    print("-" * 44)
    print(f"{'Avg F1':<16s} {'':>6s} {'':>6s} {avg_f1:>6.3f}")
    print(f"\nExact match accuracy: {exact_match:.1%}")
    print(f"Hamming accuracy:     {hamming_acc:.1%}")


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    print("Loading recordings...")
    samples = scan_recordings()
    print(f"Found {len(samples)} samples, {NUM_GESTURES} gesture labels: {GESTURE_LABELS}")

    if len(samples) == 0:
        print("No data found. Record some gestures first!")
        return

    # Per-gesture distribution (by recording)
    all_label_vecs = np.array([s[1] for s in samples])  # (N, num_gestures)
    print("\nPer-gesture sample counts (recordings):")
    for i, name in enumerate(GESTURE_LABELS):
        count = int(all_label_vecs[:, i].sum())
        print(f"  {name}: {count}")
    idle_count = int((all_label_vecs.sum(axis=1) == 0).sum())
    print(f"  (idle/no gesture): {idle_count}")

    # Chunk-split each recording into train/val by time, then window
    train_windows, val_windows = chunk_split_and_window(
        samples, VAL_FRACTION, WINDOW_SIZE, WINDOW_STRIDE, SEED
    )
    print(f"\nWindowed: {len(train_windows)} train / {len(val_windows)} val windows "
          f"(size={WINDOW_SIZE}, stride={WINDOW_STRIDE})")

    # Datasets
    train_ds = GestureDataset(train_windows, MAX_SEQ_LEN, augment=True)
    val_ds = GestureDataset(val_windows, MAX_SEQ_LEN, augment=False)

    use_pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=False, pin_memory=use_pin)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            pin_memory=use_pin)

    # Pos weights for BCEWithLogitsLoss (from windowed training samples)
    train_label_vecs = np.array([s[1] for s in train_windows])  # (N_train, num_gestures)
    pos_weights = compute_pos_weights(train_label_vecs).to(device)
    print(f"\nPos weights per gesture:")
    for i, name in enumerate(GESTURE_LABELS):
        print(f"  {name}: {pos_weights[i].item():.2f}")

    # Model
    model = GestureTransformer(
        input_dim=INPUT_DIM,
        num_gestures=NUM_GESTURES,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        max_seq_len=MAX_SEQ_LEN,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # Loss, optimizer, scheduler, AMP scaler
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # Training loop -- early stopping on average validation F1
    best_val_f1 = 0.0
    epochs_no_improve = 0
    best_state = None

    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"\nTraining for up to {EPOCHS} epochs (early stop patience: {EARLY_STOP_PATIENCE})...\n")
    print(f"{'Epoch':>5s}  {'TrLoss':>7s}  {'VlLoss':>7s} {'VlF1':>6s} {'ExMat':>6s} {'HamAc':>6s}  {'LR':>8s}")
    print("-" * 58)

    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_probs, val_labels = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        val_preds = (val_probs > THRESHOLD).astype(float)
        _, exact_match, hamming_acc, avg_f1 = compute_multilabel_metrics(val_labels, val_preds)

        lr = optimizer.param_groups[0]["lr"]
        marker = ""
        if avg_f1 > best_val_f1:
            best_val_f1 = avg_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            marker = " *"
        else:
            epochs_no_improve += 1

        print(f"{epoch:>5d}  {train_loss:>7.4f}  {val_loss:>7.4f} {avg_f1:>6.3f} "
              f"{exact_match:>6.1%} {hamming_acc:>6.1%}  {lr:>8.6f}{marker}")

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (no val F1 improvement for "
                  f"{EARLY_STOP_PATIENCE} epochs)")
            break

    elapsed = time.time() - t0
    print(f"\nTraining completed in {elapsed:.1f}s")
    print(f"Best val avg F1: {best_val_f1:.3f}")

    # Save best model and optimize thresholds
    if best_state is not None:
        model_path = os.path.join(MODELS_DIR, "gesture_transformer.pt")
        torch.save(best_state, model_path)
        print(f"Saved model to {model_path}")

        # Final evaluation with best model
        model.load_state_dict(best_state)
        model.to(device)
        _, final_probs, final_labels = evaluate(model, val_loader, criterion, device)

        # Optimize per-gesture thresholds on validation set
        per_thresholds = optimize_thresholds(final_probs, final_labels)
        print(f"\nOptimized per-gesture thresholds:")
        for i, name in enumerate(GESTURE_LABELS):
            print(f"  {name}: {per_thresholds[i]:.2f}")

        # Show metrics with optimized thresholds
        final_preds = np.zeros_like(final_probs)
        for g in range(NUM_GESTURES):
            final_preds[:, g] = (final_probs[:, g] > per_thresholds[g]).astype(float)
        print_metrics(final_labels, final_preds)

        # Save config with per-gesture thresholds
        config = {
            "gesture_labels": GESTURE_LABELS,
            "num_gestures": NUM_GESTURES,
            "input_dim": INPUT_DIM,
            "max_seq_len": MAX_SEQ_LEN,
            "d_model": D_MODEL,
            "nhead": NHEAD,
            "num_layers": NUM_LAYERS,
            "dim_feedforward": DIM_FEEDFORWARD,
            "dropout": DROPOUT,
            "threshold": THRESHOLD,
            "per_gesture_thresholds": {
                name: float(per_thresholds[i])
                for i, name in enumerate(GESTURE_LABELS)
            },
        }
        config_path = os.path.join(MODELS_DIR, "gesture_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Saved config to {config_path}")


if __name__ == "__main__":
    main()
