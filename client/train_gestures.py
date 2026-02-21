"""Training script for the gesture Transformer classifier.

Usage:
    cd client
    python train_gestures.py

Trains on recorded .npz data in client/recordings/, saves best model to
client/models/gesture_transformer.pt and config to client/models/gesture_config.json.

Requirements: torch, numpy, scikit-learn (pip install scikit-learn)
"""

import os
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from gesture_dataset import (
    GestureDataset, scan_recordings, stratified_split, create_windows,
    INPUT_DIM, DEFAULT_MAX_SEQ_LEN, WINDOW_SIZE, WINDOW_STRIDE,
)
from gesture_model import GestureTransformer

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
MAX_SEQ_LEN = DEFAULT_MAX_SEQ_LEN   # 30
BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-4
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 3
DIM_FEEDFORWARD = 256
DROPOUT = 0.3
EARLY_STOP_PATIENCE = 20
VAL_FRACTION = 0.2
SEED = 42

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(_SCRIPT_DIR, "models")


def compute_class_weights(labels: list, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights for CrossEntropyLoss."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    # Avoid division by zero for classes with no samples
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes  # normalize so mean weight = 1
    return torch.from_numpy(weights)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for features, mask, labels in loader:
        features = features.to(device)
        mask = mask.to(device)
        labels = labels.clone().detach().to(dtype=torch.long, device=device)

        optimizer.zero_grad()
        logits = model(features, mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for features, mask, labels in loader:
        features = features.to(device)
        mask = mask.to(device)
        labels = labels.clone().detach().to(dtype=torch.long, device=device)

        logits = model(features, mask)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return total_loss / total, correct / total, all_preds, all_labels


def print_metrics(all_labels, all_preds, class_names):
    """Print per-class precision/recall/F1 and confusion matrix."""
    num_classes = len(class_names)

    # Confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(all_labels, all_preds):
        cm[true][pred] += 1

    # Per-class metrics
    print(f"\n{'Class':<16s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'Support':>8s}")
    print("-" * 44)
    for i, name in enumerate(class_names):
        tp = cm[i][i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = cm[i, :].sum()
        print(f"{name:<16s} {precision:>6.3f} {recall:>6.3f} {f1:>6.3f} {support:>8d}")

    print(f"\nConfusion Matrix (rows=true, cols=predicted):")
    # Header
    header = f"{'':>16s} " + " ".join(f"{n[:8]:>8s}" for n in class_names)
    print(header)
    for i, name in enumerate(class_names):
        row = f"{name:<16s} " + " ".join(f"{cm[i][j]:>8d}" for j in range(num_classes))
        print(row)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    print("Loading recordings...")
    samples, class_names, label_map = scan_recordings()
    num_classes = len(class_names)
    print(f"Found {len(samples)} samples across {num_classes} classes: {class_names}")

    # Class distribution (by recording)
    label_counts = {}
    for _, label in samples:
        label_counts[label] = label_counts.get(label, 0) + 1
    for name in class_names:
        print(f"  {name}: {label_counts.get(label_map[name], 0)} recordings")

    if len(samples) == 0:
        print("No data found. Record some gestures first!")
        return

    # Split at recording level first (prevents data leakage), then window
    train_recordings, val_recordings = stratified_split(samples, VAL_FRACTION, SEED)
    print(f"Split: {len(train_recordings)} train / {len(val_recordings)} val recordings")

    train_windows = create_windows(train_recordings, WINDOW_SIZE, WINDOW_STRIDE)
    val_windows = create_windows(val_recordings, WINDOW_SIZE, WINDOW_STRIDE)
    print(f"Windowed: {len(train_windows)} train / {len(val_windows)} val windows "
          f"(size={WINDOW_SIZE}, stride={WINDOW_STRIDE})")

    # Datasets
    train_ds = GestureDataset(train_windows, MAX_SEQ_LEN, augment=True)
    val_ds = GestureDataset(val_windows, MAX_SEQ_LEN, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Class weights (from windowed samples)
    train_labels = [s[1] for s in train_windows]
    class_weights = compute_class_weights(train_labels, num_classes).to(device)
    print(f"Class weights: {class_weights.cpu().numpy()}")

    # Model
    model = GestureTransformer(
        input_dim=INPUT_DIM,
        num_classes=num_classes,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        max_seq_len=MAX_SEQ_LEN,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Training loop
    best_val_acc = 0.0
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = None

    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"\nTraining for up to {EPOCHS} epochs (early stop patience: {EARLY_STOP_PATIENCE})...\n")
    print(f"{'Epoch':>5s}  {'TrLoss':>7s} {'TrAcc':>6s}  {'VlLoss':>7s} {'VlAcc':>6s}  {'LR':>8s}")
    print("-" * 50)

    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            marker = " *"

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(f"{epoch:>5d}  {train_loss:>7.4f} {train_acc:>6.1%}  {val_loss:>7.4f} {val_acc:>6.1%}  {lr:>8.6f}{marker}")

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (no val loss improvement for {EARLY_STOP_PATIENCE} epochs)")
            break

    elapsed = time.time() - t0
    print(f"\nTraining completed in {elapsed:.1f}s")
    print(f"Best val accuracy: {best_val_acc:.1%}")

    # Save best model
    if best_state is not None:
        model_path = os.path.join(MODELS_DIR, "gesture_transformer.pt")
        torch.save(best_state, model_path)
        print(f"Saved model to {model_path}")

        # Save config
        config = {
            "class_names": class_names,
            "label_map": label_map,
            "input_dim": INPUT_DIM,
            "max_seq_len": MAX_SEQ_LEN,
            "num_classes": num_classes,
            "d_model": D_MODEL,
            "nhead": NHEAD,
            "num_layers": NUM_LAYERS,
            "dim_feedforward": DIM_FEEDFORWARD,
            "dropout": DROPOUT,
        }
        config_path = os.path.join(MODELS_DIR, "gesture_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Saved config to {config_path}")

    # Final evaluation with best model
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
        _, _, final_preds, final_labels = evaluate(model, val_loader, criterion, device)
        print_metrics(final_labels, final_preds, class_names)


if __name__ == "__main__":
    main()
