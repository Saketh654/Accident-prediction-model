"""
train_two_stream_resnet.py — Two-Stream ResNet training with train/val split.

Optimized for: Ryzen 5 5600H (6C/12T) + RTX 3050 4 GB VRAM + 16 GB RAM

Key changes from baseline:
    BATCH_SIZE   8  → 4    4 GB VRAM can't fit two ResNet18 streams at B=8/16
                           comfortably; B=4 leaves ~0.5 GB headroom for AMP
    GRAD_ACCUM   4         accumulates 4 × B=4 steps → effective batch = 16,
                           matching original training dynamics without OOM
    NUM_WORKERS  6  → 4    5600H sweet spot; 6 workers over-subscribes RAM
    AMP                    float16 forward/backward halves VRAM usage
    grad clip              already present — kept at 1.0
    empty_cache            called once per epoch to release CUDA fragmentation
"""

import os
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from dataset.two_stream_resnet_dataloader import get_two_stream_dataloader
from models.two_stream_resnet import TwoStreamCNNRes

# ── Hardware-tuned config ──────────────────────────────────────────────────────
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS       = 10
BATCH_SIZE   = 4    # RTX 3050 4 GB: safe ceiling for two-stream ResNet18 + AMP
GRAD_ACCUM   = 4    # effective batch = BATCH_SIZE × GRAD_ACCUM = 16
LR           = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS  = 4    # Ryzen 5 5600H: 4 workers, leaves cores free for GPU transfers

# ── Paths ──────────────────────────────────────────────────────────────────────
TRAIN_CSV      = "data/processed/labels_two_stream_train.csv"
VAL_CSV        = "data/processed/labels_two_stream_val.csv"
CHECKPOINT_DIR = "checkpoints"
BEST_CKPT      = os.path.join(CHECKPOINT_DIR, "two_stream_resnet_best.pth")
FINAL_CKPT     = os.path.join(CHECKPOINT_DIR, "two_stream_resnet_final.pth")

# ── Early stopping ─────────────────────────────────────────────────────────────
EARLY_STOP_PATIENCE  = 3
EARLY_STOP_MIN_DELTA = 1e-4


class EarlyStopping:
    """Stops training when val loss stops improving."""

    def __init__(self, patience: int = 3, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best_loss = float("inf")
        self.stop      = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # benchmark=True profiles convolution algos on first batch — worthwhile
    # because RTX 3050 Ampere has a fixed input size (224×224)
    torch.backends.cudnn.benchmark = True

    loader_kw = dict(
        num_workers        = NUM_WORKERS,
        pin_memory         = True,   # DMA to GPU without extra copy
        persistent_workers = True,   # avoids ~8 s worker respawn per epoch
        prefetch_factor    = 2,
    )
    train_loader = get_two_stream_dataloader(TRAIN_CSV, batch_size=BATCH_SIZE,
                                             shuffle=True,  **loader_kw)
    val_loader   = get_two_stream_dataloader(VAL_CSV,   batch_size=BATCH_SIZE,
                                             shuffle=False, **loader_kw)

    print(f"Device      : {DEVICE}")
    if DEVICE == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU         : {props.name}  ({props.total_memory // 1024**2} MB VRAM)")
    print(f"Train clips : {len(train_loader.dataset)}")
    print(f"Val clips   : {len(val_loader.dataset)}")
    print(f"Batch size  : {BATCH_SIZE}  (effective {BATCH_SIZE * GRAD_ACCUM} with grad accum)")

    model      = TwoStreamCNNRes(fusion="concat").to(DEVICE)
    pos_weight = torch.tensor([3.0]).to(DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # GradScaler: AMP keeps activations in float16, weights in float32
    # — halves VRAM usage on the forward pass (critical for 4 GB)
    scaler        = GradScaler("cuda")
    early_stopper = EarlyStopping(patience=EARLY_STOP_PATIENCE,
                                  min_delta=EARLY_STOP_MIN_DELTA)

    best_val_loss = float("inf")
    loss_history  = []

    for epoch in range(EPOCHS):
        print(f"\nEpoch [{epoch+1}/{EPOCHS}]  LR={scheduler.get_last_lr()[0]:.2e}")

        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        optimizer.zero_grad(set_to_none=True)   # reset once before accum loop

        for step, (rgb, flow, labels) in enumerate(
                tqdm(train_loader, desc="  train")):

            rgb    = rgb.to(DEVICE, non_blocking=True)
            flow   = flow.to(DEVICE, non_blocking=True)
            labels = labels.float().to(DEVICE, non_blocking=True).unsqueeze(1)

            with autocast(DEVICE, enabled=(DEVICE == "cuda")):
                # Scale loss by 1/GRAD_ACCUM so gradients are averaged,
                # not summed, across the accumulation window
                loss = criterion(model(rgb, flow), labels) / GRAD_ACCUM

            scaler.scale(loss).backward()

            # Only update weights every GRAD_ACCUM steps (or at end of epoch)
            if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item() * GRAD_ACCUM   # undo the /GRAD_ACCUM for logging

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for rgb, flow, labels in tqdm(val_loader, desc="  val  ", leave=False):
                rgb    = rgb.to(DEVICE, non_blocking=True)
                flow   = flow.to(DEVICE, non_blocking=True)
                labels = labels.float().to(DEVICE, non_blocking=True).unsqueeze(1)
                with autocast(DEVICE, enabled=(DEVICE == "cuda")):
                    val_loss += criterion(model(rgb, flow), labels).item()

        # Free CUDA memory fragments accumulated during the epoch
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        scheduler.step()
        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)
        loss_history.append((avg_train, avg_val))

        if DEVICE == "cuda":
            used_mb = torch.cuda.memory_reserved() / 1024**2
            print(f"  train={avg_train:.4f}  val={avg_val:.4f}  "
                  f"VRAM={used_mb:.0f} MB", end="")
        else:
            print(f"  train={avg_train:.4f}  val={avg_val:.4f}", end="")

        # ── Checkpoint ────────────────────────────────────────────────────────
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), BEST_CKPT)
            print("  -> saved best", end="")

        # ── Early stopping ────────────────────────────────────────────────────
        if early_stopper.step(avg_val):
            print(f"\n  Early stopping — no improvement for "
                  f"{EARLY_STOP_PATIENCE} consecutive epochs.")
            break

        print(f"  [patience {early_stopper.counter}/{EARLY_STOP_PATIENCE}]")

    torch.save(model.state_dict(), FINAL_CKPT)
    np.save("two_stream_resnet_loss_history.npy", np.array(loss_history))
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best checkpoint : {BEST_CKPT}")