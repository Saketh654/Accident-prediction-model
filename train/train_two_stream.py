"""
train_two_stream.py — Two-Stream CNN training with train/val split.

Uses:
    data/processed/labels_two_stream_train.csv
    data/processed/labels_two_stream_val.csv
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from dataset.two_stream_dataloader import get_two_stream_dataloader
from models.two_stream_cnn import TwoStreamCNN

DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS         = 10
BATCH_SIZE     = 6
LR             = 1e-4
WEIGHT_DECAY   = 1e-4
NUM_WORKERS    = 6

TRAIN_CSV      = "data/processed/labels_two_stream_train.csv"
VAL_CSV        = "data/processed/labels_two_stream_val.csv"
CHECKPOINT_DIR = "checkpoints"
BEST_CKPT      = os.path.join(CHECKPOINT_DIR, "two_stream_best.pth")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.backends.cudnn.benchmark = True

    loader_kw = dict(num_workers=NUM_WORKERS, pin_memory=True,
                     persistent_workers=True, prefetch_factor=2)
    train_loader = get_two_stream_dataloader(TRAIN_CSV, batch_size=BATCH_SIZE,
                                             shuffle=True,  **loader_kw)
    val_loader   = get_two_stream_dataloader(VAL_CSV,   batch_size=BATCH_SIZE,
                                             shuffle=False, **loader_kw)

    print(f"Device: {DEVICE}")
    print(f"Train: {len(train_loader.dataset)} clips | Val: {len(val_loader.dataset)} clips")

    model      = TwoStreamCNN(base_ch=32, fusion="concat").to(DEVICE)
    pos_weight = torch.tensor([3.0]).to(DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler     = GradScaler("cuda")

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        print(f"Epoch [{epoch+1}/{EPOCHS}]  LR={scheduler.get_last_lr()[0]:.2e}")

        # Train
        model.train()
        train_loss = 0.0
        for rgb, flow, labels in tqdm(train_loader, desc="  train"):
            rgb    = rgb.to(DEVICE, non_blocking=True)
            flow   = flow.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True).unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda"):
                loss = criterion(model(rgb, flow), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for rgb, flow, labels in tqdm(val_loader, desc="  val  ", leave=False):
                rgb    = rgb.to(DEVICE, non_blocking=True)
                flow   = flow.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True).unsqueeze(1)
                with autocast("cuda"):
                    val_loss += criterion(model(rgb, flow), labels).item()

        scheduler.step()
        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)
        print(f"  train={avg_train:.4f}  val={avg_val:.4f}", end="")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), BEST_CKPT)
            print(f"  -> saved best")
        else:
            print()

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {BEST_CKPT}")