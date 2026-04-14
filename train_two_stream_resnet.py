"""
train_two_stream.py  —  optimized for i7-12650H + RTX 4060 Laptop (8 GB VRAM)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from dataset.two_stream_dataloader import get_two_stream_dataloader
from models.two_stream_resnet import TwoStreamCNNRes

# ── Hyperparameters ────────────────────────────────────────────────────────────
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS         = 10
BATCH_SIZE     = 16          # raise to 32 if nvidia-smi shows < 6.5 GB used
LEARNING_RATE  = 1e-4
WEIGHT_DECAY   = 1e-4
NUM_WORKERS    = 6           # sweet spot for 10-core i7-12650H
LABELS_CSV     = "data/processed/labels_two_stream.csv"
CHECKPOINT_DIR = "checkpoints"
# ──────────────────────────────────────────────────────────────────────────────


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    """Run a single training epoch, return average loss."""
    model.train()
    running_loss = 0.0

    for rgb, flow, labels in tqdm(loader, leave=False):
        rgb    = rgb.to(device, non_blocking=True)
        flow   = flow.to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True).unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device, enabled=(device == "cuda")):
            logits = model(rgb, flow)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    return running_loss / len(loader)


def save_checkpoint(model, path):
    """Save model state dict to path."""
    torch.save(model.state_dict(), path)
    print(f"  Checkpoint saved → {path}")


def print_vram_summary(device):
    """Print a brief VRAM usage summary (useful for tuning batch size)."""
    if device == "cuda":
        allocated = torch.cuda.memory_allocated(device) / 1024 ** 3
        reserved  = torch.cuda.memory_reserved(device)  / 1024 ** 3
        print(f"  VRAM  allocated: {allocated:.2f} GB  |  reserved: {reserved:.2f} GB")


if __name__ == "__main__":

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ── cuDNN auto-tuner: big win for fixed input sizes ────────────────────────
    torch.backends.cudnn.benchmark = True

    # ── DataLoader ─────────────────────────────────────────────────────────────
    print("Loading dataset...")
    train_loader = get_two_stream_dataloader(
        labels_csv=LABELS_CSV,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,          # faster CPU → GPU transfer
        prefetch_factor=2,        # prefetch 2 batches per worker
        persistent_workers=True   # keep workers alive between epochs
    )
    print(f"  {len(train_loader.dataset)} clips | {len(train_loader)} batches/epoch")

    # ── Model ──────────────────────────────────────────────────────────────────
    model = TwoStreamCNNRes(fusion="concat").to(DEVICE)

    print(f"\nModel on : {DEVICE}")
    print(f"Params   : {sum(p.numel() for p in model.parameters()):,}")

    # ── Loss, optimiser, scheduler ─────────────────────────────────────────────
    pos_weight = torch.tensor([3.0]).to(DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    scaler = GradScaler("cuda")

    # ── Training loop ──────────────────────────────────────────────────────────
    best_loss    = float("inf")
    loss_history = []

    print("\nStarting training...\n")

    for epoch in range(EPOCHS):
        print(f"Epoch [{epoch+1}/{EPOCHS}]  LR={scheduler.get_last_lr()[0]:.2e}")

        avg_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, DEVICE
        )
        loss_history.append(avg_loss)
        scheduler.step()

        print(f"  Avg Loss : {avg_loss:.4f}")

        # Print VRAM on first epoch so you can decide whether to raise BATCH_SIZE
        if epoch == 0:
            print_vram_summary(DEVICE)

        # Save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(
                model,
                os.path.join(CHECKPOINT_DIR, "two_stream_best.pth")
            )

    # ── Final save ─────────────────────────────────────────────────────────────
    save_checkpoint(
        model,
        os.path.join(CHECKPOINT_DIR, "two_stream_final.pth")
    )
    np.save("two_stream_loss_history.npy", np.array(loss_history))

    print("\nTraining complete.")
    print(f"  Best loss  : {best_loss:.4f}")
    print(f"  Checkpoint : {CHECKPOINT_DIR}/two_stream_best.pth")
