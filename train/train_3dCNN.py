"""
train.py — 3D CNN training with train/val split + full speed optimisations.

Uses:
    data/processed/labels_enhanced_npz_train.csv
    data/processed/labels_enhanced_npz_val.csv
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from dataset.dataloader import get_dataloader
from models.accident_3d_cnn import Accident3DCNN

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS     = 10
BATCH_SIZE = 4
LR         = 1e-4

TRAIN_CSV  = "data/processed/labels_enhanced_npz_train.csv"
VAL_CSV    = "data/processed/labels_enhanced_npz_val.csv"
CHECKPOINT = "checkpoints/accident_model.pth"


if __name__ == "__main__":
    # Windows requires all multiprocessing code inside this guard.
    # Without it, each worker tries to re-run the whole script on spawn.
    import multiprocessing
    multiprocessing.freeze_support()

    os.makedirs("checkpoints", exist_ok=True)
    torch.backends.cudnn.benchmark = True

    train_loader = get_dataloader(TRAIN_CSV, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = get_dataloader(VAL_CSV,   batch_size=BATCH_SIZE, shuffle=False)

    print(f"Device      : {DEVICE}")
    print(f"Train clips : {len(train_loader.dataset)}")
    print(f"Val   clips : {len(val_loader.dataset)}")

    model      = Accident3DCNN().to(DEVICE)
    pos_weight = torch.tensor([3.0]).to(DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = optim.Adam(model.parameters(), lr=LR)
    scaler     = GradScaler("cuda")

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0.0
        for clips, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [train]"):
            clips  = clips.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True).unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda"):
                loss = criterion(model(clips), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for clips, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [val]  ", leave=False):
                clips  = clips.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True).unsqueeze(1)
                with autocast("cuda"):
                    val_loss += criterion(model(clips), labels).item()

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)
        print(f"Epoch {epoch+1:02d} | train={avg_train:.4f}  val={avg_val:.4f}", end="")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), CHECKPOINT)
            print(f"  -> saved best")
        else:
            print()

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {CHECKPOINT}")