"""
train_cnn_transformer.py — CNN+Transformer training with train/val split.

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
from models.cnn_transformer import CNNTransformer

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS       = 10
BATCH_SIZE   = 4
LR           = 1e-4
WEIGHT_DECAY = 1e-4

TRAIN_CSV  = "data/processed/labels_enhanced_npz_train.csv"
VAL_CSV    = "data/processed/labels_enhanced_npz_val.csv"
CHECKPOINT = "checkpoints/cnn_transformer_best.pth"


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    os.makedirs("checkpoints", exist_ok=True)
    torch.backends.cudnn.benchmark = True

    train_loader = get_dataloader(TRAIN_CSV, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = get_dataloader(VAL_CSV,   batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train: {len(train_loader.dataset)} clips | Val: {len(val_loader.dataset)} clips")

    model = CNNTransformer(
        feature_dim=512, d_model=256, nhead=8,
        num_layers=4, dim_feedforward=512
    ).to(DEVICE)

    print(f"Device: {DEVICE} | Params: {sum(p.numel() for p in model.parameters()):,}")

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
        for clips, labels in tqdm(train_loader, desc="  train"):
            # (B, C, T, H, W) -> (B, T, C, H, W) for Transformer
            clips  = clips.permute(0, 2, 1, 3, 4).to(DEVICE, non_blocking=True)
            labels = labels.float().unsqueeze(1).to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda"):
                loss = criterion(model(clips), labels)
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
            for clips, labels in tqdm(val_loader, desc="  val  ", leave=False):
                clips  = clips.permute(0, 2, 1, 3, 4).to(DEVICE, non_blocking=True)
                labels = labels.float().unsqueeze(1).to(DEVICE, non_blocking=True)
                with autocast("cuda"):
                    val_loss += criterion(model(clips), labels).item()

        scheduler.step()
        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)
        print(f"  train={avg_train:.4f}  val={avg_val:.4f}", end="")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), CHECKPOINT)
            print(f"  -> saved best")
        else:
            print()

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {CHECKPOINT}")
