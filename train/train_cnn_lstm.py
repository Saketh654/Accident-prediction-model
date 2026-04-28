"""
train_cnn_lstm.py — CNN+LSTM training with proper train/val split.

Uses:
    data/processed/labels_enhanced_npz_train.csv
    data/processed/labels_enhanced_npz_val.csv
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
from tqdm import tqdm

from dataset.video_clip_dataset import VideoClipDataset
from models.cnn_lstm import CNNLSTM

os.makedirs("checkpoints", exist_ok=True)
torch.backends.cudnn.benchmark = True

# ── Config ─────────────────────────────────────────────────────────────────────
BATCH_SIZE = 16
NUM_EPOCHS = 10
LR         = 3e-5
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_CSV  = "data/processed/labels_enhanced_npz_train.csv"
VAL_CSV    = "data/processed/labels_enhanced_npz_val.csv"
CHECKPOINT = "checkpoints/cnn_lstm_best.pth"


def get_cnn_backbone():
    cnn = models.resnet18(weights="IMAGENET1K_V1")
    feature_dim = cnn.fc.in_features
    cnn.fc = nn.Identity()
    return cnn, feature_dim


def to_cnnlstm_input(clips):
    """(B, C, T, H, W) → (B, T, C, H, W)"""
    if clips.shape[1] == 3:
        return clips.permute(0, 2, 1, 3, 4).contiguous()
    return clips.contiguous()


if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    # ── Data ───────────────────────────────────────────────────────────────────
    train_ds = VideoClipDataset(TRAIN_CSV)
    val_ds   = VideoClipDataset(VAL_CSV)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    print(f"Train clips: {len(train_ds)}  Val clips: {len(val_ds)}")

    # ── Model ──────────────────────────────────────────────────────────────────
    cnn, feature_dim = get_cnn_backbone()
    model = CNNLSTM(cnn=cnn, feature_dim=feature_dim,
                    hidden_dim=256, num_classes=1).to(DEVICE)

    # Freeze early CNN layers, unfreeze last block
    for param in model.cnn.parameters():
        param.requires_grad = False
    for param in model.cnn.layer4.parameters():
        param.requires_grad = True

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR,
    weight_decay=1e-4
)
    scaler    = torch.amp.GradScaler("cuda")

    best_val_loss = float("inf")

    # ── Training loop ───────────────────────────────────────────────────────────
    for epoch in range(NUM_EPOCHS):
        # Train
        model.train()
        train_loss = 0.0
        for clips, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [train]"):
            clips  = to_cnnlstm_input(clips).to(DEVICE)
            labels = labels.float().unsqueeze(1).to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                loss = criterion(model(clips), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for clips, labels in val_loader:
                clips  = to_cnnlstm_input(clips).to(DEVICE)
                labels = labels.float().unsqueeze(1).to(DEVICE)
                with torch.amp.autocast("cuda"):
                    val_loss += criterion(model(clips), labels).item()

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)
        print(f"Epoch {epoch+1:02d} | train={avg_train:.4f}  val={avg_val:.4f}", end="")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), CHECKPOINT)
            print(f"  ✓ saved")
        else:
            print()

    print(f"\nBest val loss: {best_val_loss:.4f}  →  {CHECKPOINT}")
