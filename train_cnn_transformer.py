# train_cnn_transformer.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from dataset.dataloader import get_dataloader
from models.cnn_transformer import CNNTransformer

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS        = 10
BATCH_SIZE    = 4
LR            = 1e-4
WEIGHT_DECAY  = 1e-4
LABELS_CSV    = "data/processed/labels_enhanced_npz.csv"
CHECKPOINT    = "checkpoints/cnn_transformer_best.pth"

os.makedirs("checkpoints", exist_ok=True)
torch.backends.cudnn.benchmark = True

def to_transformer_input(clips):
    """
    DataLoader returns (B, C, T, H, W).
    CNNTransformer expects  (B, T, C, H, W).
    """
    if clips.shape[1] == 3:
        return clips.permute(0, 2, 1, 3, 4).contiguous()
    return clips.contiguous()

if __name__ == "__main__":
    loader = get_dataloader(
        labels_csv=LABELS_CSV,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    model = CNNTransformer(
        feature_dim=512, d_model=256, nhead=8,
        num_layers=4, dim_feedforward=512
    ).to(DEVICE)

    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    pos_weight = torch.tensor([3.0]).to(DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler     = GradScaler("cuda")

    best_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for clips, labels in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            clips  = to_transformer_input(clips).to(DEVICE)
            labels = labels.float().unsqueeze(1).to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda"):
                logits = model(clips)
                loss   = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        avg = running_loss / len(loader)
        scheduler.step()
        print(f"  Avg loss: {avg:.4f}")

        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), CHECKPOINT)
            print(f"  Saved best → {CHECKPOINT}")