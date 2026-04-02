"""
train_two_stream.py
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
LEARNING_RATE  = 1e-4
WEIGHT_DECAY   = 1e-4
NUM_WORKERS    = 8
LABELS_CSV     = "data/processed/labels_two_stream.csv"
CHECKPOINT_DIR = "checkpoints"


if __name__ == "__main__":

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("Loading dataset...")
    train_loader = get_two_stream_dataloader(
        labels_csv=LABELS_CSV,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )
    print(f"  {len(train_loader.dataset)} clips | {len(train_loader)} batches/epoch")

    model = TwoStreamCNN(base_ch=32, fusion="concat").to(DEVICE)
    print(f"\nModel on : {DEVICE}")
    print(f"Params   : {sum(p.numel() for p in model.parameters()):,}")

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

    best_loss    = float("inf")
    loss_history = []

    print("\nStarting training...\n")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        print(f"Epoch [{epoch+1}/{EPOCHS}]  LR={scheduler.get_last_lr()[0]:.2e}")

        for rgb, flow, labels in tqdm(train_loader, leave=False):
            rgb    = rgb.to(DEVICE, non_blocking=True)
            flow   = flow.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True).unsqueeze(1)

            optimizer.zero_grad()

            with autocast("cuda"):
                logits = model(rgb, flow)
                loss   = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        loss_history.append(avg_loss)
        scheduler.step()

        print(f"  -> Avg Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = os.path.join(CHECKPOINT_DIR, "two_stream_best.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Best model saved ({avg_loss:.4f})")

    torch.save(
        model.state_dict(),
        os.path.join(CHECKPOINT_DIR, "two_stream_final.pth")
    )
    np.save("two_stream_loss_history.npy", np.array(loss_history))

    print("\nTraining complete.")
    print(f"   Best loss  : {best_loss:.4f}")
    print(f"   Checkpoint : checkpoints/two_stream_best.pth")
