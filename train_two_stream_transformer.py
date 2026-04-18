import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from dataset.two_stream_resnet_dataloader import get_two_stream_dataloader
from models.two_stream_transformer import TwoStreamTransformer

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS       = 10
BATCH_SIZE   = 8    # lower than ResNet version due to attention memory overhead
LR           = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS  = 6
LABELS_CSV   = "data/processed/labels_two_stream.csv"
CHECKPOINT   = "checkpoints/two_stream_transformer_best.pth"

os.makedirs("checkpoints", exist_ok=True)
torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    loader = get_two_stream_dataloader(
        labels_csv=LABELS_CSV,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    model = TwoStreamTransformer(
        d_model=512, nhead=8, num_layers=2, dropout=0.1
    ).to(DEVICE)

    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    pos_weight = torch.tensor([3.0]).to(DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )
    scaler = GradScaler("cuda")
    best_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for rgb, flow, labels in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            rgb    = rgb.to(DEVICE, non_blocking=True)
            flow   = flow.to(DEVICE, non_blocking=True)
            labels = labels.float().unsqueeze(1).to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda"):
                logits = model(rgb, flow)
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
            print(f"  Saved → {CHECKPOINT}")