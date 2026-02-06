import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataset.dataloader import get_dataloader
from models.accident_3d_cnn import Accident3DCNN

# -------------------------
# Configuration
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 5
BATCH_SIZE = 4
LEARNING_RATE = 1e-4

LABELS_CSV = "data/processed/labels_enhanced_npz.csv"

# -------------------------
# DataLoader
# -------------------------
train_loader = get_dataloader(
    labels_csv=LABELS_CSV,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0  # safer on Windows
)

# -------------------------
# Model
# -------------------------
model = Accident3DCNN().to(DEVICE)

# -------------------------
# Class-aware loss
# -------------------------
pos_weight = torch.tensor([3.0]).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# -------------------------
# Optimizer
# -------------------------
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# -------------------------
# Training loop
# -------------------------
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    print(f"\nEpoch [{epoch+1}/{EPOCHS}]")

    for clips, labels in tqdm(train_loader):
        clips = clips.to(DEVICE)
        labels = labels.to(DEVICE).unsqueeze(1)  # (B, 1)

        # Forward
        outputs = model(clips)
        loss = criterion(outputs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Average Loss: {avg_loss:.4f}")

print("\n✅ Training completed")
