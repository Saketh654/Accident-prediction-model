import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
from tqdm import tqdm

from dataset.video_clip_dataset import VideoClipDataset
from models.cnn_lstm import CNNLSTM
import os


os.makedirs("checkpoints", exist_ok=True)
torch.backends.cudnn.benchmark = True
# --------------------------------------------------
# CONFIG
# --------------------------------------------------
BATCH_SIZE = 4
NUM_EPOCHS = 5
LR = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------
# CNN BACKBONE
# --------------------------------------------------
def get_cnn_backbone():
    cnn = models.resnet18(pretrained=True)
    feature_dim = cnn.fc.in_features
    cnn.fc = nn.Identity()
    return cnn, feature_dim


# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":

    print("Training on:", DEVICE)

    # Dataset
    train_dataset = VideoClipDataset(
        "data/processed/labels_enhanced_npz.csv"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Model
    cnn, feature_dim = get_cnn_backbone()

    model = CNNLSTM(
        cnn=cnn,
        feature_dim=feature_dim,
        hidden_dim=256,
        num_classes=1
    ).to(DEVICE)

    # --------------------------------------------------
    # FREEZE EARLY CNN LAYERS
    # --------------------------------------------------
    for param in model.cnn.parameters():
        param.requires_grad = False

    # 🔥 UNFREEZE LAST BLOCK (IMPORTANT)
    for param in model.cnn.layer4.parameters():
        param.requires_grad = True

    # Loss & Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR
    )

    best_loss = float("inf")

    # --------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------
    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(NUM_EPOCHS):

        model.train()
        running_loss = 0.0

        progress = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}]")

        for clips, labels in progress:

            clips = clips.to(DEVICE)
            labels = labels.float().unsqueeze(1).to(DEVICE)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):

                outputs = model(clips)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            progress.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} Avg Loss: {epoch_loss:.4f}")

        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(
                model.state_dict(),
                "checkpoints/cnn_lstm_best.pth"
            )
            print("✅ Best model saved")
