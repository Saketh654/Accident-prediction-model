"""
evaluate_3d_cnn.py

Evaluates the Accident3DCNN model.
Outputs:
  - AUC, Accuracy, Precision, Recall, F1
  - ROC curve saved as roc_3dcnn.png
  - Confusion matrix saved as cm_3dcnn.png

Run from project root:
    python evaluation/evaluate_3d_cnn.py
"""

import sys
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, accuracy_score,
    precision_score, recall_score, f1_score,
    roc_curve, confusion_matrix
)
from tqdm import tqdm

from dataset.dataloader import get_dataloader
from models.accident_3d_cnn import Accident3DCNN

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
LABELS_CSV = "data/processed/labels_enhanced_npz.csv"
THRESHOLD  = 0.5
BATCH_SIZE = 4


def ensure_shape(clips):
    """
    Accident3DCNN expects (B, 3, T, H, W).
    VideoClipDataset should return this, but guard against
    (B, T, 3, H, W) or (B, T, H, W, 3) just in case.
    """
    # clips shape after DataLoader batch: should be (B, C, T, H, W)
    B = clips.shape[0]

    # If channel dim (index 1) is not 3, find where 3 is and permute
    if clips.shape[1] == 3:
        return clips  # already correct (B, 3, T, H, W)
    elif clips.shape[2] == 3:
        # (B, T, 3, H, W) → (B, 3, T, H, W)
        return clips.permute(0, 2, 1, 3, 4).contiguous()
    elif clips.shape[4] == 3:
        # (B, T, H, W, 3) → (B, 3, T, H, W)
        return clips.permute(0, 4, 1, 2, 3).contiguous()
    else:
        raise ValueError(f"Cannot find channel dim of size 3 in shape {clips.shape}")


def main():
    # ── Load model ────────────────────────────────────────────────────────────
    model = Accident3DCNN().to(DEVICE)
    model.load_state_dict(
        torch.load(
            "checkpoints/accident_model.pth",
            map_location=DEVICE,
            weights_only=False
        )
    )
    model.eval()
    print(f"Loaded : checkpoints/accident_model.pth")
    print(f"Device : {DEVICE}\n")

    # ── DataLoader ────────────────────────────────────────────────────────────
    loader = get_dataloader(
        labels_csv=LABELS_CSV,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    print(f"Dataset: {len(loader.dataset)} clips")

    # ── Debug first batch shape ───────────────────────────────────────────────
    sample_clips, _ = next(iter(loader))
    print(f"Raw clip shape from DataLoader: {sample_clips.shape}")
    sample_clips = ensure_shape(sample_clips)
    print(f"Corrected clip shape           : {sample_clips.shape}")
    # Expected: torch.Size([4, 3, 16, 224, 224])

    # ── Inference ─────────────────────────────────────────────────────────────
    all_preds  = []
    all_labels = []

    print("\nRunning inference...")
    with torch.no_grad():
        for clips, labels in tqdm(loader):
            clips  = ensure_shape(clips).to(DEVICE)
            labels = labels.numpy()

            logits = model(clips)
            probs  = torch.sigmoid(logits).squeeze(1).cpu().numpy()

            all_preds.extend(probs.tolist())
            all_labels.extend(labels.tolist())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Binarize soft labels at 0.5
    binary_labels = (all_labels >= 0.5).astype(int)
    binary_preds  = (all_preds  >= THRESHOLD).astype(int)

    # ── Metrics ───────────────────────────────────────────────────────────────
    auc       = roc_auc_score(binary_labels, all_preds)
    acc       = accuracy_score(binary_labels, binary_preds)
    precision = precision_score(binary_labels, binary_preds, zero_division=0)
    recall    = recall_score(binary_labels, binary_preds, zero_division=0)
    f1        = f1_score(binary_labels, binary_preds, zero_division=0)
    cm        = confusion_matrix(binary_labels, binary_preds)

    print("\n── 3D CNN Evaluation Results ───────────────────────")
    print(f"  AUC       : {auc:.4f}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    print("────────────────────────────────────────────────────")

    # ── ROC Curve ─────────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(binary_labels, all_preds)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"3D CNN (AUC={auc:.3f})", color="steelblue", lw=2)
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — 3D CNN")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("roc_3dcnn.png", dpi=150)
    plt.show()
    print("Saved: roc_3dcnn.png")

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Normal", "Crash"],
        yticklabels=["Normal", "Crash"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix — 3D CNN")
    plt.tight_layout()
    plt.savefig("cm_3dcnn.png", dpi=150)
    plt.show()
    print("Saved: cm_3dcnn.png")

    # Save raw predictions for combined comparison
    np.save("preds_3dcnn.npy",  all_preds)
    np.save("labels_gt.npy",    all_labels)
    print("Saved: preds_3dcnn.npy, labels_gt.npy")


if __name__ == "__main__":
    main()