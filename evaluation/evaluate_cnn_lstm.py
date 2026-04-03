"""
evaluate_cnn_lstm.py

Run from project root:
    python evaluation/evaluate_cnn_lstm.py
"""

import sys
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, accuracy_score,
    precision_score, recall_score, f1_score,
    roc_curve, confusion_matrix
)
from tqdm import tqdm

from dataset.dataloader import get_dataloader
from models.cnn_lstm import CNNLSTM

# ─────────────────────────────────────────────
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
LABELS_CSV = "data/processed/labels_enhanced_npz.csv"
THRESHOLD  = 0.5
BATCH_SIZE = 4


def get_cnn_backbone():
    cnn = models.resnet18(weights=None)
    feature_dim = cnn.fc.in_features
    cnn.fc = nn.Identity()
    return cnn, feature_dim


def to_cnnlstm_input(clips):
    """
    Converts whatever shape the DataLoader returns into (B, T, C, H, W)
    which is what CNNLSTM.forward() expects.

    VideoClipDataset does: (T,H,W,C) -> permute(3,0,1,2) -> (C,T,H,W)
    After DataLoader batch:  (B, C, T, H, W)
    CNNLSTM needs:           (B, T, C, H, W)

    We detect shape and normalise accordingly.
    """
    s = clips.shape  # (B, ?, ?, ?, ?)
    B = s[0]

    if s[1] == 3:
        # (B, C, T, H, W) — what VideoClipDataset actually returns after batch
        # permute to (B, T, C, H, W)
        return clips.permute(0, 2, 1, 3, 4).contiguous()

    elif s[1] == 16 and s[2] == 3:
        # (B, T, C, H, W) — already correct
        return clips.contiguous()

    elif s[1] == 16 and s[2] == 224:
        # (B, T, H, W, C) — permute to (B, T, C, H, W)
        return clips.permute(0, 1, 4, 2, 3).contiguous()

    else:
        raise ValueError(
            f"Cannot interpret clip shape {s} for CNNLSTM. "
            f"Expected one of: (B,3,T,H,W), (B,T,3,H,W), (B,T,H,W,3)"
        )


def main():
    print("=" * 52)
    print("  CNN+LSTM Evaluation")
    print("=" * 52)

    # ── Load model ────────────────────────────────────────────────────────────
    print("Loading model...")
    cnn, feature_dim = get_cnn_backbone()
    model = CNNLSTM(
        cnn=cnn, feature_dim=feature_dim, hidden_dim=256, num_classes=1
    ).to(DEVICE)
    model.load_state_dict(
        torch.load("checkpoints/cnn_lstm_best.pth",
                   map_location=DEVICE, weights_only=False)
    )
    model.eval()
    print(f"Loaded : checkpoints/cnn_lstm_best.pth")
    print(f"Device : {DEVICE}")

    # ── DataLoader ────────────────────────────────────────────────────────────
    print("Loading dataset...")
    loader = get_dataloader(
        labels_csv=LABELS_CSV,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    print(f"Dataset: {len(loader.dataset)} clips")

    # ── Debug shape on first batch ────────────────────────────────────────────
    sample_clips, _ = next(iter(loader))
    print(f"Raw clip shape  : {sample_clips.shape}")
    converted = to_cnnlstm_input(sample_clips)
    print(f"CNNLSTM input   : {converted.shape}  (expected [B, T=16, C=3, 224, 224])")

    # ── Inference ─────────────────────────────────────────────────────────────
    all_preds  = []
    all_labels = []

    print("\nRunning inference...")
    with torch.no_grad():
        for clips, labels in tqdm(loader):
            clips  = to_cnnlstm_input(clips).to(DEVICE)
            labels = labels.numpy()

            logits = model(clips)
            probs  = torch.sigmoid(logits).squeeze(1).cpu().numpy()

            all_preds.extend(probs.tolist())
            all_labels.extend(labels.tolist())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    binary_labels = (all_labels >= 0.5).astype(int)
    binary_preds  = (all_preds  >= THRESHOLD).astype(int)

    # ── Metrics ───────────────────────────────────────────────────────────────
    auc       = roc_auc_score(binary_labels, all_preds)
    acc       = accuracy_score(binary_labels, binary_preds)
    precision = precision_score(binary_labels, binary_preds, zero_division=0)
    recall    = recall_score(binary_labels, binary_preds, zero_division=0)
    f1        = f1_score(binary_labels, binary_preds, zero_division=0)
    cm        = confusion_matrix(binary_labels, binary_preds)

    print("\n── CNN+LSTM Evaluation Results ─────────────────────")
    print(f"  AUC       : {auc:.4f}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")
    print("────────────────────────────────────────────────────")

    # ── ROC Curve ─────────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(binary_labels, all_preds)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"CNN+LSTM (AUC={auc:.3f})", color="darkorange", lw=2)
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — CNN+LSTM")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("roc_cnn_lstm.png", dpi=150)
    plt.close()
    print("Saved: roc_cnn_lstm.png")

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Oranges",
        xticklabels=["Normal", "Crash"],
        yticklabels=["Normal", "Crash"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix — CNN+LSTM")
    plt.tight_layout()
    plt.savefig("cm_cnn_lstm.png", dpi=150)
    plt.close()
    print("Saved: cm_cnn_lstm.png")

    np.save("preds_cnn_lstm.npy", all_preds)
    print("Saved: preds_cnn_lstm.npy")
    print("\nEvaluation complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("\n[ERROR] Script failed:")
        traceback.print_exc()