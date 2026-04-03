"""
evaluate_two_stream.py

Evaluates the Two-Stream model.
Outputs:
  - AUC, Accuracy, Precision, Recall, F1
  - ROC curve saved as roc_two_stream.png
  - Confusion matrix saved as cm_two_stream.png
  - Risk score comparison plot saved as risk_comparison.png

Run from project root:
    python evaluation/evaluate_two_stream.py
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

from dataset.two_stream_dataloader import get_two_stream_dataloader
from models.two_stream_cnn import TwoStreamCNN

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
LABELS_CSV = "data/processed/labels_two_stream.csv"
THRESHOLD  = 0.5
BATCH_SIZE = 6


def main():
    # ── Load model ────────────────────────────────────────────────────────────
    model = TwoStreamCNN(base_ch=32, fusion="concat").to(DEVICE)
    model.load_state_dict(
        torch.load("checkpoints/two_stream_best.pth", map_location=DEVICE)
    )
    model.eval()
    print(f"Loaded: checkpoints/two_stream_best.pth")
    print(f"Device: {DEVICE}\n")

    # ── DataLoader ────────────────────────────────────────────────────────────
    loader = get_two_stream_dataloader(
        labels_csv=LABELS_CSV,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8
    )
    print(f"Dataset: {len(loader.dataset)} clips")

    # ── Inference ─────────────────────────────────────────────────────────────
    all_preds  = []
    all_labels = []

    print("Running inference...")
    with torch.no_grad():
        for rgb, flow, labels in tqdm(loader):
            rgb    = rgb.to(DEVICE)
            flow   = flow.to(DEVICE)
            labels = labels.numpy()

            logits = model(rgb, flow)
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

    print("\n── Two-Stream Evaluation Results ───────────────────")
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
    plt.plot(fpr, tpr, label=f"Two-Stream (AUC={auc:.3f})", color="green", lw=2)
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Two-Stream")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("roc_two_stream.png", dpi=150)
    plt.show()
    print("Saved: roc_two_stream.png")

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Greens",
        xticklabels=["Normal", "Crash"],
        yticklabels=["Normal", "Crash"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix — Two-Stream")
    plt.tight_layout()
    plt.savefig("cm_two_stream.png", dpi=150)
    plt.show()
    print("Saved: cm_two_stream.png")

    # Save raw predictions
    np.save("preds_two_stream.npy", all_preds)
    print("Saved: preds_two_stream.npy")

    # ── Risk curve comparison ──────────────────────────────────────────────────
    FPS = 10
    risk_files = {
        "3D CNN"     : ("risk_normal.npy",      "steelblue"),
        "CNN+LSTM"   : ("risk_cnn_lstm.npy",    "darkorange"),
        "Two-Stream" : ("risk_two_stream.npy",  "green"),
    }

    plt.figure(figsize=(10, 5))
    for name, (path, color) in risk_files.items():
        if os.path.exists(path):
            risk = np.load(path)
            time = [i / FPS for i in range(len(risk))]
            plt.plot(time, risk, label=name, color=color, lw=2)

    plt.axhline(y=THRESHOLD, linestyle="--", color="red", label="Threshold")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Risk Score")
    plt.title("Risk Score Comparison — All Models")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("risk_comparison.png", dpi=150)
    plt.show()
    print("Saved: risk_comparison.png")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
