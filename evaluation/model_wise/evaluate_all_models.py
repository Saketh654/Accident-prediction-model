"""
evaluate_all_models.py

Runs evaluation for all three models and produces a combined comparison.

Outputs:
  - Printed metrics table
  - roc_comparison.png
  - cm_comparison.png
  - metrics_summary.csv

Run from project root:
    python evaluation/evaluate_all_models.py
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
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, accuracy_score,
    precision_score, recall_score, f1_score,
    roc_curve, confusion_matrix
)
from tqdm import tqdm

from dataset.dataloader import get_dataloader
from dataset.two_stream_dataloader import get_two_stream_dataloader
from models.accident_3d_cnn import Accident3DCNN
from models.cnn_lstm import CNNLSTM
from models.two_stream_cnn import TwoStreamCNN

# ─────────────────────────────────────────────
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD  = 0.5
BATCH_SIZE = 4

LABELS_CSV_NPZ        = "data/processed/labels_enhanced_npz.csv"
LABELS_CSV_TWO_STREAM = "data/processed/labels_two_stream.csv"


# ─────────────────────────────────────────────
# Shape utilities
# ─────────────────────────────────────────────
def to_3dcnn_input(clips):
    """
    Accident3DCNN expects (B, 3, T, H, W).
    Handles whatever shape VideoClipDataset returns after batching.
    """
    s = clips.shape
    if s[1] == 3:
        return clips.contiguous()                        # (B, 3, T, H, W) ✓
    elif s[2] == 3:
        return clips.permute(0, 2, 1, 3, 4).contiguous() # (B, T, 3, H, W) → fix
    elif s[4] == 3:
        return clips.permute(0, 4, 1, 2, 3).contiguous() # (B, T, H, W, 3) → fix
    else:
        raise ValueError(f"Cannot find C=3 in shape {s}")


def to_cnnlstm_input(clips):
    """
    CNNLSTM expects (B, T, 3, H, W).
    """
    s = clips.shape
    if s[1] == 3:
        return clips.permute(0, 2, 1, 3, 4).contiguous() # (B, 3, T, H, W) → fix
    elif s[2] == 3:
        return clips.contiguous()                         # (B, T, 3, H, W) ✓
    elif s[4] == 3:
        return clips.permute(0, 1, 4, 2, 3).contiguous() # (B, T, H, W, 3) → fix
    else:
        raise ValueError(f"Cannot find C=3 in shape {s}")


# ─────────────────────────────────────────────
# Metric helper
# ─────────────────────────────────────────────
def compute_metrics(all_labels, all_preds):
    binary_labels = (np.array(all_labels) >= 0.5).astype(int)
    binary_preds  = (np.array(all_preds)  >= THRESHOLD).astype(int)
    fpr, tpr, _   = roc_curve(binary_labels, all_preds)
    return {
        "AUC"      : roc_auc_score(binary_labels, all_preds),
        "Accuracy" : accuracy_score(binary_labels, binary_preds),
        "Precision": precision_score(binary_labels, binary_preds, zero_division=0),
        "Recall"   : recall_score(binary_labels, binary_preds, zero_division=0),
        "F1"       : f1_score(binary_labels, binary_preds, zero_division=0),
        "CM"       : confusion_matrix(binary_labels, binary_preds),
        "FPR"      : fpr,
        "TPR"      : tpr,
    }


# ─────────────────────────────────────────────
# Evaluate 3D CNN
# ─────────────────────────────────────────────
def evaluate_3d_cnn():
    print("\n[1/3] Evaluating 3D CNN...")
    model = Accident3DCNN().to(DEVICE)
    model.load_state_dict(torch.load(
        "checkpoints/accident_model.pth",
        map_location=DEVICE, weights_only=False
    ))
    model.eval()

    loader = get_dataloader(LABELS_CSV_NPZ, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0)

    # Print shape debug once
    sample, _ = next(iter(loader))
    print(f"  Raw shape : {sample.shape} → 3DCNN input: {to_3dcnn_input(sample).shape}")

    preds, labels = [], []
    with torch.no_grad():
        for clips, lbl in tqdm(loader, leave=False):
            clips = to_3dcnn_input(clips).to(DEVICE)
            logits = model(clips)
            preds.extend(torch.sigmoid(logits).squeeze(1).cpu().numpy().tolist())
            labels.extend(lbl.numpy().tolist())

    return compute_metrics(labels, preds), np.array(preds), np.array(labels)


# ─────────────────────────────────────────────
# Evaluate CNN+LSTM
# ─────────────────────────────────────────────
def evaluate_cnn_lstm():
    print("\n[2/3] Evaluating CNN+LSTM...")
    cnn = models.resnet18(weights=None)
    cnn.fc = nn.Identity()
    feature_dim = 512

    model = CNNLSTM(cnn=cnn, feature_dim=feature_dim,
                    hidden_dim=256, num_classes=1).to(DEVICE)
    model.load_state_dict(torch.load(
        "checkpoints/cnn_lstm_best.pth",
        map_location=DEVICE, weights_only=False
    ))
    model.eval()

    loader = get_dataloader(LABELS_CSV_NPZ, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0)

    sample, _ = next(iter(loader))
    print(f"  Raw shape : {sample.shape} → CNNLSTM input: {to_cnnlstm_input(sample).shape}")

    preds, labels = [], []
    with torch.no_grad():
        for clips, lbl in tqdm(loader, leave=False):
            clips = to_cnnlstm_input(clips).to(DEVICE)
            logits = model(clips)
            preds.extend(torch.sigmoid(logits).squeeze(1).cpu().numpy().tolist())
            labels.extend(lbl.numpy().tolist())

    return compute_metrics(labels, preds), np.array(preds), np.array(labels)


# ─────────────────────────────────────────────
# Evaluate Two-Stream
# ─────────────────────────────────────────────
def evaluate_two_stream():
    print("\n[3/3] Evaluating Two-Stream...")
    model = TwoStreamCNN(base_ch=32, fusion="concat").to(DEVICE)
    model.load_state_dict(torch.load(
        "checkpoints/two_stream_best.pth",
        map_location=DEVICE, weights_only=False
    ))
    model.eval()

    loader = get_two_stream_dataloader(
        LABELS_CSV_TWO_STREAM, batch_size=6,
        shuffle=False, num_workers=8
    )

    preds, labels = [], []
    with torch.no_grad():
        for rgb, flow, lbl in tqdm(loader, leave=False):
            logits = model(rgb.to(DEVICE), flow.to(DEVICE))
            preds.extend(torch.sigmoid(logits).squeeze(1).cpu().numpy().tolist())
            labels.extend(lbl.numpy().tolist())

    return compute_metrics(labels, preds), np.array(preds), np.array(labels)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")

    m3d,   p3d,   l3d   = evaluate_3d_cnn()
    mlstm, plstm, _     = evaluate_cnn_lstm()
    mts,   pts,   _     = evaluate_two_stream()

    results = {
        "3D CNN"     : m3d,
        "CNN+LSTM"   : mlstm,
        "Two-Stream" : mts,
    }

    # ── Print metrics table ───────────────────────────────────────────────────
    print("\n\n══════════════════════════════════════════════════════")
    print("          MODEL COMPARISON — ALL METRICS              ")
    print("══════════════════════════════════════════════════════")
    print(f"{'Metric':<12} {'3D CNN':>10} {'CNN+LSTM':>10} {'Two-Stream':>12}")
    print("──────────────────────────────────────────────────────")
    for metric in ["AUC", "Accuracy", "Precision", "Recall", "F1"]:
        vals = [results[m][metric] for m in ["3D CNN", "CNN+LSTM", "Two-Stream"]]
        best = max(vals)
        row  = f"{metric:<12}"
        for v in vals:
            marker = " ◀" if v == best else "  "
            row += f" {v:>9.4f}{marker}"
        print(row)
    print("══════════════════════════════════════════════════════")
    print("◀ = best value for that metric\n")

    # ── Save metrics CSV ──────────────────────────────────────────────────────
    rows = []
    for name, m in results.items():
        rows.append({
            "Model"    : name,
            "AUC"      : round(m["AUC"],       4),
            "Accuracy" : round(m["Accuracy"],  4),
            "Precision": round(m["Precision"], 4),
            "Recall"   : round(m["Recall"],    4),
            "F1"       : round(m["F1"],        4),
        })
    pd.DataFrame(rows).to_csv("metrics_summary.csv", index=False)
    print("Saved: metrics_summary.csv")

    # ── Combined ROC Curve ────────────────────────────────────────────────────
    colors = {"3D CNN": "steelblue", "CNN+LSTM": "darkorange", "Two-Stream": "green"}
    plt.figure(figsize=(8, 6))
    for name, m in results.items():
        plt.plot(m["FPR"], m["TPR"],
                 label=f"{name} (AUC={m['AUC']:.3f})",
                 color=colors[name], lw=2)
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — All Models")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("roc_comparison.png", dpi=150)
    plt.close()
    print("Saved: roc_comparison.png")

    # ── Side-by-side Confusion Matrices ───────────────────────────────────────
    cmaps = {"3D CNN": "Blues", "CNN+LSTM": "Oranges", "Two-Stream": "Greens"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (name, m) in zip(axes, results.items()):
        sns.heatmap(
            m["CM"], annot=True, fmt="d", cmap=cmaps[name],
            xticklabels=["Normal", "Crash"],
            yticklabels=["Normal", "Crash"],
            ax=ax
        )
        ax.set_title(f"{name}\nF1={m['F1']:.3f}  AUC={m['AUC']:.3f}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.suptitle("Confusion Matrices — All Models", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig("cm_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: cm_comparison.png")
    print("\nAll done.")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    try:
        main()
    except Exception:
        import traceback
        print("\n[ERROR]")
        traceback.print_exc()