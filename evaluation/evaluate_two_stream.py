"""
evaluate_two_stream.py

Evaluates the Two-Stream model and compares against your existing models.
Outputs:
  - AUC, accuracy, precision, recall, F1
  - Side-by-side risk curve plots
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, accuracy_score,
    precision_score, recall_score, f1_score,
    roc_curve
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

# ─────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────
model = TwoStreamCNN(base_ch=32, fusion="concat").to(DEVICE)
model.load_state_dict(
    torch.load("checkpoints/two_stream_best.pth", map_location=DEVICE)
)
model.eval()

# ─────────────────────────────────────────────
# DataLoader (no shuffle for evaluation)
# ─────────────────────────────────────────────
loader = get_two_stream_dataloader(
    labels_csv=LABELS_CSV,
    batch_size=4,
    shuffle=False,
    num_workers=0
)

# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────
all_preds  = []
all_labels = []

print("Running evaluation...")
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

# Binarize labels at 0.5 (soft labels → hard)
binary_labels = (all_labels >= 0.5).astype(int)
binary_preds  = (all_preds  >= THRESHOLD).astype(int)

# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────
auc       = roc_auc_score(binary_labels, all_preds)
acc       = accuracy_score(binary_labels, binary_preds)
precision = precision_score(binary_labels, binary_preds, zero_division=0)
recall    = recall_score(binary_labels, binary_preds, zero_division=0)
f1        = f1_score(binary_labels, binary_preds, zero_division=0)

print("\n── Two-Stream Evaluation Results ──────────────────")
print(f"  AUC       : {auc:.4f}")
print(f"  Accuracy  : {acc:.4f}")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1 Score  : {f1:.4f}")
print("────────────────────────────────────────────────────")

# ─────────────────────────────────────────────
# ROC Curve
# ─────────────────────────────────────────────
fpr, tpr, _ = roc_curve(binary_labels, all_preds)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"Two-Stream (AUC={auc:.3f})", color="royalblue", lw=2)
plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Two-Stream Model")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_two_stream.png", dpi=150)
plt.show()
print("ROC curve saved: roc_two_stream.png")

# ─────────────────────────────────────────────
# Risk curve comparison (requires saved .npy files)
# ─────────────────────────────────────────────
FPS       = 10
THRESHOLD_LINE = 0.5

risk_files = {
    "3D CNN"       : "risk_normal.npy",
    "CNN+LSTM"     : "risk_cnn_lstm.npy",
    "Two-Stream"   : "risk_two_stream.npy",
}

plt.figure(figsize=(10, 5))
colors = ["steelblue", "darkorange", "green"]

for (name, path), color in zip(risk_files.items(), colors):
    if os.path.exists(path):
        risk = np.load(path)
        time = [i / FPS for i in range(len(risk))]
        plt.plot(time, risk, label=name, color=color, lw=2)

plt.axhline(y=THRESHOLD_LINE, linestyle="--", color="red", label="Threshold")
plt.xlabel("Time (seconds)")
plt.ylabel("Risk Score")
plt.title("Risk Score Comparison — All Models")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("risk_comparison.png", dpi=150)
plt.show()
print("Comparison plot saved: risk_comparison.png")
