"""
evaluation/evaluate_all_models.py

Evaluates ALL six models on the held-out test set.

VideoClipDataset now returns (B, C, T, H, W) after batching.
  - 3D CNN                : uses clips as-is  (B, C, T, H, W)
  - CNN+LSTM              : permutes to        (B, T, C, H, W)
  - CNN+Transformer       : permutes to        (B, T, C, H, W)
  - Two-Stream models     : rgb/flow direct from TwoStreamDataset

Models:
    1. 3D CNN                   checkpoints/accident_model.pth
    2. CNN+LSTM                 checkpoints/cnn_lstm_best.pth
    3. CNN+Transformer          checkpoints/cnn_transformer_best.pth
    4. Two-Stream CNN           checkpoints/two_stream_best.pth
    5. Two-Stream ResNet        checkpoints/two_stream_resnet_final.pth
    6. Two-Stream Transformer   checkpoints/two_stream_transformer_best.pth

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
from dataset.two_stream_resnet_dataloader import get_two_stream_dataloader
from models.accident_3d_cnn import Accident3DCNN
from models.cnn_lstm import CNNLSTM
from models.cnn_transformer import CNNTransformer
from models.two_stream_cnn import TwoStreamCNN
from models.two_stream_resnet import TwoStreamCNNRes
from models.two_stream_transformer import TwoStreamTransformer

# ── Config ─────────────────────────────────────────────────────────────────────
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD  = 0.5
BATCH_SIZE = 4

NPZ_TEST_CSV        = "data/processed/labels_enhanced_npz_test.csv"
TWO_STREAM_TEST_CSV = "data/processed/labels_two_stream_test.csv"

CHECKPOINTS = {
    "3D CNN"                : "checkpoints/accident_model.pth",
    "CNN+LSTM"              : "checkpoints/cnn_lstm_best.pth",
    "CNN+Transformer"       : "checkpoints/cnn_transformer_best.pth",
    "Two-Stream CNN"        : "checkpoints/two_stream_best.pth",
    "Two-Stream ResNet"     : "checkpoints/two_stream_resnet_final.pth",
    "Two-Stream Transformer": "checkpoints/two_stream_transformer_best.pth",
}

COLORS = {
    "3D CNN"                : "steelblue",
    "CNN+LSTM"              : "darkorange",
    "CNN+Transformer"       : "mediumpurple",
    "Two-Stream CNN"        : "green",
    "Two-Stream ResNet"     : "crimson",
    "Two-Stream Transformer": "saddlebrown",
}

CMAPS = {
    "3D CNN"                : "Blues",
    "CNN+LSTM"              : "Oranges",
    "CNN+Transformer"       : "Purples",
    "Two-Stream CNN"        : "Greens",
    "Two-Stream ResNet"     : "Reds",
    "Two-Stream Transformer": "YlOrBr",
}


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(all_labels, all_preds):
    bl  = (np.array(all_labels) >= 0.5).astype(int)
    bp  = (np.array(all_preds)  >= THRESHOLD).astype(int)
    fpr, tpr, _ = roc_curve(bl, all_preds)
    return {
        "AUC"      : roc_auc_score(bl, all_preds),
        "Accuracy" : accuracy_score(bl, bp),
        "Precision": precision_score(bl, bp, zero_division=0),
        "Recall"   : recall_score(bl, bp, zero_division=0),
        "F1"       : f1_score(bl, bp, zero_division=0),
        "CM"       : confusion_matrix(bl, bp),
        "FPR"      : fpr,
        "TPR"      : tpr,
    }


# ── Shared inference loops ─────────────────────────────────────────────────────

def run_npz_inference(model, batch_size=BATCH_SIZE, sequence_input=False):
    """
    sequence_input=False : clips passed as (B, C, T, H, W)  -- 3D CNN
    sequence_input=True  : clips permuted to (B, T, C, H, W) -- LSTM / Transformer
    """
    loader = get_dataloader(NPZ_TEST_CSV, batch_size=batch_size,
                            shuffle=False, num_workers=0)
    print(f"  Test clips : {len(loader.dataset)}")
    preds, labels = [], []
    with torch.no_grad():
        for clips, lbl in tqdm(loader, leave=False):
            if sequence_input:
                clips = clips.permute(0, 2, 1, 3, 4)   # (B,C,T,H,W)->(B,T,C,H,W)
            logits = model(clips.to(DEVICE))
            preds.extend(torch.sigmoid(logits).squeeze(1).cpu().numpy().tolist())
            labels.extend(lbl.numpy().tolist())
    return preds, labels


def run_twostream_inference(model, batch_size=BATCH_SIZE):
    loader = get_two_stream_dataloader(TWO_STREAM_TEST_CSV, batch_size=batch_size,
                                       shuffle=False, num_workers=0)
    print(f"  Test clips : {len(loader.dataset)}")
    preds, labels = [], []
    with torch.no_grad():
        for rgb, flow, lbl in tqdm(loader, leave=False):
            logits = model(rgb.to(DEVICE), flow.to(DEVICE))
            preds.extend(torch.sigmoid(logits).squeeze(1).cpu().numpy().tolist())
            labels.extend(lbl.numpy().tolist())
    return preds, labels


# ── Per-model evaluators ───────────────────────────────────────────────────────

def evaluate_3d_cnn(ckpt):
    model = Accident3DCNN().to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False))
    model.eval()
    preds, labels = run_npz_inference(model, sequence_input=False)
    return compute_metrics(labels, preds)


def evaluate_cnn_lstm(ckpt):
    cnn = models.resnet18(weights=None)
    cnn.fc = nn.Identity()
    model = CNNLSTM(cnn=cnn, feature_dim=512, hidden_dim=256, num_classes=1).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False))
    model.eval()
    preds, labels = run_npz_inference(model, sequence_input=True)
    return compute_metrics(labels, preds)


def evaluate_cnn_transformer(ckpt):
    model = CNNTransformer(
        feature_dim=512, d_model=256, nhead=8,
        num_layers=4, dim_feedforward=512
    ).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False))
    model.eval()
    preds, labels = run_npz_inference(model, sequence_input=True)
    return compute_metrics(labels, preds)


def evaluate_two_stream_cnn(ckpt):
    model = TwoStreamCNN(base_ch=32, fusion="concat").to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False))
    model.eval()
    preds, labels = run_twostream_inference(model, batch_size=6)
    return compute_metrics(labels, preds)


def evaluate_two_stream_resnet(ckpt):
    model = TwoStreamCNNRes(fusion="concat").to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False))
    model.eval()
    preds, labels = run_twostream_inference(model, batch_size=16)
    return compute_metrics(labels, preds)


def evaluate_two_stream_transformer(ckpt):
    model = TwoStreamTransformer(d_model=512, nhead=8, num_layers=2).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False))
    model.eval()
    preds, labels = run_twostream_inference(model, batch_size=8)
    return compute_metrics(labels, preds)


EVALUATORS = {
    "3D CNN"                : evaluate_3d_cnn,
    "CNN+LSTM"              : evaluate_cnn_lstm,
    "CNN+Transformer"       : evaluate_cnn_transformer,
    "Two-Stream CNN"        : evaluate_two_stream_cnn,
    "Two-Stream ResNet"     : evaluate_two_stream_resnet,
    "Two-Stream Transformer": evaluate_two_stream_transformer,
}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"Device : {DEVICE}")
    print("Evaluating ALL models on held-out TEST SET\n")

    results = {}
    skipped = []

    for idx, (name, ckpt) in enumerate(CHECKPOINTS.items(), 1):
        print(f"[{idx}/{len(CHECKPOINTS)}] {name}")
        if not os.path.exists(ckpt):
            print(f"  Checkpoint not found: {ckpt} -- skipping\n")
            skipped.append(name)
            continue
        try:
            results[name] = EVALUATORS[name](ckpt)
            print()
        except Exception as e:
            print(f"  ERROR: {e} -- skipping\n")
            skipped.append(name)

    if not results:
        print("No models could be evaluated. Train at least one model first.")
        return

    if skipped:
        print(f"Skipped: {', '.join(skipped)}\n")

    # Metrics table
    names   = list(results.keys())
    col_w   = 16
    divider = "=" * (12 + col_w * len(names))

    print(f"\n{divider}")
    print("  MODEL COMPARISON — TEST SET")
    print(divider)
    print(f"{'Metric':<12}" + "".join(f"{n:>{col_w}}" for n in names))
    print("-" * (12 + col_w * len(names)))

    for metric in ["AUC", "Accuracy", "Precision", "Recall", "F1"]:
        vals = [results[n][metric] for n in names]
        best = max(vals)
        row  = f"{metric:<12}"
        for v in vals:
            cell = f"{v:.4f}{'*' if v == best else ' '}"
            row += f"{cell:>{col_w}}"
        print(row)

    print(divider)
    print("* = best for that metric\n")

    # Save CSV
    rows = [{"Model": n,
             **{k: round(results[n][k], 4) for k in ["AUC","Accuracy","Precision","Recall","F1"]}}
            for n in names]
    pd.DataFrame(rows).to_csv("metrics_summary_test.csv", index=False)
    print("Saved: metrics_summary_test.csv")

    # ROC curve
    plt.figure(figsize=(9, 7))
    for name, m in results.items():
        plt.plot(m["FPR"], m["TPR"],
                 label=f"{name} (AUC={m['AUC']:.3f})",
                 color=COLORS[name], lw=2)
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — All Models (Test Set)")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("roc_comparison_test.png", dpi=150)
    plt.close()
    print("Saved: roc_comparison_test.png")

    # Confusion matrices
    n     = len(results)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for ax, (name, m) in zip(axes, results.items()):
        sns.heatmap(m["CM"], annot=True, fmt="d", cmap=CMAPS[name],
                    xticklabels=["Normal", "Crash"],
                    yticklabels=["Normal", "Crash"], ax=ax)
        ax.set_title(f"{name}\nF1={m['F1']:.3f}  AUC={m['AUC']:.3f}", fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    for ax in axes[len(results):]:
        ax.set_visible(False)

    plt.suptitle("Confusion Matrices — All Models (Test Set)", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig("cm_comparison_test.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: cm_comparison_test.png")
    print("\nAll done.")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
