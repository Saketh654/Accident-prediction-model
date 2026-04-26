"""
evaluate_pr_curves.py
─────────────────────
Generates Precision-Recall curves for all 6 models:
    1. Accident3DCNN          (3D CNN baseline)
    2. CNNLSTM                (CNN + LSTM)
    3. CNNTransformer          (CNN + Transformer)
    4. TwoStreamCNN           (Two-Stream 3D CNN)
    5. TwoStreamCNNRes        (Two-Stream ResNet)
    6. TwoStreamTransformer   (Two-Stream Transformer)

Outputs:
    pr_curves_all_models.png  — overlaid PR curves for all models
    pr_curves_individual/     — one PNG per model (for appendix)

Usage:
    python evaluate_pr_curves.py \
        --val-csv   data/processed/labels_two_stream_val.csv \
        --npz-csv   data/processed/labels_enhanced_npz_val.csv \
        --device    cuda

Checkpoint paths (edit below or pass via --ckpt-* flags):
    checkpoints/accident_model.pth
    checkpoints/cnn_lstm_best.pth
    checkpoints/cnn_transformer_best.pth
    checkpoints/two_stream_best.pth
    checkpoints/two_stream_resnet_best.pth
    checkpoints/two_stream_transformer_best.pth
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

# ── Project imports ────────────────────────────────────────────────────────────
from dataset.video_clip_dataset import VideoClipDataset
from dataset.two_stream_resnet_dataset import TwoStreamDataset

from models.accident_3d_cnn import Accident3DCNN
from models.cnn_lstm import CNNLSTM
from models.cnn_transformer import CNNTransformer
from models.two_stream_cnn import TwoStreamCNN
from models.two_stream_resnet import TwoStreamCNNRes
from models.two_stream_transformer import TwoStreamTransformer


# ══════════════════════════════════════════════════════════════════════════════
# Default checkpoint paths (override via CLI flags)
# ══════════════════════════════════════════════════════════════════════════════
CKPT_DEFAULTS = {
    "3DCNN":               "checkpoints/accident_model.pth",
    "CNN+LSTM":            "checkpoints/cnn_lstm_best.pth",
    "CNN+Transformer":     "checkpoints/cnn_transformer_best.pth",
    "TwoStream-CNN":       "checkpoints/two_stream_best.pth",
    "TwoStream-ResNet":    "checkpoints/two_stream_resnet_final.pth",
    "TwoStream-Transformer": "checkpoints/two_stream_transformer_final.pth",
}

# Colour palette — distinct, print-safe
COLORS = {
    "3DCNN":               "#E63946",   # red
    "CNN+LSTM":            "#F4A261",   # orange
    "CNN+Transformer":     "#2A9D8F",   # teal
    "TwoStream-CNN":       "#457B9D",   # steel blue
    "TwoStream-ResNet":    "#6A0572",   # purple
    "TwoStream-Transformer": "#1D3557", # navy
}

LINESTYLES = {
    "3DCNN":               "-",
    "CNN+LSTM":            "--",
    "CNN+Transformer":     "-.",
    "TwoStream-CNN":       ":",
    "TwoStream-ResNet":    (0, (5, 1)),
    "TwoStream-Transformer": "-",
}


# ══════════════════════════════════════════════════════════════════════════════
# Argument parser
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="PR-curve evaluation for all 6 models")
    p.add_argument("--val-csv",   default="data/processed/labels_two_stream_val.csv",
                   help="CSV for two-stream models (rgb_dir, flow_dir, start, end, label)")
    p.add_argument("--npz-csv",   default="data/processed/labels_enhanced_npz_val.csv",
                   help="CSV for single-stream models (clip_path, label)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", default="pr_curve_outputs")
    # Individual checkpoint overrides
    for key in CKPT_DEFAULTS:
        flag = "--ckpt-" + key.lower().replace("+", "-").replace(" ", "-")
        p.add_argument(flag, default=CKPT_DEFAULTS[key])
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Model builders
# ══════════════════════════════════════════════════════════════════════════════
def build_3dcnn():
    return Accident3DCNN()


def build_cnn_lstm():
    backbone = models.resnet18(weights=None)
    feature_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    return CNNLSTM(cnn=backbone, feature_dim=feature_dim,
                   hidden_dim=256, num_classes=1)


def build_cnn_transformer():
    return CNNTransformer(
        feature_dim=512, d_model=512, nhead=8,
        num_layers=8, dim_feedforward=2048, dropout=0.1
    )


def build_two_stream_cnn():
    return TwoStreamCNN(base_ch=32, fusion="concat")


def build_two_stream_resnet():
    return TwoStreamCNNRes(fusion="concat")


def build_two_stream_transformer():
    return TwoStreamTransformer(
        d_model=256, nhead=8, num_layers=4,
        dim_ff=512, fusion="concat", grad_ckpt=False
    )


# ══════════════════════════════════════════════════════════════════════════════
# Inference helpers
# ══════════════════════════════════════════════════════════════════════════════
def to_cnnlstm_input(clips):
    """(B, C, T, H, W) → (B, T, C, H, W)"""
    return clips.permute(0, 2, 1, 3, 4).contiguous()


@torch.no_grad()
def collect_preds_single_stream(model, loader, device, model_name):
    """Collect logits + labels for single-stream models (clip, label)."""
    model.eval()
    all_probs, all_labels = [], []

    for clips, labels in tqdm(loader, desc=f"  Evaluating {model_name}"):
        clips = clips.to(device, non_blocking=True)

        if model_name == "CNN+LSTM":
            clips = to_cnnlstm_input(clips)
        elif model_name == "CNN+Transformer":
            clips = clips.permute(0, 2, 1, 3, 4).contiguous()

        logits = model(clips)                          # (B, 1)
        probs  = torch.sigmoid(logits).squeeze(1)     # (B,)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_probs), np.concatenate(all_labels)


@torch.no_grad()
def collect_preds_two_stream(model, loader, device, model_name):
    """Collect logits + labels for two-stream models (rgb, flow, label)."""
    model.eval()
    all_probs, all_labels = [], []

    for rgb, flow, labels in tqdm(loader, desc=f"  Evaluating {model_name}"):
        rgb  = rgb.to(device, non_blocking=True)
        flow = flow.to(device, non_blocking=True)

        logits = model(rgb, flow)                      # (B, 1)
        probs  = torch.sigmoid(logits).squeeze(1)     # (B,)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_probs), np.concatenate(all_labels)


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════
def plot_combined(results: dict, out_path: str, baseline_ap: float = None):
    """
    Overlaid PR curves for all models.
    `results` = { model_name: {"precision", "recall", "ap", "auc_roc"} }
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#0F0F1A")
    ax.set_facecolor("#0F0F1A")

    # Sort by AP descending so best model legend entry is on top
    sorted_models = sorted(results.items(), key=lambda x: x[1]["ap"], reverse=True)

    for name, res in sorted_models:
        ax.plot(
            res["recall"], res["precision"],
            color=COLORS[name],
            linestyle=LINESTYLES[name],
            linewidth=2.2,
            label=f"{name}  (AP={res['ap']:.3f}  AUC-ROC={res['auc_roc']:.3f})",
            alpha=0.92,
        )

    # Random-classifier baseline (fraction of positives)
    if baseline_ap is not None:
        ax.axhline(y=baseline_ap, color="#888", linestyle=":", linewidth=1.2,
                   label=f"Random baseline (AP={baseline_ap:.3f})")

    ax.set_xlabel("Recall",    fontsize=13, color="#DDDDDD", labelpad=8)
    ax.set_ylabel("Precision", fontsize=13, color="#DDDDDD", labelpad=8)
    ax.set_title("Precision-Recall Curves — All Models",
                 fontsize=15, color="#FFFFFF", fontweight="bold", pad=14)

    ax.set_xlim([0.0, 1.01])
    ax.set_ylim([0.0, 1.05])
    ax.tick_params(colors="#AAAAAA")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

    ax.grid(True, color="#222244", linewidth=0.6, linestyle="--")

    legend = ax.legend(
        loc="lower left",
        fontsize=10,
        facecolor="#16162A",
        edgecolor="#333355",
        labelcolor="#DDDDDD",
        framealpha=0.9,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {out_path}")


def plot_individual(name: str, res: dict, out_path: str, baseline_ap: float = None):
    """Single-model PR curve with filled area."""
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#0F0F1A")
    ax.set_facecolor("#0F0F1A")

    color = COLORS[name]
    ax.plot(res["recall"], res["precision"],
            color=color, linewidth=2.4, label=f"AP = {res['ap']:.3f}")
    ax.fill_between(res["recall"], res["precision"],
                    alpha=0.15, color=color)

    if baseline_ap is not None:
        ax.axhline(y=baseline_ap, color="#888", linestyle=":", linewidth=1.2,
                   label=f"Random (AP={baseline_ap:.3f})")

    ax.set_xlabel("Recall",    fontsize=12, color="#DDDDDD")
    ax.set_ylabel("Precision", fontsize=12, color="#DDDDDD")
    ax.set_title(f"PR Curve — {name}\nAP={res['ap']:.3f}  AUC-ROC={res['auc_roc']:.3f}",
                 fontsize=13, color="#FFFFFF", fontweight="bold")
    ax.set_xlim([0.0, 1.01])
    ax.set_ylim([0.0, 1.05])
    ax.tick_params(colors="#AAAAAA")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    ax.grid(True, color="#222244", linewidth=0.6, linestyle="--")
    ax.legend(fontsize=11, facecolor="#16162A", edgecolor="#333355",
              labelcolor="#DDDDDD")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"    Saved → {out_path}")


def plot_ap_bar(results: dict, out_path: str):
    """Horizontal bar chart of AP scores — quick comparison for reports."""
    names = list(results.keys())
    aps   = [results[n]["ap"] for n in names]
    aucs  = [results[n]["auc_roc"] for n in names]

    # Sort by AP
    order = np.argsort(aps)
    names = [names[i] for i in order]
    aps   = [aps[i]   for i in order]
    aucs  = [aucs[i]  for i in order]
    colors = [COLORS[n] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#0F0F1A")

    for ax, vals, title in zip(axes, [aps, aucs], ["Average Precision (AP)", "AUC-ROC"]):
        ax.set_facecolor("#0F0F1A")
        bars = ax.barh(names, vals, color=colors, height=0.55, edgecolor="#333355")
        for bar, val in zip(bars, vals):
            ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=10, color="#DDDDDD")
        ax.set_xlim([0, 1.08])
        ax.set_title(title, fontsize=13, color="#FFFFFF", fontweight="bold", pad=10)
        ax.tick_params(colors="#AAAAAA", labelsize=10)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")
        ax.grid(True, axis="x", color="#222244", linewidth=0.6, linestyle="--")

    fig.suptitle("Model Comparison — All Metrics",
                 fontsize=15, color="#FFFFFF", fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    device = args.device
    os.makedirs(args.out_dir, exist_ok=True)
    individual_dir = os.path.join(args.out_dir, "individual")
    os.makedirs(individual_dir, exist_ok=True)

    loader_kw = dict(batch_size=args.batch_size, shuffle=False,
                     num_workers=args.num_workers, pin_memory=True)

    # Build loaders
    npz_loader        = DataLoader(VideoClipDataset(args.npz_csv),   **loader_kw)
    two_stream_loader = DataLoader(TwoStreamDataset(args.val_csv),   **loader_kw)

    # ── Model registry ─────────────────────────────────────────────────────────
    # (model_name, build_fn, checkpoint_flag_key, loader, is_two_stream)
    registry = [
        ("3DCNN",
         build_3dcnn,
         args.ckpt_3dcnn,
         npz_loader, False),

        ("CNN+LSTM",
         build_cnn_lstm,
         args.ckpt_cnn_lstm,
         npz_loader, False),

        ("CNN+Transformer",
         build_cnn_transformer,
         args.ckpt_cnn_transformer,
         npz_loader, False),

        ("TwoStream-CNN",
         build_two_stream_cnn,
         args.ckpt_twostream_cnn,
         two_stream_loader, True),

        ("TwoStream-ResNet",
         build_two_stream_resnet,
         args.ckpt_twostream_resnet,
         two_stream_loader, True),

        ("TwoStream-Transformer",
         build_two_stream_transformer,
         args.ckpt_twostream_transformer,
         two_stream_loader, True),
    ]

    results = {}

    print("\n─── Evaluating models ─────────────────────────────────────────────")
    for name, build_fn, ckpt_path, loader, is_two_stream in registry:
        print(f"\n[{name}]  checkpoint: {ckpt_path}")

        if not os.path.exists(ckpt_path):
            print(f"  ⚠  Checkpoint not found — skipping.")
            continue

        # Build + load weights
        model = build_fn()
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state)
        model.to(device)
        model.eval()

        # Collect predictions
        if is_two_stream:
            probs, labels = collect_preds_two_stream(model, loader, device, name)
        else:
            probs, labels = collect_preds_single_stream(model, loader, device, name)

        # Compute metrics
        precision, recall, _ = precision_recall_curve(labels, probs)
        ap      = average_precision_score(labels, probs)
        auc_roc = roc_auc_score(labels, probs)

        results[name] = {
            "precision": precision,
            "recall":    recall,
            "ap":        ap,
            "auc_roc":   auc_roc,
            "labels":    labels,
        }

        print(f"  AP={ap:.4f}   AUC-ROC={auc_roc:.4f}")

        # Free GPU memory between models
        del model
        torch.cuda.empty_cache() if device == "cuda" else None

    if not results:
        print("\nNo checkpoints found — nothing to plot.")
        return

    # Baseline AP (fraction of positives in dataset)
    all_labels = next(iter(results.values()))["labels"]
    baseline_ap = float(all_labels.mean())
    print(f"\nRandom-baseline AP (class imbalance ratio): {baseline_ap:.4f}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    print("\n─── Generating plots ──────────────────────────────────────────────")

    # 1) Combined PR curves
    plot_combined(
        results,
        os.path.join(args.out_dir, "pr_curves_all_models.png"),
        baseline_ap=baseline_ap,
    )

    # 2) AP + AUC bar comparison
    plot_ap_bar(
        results,
        os.path.join(args.out_dir, "model_comparison_bar.png"),
    )

    # 3) Individual PR curves
    print("\n  Individual curves:")
    for name, res in results.items():
        safe_name = name.replace("+", "_").replace("-", "_").lower()
        plot_individual(
            name, res,
            os.path.join(individual_dir, f"pr_{safe_name}.png"),
            baseline_ap=baseline_ap,
        )

    # ── Summary table ──────────────────────────────────────────────────────────
    print("\n─── Summary ────────────────────────────────────────────────────────")
    print(f"{'Model':<28}  {'AP':>7}  {'AUC-ROC':>9}")
    print("─" * 50)
    for name, res in sorted(results.items(), key=lambda x: x[1]["ap"], reverse=True):
        print(f"{name:<28}  {res['ap']:>7.4f}  {res['auc_roc']:>9.4f}")

    best = max(results, key=lambda n: results[n]["ap"])
    print(f"\n★  Best model by AP: {best}  (AP={results[best]['ap']:.4f})")
    print(f"\nAll outputs saved to: {args.out_dir}/")


if __name__ == "__main__":
    main()