"""
gradcam_two_stream_transformer.py

GradCAM++ visualisation for TwoStreamTransformer.

What it produces
----------------
For each frame in the video clip, this script:
    1. Hooks into the LAST ResNet18 conv layer (layer4) of the RGB (spatial)
       stream and computes GradCAM++ activation maps.
    2. Resizes the heatmap to 224×224 and overlays it on the original frame
       as a colour jet overlay (blue=ignored → red=attended).
    3. Saves a side-by-side strip:  original | heatmap | overlay
    4. Also saves an animated GIF of the overlay frames so you can watch
       attention move through time.

Why GradCAM++ over vanilla GradCAM
-----------------------------------
GradCAM++ weights each activation map channel by the *second-order* gradient,
which gives sharper, better-localised maps — especially useful for small
objects (vehicles, pedestrians) in accident scenes.

Why layer4 of ResNet18
-----------------------
layer4 is the last convolutional block before the global average pool.
Its feature maps are 7×7 spatial × 512 channels, capturing high-level
semantic content (object shapes, motion blobs) while retaining enough
spatial resolution to produce interpretable heatmaps at 224×224.

Usage
-----
    python gradcam_two_stream_transformer.py \
        --checkpoint checkpoints/two_stream_transformer_best.pth \
        --rgb-dir    data/video_001/frames_enhanced \
        --flow-dir   data/video_001/optical_flow_png \
        --start      0 \
        --end        15 \
        --out-dir    gradcam_output

Optional flags
--------------
    --target-class   1        force positive (accident) class gradient
                              (default: use predicted class)
    --alpha          0.5      overlay transparency (0=invisible, 1=opaque)
    --no-gif                  skip GIF generation
    --fps            4        GIF frames per second
    --d-model        256
    --nhead          8
    --num-layers     4
    --dim-ff         512
    --fusion         concat
"""

import os
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from models.two_stream_transformer import TwoStreamTransformer
from dataset.two_stream_resnet_dataset import TwoStreamDataset   # reuse loaders

# ─── Constants (must match training) ─────────────────────────────────────────
FRAME_SIZE = (224, 224)
FRAME_EXT  = (".jpg", ".png", ".jpeg")
FLOW_CLIP  = 20.0
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


# ─── Frame loading (mirrors two_stream_resnet_dataset.py) ─────────────────────

def load_rgb_frame(path: str) -> np.ndarray:
    """Load, resize, ImageNet-normalise → (H,W,3) float32."""
    img = cv2.imread(path)
    if img is None:
        return np.zeros((*FRAME_SIZE, 3), dtype=np.float32)
    img = cv2.cvtColor(
        cv2.resize(img, FRAME_SIZE, interpolation=cv2.INTER_LINEAR),
        cv2.COLOR_BGR2RGB,
    ).astype(np.float32) / 255.0
    return (img - _MEAN) / _STD


def load_rgb_frame_raw(path: str) -> np.ndarray:
    """Load original frame (no normalisation) → (H,W,3) uint8 RGB for display."""
    img = cv2.imread(path)
    if img is None:
        return np.zeros((*FRAME_SIZE, 3), dtype=np.uint8)
    img = cv2.cvtColor(
        cv2.resize(img, FRAME_SIZE, interpolation=cv2.INTER_LINEAR),
        cv2.COLOR_BGR2RGB,
    )
    return img


def load_flow_frame(path: str) -> np.ndarray:
    """PNG uint8 → float32 (H,W,2) in [-1,1]."""
    img = cv2.imread(path)
    if img is None:
        return np.zeros((*FRAME_SIZE, 2), dtype=np.float32)
    img = cv2.resize(img, FRAME_SIZE, interpolation=cv2.INTER_LINEAR)
    dx = ((img[..., 0].astype(np.float32) / 255.0) * 2 * FLOW_CLIP - FLOW_CLIP) / FLOW_CLIP
    dy = ((img[..., 1].astype(np.float32) / 255.0) * 2 * FLOW_CLIP - FLOW_CLIP) / FLOW_CLIP
    return np.stack([dx, dy], axis=-1)


def get_sorted_frames(directory: str) -> list:
    return sorted([
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(FRAME_EXT)
    ])


# ─── GradCAM++ implementation ─────────────────────────────────────────────────

class GradCAMPlusPlus:
    """
    GradCAM++ for a ResNet18 backbone inside TwoStreamTransformer.

    Hooks into `spatial_stream.backbone` → layer4 (index 7 of the Sequential).
    Computes per-frame heatmaps for a full (B, C, T, H, W) clip.
    """

    def __init__(self, model: TwoStreamTransformer, device: str):
        self.model  = model
        self.device = device

        # Target layer: last conv block of the RGB backbone
        # model.spatial_stream.backbone is nn.Sequential(*resnet18.children()[:-1])
        # children: [conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool]
        # index 7 = layer4
        self.target_layer = model.spatial_stream.backbone[7]

        self._fmaps   = None   # forward feature maps
        self._grads   = None   # backward gradients

        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(module, input, output):
            self._fmaps = output.detach()   # (B*T, 512, 7, 7)

        def bwd_hook(module, grad_in, grad_out):
            self._grads = grad_out[0]       # (B*T, 512, 7, 7)

        self._fwd_handle = self.target_layer.register_forward_hook(fwd_hook)
        self._bwd_handle = self.target_layer.register_full_backward_hook(bwd_hook)

    def remove_hooks(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def compute(
        self,
        rgb: torch.Tensor,
        flow: torch.Tensor,
        target_class: int = None,
    ) -> np.ndarray:
        """
        Args:
            rgb          : (1, 3, T, H, W)
            flow         : (1, 2, T, H, W)
            target_class : 0 or 1; None = use predicted class

        Returns:
            heatmaps : (T, H, W) float32 in [0, 1]
                       one normalised heatmap per frame
        """
        self.model.eval()
        rgb  = rgb.to(self.device).requires_grad_(False)
        flow = flow.to(self.device).requires_grad_(False)

        # Enable gradients for the backbone input only
        rgb_grad = rgb.detach().requires_grad_(True)

        # Forward pass — hooks capture fmaps
        logit = self.model(rgb_grad, flow)           # (1, 1)
        score = logit[0, 0]

        # Use predicted class if target not specified
        if target_class is None:
            target_class = 1 if score.item() > 0 else 0

        # Sign: positive class → maximise score, negative → minimise
        sign = 1.0 if target_class == 1 else -1.0

        self.model.zero_grad()
        (sign * score).backward()

        # ── GradCAM++ weights ──────────────────────────────────────────────
        # grads  : (B*T, C, h, w)   where h=w=7 for ResNet18
        # fmaps  : (B*T, C, h, w)
        grads = self._grads                          # (B*T, 512, 7, 7)
        fmaps = self._fmaps                          # (B*T, 512, 7, 7)

        # GradCAM++ formula:
        #   alpha_kc = grad^2 / (2*grad^2 + sum(A * grad^3) + eps)
        #   weight_k = sum_ij( alpha_kc * ReLU(grad) )
        grads_sq  = grads ** 2
        grads_cub = grads ** 3
        # sum over spatial dims (h, w)
        denom = 2.0 * grads_sq + (fmaps * grads_cub).sum(dim=(2, 3), keepdim=True) + 1e-8
        alpha = grads_sq / denom                     # (B*T, 512, 7, 7)

        weights = (alpha * F.relu(grads)).sum(dim=(2, 3))   # (B*T, 512)

        # Weighted combination of feature maps
        BT, C, h, w = fmaps.shape
        cam = (weights.view(BT, C, 1, 1) * fmaps).sum(dim=1)   # (B*T, h, w)
        cam = F.relu(cam)

        # Resize each frame's CAM to (H, W) and normalise to [0,1]
        T = rgb.shape[2]
        heatmaps = []
        for t in range(T):
            hm = cam[t].detach().cpu().numpy()           # (7, 7)
            hm = cv2.resize(hm, FRAME_SIZE,
                            interpolation=cv2.INTER_CUBIC)
            mn, mx = hm.min(), hm.max()
            if mx > mn:
                hm = (hm - mn) / (mx - mn)
            else:
                hm = np.zeros_like(hm)
            heatmaps.append(hm)

        return np.stack(heatmaps)   # (T, H, W)


# ─── Overlay helpers ──────────────────────────────────────────────────────────

def heatmap_to_rgb(hm: np.ndarray, colormap=cv2.COLORMAP_JET) -> np.ndarray:
    """Convert (H,W) float [0,1] heatmap → (H,W,3) uint8 RGB."""
    hm_u8 = (hm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(hm_u8, colormap)   # BGR
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def overlay_heatmap(
    frame_rgb: np.ndarray,   # (H,W,3) uint8
    heatmap:   np.ndarray,   # (H,W) float [0,1]
    alpha:     float = 0.5,
) -> np.ndarray:
    """Alpha-blend heatmap over frame → (H,W,3) uint8."""
    hm_rgb = heatmap_to_rgb(heatmap)
    blended = cv2.addWeighted(frame_rgb, 1 - alpha, hm_rgb, alpha, 0)
    return blended


def make_strip(original: np.ndarray, heatmap_img: np.ndarray,
               overlay: np.ndarray, frame_idx: int,
               risk_score: float) -> np.ndarray:
    """
    Returns a horizontal strip:  [original | heatmap | overlay]
    with a label bar at the top.
    """
    H, W = original.shape[:2]
    bar_h = 28
    strip_w = W * 3

    # Label bar
    bar = np.zeros((bar_h, strip_w, 3), dtype=np.uint8)
    label = f"Frame {frame_idx:03d}   Risk score: {risk_score:.3f}"
    cv2.putText(bar, label, (8, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)

    # Column headers
    for i, title in enumerate(["Original", "GradCAM++ heatmap", "Overlay"]):
        cv2.putText(bar, title, (W * i + 8, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (200, 200, 200), 1, cv2.LINE_AA)

    # Assemble strip
    row = np.concatenate([original, heatmap_img, overlay], axis=1)
    return np.concatenate([bar, row], axis=0)


# ─── GIF writer (pure OpenCV, no Pillow dependency) ──────────────────────────

def save_gif(frames: list, path: str, fps: int = 4) -> None:
    """Save list of (H,W,3) uint8 RGB frames as an animated GIF using Pillow."""
    try:
        from PIL import Image
        imgs = [Image.fromarray(f) for f in frames]
        duration_ms = int(1000 / fps)
        imgs[0].save(
            path,
            save_all=True,
            append_images=imgs[1:],
            loop=0,
            duration=duration_ms,
        )
        print(f"Saved GIF → {path}")
    except ImportError:
        print("Pillow not installed — skipping GIF. "
              "Install with: pip install Pillow")


# ─── Summary plot ─────────────────────────────────────────────────────────────

def save_summary_grid(
    raw_frames:  list,    # list of (H,W,3) uint8
    heatmaps:    np.ndarray,   # (T, H, W)
    overlays:    list,    # list of (H,W,3) uint8
    risk_scores: list,    # float per frame
    out_path:    str,
    max_frames:  int = 8,
) -> None:
    """
    Grid plot: each column = one sampled frame
    Rows: original / heatmap / overlay / risk bar
    """
    T = len(raw_frames)
    indices = np.linspace(0, T - 1, min(T, max_frames), dtype=int)
    n = len(indices)

    fig, axes = plt.subplots(3, n, figsize=(n * 2.5, 8))
    fig.suptitle("GradCAM++ — TwoStreamTransformer  (spatial RGB stream)",
                 fontsize=13, y=1.01)

    cmap = cm.get_cmap("jet")

    for col, t in enumerate(indices):
        # Row 0: original frame
        ax = axes[0, col] if n > 1 else axes[0]
        ax.imshow(raw_frames[t])
        ax.set_title(f"f{t}", fontsize=8)
        ax.axis("off")

        # Row 1: pure heatmap
        ax = axes[1, col] if n > 1 else axes[1]
        ax.imshow(heatmaps[t], cmap="jet", vmin=0, vmax=1)
        ax.axis("off")

        # Row 2: overlay
        ax = axes[2, col] if n > 1 else axes[2]
        ax.imshow(overlays[t])
        ax.set_xlabel(f"risk={risk_scores[t]:.2f}", fontsize=8)
        ax.xaxis.set_label_position("bottom")
        ax.axis("off")

    # Row labels
    for row, label in enumerate(["Original", "GradCAM++ map", "Overlay"]):
        ax = axes[row, 0] if n > 1 else axes[row]
        ax.set_ylabel(label, fontsize=9, rotation=90, labelpad=4)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap="jet",
                               norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(),
                        orientation="vertical", fraction=0.015, pad=0.02)
    cbar.set_label("Attention intensity", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved summary grid → {out_path}")


# ─── Argument parser ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="GradCAM++ for TwoStreamTransformer"
    )
    p.add_argument("--checkpoint",    type=str, required=True)
    p.add_argument("--rgb-dir",       type=str, required=True,
                   help="Directory of RGB frames")
    p.add_argument("--flow-dir",      type=str, required=True,
                   help="Directory of optical flow PNGs")
    p.add_argument("--start",         type=int, default=0,
                   help="Start frame index within rgb-dir")
    p.add_argument("--end",           type=int, default=15,
                   help="End frame index (inclusive)")
    p.add_argument("--target-class",  type=int, default=None,
                   choices=[0, 1],
                   help="Force gradient direction (default: use prediction)")
    p.add_argument("--alpha",         type=float, default=0.5,
                   help="Heatmap overlay transparency")
    p.add_argument("--out-dir",       type=str, default="gradcam_output")
    p.add_argument("--no-gif",        action="store_true")
    p.add_argument("--fps",           type=int, default=4)
    p.add_argument("--d-model",       type=int, default=256)
    p.add_argument("--nhead",         type=int, default=8)
    p.add_argument("--num-layers",    type=int, default=4)
    p.add_argument("--dim-ff",        type=int, default=512)
    p.add_argument("--fusion",        type=str, default="concat")
    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    model = TwoStreamTransformer(
        d_model    = args.d_model,
        nhead      = args.nhead,
        num_layers = args.num_layers,
        dim_ff     = args.dim_ff,
        fusion     = args.fusion,
        grad_ckpt  = False,   # MUST be off for GradCAM (no checkpointing during eval)
    ).to(device)

    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print("Model loaded.")

    # ── Load frames ───────────────────────────────────────────────────────────
    all_rgb_files  = get_sorted_frames(args.rgb_dir)
    all_flow_files = get_sorted_frames(args.flow_dir)
    flow_lookup    = {Path(p).stem: p for p in all_flow_files}

    frame_indices = list(range(args.start, args.end + 1))
    T = len(frame_indices)
    print(f"Processing {T} frames (indices {args.start}–{args.end}) …")

    # Build (1, C, T, H, W) tensors
    rgb_frames_norm  = []
    flow_frames_norm = []
    rgb_frames_raw   = []

    for i in frame_indices:
        fpath = all_rgb_files[i]
        stem  = Path(fpath).stem
        rgb_frames_norm.append(load_rgb_frame(fpath))
        flow_frames_norm.append(load_flow_frame(flow_lookup.get(stem, "")))
        rgb_frames_raw.append(load_rgb_frame_raw(fpath))

    rgb_t  = torch.from_numpy(
        np.stack(rgb_frames_norm)              # (T, H, W, 3)
    ).permute(3, 0, 1, 2).unsqueeze(0)        # (1, 3, T, H, W)

    flow_t = torch.from_numpy(
        np.stack(flow_frames_norm)             # (T, H, W, 2)
    ).permute(3, 0, 1, 2).unsqueeze(0)        # (1, 2, T, H, W)

    # ── Quick overall risk score ──────────────────────────────────────────────
    with torch.no_grad():
        logit_clip = model(rgb_t.to(device), flow_t.to(device))
        clip_risk  = torch.sigmoid(logit_clip).item()
    print(f"Clip risk score: {clip_risk:.4f}  "
          f"({'ACCIDENT' if clip_risk > 0.5 else 'SAFE'})")

    # ── GradCAM++ per-frame ───────────────────────────────────────────────────
    # We compute a separate GradCAM pass per frame by sliding a single-frame
    # window, but use the FULL clip context for the transformer's attention.
    # Strategy: compute GradCAM on the full clip once; the (B*T) dimension
    # in the hooked layer already gives us per-frame maps.

    gradcam    = GradCAMPlusPlus(model, device)
    heatmaps   = gradcam.compute(rgb_t, flow_t, target_class=args.target_class)
    gradcam.remove_hooks()
    print(f"Heatmaps computed: {heatmaps.shape}")  # (T, H, W)

    # ── Per-frame risk scores (sliding single-frame context) ──────────────────
    # For the risk-score label on each frame we use the full-clip score
    # since the transformer needs temporal context.
    # Alternatively compute a per-frame score by repeating the frame T times.
    per_frame_risk = []
    with torch.no_grad():
        for t in range(T):
            # Repeat single frame T times to give transformer a valid input
            single_rgb  = rgb_t[:, :, t:t+1, :, :].expand(-1, -1, T, -1, -1).to(device)
            single_flow = flow_t[:, :, t:t+1, :, :].expand(-1, -1, T, -1, -1).to(device)
            r = torch.sigmoid(model(single_rgb, single_flow)).item()
            per_frame_risk.append(r)

    # ── Build overlay images ──────────────────────────────────────────────────
    overlays = []
    strip_frames = []

    for t in range(T):
        hm_img  = heatmap_to_rgb(heatmaps[t])
        ov      = overlay_heatmap(rgb_frames_raw[t], heatmaps[t], alpha=args.alpha)
        overlays.append(ov)

        # Individual frame strip
        strip = make_strip(
            rgb_frames_raw[t], hm_img, ov,
            frame_indices[t], per_frame_risk[t],
        )
        strip_frames.append(strip)

        frame_out = os.path.join(args.out_dir, f"frame_{frame_indices[t]:04d}.png")
        cv2.imwrite(frame_out, cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))

    print(f"Saved {T} individual frame strips to {args.out_dir}/")

    # ── Summary grid ──────────────────────────────────────────────────────────
    grid_path = os.path.join(args.out_dir, "gradcam_summary_grid.png")
    save_summary_grid(
        raw_frames  = rgb_frames_raw,
        heatmaps    = heatmaps,
        overlays    = overlays,
        risk_scores = per_frame_risk,
        out_path    = grid_path,
        max_frames  = 8,
    )

    # ── Animated GIF ─────────────────────────────────────────────────────────
    if not args.no_gif:
        gif_path = os.path.join(args.out_dir, "gradcam_overlay.gif")
        save_gif(overlays, gif_path, fps=args.fps)

    # ── Risk score over time plot ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(frame_indices, per_frame_risk,
            color="#E24B4A", linewidth=2, label="Per-frame risk")
    ax.axhline(0.5, color="#444", linewidth=1, linestyle="--",
               label="Threshold (0.5)")
    ax.axhline(clip_risk, color="#378ADD", linewidth=1.5, linestyle="-.",
               label=f"Full-clip risk ({clip_risk:.3f})")
    ax.fill_between(frame_indices, per_frame_risk, 0.5,
                    where=[r > 0.5 for r in per_frame_risk],
                    alpha=0.15, color="#E24B4A", label="High-risk frames")
    ax.set_xlabel("Frame index", fontsize=11)
    ax.set_ylabel("Risk score", fontsize=11)
    ax.set_title("Per-frame risk score — GradCAM++ clip", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    risk_plot_path = os.path.join(args.out_dir, "per_frame_risk.png")
    plt.savefig(risk_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved risk plot  → {risk_plot_path}")

    print(f"\nAll outputs in: {args.out_dir}/")
    print(f"  gradcam_summary_grid.png  — {T}-frame grid")
    print(f"  frame_XXXX.png            — individual strips")
    print(f"  gradcam_overlay.gif       — animated overlay")
    print(f"  per_frame_risk.png        — risk curve")