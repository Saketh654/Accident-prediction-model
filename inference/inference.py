"""
inference.py — Unified inference for all accident anticipation models.

Supports:
    - Accident3DCNN
    - CNNLSTM
    - CNNTransformer
    - TwoStreamCNN
    - TwoStreamCNNRes (Two-Stream ResNet)
    - TwoStreamTransformer

Features:
    - Per-frame bounding box overlay (via background subtraction or saliency)
    - Risk score HUD (colour-coded gauge)
    - Per-frame risk timeline bar
    - Optional video output (MP4)
    - Supports video files, webcam, and frame directories

Usage:
    python inference.py --model two_stream_transformer \
                        --checkpoint checkpoints/two_stream_transformer_best.pth \
                        --source path/to/video.mp4 \
                        --output output/result.mp4 \
                        --clip-len 16 \
                        --threshold 0.5

    python inference.py --model accident_3dcnn \
                        --checkpoint checkpoints/accident_model.pth \
                        --source 0                  # webcam

    python inference.py --list-models               # list all available models
"""

import os
import sys
import cv2
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from collections import deque
from typing import Optional, List, Tuple

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

# ── Model imports ─────────────────────────────────────────────────────────────
from models.accident_3d_cnn import Accident3DCNN
from models.cnn_lstm import CNNLSTM
from models.cnn_transformer import CNNTransformer
from models.two_stream_cnn import TwoStreamCNN
from models.two_stream_resnet import TwoStreamCNNRes
from models.two_stream_transformer import TwoStreamTransformer

# ── Constants ─────────────────────────────────────────────────────────────────
FRAME_SIZE      = (224, 224)
FLOW_CLIP       = 20.0
IMAGENET_MEAN   = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD    = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DISPLAY_SIZE    = (1280, 720)   # output frame resolution

# Colour palette  (BGR)
COL_SAFE        = (60,  200, 60)    # green
COL_WARN        = (0,   200, 220)   # yellow
COL_DANGER      = (30,  30,  220)   # red
COL_HUD_BG      = (20,  20,  20)
COL_WHITE       = (240, 240, 240)
COL_BOX         = (0,   140, 255)   # orange bounding box

AVAILABLE_MODELS = [
    "accident_3dcnn",
    "cnn_lstm",
    "cnn_transformer",
    "two_stream_cnn",
    "two_stream_resnet",
    "two_stream_transformer",
]

TWO_STREAM_MODELS = {
    "two_stream_cnn",
    "two_stream_resnet",
    "two_stream_transformer",
}

# ── Checkpoint mapping ────────────────────────────────────────────────────────
# Maps each model name to its default checkpoint path (relative to project root)
CHECKPOINT_MAP = {
    "accident_3dcnn":         "checkpoints/accident_model.pth",
    "cnn_lstm":               "checkpoints/cnn_lstm_best.pth",
    "cnn_transformer":        "checkpoints/cnn_transformer_best.pth",
    "two_stream_cnn":         "checkpoints/two_stream_best.pth",
    "two_stream_resnet":      "checkpoints/two_stream_resnet_best.pth",
    "two_stream_transformer": "checkpoints/two_stream_transformer_final.pth",
}


def _resolve_checkpoint(model_name: str) -> Optional[str]:
    """
    Return the default checkpoint path for *model_name*, resolved relative to
    the project root.  Returns None if the file does not exist on disk.
    """
    rel_path = CHECKPOINT_MAP.get(model_name)
    if rel_path is None:
        return None
    abs_path = os.path.join(ROOT, rel_path)
    return abs_path if os.path.isfile(abs_path) else rel_path   # always return the path string; build_model will warn if missing


# ═══════════════════════════════════════════════════════════════════════════════
# Model factory
# ═══════════════════════════════════════════════════════════════════════════════

def build_model(name: str, checkpoint: Optional[str], device: str) -> nn.Module:
    """Instantiate model and optionally load checkpoint weights."""

    name = name.lower().strip()

    if name == "accident_3dcnn":
        model = Accident3DCNN()

    elif name == "cnn_lstm":
        backbone = models.resnet18(weights=None)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        model = CNNLSTM(cnn=backbone, feature_dim=feat_dim,
                        hidden_dim=256, num_classes=1)

    elif name == "cnn_transformer":
        model = CNNTransformer(
            feature_dim=512, d_model=256, nhead=8,
            num_layers=4, dim_feedforward=512, dropout=0.1
        )

    elif name == "two_stream_cnn":
        model = TwoStreamCNN(base_ch=32, fusion="concat")

    elif name == "two_stream_resnet":
        model = TwoStreamCNNRes(fusion="concat")

    elif name == "two_stream_transformer":
        model = TwoStreamTransformer(
            d_model=256, nhead=8, num_layers=4,
            dim_ff=512, fusion="concat", grad_ckpt=False
        )

    else:
        raise ValueError(
            f"Unknown model '{name}'. Choose from: {AVAILABLE_MODELS}"
        )

    if checkpoint and os.path.isfile(checkpoint):
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f"  ✓ Loaded weights: {checkpoint}")
    elif checkpoint:
        print(f"  ⚠  Checkpoint not found: {checkpoint} — running with random weights")
    else:
        print("  ⚠  No checkpoint provided — running with random weights")

    model.to(device)
    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Pre-processing helpers
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_rgb(frame: np.ndarray) -> np.ndarray:
    """BGR frame (H,W,3 uint8) → normalized float32 (H,W,3)."""
    img = cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img


def compute_optical_flow(prev_gray: np.ndarray,
                          curr_gray: np.ndarray) -> np.ndarray:
    """Returns float32 (H,W,2) flow in [-1, 1]."""
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    flow_resized = cv2.resize(flow, FRAME_SIZE, interpolation=cv2.INTER_LINEAR)
    flow_clipped = np.clip(flow_resized, -FLOW_CLIP, FLOW_CLIP) / FLOW_CLIP
    return flow_clipped.astype(np.float32)


def build_tensors(rgb_clip: list, flow_clip: list,
                  model_name: str, device: str):
    """
    Convert frame lists to the exact tensor shapes each model expects.
    Returns (rgb_tensor, flow_tensor) — flow is None for single-stream models.
    """
    rgb_arr  = np.stack(rgb_clip)   # (T, H, W, 3)
    rgb_t    = torch.from_numpy(rgb_arr).permute(3, 0, 1, 2).unsqueeze(0)  # (1,3,T,H,W)

    if model_name in TWO_STREAM_MODELS:
        flow_arr = np.stack(flow_clip)  # (T, H, W, 2)
        flow_t   = torch.from_numpy(flow_arr).permute(3, 0, 1, 2).unsqueeze(0)  # (1,2,T,H,W)
    else:
        flow_t = None

    # CNNLSTM / CNNTransformer expect (B, T, C, H, W)
    if model_name in ("cnn_lstm", "cnn_transformer"):
        rgb_t = rgb_t.permute(0, 2, 1, 3, 4)   # (1, T, 3, H, W)

    return rgb_t.to(device), (flow_t.to(device) if flow_t is not None else None)


# ═══════════════════════════════════════════════════════════════════════════════
# Bounding-box detection (motion-based + lightweight contour)
# ═══════════════════════════════════════════════════════════════════════════════

class ObjectDetector:
    """
    YOLOv8-based detector filtered to vehicles and pedestrians only.
    Returns (x, y, w, h, label) tuples in the original frame coordinates.
    """

    # COCO class IDs we care about → display label
    TARGET_CLASSES = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }

    def __init__(self, model_size: str = "n", conf: float = 0.35):
        """
        Args:
            model_size : YOLOv8 variant — 'n' (nano), 's', 'm', 'l', 'x'
            conf       : minimum confidence threshold
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics is required for object detection.\n"
                "Install it with:  pip install ultralytics"
            )
        self.model = YOLO(f"yolov8{model_size}.pt")   # auto-downloads on first run
        self.conf  = conf

    def detect(self, frame: np.ndarray,
               min_area: int = 800) -> List[Tuple]:
        """
        Args:
            frame    : BGR uint8 original-resolution frame
            min_area : minimum box area (px²) to keep
        Returns:
            list of (x, y, w, h, label)
        """
        results = self.model(frame, conf=self.conf, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in self.TARGET_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bw, bh = x2 - x1, y2 - y1
            if bw * bh < min_area:
                continue
            label = self.TARGET_CLASSES[cls_id]
            detections.append((x1, y1, bw, bh, label))
        return detections


# ═══════════════════════════════════════════════════════════════════════════════
# HUD drawing helpers
# ═══════════════════════════════════════════════════════════════════════════════

def risk_colour(score: float) -> Tuple[int, int, int]:
    """Interpolate green → yellow → red based on risk score [0, 1]."""
    if score < 0.5:
        t = score * 2
        b = int((1 - t) * COL_SAFE[0] + t * COL_WARN[0])
        g = int((1 - t) * COL_SAFE[1] + t * COL_WARN[1])
        r = int((1 - t) * COL_SAFE[2] + t * COL_WARN[2])
    else:
        t = (score - 0.5) * 2
        b = int((1 - t) * COL_WARN[0] + t * COL_DANGER[0])
        g = int((1 - t) * COL_WARN[1] + t * COL_DANGER[1])
        r = int((1 - t) * COL_WARN[2] + t * COL_DANGER[2])
    return (b, g, r)


def draw_hud(canvas: np.ndarray, score: float, model_name: str,
             fps: float, frame_idx: int, history: deque,
             threshold: float, is_two_stream: bool) -> None:
    """
    Draws an information overlay on `canvas` (in-place).

    Layout:
        Top-left  : risk gauge + score + label
        Top-right : model info + FPS
        Bottom    : scrolling risk timeline
    """
    h, w = canvas.shape[:2]
    col  = risk_colour(score)
    label = "DANGER" if score >= threshold else ("WARNING" if score >= threshold * 0.6 else "SAFE")

    # ── Semi-transparent top banner ───────────────────────────────────────────
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), COL_HUD_BG, -1)
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

    # ── Risk gauge bar ────────────────────────────────────────────────────────
    gauge_x, gauge_y, gauge_w, gauge_h = 20, 18, 340, 32
    cv2.rectangle(canvas, (gauge_x, gauge_y),
                  (gauge_x + gauge_w, gauge_y + gauge_h), (60, 60, 60), -1)
    fill = int(score * gauge_w)
    if fill > 0:
        cv2.rectangle(canvas, (gauge_x, gauge_y),
                      (gauge_x + fill, gauge_y + gauge_h), col, -1)
    cv2.rectangle(canvas, (gauge_x, gauge_y),
                  (gauge_x + gauge_w, gauge_y + gauge_h), COL_WHITE, 1)

    # Threshold marker
    tx = gauge_x + int(threshold * gauge_w)
    cv2.line(canvas, (tx, gauge_y - 3), (tx, gauge_y + gauge_h + 3), (200, 200, 200), 2)

    # ── Score text ─────────────────────────────────────────────────────────────
    score_txt = f"{score:.3f}"
    cv2.putText(canvas, score_txt, (gauge_x + gauge_w + 12, gauge_y + 24),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, col, 2, cv2.LINE_AA)

    # ── Label pill ─────────────────────────────────────────────────────────────
    lbl_x = gauge_x + gauge_w + 90
    cv2.rectangle(canvas, (lbl_x, gauge_y + 2), (lbl_x + 100, gauge_y + 28), col, -1)
    cv2.putText(canvas, label, (lbl_x + 6, gauge_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (10, 10, 10), 2, cv2.LINE_AA)

    # ── Top-right info block ───────────────────────────────────────────────────
    stream_tag = "2-Stream" if is_two_stream else "1-Stream"
    info_lines = [
        model_name.upper().replace("_", " "),
        f"{stream_tag}  |  FPS {fps:.1f}",
        f"Frame {frame_idx:06d}",
    ]
    for i, line in enumerate(info_lines):
        cv2.putText(canvas, line, (w - 320, 22 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, COL_WHITE, 1, cv2.LINE_AA)

    # ── Bottom timeline bar ────────────────────────────────────────────────────
    timeline_h = 18
    tl_y = h - timeline_h - 4
    cv2.rectangle(canvas, (0, tl_y), (w, h), COL_HUD_BG, -1)

    if len(history) > 1:
        step = max(1, w // len(history))
        pts  = []
        for j, s in enumerate(history):
            x = int(j * w / len(history))
            y = int(tl_y + timeline_h - s * timeline_h)
            pts.append((x, y))
            bar_col = risk_colour(s)
            cv2.rectangle(canvas, (x, tl_y + 1),
                          (x + step, tl_y + timeline_h - 1), bar_col, -1)

    # Timeline label
    cv2.putText(canvas, "Risk timeline", (4, tl_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1, cv2.LINE_AA)


def draw_boxes(frame: np.ndarray,
               boxes: List[Tuple],
               score: float, threshold: float) -> None:
    """Draw bounding boxes on frame (in-place). Colour reflects current risk."""
    col       = risk_colour(score)
    thickness = 3 if score >= threshold else 2
    for det in boxes:
        x, y, bw, bh, label = det
        # Main box
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), col, thickness)
        # Corner accents (top-left)
        cl = min(bw, bh, 20)
        cv2.line(frame, (x, y), (x + cl, y), COL_WHITE, thickness)
        cv2.line(frame, (x, y), (x, y + cl), COL_WHITE, thickness)
        # Corner accents (bottom-right)
        cv2.line(frame, (x + bw, y + bh), (x + bw - cl, y + bh), COL_WHITE, thickness)
        cv2.line(frame, (x + bw, y + bh), (x + bw, y + bh - cl), COL_WHITE, thickness)
        # Class label above box
        tag_y = max(y - 6, 14)
        cv2.putText(frame, label.upper(), (x, tag_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════════
# Main inference engine
# ═══════════════════════════════════════════════════════════════════════════════

def run_inference(args) -> None:
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = args.model.lower()
    print(f"\n{'='*60}")
    print(f"  Model      : {model_name}")
    print(f"  Checkpoint : {args.checkpoint or '(none)'}")
    print(f"  Device     : {device}")
    print(f"  Source     : {args.source}")
    print(f"  Clip len   : {args.clip_len} frames")
    print(f"  Threshold  : {args.threshold}")
    print(f"{'='*60}\n")

    model          = build_model(model_name, args.checkpoint, device)
    is_two_stream  = model_name in TWO_STREAM_MODELS
    detector       = ObjectDetector()
    score_history  = deque(maxlen=300)

    # ── Open source ───────────────────────────────────────────────────────────
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # ── Video writer ──────────────────────────────────────────────────────────
    writer = None
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, src_fps, DISPLAY_SIZE)
        print(f"  ✓ Writing output to: {args.output}")

    # ── Frame buffers ─────────────────────────────────────────────────────────
    rgb_buf    = deque(maxlen=args.clip_len)
    flow_buf   = deque(maxlen=args.clip_len)
    prev_gray  = None

    frame_idx  = 0
    score      = 0.0
    fps        = 0.0
    t_prev     = time.time()

    print("  Running... Press 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── Pre-process ───────────────────────────────────────────────────────
        rgb_proc = preprocess_rgb(frame)
        rgb_buf.append(rgb_proc)

        curr_gray = cv2.cvtColor(
            cv2.resize(frame, FRAME_SIZE), cv2.COLOR_BGR2GRAY
        )
        if prev_gray is not None:
            flow = compute_optical_flow(prev_gray, curr_gray)
            flow_buf.append(flow)
        prev_gray = curr_gray

        # ── Inference (once buffer is full) ───────────────────────────────────
        buf_ready = (len(rgb_buf) == args.clip_len and
                     (len(flow_buf) == args.clip_len or not is_two_stream))

        if buf_ready:
            with torch.no_grad():
                rgb_t, flow_t = build_tensors(
                    list(rgb_buf), list(flow_buf), model_name, device
                )
                if is_two_stream:
                    logit = model(rgb_t, flow_t)
                else:
                    logit = model(rgb_t)

            score = torch.sigmoid(logit).item()

        score_history.append(score)

        # ── Object detection (vehicles + pedestrians) ─────────────────────────
        boxes = detector.detect(frame, min_area=args.min_area)

        # ── Build display frame ───────────────────────────────────────────────
        display = cv2.resize(frame, DISPLAY_SIZE)

        # Scale boxes from original frame resolution to display resolution
        sx = DISPLAY_SIZE[0] / orig_w
        sy = DISPLAY_SIZE[1] / orig_h
        scaled_boxes = [
            (int(x * sx), int(y * sy), int(bw * sx), int(bh * sy), label)
            for (x, y, bw, bh, label) in boxes
        ]
        draw_boxes(display, scaled_boxes, score, args.threshold)

        # FPS
        now   = time.time()
        fps   = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-9))
        t_prev = now

        draw_hud(display, score, model_name, fps, frame_idx,
                 score_history, args.threshold, is_two_stream)

        # ── Show / write ──────────────────────────────────────────────────────
        if not args.no_display:
            cv2.imshow("Accident Anticipation — Inference", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n  Interrupted by user.")
                break

        if writer:
            writer.write(display)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  Frame {frame_idx:06d}  score={score:.4f}  fps={fps:.1f}")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"\n  Done. Processed {frame_idx} frames.")
    if args.output:
        print(f"  Saved: {args.output}")


# ═══════════════════════════════════════════════════════════════════════════════
# Interactive menu
# ═══════════════════════════════════════════════════════════════════════════════

def _prompt(prompt_text: str, default: str = "") -> str:
    """Show a prompt and return stripped input; fall back to default if empty."""
    suffix = f" [{default}]" if default else ""
    value  = input(f"{prompt_text}{suffix}: ").strip()
    return value if value else default


def interactive_menu():
    """Collect model, source, and output path via an interactive terminal menu."""

    BORDER = "═" * 60

    print(f"\n{BORDER}")
    print("   Accident Anticipation — Inference Setup")
    print(BORDER)

    # ── 1. Model selection ────────────────────────────────────────────────────
    print("\n  Available models:")
    for i, m in enumerate(AVAILABLE_MODELS, 1):
        tag  = "two-stream" if m in TWO_STREAM_MODELS else "single-stream"
        print(f"    {i}.  {m:<35} [{tag}]")

    default_model_idx = AVAILABLE_MODELS.index("two_stream_transformer") + 1
    while True:
        raw = _prompt("\n  Select model (number or name)", str(default_model_idx))
        if raw.isdigit() and 1 <= int(raw) <= len(AVAILABLE_MODELS):
            model_name = AVAILABLE_MODELS[int(raw) - 1]
            break
        elif raw.lower() in AVAILABLE_MODELS:
            model_name = raw.lower()
            break
        else:
            print(f"  ✗  Invalid choice. Enter 1–{len(AVAILABLE_MODELS)} or a model name.")

    # ── 2. Auto-resolve checkpoint ────────────────────────────────────────────
    checkpoint = _resolve_checkpoint(model_name) 
    
    # ── 3. Source ─────────────────────────────────────────────────────────────
    source = _prompt("\n  Source (video path, frame dir, or webcam index)", "0")

    # ── 4. Output ─────────────────────────────────────────────────────────────
    raw_out = _prompt("\n  Output video path (.mp4)  [leave blank to skip saving]", "")
    output  = raw_out if raw_out else None

    # ── Build namespace ───────────────────────────────────────────────────────
    import types
    args            = types.SimpleNamespace()
    args.model      = model_name
    args.checkpoint = checkpoint
    args.source     = source
    args.output     = output
    args.clip_len   = 16
    args.threshold  = 0.7
    args.min_area   = 800
    args.no_display = False
    return args


if __name__ == "__main__":
    args = interactive_menu()
    run_inference(args)