"""
infer_two_stream_resnet.py

Real-time inference with the Two-Stream ResNet model on a video file.
Computes optical flow on-the-fly — no pre-saved flow needed.

Key difference from training: ImageNet normalization applied here
since the ResNet backbone expects it.
"""

import sys
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import cv2
import torch
import numpy as np
from collections import deque

from models.two_stream_resnet import TwoStreamCNNRes

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
VIDEO_PATH  = r"C:\Users\lohit\Downloads\000096.mp4"   # ← change this
OUTPUT_PATH = "output_two_stream_resnet.avi"
CHECKPOINT  = "checkpoints/two_stream_resnet_final.pth"

CLIP_LEN    = 16
THRESHOLD   = 0.5
K           = 2        # consecutive clips above threshold → alert

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ImageNet normalization (must match training)
_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1, 1)

# ─────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────
model = TwoStreamCNNRes(fusion="concat").to(DEVICE)
model.load_state_dict(
    torch.load(CHECKPOINT, map_location=DEVICE)
)
model.eval()
print(f"Loaded  : {CHECKPOINT}")
print(f"Device  : {DEVICE}")

# ─────────────────────────────────────────────
# Video I/O
# ─────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

fps    = cap.get(cv2.CAP_PROP_FPS) or 10
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"MJPG")
out    = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# ─────────────────────────────────────────────
# Buffers
# ─────────────────────────────────────────────
rgb_buffer     = deque(maxlen=CLIP_LEN)
flow_buffer    = deque(maxlen=CLIP_LEN)
risk_history   = deque(maxlen=K)
risk_over_time = []

prev_gray = None


def compute_flow_frame(prev_gray, curr_gray):
    """Dense optical flow → normalized to [-1, 1], shape (H, W, 2)."""
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    return np.clip(flow, -20, 20) / 20.0


# ─────────────────────────────────────────────
# Inference loop
# ─────────────────────────────────────────────
frame_count = 0
print("Running inference... (press Ctrl+C to stop early)\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    resized   = cv2.resize(frame, (224, 224))
    curr_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # RGB: BGR→RGB, normalize to [0,1]
    rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb_buffer.append(rgb_frame)

    # Flow
    if prev_gray is not None:
        flow_frame = compute_flow_frame(prev_gray, curr_gray)
    else:
        flow_frame = np.zeros((224, 224, 2), dtype=np.float32)
    flow_buffer.append(flow_frame)
    prev_gray = curr_gray

    risk  = None
    alert = False

    if len(rgb_buffer) == CLIP_LEN:
        # ── Build tensors ─────────────────────────────────────────
        # (T, H, W, 3) → (3, T, H, W)
        rgb_t = torch.from_numpy(
            np.stack(rgb_buffer, axis=0)
        ).permute(3, 0, 1, 2)                         # (3, T, H, W)

        # Apply ImageNet normalization (must match training dataset)
        rgb_t = (rgb_t - _MEAN) / _STD

        # (T, H, W, 2) → (2, T, H, W)
        flow_t = torch.from_numpy(
            np.stack(flow_buffer, axis=0)
        ).permute(3, 0, 1, 2)                         # (2, T, H, W)

        rgb_t  = rgb_t.unsqueeze(0).to(DEVICE)        # (1, 3, T, H, W)
        flow_t = flow_t.unsqueeze(0).to(DEVICE)       # (1, 2, T, H, W)

        with torch.no_grad():
            logits = model(rgb_t, flow_t)
            risk   = torch.sigmoid(logits).item()

        risk_over_time.append(risk)
        risk_history.append(risk)

        if len(risk_history) == K and all(r >= THRESHOLD for r in risk_history):
            alert = True

    # ── Overlay ───────────────────────────────────────────────────
    display = frame.copy()

    if risk is not None:
        bar_color = (
            (0, int(255 * (1 - risk)), int(255 * risk))   # green→red gradient
        )
        cv2.putText(display, f"Risk: {risk:.2f}",
                    (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    bar_color, 2)

    if alert:
        # Red banner across the top
        cv2.rectangle(display, (0, 0), (width, 70), (0, 0, 200), -1)
        cv2.putText(display, "ACCIDENT RISK DETECTED",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                    (255, 255, 255), 3)

    # Frame counter (bottom left)
    cv2.putText(display, f"Frame: {frame_count}",
                (20, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (200, 200, 200), 1)

    out.write(display)

# ─────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────
cap.release()
out.release()
np.save("risk_two_stream_resnet.npy", np.array(risk_over_time))

print(f"\nInference complete.")
print(f"  Frames processed : {frame_count}")
print(f"  Output video     : {OUTPUT_PATH}")
print(f"  Risk scores saved: risk_two_stream_resnet.npy")