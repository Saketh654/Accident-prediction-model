"""
infer_cnn_transformer.py

Real-time sliding-window inference with CNNTransformer on a video file.
Applies ImageNet normalization matching the training pipeline.

Usage
─────
Set VIDEO_PATH and CHECKPOINT below, then:
    python inference/infer_cnn_transformer.py
"""

import sys
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import cv2
import torch
import numpy as np
from collections import deque

from models.cnn_transformer import CNNTransformer

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
VIDEO_PATH  = r"data/videos/Crash/000001.mp4"   # ← change this
OUTPUT_PATH = "output_cnn_transformer.avi"
CHECKPOINT  = "checkpoints/cnn_transformer_best.pth"

CLIP_LEN    = 16       # frames per clip (must match training)
THRESHOLD   = 0.5      # risk score decision boundary
K           = 2        # consecutive clips above threshold → alert

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ImageNet normalization — must match training
_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1, 1)

# ─────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────
model = CNNTransformer(
    feature_dim=512, d_model=512, nhead=8,
    num_layers=8, dim_feedforward=2048, dropout=0.1
).to(DEVICE)
model.load_state_dict(
    torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
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
rgb_buffer   = deque(maxlen=CLIP_LEN)
risk_history = deque(maxlen=K)
risk_over_time = []

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
    resized = cv2.resize(frame, (224, 224))

    # BGR → RGB, normalize to [0, 1]
    rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb_buffer.append(rgb_frame)

    risk  = None
    alert = False

    if len(rgb_buffer) == CLIP_LEN:
        # ── Build tensor ──────────────────────────────────────────
        # (T, H, W, 3) → (T, 3, H, W) → apply ImageNet norm → (1, T, 3, H, W)
        rgb_t = torch.from_numpy(
            np.stack(rgb_buffer, axis=0)             # (T, H, W, 3)
        ).permute(0, 3, 1, 2)                        # (T, 3, H, W)

        # Apply ImageNet normalization
        rgb_t = (rgb_t - _MEAN.squeeze(-1)) / _STD.squeeze(-1)

        # CNNTransformer expects (B, T, C, H, W)
        rgb_t = rgb_t.unsqueeze(0).to(DEVICE)        # (1, T, 3, H, W)

        with torch.no_grad():
            with torch.amp.autocast(str(DEVICE), enabled=(DEVICE == "cuda")):
                logits = model(rgb_t)
            risk = torch.sigmoid(logits).item()

        risk_over_time.append(risk)
        risk_history.append(risk)

        if len(risk_history) == K and all(r >= THRESHOLD for r in risk_history):
            alert = True

    # ── Overlay ───────────────────────────────────────────────────
    display = frame.copy()

    if risk is not None:
        bar_color = (
            0,
            int(255 * (1 - risk)),
            int(255 * risk)
        )  # green → red gradient
        cv2.putText(display, f"Risk: {risk:.2f}",
                    (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    bar_color, 2)

    if alert:
        cv2.rectangle(display, (0, 0), (width, 70), (0, 0, 200), -1)
        cv2.putText(display, "ACCIDENT RISK DETECTED",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                    (255, 255, 255), 3)

    # Frame counter
    cv2.putText(display, f"Frame: {frame_count}",
                (20, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (200, 200, 200), 1)

    out.write(display)

# ─────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────
cap.release()
out.release()
np.save("risk_cnn_transformer.npy", np.array(risk_over_time))

print(f"\nInference complete.")
print(f"  Frames processed : {frame_count}")
print(f"  Output video     : {OUTPUT_PATH}")
print(f"  Risk scores saved: risk_cnn_transformer.npy")