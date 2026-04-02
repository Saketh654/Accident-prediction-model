"""
infer_two_stream.py

Real-time inference with the Two-Stream model on a video file.
Computes optical flow on-the-fly — no pre-saved flow needed.
"""

import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


import cv2
import torch
import numpy as np
from collections import deque

from models.two_stream_cnn import TwoStreamCNN

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
VIDEO_PATH  = r"C:\Users\lohit\Downloads\000096.mp4"
OUTPUT_PATH = "output_two_stream_alert.avi"

CLIP_LEN    = 16
THRESHOLD   = 0.5      # tune after evaluating on val set
K           = 2        # consecutive frames above threshold → alert

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────
model = TwoStreamCNN(base_ch=32, fusion="concat").to(DEVICE)
model.load_state_dict(
    torch.load("checkpoints/two_stream_best.pth", map_location=DEVICE)
)
model.eval()

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
flow_buffer  = deque(maxlen=CLIP_LEN)
risk_history = deque(maxlen=K)
risk_over_time = []

prev_gray = None   # for on-the-fly flow computation


def compute_flow_frame(prev_gray, curr_gray):
    """Dense optical flow, normalized to [-1,1]."""
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    return np.clip(flow, -20, 20) / 20.0   # (H, W, 2)


# ─────────────────────────────────────────────
# Inference loop
# ─────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    resized      = cv2.resize(frame, (224, 224))
    resized_f32  = resized.astype(np.float32) / 255.0
    curr_gray    = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    rgb_buffer.append(resized_f32)   # (H, W, 3)

    # Compute flow if we have a previous frame
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
        rgb_clip  = np.stack(rgb_buffer,  axis=0)   # (T,H,W,3)
        flow_clip = np.stack(flow_buffer, axis=0)   # (T,H,W,2)

        rgb_t  = torch.from_numpy(rgb_clip).permute(3, 0, 1, 2)   # (3,T,H,W)
        flow_t = torch.from_numpy(flow_clip).permute(3, 0, 1, 2)  # (2,T,H,W)

        rgb_t  = rgb_t.unsqueeze(0).to(DEVICE)    # (1,3,T,H,W)
        flow_t = flow_t.unsqueeze(0).to(DEVICE)   # (1,2,T,H,W)

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
        cv2.putText(display, f"Risk: {risk:.2f}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 255), 2)

    if alert:
        cv2.putText(display, "⚠️ ACCIDENT RISK!",
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    (0, 0, 255), 3)

    out.write(display)

# ─────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────
cap.release()
out.release()
np.save("risk_two_stream.npy", np.array(risk_over_time))
print(f"✅ Two-Stream inference complete → {OUTPUT_PATH}")
