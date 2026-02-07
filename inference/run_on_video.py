import os
import cv2
import torch
import numpy as np
from collections import deque

from models.accident_3d_cnn import Accident3DCNN

# -----------------------
# CONFIG
# -----------------------
VIDEO_PATH = r"L:\Accident Prediction\data\videos\crash\sample.mp4"
OUTPUT_PATH = "output_with_alert.mp4"

CLIP_LEN = 16
THRESHOLD = 0.7
K = 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# Load model
# -----------------------
model = Accident3DCNN().to(DEVICE)
model.load_state_dict(torch.load("accident_model.pth", map_location=DEVICE))
model.eval()

# -----------------------
# Video IO
# -----------------------
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# -----------------------
# Sliding window buffer
# -----------------------
frame_buffer = deque(maxlen=CLIP_LEN)
risk_history = deque(maxlen=K)

frame_idx = 0

# -----------------------
# Inference loop
# -----------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    # Resize & normalize (same as training)
    resized = cv2.resize(frame, (224, 224))
    resized = resized.astype(np.float32) / 255.0

    frame_buffer.append(resized)

    alert = False
    risk = None

    if len(frame_buffer) == CLIP_LEN:
        clip = np.stack(frame_buffer, axis=0)  # (T,H,W,C)
        clip = torch.from_numpy(clip).permute(3, 0, 1, 2).unsqueeze(0)
        clip = clip.to(DEVICE)

        with torch.no_grad():
            logit = model(clip)
            risk = torch.sigmoid(logit).item()

        risk_history.append(risk)

        if len(risk_history) == K and all(r >= THRESHOLD for r in risk_history):
            alert = True

    # -----------------------
    # Overlay text
    # -----------------------
    display = frame.copy()

    if risk is not None:
        cv2.putText(
            display,
            f"Risk: {risk:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

    if alert:
        cv2.putText(
            display,
            "⚠️ ACCIDENT RISK!",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 0, 255),
            3
        )

    out.write(display)

# -----------------------
# Cleanup
# -----------------------
cap.release()
out.release()
print("✅ Inference complete. Output saved:", OUTPUT_PATH)
