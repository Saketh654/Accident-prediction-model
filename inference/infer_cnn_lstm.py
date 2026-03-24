import cv2
import torch
import numpy as np
from collections import deque
import torchvision.models as models
import torch.nn as nn

from models.cnn_lstm import CNNLSTM

torch.backends.cudnn.benchmark = True
# -----------------------
# CONFIG
# -----------------------
VIDEO_PATH = r"C:\Users\lohit\Downloads\000096.mp4"
OUTPUT_PATH = "output_cnn_lstm_alert.avi"

CLIP_LEN = 16
THRESHOLD = 0.2      # 🔥 CALIBRATED
K = 2                # 🔥 TEMPORAL CONSISTENCY

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------
# Load model
# -----------------------
def get_cnn_backbone():
    cnn = models.resnet18(weights=None)
    feature_dim = cnn.fc.in_features
    cnn.fc = nn.Identity()
    return cnn, feature_dim


cnn, feature_dim = get_cnn_backbone()

model = CNNLSTM(
    cnn=cnn,
    feature_dim=feature_dim,
    hidden_dim=256,
    num_classes=1
).to(DEVICE)

model.load_state_dict(
    torch.load("checkpoints/cnn_lstm_best.pth", map_location=DEVICE)
)

model.eval()


# -----------------------
# Video IO
# -----------------------
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 10
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"MJPG")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))


# -----------------------
# Buffers
# -----------------------
frame_buffer = deque(maxlen=CLIP_LEN)
risk_history = deque(maxlen=K)
risk_over_time = []


# -----------------------
# Inference loop
# -----------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    resized = cv2.resize(frame, (224, 224))
    resized = resized.astype(np.float32) / 255.0
    frame_buffer.append(resized)

    risk = None
    alert = False

    if len(frame_buffer) == CLIP_LEN:
        clip = np.stack(frame_buffer, axis=0)
        clip = torch.from_numpy(clip).permute(0, 3, 1, 2)
        clip = clip.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(clip)
            risk = torch.sigmoid(logits).item()

        print(f"Risk: {risk:.3f}")
        risk_over_time.append(risk)
        risk_history.append(risk)

        if len(risk_history) == K and all(r >= THRESHOLD for r in risk_history):
            alert = True

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
np.save("risk_cnn_lstm.npy", np.array(risk_over_time))

print("✅ CNN+LSTM inference complete.")
