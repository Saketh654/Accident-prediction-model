"""
main.py — FastAPI backend for Accident Risk Prediction
Place at project root (same level as checkpoints/, models/, inference/)

Run with:
    uvicorn main:app --reload --port 8000

Models:
    3dcnn                  → checkpoints/accident_model.pth
    cnn_lstm               → checkpoints/cnn_lstm_best.pth
    two_stream             → checkpoints/two_stream_best.pth
    cnn_transformer        → checkpoints/cnn_transformer_best.pth
    two_stream_resnet      → checkpoints/two_stream_resnet_best.pth
    two_stream_transformer → checkpoints/two_stream_transformer_best.pth
"""

import os
import sys
import uuid
import tempfile
import numpy as np
from collections import deque

import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import imageio

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# ── Project root on path ───────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from models.accident_3d_cnn       import Accident3DCNN
from models.cnn_lstm               import CNNLSTM
from models.two_stream_cnn         import TwoStreamCNN
from models.cnn_transformer        import CNNTransformer
from models.two_stream_resnet      import TwoStreamCNNRes
from models.two_stream_transformer import TwoStreamTransformer

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="DashGuard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/video", StaticFiles(directory=OUTPUT_DIR), name="video")

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
TEMPERATURE = 3.0   # softens overconfident logits — tune between 2.0–5.0
print(f"[DashGuard] Device: {DEVICE}  |  Temperature: {TEMPERATURE}")


# ── Re-encode to browser-compatible H.264 ─────────────────────────────────────
def remux_to_browser_mp4(raw_path: str, out_path: str):
    """
    Converts OpenCV mp4v video → H.264 + faststart for browser streaming.
    Requires: pip install imageio[ffmpeg]
    """
    try:
        reader = imageio.get_reader(raw_path, format="ffmpeg")
        fps    = reader.get_meta_data().get("fps", 10)
        writer = imageio.get_writer(
            out_path,
            format="ffmpeg",
            fps=fps,
            codec="libx264",
            quality=7,
            macro_block_size=None,
            ffmpeg_params=["-movflags", "+faststart", "-pix_fmt", "yuv420p"]
        )
        for frame in reader:
            writer.append_data(frame)
        reader.close()
        writer.close()
        print(f"[DashGuard] Remuxed → {out_path}")
    finally:
        if os.path.exists(raw_path):
            os.remove(raw_path)


# ── CLAHE enhancement — matches training preprocessing ────────────────────────
def enhance_frame(img):
    """BGR uint8 → BGR uint8 with CLAHE contrast enhancement."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


# ── Overlay helper ─────────────────────────────────────────────────────────────
def draw_overlay(frame, risk, alert, width, height):
    display = frame.copy()
    if risk is not None:
        g = int(255 * (1 - risk))
        r = int(255 * risk)
        cv2.rectangle(display, (12, 12), (230, 46), (20, 20, 20), -1)
        cv2.rectangle(display, (12, 12), (12 + int(210 * risk), 46), (0, g, r), -1)
        cv2.putText(display, f"Risk: {risk:.2f}",
                    (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    if alert:
        cv2.rectangle(display, (0, 0), (width, 64), (0, 0, 200), -1)
        cv2.putText(display, "ACCIDENT RISK DETECTED",
                    (16, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    return display


# ── VideoWriter helper ─────────────────────────────────────────────────────────
def _open_writer(output_path, fps, width, height):
    raw = output_path.replace(".mp4", "_raw.mp4")
    writer = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    return writer, raw


# ── ImageNet normalization constants (for ResNet-based models) ─────────────────
_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1, 1)


# ── Load checkpoint helper ─────────────────────────────────────────────────────
def _load(ckpt_names, model_obj, label):
    """Try loading from a list of checkpoint names, return None if none found."""
    if isinstance(ckpt_names, str):
        ckpt_names = [ckpt_names]
    for name in ckpt_names:
        path = os.path.join(ROOT, "checkpoints", name)
        if os.path.exists(path):
            model_obj.load_state_dict(
                torch.load(path, map_location=DEVICE, weights_only=False)
            )
            model_obj.eval()
            print(f"[DashGuard] {label} loaded ✓  ({name})")
            return model_obj
    print(f"[DashGuard] WARNING: {label} — no checkpoint found. Tried: {ckpt_names}")
    return None


# ── Load all 6 models ──────────────────────────────────────────────────────────

# 1. 3D CNN
_3dcnn_model = _load(
    "accident_model.pth",
    Accident3DCNN().to(DEVICE),
    "3D CNN"
)

# 2. CNN+LSTM
_lstm_backbone = models.resnet18(weights=None)
_lstm_backbone.fc = nn.Identity()
_lstm_model = _load(
    "cnn_lstm_best.pth",
    CNNLSTM(cnn=_lstm_backbone, feature_dim=512,
            hidden_dim=256, num_classes=1).to(DEVICE),
    "CNN+LSTM"
)

# 3. Two-Stream CNN (lightweight 3D CNN streams)
_twostream_model = _load(
    ["two_stream_best.pth", "two_stream_final.pth"],
    TwoStreamCNN(base_ch=32, fusion="concat").to(DEVICE),
    "Two-Stream CNN"
)

# 4. CNN+Transformer
_transformer_model = _load(
    "cnn_transformer_best.pth",
    CNNTransformer(
        feature_dim=512, d_model=512, nhead=8,
        num_layers=8, dim_feedforward=2048, dropout=0.1
    ).to(DEVICE),
    "CNN+Transformer"
)

# 5. Two-Stream ResNet
_twostream_resnet_model = _load(
    "two_stream_resnet_best.pth",
    TwoStreamCNNRes(fusion="concat").to(DEVICE),
    "Two-Stream ResNet"
)

# 6. Two-Stream Transformer
_twostream_transformer_model = _load(
     "two_stream_transformer_final.pth",
    TwoStreamTransformer(
        d_model=256, nhead=8, num_layers=4,
        dim_ff=512, fusion="concat",
        grad_ckpt=False   # disable grad checkpointing at inference
    ).to(DEVICE),
    "Two-Stream Transformer"
)


# ── Inference: 3D CNN ──────────────────────────────────────────────────────────
def infer_3dcnn(input_path: str, output_path: str) -> list:
    if _3dcnn_model is None:
        raise RuntimeError("3D CNN checkpoint not loaded")

    CLIP_LEN = 16; THRESHOLD = 0.5; K = 3
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out, raw = _open_writer(output_path, fps, W, H)

    buf = deque(maxlen=CLIP_LEN); hist = deque(maxlen=K)
    scores = []; fc = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        fc += 1
        norm = enhance_frame(cv2.resize(frame, (224, 224))).astype(np.float32) / 255.0
        buf.append(norm)

        risk = None; alert = False
        if len(buf) == CLIP_LEN:
            clip = torch.from_numpy(
                np.stack(list(buf))
            ).permute(3, 0, 1, 2).unsqueeze(0).to(DEVICE)          # (1,C,T,H,W)
            with torch.no_grad():
                logit = _3dcnn_model(clip)
                risk  = torch.sigmoid(logit / TEMPERATURE).item()
            print(f"[3DCNN] Frame {fc:4d} | logit={logit.item():.3f} | prob={risk:.3f}")
            scores.append(risk); hist.append(risk)
            if len(hist) == K and all(r >= THRESHOLD for r in hist): alert = True

        out.write(draw_overlay(frame, risk, alert, W, H))

    cap.release(); out.release()
    _log_done("3DCNN", scores)
    remux_to_browser_mp4(raw, output_path)
    return scores


# ── Inference: CNN+LSTM ────────────────────────────────────────────────────────
def infer_cnn_lstm(input_path: str, output_path: str) -> list:
    if _lstm_model is None:
        raise RuntimeError("CNN+LSTM checkpoint not loaded")

    CLIP_LEN = 16; THRESHOLD = 0.4; K = 2
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out, raw = _open_writer(output_path, fps, W, H)

    buf = deque(maxlen=CLIP_LEN); hist = deque(maxlen=K)
    scores = []; fc = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        fc += 1
        norm = enhance_frame(cv2.resize(frame, (224, 224))).astype(np.float32) / 255.0
        buf.append(norm)

        risk = None; alert = False
        if len(buf) == CLIP_LEN:
            # CNN+LSTM expects (B, T, C, H, W)
            clip = torch.from_numpy(
                np.stack(list(buf))
            ).permute(0, 3, 1, 2).unsqueeze(0).to(DEVICE)          # (1,T,C,H,W)
            with torch.no_grad():
                logit = _lstm_model(clip)
                risk  = torch.sigmoid(logit / TEMPERATURE).item()
            print(f"[LSTM] Frame {fc:4d} | logit={logit.item():.3f} | prob={risk:.3f}")
            scores.append(risk); hist.append(risk)
            if len(hist) == K and all(r >= THRESHOLD for r in hist): alert = True

        out.write(draw_overlay(frame, risk, alert, W, H))

    cap.release(); out.release()
    _log_done("LSTM", scores)
    remux_to_browser_mp4(raw, output_path)
    return scores


# ── Inference: Two-Stream CNN ──────────────────────────────────────────────────
def infer_two_stream(input_path: str, output_path: str) -> list:
    if _twostream_model is None:
        raise RuntimeError("Two-Stream CNN checkpoint not loaded")

    CLIP_LEN = 16; THRESHOLD = 0.5; K = 2
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out, raw = _open_writer(output_path, fps, W, H)

    rgb_buf = deque(maxlen=CLIP_LEN); flow_buf = deque(maxlen=CLIP_LEN)
    hist = deque(maxlen=K); scores = []; prev_gray = None; fc = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        fc += 1
        resized   = enhance_frame(cv2.resize(frame, (224, 224)))
        curr_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        rgb_buf.append(resized.astype(np.float32) / 255.0)

        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow_buf.append(np.clip(flow, -20, 20) / 20.0)
        else:
            flow_buf.append(np.zeros((224, 224, 2), dtype=np.float32))
        prev_gray = curr_gray

        risk = None; alert = False
        if len(rgb_buf) == CLIP_LEN:
            rgb_t  = torch.from_numpy(np.stack(list(rgb_buf))).permute(3, 0, 1, 2).unsqueeze(0).to(DEVICE)
            flow_t = torch.from_numpy(np.stack(list(flow_buf))).permute(3, 0, 1, 2).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logit = _twostream_model(rgb_t, flow_t)
                risk  = torch.sigmoid(logit / TEMPERATURE).item()
            print(f"[TwoStream] Frame {fc:4d} | logit={logit.item():.3f} | prob={risk:.3f}")
            scores.append(risk); hist.append(risk)
            if len(hist) == K and all(r >= THRESHOLD for r in hist): alert = True

        out.write(draw_overlay(frame, risk, alert, W, H))

    cap.release(); out.release()
    _log_done("TwoStream", scores)
    remux_to_browser_mp4(raw, output_path)
    return scores


# ── Inference: CNN+Transformer ─────────────────────────────────────────────────
def infer_cnn_transformer(input_path: str, output_path: str) -> list:
    if _transformer_model is None:
        raise RuntimeError("CNN+Transformer checkpoint not loaded")

    CLIP_LEN = 16; THRESHOLD = 0.5; K = 2
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out, raw = _open_writer(output_path, fps, W, H)

    buf = deque(maxlen=CLIP_LEN); hist = deque(maxlen=K)
    scores = []; fc = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        fc += 1

        # CNNTransformer: ResNet18 backbone → needs ImageNet normalization + RGB
        resized = cv2.cvtColor(
            enhance_frame(cv2.resize(frame, (224, 224))),
            cv2.COLOR_BGR2RGB
        ).astype(np.float32) / 255.0
        buf.append(resized)

        risk = None; alert = False
        if len(buf) == CLIP_LEN:
            # CNNTransformer expects (B, T, C, H, W)
            clip = torch.from_numpy(np.stack(list(buf))).permute(0, 3, 1, 2)   # (T,C,H,W)
            clip = (clip - _MEAN) / _STD                                         # ImageNet norm
            clip = clip.unsqueeze(0).to(DEVICE)                                  # (1,T,C,H,W)
            with torch.no_grad():
                logit = _transformer_model(clip)
                risk  = torch.sigmoid(logit / TEMPERATURE).item()
            print(f"[Transformer] Frame {fc:4d} | logit={logit.item():.3f} | prob={risk:.3f}")
            scores.append(risk); hist.append(risk)
            if len(hist) == K and all(r >= THRESHOLD for r in hist): alert = True

        out.write(draw_overlay(frame, risk, alert, W, H))

    cap.release(); out.release()
    _log_done("Transformer", scores)
    remux_to_browser_mp4(raw, output_path)
    return scores


# ── Inference: Two-Stream ResNet ───────────────────────────────────────────────
def infer_two_stream_resnet(input_path: str, output_path: str) -> list:
    if _twostream_resnet_model is None:
        raise RuntimeError("Two-Stream ResNet checkpoint not loaded")

    CLIP_LEN = 16; THRESHOLD = 0.5; K = 2
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out, raw = _open_writer(output_path, fps, W, H)

    rgb_buf = deque(maxlen=CLIP_LEN); flow_buf = deque(maxlen=CLIP_LEN)
    hist = deque(maxlen=K); scores = []; prev_gray = None; fc = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        fc += 1
        resized   = enhance_frame(cv2.resize(frame, (224, 224)))
        curr_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # RGB stream: BGR→RGB + ImageNet normalize
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb_buf.append(rgb_frame)

        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow_buf.append(np.clip(flow, -20, 20) / 20.0)
        else:
            flow_buf.append(np.zeros((224, 224, 2), dtype=np.float32))
        prev_gray = curr_gray

        risk = None; alert = False
        if len(rgb_buf) == CLIP_LEN:
            # (T,H,W,3) → (3,T,H,W) then ImageNet normalize
            rgb_t = torch.from_numpy(np.stack(list(rgb_buf))).permute(3, 0, 1, 2)  # (3,T,H,W)
            rgb_t = (rgb_t - _MEAN) / _STD
            rgb_t = rgb_t.unsqueeze(0).to(DEVICE)                                   # (1,3,T,H,W)

            flow_t = torch.from_numpy(
                np.stack(list(flow_buf))
            ).permute(3, 0, 1, 2).unsqueeze(0).to(DEVICE)                           # (1,2,T,H,W)

            with torch.no_grad():
                logit = _twostream_resnet_model(rgb_t, flow_t)
                risk  = torch.sigmoid(logit / TEMPERATURE).item()
            print(f"[TSResNet] Frame {fc:4d} | logit={logit.item():.3f} | prob={risk:.3f}")
            scores.append(risk); hist.append(risk)
            if len(hist) == K and all(r >= THRESHOLD for r in hist): alert = True

        out.write(draw_overlay(frame, risk, alert, W, H))

    cap.release(); out.release()
    _log_done("TSResNet", scores)
    remux_to_browser_mp4(raw, output_path)
    return scores


# ── Inference: Two-Stream Transformer ─────────────────────────────────────────
def infer_two_stream_transformer(input_path: str, output_path: str) -> list:
    if _twostream_transformer_model is None:
        raise RuntimeError("Two-Stream Transformer checkpoint not loaded")

    CLIP_LEN = 16; THRESHOLD = 0.5; K = 2
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out, raw = _open_writer(output_path, fps, W, H)

    rgb_buf = deque(maxlen=CLIP_LEN); flow_buf = deque(maxlen=CLIP_LEN)
    hist = deque(maxlen=K); scores = []; prev_gray = None; fc = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        fc += 1
        resized   = enhance_frame(cv2.resize(frame, (224, 224)))
        curr_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb_buf.append(rgb_frame)

        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow_buf.append(np.clip(flow, -20, 20) / 20.0)
        else:
            flow_buf.append(np.zeros((224, 224, 2), dtype=np.float32))
        prev_gray = curr_gray

        risk = None; alert = False
        if len(rgb_buf) == CLIP_LEN:
            rgb_t = torch.from_numpy(np.stack(list(rgb_buf))).permute(3, 0, 1, 2)
            rgb_t = (rgb_t - _MEAN) / _STD
            rgb_t = rgb_t.unsqueeze(0).to(DEVICE)                                   # (1,3,T,H,W)

            flow_t = torch.from_numpy(
                np.stack(list(flow_buf))
            ).permute(3, 0, 1, 2).unsqueeze(0).to(DEVICE)                           # (1,2,T,H,W)

            with torch.no_grad():
                logit = _twostream_transformer_model(rgb_t, flow_t)
                risk  = torch.sigmoid(logit / TEMPERATURE).item()
            print(f"[TSTrans] Frame {fc:4d} | logit={logit.item():.3f} | prob={risk:.3f}")
            scores.append(risk); hist.append(risk)
            if len(hist) == K and all(r >= THRESHOLD for r in hist): alert = True

        out.write(draw_overlay(frame, risk, alert, W, H))

    cap.release(); out.release()
    _log_done("TSTrans", scores)
    remux_to_browser_mp4(raw, output_path)
    return scores


# ── Logging helper ─────────────────────────────────────────────────────────────
def _log_done(tag, scores):
    if scores:
        print(f"[{tag}] Done | frames={len(scores)} "
              f"min={min(scores):.3f} max={max(scores):.3f} "
              f"mean={sum(scores)/len(scores):.3f}")


# ── Model registry ─────────────────────────────────────────────────────────────
INFERENCE_FNS = {
    "3dcnn":                  infer_3dcnn,
    "cnn_lstm":               infer_cnn_lstm,
    "two_stream":             infer_two_stream,
    "cnn_transformer":        infer_cnn_transformer,
    "two_stream_resnet":      infer_two_stream_resnet,
    "two_stream_transformer": infer_two_stream_transformer,
}


# ── API routes ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "temperature": TEMPERATURE,
        "models_loaded": {
            "3dcnn":                  _3dcnn_model                 is not None,
            "cnn_lstm":               _lstm_model                  is not None,
            "two_stream":             _twostream_model             is not None,
            "cnn_transformer":        _transformer_model           is not None,
            "two_stream_resnet":      _twostream_resnet_model      is not None,
            "two_stream_transformer": _twostream_transformer_model is not None,
        }
    }


@app.post("/analyze")
async def analyze(
    file:       UploadFile = File(...),
    model_name: str        = Form("3dcnn"),
):
    # Validate file type
    allowed = (
        file.content_type.startswith("video/") or
        file.content_type in ("application/octet-stream", "")
    )
    ext = os.path.splitext(file.filename or "")[1].lower()
    if not allowed and ext not in (".mp4", ".avi", ".mov", ".mkv"):
        raise HTTPException(
            400, f"Please upload a video file (got: {file.content_type})"
        )

    if model_name not in INFERENCE_FNS:
        raise HTTPException(
            400,
            f"Unknown model '{model_name}'. "
            f"Valid: {list(INFERENCE_FNS.keys())}"
        )

    # Save upload to temp
    job_id     = uuid.uuid4().hex[:12]
    input_path = os.path.join(
        tempfile.gettempdir(),
        f"dashguard_{job_id}_in{ext or '.mp4'}"
    )
    with open(input_path, "wb") as f:
        f.write(await file.read())

    print(f"[DashGuard] Job {job_id} | model={model_name} | file={file.filename}")

    output_name = f"{job_id}_{model_name}.mp4"
    output_path = os.path.join(OUTPUT_DIR, output_name)

    try:
        risk_scores = INFERENCE_FNS[model_name](input_path, output_path)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(500, f"Inference failed: {e}")
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)

    scores_arr = np.array(risk_scores) if risk_scores else np.array([0.0])

    return {
        "video":       output_name,
        "risk_scores": risk_scores,
        "model_used":  model_name,
        "summary": {
            "peak_risk":       float(scores_arr.max()),
            "avg_risk":        float(scores_arr.mean()),
            "alert_frames":    int((scores_arr >= 0.5).sum()),
            "total_frames":    len(risk_scores),
            "alert_triggered": bool(scores_arr.max() >= 0.5),
        }
    }