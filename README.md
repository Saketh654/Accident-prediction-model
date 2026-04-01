# Accident Anticipation System

A spatiotemporal deep learning pipeline for predicting road accidents from dashcam footage before they occur. Three models are implemented and compared: a 3D CNN, a CNN+LSTM, and a Two-Stream 3D CNN with optical flow.

---

## Project Overview

The system processes dashcam video clips and assigns a risk score between 0 and 1. A score above the threshold triggers an alert. The pipeline covers the full ML lifecycle — data collection, preprocessing, frame enhancement, model training, inference, and evaluation.

---

## Models

| Model | Architecture | Temporal Modelling | Input |
|---|---|---|---|
| `Accident3DCNN` | 3D CNN | Joint spatiotemporal convolution | RGB clips |
| `CNNLSTM` | ResNet18 + LSTM | LSTM over per-frame features | RGB clips |
| `TwoStreamCNN` | Dual 3D CNN | Explicit optical flow stream | RGB + Flow |

---

## Project Structure

```
your_project/
│
├── data/
│   ├── videos/
│   │   ├── crash/                          # original crash videos
│   │   └── normal/                         # original normal videos
│   ├── frames/
│   │   ├── crash/                          # raw extracted frames
│   │   └── normal/
│   ├── frames_enhanced/
│   │   ├── crash/                          # CLAHE + denoised frames
│   │   └── normal/
│   ├── optical_flow_png/
│   │   ├── crash/                          # uint8 PNG flow maps
│   │   └── normal/
│   ├── processed/
│   │   ├── labels_enhanced_npz.csv         # labels for 3DCNN + CNNLSTM
│   │   └── labels_two_stream.csv           # labels for Two-Stream
│   └── excels/
│       ├── predicted_accident_frames.csv
│       └── final_accident_frames.csv
│
├── preprocessing/
│   ├── frame_extraction.py
│   ├── frame_enhancement.py
│   ├── accident_frame_prediction.py
│   ├── accident_frames_finalisation.py
│   ├── generate_spatiotemporal_clips_enhanced_npz.py
│   ├── generate_optical_flow_png.py
│   └── generate_two_stream_labels.py
│
├── dataset/
│   ├── video_clip_dataset.py
│   ├── dataloader.py
│   ├── two_stream_dataset.py
│   └── two_stream_dataloader.py
│
├── models/
│   ├── accident_3d_cnn.py
│   ├── cnn_lstm.py
│   └── two_stream_cnn.py
│
├── inference/
│   ├── run_on_video.py
│   ├── infer_cnn_lstm.py
│   └── infer_two_stream.py
│
├── evaluation/
│   ├── plot_risk_crash.py
│   ├── plot_risk_normal.py
│   ├── plot_training_loss.py
│   └── evaluate_two_stream.py
│
├── checkpoints/
│   ├── accident_model.pth
│   ├── cnn_lstm_best.pth
│   └── two_stream_best.pth
│
├── train.py
├── train_cnn_lstm.py
├── train_two_stream.py
└── fix_labels_paths.py
```

---

## Full Run Order

### Phase 1 — Data Preprocessing (Run Once)

```bash
# Step 1: Extract frames from videos
python preprocessing/frame_extraction.py

# Step 2: Enhance frames (CLAHE + denoising + sharpening)
python preprocessing/frame_enhancement.py

# Step 3: Auto-detect accident frames using motion scores
python preprocessing/accident_frame_prediction.py

# Step 4: Human verification of accident frames (interactive)
python preprocessing/accident_frames_finalisation.py

# Step 5: Generate NPZ clips for 3DCNN and CNNLSTM
python preprocessing/generate_spatiotemporal_clips_enhanced_npz.py

# Step 6: Generate optical flow PNGs for Two-Stream (multiprocessing)
python preprocessing/generate_optical_flow_png.py

# Step 7: Generate labels CSV for Two-Stream (no NPZ clips needed)
python preprocessing/generate_two_stream_labels.py
```

### Phase 2 — Training

```bash
# Train 3D CNN
python train.py

# Train CNN+LSTM
python train_cnn_lstm.py

# Train Two-Stream (must use __main__ guard on Windows)
python train_two_stream.py
```

### Phase 3 — Inference

```bash
# 3D CNN inference on video
python inference/run_on_video.py

# CNN+LSTM inference on video
python inference/infer_cnn_lstm.py

# Two-Stream inference on video (flow computed on-the-fly)
python inference/infer_two_stream.py
```

### Phase 4 — Evaluation

```bash
# Plot risk curves
python evaluation/plot_risk_crash.py
python evaluation/plot_risk_normal.py

# Plot training loss
python evaluation/plot_training_loss.py

# Full Two-Stream evaluation: AUC, F1, ROC curve, model comparison
python evaluation/evaluate_two_stream.py
```

---

## Label Assignment (Soft Labels)

Crash clips are assigned soft labels based on how close the clip end is to the accident frame:

| Time to crash | Label |
|---|---|
| ≤ 1.0 sec | 1.0 |
| 1.0 – 1.5 sec | 0.8 |
| 1.5 – 2.0 sec | 0.6 |
| > 2.0 sec | 0.0 |

Normal clips are always labelled `0.0`.

---

## Two-Stream Pipeline

The Two-Stream model uses a separate spatial stream (RGB) and temporal stream (optical flow) which are fused at the end.

```
RGB frames   → 3D CNN → feat_s (128-dim)  ┐
                                            ├─ concat → FC → risk score
Flow PNGs    → 3D CNN → feat_t (128-dim)  ┘
```

Optical flow is saved as uint8 PNG (15–30 KB per frame vs ~400 KB for float32).  
Encoding: `dx, dy ∈ [-20, 20]` mapped to `[0, 255]`.  
At inference, flow is computed on-the-fly using Farneback dense optical flow.

---

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU VRAM | 6 GB | 8 GB |
| CPU | 6 cores | 10+ cores |
| RAM | 16 GB | 32 GB |
| Storage | SSD (frames on internal SSD) | NVMe SSD |

### Training Performance (i7-12650H + 8GB GPU)

| Setting | Value |
|---|---|
| `BATCH_SIZE` | 8 (reduce to 4 if OOM) |
| `NUM_WORKERS` | 8 |
| `prefetch_factor` | 3 |
| AMP (mixed precision) | Enabled |
| GPU utilisation | ~50–70% |

> **Note:** If GPU utilisation is low, ensure `frames_enhanced` and `optical_flow_png` are on the internal SSD, not an external drive.

---

## Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python numpy pandas tqdm scikit-learn matplotlib
```

---

## Checkpoints

| File | Model | Saved by |
|---|---|---|
| `checkpoints/accident_model.pth` | 3D CNN | `train.py` |
| `checkpoints/cnn_lstm_best.pth` | CNN+LSTM | `train_cnn_lstm.py` |
| `checkpoints/two_stream_best.pth` | Two-Stream | `train_two_stream.py` |

---

## Output Files

| File | Description |
|---|---|
| `output_with_alert.avi` | Annotated video — 3D CNN |
| `output_cnn_lstm_alert.avi` | Annotated video — CNN+LSTM |
| `output_two_stream_alert.avi` | Annotated video — Two-Stream |
| `risk_normal.npy` | Risk score curve — 3D CNN |
| `risk_cnn_lstm.npy` | Risk score curve — CNN+LSTM |
| `risk_two_stream.npy` | Risk score curve — Two-Stream |
| `two_stream_loss_history.npy` | Epoch losses for Two-Stream |
| `roc_two_stream.png` | ROC curve — Two-Stream |
| `risk_comparison.png` | All 3 models risk curves overlaid |
