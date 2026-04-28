# DashGuard — Dashcam Accident Risk Prediction

> Real-time spatiotemporal accident anticipation from dashcam footage using six deep learning models, a FastAPI backend, and a React frontend.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Prerequisites](#3-prerequisites)
4. [Environment Setup](#4-environment-setup)
5. [Dataset Download (CCD)](#5-dataset-download-ccd)
6. [Dataset Preprocessing](#6-dataset-preprocessing)
7. [Training the Models](#7-training-the-models)
8. [Evaluating the Models](#8-evaluating-the-models)
9. [Running the Application](#9-running-the-application)
10. [Project Structure](#10-project-structure)
11. [Reproducing Our Results](#11-reproducing-our-results)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Project Overview

DashGuard predicts accident risk from dashcam video clips using spatiotemporal deep learning. It supports six model architectures:

| Model ID | Architecture | Description |
|---|---|---|
| `3dcnn` | Accident 3D CNN | Fast, lightweight 3D convolutional baseline |
| `cnn_lstm` | CNN + LSTM | ResNet18 frame encoder with LSTM temporal modelling |
| `two_stream` | Two-Stream CNN | Separate spatial (RGB) and temporal (optical flow) 3D CNN streams |
| `cnn_transformer` | CNN + Transformer | ResNet18 encoder with Transformer attention over time |
| `two_stream_resnet` | Two-Stream ResNet | Two-stream network with pretrained ResNet18 backbones |
| `two_stream_transformer` | Two-Stream Transformer | Best model — ResNet18 + Transformer per stream, fused for prediction |

---

## 2. Architecture

```
dashcam video
      │
      ▼
[FastAPI Backend — main.py]
      │
      ├── frame extraction + CLAHE enhancement
      ├── optical flow (Farneback) computed on-the-fly
      ├── sliding window (16 frames, stride 1)
      ├── model inference (one of 6 models)
      └── annotated MP4 output + JSON risk scores
            │
            ▼
[React Frontend — accident-frontend/]
      ├── drag-and-drop video upload
      ├── model selector
      ├── risk score timeline chart (Chart.js)
      └── annotated video playback
```

---

## 3. Prerequisites

### Hardware (minimum)

- GPU: NVIDIA GPU with ≥ 4 GB VRAM (tested on RTX 3050 4 GB and RTX 4060 Laptop 8 GB)
- RAM: 16 GB recommended
- Disk: ~50 GB free for dataset, frames, flow, and clips

### Software

| Requirement | Version |
|---|---|
| Python | 3.10 or 3.11 |
| CUDA Toolkit | 11.8 or 12.1 (must match your PyTorch build) |
| Node.js | 20.x or 22.x (required by Vite 8 / React 19) |
| npm | 10.x+ |
| Git | any recent version |
| ffmpeg | required for video re-encoding (`imageio[ffmpeg]`) |

> **Windows users:** Install ffmpeg and add it to `PATH`. Download from https://ffmpeg.org/download.html or via `winget install ffmpeg`.

---

## 4. Environment Setup

### 4.1 Clone the repository

```bash
git clone <your-repo-url>
cd <repo-root>
```

### 4.2 Create a Python virtual environment

```bash
# Create environment
python -m venv venv

# Activate — Windows
venv\Scripts\activate

# Activate — Linux / macOS
source venv/bin/activate
```

### 4.3 Install PyTorch (GPU)

Visit https://pytorch.org/get-started/locally/ and select your CUDA version. Example for CUDA 12.1:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

For CUDA 11.8:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Verify your installation:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

Expected output: `True  12.1` (or your CUDA version).

### 4.4 Install Python dependencies

```bash
pip install fastapi uvicorn python-multipart
pip install opencv-python-headless numpy pandas tqdm
pip install scikit-learn matplotlib seaborn
pip install imageio[ffmpeg]
pip install torchvision
```

Full requirements in one command:

```bash
pip install fastapi uvicorn python-multipart opencv-python-headless \
    numpy pandas tqdm scikit-learn matplotlib seaborn imageio[ffmpeg]
```

### 4.5 Install frontend dependencies

```bash
cd accident-frontend
npm install
cd ..
```

> Node.js 20+ and npm 10+ are required. Check with `node --version` and `npm --version`.

---

## 5. Dataset Download (CCD)

The project uses the **Car Crash Dataset (CCD)**, which contains dashcam footage of crash and normal driving scenarios.

### 5.1 Download from Google Drive

Open the following link in your browser and download the dataset:

**[CCD Dataset — Google Drive](https://drive.google.com/drive/folders/1NUwC-bkka0-iPqhEhIgsXWtj0DA2MR-F?usp=sharing)**

Download both the `Crash` and `Normal` video folders.

### 5.2 Organise the dataset

After downloading, place the videos in the following structure:

```
<repo-root>/
└── data/
    └── videos/
        ├── crash/
        │   ├── 000001.mp4
        │   ├── 000002.mp4
        │   └── ...
        └── normal/
            ├── 000001.mp4
            ├── 000002.mp4
            └── ...
```

Create the folders if they do not exist:

```bash
mkdir -p data/videos/crash
mkdir -p data/videos/normal
mkdir -p data/excels
```

Move your downloaded crash videos into `data/videos/crash/` and normal videos into `data/videos/normal/`.

---

## 6. Dataset Preprocessing

Run the following scripts **in order**. Each step builds on the output of the previous one.

### Step 1 — Extract frames

Extracts individual JPEG frames from all videos at their native FPS.

```bash
python preprocessing/frame_extraction.py
```

Output directories:
- `data/frames/crash/<video_id>/frame_00000.jpg ...`
- `data/frames/normal/<video_id>/frame_00000.jpg ...`

### Step 2 — Detect accident frames (automated)

Estimates the accident frame in each crash video using motion score analysis.

```bash
python preprocessing/accident_frame_prediction.py
```

Output: `data/excels/predicted_accident_frames.csv`

### Step 3 — Verify and correct accident frames (manual)

Opens each crash video interactively so you can confirm or correct the predicted accident frame.

```bash
python preprocessing/accident_frames_finalisation.py
```

**Controls during review:**
- `d` → advance to next frame
- `a` → go back one frame
- `q` → close the frame window
- At the prompt: `y` = correct, `n` = enter the correct frame number, `q` = quit the program

Output: `data/excels/final_accident_frames.csv`

> **Note:** This step is labour-intensive. Our team divided the work across members. If you want to skip manual verification and use automated labels only, copy `predicted_accident_frames.csv` to `final_accident_frames.csv` and add a `Validity` column with value `"Verified"` for all rows.

### Step 4 — CLAHE frame enhancement

Applies Contrast Limited Adaptive Histogram Equalization to all frames. This step is required before generating clips.

Create the enhanced frames directory and run the enhancement script. The enhanced frame directories are expected at:
- `data/frames_enhanced/crash/<video_id>/`
- `data/frames_enhanced/normal/<video_id>/`

If you do not have a dedicated enhancement script, you can apply CLAHE inline. The preprocessing pipeline assumes these directories exist and will skip videos that do not.

> This step was performed as part of our preprocessing pipeline. If your frame extraction script already applies CLAHE, you can symlink or copy `data/frames/` to `data/frames_enhanced/`.

### Step 5 — Generate optical flow PNGs

Computes dense Farneback optical flow between consecutive frames and saves as compressed PNG files. Uses all available CPU cores.

```bash
python preprocessing/generate_optical_flow_png.py
```

Output:
- `data/optical_flow_png/crash/<video_id>/<stem>.png`
- `data/optical_flow_png/normal/<video_id>/<stem>.png`

> This step takes 25–60 minutes depending on your CPU. Progress is shown per video folder.

### Step 6 — Generate spatiotemporal NPZ clips

Produces 16-frame sliding-window clips in compressed `.npz` format for the 3D CNN, CNN+LSTM, and CNN+Transformer models.

```bash
python preprocessing/generate_spatiotemporal_clips_enhanced_npz.py
```

Output:
- `data/processed/clips_enhanced_npz/crash/*.npz`
- `data/processed/clips_enhanced_npz/normal/*.npz`
- `data/processed/labels_enhanced_npz.csv`

### Step 7 — Generate two-stream labels CSV

Creates the index CSV for the Two-Stream models (no NPZ files written — frames are loaded on-the-fly during training).

```bash
python preprocessing/generate_two_stream_labels.py
```

Output: `data/processed/labels_two_stream.csv`

### Step 8 — Split dataset into train / val / test

Performs a video-level stratified split (70% train, 15% val, 15% test) to prevent data leakage.

```bash
python preprocessing/split_dataset.py
```

Output (in `data/processed/`):
- `labels_enhanced_npz_train.csv`
- `labels_enhanced_npz_val.csv`
- `labels_enhanced_npz_test.csv`
- `labels_two_stream_train.csv`
- `labels_two_stream_val.csv`
- `labels_two_stream_test.csv`

---

## 7. Training the Models

All training scripts save checkpoints to the `checkpoints/` directory. The best validation-loss checkpoint is kept.

Create the checkpoint directory first:

```bash
mkdir -p checkpoints
```

### 7.1 3D CNN (fastest to train)

```bash
python train.py
```

Checkpoint: `checkpoints/accident_model.pth`

### 7.2 CNN + LSTM

```bash
python train_cnn_lstm.py
```

Checkpoint: `checkpoints/cnn_lstm_best.pth`

### 7.3 CNN + Transformer

```bash
python train_cnn_transformer.py
```

Checkpoint: `checkpoints/cnn_transformer_best.pth`

### 7.4 Two-Stream CNN

```bash
python train_two_stream.py
```

Checkpoint: `checkpoints/two_stream_best.pth`

### 7.5 Two-Stream ResNet

```bash
python train_two_stream_resnet.py
```

Checkpoints:
- `checkpoints/two_stream_resnet_best.pth` (best val loss)
- `checkpoints/two_stream_resnet_final.pth` (final epoch)

### 7.6 Two-Stream Transformer (recommended — best performance)

```bash
python train_two_stream_transformer.py
```

Checkpoints:
- `checkpoints/two_stream_transformer_best.pth`
- `checkpoints/two_stream_transformer_final.pth`

**Optional training flags:**

```bash
python train_two_stream_transformer.py \
    --epochs 15 \
    --batch-size 16 \
    --d-model 256 \
    --nhead 8 \
    --num-layers 4 \
    --compile         # torch.compile for ~20% speedup (adds ~3 min first epoch)
```

### Training time estimates (RTX 3050 4 GB)

| Model | Batch Size | Epochs | Approximate Time |
|---|---|---|---|
| 3D CNN | 4 | 10 | ~2 hours |
| CNN + LSTM | 16 | 10 | ~3 hours |
| CNN + Transformer | 32 | 10 | ~4 hours |
| Two-Stream CNN | 6 | 10 | ~3 hours |
| Two-Stream ResNet | 4 | 10 | ~5 hours |
| Two-Stream Transformer | 16 | 15 | ~8 hours |

---

## 8. Evaluating the Models

### Evaluate all six models on the held-out test set

```bash
python evaluation/evaluate_all_models.py
```

This script:
- Loads each model from its checkpoint
- Runs inference on the test CSVs
- Prints a metrics comparison table (AUC, Accuracy, Precision, Recall, F1)
- Saves `metrics_summary_test.csv`, `roc_comparison_test.png`, and `cm_comparison_test.png`

### Evaluate individual models

```bash
# 3D CNN only
python evaluation/New\ folder/evaluate_3d_cnn.py

# CNN + LSTM only
python evaluation/New\ folder/evaluate_cnn_lstm.py

# Two-Stream CNN only
python evaluation/New\ folder/evaluate_two_stream.py

# Two-Stream ResNet only
python evaluation/New\ folder/evaluate_two_stream_resnet.py
```

---

## 9. Running the Application

Both the backend and frontend must be running simultaneously.

### 9.1 Start the FastAPI backend

From the project root (with the virtual environment active):

```bash
uvicorn main:app --reload --port 8000
```

The backend will:
- Load all six models from `checkpoints/` on startup (models with missing checkpoints are skipped with a warning)
- Serve the API at `http://localhost:8000`
- Save annotated output videos to `outputs/`

Verify the backend is healthy:

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{
  "status": "ok",
  "device": "cuda",
  "models_loaded": {
    "3dcnn": true,
    "cnn_lstm": true,
    ...
  }
}
```

### 9.2 Start the React frontend

In a separate terminal:

```bash
cd accident-frontend
npm run dev
```

The frontend will be available at **http://localhost:5173**

### 9.3 Using the application

1. Open http://localhost:5173 in your browser
2. Drag and drop a dashcam video file (MP4, AVI, or MOV) onto the upload zone
3. Select one of the six models from the model selector
4. Click **Analyze Video →**
5. Wait for inference to complete (progress bar shown)
6. View the results: annotated video, risk score timeline chart, and model interpretation

---

## 10. Project Structure

```
<repo-root>/
├── accident-frontend/          # React + Vite frontend
│   ├── src/
│   │   ├── components/         # DropZone, Header, Footer, RiskChart, ModelSelector
│   │   ├── hooks/              # useAnalyze — upload + inference hook
│   │   ├── pages/              # HomePage, ResultPage
│   │   ├── styles/             # CSS modules
│   │   └── utils/              # helpers.js — formatting, model lists
│   └── package.json
│
├── checkpoints/                # Saved model weights (created after training)
│
├── data/                       # Dataset (created during preprocessing)
│   ├── videos/crash/           # Raw crash videos
│   ├── videos/normal/          # Raw normal videos
│   ├── frames/                 # Extracted JPEG frames
│   ├── frames_enhanced/        # CLAHE-enhanced frames
│   ├── optical_flow_png/       # Farneback flow as PNG
│   ├── excels/                 # Annotation CSVs
│   └── processed/              # NPZ clips and label CSVs
│
├── dataset/                    # PyTorch Dataset and DataLoader classes
│   ├── video_clip_dataset.py
│   ├── dataloader.py
│   ├── two_stream_dataset.py
│   ├── two_stream_resnet_dataset.py
│   └── ...
│
├── evaluation/                 # Evaluation scripts and metrics
│   ├── evaluate_all_models.py
│   └── New folder/             # Per-model evaluation scripts
│
├── inference/                  # Standalone inference scripts
│   ├── inference.py            # Unified multi-model inference with HUD
│   ├── infer_cnn_lstm.py
│   ├── infer_two_stream_resnet.py
│   └── ...
│
├── models/                     # Model architecture definitions
│   ├── accident_3d_cnn.py
│   ├── cnn_lstm.py
│   ├── cnn_transformer.py
│   ├── two_stream_cnn.py
│   ├── two_stream_resnet.py
│   └── two_stream_transformer.py
│
├── outputs/                    # Annotated output videos (created by backend)
│
├── preprocessing/              # Data preparation scripts
│   ├── frame_extraction.py
│   ├── accident_frame_prediction.py
│   ├── accident_frames_finalisation.py
│   ├── generate_optical_flow_png.py
│   ├── generate_spatiotemporal_clips_enhanced_npz.py
│   ├── generate_two_stream_labels.py
│   └── split_dataset.py
│
├── main.py                     # FastAPI backend
├── train.py                    # 3D CNN training
├── train_cnn_lstm.py
├── train_cnn_transformer.py
├── train_two_stream.py
├── train_two_stream_resnet.py
└── train_two_stream_transformer.py
```

---

## 11. Reproducing Our Results

Follow these steps in exact order to reproduce the evaluation numbers reported in the project.

### Step-by-step reproduction checklist

```
[ ] 1. Download CCD dataset and place videos in data/videos/crash/ and data/videos/normal/
[ ] 2. Run: python preprocessing/frame_extraction.py
[ ] 3. Run: python preprocessing/accident_frame_prediction.py
[ ] 4. Run: python preprocessing/accident_frames_finalisation.py  (manual verification)
[ ] 5. Apply CLAHE enhancement to data/frames/ → data/frames_enhanced/
[ ] 6. Run: python preprocessing/generate_optical_flow_png.py
[ ] 7. Run: python preprocessing/generate_spatiotemporal_clips_enhanced_npz.py
[ ] 8. Run: python preprocessing/generate_two_stream_labels.py
[ ] 9. Run: python preprocessing/split_dataset.py
[ ] 10. Train all six models (see Section 7)
[ ] 11. Run: python evaluation/evaluate_all_models.py
```

### Expected checkpoint filenames

The evaluation script looks for these exact filenames in `checkpoints/`:

| Model | Checkpoint File |
|---|---|
| 3D CNN | `accident_model.pth` |
| CNN + LSTM | `cnn_lstm_best.pth` |
| CNN + Transformer | `cnn_transformer_best.pth` |
| Two-Stream CNN | `two_stream_best.pth` |
| Two-Stream ResNet | `two_stream_resnet_final.pth` |
| Two-Stream Transformer | `two_stream_transformer_best.pth` |

### Random seed

The dataset split uses a fixed random seed (`RANDOM_SEED = 42` in `preprocessing/split_dataset.py`). Do not change this seed if you want to reproduce the exact train/val/test splits used in our evaluation.

---

## 12. Troubleshooting

### CUDA out of memory

Reduce the batch size in the relevant training script. For the Two-Stream ResNet on a 4 GB GPU, use `BATCH_SIZE = 4` with `GRAD_ACCUM = 4` (already set as default). For the Two-Stream Transformer on 8 GB, use `--batch-size 8` if 16 causes OOM.

### Video writer fails on Windows

Make sure `ffmpeg` is installed and on your `PATH`. The backend uses `imageio[ffmpeg]` to re-encode output videos to H.264 for browser playback. Run `ffmpeg -version` to verify.

### Frontend cannot connect to backend

Ensure the backend is running on port 8000 (`uvicorn main:app --port 8000`) and the frontend is on port 5173. The frontend is hardcoded to `http://localhost:8000`. CORS is configured to allow `http://localhost:5173`.

### `No module named 'models'` during training

Make sure you are running training scripts from the **project root directory**, not from a subdirectory:

```bash
# Correct
cd <repo-root>
python train.py

# Wrong
cd checkpoints
python ../train.py
```

### Optical flow generation is slow

The flow script automatically uses `cpu_count() - 2` workers. On a 6-core CPU this is 4 workers. You can increase `NUM_WORKERS` in `preprocessing/generate_optical_flow_png.py` if your system has more cores and sufficient RAM (each worker holds frame data in memory).

### `ModuleNotFoundError: No module named 'ultralytics'`

The `inference/inference.py` unified inference script uses YOLOv8 for object detection overlays. This dependency is optional and only required for that script. Install it with:

```bash
pip install ultralytics
```

The FastAPI backend (`main.py`) does **not** require `ultralytics`.

---

## License

This project is for academic research purposes. The CCD dataset is subject to its own terms of use. Please refer to the dataset source for licensing information.

---

## Acknowledgements

- Car Crash Dataset (CCD) — provided via Google Drive
- PyTorch and torchvision for model training infrastructure
- Chart.js for risk score visualization
- FastAPI for the inference backend
- Vite + React for the frontend interface