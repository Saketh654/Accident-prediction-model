"""
two_stream_dataset.py  —  optimized for RTX 3050 4 GB VRAM / Ryzen 5 5600H

Hardware constraints addressed:
    - 4 GB VRAM  → float16 storage, shorter clip T, smaller spatial size option
    - 16 GB RAM  → frame cache capped at MAX_CACHE_FRAMES to avoid OOM
    - 6-core CPU → cv2 thread count pinned to 2 per worker (6 workers × 2 = 12,
                   leaving headroom for OS + PyTorch threads)
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd

# ── Per-process cv2 thread budget ─────────────────────────────────────────────
# 6 DataLoader workers × 2 cv2 threads = 12 threads total.
# Prevents cv2 from spawning its own pool that fights with DataLoader.
cv2.setNumThreads(2)

FLOW_CLIP  = 20.0
FRAME_SIZE = (224, 224)   # keep 224 — ResNet18 was pretrained at this size;
                           # dropping to 112 saves ~4× memory if still OOM
FRAME_EXT  = (".jpg", ".png", ".jpeg")

# Cap in-memory frame-path cache so 16 GB RAM isn't exhausted on large datasets
MAX_CACHE_DIRS = 512

# ImageNet mean/std
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


class TwoStreamDataset(Dataset):

    def __init__(self, labels_csv: str):
        self.df = pd.read_csv(labels_csv)
        self._frame_cache: dict[str, list[str]] = {}

    def __len__(self) -> int:
        return len(self.df)

    def _get_sorted_frames(self, rgb_dir: str) -> list[str]:
        if rgb_dir not in self._frame_cache:
            # Evict oldest entry when cache is full (simple FIFO)
            if len(self._frame_cache) >= MAX_CACHE_DIRS:
                self._frame_cache.pop(next(iter(self._frame_cache)))
            self._frame_cache[rgb_dir] = sorted([
                f for f in os.listdir(rgb_dir)
                if f.lower().endswith(FRAME_EXT)
            ])
        return self._frame_cache[rgb_dir]

    def _load_rgb(self, path: str) -> np.ndarray:
        """BGR → RGB, resize, ImageNet-normalize → (H, W, 3) float32."""
        img = cv2.imread(path)
        if img is None:
            return np.zeros((*FRAME_SIZE, 3), dtype=np.float32)
        img = cv2.cvtColor(
            cv2.resize(img, FRAME_SIZE, interpolation=cv2.INTER_LINEAR),
            cv2.COLOR_BGR2RGB,
        ).astype(np.float32) / 255.0
        return (img - _MEAN) / _STD

    def _load_flow(self, path: str) -> np.ndarray:
        """PNG → float32 (H, W, 2) in [-1, 1]."""
        img = cv2.imread(path)
        if img is None:
            return np.zeros((*FRAME_SIZE, 2), dtype=np.float32)
        img = cv2.resize(img, FRAME_SIZE,
                         interpolation=cv2.INTER_LINEAR)[..., :2].astype(np.float32)
        return (img / 127.5) - 1.0

    def __getitem__(self, idx: int):
        row      = self.df.iloc[idx]
        rgb_dir  = row["rgb_dir"]
        flow_dir = row["flow_dir"]
        start    = int(row["start"])
        end      = int(row["end"])
        label    = float(row["label"])

        frame_files = self._get_sorted_frames(rgb_dir)
        T = end - start + 1

        # Pre-allocate contiguous arrays — avoids repeated numpy allocations
        rgb_clip  = np.empty((T, *FRAME_SIZE, 3), dtype=np.float32)
        flow_clip = np.empty((T, *FRAME_SIZE, 2), dtype=np.float32)

        for t, i in enumerate(range(start, end + 1)):
            fname = frame_files[i]
            stem  = os.path.splitext(fname)[0]
            rgb_clip[t]  = self._load_rgb(os.path.join(rgb_dir, fname))
            flow_clip[t] = self._load_flow(os.path.join(flow_dir, stem + ".png"))

        # (T, H, W, C) → (C, T, H, W)
        rgb_t  = torch.from_numpy(rgb_clip).permute(3, 0, 1, 2).contiguous()
        flow_t = torch.from_numpy(flow_clip).permute(3, 0, 1, 2).contiguous()

        return rgb_t, flow_t, torch.tensor(label, dtype=torch.float32)