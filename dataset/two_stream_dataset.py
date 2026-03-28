"""
two_stream_dataset.py

On-the-fly PyTorch Dataset for the Two-Stream network.
Reads directly from:
    - Enhanced RGB frames  (frames_enhanced/)
    - PNG optical flow     (optical_flow_png/)

No pre-built NPZ clips needed. Each __getitem__ call:
    1. Reads CLIP_LENGTH RGB frames from disk
    2. Reads CLIP_LENGTH PNG flow files from disk
    3. Decodes, normalizes, and returns tensors

Returns per sample:
    rgb  : (3, T, H, W)  float32  spatial stream input
    flow : (2, T, H, W)  float32  temporal stream input
    label: scalar float32
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd

# Must match value used in generate_optical_flow_png_fast.py
FLOW_CLIP  = 20.0
FRAME_SIZE = (224, 224)
FRAME_EXT  = (".jpg", ".png", ".jpeg")


class TwoStreamDataset(Dataset):

    def __init__(self, labels_csv):
        """
        Args:
            labels_csv (str): Path to labels_two_stream.csv
                              Must have columns:
                              rgb_dir, flow_dir, start, end, label
        """
        self.df = pd.read_csv(labels_csv)

        # Pre-cache sorted frame lists per video to avoid repeated os.listdir
        # Key: rgb_dir → sorted list of frame filenames
        self._frame_cache = {}

    def __len__(self):
        return len(self.df)

    def _get_sorted_frames(self, rgb_dir):
        """Returns cached sorted frame list for a directory."""
        if rgb_dir not in self._frame_cache:
            self._frame_cache[rgb_dir] = sorted([
                f for f in os.listdir(rgb_dir)
                if f.lower().endswith(FRAME_EXT)
            ])
        return self._frame_cache[rgb_dir]

    def _load_rgb(self, path):
        """Loads and normalizes one RGB frame → (H, W, 3) float32 [0,1]."""
        img = cv2.imread(path)
        if img is None:
            return np.zeros((*FRAME_SIZE, 3), dtype=np.float32)
        img = cv2.resize(img, FRAME_SIZE)
        return img.astype(np.float32) / 255.0

    def _load_flow(self, path):
        """
        Loads uint8 PNG flow → (H, W, 2) float32 [-1, 1].

        PNG channel layout (saved by generate_optical_flow_png_fast.py):
            B = dx_u8
            G = dy_u8
            R = magnitude (unused here)

        Decoding:
            dx = (dx_u8 / 255) * 2 * FLOW_CLIP - FLOW_CLIP  → [-FLOW_CLIP, FLOW_CLIP]
            normalize by FLOW_CLIP → [-1, 1]
        """
        if not os.path.exists(path):
            return np.zeros((*FRAME_SIZE, 2), dtype=np.float32)

        img = cv2.imread(path)                      # uint8 (H, W, 3) BGR
        if img is None:
            return np.zeros((*FRAME_SIZE, 2), dtype=np.float32)

        img = cv2.resize(img, FRAME_SIZE)

        dx_u8 = img[..., 0].astype(np.float32)     # B channel = dx
        dy_u8 = img[..., 1].astype(np.float32)     # G channel = dy

        dx = ((dx_u8 / 255.0) * 2 * FLOW_CLIP - FLOW_CLIP) / FLOW_CLIP
        dy = ((dy_u8 / 255.0) * 2 * FLOW_CLIP - FLOW_CLIP) / FLOW_CLIP

        return np.stack([dx, dy], axis=-1)          # (H, W, 2)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        rgb_dir  = row["rgb_dir"]
        flow_dir = row["flow_dir"]
        start    = int(row["start"])
        end      = int(row["end"])
        label    = float(row["label"])

        frame_files = self._get_sorted_frames(rgb_dir)

        rgb_clip  = []
        flow_clip = []

        for i in range(start, end + 1):
            fname = frame_files[i]
            stem  = os.path.splitext(fname)[0]

            rgb_clip.append(
                self._load_rgb(os.path.join(rgb_dir, fname))
            )
            flow_clip.append(
                self._load_flow(os.path.join(flow_dir, stem + ".png"))
            )

        rgb_clip  = np.stack(rgb_clip)              # (T, H, W, 3)
        flow_clip = np.stack(flow_clip)             # (T, H, W, 2)

        # (T, H, W, C) → (C, T, H, W)
        rgb_t  = torch.from_numpy(rgb_clip).permute(3, 0, 1, 2)    # (3, T, H, W)
        flow_t = torch.from_numpy(flow_clip).permute(3, 0, 1, 2)   # (2, T, H, W)

        label_t = torch.tensor(label, dtype=torch.float32)

        return rgb_t, flow_t, label_t
