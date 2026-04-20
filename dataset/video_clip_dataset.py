import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os

class VideoClipDataset(Dataset):
    """
    PyTorch Dataset for spatiotemporal accident anticipation clips.
    """

    def __init__(self, labels_csv):
        """
        Args:
            labels_csv (str): Path to labels_enhanced_npz.csv
        """
        self.df = pd.read_csv(labels_csv)

    def __len__(self):
        """
        Returns total number of clips
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Returns ONE training sample
        """
        row = self.df.iloc[idx]

        clip_path = row["clip_path"]
        label = row["label"]

        # ----------------------------
        # Load NPZ clip
        # ----------------------------
        data = np.load(clip_path)
        clip = data["clip"]  # (T, H, W, C)

        # ----------------------------
        # Convert to torch tensor
        # ----------------------------
        clip = torch.from_numpy(clip).float()

        # (T, H, W, C) → (C, T, H, W)
        clip = clip.permute(3, 0, 1, 2)

        label = torch.tensor(label, dtype=torch.float32)

        return clip, label
