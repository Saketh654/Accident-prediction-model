"""
two_stream_cnn.py

Two-Stream 3D CNN for spatiotemporal accident anticipation.

Architecture:
    ┌─────────────────────┐    ┌──────────────────────────┐
    │   Spatial Stream    │    │    Temporal Stream        │
    │  (RGB clips 3ch)    │    │  (Optical Flow clips 2ch) │
    │   3D CNN encoder    │    │    3D CNN encoder          │
    └────────┬────────────┘    └─────────────┬────────────┘
             │  feat_s (128)                  │  feat_t (128)
             └───────────────┬───────────────┘
                             │  concat → (256)
                        ┌────┴────┐
                        │  Fusion │  FC 256→128→1
                        └─────────┘

Memory budget for <=8GB VRAM:
    - Batch size 4, clip (3,16,224,224): ~1.4 GB
    - Both streams + gradients: ~5–6 GB total
    - Leaves headroom for OS + CUDA context
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StreamEncoder(nn.Module):
    """
    Shared 3D CNN backbone used for both streams.
    in_channels: 3 for RGB, 2 for optical flow.
    """

    def __init__(self, in_channels: int, base_ch: int = 32):
        super().__init__()

        # Block 1
        self.conv1 = nn.Conv3d(
            in_channels, base_ch,
            kernel_size=(3, 7, 7),
            stride=(1, 2, 2),
            padding=(1, 3, 3),
            bias=False
        )
        self.bn1 = nn.BatchNorm3d(base_ch)

        # Block 2
        self.conv2 = nn.Conv3d(
            base_ch, base_ch * 2,
            kernel_size=(3, 5, 5),
            stride=(1, 2, 2),
            padding=(1, 2, 2),
            bias=False
        )
        self.bn2 = nn.BatchNorm3d(base_ch * 2)

        # Block 3
        self.conv3 = nn.Conv3d(
            base_ch * 2, base_ch * 4,
            kernel_size=(3, 3, 3),
            stride=(2, 2, 2),
            padding=1,
            bias=False
        )
        self.bn3 = nn.BatchNorm3d(base_ch * 4)

        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout3d(p=0.3)

        # Output feature dim = base_ch * 4
        self.out_dim = base_ch * 4

    def forward(self, x):
        # x: (B, C, T, H, W)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool3d(x, kernel_size=(1, 2, 2))

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool3d(x, kernel_size=(1, 2, 2))

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)   # (B, out_dim)

        return x


class TwoStreamCNN(nn.Module):
    """
    Two-Stream 3D CNN.

    Args:
        base_ch (int): Base channel count per stream (default 32 → out 128).
                       Reduce to 16 if you hit OOM on 8GB GPU.
        fusion  (str): 'concat' (default) or 'average'
    """

    def __init__(self, base_ch: int = 32, fusion: str = "concat"):
        super().__init__()

        assert fusion in ("concat", "average"), \
            "fusion must be 'concat' or 'average'"
        self.fusion = fusion

        # ── Two independent stream encoders ──────────────────────────
        self.spatial_stream  = StreamEncoder(in_channels=3, base_ch=base_ch)
        self.temporal_stream = StreamEncoder(in_channels=2, base_ch=base_ch)

        feat_dim = self.spatial_stream.out_dim   # 128 by default

        # ── Fusion head ──────────────────────────────────────────────
        if fusion == "concat":
            fused_dim = feat_dim * 2             # 256
        else:
            fused_dim = feat_dim                 # 128

        self.fusion_head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(128, 1)
        )

    def forward(self, rgb, flow):
        """
        Args:
            rgb  : (B, 3, T, H, W)
            flow : (B, 2, T, H, W)
        Returns:
            logits: (B, 1)
        """
        feat_s = self.spatial_stream(rgb)    # (B, 128)
        feat_t = self.temporal_stream(flow)  # (B, 128)

        if self.fusion == "concat":
            fused = torch.cat([feat_s, feat_t], dim=1)  # (B, 256)
        else:
            fused = (feat_s + feat_t) / 2.0             # (B, 128)

        return self.fusion_head(fused)                   # (B, 1)
