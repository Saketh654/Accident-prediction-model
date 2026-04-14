"""
two_stream_cnn.py

Two-Stream CNN for spatiotemporal accident anticipation.

Architecture:
    RGB frames   → ResNet18 (pretrained) → feat_s (512-dim) ┐
                                                              ├─ concat → FC → risk score
    Flow frames  → ResNet18 (2ch adapt)  → feat_t (512-dim) ┘

Memory budget for <=8GB VRAM:
    - Batch size 4, clip (3,16,224,224): ~2.5 GB
    - Both streams + gradients: ~6–7 GB total
    - Reduce BATCH_SIZE to 2 if OOM
"""

import torch
import torch.nn as nn
import torchvision.models as models


class StreamEncoder(nn.Module):
    """
    Per-frame ResNet18 backbone with temporal average pooling.
    in_channels: 3 for RGB, 2 for optical flow.
    """

    def __init__(self, in_channels: int):
        super().__init__()

        if in_channels == 3:
            # RGB stream: full pretrained ResNet18, strip final FC
            backbone = models.resnet18(weights="IMAGENET1K_V1")
            self.encoder = nn.Sequential(*list(backbone.children())[:-1])

        else:
            # Flow stream (2ch): adapt first conv from 3ch to 2ch
            backbone = models.resnet18(weights="IMAGENET1K_V1")
            old_conv = backbone.conv1                  # weight shape: (64, 3, 7, 7)
            new_conv = nn.Conv2d(
                2, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            # Warm-start: take first 2 input channels, scale to preserve activation magnitude
            with torch.no_grad():
                new_conv.weight.copy_(
                    old_conv.weight[:, :2, :, :] * (3.0 / 2.0)
                )
            backbone.conv1 = new_conv
            self.encoder = nn.Sequential(*list(backbone.children())[:-1])

        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = 512

    def forward(self, x):
        """
        Args:
            x: (B, C, T, H, W)
        Returns:
            feat: (B, 512)
        """
        B, C, T, H, W = x.shape

        # Process all frames across batch and time simultaneously
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)  # (B*T, C, H, W)
        feat = self.encoder(x)                                   # (B*T, 512, 1, 1)
        feat = feat.view(B, T, 512).permute(0, 2, 1)            # (B, 512, T)
        feat = self.temporal_pool(feat).squeeze(-1)              # (B, 512)
        return feat


class TwoStreamCNNRes(nn.Module):
    """
    Two-Stream CNN fusing spatial (RGB) and temporal (optical flow) features.

    Args:
        fusion (str): 'concat' (1024-dim → FC) or 'average' (512-dim → FC)
    """

    def __init__(self, fusion: str = "concat"):
        super().__init__()

        assert fusion in ("concat", "average"), \
            "fusion must be 'concat' or 'average'"
        self.fusion = fusion

        self.spatial_stream  = StreamEncoder(in_channels=3)
        self.temporal_stream = StreamEncoder(in_channels=2)

        fused_dim = 1024 if fusion == "concat" else 512

        self.fusion_head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(256, 1)
        )

    def forward(self, rgb, flow):
        """
        Args:
            rgb  : (B, 3, T, H, W)
            flow : (B, 2, T, H, W)
        Returns:
            logits: (B, 1)
        """
        feat_s = self.spatial_stream(rgb)    # (B, 512)
        feat_t = self.temporal_stream(flow)  # (B, 512)

        if self.fusion == "concat":
            fused = torch.cat([feat_s, feat_t], dim=1)  # (B, 1024)
        else:
            fused = (feat_s + feat_t) / 2.0             # (B, 512)

        return self.fusion_head(fused)                   # (B, 1)