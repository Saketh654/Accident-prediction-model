import torch
import torch.nn as nn
import torch.nn.functional as F

class Accident3DCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # -------------------------
        # Block 1: Low-level motion + edges
        # -------------------------
        self.conv1 = nn.Conv3d(
            in_channels=3,
            out_channels=32,
            kernel_size=(3, 7, 7),
            stride=(1, 2, 2),
            padding=(1, 3, 3)
        )
        self.bn1 = nn.BatchNorm3d(32)

        # -------------------------
        # Block 2: Mid-level spatiotemporal features
        # -------------------------
        self.conv2 = nn.Conv3d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 5, 5),
            stride=(1, 2, 2),
            padding=(1, 2, 2)
        )
        self.bn2 = nn.BatchNorm3d(64)

        # -------------------------
        # Block 3: High-level motion patterns
        # -------------------------
        self.conv3 = nn.Conv3d(
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 3, 3),
            stride=(2, 2, 2),
            padding=1
        )
        self.bn3 = nn.BatchNorm3d(128)

        # -------------------------
        # Global pooling (key design choice)
        # -------------------------
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # -------------------------
        # Classification head
        # -------------------------
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        """
        x shape: (B, 3, 16, 224, 224)
        """

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool3d(x, kernel_size=(1, 2, 2))

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool3d(x, kernel_size=(1, 2, 2))

        x = F.relu(self.bn3(self.conv3(x)))

        # (B, 128, T, H, W) → (B, 128, 1, 1, 1)
        x = self.global_pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Output logits (NO sigmoid)
        x = self.fc(x)

        return x
