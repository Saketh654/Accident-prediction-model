# models/cnn_transformer.py
import torch
import torch.nn as nn
import torchvision.models as models
import math
import sys
import os

ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=64, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, T, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class CNNTransformer(nn.Module):
    """
    ResNet18 per-frame encoder + Transformer encoder for temporal modeling.
    Drop-in replacement for CNNLSTM.
    """

    def __init__(self, feature_dim=512, d_model=256, nhead=8,
                 num_layers=4, dim_feedforward=512, dropout=0.1):   
        super().__init__()

        # ── CNN backbone ──────────────────────────────────────
        backbone = models.resnet18(weights="IMAGENET1K_V1")
        backbone.fc = nn.Identity()
        self.cnn = backbone
        # In CNNTransformer.__init__, after loading backbone:
        for param in self.cnn.parameters():
            param.requires_grad = False
        # Project CNN features to transformer d_model
        self.input_proj = nn.Linear(feature_dim, d_model)

        # ── Positional encoding ───────────────────────────────
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        # ── Transformer encoder ───────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True   # expects (B, T, d_model)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # ── Classification head ───────────────────────────────
        # Learnable [CLS] token — aggregates sequence info
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W)  — same format as CNNLSTM
        Returns:
            logits: (B, 1)
        """
        B, T, C, H, W = x.shape

        # Extract per-frame features
        feat = self.cnn(x.reshape(B * T, C, H, W))   # (B*T, 512)
        feat = feat.view(B, T, -1)                  # (B, T, 512)

        # Project to d_model
        feat = self.input_proj(feat)                # (B, T, d_model)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)      # (B, 1, d_model)
        feat = torch.cat([cls, feat], dim=1)         # (B, T+1, d_model)

        # Add positional encoding
        feat = self.pos_enc(feat)

        # Transformer encoder
        feat = self.transformer(feat)               # (B, T+1, d_model)

        # Classify from CLS token
        cls_out = feat[:, 0]                        # (B, d_model)
        return self.classifier(cls_out)             # (B, 1)