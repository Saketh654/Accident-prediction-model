# models/two_stream_transformer.py
import torch
import torch.nn as nn
from models.two_stream_resnet import StreamEncoder

class CrossStreamAttention(nn.Module):
    """
    Bidirectional cross-attention between RGB and flow feature sequences.
    Each stream attends to the other as key/value.
    """

    def __init__(self, d_model=512, nhead=8, dropout=0.1):
        super().__init__()

        # RGB attends to Flow
        self.rgb_to_flow = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        # Flow attends to RGB
        self.flow_to_rgb = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        self.norm_rgb  = nn.LayerNorm(d_model)
        self.norm_flow = nn.LayerNorm(d_model)

        self.ffn_rgb = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model * 2, d_model)
        )
        self.ffn_flow = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model * 2, d_model)
        )
        self.norm_rgb2  = nn.LayerNorm(d_model)
        self.norm_flow2 = nn.LayerNorm(d_model)

    def forward(self, feat_rgb, feat_flow):
        """
        Args:
            feat_rgb, feat_flow: (B, T, d_model) — per-frame feature sequences
        Returns:
            fused_rgb, fused_flow: (B, T, d_model)
        """
        # RGB queries flow context
        attn_rgb, _ = self.rgb_to_flow(
            query=feat_rgb, key=feat_flow, value=feat_flow
        )
        feat_rgb = self.norm_rgb(feat_rgb + attn_rgb)
        feat_rgb = self.norm_rgb2(feat_rgb + self.ffn_rgb(feat_rgb))

        # Flow queries RGB context
        attn_flow, _ = self.flow_to_rgb(
            query=feat_flow, key=feat_rgb, value=feat_rgb
        )
        feat_flow = self.norm_flow(feat_flow + attn_flow)
        feat_flow = self.norm_flow2(feat_flow + self.ffn_flow(feat_flow))

        return feat_rgb, feat_flow


class TwoStreamTransformer(nn.Module):
    """
    Two-Stream model with cross-attention fusion instead of concatenation.
    """

    def __init__(self, d_model=512, nhead=8, num_layers=2, dropout=0.1):
        super().__init__()

        self.rgb_encoder  = StreamEncoder(in_channels=3)   # from your existing file
        self.flow_encoder = StreamEncoder(in_channels=2)

        # Stack multiple cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            CrossStreamAttention(d_model, nhead, dropout)
            for _ in range(num_layers)
        ])

        # Temporal self-attention after cross-stream fusion
        self_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True
        )
        self.self_attn = nn.TransformerEncoder(self_layer, num_layers=2)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def _get_frame_features(self, encoder, x):
        """
        Run encoder frame-by-frame to get a temporal sequence.
        x: (B, C, T, H, W)
        returns: (B, T, d_model)
        """
        B, C, T, H, W = x.shape
        frames = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        # Use the backbone layers only (strip temporal pool)
        feat = encoder.encoder(frames)          # (B*T, 512, 1, 1)
        feat = feat.view(B, T, 512)
        return feat

    def forward(self, rgb, flow):
        """
        Args:
            rgb  : (B, 3, T, H, W)
            flow : (B, 2, T, H, W)
        Returns:
            logits: (B, 1)
        """
        feat_rgb  = self._get_frame_features(self.rgb_encoder,  rgb)
        feat_flow = self._get_frame_features(self.flow_encoder, flow)

        # Iterative cross-stream attention
        for layer in self.cross_attn_layers:
            feat_rgb, feat_flow = layer(feat_rgb, feat_flow)

        # Fuse by addition, then self-attend over time
        fused = feat_rgb + feat_flow                          # (B, T, 512)

        cls = self.cls_token.expand(rgb.size(0), -1, -1)
        fused = torch.cat([cls, fused], dim=1)                # (B, T+1, 512)
        fused = self.self_attn(fused)

        return self.head(fused[:, 0])                         # (B, 1)