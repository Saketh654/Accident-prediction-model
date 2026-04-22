import torch
import torch.nn as nn


class CNNLSTM(nn.Module):

    def __init__(self, cnn, feature_dim, hidden_dim, num_classes):
        super().__init__()

        self.cnn = cnn

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        x: (B, T, C, H, W)
        """

        B, T, C, H, W = x.shape

        # reshape so CNN processes all frames at once
        x = x.view(B * T, C, H, W)

        features = self.cnn(x)          # (B*T, F)

        features = features.view(B, T, -1)  # (B, T, F)

        lstm_out, _ = self.lstm(features)

        final_feature = torch.mean(lstm_out, dim=1)
        final_feature = self.dropout(final_feature)
        out = self.fc(final_feature)

        return out