
import torch
import torch.nn as nn
import torch.nn.functional as F


class TabCNNHead(nn.Module):
    """TabCNN's dense classification head on Kong features."""

    def __init__(self, input_dim=512):
        super().__init__()

        self.dense = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 6 * 21),
        )

    def forward(self, features):
        #[B, T, 512] -> [B, T, 6, 21]
        logits = self.dense(features)
        return logits.reshape(*logits.shape[:-1], 6, 21)


class KongHeadA(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = TabCNNHead(
            input_dim=encoder.feature_dim
        )

    def forward(self, waveform):
        features = self.encoder(waveform)["features"]
        return self.head(features)

    def set_stage(self, stage):
        if stage not in (1, 2):
            raise ValueError("stage must be 1 or 2")

        #stage 1: frozen Kong, train adapter + head
        #stage 2: train all three
        self.encoder.set_backbone_frozen(stage == 1)

    @staticmethod
    def loss(logits, targets):
        #logits: [B, T, 6, 21]
        #targets: [B, T, 6]
        assert logits.shape[:-1] == targets.shape

        return 6 * F.cross_entropy(
            logits.reshape(-1, 21),
            targets.reshape(-1).long()
        )

    @torch.no_grad()
    def predict(self, waveform):
        logits = self.forward(waveform)
        return logits.argmax(dim=-1)
