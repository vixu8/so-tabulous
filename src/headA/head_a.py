"""Head A: TabCNN's dense head on Kong's (adapter-fused) features -> fret class per string.

    encoder = TunableKongEncoder.from_checkpoint(ckpt, use_adapter=True)   # starts frozen
    model = KongHeadA(encoder)
    model.set_stage(1)                                  # or 2: LoRA on Kong
    logits = model(batch["waveform"])                   # [B, 160000] -> [B, 1001, 6, 21]
    loss = tab_loss(logits, batch["tab"], batch["n_valid_frames"])

Classes per string follow the loader's `tab` layout (TabCNN's): 0 = string silent,
fret + 1 = that fret sounding (frets 0..19 -> 21 classes).

Stage 1: Kong frozen; the fusion adapter and the head train.
Stage 2: LoRA on Kong (src/kong_wrapper/lora.py); adapter and head train fully.
The loss is the tab loss alone in both stages: Head A reads `features`, not Kong's
piano outputs, so Kong's own loss isn't needed to keep anything meaningful.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Keep in sync with src/data/loader.py.
N_STRINGS = 6
N_CLASSES = 21                    # silent + frets 0..19
IGNORE_INDEX = -100


class TabCNNHead(nn.Module):
    """TabCNN's dense classification head on Kong features."""

    def __init__(self, input_dim=512):
        super().__init__()

        self.dense = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, N_STRINGS * N_CLASSES),
        )

    def forward(self, features):
        # [B, T, 512] -> [B, T, 6, 21]
        logits = self.dense(features)
        return logits.reshape(*logits.shape[:-1], N_STRINGS, N_CLASSES)


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

    def set_stage(self, stage, rank=8, alpha=16.0):
        """Stage 1: Kong frozen (adapter + head train). Stage 2: LoRA on Kong (+ adapter + head)."""
        if stage not in (1, 2):
            raise ValueError("stage must be 1 or 2")
        if not hasattr(self.encoder, "set_finetune"):
            raise TypeError("set_stage needs a TunableKongEncoder (src/kong_wrapper/lora.py)")
        if stage == 1:
            self.encoder.set_finetune("frozen")
        else:
            self.encoder.set_finetune("lora", rank=rank, alpha=alpha)

    @staticmethod
    def loss(logits, targets):
        # logits: [B, T, 6, 21]
        # targets: [B, T, 6], IGNORE_INDEX frames don't count
        assert logits.shape[:-1] == targets.shape

        return N_STRINGS * F.cross_entropy(
            logits.reshape(-1, N_CLASSES),
            targets.reshape(-1).long(),
            ignore_index=IGNORE_INDEX,
        )

    @torch.no_grad()
    def predict(self, waveform):
        logits = self.forward(waveform)
        return logits.argmax(dim=-1)


def tab_loss(logits: Tensor, tab: Tensor, n_valid: Tensor) -> Tensor:
    """KongHeadA.loss with frames >= n_valid (window padding) ignored."""
    frames = torch.arange(tab.shape[1], device=tab.device)
    pad = frames[None, :] >= n_valid.to(tab.device)[:, None]                   # (B, T)
    return KongHeadA.loss(logits, tab.long().masked_fill(pad[..., None], IGNORE_INDEX))


@torch.no_grad()
def predict_tab_grid(probs: Tensor, threshold: float | None = None) -> Tensor:
    """Class probabilities (..., 6, 21) -> tab classes (..., 6): 0 silent, fret + 1 sounding.

    threshold None: plain argmax (KongHeadA.predict). Otherwise a string sounds
    when 1 - p(silent) >= threshold, at its most likely fret.
    """
    if threshold is None:
        return probs.argmax(-1)
    fret = probs[..., 1:].argmax(-1) + 1
    return torch.where(1 - probs[..., 0] >= threshold, fret, torch.zeros_like(fret))
