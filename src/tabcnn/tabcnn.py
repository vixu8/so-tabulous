"""TabCNN: a CNN that classifies 6 fret-string positions per audio frame.

PyTorch port of the original Keras model (Wiggins & Kim, "Guitar Tablature
Estimation with a Convolutional Neural Network", ISMIR 2019).

Architecture (unchanged from the original):
    input (B, 1, H, W)   H = freq bins (192 for CQT), W = con_win_size
    Conv2d(1  -> 32, 3x3, valid) + ReLU
    Conv2d(32 -> 64, 3x3, valid) + ReLU
    Conv2d(64 -> 64, 3x3, valid) + ReLU
    MaxPool2d(2x2)
    Dropout(0.25)
    Flatten
    Linear(-> 128) + ReLU          (LazyLinear: in-features inferred on first call)
    Dropout(0.5)
    Linear(-> 6 * 21)
    reshape -> (B, 6, 21) logits   (softmax per string happens in the loss / predict)

Keras used channels-last inputs (H, W, 1); here inputs are channels-first
(B, 1, H, W). The per-string softmax cross-entropy loss (``tab_loss``) and the
``avg_acc`` metric reproduce the original ``catcross_by_string`` /
``softmax_by_string`` / ``avg_acc`` exactly.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

NUM_STRINGS = 6
NUM_CLASSES = 21  # 0 = string not played, 1..20 = fret 0..19

# Frequency-bin height per spectral representation, matching the original.
SPEC_REPR_BINS = {"c": 192, "m": 128, "cm": 320, "s": 1025}


class TabCNN(nn.Module):
    def __init__(
        self,
        num_strings: int = NUM_STRINGS,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()
        self.num_strings = num_strings
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),   # 'valid' padding (PyTorch default)
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.LazyLinear(128),  # flatten size inferred from the first forward pass
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_strings * num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, 1, H, W) -> logits (B, num_strings, num_classes)."""
        if x.ndim != 4:
            raise ValueError(f"Expected input (B, 1, H, W); got {tuple(x.shape)}")
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.shape[0], self.num_strings, self.num_classes)


def tab_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Per-string softmax cross-entropy, summed over strings and averaged over
    the batch.

    Reproduces the original ``catcross_by_string`` composed with
    ``softmax_by_string``, but applies ``log_softmax`` to raw logits for
    numerical stability instead of taking the log of softmax probabilities.

    Args:
        logits:  (B, num_strings, num_classes) raw scores from ``TabCNN``.
        targets: (B, num_strings, num_classes) per-string one-hot labels.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    per_string = -(targets * log_probs).sum(dim=-1)  # (B, num_strings)
    return per_string.sum(dim=1).mean()


@torch.no_grad()
def avg_acc(logits: Tensor, targets: Tensor) -> Tensor:
    """Mean over strings and batch of (argmax prediction == argmax label)."""
    correct = logits.argmax(dim=-1) == targets.argmax(dim=-1)
    return correct.float().mean()


@torch.no_grad()
def predict_proba(model: TabCNN, x: Tensor) -> Tensor:
    """Per-string softmax probabilities (B, num_strings, num_classes)."""
    was_training = model.training
    model.eval()
    probs = F.softmax(model(x), dim=-1)
    model.train(was_training)
    return probs
