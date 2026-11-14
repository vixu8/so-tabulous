"""Small trainable feature-fusion adapter for the pretrained Kong note model.

No guitar head, loss, or training loop is defined here. A downstream model can
train this adapter jointly with its own head while keeping Kong frozen.
"""

from collections.abc import Mapping

import torch
from torch import Tensor, nn


BRANCH_NAMES = ("frame", "onset", "offset", "velocity")


class ResidualKongFusion(nn.Module):
    """Add a learned correction from the four CRNNs to Kong's fused-frame GRU.

    Args:
        feature_dim: Dimension of the original fused-frame representation (512).
        hidden_dim: Bottleneck width for the new MLP adapter.
        dropout: Adapter dropout applied between its two linear layers.
        initial_gate_logit: Initial sigmoid gate logit; -2 is ~0.119.

    Shapes:
        pretrained_features: [B, T, 512]
        branch_features: mapping of four [B, T, 512] tensors
        output: [B, T, 512]

    The final linear projection is initialized to zero, so the first forward
    pass gives *exactly* the pretrained features (before any adapter training).
    The gate and early MLP layers gain nonzero gradients after the final layer
    starts learning; this is intentional, not a frozen adapter.
    """

    def __init__(
        self,
        feature_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        initial_gate_logit: float = -2.0,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or hidden_dim <= 0:
            raise ValueError("feature_dim and hidden_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.feature_dim = feature_dim
        self.adapter = nn.Sequential(
            nn.LayerNorm(feature_dim * len(BRANCH_NAMES)),
            nn.Linear(feature_dim * len(BRANCH_NAMES), hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        self.gate_logit = nn.Parameter(torch.tensor(float(initial_gate_logit)))

    @property
    def gate(self) -> Tensor:
        """Learned residual correction amplitude, between 0 and 1."""
        return torch.sigmoid(self.gate_logit)

    def forward(
        self,
        pretrained_features: Tensor,
        branch_features: Mapping[str, Tensor],
    ) -> Tensor:
        if pretrained_features.ndim != 3 or pretrained_features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"pretrained_features must be [B, T, {self.feature_dim}]; "
                f"got {tuple(pretrained_features.shape)}"
            )
        missing = [name for name in BRANCH_NAMES if name not in branch_features]
        if missing:
            raise KeyError(f"Missing branch features: {missing}")
        for name in BRANCH_NAMES:
            if branch_features[name].shape != pretrained_features.shape:
                raise ValueError(
                    f"branch_features[{name!r}] must match "
                    f"{tuple(pretrained_features.shape)}; got "
                    f"{tuple(branch_features[name].shape)}"
                )
        all_branches = torch.cat([branch_features[n] for n in BRANCH_NAMES], dim=-1)
        correction = self.adapter(all_branches)
        return pretrained_features + self.gate * correction
