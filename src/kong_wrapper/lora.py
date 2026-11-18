"""LoRA and fine-tuning modes for the shared Kong encoder.

    from lora import TunableKongEncoder, lora_state_dict

    encoder = TunableKongEncoder.from_checkpoint(ckpt, use_adapter=False)  # starts frozen
    encoder.set_finetune("lora", rank=8, alpha=16)    # or "frozen" / "full"
    opt = torch.optim.AdamW([*encoder.trainable_parameters(), *head.parameters()], lr=1e-4)
    ...
    torch.save(lora_state_dict(encoder.note_model), "lora.pt")      # A and B only

Modes (set_finetune):
  "frozen"  Kong fixed: no gradients, no_grad forward (KongEncoder's frozen mode).
  "lora"    Low-rank updates W + (alpha / rank) * B @ A on every nn.Linear weight
            and every nn.GRU weight_ih / weight_hh (biases untouched); only A and
            B train. B starts at zero, so adding LoRA doesn't change Kong's outputs.
  "full"    Every Kong weight trains (merge or drop LoRA first).
In "lora" and "full", Kong follows train()/eval() (dropout on when training, as
in Kong's own training), except BatchNorm, which always keeps its pretrained
statistics: a LoRA checkpoint (A / B only) then fully describes the model, and
batches of 4-8 clips would give noisy statistics. (Kong's in-place dropout bug at
models.py:149 is fixed when kong_encoder.py loads the model.)

Use set_finetune on this class, not set_backbone_frozen.

Heads, the decoder and the fusion adapter are small and task-specific: they
train fully on top of a LoRA'd Kong (decided 2026-11-17). add_lora works on any
module, but nothing applies it to them.

LoRA is implemented with torch.nn.utils.parametrize rather than peft (peft can't
wrap nn.GRU). On a parametrized GRU, torch re-reads the weights every forward;
outputs match a GRU with merged weights exactly (tests/test_kong_lora.py).
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn.utils import parametrize

from kong_encoder import KongEncoder

FINETUNE_MODES = ("frozen", "lora", "full")
LORA_KINDS = ("linear", "gru")


class LoRADelta(nn.Module):
    """Parametrization W -> W + (alpha / rank) * B @ A, with B = 0 at the start."""

    def __init__(self, out_features: int, in_features: int, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be >= 1")
        self.rank, self.alpha, self.scale = rank, alpha, alpha / rank
        self.A = nn.Parameter(torch.empty(rank, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))   # same init as nn.Linear

    def forward(self, weight):
        return weight + self.scale * (self.B @ self.A)

    def extra_repr(self):
        return f"out={self.B.shape[0]}, in={self.A.shape[1]}, rank={self.rank}, alpha={self.alpha}"


def has_lora(module: nn.Module) -> bool:
    return any(isinstance(m, LoRADelta) for m in module.modules())


def lora_targets(module: nn.Module, kinds=LORA_KINDS) -> list[tuple[str, str]]:
    """[(submodule name, weight name)] that add_lora would adapt."""
    unknown = set(kinds) - set(LORA_KINDS)
    if unknown:
        raise ValueError(f"unknown LoRA kinds {sorted(unknown)}; choose from {LORA_KINDS}")
    targets = []
    for name, sub in module.named_modules():
        if isinstance(sub, LoRADelta) or isinstance(sub, parametrize.ParametrizationList):
            continue
        if "linear" in kinds and isinstance(sub, nn.Linear):
            targets.append((name, "weight"))
        elif "gru" in kinds and isinstance(sub, nn.GRU):
            targets += [(name, w) for w in sub._flat_weights_names if w.startswith("weight_")]
    return targets


def add_lora(module: nn.Module, rank: int = 8, alpha: float = 16.0, kinds=LORA_KINDS) -> list[str]:
    """Attach a LoRADelta to every target weight in `module`. Returns their names."""
    if has_lora(module):
        raise ValueError("module already has LoRA; merge_lora() or rebuild it first")
    names = []
    for sub_name, weight_name in lora_targets(module, kinds):
        sub = module.get_submodule(sub_name)
        weight = getattr(sub, weight_name)
        delta = LoRADelta(weight.shape[0], weight.shape[1], rank, alpha)
        parametrize.register_parametrization(
            sub, weight_name, delta.to(device=weight.device, dtype=weight.dtype))
        names.append(f"{sub_name}.{weight_name}" if sub_name else weight_name)
    return names


def _lora_modules(module: nn.Module):
    return [(name, m) for name, m in module.named_modules() if isinstance(m, LoRADelta)]


def lora_parameters(module: nn.Module) -> list[nn.Parameter]:
    return [p for _, m in _lora_modules(module) for p in (m.A, m.B)]


def lora_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    """Only the LoRA A / B matrices (a few MB), keyed by their module path."""
    out = {}
    for name, m in _lora_modules(module):
        out[f"{name}.A"] = m.A.detach().cpu().clone()
        out[f"{name}.B"] = m.B.detach().cpu().clone()
    return out


def load_lora_state_dict(module: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Load A / B into a module that already has LoRA with the same targets and rank."""
    mods = dict(_lora_modules(module))
    expected = {f"{n}.{p}" for n in mods for p in ("A", "B")}
    if set(state) != expected:
        missing, extra = sorted(expected - set(state)), sorted(set(state) - expected)
        raise KeyError(f"LoRA state mismatch; missing {missing[:3]}, unexpected {extra[:3]}")
    with torch.no_grad():
        for key, value in state.items():
            name, which = key.rsplit(".", 1)
            getattr(mods[name], which).copy_(value)


def merge_lora(module: nn.Module) -> int:
    """Fold B @ A into the weights and remove the LoRA parametrizations. Returns how many."""
    merged = 0
    for sub in list(module.modules()):
        if not parametrize.is_parametrized(sub):
            continue
        for weight_name in list(sub.parametrizations.keys()):
            if any(isinstance(p, LoRADelta) for p in sub.parametrizations[weight_name]):
                parametrize.remove_parametrizations(sub, weight_name, leave_parametrized=True)
                merged += 1
    return merged


class TunableKongEncoder(KongEncoder):
    """KongEncoder with frozen / LoRA / full fine-tuning modes (see module docstring)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.finetune = "frozen"
        self.set_finetune("frozen" if self.backbone_frozen else "full")

    def set_finetune(self, mode: str, rank: int = 8, alpha: float = 16.0,
                     kinds=LORA_KINDS) -> TunableKongEncoder:
        """Switch mode. rank / alpha / kinds only matter when LoRA is first added."""
        if mode not in FINETUNE_MODES:
            raise ValueError(f"mode must be one of {FINETUNE_MODES}, got {mode!r}")
        if mode == "frozen":
            self.set_backbone_frozen(True)
        elif mode == "lora":
            if not has_lora(self.note_model):
                add_lora(self.note_model, rank=rank, alpha=alpha, kinds=kinds)
            self.set_backbone_frozen(False)          # gradients flow through Kong
            self.note_model.requires_grad_(False)    # ... but only LoRA weights update
            for p in lora_parameters(self.note_model):
                p.requires_grad_(True)
        else:
            if has_lora(self.note_model):
                raise ValueError("full fine-tuning with LoRA attached; call "
                                 "merge_lora(encoder.note_model) first")
            self.set_backbone_frozen(False)
        self.finetune = mode
        if mode != "frozen":
            self._freeze_batchnorm()
        return self

    def train(self, mode: bool = True) -> TunableKongEncoder:
        super().train(mode)          # frozen: Kong in eval; otherwise Kong follows `mode`
        if getattr(self, "finetune", "frozen") != "frozen":
            self._freeze_batchnorm()
        return self

    def _freeze_batchnorm(self) -> None:
        """Keep every Kong BatchNorm on its pretrained running statistics."""
        for m in self.note_model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Kong parameters that train in the current mode (the adapter is separate)."""
        return [p for p in self.note_model.parameters() if p.requires_grad]
