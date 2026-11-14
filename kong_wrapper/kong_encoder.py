"""Kong note-transcription backbone with accessible intermediate audio features.

Use the original bytedance/piano_transcription PyTorch source and note weights.
This wrapper DOES NOT replace or retrain pretrained Kong layers on creation.

Inputs: normalized floating-point mono waveform [B, samples] at 16,000 Hz.
Output dict:
    features:             [B, T, 512]  (residual-adapted or unmodified)
    pretrained_features:  [B, T, 512]  (output of Kong's frame_gru)
    branch_features:      frame/onset/offset/velocity -> [B, T, 512]
    refined_onset_features: [B, T, 512] (output of Kong's reg_onset_gru)
    piano:                frame_output, reg_onset_output, reg_offset_output,
                          velocity_output -> each [B, T, 88] probabilities

T is floor(samples / 160) + 1 (for standard input lengths); see README.
"""

from __future__ import annotations

import importlib.util
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from fusion_adapter import BRANCH_NAMES, ResidualKongFusion

SAMPLE_RATE = 16_000
FRAMES_PER_SECOND = 100
PIANO_PITCHES = 88
FEATURE_DIM = 512


@contextmanager
def _in_sys_path(*directories: Path):
    """Temporarily enable original repo's absolute imports (pytorch_utils)."""
    inserted = []
    try:
        for directory in directories:
            value = str(directory)
            if value not in sys.path:
                sys.path.insert(0, value)
                inserted.append(value)
        yield
    finally:
        for value in inserted:
            sys.path.remove(value)


def _make_upstream_note_model(repo_path: str | Path) -> nn.Module:
    """Import the official models.py without modifying or copying Kong source."""
    root = Path(repo_path).expanduser().resolve()
    model_file = root / "pytorch" / "models.py"
    if not model_file.is_file():
        raise FileNotFoundError(
            f"Expected the original Kong source at {model_file}. "
            "Clone https://github.com/bytedance/piano_transcription there "
            "or pass repo_path=..."
        )

    spec = importlib.util.spec_from_file_location("_bytedance_kong_models", model_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Kong module from {model_file}")
    module = importlib.util.module_from_spec(spec)
    # Official models.py imports `pytorch_utils`, which in turn imports
    # `utilities`; both are top-level imports in that repository.
    try:
        with _in_sys_path(root / "pytorch", root / "utils"):
            spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        raise ImportError(
            f"Missing dependency {exc.name!r} while importing the Kong repo. "
            "Install requirements.txt from this wrapper, and check that "
            "the original repo exists at repo_path."
        ) from exc
    return module.Regress_onset_offset_frame_velocity_CRNN(
        frames_per_second=FRAMES_PER_SECOND, classes_num=PIANO_PITCHES
    )


def _extract_note_state(checkpoint: Any) -> dict[str, Tensor]:
    """Accept official note+pedal, note-only trainer, or plain state_dict."""
    state = checkpoint
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if isinstance(state, dict) and "note_model" in state:
        state = state["note_model"]
    if not isinstance(state, dict) or not state or not all(
        isinstance(k, str) and torch.is_tensor(v) for k, v in state.items()
    ):
        raise ValueError("Checkpoint does not contain a recognizable note-model state_dict")
    if all(k.startswith("module.") for k in state):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    return state


class KongEncoder(nn.Module):
    """Shared interface exposing Kong's pretrained representations.

    `set_backbone_frozen(True)` freezes ONLY the note backbone; the optional
    `fusion_adapter` remains trainable. A frozen backbone stays in eval mode,
    even when the outer wrapper is in train mode.

    For fine-tuning or LoRA, call set_backbone_frozen(False), then manage the
    exact trainable backbone/LoRA parameters as required. The wrapper itself
    does not insert LoRA layers.

    Hooks store per-forward activations. Do not run concurrent forwards on the
    same instance or use torch.nn.DataParallel; use separate replicas / DDP.
    """

    sample_rate = SAMPLE_RATE
    frames_per_second = FRAMES_PER_SECOND
    feature_dim = FEATURE_DIM

    def __init__(
        self,
        note_model: nn.Module,
        fusion_adapter: ResidualKongFusion | None = None,
        backbone_frozen: bool = True,
    ) -> None:
        super().__init__()
        self.note_model = note_model
        self.fusion_adapter = fusion_adapter
        if fusion_adapter is not None and fusion_adapter.feature_dim != FEATURE_DIM:
            raise ValueError(f"Adapter feature_dim must be {FEATURE_DIM}")

        self._activation_cache: dict[str, Tensor] = {}
        locations = {
            "frame": "frame_model.gru",
            "onset": "reg_onset_model.gru",
            "offset": "reg_offset_model.gru",
            "velocity": "velocity_model.gru",
            "refined_onset": "reg_onset_gru",
            "pretrained": "frame_gru",
        }
        self._hook_handles = []
        for key, location in locations.items():
            block: nn.Module = note_model
            for attr in location.split("."):
                try:
                    block = getattr(block, attr)
                except AttributeError as exc:
                    raise AttributeError(
                        f"Kong note model is missing {location!r}; "
                        "upstream architecture may have changed."
                    ) from exc
            self._hook_handles.append(block.register_forward_hook(self._make_hook(key)))
        self.backbone_frozen = False
        self.set_backbone_frozen(backbone_frozen)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        repo_path: str | Path | None = None,
        use_adapter: bool = False,
        backbone_frozen: bool = True,
        adapter_hidden_dim: int = 256,
        adapter_dropout: float = 0.1,
    ) -> KongEncoder:
        """Construct from official combined checkpoint (or note-only weights).

        Checkpoints are loaded on CPU with `weights_only=True` and strict key
        matching. Call `.to(device)` after this factory method.
        """
        if repo_path is None:
            repo_path = Path(__file__).resolve().parent / "third_party" / "piano_transcription"
        note_model = _make_upstream_note_model(repo_path)
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Kong checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state = _extract_note_state(checkpoint)
        note_model.load_state_dict(state, strict=True)
        adapter = (
            ResidualKongFusion(
                feature_dim=FEATURE_DIM,
                hidden_dim=adapter_hidden_dim,
                dropout=adapter_dropout,
            )
            if use_adapter
            else None
        )
        return cls(note_model, adapter, backbone_frozen=backbone_frozen)

    def _make_hook(self, name: str):
        def capture(_module: nn.Module, _inputs: tuple, output: Any) -> None:
            # An nn.GRU returns (output_sequence, final_hidden_state).
            if not isinstance(output, (tuple, list)) or not torch.is_tensor(output[0]):
                raise TypeError(f"Expected GRU tuple output from {name}")
            self._activation_cache[name] = output[0]
        return capture

    def set_backbone_frozen(self, frozen: bool = True) -> KongEncoder:
        """Keep the pretrained note backbone fixed or allow gradient updates."""
        self.backbone_frozen = bool(frozen)
        self.note_model.requires_grad_(not self.backbone_frozen)
        self.note_model.eval() if self.backbone_frozen else self.note_model.train(self.training)
        return self

    def train(self, mode: bool = True) -> KongEncoder:
        super().train(mode)
        if self.backbone_frozen:
            self.note_model.eval()  # BN stats and dropout stay fixed
        return self

    def forward(self, waveform: Tensor) -> dict[str, Any]:
        if waveform.ndim != 2:
            raise ValueError(f"Expected waveform [B, samples]; got {tuple(waveform.shape)}")
        if not waveform.is_floating_point():
            raise TypeError("Waveform must be float32 floating-point audio, not int16")
        if waveform.shape[-1] < 2048:
            raise ValueError("Expected at least 2048 input samples for Kong's STFT frontend")
        if self._activation_cache:
            raise RuntimeError("Activation cache is nonempty: overlapping forwards are unsupported")

        try:
            # Capture pretrained layers without changing the model's forward.
            context = torch.no_grad() if self.backbone_frozen else nullcontext()
            with context:
                piano = self.note_model(waveform)
            required = (*BRANCH_NAMES, "refined_onset", "pretrained")
            missing = [name for name in required if name not in self._activation_cache]
            if missing:
                raise RuntimeError(f"Kong features not captured: {missing}")
            pretrained = self._activation_cache["pretrained"]
            branches = {name: self._activation_cache[name] for name in BRANCH_NAMES}
            refined_onset = self._activation_cache["refined_onset"]
            for name, features in {**branches, "pretrained": pretrained}.items():
                if features.shape[:2] != pretrained.shape[:2] or features.shape[-1] != FEATURE_DIM:
                    raise RuntimeError(f"Unexpected {name} features shape: {tuple(features.shape)}")
            features = (
                self.fusion_adapter(pretrained, branches)
                if self.fusion_adapter is not None
                else pretrained
            )
            return {
                "features": features,
                "pretrained_features": pretrained,
                "branch_features": branches,
                "refined_onset_features": refined_onset,
                "piano": piano,
            }
        finally:
            self._activation_cache.clear()

    def remove_hooks(self) -> None:
        """Optional cleanup when discarding this wrapper; do not forward afterward."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
