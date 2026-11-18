"""Head C: Kong's piano outputs -> a guitar string (and so a fret) per sounding pitch.

Baseline for the AR decoder. Kong already says which pitches sound in each frame;
Head C only decides which string plays each one. The fret is not learned:
fret = MIDI pitch - open-string pitch.

    encoder = KongEncoder.from_checkpoint(ckpt, use_adapter=False, backbone_frozen=True)
    model = KongHeadC(encoder)
    out = model(batch["waveform"])                       # [B, 160000] -> T = 1001
    loss = string_loss(out["string_logits"], batch["string_per_pitch"])

Shapes:
  piano (from KongEncoder)  4 x [B, T, 88] probabilities (PIANO_KEYS order)
  string_logits             [B, T, 88, 6]; strings that cannot play a pitch at
                            frets 0..MAX_FRET are filled with the dtype's minimum
  string_per_pitch target   [B, T, 88] string index, IGNORE_INDEX where silent
                            (built by src/data/loader.py)

Stage 1 (now): build the encoder with backbone_frozen=True; only the head trains.
Stage 2 (LoRA + Kong's own loss) is added once the shared LoRA setup exists.
Head C reads only `piano`, so it never uses the fusion adapter.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

# Keep in sync with src/data/preprocessing.py and src/data/loader.py.
OPEN_STRING_MIDI = [40, 45, 50, 55, 59, 64]  # standard tuning, low E -> high e
N_STRINGS = len(OPEN_STRING_MIDI)
MAX_FRET = 19
BEGIN_NOTE = 21                              # Kong pitch index 0 = MIDI 21 (A0)
N_PITCHES = 88
IGNORE_INDEX = -100
FRAMES_PER_SECOND = 100

# Order of Kong's outputs in the head's input (keys of KongEncoder(...)["piano"]).
PIANO_KEYS = ("frame_output", "reg_onset_output", "reg_offset_output", "velocity_output")


def valid_string_mask() -> Tensor:
    """Bool (88, 6): True where string s can play Kong pitch p at a fret in 0..MAX_FRET."""
    midi = torch.arange(N_PITCHES)[:, None] + BEGIN_NOTE
    fret = midi - torch.tensor(OPEN_STRING_MIDI)[None, :]
    return (fret >= 0) & (fret <= MAX_FRET)


def _fret_table() -> Tensor:
    """Long (88, 6): fret for each (pitch, string); meaningless where the mask is False."""
    midi = torch.arange(N_PITCHES)[:, None] + BEGIN_NOTE
    return midi - torch.tensor(OPEN_STRING_MIDI)[None, :]


class HeadC(nn.Module):
    """Per-frame MLP: Kong's 4 x 88 piano outputs -> 88 x 6 string logits."""

    def __init__(self, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(len(PIANO_KEYS) * N_PITCHES, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, N_PITCHES * N_STRINGS),
        )
        self.register_buffer("valid", valid_string_mask(), persistent=False)

    def forward(self, piano: dict[str, Tensor]) -> Tensor:
        x = torch.cat([piano[k] for k in PIANO_KEYS], dim=-1)          # [B, T, 352]
        logits = self.mlp(x).unflatten(-1, (N_PITCHES, N_STRINGS))     # [B, T, 88, 6]
        # Finite fill, not -inf: pitches no string can play stay NaN-free.
        return logits.masked_fill(~self.valid, torch.finfo(logits.dtype).min)


class KongHeadC(nn.Module):
    """KongEncoder (no adapter) + HeadC. Returns string logits and Kong's piano outputs."""

    def __init__(self, encoder: nn.Module, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        if getattr(encoder, "fusion_adapter", None) is not None:
            raise ValueError("Head C reads only Kong's piano outputs; build the encoder "
                             "with use_adapter=False")
        self.encoder = encoder
        self.head = HeadC(hidden=hidden, dropout=dropout)

    def forward(self, waveform: Tensor) -> dict[str, Tensor | dict[str, Tensor]]:
        piano = self.encoder(waveform)["piano"]
        return {"string_logits": self.head(piano), "piano": piano}


def string_loss(logits: Tensor, target: Tensor) -> Tensor:
    """Cross-entropy over strings, only where a pitch sounds (target != IGNORE_INDEX).

    logits [..., 88, 6], target [..., 88]. Returns 0 (still differentiable) if
    nothing sounds in the batch.
    """
    keep = target != IGNORE_INDEX
    if not keep.any():
        return (logits * 0.0).sum()  # not logits.sum() * 0: masked entries sum to -inf
    return F.cross_entropy(logits[keep], target[keep])


@torch.no_grad()
def predict_tab_grid(logits: Tensor, frame_prob: Tensor, threshold: float = 0.5) -> Tensor:
    """Frame-level tab in the loader's `tab` layout: (..., T, 6), 0 silent, fret + 1 sounding.

    A pitch is on where Kong's frame_output > threshold (guitar-range pitches
    only) and goes to its argmax string. If two pitches pick one string in the
    same frame, the one with the higher frame probability wins.
    """
    valid = valid_string_mask().to(logits.device)
    on = (frame_prob > threshold) & valid.any(-1)                         # [..., T, 88]
    string = logits.argmax(-1)                                            # [..., T, 88]
    picks = on.unsqueeze(-2) & (string.unsqueeze(-2)
                                == torch.arange(N_STRINGS, device=logits.device)[:, None])
    score = torch.where(picks, frame_prob.unsqueeze(-2), torch.full_like(frame_prob, -1.0)
                        .unsqueeze(-2))                                   # [..., T, 6, 88]
    best, pitch = score.max(-1)                                           # [..., T, 6]
    fret = pitch + BEGIN_NOTE - torch.tensor(OPEN_STRING_MIDI, device=logits.device)
    return torch.where(best >= 0, fret + 1, torch.zeros_like(fret))


@torch.no_grad()
def assign_strings(notes, logits: Tensor) -> np.ndarray:
    """Pick one string per detected piano note.

    notes:  (N, 3) [onset_s, offset_s, midi_pitch], times relative to the
            start of `logits` (e.g. Kong's RegressionPostProcessor output).
    logits: [T, 88, 6] string logits for the same stretch of audio.
    Returns (M, 4) float32 [string, fret, onset_s, offset_s] (the `tab_notes`
    layout), sorted by onset then string. Notes no string can play are dropped.
    """
    notes = np.asarray(notes, dtype=np.float64).reshape(-1, 3)
    log_probs = F.log_softmax(logits.float(), dim=-1).cpu()
    valid = valid_string_mask()
    n_frames = log_probs.shape[0]
    rows = []
    for onset, offset, midi in notes:
        p = int(round(midi)) - BEGIN_NOTE
        if not (0 <= p < N_PITCHES) or not valid[p].any():
            continue
        a = min(max(int(round(onset * FRAMES_PER_SECOND)), 0), n_frames - 1)
        b = min(max(int(round(offset * FRAMES_PER_SECOND)), a), n_frames - 1)
        score = log_probs[a:b + 1, p].mean(0).masked_fill(~valid[p], float("-inf"))
        s = int(score.argmax())
        rows.append((s, p + BEGIN_NOTE - OPEN_STRING_MIDI[s], onset, offset))
    out = np.asarray(rows, dtype=np.float32).reshape(-1, 4)
    return out[np.lexsort((out[:, 0], out[:, 2]))]
