"""Head C tests. All but the last section run without the Kong checkpoint."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

SRC = Path(__file__).resolve().parents[1] / "src"
for sub in ("headC", "data", "kong_wrapper"):
    sys.path.insert(0, str(SRC / sub))
from head_c import (BEGIN_NOTE, IGNORE_INDEX, MAX_FRET, N_PITCHES, N_STRINGS,  # noqa: E402
                    OPEN_STRING_MIDI, PIANO_KEYS, HeadC, KongHeadC, assign_strings,
                    predict_tab_grid, string_loss, valid_string_mask)
from loader import notes_to_tab_grid, tab_to_string_per_pitch  # noqa: E402

T = 50


def fake_piano(b=2, t=T, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {k: torch.rand(b, t, N_PITCHES, generator=g) for k in PIANO_KEYS}


def perfect_logits(spp):
    """[..., 88] string targets -> logits whose argmax is the target where it sounds."""
    target = torch.as_tensor(spp)
    logits = torch.zeros(*target.shape, N_STRINGS)
    sounding = target != IGNORE_INDEX
    logits[sounding] = F.one_hot(target[sounding], N_STRINGS).float() * 10
    return logits.masked_fill(~valid_string_mask(), torch.finfo(torch.float32).min)


# ----- mask -----

def test_valid_mask_covers_guitar_range():
    mask = valid_string_mask()
    playable = mask.any(-1).nonzero().flatten()
    assert len(playable) == 44
    assert playable.min() + BEGIN_NOTE == 40 and playable.max() + BEGIN_NOTE == 83
    for p, s in mask.nonzero().tolist():
        assert 0 <= p + BEGIN_NOTE - OPEN_STRING_MIDI[s] <= MAX_FRET


# ----- head -----

def test_head_shape_and_never_picks_invalid_string():
    torch.manual_seed(0)
    logits = HeadC()(fake_piano())
    assert logits.shape == (2, T, N_PITCHES, N_STRINGS)
    assert torch.isfinite(logits).all()
    picked = logits.argmax(-1)                                   # [B, T, 88]
    playable = valid_string_mask().any(-1)
    ok = valid_string_mask()[torch.arange(N_PITCHES), picked]    # [B, T, 88]
    assert ok[..., playable].all()


def test_string_loss_matches_manual_ce():
    torch.manual_seed(0)
    logits = HeadC()(fake_piano())
    target = torch.full((2, T, N_PITCHES), IGNORE_INDEX)
    target[0, 3, 40 - BEGIN_NOTE] = 0      # low E open
    target[1, 7, 64 - BEGIN_NOTE] = 4      # B string, fret 5
    target[1, 7, 60 - BEGIN_NOTE] = 3      # G string, fret 5
    rows = [(0, 3, 40 - BEGIN_NOTE, 0), (1, 7, 64 - BEGIN_NOTE, 4), (1, 7, 60 - BEGIN_NOTE, 3)]
    manual = torch.stack([-F.log_softmax(logits[b, t, p], -1)[s] for b, t, p, s in rows]).mean()
    torch.testing.assert_close(string_loss(logits, target), manual)


def test_string_loss_all_silent_is_zero_and_backprops():
    head = HeadC()
    loss = string_loss(head(fake_piano()), torch.full((2, T, N_PITCHES), IGNORE_INDEX))
    assert loss.item() == 0.0
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())


# ----- prediction helpers vs. the loader's targets -----

def test_predict_tab_grid_round_trips_loader_targets():
    notes = np.array([[0, 3, 0.05, 0.20],    # low E fret 3 (G2)
                      [2, 2, 0.05, 0.30],    # D fret 2 (E3)
                      [4, 1, 0.10, 0.40],    # B fret 1 (C4)
                      [5, 12, 0.25, 0.45]],  # high e fret 12
                     dtype=np.float32)
    tab, _ = notes_to_tab_grid(notes, n_frames=T)
    spp = tab_to_string_per_pitch(tab)
    frame_prob = torch.as_tensor(spp != IGNORE_INDEX).float()
    pred = predict_tab_grid(perfect_logits(spp), frame_prob)
    assert np.array_equal(pred.numpy(), tab)


def test_unison_keeps_lower_string():
    # E4 (MIDI 64) on the B string (fret 5) and the open high e at the same time.
    notes = np.array([[4, 5, 0.0, 0.2], [5, 0, 0.0, 0.2]], dtype=np.float32)
    tab, _ = notes_to_tab_grid(notes, n_frames=T)
    spp = tab_to_string_per_pitch(tab)
    pred = predict_tab_grid(perfect_logits(spp), torch.as_tensor(spp != IGNORE_INDEX).float())
    assert (pred[:21, 4] == 6).all()        # B string, fret 5 (+1)
    assert (pred[:, 5] == 0).all()          # high e lost: Kong labels merge unisons


def test_predict_tab_grid_conflict_goes_to_higher_frame_prob():
    logits = torch.zeros(1, N_PITCHES, N_STRINGS)
    logits[0, 50 - BEGIN_NOTE, 2] = 10      # D3 -> D string fret 0
    logits[0, 52 - BEGIN_NOTE, 2] = 10      # E3 -> D string fret 2
    frame = torch.zeros(1, N_PITCHES)
    frame[0, 50 - BEGIN_NOTE], frame[0, 52 - BEGIN_NOTE] = 0.6, 0.9
    assert predict_tab_grid(logits, frame)[0, 2] == 3


def test_assign_strings_averages_over_note_frames():
    logits = torch.zeros(T, N_PITCHES, N_STRINGS)
    p = 55 - BEGIN_NOTE                     # G3: playable on strings 0..3
    logits[10:20, p, 1] = 3.0               # A string (fret 10) wins on average
    logits[10:12, p, 3] = 5.0               # G string open briefly louder
    logits[30:40, 64 - BEGIN_NOTE, 5] = 4.0
    notes = [[0.30, 0.39, 64], [0.10, 0.19, 55], [0.0, 0.1, 100]]   # last: unplayable
    out = assign_strings(notes, logits)
    np.testing.assert_allclose(out, [[1, 10, 0.10, 0.19], [5, 0, 0.30, 0.39]], rtol=1e-6)


# ----- KongHeadC wiring (fake encoder) -----

class FakeEncoder(nn.Module):
    """Frozen stand-in for KongEncoder: waveform -> piano dict, no grads."""

    def __init__(self, with_adapter=False):
        super().__init__()
        self.proj = nn.Linear(1, N_PITCHES).requires_grad_(False)
        self.fusion_adapter = nn.Identity() if with_adapter else None

    def forward(self, waveform):
        frames = waveform.unfold(-1, 160, 160).mean(-1, keepdim=True)   # [B, T, 1]
        with torch.no_grad():
            probs = torch.sigmoid(self.proj(frames))
        return {"piano": {k: probs for k in PIANO_KEYS}}


def test_kong_head_c_trains_only_the_head():
    model = KongHeadC(FakeEncoder())
    out = model(torch.randn(2, 160 * T))
    logits = out["string_logits"]
    assert logits.shape == (2, T, N_PITCHES, N_STRINGS)
    assert set(out["piano"]) == set(PIANO_KEYS)
    target = torch.full(logits.shape[:-1], IGNORE_INDEX)
    target[:, :, 45 - BEGIN_NOTE] = 1
    string_loss(logits, target).backward()
    assert all(p.grad is not None for p in model.head.parameters())
    assert all(p.grad is None for p in model.encoder.parameters())


def test_kong_head_c_rejects_adapter():
    with pytest.raises(ValueError, match="use_adapter=False"):
        KongHeadC(FakeEncoder(with_adapter=True))


# ----- real Kong checkpoint (skips if the per-person setup is missing) -----

KONG_DIR = SRC / "kong_wrapper"
CHECKPOINT = KONG_DIR / "checkpoints" / "kong_note_pedal.pth"
REPO = KONG_DIR / "third_party" / "piano_transcription"


def test_real_kong_frozen_stage1_step():
    if not CHECKPOINT.is_file():
        pytest.skip(f"Kong checkpoint not found at {CHECKPOINT} (see src/kong_wrapper/README.md)")
    if not REPO.is_dir():
        pytest.skip(f"Kong repo not found at {REPO} (see src/kong_wrapper/README.md)")
    from kong_encoder import KongEncoder
    try:
        encoder = KongEncoder.from_checkpoint(CHECKPOINT, repo_path=REPO,
                                              use_adapter=False, backbone_frozen=True)
    except ImportError as exc:
        pytest.skip(str(exc))

    torch.manual_seed(0)
    model = KongHeadC(encoder).train()
    kong_before = {k: v.clone() for k, v in model.encoder.note_model.state_dict().items()}
    head_before = [p.clone() for p in model.head.parameters()]

    out = model(torch.randn(1, 16000 * 2) * 0.1)
    logits = out["string_logits"]
    assert logits.shape == (1, 201, N_PITCHES, N_STRINGS)
    target = torch.full(logits.shape[:-1], IGNORE_INDEX)
    target[0, 50:100, 52 - BEGIN_NOTE] = 2
    opt = torch.optim.Adam(model.head.parameters(), lr=1e-3)
    string_loss(logits, target).backward()
    opt.step()

    assert any(not torch.equal(a, b) for a, b in zip(head_before, model.head.parameters()))
    for k, v in model.encoder.note_model.state_dict().items():
        assert torch.equal(v, kong_before[k]), f"Kong weight/buffer {k} changed"
