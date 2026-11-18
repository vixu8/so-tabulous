"""Tests for the Head C stage 1 trainer: Kong-output cache and a tiny end-to-end run.

frame_metrics is tested in tests/test_tab_metrics.py.
"""

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

SRC = Path(__file__).resolve().parents[1] / "src"
for sub in ("headC",):
    sys.path.insert(0, str(SRC / sub))
from train_head_c import (PIANO_KEYS, build_cache, cache_meta, load_windows,  # noqa: E402
                          run)
from loader import (IGNORE_INDEX, OPEN_STRING_MIDI, SAMPLE_RATE,  # noqa: E402
                    GuitarSetDataset)


# ----- synthetic clips (same schema as tests/test_loader.py's write_clip; copied, tests
# aren't a package) and a fake frozen encoder -----

NOTES = [[0, 3, 0.50, 1.20], [2, 2, 0.50, 1.20], [1, 0, 9.50, 10.80],
         [5, 12, 12.00, 12.50], [3, 5, 24.00, 24.90]]


def write_clip(path, duration_s, tab_notes, player):
    n = int(duration_s * SAMPLE_RATE)
    tab_notes = np.asarray(tab_notes, dtype=np.float32)
    events, times = [], []
    for s, f, on, off in tab_notes:
        pitch = OPEN_STRING_MIDI[int(s)] + int(f)
        events += [(on, 1, f"note_on channel=0 note={pitch} velocity=64 time=0"),
                   (off, 0, f"note_on channel=0 note={pitch} velocity=0 time=0")]
    events.sort(key=lambda e: (e[0], e[1]))
    with h5py.File(path, "w") as hf:
        hf.attrs.create("player_id", data=player.encode(), dtype="S100")
        hf.attrs.create("sample_rate", data=SAMPLE_RATE, dtype=np.int32)
        hf.create_dataset("waveform", data=(np.arange(n) % 2000 - 1000).astype(np.int16))
        hf.create_dataset("midi_event", data=[e[2].encode() for e in events], dtype="S100")
        hf.create_dataset("midi_event_time", data=np.array([e[0] for e in events], np.float32))
        hf.create_dataset("tab_notes", data=tab_notes)


@pytest.fixture
def root(tmp_path):
    (tmp_path / "hdf5s").mkdir()
    rows = []
    for player in ["00", "01", "02", "03", "04", "05"]:
        track = f"{player}_Test0-100-C_comp"
        write_clip(tmp_path / "hdf5s" / f"{track}.h5", 25.0, NOTES, player)
        rows.append({"track_id": track, "player_id": player, "duration": 25.0,
                     "hdf5_path": f"hdf5s/{track}.h5"})
    pd.DataFrame(rows).to_csv(tmp_path / "manifest.csv", index=False)
    (tmp_path / "fake.pth").write_bytes(b"0" * 10)
    return tmp_path


class FakeEncoder(nn.Module):
    """Frozen stand-in for KongEncoder: waveform -> piano dict with T = samples // 160 + 1."""

    calls = 0

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(self, waveform):
        FakeEncoder.calls += 1
        b, t = waveform.shape[0], waveform.shape[1] // 160 + 1
        probs = torch.full((b, t, 88), 0.9) * self.w
        return {"piano": {k: probs for k in PIANO_KEYS}}


def test_build_cache_shapes_reuse_and_meta_check(root, tmp_path):
    cache = tmp_path / "cache"
    meta = cache_meta(root, root / "fake.pth")
    FakeEncoder.calls = 0
    assert build_cache(root, cache, meta, FakeEncoder, "cpu", log=lambda *_: None) == 6
    assert FakeEncoder.calls == 6                     # one batched forward per clip

    with np.load(cache / "00_Test0-100-C_comp.npz") as z:
        assert z["piano"].shape == (3, 4, 1001, 88) and z["piano"].dtype == np.float16
        assert z["spp"].shape == (3, 1001, 88) and z["spp"].dtype == np.int8
        assert z["tab"].shape == (3, 1001, 6)
        rows = pd.read_csv(root / "manifest.csv", dtype={"player_id": str}).iloc[:1]
        ds = GuitarSetDataset(rows, root, mode="windows", targets=("string_per_pitch",))
        assert list(z["n_valid"]) == [ds[k]["n_valid_frames"] for k in range(3)]
        assert np.array_equal(z["spp"][0], ds[0]["string_per_pitch"])

    def must_not_run():
        raise AssertionError("encoder built although everything is cached")
    assert build_cache(root, cache, meta, must_not_run, "cpu", log=lambda *_: None) == 0
    with pytest.raises(ValueError, match="built with"):
        build_cache(root, cache, {**meta, "checkpoint": "other.pth"}, must_not_run, "cpu")

    data = load_windows(cache, ["00_Test0-100-C_comp"])
    last = data["n_valid"][-1]
    assert last == 501 and (data["spp"][-1, last:] == IGNORE_INDEX).all()


def test_run_end_to_end(root, tmp_path):
    args = argparse.Namespace(
        root=str(root), val_player="04", test_player="05", checkpoint=str(root / "fake.pth"),
        cache_dir=str(tmp_path / "cache"), out_dir=str(tmp_path / "runs"), epochs=5,
        patience=5, lr=1e-2, weight_decay=0.0, batch_size=4, hidden=32, dropout=0.0,
        seed=0, device="cpu")
    res = run(args, make_encoder=FakeEncoder, log=lambda *_: None)

    hist = res["history"]
    assert hist[0]["epoch"] == 0 and len(hist) == 6
    # Every clip has the same notes, so the head can learn them; val loss must drop.
    assert min(h["val_string_loss"] for h in hist[1:]) < hist[0]["val_string_loss"]
    out = Path(args.out_dir) / "stage1_val04_test05"
    ckpt = torch.load(out / "best.pt", weights_only=False)
    assert ckpt["threshold"] == res["threshold"] and "mlp.0.weight" in ckpt["head"]
    saved = json.loads((out / "metrics.json").read_text())
    assert set(saved) >= {"history", "validation", "test", "threshold", "best_epoch"}
