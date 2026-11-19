"""Tests for Head C's note-level evaluation (src/headC/eval_head_c.py).

Tests that use Kong's RegressionPostProcessor skip if third_party/ isn't cloned.
"""

import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC / "headC"))
from eval_head_c import (DEFAULT_KONG_REPO, GUITAR_INDEX_OFFSET, _kong_utilities,  # noqa: E402
                         detect_notes, run, stitch_windows)
from head_c import HeadC  # noqa: E402


def kong_available():
    try:
        _kong_utilities(DEFAULT_KONG_REPO)
        return True
    except (FileNotFoundError, ImportError):
        return False


needs_kong = pytest.mark.skipif(not kong_available(),
                                reason=f"Kong repo not found at {DEFAULT_KONG_REPO}")


def bump(length, center, width, height):
    """A smooth peak, like Kong's regression targets."""
    x = np.arange(length)
    return (height * np.clip(1 - np.abs(x - center) / width, 0, 1)).astype(np.float32)


def one_note_piano(t=300, pitch_idx=52 - 40, on=100, off=150):
    """(4, T, 44) Kong outputs with one note on guitar pitch index `pitch_idx`."""
    piano = np.zeros((4, t, 44), dtype=np.float32)
    piano[0, on:off, pitch_idx] = 0.9                      # frame
    piano[1, :, pitch_idx] = bump(t, on, 5, 0.8)           # reg_onset
    piano[2, :, pitch_idx] = bump(t, off, 5, 0.8)          # reg_offset
    piano[3, :, pitch_idx] = 0.5                           # velocity
    return piano


def test_stitch_windows():
    x = np.arange(3 * 1001).reshape(3, 1001, 1)
    out = stitch_windows(x, [1001, 1001, 501])
    assert out.shape == (2501, 1)
    assert out[999, 0] == 999 and out[1000, 0] == 1001        # frame 1000 of window 0 dropped
    assert out[-1, 0] == 2 * 1001 + 500
    t = stitch_windows(torch.from_numpy(x), [1001, 1001, 501])
    assert torch.equal(t, torch.from_numpy(out))


@needs_kong
def test_detect_notes_finds_handmade_note():
    notes = detect_notes(one_note_piano(), [0.3], [0.1])[(0.3, 0.1)]
    assert notes.shape == (1, 3)
    onset, offset, midi = notes[0]
    assert midi == 52 and abs(onset - 1.0) < 0.02 and abs(offset - 1.5) < 0.02


@needs_kong
def test_detect_notes_no_onsets():
    assert detect_notes(np.zeros((4, 200, 44), np.float32), [0.3], [0.1])[(0.3, 0.1)].shape == (0, 3)


@needs_kong
def test_fast_path_matches_kong_postprocessor():
    rng = np.random.default_rng(0)
    t = 800
    piano = np.zeros((4, t, 44), dtype=np.float32)
    for p in rng.choice(44, 10, replace=False):            # peaky random notes on 10 pitches
        for on in rng.choice(np.arange(10, t - 80, 60), 4, replace=False):
            off = on + rng.integers(10, 60)
            h = rng.uniform(0.05, 0.9)
            piano[0, on:off, p] = np.maximum(piano[0, on:off, p], rng.uniform(0.03, 0.9))
            piano[1, :, p] = np.maximum(piano[1, :, p], bump(t, on, 4, h))
            piano[2, :, p] = np.maximum(piano[2, :, p], bump(t, off, 4, rng.uniform(0.1, 0.9)))
    piano[3] = 0.5
    piano += rng.uniform(0, 0.01, piano.shape).astype(np.float32)

    onset_ts, frame_ts = [0.05, 0.3], [0.01, 0.1]
    fast = detect_notes(piano, onset_ts, frame_ts)
    utils = _kong_utilities(DEFAULT_KONG_REPO)
    for o in onset_ts:
        for f in frame_ts:
            pp = utils.RegressionPostProcessor(100, classes_num=44, onset_threshold=o,
                                               offset_threshold=0.3, frame_threshold=f,
                                               pedal_offset_threshold=0.2)
            ref, _ = pp.output_dict_to_note_pedal_arrays({
                "frame_output": piano[0], "reg_onset_output": piano[1],
                "reg_offset_output": piano[2], "velocity_output": piano[3]})
            ref = ref[:, :3].copy()
            ref[:, 2] += GUITAR_INDEX_OFFSET
            assert len(ref) > 0
            np.testing.assert_array_equal(fast[(o, f)], ref)


# ----- end to end on a tiny synthetic setup -----

def write_clip(path, tab_notes, player, duration_s=12.0):
    with h5py.File(path, "w") as hf:
        hf.attrs.create("player_id", data=player.encode(), dtype="S100")
        hf.create_dataset("waveform", data=np.zeros(int(duration_s * 16000), np.int16))
        hf.create_dataset("tab_notes", data=np.asarray(tab_notes, np.float32))


@needs_kong
def test_run_end_to_end(tmp_path):
    root, cache = tmp_path / "data", tmp_path / "cache"
    (root / "hdf5s").mkdir(parents=True)
    cache.mkdir()
    ref = [[2, 2, 1.0, 1.5]]                                # E3 on the D string, fret 2
    rows = []
    for player in ["00", "01", "02", "03", "04", "05"]:
        track = f"{player}_Test0-100-C_comp"
        write_clip(root / "hdf5s" / f"{track}.h5", ref, player)
        rows.append({"track_id": track, "player_id": player, "duration": 12.0,
                     "hdf5_path": f"hdf5s/{track}.h5"})
        # Cache: 2 windows (12 s clip), the note in window 0 at Kong index 52 - 21.
        piano = np.zeros((2, 4, 1001, 88), dtype=np.float16)
        piano[0, :, :, 19:63] = one_note_piano(1001, 52 - 40).astype(np.float16)
        np.savez(cache / f"{track}.npz", piano=piano, n_valid=np.array([1001, 201]),
                 spp=np.zeros((2, 1001, 88), np.int8), tab=np.zeros((2, 1001, 6), np.int8))
    pd.DataFrame(rows).to_csv(root / "manifest.csv", index=False)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    torch.manual_seed(0)
    torch.save({"head": HeadC(hidden=16).state_dict(), "hidden": 16, "dropout": 0.0,
                "args": {"root": str(root), "cache_dir": str(cache),
                         "test_player": "05"}}, run_dir / "final.pt")

    res = run(run_dir, workers=1, log=lambda *_: None)
    assert set(res["test"]) == {"fixed"} and "validation" not in res
    # The note is detected with the right pitch and timing whatever string the random head picks.
    assert res["test"]["fixed"]["note_onset"]["pooled"]["f1"] == 1.0
    assert json.loads((run_dir / "eval.json").read_text())["thresholds"]["fixed"] == {
        "onset": 0.3, "offset": 0.3, "frame": 0.1}
    with np.load(run_dir / "test_notes.npz") as z:
        est = z["05_Test0-100-C_comp__est"]
        assert est.shape == (1, 4) and est[0, 1] == 52 - [40, 45, 50, 55, 59, 64][int(est[0, 0])]
