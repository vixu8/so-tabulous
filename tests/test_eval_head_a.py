"""Tests for Head A's note-level evaluation (src/headA/eval_head_a.py). No Kong needed."""

import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
for sub in ("headA", "data"):
    sys.path.insert(0, str(SRC / sub))
from eval_head_a import clip_notes, run, score_clips, stitch_windows  # noqa: E402
from loader import notes_to_tab_grid  # noqa: E402

N_CLASSES = 21
REF = np.array([[0, 3, 1.00, 1.50], [2, 0, 1.00, 2.00], [5, 12, 4.00, 4.02],
                [0, 3, 6.00, 6.50]], np.float32)     # string 0 fret 3 twice, with a gap


def one_hot(tab):
    """(T, 6) tab classes -> (T, 6, 21) probabilities."""
    return torch.nn.functional.one_hot(torch.as_tensor(tab), N_CLASSES).float().numpy()


def test_clip_notes_runs_and_min_frames():
    probs = one_hot(np.zeros((50, 6), np.int64))
    probs[10:20, 0] = one_hot(np.full((10, 1), 4))[:, 0]       # string 0, fret 3, 10 frames
    probs[30:32, 2] = one_hot(np.full((2, 1), 1))[:, 0]        # string 2, fret 0, 2 frames
    notes = clip_notes(probs)
    np.testing.assert_allclose(notes, [[0, 3, 0.10, 0.19], [2, 0, 0.30, 0.31]], atol=1e-6)
    assert len(clip_notes(probs, min_frames=3)) == 1
    assert len(clip_notes(probs, min_frames=11)) == 0


def test_perfect_probabilities_score_one():
    tab, _ = notes_to_tab_grid(REF)
    summary, notes = score_clips({"clip": one_hot(tab)}, {"clip": REF}, sounds=[None, 0.5],
                                 min_frames=[1, 3], log=lambda *_: None)
    assert summary[(None, 1)]["tab_onset"]["pooled"]["f1"] == 1.0
    assert summary[(None, 1)]["tab_offset"]["pooled"]["f1"] == 1.0
    # 4.00-4.02 s covers frames 400..402 (3 frames), so min_frames 3 keeps it.
    assert summary[(None, 3)]["tab_onset"]["pooled"]["n_est"] == 4
    assert len(notes[(0.5, 1)]["clip"]) == 4


def test_stitch_windows():
    x = np.arange(3 * 1001).reshape(3, 1001)
    out = stitch_windows(x, [1001, 1001, 201])
    assert out.shape == (2201,) and out[1000] == x[1, 0] and out[-1] == x[2, 200]


def write_clip(path, tab_notes, player, duration_s=12.0):
    with h5py.File(path, "w") as hf:
        hf.attrs.create("player_id", data=player.encode(), dtype="S100")
        hf.create_dataset("waveform", data=np.zeros(int(duration_s * 16000), np.int16))
        hf.create_dataset("tab_notes", data=np.asarray(tab_notes, np.float32))


def test_run_end_to_end(tmp_path):
    root, cache = tmp_path / "data", tmp_path / "cache"
    (root / "hdf5s").mkdir(parents=True)
    cache.mkdir()
    rows = []
    for player in ["00", "01", "02", "03", "04", "05"]:
        track = f"{player}_Test0-100-C_comp"
        write_clip(root / "hdf5s" / f"{track}.h5", REF, player)
        rows.append({"track_id": track, "player_id": player, "duration": 12.0,
                     "hdf5_path": f"hdf5s/{track}.h5"})
        # 2 windows (12 s clip), all notes in window 0, perfect probabilities.
        probs = np.zeros((2, 1001, 6, N_CLASSES), np.float16)
        probs[:, :, :, 0] = 1
        probs[0] = one_hot(notes_to_tab_grid(REF)[0]).astype(np.float16)
        np.savez(cache / f"{track}.npz", probs=probs, n_valid=np.array([1001, 201]))
    pd.DataFrame(rows).to_csv(root / "manifest.csv", index=False)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    torch.save({"args": {"root": str(root), "cache_dir": str(cache),
                         "test_player": "05"}}, run_dir / "final.pt")
    res = run(run_dir, workers=1, log=lambda *_: None)
    assert set(res["test"]) == {"fixed", "argmax"}
    assert res["thresholds"]["fixed"] == {"sound": 0.5, "min_frames": 1}
    assert res["test"]["fixed"]["tab_onset"]["pooled"]["f1"] == 1.0
    assert res["test"]["argmax"]["tab_onset"]["pooled"]["f1"] == 1.0
    saved = json.loads((run_dir / "eval.json").read_text())
    assert saved["thresholds"]["argmax"] == {"sound": None, "min_frames": 1}
    assert "validation" not in saved
    with np.load(run_dir / "test_notes.npz") as z:
        np.testing.assert_allclose(z["05_Test0-100-C_comp__est"], REF, atol=1e-6)
