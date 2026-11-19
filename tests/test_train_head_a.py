"""Tests for the Head A trainer (src/headA/train_head_a.py, both stages).

The end-to-end test runs real Kong (frozen, then LoRA) on CPU over a tiny synthetic
dataset and skips if the Kong checkpoint / repo is missing.
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
sys.path.insert(0, str(SRC / "headA"))
from train_head_a import parse_args, run  # noqa: E402

KONG_DIR = SRC / "kong_wrapper"
CHECKPOINT = KONG_DIR / "checkpoints" / "kong_note_pedal.pth"
KONG_REPO = KONG_DIR / "third_party" / "piano_transcription"
OPEN = [40, 45, 50, 55, 59, 64]


def write_clip(path, notes, player, seconds=12.0, seed=0):
    rng = np.random.default_rng(seed)
    events = []
    for s, f, on, off in notes:
        pitch = OPEN[int(s)] + int(f)
        events += [(on, 1, f"note_on channel=0 note={pitch} velocity=64 time=0"),
                   (off, 0, f"note_on channel=0 note={pitch} velocity=0 time=0")]
    events.sort(key=lambda e: (e[0], e[1]))
    with h5py.File(path, "w") as hf:
        hf.attrs.create("player_id", data=player.encode(), dtype="S100")
        hf.create_dataset("waveform", data=(rng.standard_normal(int(seconds * 16000)) * 2000)
                          .astype(np.int16))
        hf.create_dataset("midi_event", data=[e[2].encode() for e in events], dtype="S100")
        hf.create_dataset("midi_event_time", data=np.array([e[0] for e in events], np.float32))
        hf.create_dataset("tab_notes", data=np.asarray(notes, np.float32))


def test_parse_args_stage_defaults():
    a1 = parse_args(["--stage", "1"])
    a2 = parse_args(["--stage", "2", "--epochs", "3", "--val_player", "4"])
    assert (a1.lr, a1.weight_decay, a1.epochs, a1.tag) == (1e-3, 1e-4, 50, "stage1")
    assert (a2.lr, a2.weight_decay, a2.epochs, a2.tag) == (1e-4, 0.0, 3, "stage2")
    assert a1.out_dir.endswith("runs/head_a") and a2.val_player == "04"


@pytest.mark.skipif(not (CHECKPOINT.is_file() and KONG_REPO.is_dir()),
                    reason="Kong checkpoint / repo missing (see src/kong_wrapper/README.md)")
def test_stage1_then_stage2_end_to_end_and_resume(tmp_path):
    root = tmp_path / "data"
    (root / "hdf5s").mkdir(parents=True)
    notes = [[2, 2, 1.0, 1.5], [4, 1, 1.0, 1.5], [0, 3, 5.0, 6.0]]
    rows = []
    for i, player in enumerate(["00", "01", "02", "03", "04", "05"]):
        track = f"{player}_Test0-100-C_comp"
        write_clip(root / "hdf5s" / f"{track}.h5", notes, player, seed=i)
        rows.append({"track_id": track, "player_id": player, "duration": 12.0,
                     "hdf5_path": f"hdf5s/{track}.h5"})
    pd.DataFrame(rows).to_csv(root / "manifest.csv", index=False)
    out_dir = tmp_path / "runs"
    base = ["--root", str(root), "--out_dir", str(out_dir), "--device", "cpu",
            "--batch_size", "2", "--crops_per_clip", "1", "--num_workers", "0",
            "--workers", "1", "--checkpoint", str(CHECKPOINT), "--kong_repo", str(KONG_REPO)]
    quiet = lambda *_: None  # noqa: E731

    # Stage 1: frozen Kong; adapter + head train.
    run(parse_args(base + ["--stage", "1", "--epochs", "1"]), log=quiet)
    s1 = out_dir / "stage1_val04_test05"
    best1 = torch.load(s1 / "best.pt", weights_only=False)
    assert best1["stage"] == 1 and "lora" not in best1 and best1["adapter"] and best1["head"]
    m1 = json.loads((s1 / "metrics.json").read_text())
    assert [h["epoch"] for h in m1["history"]] == [0, 1]
    assert set(m1["history"][1]["train_loss"]) == {"tab", "total"}
    assert m1["threshold"] in m1["test"]["frame"]
    ev1 = json.loads((s1 / "eval.json").read_text())
    assert {"chosen", "default"} <= set(ev1["test"]) and (s1 / "test_notes.npz").exists()
    assert not (s1 / "probs_cache").exists()                # removed unless --keep_cache
    last1 = torch.load(s1 / "last.pt", weights_only=False)
    assert not last1["state"]["lora"]
    # The adapter trained (its last layer starts at zero).
    assert last1["state"]["adapter"]["adapter.4.weight"].abs().sum() > 0

    # Stage 2: LoRA on Kong, adapter + head from stage 1.
    run(parse_args(base + ["--stage", "2", "--epochs", "1"]), log=quiet)
    run_dir = out_dir / "stage2_val04_test05"
    hist = json.loads((run_dir / "metrics.json").read_text())["history"]
    assert [h["epoch"] for h in hist] == [0, 1]
    # Epoch 0 of stage 2 is stage 1's best model with Kong unchanged.
    assert hist[0]["val_tab_f1"] == pytest.approx(m1["history"][best1["best_epoch"]]["val_tab_f1"])
    assert hist[0]["val_loss"]["total"] == pytest.approx(
        m1["history"][best1["best_epoch"]]["val_loss"]["total"], rel=1e-4)
    best = torch.load(run_dir / "best.pt", weights_only=False)
    assert best["stage"] == 2 and best["lora"] and best["lora_config"] == {"rank": 8, "alpha": 16.0}
    last = torch.load(run_dir / "last.pt", weights_only=False)
    trained_b = [v for k, v in last["state"]["lora"].items() if k.endswith(".B")]
    assert any(v.abs().sum() > 0 for v in trained_b)          # LoRA actually trained

    # A finished run is skipped by --resume; an unfinished one continues after its last epoch.
    stage2 = base + ["--stage", "2", "--epochs", "2", "--resume"]
    mtime = (run_dir / "eval.json").stat().st_mtime
    run(parse_args(stage2), log=quiet)
    assert (run_dir / "eval.json").stat().st_mtime == mtime
    (run_dir / "eval.json").unlink()                          # as if interrupted after epoch 1
    run(parse_args(stage2), log=quiet)
    hist = json.loads((run_dir / "metrics.json").read_text())["history"]
    assert [h["epoch"] for h in hist] == [0, 1, 2]
