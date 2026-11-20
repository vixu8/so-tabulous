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
    a2 = parse_args(["--stage", "2", "--epochs", "3", "--test_player", "4"])
    assert (a1.lr, a1.weight_decay, a1.epochs, a1.tag) == (1e-3, 1e-4, 30, "stage1")
    assert (a2.lr, a2.weight_decay, a2.epochs, a2.tag) == (1e-4, 0.0, 3, "stage2")
    assert a1.out_dir.endswith("runs/head_a") and a2.test_player == "04"
    assert not hasattr(a1, "val_player")
    assert parse_args(["--stage", "2"]).epochs == 20


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
    tabcnn = tmp_path / "weights.pt"
    classifier = torch.nn.Sequential(torch.nn.Linear(5952, 128), torch.nn.ReLU(),
                                     torch.nn.Dropout(0.5), torch.nn.Linear(128, 126))
    torch.save({f"classifier.{k}": v for k, v in classifier.state_dict().items()}, tabcnn)
    base = ["--root", str(root), "--out_dir", str(out_dir), "--device", "cpu",
            "--tabcnn_checkpoint", str(tabcnn),
            "--batch_size", "2", "--crops_per_clip", "1", "--num_workers", "0",
            "--workers", "1", "--checkpoint", str(CHECKPOINT), "--kong_repo", str(KONG_REPO)]
    quiet = lambda *_: None  # noqa: E731

    # Stage 1: frozen Kong; adapter + head train for a fixed number of epochs.
    run(parse_args(base + ["--stage", "1", "--epochs", "1"]), log=quiet)
    s1 = out_dir / "stage1_test05"
    final1 = torch.load(s1 / "final.pt", weights_only=False)
    assert final1["stage"] == 1 and "lora" not in final1 and final1["adapter"] and final1["head"]
    assert final1["epochs"] == 1 and final1["threshold"] == "0.5"
    m1 = json.loads((s1 / "metrics.json").read_text())
    assert [h["epoch"] for h in m1["history"]] == [1]
    assert set(m1["history"][0]["train_loss"]) == {"tab", "total"}
    assert set(m1["test"]["frame"]) == {"0.5", "argmax"} and "validation" not in m1
    ev1 = json.loads((s1 / "eval.json").read_text())
    assert set(ev1["test"]) == {"fixed", "argmax"} and (s1 / "test_notes.npz").exists()
    with np.load(s1 / "test_notes.npz") as z:                 # test player only
        assert {k.split("_")[0] for k in z.files} == {"05"}
    assert not (s1 / "probs_cache").exists()                # removed unless --keep_cache
    last1 = torch.load(s1 / "last.pt", weights_only=False)
    assert not last1["state"]["lora"]
    # The final model is the last epoch, and the adapter trained (its last layer starts at zero).
    assert torch.equal(last1["state"]["adapter"]["adapter.4.weight"],
                       final1["adapter"]["adapter.4.weight"])
    assert last1["state"]["adapter"]["adapter.4.weight"].abs().sum() > 0

    # Stage 2: LoRA on Kong, adapter + head from stage 1's final model.
    run(parse_args(base + ["--stage", "2", "--epochs", "1"]), log=quiet)
    run_dir = out_dir / "stage2_test05"
    hist = json.loads((run_dir / "metrics.json").read_text())["history"]
    assert [h["epoch"] for h in hist] == [1]
    final = torch.load(run_dir / "final.pt", weights_only=False)
    assert final["stage"] == 2 and final["lora"] and final["lora_config"] == {"rank": 8, "alpha": 16.0}
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
    assert [h["epoch"] for h in hist] == [1, 2]
