"""Tests for Head C stage 2 (src/headC/train_head_c_stage2.py).

The loss test needs nothing; the end-to-end test runs real Kong + LoRA on CPU over a
tiny synthetic dataset and skips if the Kong checkpoint / repo is missing.
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
from train_head_c_stage2 import parse_args, run, stage2_loss  # noqa: E402
from head_c import IGNORE_INDEX, PIANO_KEYS, HeadC  # noqa: E402

KONG_DIR = SRC / "kong_wrapper"
CHECKPOINT = KONG_DIR / "checkpoints" / "kong_note_pedal.pth"
REPO = KONG_DIR / "third_party" / "piano_transcription"
OPEN = [40, 45, 50, 55, 59, 64]


def bce(output, target, mask):
    """Same formula as Kong's pytorch/losses.py bce."""
    output = torch.clamp(output, 1e-7, 1 - 1e-7)
    matrix = -target * torch.log(output) - (1 - target) * torch.log(1 - output)
    return torch.sum(matrix * mask) / torch.sum(mask)


def test_stage2_loss_masks_padding_and_weights_lambda():
    torch.manual_seed(0)
    b, t = 2, 30
    out = {"piano": {k: torch.rand(b, t, 88) * 0.98 + 0.01 for k in PIANO_KEYS},
           "string_logits": torch.randn(b, t, 88, 6)}
    spp = torch.full((b, t, 88), IGNORE_INDEX)
    spp[:, :, 40 - 21] = 0
    batch = {"n_valid_frames": torch.tensor([30, 12]), "string_per_pitch": spp,
             "mask_roll": torch.ones(b, t, 88)}
    for roll in ("reg_onset_roll", "reg_offset_roll", "frame_roll"):
        batch[roll] = (torch.rand(b, t, 88) > 0.9).float()

    total1, parts1 = stage2_loss(out, batch, 1.0, bce)
    total3, parts3 = stage2_loss(out, batch, 3.0, bce)
    kong = parts1["onset"] + parts1["offset"] + parts1["frame"]
    torch.testing.assert_close(total1, kong + parts1["string"])
    torch.testing.assert_close(total3, kong + 3 * parts1["string"])

    # Changing anything in the padded frames of item 1 doesn't change the loss.
    for k in PIANO_KEYS:
        out["piano"][k][1, 12:] = 0.5
    out["string_logits"][1, 12:] = 100.0
    for roll in ("reg_onset_roll", "reg_offset_roll", "frame_roll"):
        batch[roll][1, 12:] = 1.0
    total_after, _ = stage2_loss(out, batch, 1.0, bce)
    torch.testing.assert_close(total_after, total1)


# ----- end to end with real Kong (CPU, tiny synthetic data) -----

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


@pytest.mark.skipif(not (CHECKPOINT.is_file() and REPO.is_dir()),
                    reason="Kong checkpoint / repo missing (see src/kong_wrapper/README.md)")
def test_stage2_end_to_end_and_resume(tmp_path):
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
    stage1 = out_dir / "stage1_val04_test05"
    stage1.mkdir(parents=True)
    torch.manual_seed(0)
    head = HeadC(hidden=16, dropout=0.0)
    torch.save({"head": head.state_dict(), "hidden": 16, "dropout": 0.0}, stage1 / "best.pt")

    base = ["--root", str(root), "--out_dir", str(out_dir), "--device", "cpu",
            "--batch_size", "2", "--crops_per_clip", "1", "--num_workers", "0",
            "--workers", "1", "--checkpoint", str(CHECKPOINT), "--kong_repo", str(REPO),
            "--lr", "1e-3"]
    run(parse_args(base + ["--epochs", "1"]), log=lambda *_: None)
    run_dir = out_dir / "stage2_val04_test05"
    hist = json.loads((run_dir / "metrics.json").read_text())["history"]
    assert [h["epoch"] for h in hist] == [0, 1]
    assert {"onset", "offset", "frame", "string", "total"} <= set(hist[1]["train_loss"])
    assert {"total", "string"} <= set(hist[1]["val_loss"])
    best = torch.load(run_dir / "best.pt", weights_only=False)
    assert best["stage"] == 2 and best["lora"] and best["args"]["lam"] == 0.2
    ev = json.loads((run_dir / "eval.json").read_text())
    assert "chosen" in ev["test"] and (run_dir / "test_notes.npz").exists()
    assert not (run_dir / "kong_cache").exists()            # removed unless --keep_cache
    last = torch.load(run_dir / "last.pt", weights_only=False)
    trained_b = [v for k, v in last["state"]["lora"].items() if k.endswith(".B")]
    assert any(v.abs().sum() > 0 for v in trained_b)          # LoRA actually trained

    # A finished run is skipped by --resume; an unfinished one continues after its last epoch.
    mtime = (run_dir / "eval.json").stat().st_mtime
    run(parse_args(base + ["--epochs", "2", "--resume"]), log=lambda *_: None)
    assert (run_dir / "eval.json").stat().st_mtime == mtime
    (run_dir / "eval.json").unlink()                          # as if interrupted after epoch 1
    run(parse_args(base + ["--epochs", "2", "--resume"]), log=lambda *_: None)
    hist = json.loads((run_dir / "metrics.json").read_text())["history"]
    assert [h["epoch"] for h in hist] == [0, 1, 2]
