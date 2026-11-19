"""Tests for the Head C trainer (src/headC/train_head_c.py, both stages).

The cache and loss tests need nothing; the end-to-end test runs real Kong (frozen,
then LoRA) on CPU over a tiny synthetic dataset and skips if the Kong checkpoint /
repo is missing. frame_metrics is tested in tests/test_tab_metrics.py.
"""

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
sys.path.insert(0, str(SRC / "headC"))
from train_head_c import (PIANO_KEYS, build_cache, cache_meta, load_windows,  # noqa: E402
                          parse_args, run, stage1_loss, stage2_loss)
from loader import (IGNORE_INDEX, OPEN_STRING_MIDI, SAMPLE_RATE,  # noqa: E402
                    GuitarSetDataset)

KONG_DIR = SRC / "kong_wrapper"
CHECKPOINT = KONG_DIR / "checkpoints" / "kong_note_pedal.pth"
KONG_REPO = KONG_DIR / "third_party" / "piano_transcription"
needs_kong = pytest.mark.skipif(not (CHECKPOINT.is_file() and KONG_REPO.is_dir()),
                                reason="Kong checkpoint / repo missing (see src/kong_wrapper/README.md)")


# ----- synthetic clips (same schema as tests/test_loader.py's write_clip; copied, tests
# aren't a package) -----

NOTES = [[0, 3, 0.50, 1.20], [2, 2, 0.50, 1.20], [1, 0, 9.50, 10.80],
         [5, 12, 12.00, 12.50], [3, 5, 24.00, 24.90]]


def write_clip(path, duration_s, tab_notes, player, noise_seed=None):
    n = int(duration_s * SAMPLE_RATE)
    tab_notes = np.asarray(tab_notes, dtype=np.float32)
    events = []
    for s, f, on, off in tab_notes:
        pitch = OPEN_STRING_MIDI[int(s)] + int(f)
        events += [(on, 1, f"note_on channel=0 note={pitch} velocity=64 time=0"),
                   (off, 0, f"note_on channel=0 note={pitch} velocity=0 time=0")]
    events.sort(key=lambda e: (e[0], e[1]))
    if noise_seed is None:
        audio = (np.arange(n) % 2000 - 1000).astype(np.int16)
    else:
        audio = (np.random.default_rng(noise_seed).standard_normal(n) * 2000).astype(np.int16)
    with h5py.File(path, "w") as hf:
        hf.attrs.create("player_id", data=player.encode(), dtype="S100")
        hf.attrs.create("sample_rate", data=SAMPLE_RATE, dtype=np.int32)
        hf.create_dataset("waveform", data=audio)
        hf.create_dataset("midi_event", data=[e[2].encode() for e in events], dtype="S100")
        hf.create_dataset("midi_event_time", data=np.array([e[0] for e in events], np.float32))
        hf.create_dataset("tab_notes", data=tab_notes)


def make_root(root, duration_s, notes, noise=False):
    (root / "hdf5s").mkdir(parents=True)
    rows = []
    for i, player in enumerate(["00", "01", "02", "03", "04", "05"]):
        track = f"{player}_Test0-100-C_comp"
        write_clip(root / "hdf5s" / f"{track}.h5", duration_s, notes, player,
                   noise_seed=i if noise else None)
        rows.append({"track_id": track, "player_id": player, "duration": duration_s,
                     "hdf5_path": f"hdf5s/{track}.h5"})
    pd.DataFrame(rows).to_csv(root / "manifest.csv", index=False)
    return root


@pytest.fixture
def root(tmp_path):
    make_root(tmp_path, 25.0, NOTES)
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


# ----- losses -----

def bce(output, target, mask):
    """Same formula as Kong's pytorch/losses.py bce."""
    output = torch.clamp(output, 1e-7, 1 - 1e-7)
    matrix = -target * torch.log(output) - (1 - target) * torch.log(1 - output)
    return torch.sum(matrix * mask) / torch.sum(mask)


def loss_inputs():
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
    return out, batch


def scramble_padding(out, batch):
    """Change everything in item 1's padded frames (>= 12)."""
    for k in PIANO_KEYS:
        out["piano"][k][1, 12:] = 0.5
    out["string_logits"][1, 12:] = 100.0
    for roll in ("reg_onset_roll", "reg_offset_roll", "frame_roll"):
        batch[roll][1, 12:] = 1.0


def test_stage2_loss_masks_padding_and_weights_lambda():
    out, batch = loss_inputs()
    total1, parts1 = stage2_loss(out, batch, 1.0, bce)
    total3, parts3 = stage2_loss(out, batch, 3.0, bce)
    kong = parts1["onset"] + parts1["offset"] + parts1["frame"]
    torch.testing.assert_close(total1, kong + parts1["string"])
    torch.testing.assert_close(total3, kong + 3 * parts1["string"])
    scramble_padding(out, batch)
    total_after, _ = stage2_loss(out, batch, 1.0, bce)
    torch.testing.assert_close(total_after, total1)


def test_stage1_loss_is_string_loss_and_masks_padding():
    out, batch = loss_inputs()
    total, parts = stage1_loss(out, batch)
    assert set(parts) == {"string"}
    torch.testing.assert_close(total, stage2_loss(out, batch, 1.0, bce)[1]["string"])
    scramble_padding(out, batch)
    torch.testing.assert_close(stage1_loss(out, batch)[0], total)


# ----- end to end with real Kong (CPU, tiny synthetic data): stage 1 -> stage 2 -> resume -----

@needs_kong
def test_stage1_then_stage2_end_to_end_and_resume(tmp_path):
    root = make_root(tmp_path / "data", 12.0, [[2, 2, 1.0, 1.5], [4, 1, 1.0, 1.5], [0, 3, 5.0, 6.0]],
                     noise=True)
    out_dir = tmp_path / "runs"
    base = ["--root", str(root), "--out_dir", str(out_dir), "--device", "cpu",
            "--batch_size", "2", "--crops_per_clip", "1", "--num_workers", "0",
            "--workers", "1", "--checkpoint", str(CHECKPOINT), "--kong_repo", str(KONG_REPO),
            "--lr", "1e-3"]
    quiet = lambda *_: None  # noqa: E731

    # Stage 1: frozen Kong, fresh head, fixed number of epochs.
    run(parse_args(base + ["--stage", "1", "--epochs", "1", "--hidden", "16"]), log=quiet)
    s1 = out_dir / "stage1_test05"
    final1 = torch.load(s1 / "final.pt", weights_only=False)
    assert final1["stage"] == 1 and "lora" not in final1 and final1["hidden"] == 16
    assert final1["epochs"] == 1 and final1["threshold"] == 0.5
    m1 = json.loads((s1 / "metrics.json").read_text())
    assert [h["epoch"] for h in m1["history"]] == [1]
    assert set(m1["history"][0]["train_loss"]) == {"string", "total"}
    assert set(m1["test"]["frame"]) == {"0.5"} and "validation" not in m1
    assert set(json.loads((s1 / "eval.json").read_text())["test"]) == {"fixed"}
    assert (s1 / "test_notes.npz").exists() and not (s1 / "kong_cache").exists()
    with np.load(s1 / "test_notes.npz") as z:                 # test player only
        assert {k.split("_")[0] for k in z.files} == {"05"}

    # Stage 2: LoRA'd Kong, head from stage 1's final model.
    run(parse_args(base + ["--stage", "2", "--epochs", "1"]), log=quiet)
    run_dir = out_dir / "stage2_test05"
    hist = json.loads((run_dir / "metrics.json").read_text())["history"]
    assert [h["epoch"] for h in hist] == [1]
    assert {"onset", "offset", "frame", "string", "total"} <= set(hist[0]["train_loss"])
    final = torch.load(run_dir / "final.pt", weights_only=False)
    assert final["stage"] == 2 and final["lora"] and final["args"]["lam"] == 0.2
    ev = json.loads((run_dir / "eval.json").read_text())
    assert "fixed" in ev["test"] and (run_dir / "test_notes.npz").exists()
    assert not (run_dir / "kong_cache").exists()            # removed unless --keep_cache
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


def test_parse_args_stage_defaults():
    a1 = parse_args(["--stage", "1"])
    a2 = parse_args(["--stage", "2", "--lr", "3e-4", "--test_player", "5"])
    assert (a1.lr, a1.weight_decay, a1.epochs, a1.tag) == (1e-3, 1e-4, 30, "stage1")
    assert (a2.lr, a2.weight_decay, a2.epochs, a2.tag) == (3e-4, 0.0, 20, "stage2")
    assert not hasattr(a1, "val_player") and not hasattr(a1, "patience")
    assert a2.test_player == "05"
