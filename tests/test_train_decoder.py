"""Tests for the decoder trainer and eval (src/decoder/train_decoder.py, eval_decoder.py).

The end-to-end test runs real Kong (frozen, then LoRA) on CPU over a tiny synthetic
dataset with a tiny decoder, and skips if the Kong checkpoint / repo is missing.
"""

import json
import math
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC / "decoder"))
import eval_decoder  # noqa: E402
from train_decoder import lr_lambda, parse_args, run  # noqa: E402

KONG_DIR = SRC / "kong_wrapper"
CHECKPOINT = KONG_DIR / "checkpoints" / "kong_note_pedal.pth"
KONG_REPO = KONG_DIR / "third_party" / "piano_transcription"


def write_clip(path, notes, player, seconds=12.0, seed=0):
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as hf:
        hf.attrs.create("player_id", data=player.encode(), dtype="S100")
        hf.create_dataset("waveform", data=(rng.standard_normal(int(seconds * 16000)) * 2000)
                          .astype(np.int16))
        hf.create_dataset("tab_notes", data=np.asarray(notes, np.float32))


def test_parse_args_stage_defaults():
    a1 = parse_args(["--stage", "1"])
    a2 = parse_args(["--stage", "2", "--epochs", "3", "--test_player", "4"])
    assert (a1.lr, a1.epochs, a1.tag, a1.weight_decay) == (3e-4, 50, "stage1", 0.01)
    assert (a2.lr, a2.epochs, a2.tag) == (1e-4, 3, "stage2")
    assert (a1.d_model, a1.n_layers, a1.n_heads, a1.d_ff, a1.max_len) == (256, 4, 4, 1024, 1280)
    assert a1.out_dir.endswith("runs/decoder") and a2.test_player == "04"
    assert parse_args(["--stage", "2"]).epochs == 25
    assert not hasattr(a1, "val_player")


def test_lr_schedule_warmup_then_cosine_to_zero():
    f = lr_lambda(warmup_steps=10, total_steps=110)
    assert f(0) == pytest.approx(0.1) and f(9) == pytest.approx(1.0)
    assert f(60) == pytest.approx(0.5) and f(110) == pytest.approx(0.0, abs=1e-12)
    assert all(f(s) >= f(s + 1) for s in range(10, 110))
    assert lr_lambda(0, 10)(0) == pytest.approx(1.0)


def test_score_perfect_and_empty():
    ref = {"a": np.array([[2, 2, 1.0, 1.5], [3, 0, 2.0, 2.5]])}
    perfect = eval_decoder.score({"a": ref["a"].copy()}, ref)
    assert perfect["tab_onset"]["pooled"]["f1"] == 1.0
    empty = eval_decoder.score({"a": np.zeros((0, 4))}, ref)
    assert empty["tab_onset"]["pooled"]["f1"] == 0.0


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
            "--checkpoint", str(CHECKPOINT), "--kong_repo", str(KONG_REPO),
            "--warmup_epochs", "0", "--decode_batch", "4"]
    tiny = ["--d_model", "32", "--n_layers", "1", "--n_heads", "2", "--d_ff", "64",
            "--max_len", "64"]
    quiet = lambda *_: None  # noqa: E731

    # Stage 1: frozen Kong; adapter + decoder train for a fixed number of epochs.
    run(parse_args(base + tiny + ["--stage", "1", "--epochs", "1", "--lr", "1e-3"]), log=quiet)
    s1 = out_dir / "stage1_test05"
    final1 = torch.load(s1 / "final.pt", weights_only=False)
    assert final1["stage"] == 1 and "lora" not in final1 and final1["epochs"] == 1
    assert final1["config"]["d_model"] == 32 and final1["tokenizer"]["segment_seconds"] == 10.0
    m1 = json.loads((s1 / "metrics.json").read_text())
    assert [h["epoch"] for h in m1["history"]] == [1]
    first = m1["history"][0]["train"]
    assert first["loss"] < math.log(1104) and first["truncated_sequences"] == 0
    ev1 = json.loads((s1 / "eval.json").read_text())
    assert set(ev1["test"]) == {"greedy"} and ev1["decode_stats"]["windows"] == 2
    assert ev1["decode_stats"]["malformed_groups"] == 0
    with np.load(s1 / "test_notes.npz") as z:                 # test player only
        assert set(z.files) == {"05_Test0-100-C_comp__est", "05_Test0-100-C_comp__ref"}
    last1 = torch.load(s1 / "last.pt", weights_only=False)
    assert not last1["state"]["lora"]
    # The final model is the last epoch, and the adapter trained (its last layer starts at zero).
    assert torch.equal(last1["state"]["adapter"]["adapter.4.weight"],
                       final1["adapter"]["adapter.4.weight"])
    assert last1["state"]["adapter"]["adapter.4.weight"].abs().sum() > 0

    # The standalone eval rebuilds the same model from final.pt and reproduces eval.json.
    again = eval_decoder.run(s1, device="cpu", batch_size=4, log=quiet)
    assert again["test"] == ev1["test"]

    # Stage 2: LoRA on Kong; adapter + decoder (and its config) from stage 1's final model.
    run(parse_args(base + ["--stage", "2", "--epochs", "1"]), log=quiet)
    run_dir = out_dir / "stage2_test05"
    final = torch.load(run_dir / "final.pt", weights_only=False)
    assert final["stage"] == 2 and final["lora"] and final["lora_config"] == {"rank": 8, "alpha": 16.0}
    assert final["config"] == final1["config"]
    last = torch.load(run_dir / "last.pt", weights_only=False)
    trained_b = [v for k, v in last["state"]["lora"].items() if k.endswith(".B")]
    assert any(v.abs().sum() > 0 for v in trained_b)          # LoRA actually trained
    assert (run_dir / "eval.json").exists() and (run_dir / "test_notes.npz").exists()

    # A finished run is skipped by --resume; an unfinished one continues after its last epoch.
    stage2 = base + ["--stage", "2", "--epochs", "2", "--resume"]
    mtime = (run_dir / "eval.json").stat().st_mtime
    run(parse_args(stage2), log=quiet)
    assert (run_dir / "eval.json").stat().st_mtime == mtime
    (run_dir / "eval.json").unlink()                          # as if interrupted after epoch 1
    run(parse_args(stage2), log=quiet)
    hist = json.loads((run_dir / "metrics.json").read_text())["history"]
    assert [h["epoch"] for h in hist] == [1, 2]
