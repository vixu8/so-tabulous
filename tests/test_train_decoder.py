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
from train_decoder import lr_lambda, parse_args, run, train_step, training_data  # noqa: E402
from loader import collate  # noqa: E402
from tab_decoder import KongTabDecoder  # noqa: E402
from tokenizer import TabTokenizer  # noqa: E402

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


def write_dataset(root, tracks, notes, seconds=12.0, seed=0, splits=None):
    """A features/v1-style folder: manifest.csv + one HDF5 per (track_id, player).

    `splits`, if given, is a per-track split label (adds a `split` column, like the
    SynthTab manifest); otherwise there is no split column.
    """
    (root / "hdf5s").mkdir(parents=True)
    rows = []
    for i, (track, player) in enumerate(tracks):
        write_clip(root / "hdf5s" / f"{track}.h5", notes, player, seconds, seed=seed + i)
        row = {"track_id": track, "player_id": player, "duration": seconds,
               "hdf5_path": f"hdf5s/{track}.h5"}
        if splits is not None:
            row["split"] = splits[i]
        rows.append(row)
    pd.DataFrame(rows).to_csv(root / "manifest.csv", index=False)
    return root


def test_parse_args_stage_and_arch_defaults():
    a1 = parse_args(["--stage", "1"])
    a2 = parse_args(["--stage", "2", "--epochs", "3", "--test_player", "4"])
    assert (a1.lr, a1.epochs, a1.tag, a1.weight_decay) == (3e-4, 50, "stage1", 0.01)
    assert (a2.lr, a2.epochs, a2.tag) == (1e-4, 3, "stage2")
    assert (a1.arch, a1.data, a1.batch_size, a1.grad_accum) == ("decoder1", "guitarset", 8, 1)
    assert (a1.d_model, a1.n_layers, a1.n_heads, a1.d_ff, a1.ffn, a1.max_len) == (
        256, 4, 4, 1024, "gelu", 1280)
    assert a1.out_dir.endswith("runs/decoder1") and a2.test_player == "04"
    assert parse_args(["--stage", "2"]).epochs == 25
    assert not hasattr(a1, "val_player")
    assert (a1.decoding, a1.beam_size, a1.length_penalty, a1.max_chord_span) == (
        ["greedy", "beam"], 4, 1.0, None)
    # decoder2: TART's transformer, SynthTab, microbatch 4 x 4 accumulation = 16.
    d2 = parse_args(["--arch", "decoder2", "--stage", "1"])
    assert (d2.n_layers, d2.n_heads, d2.d_model, d2.d_ff, d2.ffn) == (6, 8, 256, 1024, "geglu")
    assert (d2.data, d2.batch_size, d2.grad_accum) == ("synthtab", 4, 4)
    assert d2.out_dir.endswith("runs/decoder2") and d2.train_root.endswith("features/synthtab")
    over = parse_args(["--arch", "decoder2", "--stage", "1", "--n_layers", "2",
                       "--grad_accum", "2", "--data", "guitarset", "--max_chord_span", "5"])
    assert (over.n_layers, over.n_heads, over.grad_accum, over.data, over.max_chord_span) == (
        2, 8, 2, "guitarset", 5)


class FakeEncoder(torch.nn.Module):
    feature_dim = 512

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(1, 512)

    def forward(self, waveform):
        return {"features": self.proj(waveform[:, :1001, None])}


def test_grad_accumulation_equals_one_big_batch():
    tok = TabTokenizer()
    rng = np.random.default_rng(0)
    items = []
    for i in range(8):                        # sequences of very different lengths
        n = int(rng.integers(0, 12))
        on = np.sort(rng.uniform(0, 9, n))
        notes = np.stack([rng.integers(0, 6, n), rng.integers(0, 20, n), on, on + 0.5], 1)
        items.append({"waveform": rng.standard_normal(2000).astype(np.float32),
                      "n_valid_frames": 1001, "tokens": tok.encode(notes)})
    args = parse_args(["--stage", "1", "--grad_clip", "1e9"])
    deltas = []
    for micro, accum in ((8, 1), (4, 2), (2, 4)):
        torch.manual_seed(0)
        model = KongTabDecoder(FakeEncoder(), tok, d_model=32, n_layers=2, n_heads=4, d_ff=64,
                               dropout=0.0, max_len=64, ffn="geglu").train()
        torch.nn.init.normal_(model.decoder.out.weight, std=0.5)
        before = [p.detach().clone() for p in model.parameters()]
        params = list(model.parameters())
        opt = torch.optim.SGD(params, lr=1.0)
        group = [collate(items[i:i + micro]) for i in range(0, 8, micro)]
        assert len(group) == accum
        loss, acc, _, _ = train_step(model, group, opt, params, args, torch.device("cpu"))
        deltas.append((loss, acc, [p.detach() - b for p, b in zip(model.parameters(), before)]))
    for loss, acc, delta in deltas[1:]:
        assert loss == pytest.approx(deltas[0][0], rel=1e-5)
        assert acc == pytest.approx(deltas[0][1], rel=1e-6)
        for d, d0 in zip(delta, deltas[0][2]):
            torch.testing.assert_close(d, d0, rtol=1e-4, atol=1e-6)


def test_synthtab_data_never_includes_guitarset(tmp_path):
    notes = [[2, 2, 1.0, 1.5]]
    gs = write_dataset(tmp_path / "gs", [("00_A", "00"), ("01_B", "01")], notes)
    st = write_dataset(tmp_path / "st", [("synth_0", "s"), ("synth_1", "s")], notes)
    a = parse_args(["--arch", "decoder2", "--stage", "1", "--root", str(gs),
                    "--train_root", str(st)])
    train, val, root, test = training_data(a)
    assert list(train.track_id) == ["synth_0", "synth_1"] and root == str(st)
    assert val is None                                          # no split column -> no val
    assert list(test.track_id) == ["00_A", "01_B"]              # every GuitarSet clip
    with pytest.raises(ValueError, match="never"):
        training_data(parse_args(["--arch", "decoder2", "--stage", "1", "--root", str(gs),
                                  "--train_root", str(gs)]))
    leak = write_dataset(tmp_path / "leak", [("synth_0", "s"), ("01_B", "01")], notes)
    with pytest.raises(ValueError, match="never see"):
        training_data(parse_args(["--arch", "decoder2", "--stage", "1", "--root", str(gs),
                                  "--train_root", str(leak)]))


def test_synthtab_split_column_yields_val(tmp_path):
    notes = [[2, 2, 1.0, 1.5]]
    gs = write_dataset(tmp_path / "gs", [("00_A", "00")], notes)
    st = write_dataset(tmp_path / "st",
                       [("tr0", "s"), ("tr1", "s"), ("va0", "s"), ("te0", "s")], notes,
                       splits=["train", "train", "val", "test"])
    a = parse_args(["--arch", "decoder2", "--stage", "1", "--root", str(gs),
                    "--train_root", str(st)])
    train, val, root, test = training_data(a)
    assert list(train.track_id) == ["tr0", "tr1"]       # split == train
    assert list(val.track_id) == ["va0"]                # split == val
    assert "te0" not in set(train.track_id) | set(val.track_id)   # test rows never trained on


def test_lr_schedule_warmup_then_cosine_to_zero():
    f = lr_lambda(warmup_steps=10, total_steps=110)
    assert f(0) == pytest.approx(0.1) and f(9) == pytest.approx(1.0)
    assert f(60) == pytest.approx(0.5) and f(110) == pytest.approx(0.0, abs=1e-12)
    assert all(f(s) >= f(s + 1) for s in range(10, 110))
    assert lr_lambda(0, 10)(0) == pytest.approx(1.0)


def test_score_perfect_and_empty():
    ref = {"a": np.array([[2, 2, 1.0, 1.5], [3, 0, 2.0, 2.5], [1, 5, 11.0, 12.0]])}
    windows = [("a", 0.0, 1001), ("a", 10.0, 201)]
    perfect = eval_decoder.score({"a": ref["a"].copy()}, ref, windows)
    assert perfect["tab_onset"]["pooled"]["f1"] == 1.0
    assert perfect["frame"]["tab"]["f1"] == 1.0 and perfect["frame"]["tdr"] == 1.0
    empty = eval_decoder.score({"a": np.zeros((0, 4))}, ref, windows)
    assert empty["tab_onset"]["pooled"]["f1"] == 0.0 and empty["frame"]["tab"]["f1"] == 0.0
    # Right pitch, wrong string: frame pitch F1 stays 1, tab F1 drops, TDR < 1.
    wrong = ref["a"].copy()
    wrong[0, :2] = [1, 7]                                     # string 2 fret 2 = string 1 fret 7
    off = eval_decoder.score({"a": wrong}, ref, windows)
    assert off["frame"]["pitch"]["f1"] == 1.0 and off["frame"]["tab"]["f1"] < 1.0
    assert off["frame"]["tdr"] < 1.0
    assert "frame" not in eval_decoder.score({"a": wrong}, ref)


@pytest.mark.skipif(not (CHECKPOINT.is_file() and KONG_REPO.is_dir()),
                    reason="Kong checkpoint / repo missing (see src/kong_wrapper/README.md)")
def test_stage1_then_stage2_end_to_end_and_resume(tmp_path):
    notes = [[2, 2, 1.0, 1.5], [4, 1, 1.0, 1.5], [0, 3, 5.0, 6.0]]
    root = write_dataset(tmp_path / "data", [(f"{p}_Test0-100-C_comp", p) for p in
                                             ["00", "01", "02", "03", "04", "05"]], notes)
    out_dir = tmp_path / "runs"
    base = ["--root", str(root), "--out_dir", str(out_dir), "--device", "cpu",
            "--batch_size", "2", "--crops_per_clip", "1", "--num_workers", "0",
            "--checkpoint", str(CHECKPOINT), "--kong_repo", str(KONG_REPO),
            "--warmup_epochs", "0", "--decode_batch", "4"]
    tiny = ["--d_model", "32", "--n_layers", "1", "--n_heads", "2", "--d_ff", "64",
            "--max_len", "64"]
    quiet = lambda *_: None  # noqa: E731

    # Stage 1: frozen Kong (no fusion adapter); the decoder trains for a fixed number of epochs.
    run(parse_args(base + tiny + ["--stage", "1", "--epochs", "1", "--lr", "1e-3"]), log=quiet)
    s1 = out_dir / "stage1_test05"
    final1 = torch.load(s1 / "final.pt", weights_only=False)
    assert final1["stage"] == 1 and "lora" not in final1 and final1["epochs"] == 1
    assert "adapter" not in final1 and final1["arch"] == "decoder1"
    assert final1["config"]["d_model"] == 32 and final1["tokenizer"]["segment_seconds"] == 10.0
    m1 = json.loads((s1 / "metrics.json").read_text())
    assert [h["epoch"] for h in m1["history"]] == [1]
    first = m1["history"][0]["train"]
    assert first["loss"] < math.log(1104) and first["truncated_sequences"] == 0
    ev1 = json.loads((s1 / "eval.json").read_text())
    assert set(ev1["test"]) == {"greedy", "beam4"}             # both decodings by default
    assert ev1["decodings"]["beam4"] == {"method": "beam", "grammar_masked": True,
                                         "max_chord_span": None, "beam_size": 4,
                                         "length_penalty": 1.0}
    for name in ("greedy", "beam4"):
        assert ev1["decode_stats"][name]["windows"] == 2
        assert ev1["decode_stats"][name]["malformed_groups"] == 0
        assert {"pitch", "tab", "tdr"} <= set(ev1["test"][name]["frame"])
        with np.load(s1 / f"test_notes_{name}.npz") as z:     # test player only
            assert set(z.files) == {"05_Test0-100-C_comp__est", "05_Test0-100-C_comp__ref"}
    last1 = torch.load(s1 / "last.pt", weights_only=False)
    assert not last1["state"]["lora"]
    # The final model is the last epoch, and the decoder trained (its output layer starts at zero).
    assert torch.equal(last1["state"]["decoder"]["out.weight"], final1["decoder"]["out.weight"])
    assert last1["state"]["decoder"]["out.weight"].abs().sum() > 0

    # The standalone eval rebuilds the same model from final.pt and reproduces eval.json.
    again = eval_decoder.run(s1, device="cpu", batch_size=4, log=quiet)
    assert again["test"] == ev1["test"]
    # ... and can rerun a single decoding with other settings.
    span = eval_decoder.run(s1, device="cpu", batch_size=4, log=quiet, decoding=["greedy"],
                            max_chord_span=5)
    assert set(span["test"]) == {"greedy"} and span["decodings"]["greedy"]["max_chord_span"] == 5

    # Stage 2: LoRA on Kong; the decoder (and its config) from stage 1's final model.
    run(parse_args(base + ["--stage", "2", "--epochs", "1"]), log=quiet)
    run_dir = out_dir / "stage2_test05"
    final = torch.load(run_dir / "final.pt", weights_only=False)
    assert final["stage"] == 2 and final["lora"] and final["lora_config"] == {"rank": 8, "alpha": 16.0}
    assert final["config"] == final1["config"]
    last = torch.load(run_dir / "last.pt", weights_only=False)
    trained_b = [v for k, v in last["state"]["lora"].items() if k.endswith(".B")]
    assert any(v.abs().sum() > 0 for v in trained_b)          # LoRA actually trained
    assert (run_dir / "eval.json").exists() and (run_dir / "test_notes_beam4.npz").exists()

    # A finished run is skipped by --resume; an unfinished one continues after its last epoch.
    stage2 = base + ["--stage", "2", "--epochs", "2", "--resume"]
    mtime = (run_dir / "eval.json").stat().st_mtime
    run(parse_args(stage2), log=quiet)
    assert (run_dir / "eval.json").stat().st_mtime == mtime
    (run_dir / "eval.json").unlink()                          # as if interrupted after epoch 1
    run(parse_args(stage2), log=quiet)
    hist = json.loads((run_dir / "metrics.json").read_text())["history"]
    assert [h["epoch"] for h in hist] == [1, 2]


@pytest.mark.skipif(not (CHECKPOINT.is_file() and KONG_REPO.is_dir()),
                    reason="Kong checkpoint / repo missing (see src/kong_wrapper/README.md)")
def test_decoder2_synthtab_zeroshot_end_to_end(tmp_path):
    notes = [[2, 2, 1.0, 1.5], [4, 1, 1.0, 1.5], [0, 3, 5.0, 6.0]]
    gs = write_dataset(tmp_path / "gs", [(f"{p}_Test0-100-C_comp", p) for p in
                                         ["00", "01", "02"]], notes)
    st = write_dataset(tmp_path / "st", [(f"synth_{i}", "s") for i in range(4)], notes,
                       seconds=11.0, seed=10)
    out_dir = tmp_path / "runs"
    base = ["--arch", "decoder2", "--root", str(gs), "--train_root", str(st),
            "--out_dir", str(out_dir), "--device", "cpu", "--batch_size", "1",
            "--grad_accum", "2", "--crops_per_clip", "1", "--num_workers", "0",
            "--checkpoint", str(CHECKPOINT), "--kong_repo", str(KONG_REPO),
            "--warmup_epochs", "0", "--decode_batch", "4", "--beam_size", "2"]
    tiny = ["--d_model", "32", "--n_layers", "2", "--n_heads", "4", "--d_ff", "64",
            "--max_len", "64"]
    quiet = lambda *_: None  # noqa: E731

    run(parse_args(base + tiny + ["--stage", "1", "--epochs", "1", "--lr", "1e-3"]), log=quiet)
    s1 = out_dir / "stage1_zeroshot"
    final1 = torch.load(s1 / "final.pt", weights_only=False)
    assert (final1["arch"], final1["data"]) == ("decoder2", "synthtab")
    assert final1["config"]["ffn"] == "geglu" and final1["config"]["n_heads"] == 4
    assert "blocks.0.ff.wi.weight" in final1["decoder"]                 # gated-GELU FFN
    m1 = json.loads((s1 / "metrics.json").read_text())
    assert m1["batch"] == {"micro": 1, "grad_accum": 2, "effective": 2}
    ev1 = json.loads((s1 / "eval.json").read_text())
    assert set(ev1["test"]) == {"greedy", "beam2"}
    for name in ("greedy", "beam2"):                  # every GuitarSet clip, 2 windows each
        assert ev1["decode_stats"][name]["windows"] == 6
        with np.load(s1 / f"test_notes_{name}.npz") as z:
            assert {f.split("__")[0] for f in z.files} == {
                f"{p}_Test0-100-C_comp" for p in ["00", "01", "02"]}
    assert eval_decoder.run(s1, device="cpu", batch_size=4, log=quiet)["test"] == ev1["test"]

    # Stage 2 (LoRA) continues the zero-shot run; a decoder1 stage 2 can't load it.
    run(parse_args(base + ["--stage", "2", "--epochs", "1"]), log=quiet)
    final2 = torch.load(out_dir / "stage2_zeroshot" / "final.pt", weights_only=False)
    assert final2["lora"] and final2["config"] == final1["config"]
    with pytest.raises(ValueError, match="decoder2"):
        run(parse_args([a for a in base if a != "decoder2" and a != "--arch"]
                       + ["--data", "synthtab", "--stage", "2", "--epochs", "1",
                          "--stage1_dir", str(s1), "--tag", "other"]), log=quiet)
