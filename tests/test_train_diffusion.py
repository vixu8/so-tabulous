"""Train/resume/evaluate integration using real loader files and a tiny encoder."""

import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC / "diffusion"))
import eval_diffusion as ev  # noqa: E402
import train_diffusion as tr  # noqa: E402
from kong_tab_diffusion import KongTabDiffusion  # noqa: E402
from test_tab_diffusion import FakeEncoder  # noqa: E402


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def write_dataset(root):
    root.mkdir()
    rows = []
    for index, player in enumerate(("00", "01", "05")):
        track = f"track{index}"
        with h5py.File(root / f"{track}.h5", "w") as hf:
            hf["waveform"] = np.arange(800, dtype=np.int16)
            hf["tab_notes"] = np.array([[0, 3, 0, 0.04]], dtype=np.float32)
        rows.append({"track_id": track, "player_id": player, "hdf5_path": f"{track}.h5"})
    pd.DataFrame(rows).to_csv(root / "manifest.csv", index=False)
    return root


def arguments(root, output, *extras):
    return tr.parse_args(["--root", str(root), "--out_dir", str(output), "--device", "cpu",
        "--epochs", "2", "--batch_size", "2", "--num_workers", "0", "--timesteps", "2",
        "--inference_steps", "1", "2", "--hidden_dim", "8", "--embed_dim", "4",
        "--lstm_hidden", "4", "--heads", "2", "--d_ff", "16", "--dilations", "1",
        "--crop_mode", "windows", "--dropout", "0.1", "--log_every", "0", *extras])


def fake_build(args, device):
    return KongTabDiffusion(FakeEncoder(frames=1001), timesteps=args.timesteps,
        hidden_dim=args.hidden_dim, embed_dim=args.embed_dim, lstm_hidden=args.lstm_hidden,
        heads=args.heads, d_ff=args.d_ff, dilations=args.dilations, dropout=args.dropout).to(device)


def test_end_to_end_resume_matches_uninterrupted_and_evaluation(tmp_path, monkeypatch):
    root = write_dataset(tmp_path / "data")
    monkeypatch.setattr(tr, "build_model", fake_build)
    full = arguments(root, tmp_path / "full", "--skip_eval")
    full_history = tr.run(full)
    interrupted = arguments(root, tmp_path / "resume", "--skip_eval")
    actual_step, calls = tr.train_step, []
    def stop_in_second_epoch(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("simulated interruption")
        return actual_step(*args, **kwargs)
    monkeypatch.setattr(tr, "train_step", stop_in_second_epoch)
    with pytest.raises(RuntimeError, match="simulated"):
        tr.run(interrupted)
    monkeypatch.setattr(tr, "train_step", actual_step)
    interrupted.resume = True
    resumed_history = tr.run(interrupted)
    assert resumed_history[-1]["train"]["loss"] == full_history[-1]["train"]["loss"]
    full_state = torch.load(tmp_path / "full/stage1_test05/final.pt", weights_only=False)
    resumed_dir = tmp_path / "resume/stage1_test05"
    resumed_state = torch.load(resumed_dir / "final.pt", weights_only=False)
    for key, value in full_state["state"]["denoiser"].items():
        torch.testing.assert_close(value, resumed_state["state"]["denoiser"][key], rtol=0, atol=0)
    # Test real checkpoint reconstruction while replacing just the external
    # pretrained Kong factory with the same deterministic tiny encoder.
    def encoder_factory(*args, **kwargs):
        torch.manual_seed(interrupted.seed)
        return FakeEncoder(frames=1001)
    monkeypatch.setattr(ev.TunableKongEncoder, "from_checkpoint", encoder_factory)
    model, _ = ev.load_model(resumed_dir / "final.pt", torch.device("cpu"))
    result = ev.evaluate(model, resumed_dir, vars(interrupted), torch.device("cpu"))
    assert set(result["test"]) == {"one_pass", "diffusion1", "diffusion2"}
    assert all("frame" in score and "tab_offset" in score for score in result["test"].values())
    notes = np.load(resumed_dir / "test_notes_diffusion2.npz")
    assert set(notes.files) == {"track2__est", "track2__ref"}  # only held-out player
    assert (resumed_dir / "eval.json").exists()
    assert json.loads((resumed_dir / "metrics.json").read_text())["selection"].startswith("last epoch")


def test_real_kong_frozen_training_and_generation():
    checkpoint = SRC / "kong_wrapper/checkpoints/kong_note_pedal.pth"
    kong_repo = SRC / "kong_wrapper/third_party/piano_transcription"
    if not checkpoint.exists() or not (kong_repo / "pytorch/models.py").exists():
        pytest.skip("pretrained Kong checkpoint/source unavailable")
    pytest.importorskip("torchlibrosa")
    args = tr.parse_args(["--checkpoint", str(checkpoint), "--kong_repo", str(kong_repo)])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = tr.build_model(args, device).train()
    batch = {"waveform": torch.randn(1, 160000, device=device) * 0.02,
             "tab": torch.zeros(1, 1001, 6, device=device, dtype=torch.long),
             "tab_onset": torch.zeros(1, 1001, 6, device=device, dtype=torch.bool),
             "n_valid_frames": torch.tensor([1001], device=device)}
    batch["tab"][:, 10:50, 0] = 4
    batch["tab_onset"][:, 10, 0] = True
    with tr.autocast(device, True):
        features = model.features(batch["waveform"])
        losses = model.training_loss(batch, features=features)
    losses["loss"].backward()
    assert all(p.grad is None for p in model.encoder.note_model.parameters())
    assert model.denoiser.acoustic.weight.grad is not None
    optimizer = torch.optim.AdamW(model.denoiser.parameters(), lr=3e-4)
    optimizer.step()
    output = model.eval().generate(None, batch["n_valid_frames"], steps=2, features=features.float())
    assert output["tab"].shape == (1, 1001, 6)
    assert int(output["tab"].max()) <= 20


def test_summary_fold_statistics_and_settings_consistency(tmp_path):
    from summarize_diffusion import summarize
    for player, value in [("00", 0.4), ("01", 0.6)]:
        directory = tmp_path / f"stage1_test{player}"
        directory.mkdir()
        scores = {k: {"pooled": {"f1": value}} for k in (
            "note_onset", "note_offset", "tab_onset", "tab_offset")}
        scores.update(playability={"pooled": {"rate": value}},
                      frame={"pitch": {"f1": value}, "tab": {"f1": value}, "tdr": value})
        record = {"test": {"one_pass": scores}, "decodings": {"one_pass": {"steps": 1}},
                  "decode_stats": {"one_pass": {"seconds_per_window": 1}}}
        (directory / "eval.json").write_text(json.dumps(record))
    result = summarize(tmp_path, ["00", "01"])["results"]["one_pass"]
    assert result["mean"]["tdr"] == pytest.approx(0.5)
    assert result["std"]["tdr"] == pytest.approx(np.std([0.4, 0.6], ddof=1))
    path = tmp_path / "stage1_test01/eval.json"
    changed = json.loads(path.read_text())
    changed["decodings"]["one_pass"]["steps"] = 2
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="different decoding"):
        summarize(tmp_path, ["00", "01"])
