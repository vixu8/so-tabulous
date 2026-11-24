"""Mathematics, locality, padding, optimization and reproducible diffusion inference."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch import nn

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC / "diffusion"))
from categorical_diffusion import CategoricalDiffusion, diffusion_loss  # noqa: E402
from kong_tab_diffusion import KongTabDiffusion  # noqa: E402
from tab_denoiser import LocalTemporalAttention, MASK, TabDenoiser, valid_frames  # noqa: E402


@pytest.fixture(autouse=True)
def reproducible_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(7)
    yield
    torch.set_num_threads(previous)


class FakeEncoder(nn.Module):
    feature_dim = 512

    def __init__(self, frames=9, adapter=False):
        super().__init__()
        self.frames, self.calls = frames, 0
        self.note_model = nn.Linear(1, 512)
        self.fusion_adapter = nn.Linear(512, 512) if adapter else None

    def set_finetune(self, mode):
        assert mode == "frozen"
        self.note_model.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.note_model.eval()
        return self

    def forward(self, waveform):
        self.calls += 1
        features = self.note_model(waveform[:, :self.frames, None])
        if self.fusion_adapter is not None:
            features = features + self.fusion_adapter(features)
        return {"features": features}


def tiny_model(frames=9, adapter=False):
    return KongTabDiffusion(FakeEncoder(frames, adapter), timesteps=10, hidden_dim=16,
                            embed_dim=8, lstm_hidden=8, heads=4, d_ff=32,
                            dilations=(1, 2), dropout=0)


def batch(frames=9, device="cpu"):
    return {"waveform": torch.randn(2, max(frames, 32), device=device),
            "tab": torch.randint(0, 21, (2, frames, 6), device=device),
            "tab_onset": torch.rand(2, frames, 6, device=device) < 0.1,
            "n_valid_frames": torch.tensor([frames, frames - 3], device=device)}


@pytest.mark.parametrize("schedule", ["linear", "cosine"])
def test_forward_endpoints_categories_and_padding(schedule):
    process = CategoricalDiffusion(100, schedule)
    data = batch()
    clean, valid = data["tab"], data["n_valid_frames"]
    keep = valid_frames(valid, 9, "cpu")[..., None].expand_as(clean)
    torch.testing.assert_close(process.q_sample(clean, torch.zeros(2, dtype=torch.long), valid)[keep], clean[keep])
    maximum = process.q_sample(clean, torch.full((2,), 100), valid)
    assert (maximum[keep] == MASK).all()
    assert (maximum[~keep] == 0).all()
    for t in (1, 25, 50, 99):
        noisy = process.q_sample(clean, torch.full((2,), t), valid)
        assert ((noisy[keep] == clean[keep]) | (noisy[keep] == MASK)).all()
    assert (process.alpha_bar[1:] <= process.alpha_bar[:-1]).all()


def test_corruption_matches_declared_probability():
    process = CategoricalDiffusion()
    clean = torch.ones(200, 100, 6, dtype=torch.long)
    noisy = process.q_sample(clean, torch.full((200,), 25), torch.full((200,), 100))
    assert abs(float((noisy == MASK).float().mean()) - 0.25) < 0.005
    sampled = process.sample_timesteps(20000, "cpu")
    assert set(sampled.tolist()) == set(range(101))


@pytest.mark.parametrize("steps", [1, 10, 25, 50, 100])
def test_respacing_and_normalized_posterior(steps):
    process = CategoricalDiffusion()
    schedule = process.inference_schedule(steps)
    assert schedule[0] == 100 and schedule[-1] == 0 and len(schedule) == steps + 1
    assert all(t > s for t, s in zip(schedule[:-1], schedule[1:]))
    clean = torch.randint(0, 21, (1, 3, 6))
    noisy = torch.full_like(clean, MASK)
    noisy[0, 0, 0] = 7
    logits = torch.randn(1, 3, 6, 21)
    for t, s in zip(schedule[:-1], schedule[1:]):
        posterior = process.reverse_probabilities(noisy, logits, t, s)
        torch.testing.assert_close(posterior.sum(-1), torch.ones_like(noisy, dtype=torch.float32))
        assert (posterior >= 0).all()
        assert posterior[0, 0, 0, 7] == 1
        noisy = process.reverse_step(noisy, logits, t, s, torch.tensor([3]))
        assert noisy[0, 0, 0] == 7
    assert (noisy < MASK).all()


def test_skipped_transition_composes_and_empirical_reveal_rate():
    process = CategoricalDiffusion()
    direct = process.reveal_probability(90, 20)
    first = process.reveal_probability(90, 50)
    second = process.reveal_probability(50, 20)
    torch.testing.assert_close(direct, first + (1 - first) * second)
    noisy = torch.full((100, 100, 6), MASK)
    logits = torch.full((*noisy.shape, 21), -100.0)
    logits[..., 4] = 100
    sample = process.reverse_step(noisy, logits, 90, 20, torch.full((100,), 100))
    assert abs(float((sample == 4).float().mean()) - float(direct)) < 0.008
    assert ((sample == 4) | (sample == MASK)).all()


@pytest.mark.parametrize("dilation", [1, 2, 4, 8])
def test_local_attention_uses_exact_neighbors_and_masks(dilation):
    # Zero q/k => uniform attention. Identity v/output makes the result the
    # arithmetic mean of just the valid dilated neighbors, including boundaries.
    attention = LocalTemporalAttention(4, 1, window=5, dilation=dilation, dropout=0).eval()
    with torch.no_grad():
        attention.qkv.weight.zero_()
        attention.qkv.bias.zero_()
        attention.qkv.weight[8:] = torch.eye(4)
        attention.out.weight.copy_(torch.eye(4))
        attention.out.bias.zero_()
    x = torch.arange(13).float()[None, :, None].expand(2, 13, 4).clone()
    keep = torch.arange(13)[None] < torch.tensor([13, 7])[:, None]
    result = attention(x, keep)
    for row, length in enumerate([13, 7]):
        for t in range(length):
            neighbors = [t + j * dilation for j in range(-2, 3) if 0 <= t + j * dilation < length]
            torch.testing.assert_close(result[row, t], x[row, neighbors].mean(0))
    assert (result[1, 7:] == 0).all()
    assert torch.isfinite(result).all()
    # A directional relative bias must distinguish past from future.
    with torch.no_grad():
        attention.bias[0, 1] = 5
    if dilation < 7:
        assert not torch.allclose(result[0, 6], attention(x, keep)[0, 6])


def test_default_shapes_including_full_window():
    denoiser = TabDenoiser(dropout=0).eval()
    observed = {}
    def capture(name):
        def hook(module, inputs, output):
            observed[name] = tuple(output.shape)
        return hook
    handles = [denoiser.condition.register_forward_hook(capture("condition"))]
    handles += [block.register_forward_hook(capture(f"block{i}")) for i, block in enumerate(denoiser.blocks)]
    features = torch.randn(1, 1001, 512)
    acoustics = denoiser.project_acoustics(features)
    assert acoustics.shape == (1, 1001, 6, 64)
    with torch.no_grad():
        result = denoiser(torch.full((1, 1001, 6), MASK), acoustics, torch.tensor([100]), torch.tensor([1001]))
    assert result["fret_logits"].shape == (1, 1001, 6, 21)
    assert result["onset_logits"].shape == (1, 1001, 6)
    assert all(shape == (1, 1001, 6, 64) for shape in observed.values())
    assert len(observed) == 5
    assert (result["fret_logits"][:, 1000] == 0).all()
    for handle in handles:
        handle.remove()


def test_padding_cannot_change_valid_outputs_in_backward_lstm_or_attention():
    model = tiny_model().eval()
    denoiser = model.denoiser
    clean = torch.randint(0, 22, (1, 9, 6))
    acoustic = torch.randn(1, 9, 6, 16)
    clean[:, 6:] = 999  # ignored sentinel padding never reaches an embedding
    result = denoiser(clean, acoustic, torch.tensor([3]), torch.tensor([6]))
    prefix = denoiser(clean[:, :6], acoustic[:, :6], torch.tensor([3]), torch.tensor([6]))
    for key in result:
        torch.testing.assert_close(result[key][:, :6], prefix[key], atol=2e-6, rtol=2e-5)
    empty = denoiser(clean, acoustic, torch.tensor([3]), torch.tensor([0]))
    assert all(torch.isfinite(value).all() and (value == 0).all() for value in empty.values())


def test_loss_masks_padding_and_reports_active_and_corrupted_accuracy():
    data = batch()
    clean, valid = data["tab"], data["n_valid_frames"]
    noisy = CategoricalDiffusion(10).q_sample(clean, torch.tensor([10, 10]), valid)
    outputs = {"fret_logits": torch.randn(2, 9, 6, 21, requires_grad=True),
               "onset_logits": torch.randn(2, 9, 6, requires_grad=True)}
    losses = diffusion_loss(outputs, clean, noisy, data["tab_onset"], valid)
    assert int(losses["correct"][1]) == 15 * 6
    assert int(losses["corrupted_correct"][1]) == 15 * 6
    altered = clean.clone()
    altered[1, 6:] = -100
    other = diffusion_loss(outputs, altered, noisy, ~data["tab_onset"], valid, lambda_onset=0)
    fret_only = diffusion_loss(outputs, clean, noisy, data["tab_onset"], valid, lambda_onset=0)
    torch.testing.assert_close(other["loss"], fret_only["loss"])
    losses["loss"].backward()
    assert (outputs["fret_logits"].grad[1, 6:] == 0).all()
    assert (outputs["onset_logits"].grad[1, 6:] == 0).all()


def test_loss_weights_select_corrupted_positions_and_onset_positives():
    clean = torch.zeros(1, 2, 6, dtype=torch.long)
    noisy = clean.clone()
    noisy[:, 0] = MASK
    logits = torch.zeros(1, 2, 6, 21)
    logits[:, 0, :, 0] = 10
    logits[:, 1, :, 1] = 10
    outputs = {"fret_logits": logits, "onset_logits": torch.zeros(1, 2, 6)}
    onsets = torch.ones_like(clean, dtype=torch.bool)
    masked = diffusion_loss(outputs, clean, noisy, onsets, [2], masked_weight=0.01, unmasked_weight=0)
    unmasked = diffusion_loss(outputs, clean, noisy, onsets, [2], masked_weight=0, unmasked_weight=1)
    assert float(masked["fret_loss"]) < 0.01
    assert float(unmasked["fret_loss"]) > 9
    positive = diffusion_loss(outputs, clean, noisy, onsets, [2], onset_pos_weight=3)
    torch.testing.assert_close(positive["onset_loss"], 3 * masked["onset_loss"])


@pytest.mark.parametrize("adapter", [False, True])
def test_frozen_backbone_gradients_and_trainable_adapter(adapter):
    model = tiny_model(adapter=adapter).train()
    losses = model.training_loss(batch())
    losses["loss"].backward()
    assert all(p.grad is None and not p.requires_grad for p in model.encoder.note_model.parameters())
    assert all(p.grad is not None for p in model.denoiser.parameters())
    if adapter:
        assert all(p.grad is not None for p in model.encoder.fusion_adapter.parameters())
    assert not model.encoder.note_model.training


@pytest.mark.parametrize("method,steps", [("one_pass", 1), ("diffusion", 1), ("diffusion", 5), ("diffusion", 10)])
def test_sampling_reuses_acoustics_and_has_no_masks(method, steps):
    model = tiny_model().eval()
    data = batch()
    output = model.generate(data["waveform"], data["n_valid_frames"], steps=steps, method=method)
    assert model.encoder.calls == 1
    assert output["tab"].shape == (2, 9, 6)
    assert int(output["tab"].max()) < MASK and int(output["tab"].min()) >= 0
    assert (output["tab"][1, 6:] == 0).all()
    assert (output["onset_probs"][1, 6:] == 0).all()


def test_checkpoint_reproduces_inference(tmp_path):
    from train_diffusion import restore, snapshot
    model = tiny_model().eval()
    data = batch()
    path = tmp_path / "checkpoint.pt"
    torch.save(snapshot(model), path)
    expected = model.generate(data["waveform"], data["n_valid_frames"], steps=5,
                              generator=torch.Generator().manual_seed(8))
    restored = tiny_model().eval()
    # The pretrained backbone is external to decoder checkpoints, as in AR.
    restored.encoder.load_state_dict(model.encoder.state_dict())
    restore(restored, torch.load(path, weights_only=True))
    actual = restored.generate(data["waveform"], data["n_valid_frames"], steps=5,
                               generator=torch.Generator().manual_seed(8))
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key])


def test_single_synthetic_clip_overfit():
    model = tiny_model().train()
    data = batch()
    data = {k: value[:1] for k, value in data.items()}
    data["tab"][:] = torch.tensor([0, 1, 5, 10, 15, 20])
    data["tab_onset"][:] = False
    features = model.features(data["waveform"]).detach()
    noisy = torch.full_like(data["tab"], MASK)
    optimizer = torch.optim.AdamW(model.denoiser.parameters(), lr=0.02)
    initial = None
    for _ in range(100):
        optimizer.zero_grad()
        outputs = model(None, noisy, torch.tensor([10]), data["n_valid_frames"], features)
        losses = diffusion_loss(outputs, data["tab"], noisy, data["tab_onset"], data["n_valid_frames"])
        if initial is None:
            initial = float(losses["loss"].detach())
        losses["loss"].backward()
        optimizer.step()
    assert float(losses["loss"].detach()) < initial * 0.1
    assert float(losses["correct"][0] / losses["correct"][1]) > 0.98


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_training_batch():
    model = tiny_model().cuda().train()
    optimizer = torch.optim.AdamW(model.denoiser.parameters(), lr=3e-4)
    losses = model.training_loss(batch(device="cuda"))
    losses["loss"].backward()
    optimizer.step()
    assert torch.isfinite(losses["loss"])


def test_repeated_attack_note_conversion_and_window_clipping():
    from eval_diffusion import grid_to_notes
    grid = np.zeros((1001, 6), dtype=np.int64)
    grid[:10, 0] = 6
    onset = np.zeros_like(grid, dtype=np.float32)
    onset[5, 0] = 1
    notes = grid_to_notes(grid, onset, 1001, start_s=10)
    np.testing.assert_allclose(notes, [[0, 5, 10, 10.05], [0, 5, 10.05, 10.1]])
    grid[:, 0] = 6
    clipped = grid_to_notes(grid, np.zeros_like(onset), 8)
    np.testing.assert_allclose(clipped, [[0, 5, 0, 0.07]])
    full = grid_to_notes(grid, np.zeros_like(onset), 1001)
    np.testing.assert_allclose(full, [[0, 5, 0, 10]])


def test_feature_cache_exact_crop_and_frozen_guard(tmp_path):
    from train_diffusion import FeatureCache
    model = tiny_model().eval()
    cache = FeatureCache(tmp_path, model.encoder)
    data = batch()
    expected = cache.get(model, data["waveform"])
    calls = model.encoder.calls
    torch.testing.assert_close(cache.get(model, data["waveform"]), expected)
    assert model.encoder.calls == calls
    cache.get(model, data["waveform"] + 1)
    assert model.encoder.calls == calls + 1
    with pytest.raises(ValueError, match="fully frozen"):
        FeatureCache(tmp_path, tiny_model(adapter=True).encoder)


def test_training_cli_defaults_and_invalid_cache():
    from train_diffusion import parse_args
    args = parse_args([])
    assert (args.batch_size, args.epochs, args.lr, args.timesteps, args.use_adapter) == (8, 25, 3e-4, 100, False)
    assert args.dilations == [1, 2, 4, 8]
    with pytest.raises(SystemExit):
        parse_args(["--use_adapter", "--feature_cache", "cache"])
