"""LoRA / fine-tuning modes for the Kong encoder (src/kong_wrapper/lora.py).

The first group uses a toy module with real nn.Linear and nn.GRU layers; the
second loads the real Kong checkpoint and skips if the per-person setup is missing.
"""

import copy
from pathlib import Path
import sys

import pytest
import torch
from torch import nn
from torch.nn.utils import parametrize

KONG_DIR = Path(__file__).resolve().parents[1] / "src" / "kong_wrapper"
sys.path.insert(0, str(KONG_DIR))
from lora import (LoRADelta, TunableKongEncoder, add_lora, has_lora,  # noqa: E402
                  load_lora_state_dict, lora_parameters, lora_state_dict, lora_targets,
                  merge_lora)

CHECKPOINT = KONG_DIR / "checkpoints" / "kong_note_pedal.pth"
REPO = KONG_DIR / "third_party" / "piano_transcription"


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(6, 10)
        self.gru = nn.GRU(10, 4, num_layers=2, batch_first=True, bidirectional=True)
        self.out = nn.Linear(8, 3)

    def forward(self, x):
        return self.out(self.gru(torch.relu(self.fc(x)))[0])


def toy_and_input(seed=0):
    torch.manual_seed(seed)
    return Toy(), torch.randn(2, 7, 6)


def nudge(module, seed=1):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in lora_parameters(module):
            p.add_(torch.randn(p.shape, generator=g) * 0.1)


# ----- toy module -----

def test_targets_are_linear_and_gru_weights_only():
    toy, _ = toy_and_input()
    targets = lora_targets(toy)
    gru = [w for m, w in targets if m == "gru"]
    assert ("fc", "weight") in targets and ("out", "weight") in targets
    assert len(gru) == 8 and all(w.startswith("weight_") for w in gru)   # 2 layers x 2 dirs x ih/hh
    assert lora_targets(toy, kinds=("linear",)) == [("fc", "weight"), ("out", "weight")]
    with pytest.raises(ValueError):
        lora_targets(toy, kinds=("conv",))


def test_fresh_lora_is_identity():
    toy, x = toy_and_input()
    before = toy(x)
    names = add_lora(toy, rank=2, alpha=4)
    assert len(names) == 10 and has_lora(toy)
    torch.testing.assert_close(toy(x), before, rtol=0, atol=0)
    with pytest.raises(ValueError, match="already has LoRA"):
        add_lora(toy)


def test_lora_equals_hand_merged_weights():
    toy, x = toy_and_input()
    reference = copy.deepcopy(toy)
    add_lora(toy, rank=2, alpha=4)
    nudge(toy)
    with torch.no_grad():
        for sub_name, weight_name in lora_targets(reference):
            delta = toy.get_submodule(sub_name).parametrizations[weight_name][0]
            w = getattr(reference.get_submodule(sub_name), weight_name)
            w.add_(delta.scale * delta.B @ delta.A)
    torch.testing.assert_close(toy(x), reference(x))


def test_only_lora_trains():
    toy, x = toy_and_input()
    add_lora(toy, rank=2)
    toy.requires_grad_(False)
    for p in lora_parameters(toy):
        p.requires_grad_(True)
    base = {n: p.clone() for n, p in toy.named_parameters() if not p.requires_grad}
    b_before = [m.B.clone() for m in toy.modules() if isinstance(m, LoRADelta)]
    opt = torch.optim.SGD(lora_parameters(toy), lr=0.1)
    toy(x).pow(2).sum().backward()
    opt.step()
    b_after = [m.B for m in toy.modules() if isinstance(m, LoRADelta)]
    assert all(not torch.equal(a, b) for a, b in zip(b_before, b_after))
    for n, p in toy.named_parameters():
        if not p.requires_grad:
            assert p.grad is None and torch.equal(p, base[n])


def test_state_dict_round_trip():
    toy, x = toy_and_input()
    add_lora(toy, rank=3)
    nudge(toy)
    state = lora_state_dict(toy)
    assert len(state) == 20 and all(k.endswith((".A", ".B")) for k in state)
    fresh, _ = toy_and_input()
    add_lora(fresh, rank=3)
    load_lora_state_dict(fresh, state)
    torch.testing.assert_close(fresh(x), toy(x))
    with pytest.raises(KeyError):
        load_lora_state_dict(fresh, {k: v for k, v in list(state.items())[1:]})


def test_merge_lora():
    toy, x = toy_and_input()
    plain_names = {n for n, _ in toy.named_parameters()}
    add_lora(toy, rank=2)
    nudge(toy)
    expected = toy(x)
    assert merge_lora(toy) == 10
    assert not has_lora(toy) and not any(parametrize.is_parametrized(m) for m in toy.modules())
    assert {n for n, _ in toy.named_parameters()} == plain_names
    torch.testing.assert_close(toy(x), expected)


# ----- real Kong checkpoint -----

@pytest.fixture(scope="module")
def kong():
    if not CHECKPOINT.is_file():
        pytest.skip(f"Kong checkpoint not found at {CHECKPOINT} (see src/kong_wrapper/README.md)")
    if not REPO.is_dir():
        pytest.skip(f"Kong repo not found at {REPO} (see src/kong_wrapper/README.md)")
    try:
        return TunableKongEncoder.from_checkpoint(CHECKPOINT, repo_path=REPO, use_adapter=True)
    except ImportError as exc:
        pytest.skip(str(exc))


def run(encoder, audio):
    out = encoder(audio)
    return {"features": out["features"], **out["piano"],
            **{f"branch_{k}": v for k, v in out["branch_features"].items()}}


def batchnorms(encoder):
    return [m for m in encoder.note_model.modules()
            if isinstance(m, nn.modules.batchnorm._BatchNorm)]


def assert_batchnorm_frozen(encoder):
    bns = batchnorms(encoder)
    assert bns and not any(m.training for m in bns)


def batchnorm_stats(encoder):
    return [t.clone() for m in batchnorms(encoder) for t in (m.running_mean, m.running_var)]


def expected_lora_size(note_model, rank):
    return sum(rank * sum(getattr(note_model.get_submodule(m), w).shape)
               for m, w in lora_targets(note_model))


def test_kong_modes(kong):
    torch.manual_seed(0)
    audio = torch.randn(1, 16000 * 2) * 0.1
    assert kong.finetune == "frozen" and not kong.trainable_parameters()
    kong.eval()
    with torch.no_grad():
        frozen = run(kong, audio)

    # LoRA: identical outputs at the start, only A / B trainable, all 10 Linear + 6 GRUs.
    expected = expected_lora_size(kong.note_model, 8)
    kong.set_finetune("lora", rank=8)
    targets = lora_targets(kong.note_model)
    assert len(targets) == 10 + 4 * 8 + 2 * 4          # 10 Linear; 4 branch GRUs x 8; 2 GRUs x 4
    trainable = kong.trainable_parameters()
    assert sum(p.numel() for p in trainable) == expected
    assert {id(p) for p in trainable} == {id(p) for p in lora_parameters(kong.note_model)}
    print(f"\nLoRA rank 8: {expected:,} trainable of "
          f"{sum(p.numel() for p in kong.note_model.parameters()) - expected:,} Kong weights")
    with torch.no_grad():
        lora_out = run(kong, audio)
    for key in frozen:
        torch.testing.assert_close(lora_out[key], frozen[key], rtol=0, atol=0, msg=key)

    # Train mode: dropout on, BatchNorm frozen; backward reaches LoRA weights.
    kong.train()
    assert kong.training and kong.note_model.training and kong.fusion_adapter.training
    assert_batchnorm_frozen(kong)
    stats = batchnorm_stats(kong)
    out = kong(audio)
    assert not torch.equal(out["piano"]["frame_output"], kong(audio)["piano"]["frame_output"])
    (out["piano"]["frame_output"].mean() + out["features"].mean()).backward()
    frame_fc = kong.note_model.frame_fc.parametrizations.weight[0]
    frame_gru = kong.note_model.frame_gru.parametrizations.weight_ih_l0[0]
    assert frame_fc.B.grad.abs().sum() > 0 and frame_gru.B.grad.abs().sum() > 0
    assert all(torch.equal(a, b) for a, b in zip(stats, batchnorm_stats(kong)))
    kong.zero_grad(set_to_none=True)

    # Frozen and back keeps the LoRA weights.
    nudge(kong.note_model)
    state = lora_state_dict(kong.note_model)
    kong.set_finetune("frozen")
    assert not kong.trainable_parameters()
    kong.set_finetune("lora")
    after = lora_state_dict(kong.note_model)
    assert all(torch.equal(state[k], after[k]) for k in state)

    # Full fine-tuning needs LoRA merged first; then everything trains, BatchNorm still frozen.
    with pytest.raises(ValueError, match="merge_lora"):
        kong.set_finetune("full")
    merge_lora(kong.note_model)
    kong.set_finetune("full").train()
    assert kong.note_model.training
    assert_batchnorm_frozen(kong)
    assert len(kong.trainable_parameters()) == len(list(kong.note_model.parameters()))
    stats = batchnorm_stats(kong)
    kong(audio)["piano"]["frame_output"].mean().backward()
    assert kong.note_model.frame_model.fc5.weight.grad is not None
    assert kong.note_model.frame_model.conv_block1.conv1.weight.grad is not None
    assert all(torch.equal(a, b) for a, b in zip(stats, batchnorm_stats(kong)))

    # eval() turns dropout off again: repeated forwards match.
    kong.eval()
    with torch.no_grad():
        torch.testing.assert_close(kong(audio)["features"], kong(audio)["features"], rtol=0, atol=0)


# ----- random-init Kong ("scratch" mode, pretraining ablation) -----

FRONTEND = ("spectrogram_extractor", "logmel_extractor")


@pytest.fixture(scope="module")
def kong_repo():
    if not REPO.is_dir():
        pytest.skip(f"Kong repo not found at {REPO} (see src/kong_wrapper/README.md)")
    return REPO


def test_from_scratch_is_kongs_own_init_not_the_checkpoint(kong_repo):
    from kong_encoder import _make_upstream_note_model
    torch.manual_seed(0)
    enc = TunableKongEncoder.from_scratch(repo_path=kong_repo)
    torch.manual_seed(0)
    plain = _make_upstream_note_model(kong_repo)
    for (name, a), b in zip(enc.note_model.state_dict().items(), plain.state_dict().values()):
        assert torch.equal(a, b), name                     # just Kong's constructor init
    if CHECKPOINT.is_file():
        from kong_encoder import _extract_note_state
        ckpt = _extract_note_state(torch.load(CHECKPOINT, map_location="cpu", weights_only=True))
        w = "frame_model.conv_block1.conv1.weight"
        assert not torch.equal(enc.note_model.state_dict()[w], ckpt[w])
    assert enc.note_model.kong_detach is False


def test_no_detach_forward_same_values_and_gradient_reaches_every_branch(kong_repo):
    torch.manual_seed(0)
    enc = TunableKongEncoder.from_scratch(repo_path=kong_repo)      # no .detach()
    torch.manual_seed(0)
    kept = TunableKongEncoder.from_scratch(repo_path=kong_repo, detach=True)
    audio = torch.randn(2, 16000 * 2) * 0.1
    for e in (enc, kept):
        e.set_finetune("scratch")
        e.eval()
    with torch.no_grad():
        a, b = run(enc, audio), run(kept, audio)
    for k in a:
        torch.testing.assert_close(a[k], b[k], rtol=0, atol=0)

    def reached(e):
        e.train()
        e.zero_grad(set_to_none=True)
        e(audio)["features"].sum().backward()
        return {n.split(".")[0] for n, p in e.note_model.named_parameters()
                if p.grad is not None and p.grad.abs().sum() > 0}

    branches = {"frame_model", "reg_onset_model", "reg_offset_model", "velocity_model",
                "reg_onset_gru", "reg_onset_fc", "frame_gru", "bn0"}
    assert reached(enc) == branches                         # frame_fc only feeds piano outputs
    assert reached(kept) == {"frame_model", "frame_gru", "bn0"}   # Kong's detaches


def test_scratch_mode_trains_batchnorm_and_freezes_frontend(kong_repo):
    torch.manual_seed(0)
    enc = TunableKongEncoder.from_scratch(repo_path=kong_repo).set_finetune("scratch")
    trainable = {id(p) for p in enc.trainable_parameters()}
    for name, p in enc.note_model.named_parameters():
        assert (id(p) in trainable) == (name.split(".")[0] not in FRONTEND), name
    assert sum(p.numel() for p in enc.trainable_parameters()) == 20_218_778
    enc.train()
    assert all(m.training for m in batchnorms(enc))
    before = batchnorm_stats(enc)
    enc(torch.randn(2, 16000 * 2) * 0.1)
    after = batchnorm_stats(enc)
    assert any(not torch.equal(x, y) for x, y in zip(before, after))   # stats update
    enc.eval()
    assert_batchnorm_frozen(enc)
    with pytest.raises(ValueError):
        torch.manual_seed(0)
        lora = TunableKongEncoder.from_scratch(repo_path=kong_repo).set_finetune("lora")
        lora.set_finetune("scratch")                        # LoRA attached: merge first
