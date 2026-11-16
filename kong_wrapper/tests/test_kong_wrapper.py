
"""Offline unit tests: exercise hooks, fusion and gradient behavior with 
a fake note model.
"""

from pathlib import Path
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fusion_adapter import BRANCH_NAMES, ResidualKongFusion
from kong_encoder import KongEncoder, _extract_note_state


class FakeGRU(nn.Module):
    """Small differentiable stand-in for GRU's (sequence, hidden) contract."""
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.randn(512) * 0.01)
        self.bias = nn.Parameter(torch.randn(512) * 0.01)

    def forward(self, x):
        # Summarize the input's last dimension; create 512 features per frame.
        sequence = x.mean(dim=-1, keepdim=True) * self.scale + self.bias
        return sequence, None


class FakeBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = FakeGRU()

    def forward(self, x):
        features, _ = self.gru(x)
        return torch.sigmoid(features[..., :88])


class FakeNoteModel(nn.Module):
    """Same hook names and outputs as official Kong, but tiny computations."""
    def __init__(self):
        super().__init__()
        self.frame_model = FakeBranch()
        self.reg_onset_model = FakeBranch()
        self.reg_offset_model = FakeBranch()
        self.velocity_model = FakeBranch()
        self.reg_onset_gru = FakeGRU()
        self.frame_gru = FakeGRU()
        self.dropout = nn.Dropout(p=0.5)
        self.bn = nn.BatchNorm1d(1)
        self.classifier = nn.Linear(512, 88)

    def forward(self, waveform):
        x = waveform[:, ::160].unsqueeze(-1)
        x = torch.cat((x, waveform[:, -1:].unsqueeze(-1)), dim=1)
        x = self.dropout(x)
        # BN is only for checking frozen/eval semantics, not central to features
        x = self.bn(x.transpose(1, 2)).transpose(1, 2)
        frame = self.frame_model(x)
        onset = self.reg_onset_model(x)
        offset = self.reg_offset_model(x)
        velocity = self.velocity_model(x)
        refined, _ = self.reg_onset_gru(torch.cat((onset, velocity), dim=-1))
        final, _ = self.frame_gru(torch.cat((frame, offset, refined), dim=-1))
        return {
            "frame_output": torch.sigmoid(self.classifier(final)),
            "reg_onset_output": onset,
            "reg_offset_output": offset,
            "velocity_output": velocity,
        }


def sample_audio(batch=2, seconds=2):
    return torch.randn(batch, seconds * 16000)


def test_adapter_initially_identity_and_allows_learning():
    fusion = ResidualKongFusion(dropout=0.0)
    pretrained = torch.randn(2, 21, 512)
    branches = {name: torch.randn_like(pretrained) for name in BRANCH_NAMES}
    torch.testing.assert_close(fusion(pretrained, branches), pretrained, rtol=0, atol=0)
    # Gradient enters the new output projection immediately.
    fusion(pretrained, branches).square().mean().backward()
    assert fusion.adapter[-1].weight.grad is not None
    assert fusion.adapter[-1].weight.grad.abs().sum() > 0
    assert fusion.adapter[-1].bias.grad is not None


def test_wrapper_feature_contract_and_freezing():
    encoder = KongEncoder(FakeNoteModel(), backbone_frozen=True)
    encoder.train()
    assert not encoder.note_model.training
    assert not any(param.requires_grad for param in encoder.note_model.parameters())
    outputs = encoder(sample_audio())
    assert outputs["features"].shape == (2, 201, 512)
    assert outputs["pretrained_features"].shape == (2, 201, 512)
    assert outputs["refined_onset_features"].shape == (2, 201, 512)
    assert set(outputs["branch_features"]) == set(BRANCH_NAMES)
    for value in outputs["branch_features"].values():
        assert value.shape == (2, 201, 512)
    assert set(outputs["piano"]) == {
        "frame_output", "reg_onset_output", "reg_offset_output", "velocity_output"
    }
    assert all(v.shape == (2, 201, 88) for v in outputs["piano"].values())
    assert outputs["features"] is outputs["pretrained_features"]
    assert not outputs["features"].requires_grad


def test_frozen_kong_adapter_and_downstream_head_receive_gradients():
    encoder = KongEncoder(
        FakeNoteModel(), ResidualKongFusion(dropout=0.0), backbone_frozen=True
    )
    encoder.train()
    head = nn.Linear(512, 126)
    logits = head(encoder(sample_audio())["features"])
    logits.square().mean().backward()
    assert head.weight.grad is not None
    assert encoder.fusion_adapter.adapter[-1].weight.grad is not None
    assert encoder.fusion_adapter.adapter[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in encoder.note_model.parameters())
    assert encoder.note_model.bn.running_mean.sum() == 0  # frozen BN eval


def test_unfrozen_kong_receives_gradients():
    encoder = KongEncoder(FakeNoteModel(), backbone_frozen=False)
    encoder.train()
    assert encoder.note_model.training
    features = encoder(sample_audio())["features"]
    features.square().mean().backward()
    assert encoder.note_model.frame_gru.scale.grad is not None
    assert encoder.note_model.frame_gru.scale.grad.abs().sum() > 0


def test_input_checks_and_cache_clearing():
    encoder = KongEncoder(FakeNoteModel())
    with pytest.raises(ValueError):
        encoder(torch.randn(2, 3, 32000))
    with pytest.raises(TypeError):
        encoder(torch.zeros(2, 32000, dtype=torch.int16))
    with pytest.raises(ValueError):
        encoder(torch.randn(2, 1000))
    assert encoder._activation_cache == {}
    encoder(sample_audio())
    assert encoder._activation_cache == {}


def test_checkpoint_formats():
    state = {"foo": torch.tensor([1.0])}
    assert _extract_note_state({"model": {"note_model": state, "pedal_model": {}}}) == state
    assert _extract_note_state({"model": state}) == state
    assert _extract_note_state(state) == state
    stripped = _extract_note_state({"model": {"module.foo": torch.tensor([1.0])}})
    assert list(stripped) == ["foo"]
    with pytest.raises(ValueError):
        _extract_note_state({"model": "not a state dict"})
