"""Tests for Head A (src/headA/head_a.py).

Replaces the old src/headA/test_head_a.py and test_head_a_training.py scripts. The
unit tests need nothing; the real-Kong tests skip if the checkpoint / repo is missing.
"""

import math
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

SRC = Path(__file__).resolve().parents[1] / "src"
for sub in ("headA", "kong_wrapper"):
    sys.path.insert(0, str(SRC / sub))
from head_a import (IGNORE_INDEX, N_CLASSES, N_STRINGS, KongHeadA, TabCNNHead,  # noqa: E402
                    predict_tab_grid, tab_loss)

KONG_DIR = SRC / "kong_wrapper"
CHECKPOINT = KONG_DIR / "checkpoints" / "kong_note_pedal.pth"
KONG_REPO = KONG_DIR / "third_party" / "piano_transcription"
needs_kong = pytest.mark.skipif(not (CHECKPOINT.is_file() and KONG_REPO.is_dir()),
                                reason="Kong checkpoint / repo missing (see src/kong_wrapper/README.md)")


def test_head_shape():
    head = TabCNNHead(input_dim=512)
    assert head(torch.randn(2, 7, 512)).shape == (2, 7, N_STRINGS, N_CLASSES)


def test_pretrained_classifier_transfer_and_roundtrip(tmp_path):
    from head_a import load_tabcnn_head
    classifier = nn.Sequential(nn.Linear(5952, 128), nn.ReLU(),
                               nn.Dropout(0.5), nn.Linear(128, 126))
    path = tmp_path / "weights.pt"
    torch.save({f"classifier.{k}": v for k, v in classifier.state_dict().items()}, path)
    head = load_tabcnn_head(path)
    for key, value in classifier.state_dict().items():
        torch.testing.assert_close(head.dense.state_dict()[key], value)
    features = torch.randn(2, 3, 512)
    head.eval()
    expected = classifier.eval()(head.projection(features)).reshape(2, 3, 6, 21)
    torch.testing.assert_close(head(features), expected)
    restored = TabCNNHead(512, head.dense[0].in_features).eval()
    restored.load_state_dict(head.state_dict())
    torch.testing.assert_close(restored(features), expected)
    head(features).sum().backward()
    assert head.projection.weight.grad is not None
    assert head.dense[0].weight.grad is not None


def test_loss_uniform_and_ignores_padding():
    logits = torch.zeros(2, 10, N_STRINGS, N_CLASSES)
    tab = torch.randint(0, N_CLASSES, (2, 10, N_STRINGS))
    torch.testing.assert_close(KongHeadA.loss(logits, tab), torch.tensor(6 * math.log(21)))

    logits = torch.randn(2, 10, N_STRINGS, N_CLASSES)
    n_valid = torch.tensor([10, 4])
    before = tab_loss(logits, tab, n_valid)
    logits[1, 4:] = 50.0                        # padded frames of item 1
    tab[1, 4:] = 7
    torch.testing.assert_close(tab_loss(logits, tab, n_valid), before)
    masked = tab.clone()
    masked[1, 4:] = IGNORE_INDEX
    torch.testing.assert_close(KongHeadA.loss(logits, masked), before)


def test_predict_tab_grid_argmax_vs_threshold():
    probs = torch.zeros(1, 2, N_STRINGS, N_CLASSES)
    probs[..., 0] = 1.0                          # every string silent ...
    probs[0, 0, 3, 0], probs[0, 0, 3, 6] = 0.6, 0.4    # ... except string 3: p(fret 5) = 0.4
    assert predict_tab_grid(probs)[0, 0, 3] == 0                  # argmax: silent
    assert predict_tab_grid(probs, 0.3)[0, 0, 3] == 6             # 1 - 0.6 >= 0.3: fret 5
    assert predict_tab_grid(probs, 0.5)[0, 0, 3] == 0
    assert (predict_tab_grid(probs, 0.3)[0, 1] == 0).all()


def test_set_stage_needs_tunable_encoder():
    class Plain(nn.Module):
        feature_dim = 512
    model = KongHeadA(Plain())
    with pytest.raises(TypeError, match="TunableKongEncoder"):
        model.set_stage(1)
    with pytest.raises(ValueError):
        model.set_stage(3)


# ----- real Kong (CPU, 2 s of synthetic audio) -----

def make_model():
    from lora import TunableKongEncoder
    torch.manual_seed(0)
    encoder = TunableKongEncoder.from_checkpoint(CHECKPOINT, repo_path=KONG_REPO, use_adapter=True)
    return KongHeadA(encoder)


def one_step(model):
    torch.manual_seed(1)
    x = torch.randn(1, 32000) * 0.1
    tab = torch.randint(0, N_CLASSES, (1, 201, N_STRINGS))
    model.train()
    logits = model(x)
    assert logits.shape == (1, 201, N_STRINGS, N_CLASSES)
    loss = tab_loss(logits, tab, torch.tensor([201]))
    assert torch.isfinite(loss)
    loss.backward()


@needs_kong
def test_stage1_trains_adapter_and_head_only():
    model = make_model()
    model.set_stage(1)
    one_step(model)
    kong = list(model.encoder.note_model.parameters())
    assert not any(p.requires_grad for p in kong) and all(p.grad is None for p in kong)
    # The adapter's last layer is zero-initialized, so only it (not earlier layers) gets
    # a nonzero gradient on the first step.
    for p in (model.head.dense[-1].weight, model.encoder.fusion_adapter.adapter[-1].weight):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0


@needs_kong
def test_stage2_trains_lora_adapter_and_head():
    from lora import lora_parameters
    model = make_model()
    model.set_stage(2, rank=4, alpha=8)
    lora = lora_parameters(model.encoder.note_model)
    expected = {id(p) for p in [*lora, *model.encoder.fusion_adapter.parameters(),
                                *model.head.parameters()]}
    assert {id(p) for p in model.parameters() if p.requires_grad} == expected
    one_step(model)
    b_grads = [p.grad for p in lora if p.shape[1] == 4]          # B: (out, rank)
    assert any(g is not None and g.abs().sum() > 0 for g in b_grads)
