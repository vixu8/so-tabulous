"""Integration test with the real pretrained Kong weights (no training required).

Skips unless the per-person setup in src/kong_wrapper/README.md has been done:
the checkpoint in src/kong_wrapper/checkpoints/, the Kong repo cloned into
src/kong_wrapper/third_party/, and requirements/model.txt installed.
"""

from pathlib import Path
import sys

import pytest
import torch

KONG_DIR = Path(__file__).resolve().parents[1] / "src" / "kong_wrapper"
sys.path.insert(0, str(KONG_DIR))
from kong_encoder import KongEncoder  # noqa: E402

CHECKPOINT = KONG_DIR / "checkpoints" / "kong_note_pedal.pth"
REPO = KONG_DIR / "third_party" / "piano_transcription"
SECONDS = 2
EXPECTED_FRAMES = SECONDS * 100 + 1


@pytest.fixture(scope="module")
def outputs():
    if not CHECKPOINT.is_file():
        pytest.skip(f"Kong checkpoint not found at {CHECKPOINT} (see src/kong_wrapper/README.md)")
    if not REPO.is_dir():
        pytest.skip(f"Kong repo not found at {REPO} (see src/kong_wrapper/README.md)")
    try:
        encoder = KongEncoder.from_checkpoint(
            CHECKPOINT, repo_path=REPO, use_adapter=True, backbone_frozen=True
        )
    except ImportError as exc:  # a Kong dependency from requirements/model.txt is missing
        pytest.skip(str(exc))
    encoder.eval()
    audio = torch.zeros(1, 16000 * SECONDS, dtype=torch.float32)
    with torch.no_grad():
        return encoder(audio)


def test_feature_shapes(outputs):
    assert outputs["features"].shape == (1, EXPECTED_FRAMES, 512)
    assert outputs["pretrained_features"].shape == (1, EXPECTED_FRAMES, 512)


def test_all_branches_captured(outputs):
    # 4 CRNN branches captured via hooks, each [B, T, 512]
    assert len(outputs["branch_features"]) == 4
    assert all(x.shape == (1, EXPECTED_FRAMES, 512) for x in outputs["branch_features"].values())


def test_piano_output_shapes(outputs):
    assert all(x.shape == (1, EXPECTED_FRAMES, 88) for x in outputs["piano"].values())


def test_adapter_starts_as_identity(outputs):
    # Zero-initialized last adapter layer -> features == Kong's frame_gru output exactly
    torch.testing.assert_close(outputs["features"], outputs["pretrained_features"], rtol=0, atol=0)


def load_unfrozen():
    if not CHECKPOINT.is_file() or not REPO.is_dir():
        pytest.skip("Kong checkpoint or repo missing (see src/kong_wrapper/README.md)")
    try:
        return KongEncoder.from_checkpoint(CHECKPOINT, repo_path=REPO, backbone_frozen=False)
    except ImportError as exc:
        pytest.skip(str(exc))


def kong_functional(encoder):
    """The `F` that Kong's models.py sees (its private module's globals)."""
    return type(encoder.note_model.frame_model).forward.__globals__


def test_full_finetune_backprops_in_train_mode():
    # Kong's models.py:149 does an in-place dropout on a ReLU output; the wrapper
    # swaps in an out-of-place dropout at load time so backward works in train mode.
    torch.manual_seed(0)
    audio = torch.randn(1, 16000 * SECONDS) * 0.1
    patched = load_unfrozen()
    original = load_unfrozen()
    kong_functional(original)["F"] = torch.nn.functional   # Kong exactly as published

    # Same outputs in eval mode with and without the fix.
    patched.eval(), original.eval()
    with torch.no_grad():
        a, b = patched(audio), original(audio)
    torch.testing.assert_close(a["features"], b["features"], rtol=0, atol=0)
    for key in a["piano"]:
        torch.testing.assert_close(a["piano"][key], b["piano"][key], rtol=0, atol=0)

    # The published code fails in train mode; the fixed one trains, with dropout on.
    original.train()
    with pytest.raises(RuntimeError, match="inplace"):
        original(audio)["piano"]["frame_output"].mean().backward()
    patched.train()
    assert patched.note_model.training
    out = patched(audio)
    assert not torch.equal(out["features"], patched(audio)["features"])   # dropout active
    (out["piano"]["frame_output"].mean() + out["features"].mean()).backward()
    assert patched.note_model.frame_model.fc5.weight.grad.abs().sum() > 0
    assert patched.note_model.frame_model.conv_block1.conv1.weight.grad is not None
