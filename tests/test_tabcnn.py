"""Unit tests for the PyTorch TabCNN port: model shapes, loss/metric behavior,
the metric functions, and the dataset's context-window loading.
"""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "tabcnn"))
from dataset import TabFrameDataset, partition_ids
from metrics import pitch_precision, pitch_recall, tab_precision
from tabcnn import NUM_CLASSES, NUM_STRINGS, SPEC_REPR_BINS, TabCNN, avg_acc, tab_loss


def test_forward_shape():
    model = TabCNN()
    x = torch.randn(4, 1, SPEC_REPR_BINS["c"], 9)
    out = model(x)
    assert out.shape == (4, NUM_STRINGS, NUM_CLASSES)


@pytest.mark.parametrize("spec_repr", list(SPEC_REPR_BINS))
def test_forward_all_spec_reprs(spec_repr):
    model = TabCNN()
    out = model(torch.randn(2, 1, SPEC_REPR_BINS[spec_repr], 9))
    assert out.shape == (2, NUM_STRINGS, NUM_CLASSES)


def test_perfect_prediction_zero_loss_full_acc():
    # One-hot targets; feed huge logits on the right class -> loss ~0, acc 1.
    targets = torch.zeros(3, NUM_STRINGS, NUM_CLASSES)
    idx = torch.randint(0, NUM_CLASSES, (3, NUM_STRINGS))
    targets.scatter_(-1, idx.unsqueeze(-1), 1.0)
    logits = (targets - 0.5) * 50.0
    assert tab_loss(logits, targets).item() == pytest.approx(0.0, abs=1e-4)
    assert avg_acc(logits, targets).item() == pytest.approx(1.0)


def test_loss_matches_manual_crossentropy():
    torch.manual_seed(0)
    logits = torch.randn(5, NUM_STRINGS, NUM_CLASSES)
    targets = torch.zeros(5, NUM_STRINGS, NUM_CLASSES)
    idx = torch.randint(0, NUM_CLASSES, (5, NUM_STRINGS))
    targets.scatter_(-1, idx.unsqueeze(-1), 1.0)
    # Reference: sum over strings of class-index cross-entropy, mean over batch.
    ce = torch.nn.functional.cross_entropy(
        logits.reshape(-1, NUM_CLASSES), idx.reshape(-1), reduction="none"
    ).reshape(5, NUM_STRINGS)
    expected = ce.sum(dim=1).mean()
    assert tab_loss(logits, targets).item() == pytest.approx(expected.item(), rel=1e-5)


def test_gradients_flow():
    model = TabCNN()
    x = torch.randn(2, 1, SPEC_REPR_BINS["c"], 9)
    targets = torch.zeros(2, NUM_STRINGS, NUM_CLASSES)
    targets[..., 0] = 1.0
    tab_loss(model(x), targets).backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def _onehot(frame_frets):
    """(6,) fret classes -> (6, 21) one-hot frame."""
    y = np.zeros((NUM_STRINGS, NUM_CLASSES))
    for s, c in enumerate(frame_frets):
        y[s, c] = 1.0
    return y


def test_metrics_perfect_match():
    # Two frames; some strings played, some closed.
    gt = np.stack([_onehot([0, 5, 0, 3, 0, 1]), _onehot([2, 0, 0, 0, 7, 0])])
    assert pitch_precision(gt, gt) == pytest.approx(1.0)
    assert pitch_recall(gt, gt) == pytest.approx(1.0)
    assert tab_precision(gt, gt) == pytest.approx(1.0)


def test_partition_leaves_one_guitarist_out():
    ids = [f"{g}_clip_{f}" for g in range(6) for f in range(3)]
    part = partition_ids(ids, val_split=2)
    assert all(i.startswith("2_") for i in part["validation"])
    assert not any(i.startswith("2_") for i in part["training"])
    assert len(part["training"]) + len(part["validation"]) == len(ids)


def test_dataset_context_window(tmp_path):
    bins, con_win = SPEC_REPR_BINS["c"], 9
    spec_dir = tmp_path / "c"
    spec_dir.mkdir()
    n_frames = 20
    repr_arr = np.random.randn(n_frames, bins).astype(np.float32)
    labels = np.zeros((n_frames, NUM_STRINGS, NUM_CLASSES), dtype=np.float32)
    labels[:, :, 0] = 1.0
    np.savez(spec_dir / "00_song.npz", repr=repr_arr, labels=labels)

    ids = [f"00_song_{i}" for i in range(n_frames)]
    ds = TabFrameDataset(ids, data_path=tmp_path, spec_repr="c", con_win_size=con_win)
    x, y = ds[0]
    assert x.shape == (1, bins, con_win)
    assert y.shape == (NUM_STRINGS, NUM_CLASSES)
    # Frame 0 is at a clip edge: the first halfwin columns are zero-padded.
    assert torch.count_nonzero(x[0, :, : con_win // 2]) == 0
    # The center column is the frame itself.
    assert torch.allclose(x[0, :, con_win // 2], torch.from_numpy(repr_arr[0]))
