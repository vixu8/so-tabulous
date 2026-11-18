"""Tests for src/metrics/tab_metrics.py (Note F1, Tab F1, playability, frame metrics)."""

from pathlib import Path
import sys

import mir_eval
import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
for sub in ("metrics", "tabcnn", "data"):
    sys.path.insert(0, str(REPO / "src" / sub))
from tab_metrics import (OPEN_STRING_MIDI, aggregate, frame_metrics,  # noqa: E402
                         note_metrics, playability, tab_grid_to_notes)
from metrics import pitch_precision, pitch_recall, tab_precision, tab_recall  # noqa: E402
from loader import notes_to_tab_grid  # noqa: E402

REAL_ROOT = REPO / "features" / "v1"

# A small clip: C major chord, then a melody line. [string, fret, onset, offset]
REF = np.array([
    [1, 3, 0.50, 1.50],    # C3
    [2, 2, 0.50, 1.50],    # E3
    [3, 0, 0.50, 1.50],    # G3 (open)
    [4, 1, 0.50, 1.50],    # C4
    [5, 0, 2.00, 2.40],    # E4 (open)
    [5, 3, 2.50, 2.90],    # G4
], dtype=np.float32)


def f1(m, key):
    return m[key]["f1"]


# ----- frame metrics vs. the TabCNN port (moved from test_train_head_c.py) -----

def onehot(tab):
    return np.eye(21)[tab]                          # (N, 6) classes -> (N, 6, 21)


def test_frame_metrics_match_tabcnn():
    rng = np.random.default_rng(0)
    gt = np.where(rng.random((2, 300, 6)) < 0.3, rng.integers(1, 21, (2, 300, 6)), 0)
    pred = np.where(rng.random(gt.shape) < 0.5, gt,
                    np.where(rng.random(gt.shape) < 0.3, rng.integers(1, 21, gt.shape), 0))
    m = frame_metrics(pred, gt, n_valid=[300, 300])
    p1, g1 = onehot(pred.reshape(-1, 6)), onehot(gt.reshape(-1, 6))
    assert m["pitch"]["precision"] == pytest.approx(pitch_precision(p1, g1))
    assert m["pitch"]["recall"] == pytest.approx(pitch_recall(p1, g1))
    assert m["tab"]["precision"] == pytest.approx(tab_precision(p1, g1))
    assert m["tab"]["recall"] == pytest.approx(tab_recall(p1, g1))


def test_frame_metrics_skip_padding_and_frame_1000():
    gt = np.zeros((2, 1001, 6), dtype=np.int64)
    gt[:, :, 0] = 4
    pred = gt.copy()
    pred[0, 1000] = 9          # frame 1000 (= next window's frame 0): ignored
    pred[1, 600:] = 9          # padding in a last window with 600 valid frames: ignored
    m = frame_metrics(pred, gt, n_valid=[1001, 600])
    assert m["tab"]["precision"] == m["tab"]["recall"] == 1.0
    assert m["pitch"]["f1"] == 1.0


# ----- note_metrics -----

def test_perfect_prediction():
    m = note_metrics(REF, REF)
    assert all(v["f1"] == 1.0 for v in m.values())
    assert m["note_onset"]["n_match"] == len(REF)


def test_onset_tolerance_50ms():
    est = REF.copy()
    est[:, 2] += 0.03
    assert f1(note_metrics(REF, est), "note_onset") == 1.0
    est[:, 2] = REF[:, 2] + 0.07
    assert f1(note_metrics(REF, est), "note_onset") == 0.0


def test_wrong_string_same_pitch_hurts_only_tab_f1():
    est = REF.copy()
    est[3] = [3, 5, 0.50, 1.50]                     # C4 on the G string (fret 5), not B fret 1
    assert OPEN_STRING_MIDI[3] + 5 == OPEN_STRING_MIDI[4] + 1
    m = note_metrics(REF, est)
    assert f1(m, "note_onset") == 1.0
    assert m["tab_onset"]["n_match"] == len(REF) - 1


def test_bad_offset_hurts_only_offset_variants():
    est = REF.copy()
    est[:, 3] = est[:, 2] + 0.05                    # far too short
    m = note_metrics(REF, est)
    assert f1(m, "note_onset") == f1(m, "tab_onset") == 1.0
    assert f1(m, "note_offset") == f1(m, "tab_offset") == 0.0


def test_empty_and_zero_length():
    m = note_metrics(REF, np.zeros((0, 4)))
    assert all(v["f1"] == 0.0 and v["n_est"] == 0 for v in m.values())
    assert note_metrics(np.zeros((0, 4)), REF)["note_onset"]["precision"] == 0.0
    est = REF.copy()
    est[:, 3] = est[:, 2]                           # zero-length predictions don't crash
    assert f1(note_metrics(REF, est), "note_onset") == 1.0


def test_fret_out_of_range_raises():
    with pytest.raises(ValueError):
        note_metrics(REF, [[0, 20, 0.5, 1.0]])


def test_matches_mir_eval_on_random_notes():
    rng = np.random.default_rng(1)

    def random_notes(n):
        on = np.sort(rng.uniform(0, 10, n))
        return np.stack([rng.integers(0, 6, n), rng.integers(0, 20, n), on,
                         on + rng.uniform(0.05, 1.0, n)], 1)

    ref, est = random_notes(80), random_notes(80)
    est[:40] = ref[:40] + [0, 0, 0.02, 0.1]         # half close to the reference
    hz = lambda n: 440.0 * 2 ** ((OPEN_STRING_MIDI[n[:, 0].astype(int)] + n[:, 1] - 69) / 12)
    for ratio, key in ((None, "note_onset"), (0.2, "note_offset")):
        expected = mir_eval.transcription.precision_recall_f1_overlap(
            ref[:, 2:4], hz(ref), est[:, 2:4], hz(est), offset_ratio=ratio)
        got = note_metrics(ref, est)[key]
        assert (got["precision"], got["recall"], got["f1"]) == pytest.approx(expected[:3])


# ----- playability -----

def test_playable_chord_and_open_strings():
    p = playability(REF)
    assert p["rate"] == 1.0
    assert p["n_sounding_frames"] == 100 + 40 + 40   # silent gaps not counted


def test_wide_span_not_playable():
    chord = np.array([[0, 1, 0.0, 1.0], [5, 5, 0.0, 1.0]])   # frets 1..5: 5-fret span
    assert playability(chord)["rate"] == 0.0
    assert playability(chord, max_fret_span=5)["rate"] == 1.0
    chord[1, 1] = 4                                          # frets 1..4: fits
    assert playability(chord)["rate"] == 1.0
    chord[1, 1] = 0                                          # open string ignored
    chord = np.vstack([chord, [3, 12, 0.0, 1.0]])            # ... but 1 and 12 don't fit
    assert playability(chord)["n_wide_span_frames"] == 100


def test_same_string_overlap_flagged_but_back_to_back_ok():
    back_to_back = np.array([[2, 2, 0.0, 0.5], [2, 4, 0.5, 1.0]])
    assert playability(back_to_back)["rate"] == 1.0
    overlap = np.array([[2, 2, 0.0, 0.6], [2, 4, 0.5, 1.0]])
    p = playability(overlap)
    assert p["n_same_string_frames"] == 10 and p["n_playable_frames"] == 90


def test_playability_empty():
    assert playability(np.zeros((0, 4)))["rate"] == 0.0


# ----- tab_grid_to_notes -----

def test_grid_round_trip():
    tab, onsets = notes_to_tab_grid(REF, n_frames=301)
    notes = tab_grid_to_notes(tab, onsets=onsets)
    m = note_metrics(REF, notes)
    assert f1(m, "tab_onset") == 1.0 and f1(m, "tab_offset") == 1.0


def test_grid_onsets_split_repeated_note():
    repeated = np.array([[0, 5, 0.0, 0.5], [0, 5, 0.5, 1.0]], dtype=np.float32)
    tab, onsets = notes_to_tab_grid(repeated, n_frames=120)
    assert len(tab_grid_to_notes(tab)) == 1                  # merged without onsets
    assert len(tab_grid_to_notes(tab, onsets=onsets)) == 2
    assert tab_grid_to_notes(tab, start_s=10.0)[0, 2] == 10.0


# ----- aggregate -----

def test_aggregate_pooled_and_mean():
    clip_a = note_metrics(REF, REF)                          # 6/6
    clip_b = note_metrics(REF, REF[:2])                      # 2 of 6, precision 1
    agg = aggregate([{**clip_a, "playability": playability(REF)},
                     {**clip_b, "playability": playability(REF[:2])}])
    pooled = agg["note_onset"]["pooled"]
    assert (pooled["n_match"], pooled["n_est"], pooled["n_ref"]) == (8, 8, 12)
    assert pooled["recall"] == pytest.approx(8 / 12) and pooled["precision"] == 1.0
    assert agg["note_onset"]["mean_f1"] == pytest.approx((1.0 + 0.5) / 2)
    assert agg["playability"]["pooled"]["rate"] == 1.0 and agg["playability"]["n_clips"] == 2


# ----- real GuitarSet labels (skips without features/v1) -----

@pytest.fixture(scope="module")
def real_clips():
    import h5py
    if not (REAL_ROOT / "manifest.csv").is_file():
        pytest.skip(f"no preprocessed data at {REAL_ROOT}")
    manifest = pd.read_csv(REAL_ROOT / "manifest.csv", dtype={"player_id": str})
    clips = []
    for rel in manifest.hdf5_path:
        with h5py.File(REAL_ROOT / rel, "r") as hf:
            clips.append(hf["tab_notes"][:])
    return clips


def test_real_labels_self_consistent(real_clips):
    per_clip = [{**note_metrics(n, n), "playability": playability(n)} for n in real_clips]
    agg = aggregate(per_clip)
    assert agg["note_onset"]["pooled"]["f1"] == 1.0 and agg["tab_offset"]["pooled"]["f1"] == 1.0
    rate = agg["playability"]["pooled"]["rate"]
    print(f"\nGuitarSet label playability: {rate:.4f}")
    assert 0.985 < rate < 0.999


def test_real_grid_round_trip(real_clips):
    per_clip = []
    for notes in real_clips[::10]:
        n_frames = int(np.ceil(notes[:, 3].max() * 100)) + 2
        tab, onsets = notes_to_tab_grid(notes, n_frames=n_frames)
        per_clip.append(note_metrics(notes, tab_grid_to_notes(tab, onsets=onsets)))
    f = aggregate(per_clip)["tab_onset"]["pooled"]["f1"]
    print(f"\ngrid -> notes round trip, onset Tab F1: {f:.4f}")
    assert f >= 0.95
