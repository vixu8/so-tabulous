import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "data"))
from tokenizer import (BOS_ID, DUR, EOS_ID, FRET, ONSET, PAD_ID, STRING,  # noqa: E402
                       TabTokenizer, stitch)

FPS = 100


@pytest.fixture
def tok():
    return TabTokenizer()


def note(string, fret, onset, offset):
    return [string, fret, onset, offset]


def random_on_grid_notes(rng, n, window_s=10.0, start_s=0.0):
    """Random notes with onsets on the 10 ms grid and unique (onset, string)."""
    onset_frames = rng.integers(0, int(window_s * FPS), size=n)
    strings = rng.integers(0, 6, size=n)
    _, keep = np.unique(onset_frames * 10 + strings, return_index=True)
    onset_frames, strings = onset_frames[keep], strings[keep]
    frets = rng.integers(0, 20, size=len(keep))
    durs = rng.integers(1, 300, size=len(keep))
    onsets = start_s + onset_frames / FPS
    offsets = np.minimum(onsets + durs / FPS, start_s + window_s)
    notes = np.stack([strings, frets, onsets, offsets], axis=1)
    return notes[np.lexsort((notes[:, 0], notes[:, 2]))]


# ----- vocabulary -----

def test_default_vocab(tok):
    assert tok.segment_seconds == 10.0 and tok.n_onset_frames == 1000
    assert tok.n_duration_tokens == 75 and tok.vocab_size == 1104


@pytest.mark.parametrize("seconds", [2.0, 4.0, 10.0])
def test_vocab_layout(seconds):
    tok = TabTokenizer(segment_seconds=seconds)
    n_onset = int(seconds * FPS)
    assert tok.n_onset_frames == n_onset
    assert tok.vocab_size == 3 + n_onset + tok.n_duration_tokens + 6 + 20
    sizes = {ONSET: n_onset, DUR: tok.n_duration_tokens, STRING: 6, FRET: 20}
    types = tok.token_types(np.arange(tok.vocab_size))
    for t, size in sizes.items():
        idx = np.flatnonzero(types == t)
        assert len(idx) == size and np.all(np.diff(idx) == 1)  # one contiguous range each
    assert tok.duration_frames[0] == 1 and tok.duration_frames[-1] == n_onset
    assert np.all(np.diff(tok.duration_frames) > 0)


def test_id_to_str_round_trip(tok):
    for i in range(tok.vocab_size):
        assert tok.str_to_id(tok.id_to_str(i)) == i
    assert tok.id_to_str(tok.vocab_size) == f"INVALID_{tok.vocab_size}"


def test_config_round_trip():
    tok = TabTokenizer(segment_seconds=2.0)
    tok2 = TabTokenizer.from_dict(tok.to_dict())
    assert tok2 == tok and tok2.vocab_size == tok.vocab_size


# ----- encode / decode -----

def test_round_trip_random(tok):
    rng = np.random.default_rng(0)
    for start in [0.0, 20.0]:
        notes = random_on_grid_notes(rng, 60, start_s=start)
        out = tok.decode(tok.encode(notes, start_s=start), start_s=start)
        assert out.shape == notes.shape
        np.testing.assert_array_equal(out[:, :2], notes[:, :2])
        np.testing.assert_allclose(out[:, 2], notes[:, 2], atol=1e-9)
        true_dur, got_dur = notes[:, 3] - notes[:, 2], out[:, 3] - out[:, 2]
        err = np.abs(got_dur - true_dur)
        assert np.all(err <= np.maximum(0.045 * true_dur, 0.01) + 1e-9)


def test_short_durations_exact(tok):
    durs = np.arange(1, 18)  # 10..170 ms: one bin per frame
    notes = [note(0, 0, 0.2 * i, 0.2 * i + d / FPS) for i, d in enumerate(durs)]
    out = tok.decode(tok.encode(notes))
    np.testing.assert_allclose(out[:, 3] - out[:, 2], durs / FPS, atol=1e-9)


def test_sequence_format(tok):
    ids = tok.encode([note(2, 3, 0.5, 1.0)])
    assert ids[0] == BOS_ID and ids[-1] == EOS_ID and len(ids) == 6
    assert tok.describe(ids) == "BOS ONSET_50 DUR_%d STRING_2 FRET_3 EOS" % tok.duration_bin(50)


def test_chord_shares_onset_and_orders_strings(tok):
    e_major = [(0, 0), (1, 2), (2, 2), (3, 1), (4, 0), (5, 0)]
    notes = [note(s, f, 1.0, 2.0) for s, f in reversed(e_major)]  # given high -> low
    ids = tok.encode(notes)
    groups = np.array(ids[1:-1]).reshape(-1, 4)
    assert len(set(groups[:, 0])) == 1
    assert list(groups[:, 2] - tok.string_offset) == [0, 1, 2, 3, 4, 5]
    assert list(groups[:, 3] - tok.fret_offset) == [f for _, f in e_major]


def test_resort_after_quantization(tok):
    # Raw order: string 3 at 1.001 s, then string 1 at 1.004 s; both land in frame 100.
    notes = [note(3, 5, 1.001, 2.0), note(1, 7, 1.004, 2.0)]
    out = tok.decode(tok.encode(notes))
    assert list(out[:, 0]) == [1, 3]
    np.testing.assert_allclose(out[:, 2], [1.0, 1.0])


def test_same_frame_collision_keeps_longer(tok):
    notes = [note(2, 3, 1.000, 1.05), note(2, 5, 1.003, 2.0)]
    ids, stats = tok.encode(notes, return_stats=True)
    out = tok.decode(ids)
    assert stats["same_frame_merged"] == 1 and len(out) == 1
    assert out[0, 1] == 5


def test_window_edges(tok):
    notes = [note(0, 0, 9.9, 10.5),    # starts before window [10, 20): dropped
             note(1, 1, 10.0, 10.5),   # exactly at start: kept
             note(2, 2, 19.5, 21.0),   # rings past the end: offset clipped to 20.0
             note(3, 3, 20.0, 20.5)]   # exactly at the end: next window's
    ids, stats = tok.encode(notes, start_s=10.0, return_stats=True)
    out = tok.decode(ids, start_s=10.0)
    assert list(out[:, 0]) == [1, 2]
    assert stats["offset_clipped"] == 1
    np.testing.assert_allclose(out[1, 3], 20.0, atol=0.12)
    assert out[1, 3] <= 20.0
    rel = tok.notes_in_window(notes, start_s=10.0)
    np.testing.assert_allclose(rel[:, 2:], [[0.0, 0.5], [9.5, 10.0]])


def test_onset_just_before_window_end(tok):
    out = tok.decode(tok.encode([note(0, 0, 9.999, 10.0)]))
    assert len(out) == 1 and out[0, 2] == pytest.approx(9.99) and out[0, 3] <= 10.0


@pytest.mark.parametrize("bad", [note(0, 20, 1.0, 2.0), note(6, 0, 1.0, 2.0),
                                 note(-1, 0, 1.0, 2.0), note(0, 1.5, 1.0, 2.0),
                                 note(0, 0, 2.0, 1.0)])
def test_out_of_range_raises(tok, bad):
    with pytest.raises(ValueError):
        tok.encode([bad])


def test_empty_window(tok):
    assert tok.encode(np.zeros((0, 4))) == [BOS_ID, EOS_ID]
    assert tok.encode([note(0, 0, 12.0, 12.5)]) == [BOS_ID, EOS_ID]  # notes elsewhere
    assert tok.decode([BOS_ID, EOS_ID]).shape == (0, 4)


def test_decode_malformed(tok):
    ids = tok.encode([note(0, 0, 1.0, 2.0), note(1, 2, 2.0, 3.0)])
    truncated = ids[:-2]  # last group lost its FRET and EOS
    with pytest.raises(ValueError):
        tok.decode(truncated, strict=True)
    out, stats = tok.decode(truncated, return_stats=True)
    assert len(out) == 1 and stats["malformed_groups"] == 1 and not stats["ended_with_eos"]

    # A stray STRING token, then two valid notes: the stray is skipped, both notes kept.
    garbage = [BOS_ID, ids[3], *ids[1:5], *ids[5:9], EOS_ID]
    out, stats = tok.decode(garbage, return_stats=True)
    assert len(out) == 2 and stats["malformed_groups"] == 1 and stats["ended_with_eos"]


def test_decode_stops_at_pad_and_accepts_arrays(tok):
    ids = tok.encode([note(0, 0, 1.0, 2.0)]) + [PAD_ID] * 5
    assert len(tok.decode(np.array(ids))) == 1
    assert len(tok.decode(ids[1:])) == 1  # no BOS is fine


# ----- grammar mask -----

def test_encoded_sequences_follow_mask(tok):
    rng = np.random.default_rng(1)
    notes = random_on_grid_notes(rng, 80)
    ids = tok.encode(notes)
    for t in range(1, len(ids)):
        assert tok.next_token_mask(ids[:t])[ids[t]], f"step {t}: {tok.id_to_str(ids[t])}"
    after_eos = tok.next_token_mask(ids)
    assert after_eos.sum() == 1 and after_eos[PAD_ID]
    assert tok.next_token_mask(ids + [PAD_ID, PAD_ID])[PAD_ID]


def test_mask_rules(tok):
    m = tok.next_token_mask([BOS_ID])
    assert m[EOS_ID] and m[tok.onset_offset:tok.dur_offset].all() and m.sum() == 1001
    chord = tok.encode([note(5, 0, 1.0, 2.0)])[:-1]  # high e already used at onset 100
    m = tok.next_token_mask(chord)
    assert not m[tok.onset_offset + 100] and m[tok.onset_offset + 101] and m[EOS_ID]
    partial = tok.encode([note(2, 0, 1.0, 2.0)])[:-1] + [tok.onset_offset + 100, tok.dur_offset]
    m = tok.next_token_mask(partial)
    assert m[tok.string_offset:tok.fret_offset].tolist() == [False, False, False, True, True, True]
    for bad in [[], [tok.onset_offset], [BOS_ID, tok.dur_offset], [BOS_ID, EOS_ID, BOS_ID]]:
        with pytest.raises(ValueError):
            tok.next_token_mask(bad)


def test_mask_random_walks_decode_cleanly(tok):
    rng = np.random.default_rng(2)
    for _ in range(200):
        ids = [BOS_ID]
        while ids[-1] != EOS_ID and len(ids) < 2000:
            ids.append(int(rng.choice(np.flatnonzero(tok.next_token_mask(ids)))))
        out = tok.decode(ids, strict=True)
        key = np.rint(out[:, 2] * FPS) * 10 + out[:, 0]  # (onset frame, string) strictly increasing
        assert np.all(np.diff(key) > 0)
        assert np.all(out[:, 3] > out[:, 2]) and np.all(out[:, 3] <= tok.window_seconds + 1e-9)


# ----- stitching -----

def test_window_starts(tok):
    assert tok.window_starts(25.0).tolist() == [0.0, 10.0, 20.0]
    assert tok.window_starts(20.0).tolist() == [0.0, 10.0]
    assert tok.window_starts(0.5).tolist() == [0.0]


def test_stitch_full_clip(tok):
    rng = np.random.default_rng(3)
    notes = np.concatenate([random_on_grid_notes(rng, 30, start_s=s) for s in [0.0, 10.0, 20.0]])
    segments = [tok.decode(tok.encode(notes, s), s) for s in tok.window_starts(30.0)]
    full = stitch(segments)
    assert full.shape == notes.shape
    np.testing.assert_array_equal(full[:, :2], notes[:, :2])
    np.testing.assert_allclose(full[:, 2], notes[:, 2], atol=1e-9)
    assert stitch([]).shape == (0, 4)
