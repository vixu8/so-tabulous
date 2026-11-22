import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "data"))
from loader import GuitarSetDataset  # noqa: E402
from preprocess_synthtab import (FLIP_STRING_ORDER, N_STRINGS,  # noqa: E402
                                 OPEN_STRING_MIDI, SPLITS, SYNTHTAB_PPQ,
                                 SkipTrack, SynthTabTrack, _flip_string,
                                 clean_tab_notes, load_raw_notes, split_for_tab,
                                 write_hdf5)
from tokenizer import TabTokenizer  # noqa: E402

# Guitar Pro / DadaGP open-string pitches in SOURCE order (string 0 = high e,
# descending to string 5 = low E) -- the reverse of the repo's OPEN_STRING_MIDI.
GUITARPRO_OPEN_MIDI = [64, 59, 55, 50, 45, 40]


# ----- the string flip must land notes at their true pitch -----

@pytest.mark.skipif(not FLIP_STRING_ORDER,
                    reason="source already numbers 0 = low E; no flip to check")
@pytest.mark.parametrize("raw_string", range(N_STRINGS))
@pytest.mark.parametrize("fret", [0, 3, 12])
def test_flip_recovers_true_pitch(raw_string, fret):
    """A Guitar-Pro-numbered note, after _flip_string, must compose to the pitch
    that string actually sounds. Catches a silent string reversal that would
    otherwise train cleanly and only show up as bad Note F1."""
    repo_string = _flip_string(raw_string)
    pitch_from_repo = OPEN_STRING_MIDI[repo_string] + fret
    true_pitch = GUITARPRO_OPEN_MIDI[raw_string] + fret
    assert pitch_from_repo == true_pitch


def test_flip_is_a_permutation_and_self_inverse():
    flipped = [_flip_string(s) for s in range(N_STRINGS)]
    assert sorted(flipped) == list(range(N_STRINGS))                     # a permutation
    assert [_flip_string(s) for s in flipped] == list(range(N_STRINGS))  # flip twice = identity


# ----- clean_tab_notes: validation, pitch, overlap truncation -----

def test_clean_drops_out_of_range_and_computes_pitch():
    stats = Counter()
    raw = [(0, 3, 0.0, 1.0),      # low E fret 3 -> pitch 43, kept
           (2, 2, 0.5, 1.5),      # D fret 2 -> 52, kept
           (6, 0, 0.0, 1.0),      # string 6 out of range, dropped
           (1, 25, 0.0, 1.0),     # fret 25 > MAX_FRET, dropped
           (3, 5, 2.0, 2.0)]      # zero-length, dropped
    clean = clean_tab_notes(raw, stats)
    assert [(s, f, p) for s, f, on, off, p in clean] == [(0, 3, 43), (2, 2, 52)]
    assert stats["dropped_string_out_of_range"] == 1
    assert stats["dropped_fret_out_of_range"] == 1
    assert stats["dropped_non_positive_duration"] == 1


def test_clean_truncates_same_string_overlap():
    stats = Counter()
    clean = clean_tab_notes([(0, 1, 0.0, 2.0), (0, 3, 1.0, 3.0)], stats)
    offs = {round(on, 2): round(off, 2) for s, f, on, off, p in clean}
    assert offs == {0.0: 1.0, 1.0: 3.0}            # first note cut at the second's onset
    assert stats["truncated_same_string_overlap"] == 1


# ----- load_raw_notes: JAMS parsing (ticks->s, string flip, tuning skip) -----

def _note_tab(string_index, open_tuning, notes):
    """One note_tab annotation; notes = list of (time_ticks, dur_ticks, fret)."""
    return {"namespace": "note_tab",
            "sandbox": {"string_index": string_index, "open_tuning": open_tuning},
            "data": [{"time": t, "duration": d, "value": {"fret": f, "velocity": 95},
                      "confidence": None} for t, d, f in notes]}


def _write_jams(path, annotations, bpm=140):
    anns = list(annotations)
    if bpm is not None:
        anns.append({"namespace": "tempo", "sandbox": {},
                     "data": [{"time": 0.0, "duration": 0.0, "value": bpm,
                               "confidence": 1.0}]})
    path.write_text(json.dumps({"annotations": anns}), encoding="utf-8")


def _track(path):
    return SynthTabTrack(track_id="t", tab_id="song", synth="eff/s", split="train",
                         audio_path="", label_path=str(path))


def test_load_raw_notes_ticks_flip_and_pitch(tmp_path):
    j = tmp_path / "s.jams"
    # Guitar Pro numbering: string_index 6 = low E (open 40), 1 = high e (open 64).
    _write_jams(j, [_note_tab(6, 40, [(960, 480, 3)]),     # low E,  fret 3
                    _note_tab(1, 64, [(1920, 240, 0)])],   # high e, fret 0
                bpm=140)
    notes = sorted(load_raw_notes(_track(j)))

    # string_index 6 -> repo 0 (low E), string_index 1 -> repo 5 (high e)
    assert [n[0] for n in notes] == [0, 5]
    # onset = ticks * 60 / (bpm * PPQ); 960 ticks at 140 BPM, 960 PPQ
    on0 = 960 * 60 / (140 * SYNTHTAB_PPQ)
    off0 = (960 + 480) * 60 / (140 * SYNTHTAB_PPQ)
    assert notes[0][1] == 3                                   # fret preserved
    np.testing.assert_allclose(notes[0][2:], (on0, off0), rtol=1e-9)
    # and the flip lands the low-E note at its true pitch
    assert OPEN_STRING_MIDI[notes[0][0]] + notes[0][1] == 40 + 3


def test_load_raw_notes_skips_non_standard_tuning(tmp_path):
    j = tmp_path / "dropd.jams"
    _write_jams(j, [_note_tab(6, 38, [(0, 480, 0)])], bpm=140)  # drop-D low string
    with pytest.raises(SkipTrack):
        load_raw_notes(_track(j))


def test_load_raw_notes_skips_missing_tempo(tmp_path):
    j = tmp_path / "notempo.jams"
    _write_jams(j, [_note_tab(6, 40, [(0, 480, 0)])], bpm=None)
    with pytest.raises(SkipTrack):
        load_raw_notes(_track(j))


def test_split_for_tab_is_deterministic_and_groups_renders():
    assert split_for_tab("song-A") in SPLITS
    assert split_for_tab("song-A") == split_for_tab("song-A")   # stable per song


# ----- end to end: write a clip, read it back through the loader, tokenize -----

def _write_clip(path, tab_notes, duration_s=13.0):
    waveform = np.zeros(int(duration_s * 16_000), dtype=np.float32)
    meta = {"split": "train", "track_id": path.stem, "tab_id": "song1",
            "synth": "les_paul", "duration": duration_s}
    write_hdf5(path, waveform, None, None, tab_notes, meta)


def test_clip_round_trips_through_loader_and_tokenizer(tmp_path):
    hdf5_dir = tmp_path / "hdf5s"
    hdf5_dir.mkdir()
    tab_notes = np.array([[0, 3, 0.50, 1.20],      # low E fret 3, window 0
                          [2, 2, 0.50, 1.20],      # D fret 2, same onset (a chord), window 0
                          [5, 7, 11.00, 11.50]],   # high e fret 7, falls in window 1
                         dtype=np.float32)
    _write_clip(hdf5_dir / "clip0.h5", tab_notes)

    rows = pd.DataFrame([{"track_id": "clip0", "hdf5_path": "hdf5s/clip0.h5"}])
    ds = GuitarSetDataset(rows, str(tmp_path), mode="windows", targets=("tokens",))
    item = ds[0]  # first 10 s window, start_s = 0

    tok = TabTokenizer()
    assert item["tokens"] == tok.encode(tab_notes, item["start_s"])
    decoded = tok.decode(item["tokens"], start_s=item["start_s"])
    # Only the two notes with onset in [0, 10) survive; string/fret are exact.
    assert decoded.shape[0] == 2
    np.testing.assert_array_equal(decoded[:, :2], tab_notes[:2, :2])
    np.testing.assert_allclose(decoded[:, 2], tab_notes[:2, 2], atol=0.01)
