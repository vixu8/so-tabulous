import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "data"))
from loader import (BEGIN_NOTE, IGNORE_INDEX, KONG_ROLLS, N_FRAMES, OPEN_STRING_MIDI,  # noqa: E402
                     SAMPLE_RATE, SEGMENT_SAMPLES, GuitarSetDataset, _kong_target_processor,
                     collate, load_split, notes_to_tab_grid, tab_to_string_per_pitch)
from tokenizer import PAD_ID, TabTokenizer  # noqa: E402

REAL_ROOT = REPO / "features" / "v1"


# ----- synthetic clips in preprocessing.py's HDF5 schema -----

def kong_events(tab_notes):
    """Pitch-level note_on events like preprocessing.to_kong_events (no unisons here)."""
    events = []
    for string, fret, on, off in tab_notes:
        pitch = OPEN_STRING_MIDI[int(string)] + int(fret)
        events.append((on, 1, f"note_on channel=0 note={pitch} velocity=64 time=0"))
        events.append((off, 0, f"note_on channel=0 note={pitch} velocity=0 time=0"))
    events.sort(key=lambda e: (e[0], e[1]))
    return [e[2] for e in events], np.array([e[0] for e in events], dtype=np.float32)


def write_clip(path, duration_s, tab_notes, waveform=None, player="00"):
    n = int(duration_s * SAMPLE_RATE)
    if waveform is None:
        waveform = (np.arange(n) % 2000 - 1000).astype(np.int16)
    tab_notes = np.asarray(tab_notes, dtype=np.float32).reshape(-1, 4)
    events, times = kong_events(tab_notes)
    with h5py.File(path, "w") as hf:
        hf.attrs.create("player_id", data=player.encode(), dtype="S100")
        hf.attrs.create("duration", data=duration_s, dtype=np.float32)
        hf.attrs.create("sample_rate", data=SAMPLE_RATE, dtype=np.int32)
        hf.create_dataset("waveform", data=waveform, dtype=np.int16)
        hf.create_dataset("midi_event", data=[e.encode() for e in events], dtype="S100")
        hf.create_dataset("midi_event_time", data=times, dtype=np.float32)
        hf.create_dataset("tab_notes", data=tab_notes, dtype=np.float32)


# Notes for the 25 s clip. Times chosen off the rounding midpoints.
NOTES = [
    [0, 3, 0.50, 1.20],    # low E fret 3 (pitch 43)
    [2, 2, 0.50, 1.20],    # same onset, D string fret 2 (pitch 52)
    [1, 0, 9.50, 10.80],   # A open (45), rings across the 10 s window boundary
    [5, 12, 12.00, 12.50],  # high e fret 12 (76)
    [3, 5, 24.00, 24.90],  # G fret 5 (60), in the last (padded) window
]


@pytest.fixture
def root(tmp_path):
    (tmp_path / "hdf5s").mkdir()
    rows = []
    for player in ["00", "01", "02", "03", "04", "05"]:
        for k in range(2):
            track = f"{player}_Test{k}-100-C_comp"
            write_clip(tmp_path / "hdf5s" / f"{track}.h5", 25.0, NOTES, player=player)
            rows.append({"track_id": track, "player_id": player, "split": "train",
                         "duration": 25.0, "hdf5_path": f"hdf5s/{track}.h5"})
    pd.DataFrame(rows).to_csv(tmp_path / "manifest.csv", index=False)
    return tmp_path


@pytest.fixture
def splits(root):
    return load_split(root, "04", "05")


def kong_available():
    try:
        _kong_target_processor()
        return True, ""
    except (FileNotFoundError, ImportError) as exc:
        return False, str(exc)


# ----- splits -----

def test_load_split_by_player(splits):
    assert sorted(set(splits["train"].player_id)) == ["00", "01", "02", "03"]
    assert set(splits["validation"].player_id) == {"04"}
    assert set(splits["test"].player_id) == {"05"}
    assert [len(splits[k]) for k in ("train", "validation", "test")] == [8, 2, 2]


def test_load_split_ignores_baked_in_split_and_accepts_ints(root):
    s = load_split(root, 0, 1)  # every row says split=train in the manifest
    assert set(s["validation"].player_id) == {"00"} and set(s["test"].player_id) == {"01"}


def test_load_split_rejects_bad_players(root):
    with pytest.raises(ValueError):
        load_split(root, "04", "04")
    with pytest.raises(ValueError):
        load_split(root, "04", "09")


# ----- target helpers -----

def test_tab_grid_basic_note():
    tab, onset = notes_to_tab_grid([[0, 3, 0.50, 1.20]], start_s=0.0)
    assert tab.shape == (N_FRAMES, 6) and tab.dtype == np.int64
    assert (tab[50:121, 0] == 4).all() and tab[49, 0] == 0 and tab[121, 0] == 0
    assert np.argwhere(onset).tolist() == [[50, 0]]


def test_tab_grid_note_from_before_crop_has_no_onset():
    tab, onset = notes_to_tab_grid([[1, 0, 9.50, 10.80]], start_s=10.0)
    assert (tab[0:81, 1] == 1).all() and tab[81, 1] == 0
    assert not onset.any()


def test_tab_grid_note_past_end_fills_to_last_frame():
    tab, onset = notes_to_tab_grid([[1, 0, 9.50, 10.80]], start_s=0.0)
    assert (tab[950:, 1] == 1).all() and onset[950, 1]


def test_tab_grid_skips_notes_outside_crop():
    tab, onset = notes_to_tab_grid([[0, 1, 1.0, 2.0], [0, 1, 25.0, 26.0]], start_s=10.0)
    assert not tab.any() and not onset.any()


def test_tab_grid_later_note_wins_shared_frame():
    # Rounding puts both notes on frame 100 of string 0; the later one keeps it.
    tab, _ = notes_to_tab_grid([[0, 5, 0.5, 1.0], [0, 7, 1.004, 1.5]], start_s=0.0)
    assert tab[99, 0] == 6 and tab[100, 0] == 8


def test_string_per_pitch():
    tab, _ = notes_to_tab_grid([[0, 3, 0.5, 1.2], [2, 2, 0.5, 1.2]], start_s=0.0)
    spp = tab_to_string_per_pitch(tab)
    assert spp.shape == (N_FRAMES, 88) and spp.dtype == np.int64
    assert (spp[50:121, 43 - BEGIN_NOTE] == 0).all()
    assert (spp[50:121, 52 - BEGIN_NOTE] == 2).all()
    assert (spp == IGNORE_INDEX).sum() == spp.size - 2 * 71


def test_string_per_pitch_unison_keeps_lower_string():
    # Pitch 50 on the A string (fret 5) and the D string (open) at once.
    tab, _ = notes_to_tab_grid([[1, 5, 0.5, 1.0], [2, 0, 0.5, 1.0]], start_s=0.0)
    assert (tab_to_string_per_pitch(tab)[50:101, 50 - BEGIN_NOTE] == 1).all()


# ----- dataset -----

def test_random_item_shapes(splits, root):
    ds = GuitarSetDataset(splits["train"], root, mode="random")
    assert len(ds) == 8
    item = ds[0]
    assert item["waveform"].shape == (SEGMENT_SAMPLES,) and item["waveform"].dtype == np.float32
    assert item["tab"].shape == (N_FRAMES, 6) and item["tab_onset"].dtype == bool
    assert item["n_valid_frames"] == N_FRAMES
    assert 0.0 <= item["start_s"] <= 15.0


def test_random_crops_reproducible_and_per_clip_count(splits, root):
    ds = GuitarSetDataset(splits["train"], root, mode="random", crops_per_clip=3)
    assert len(ds) == 24
    torch.manual_seed(0)
    a = [ds[i]["start_s"] for i in range(5)]
    torch.manual_seed(0)
    b = [ds[i]["start_s"] for i in range(5)]
    assert a == b and len(set(a)) > 1


def test_random_crop_audio_and_labels_line_up(splits, root):
    ds = GuitarSetDataset(splits["train"], root, mode="random")
    torch.manual_seed(1)
    item = ds[0]
    start = round(item["start_s"] * SAMPLE_RATE)
    full = (np.arange(int(25 * SAMPLE_RATE)) % 2000 - 1000).astype(np.int16)
    np.testing.assert_allclose(item["waveform"], full[start:start + SEGMENT_SAMPLES] / 32767.0,
                               rtol=1e-6)
    # float32 times, as stored in the HDF5, so frame rounding matches exactly
    tab, onset = notes_to_tab_grid(np.array(NOTES, dtype=np.float32), item["start_s"])
    np.testing.assert_array_equal(item["tab"], tab)
    np.testing.assert_array_equal(item["tab_onset"], onset)


def test_windows_cover_clip_and_pad_last(splits, root):
    ds = GuitarSetDataset(splits["validation"], root, mode="windows")
    assert len(ds) == 2 * 3  # 25 s -> windows at 0, 10, 20
    items = [ds[i] for i in range(3)]
    assert [it["start_s"] for it in items] == [0.0, 10.0, 20.0]
    assert [it["n_valid_frames"] for it in items] == [N_FRAMES, N_FRAMES, 5 * 100 + 1]
    assert not items[2]["waveform"][5 * SAMPLE_RATE:].any()  # padding
    # The note at 24.0-24.9 s on the G string lands at frames 400-490 of the last window.
    assert (items[2]["tab"][400:491, 3] == 6).all() and items[2]["tab_onset"][400, 3]


def test_tokens_match_tokenizer(splits, root):
    ds = GuitarSetDataset(splits["validation"], root, mode="windows", targets=("tokens",))
    tok = TabTokenizer()
    for i in range(3):
        item = ds[i]
        assert item["tokens"] == tok.encode(np.array(NOTES, dtype=np.float32), item["start_s"])


def test_collate(splits, root):
    ds = GuitarSetDataset(splits["validation"], root, mode="windows",
                          targets=("tokens", "string_per_pitch"))
    batch = collate([ds[0], ds[2]])
    assert batch["waveform"].shape == (2, SEGMENT_SAMPLES) and batch["waveform"].dtype == torch.float32
    assert batch["tab"].shape == (2, N_FRAMES, 6) and batch["tab"].dtype == torch.int64
    assert batch["string_per_pitch"].shape == (2, N_FRAMES, 88)
    assert batch["n_valid_frames"].tolist() == [N_FRAMES, 501]
    assert batch["start_s"] == [0.0, 20.0] and len(batch["track_id"]) == 2
    lengths = batch["token_lengths"].tolist()
    assert batch["tokens"].shape == (2, max(lengths))
    short = int(np.argmin(lengths))
    assert (batch["tokens"][short, lengths[short]:] == PAD_ID).all()


def test_collate_works_in_dataloader(splits, root):
    ds = GuitarSetDataset(splits["train"], root, mode="random", targets=("string_per_pitch",))
    batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=4, collate_fn=collate)))
    assert batch["tab"].shape == (4, N_FRAMES, 6)


def test_bad_arguments(splits, root):
    with pytest.raises(ValueError):
        GuitarSetDataset(splits["train"], root, mode="sliding")
    with pytest.raises(ValueError):
        GuitarSetDataset(splits["train"], root, targets=("velocity",))
    with pytest.raises(ValueError):
        GuitarSetDataset(splits["train"], root, crops_per_clip=0)


# ----- Kong targets (need the cloned Kong repo) -----

def test_kong_targets(splits, root):
    ok, reason = kong_available()
    if not ok:
        pytest.skip(reason)
    ds = GuitarSetDataset(splits["validation"], root, mode="windows", targets=("kong",))
    first, second = ds[0], ds[1]
    for key in KONG_ROLLS:
        assert first[key].shape == (N_FRAMES, 88) and first[key].dtype == np.float32
    # Every sounding tab frame is on in Kong's frame_roll (or masked out by Kong).
    for item in (first, second):
        tab, frame, mask = item["tab"], item["frame_roll"], item["mask_roll"]
        for s in range(6):
            frames = np.nonzero(tab[:, s])[0]
            pitch = OPEN_STRING_MIDI[s] + tab[frames, s] - 1 - BEGIN_NOTE
            assert ((frame[frames, pitch] == 1) | (mask[frames, pitch] == 0)).all()
    # Onsets agree: low E fret 3 (pitch 43) and D fret 2 (52) at frame 50 of window 0.
    assert first["onset_roll"][50, 43 - BEGIN_NOTE] == 1 and first["onset_roll"][50, 52 - BEGIN_NOTE] == 1
    # The A-string note from before window 1: no onset in either label set, and Kong masks it.
    assert not second["tab_onset"][:, 1].any() and not second["onset_roll"][:, 45 - BEGIN_NOTE].any()
    assert (second["mask_roll"][:81, 45 - BEGIN_NOTE] == 0).all()


# ----- real data (features/v1) -----

@pytest.mark.skipif(not (REAL_ROOT / "manifest.csv").is_file(), reason="features/v1 not present")
def test_real_data_every_note_onset_in_windows():
    manifest = pd.read_csv(REAL_ROOT / "manifest.csv", dtype={"player_id": str})
    ds = GuitarSetDataset(manifest, REAL_ROOT, mode="windows")
    by_clip = {}
    for i in range(len(ds)):
        item = ds[i]
        by_clip.setdefault(item["track_id"], []).append(item)
    assert len(by_clip) == len(manifest) == 360
    missing = 0
    total = 0
    for track_id, rel in zip(manifest.track_id, manifest.hdf5_path):
        items = by_clip[track_id]
        # Stitch: frame 1000 of window k is frame 0 of window k+1, so keep 1000 per window.
        onset = np.concatenate([it["tab_onset"][:1000] for it in items])
        tab = np.concatenate([it["tab"][:1000] for it in items])
        with h5py.File(REAL_ROOT / rel, "r") as hf:
            notes = hf["tab_notes"][:]
        for string, fret, on, _ in notes:
            # Expected frame computed as the loader does (relative to the window start,
            # in float64); rounding absolute float32 time differs on exact .5 frames.
            k = int(float(on) // 10)
            f, s = 1000 * k + int(round((float(on) - 10.0 * k) * 100)), int(string)
            total += 1
            missing += not (onset[f, s] and tab[f, s] == int(fret) + 1)
    assert total == 62476
    assert missing == 0, f"{missing} of {total} notes missing from the stitched grid"
