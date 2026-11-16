"""Download GuitarSet (mic audio + annotations) with mirdata and pack each clip
into a Kong-style HDF5 file with extra tab labels.

Pipeline (one pass over all 360 clips):
  1. Download annotations + mono mic audio via mirdata (skippable).
  2. Assign each clip to train / validation / test by player.
  3. Per clip:
       - notes -> round to semitone -> (string, fret); drop frets outside 0..MAX_FRET
       - truncate the earlier note when two notes overlap on the same string
       - build pitch-level note events in Kong's format (unisons merged)
       - load mic audio resampled to 16 kHz mono, stored as int16
  4. Write <out_dir>/hdf5s/<track_id>.h5, <out_dir>/manifest.csv, <out_dir>/stats.json

HDF5 contents (Kong-compatible fields + tab additions):
  attrs:    split (bytes: train|validation|test), duration, sample_rate,
            track_id, player_id, style, mode
  waveform:        int16, 16 kHz mono
  midi_event:      bytes strings like "note_on channel=0 note=60 velocity=64 time=0"
                   (offsets are note_on with velocity=0, as in MAESTRO MIDI)
  midi_event_time: float32 seconds
  tab_notes:       float32 (N, 4) columns [string, fret, onset, offset];
                   string 0 = low E ... 5 = high e

Velocity is a placeholder (GuitarSet has none): mask the velocity loss in training.

Usage:
  python src/data/preprocessing.py --val_player 04 --test_player 05
  python src/data/preprocessing.py --val_player 04 --test_player 05 --skip_download
"""

# To open one clip:
# import h5py
# import numpy as np

# with h5py.File("features/v1/hdf5s/00_BN1-129-Eb_comp.h5", "r") as hf:
#     split = hf.attrs["split"].decode()           # "train" / "validation" / "test"
#     sr = int(hf.attrs["sample_rate"])            # 16000
#     duration = float(hf.attrs["duration"])

#     waveform = hf["waveform"][:].astype(np.float32) / 32767.0   # back to [-1, 1]

#     midi_events = [e.decode() for e in hf["midi_event"][:]]     # "note_on channel=0 note=50 velocity=64 time=0"
#     midi_event_times = hf["midi_event_time"][:]                 # seconds, same length as midi_events

#     tab_notes = hf["tab_notes"][:]               # (N, 4): string, fret, onset_s, offset_s

import argparse
import csv
import json
import os
from collections import Counter, defaultdict

import h5py
import librosa
import mirdata
import numpy as np

SAMPLE_RATE = 16000                       # Kong's input sample rate
OPEN_STRING_MIDI = [40, 45, 50, 55, 59, 64]  # standard tuning, low E -> high e
STRING_NAMES = ["E", "A", "D", "G", "B", "e"]  # mirdata's keys for track.notes
MAX_FRET = 19                             # matches TabCNN's output range
PLACEHOLDER_VELOCITY = 64
PLAYERS = ["00", "01", "02", "03", "04", "05"]


def assign_split(player_id, val_player, test_player):
    if player_id == test_player:
        return "test"
    if player_id == val_player:
        return "validation"
    return "train"


def clean_string_notes(note_data, string_idx, stats):
    """Turn one string's mirdata NoteData into cleaned tab notes.

    Returns a list of (string, fret, onset, offset, pitch).
    """
    if note_data is None:  # mirdata returns None for a string with no notes
        return []
    assert note_data.pitch_unit == "midi", note_data.pitch_unit

    notes = []
    for (onset, offset), midi in zip(note_data.intervals, note_data.pitches):
        pitch = int(round(float(midi)))
        fret = pitch - OPEN_STRING_MIDI[string_idx]
        stats["notes_total"] += 1
        if not 0 <= fret <= MAX_FRET:
            stats["dropped_fret_out_of_range"] += 1
            continue
        notes.append([float(onset), float(offset), pitch, fret])

    notes.sort(key=lambda n: n[0])
    cleaned = []
    for note in notes:
        if cleaned and cleaned[-1][1] > note[0]:
            # Overlap on the same string: cut the earlier note at this onset.
            cleaned[-1][1] = note[0]
            stats["truncated_same_string_overlap"] += 1
            if cleaned[-1][1] <= cleaned[-1][0]:  # same onset -> nothing left
                cleaned.pop()
                stats["dropped_zero_length_after_truncation"] += 1
        cleaned.append(note)

    return [(string_idx, fret, on, off, pitch) for on, off, pitch, fret in cleaned]


def to_kong_events(tab_notes, stats):
    """Pitch-level note events for Kong's target processor.

    Unisons (same pitch on two strings at once) are merged into one note,
    since Kong's targets are per pitch, not per string.
    """
    intervals_by_pitch = defaultdict(list)
    for _, _, onset, offset, pitch in tab_notes:
        intervals_by_pitch[pitch].append([onset, offset])

    events = []  # (time, order, message); offsets sort before onsets at equal times
    for pitch, intervals in intervals_by_pitch.items():
        intervals.sort()
        merged = []
        for onset, offset in intervals:
            if merged and onset < merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], offset)
                stats["merged_unisons"] += 1
            else:
                merged.append([onset, offset])
        for onset, offset in merged:
            events.append((onset, 1, f"note_on channel=0 note={pitch} "
                                     f"velocity={PLACEHOLDER_VELOCITY} time=0"))
            events.append((offset, 0, f"note_on channel=0 note={pitch} "
                                      f"velocity=0 time=0"))
    events.sort(key=lambda e: (e[0], e[1]))
    return [e[2] for e in events], np.array([e[0] for e in events], dtype=np.float32)


def float32_to_int16(x):
    # Same conversion as Kong's piano_transcription utilities.
    return (np.clip(x, -1, 1) * 32767.0).astype(np.int16)


def write_hdf5(path, waveform, midi_events, midi_event_times, tab_notes, meta):
    with h5py.File(path, "w") as hf:
        # Kong's code calls .decode() on string attrs, so store them as bytes.
        for key in ["split", "track_id", "player_id", "style", "mode"]:
            hf.attrs.create(key, data=meta[key].encode(), dtype="S100")
        hf.attrs.create("duration", data=meta["duration"], dtype=np.float32)
        hf.attrs.create("sample_rate", data=SAMPLE_RATE, dtype=np.int32)

        hf.create_dataset("waveform", data=float32_to_int16(waveform), dtype=np.int16)
        hf.create_dataset("midi_event", data=[e.encode() for e in midi_events], dtype="S100")
        hf.create_dataset("midi_event_time", data=midi_event_times, dtype=np.float32)
        tab_array = np.array([n[:4] for n in tab_notes], dtype=np.float32).reshape(-1, 4)
        hf.create_dataset("tab_notes", data=tab_array, dtype=np.float32)


def process_track(track, split, hdf5_dir, stats, fret_counts):
    tab_notes = []
    for string_idx, name in enumerate(STRING_NAMES):
        tab_notes += clean_string_notes(track.notes[name], string_idx, stats)
    tab_notes.sort(key=lambda n: (n[2], n[0]))  # by onset, then string
    for note in tab_notes:
        fret_counts[note[1]] += 1

    midi_events, midi_event_times = to_kong_events(tab_notes, stats)

    waveform, _ = librosa.load(track.audio_mic_path, sr=SAMPLE_RATE, mono=True)
    duration = len(waveform) / SAMPLE_RATE
    last_offset = max((n[3] for n in tab_notes), default=0.0)
    if last_offset > duration + 0.5:
        print(f"WARNING {track.track_id}: last note ends at {last_offset:.2f}s "
              f"but audio is {duration:.2f}s")
        stats["duration_mismatch_clips"] += 1

    meta = {"split": split, "track_id": track.track_id, "player_id": track.player_id,
            "style": track.style, "mode": track.mode, "duration": duration}
    hdf5_path = os.path.join(hdf5_dir, f"{track.track_id}.h5")
    write_hdf5(hdf5_path, waveform, midi_events, midi_event_times, tab_notes, meta)

    return {**meta, "tempo": track.tempo, "n_notes": len(tab_notes),
            "hdf5_path": os.path.relpath(hdf5_path, os.path.dirname(hdf5_dir)),
            "audio_mic_path": track.audio_mic_path, "jams_path": track.jams_path}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--val_player", required=True, choices=PLAYERS)
    parser.add_argument("--test_player", required=True, choices=PLAYERS)
    parser.add_argument("--data_home", default=None,
                        help="where mirdata stores GuitarSet (default: ~/mir_datasets/guitarset)")
    parser.add_argument("--out_dir", default="features/v1")
    parser.add_argument("--skip_download", action="store_true")
    args = parser.parse_args()
    if args.val_player == args.test_player:
        parser.error("val_player and test_player must differ")

    # --- Stage 0: download only what we need ---
    dataset = mirdata.initialize("guitarset", data_home=args.data_home)
    if not args.skip_download:
        dataset.download(partial_download=["annotations", "audio_mic"])

    track_ids = dataset.track_ids
    missing = [t for t in track_ids
               if not (os.path.exists(dataset.track(t).audio_mic_path)
                       and os.path.exists(dataset.track(t).jams_path))]
    if missing:
        raise SystemExit(f"{len(missing)} clips missing mic audio or JAMS, e.g. {missing[:3]}")
    print(f"Found {len(track_ids)} clips with mic audio + annotations")

    # --- Stages 1-4: split, labels, audio, pack ---
    hdf5_dir = os.path.join(args.out_dir, "hdf5s")
    os.makedirs(hdf5_dir, exist_ok=True)
    stats, fret_counts, rows = Counter(), Counter(), []

    for i, track_id in enumerate(sorted(track_ids)):
        track = dataset.track(track_id)
        split = assign_split(track.player_id, args.val_player, args.test_player)
        rows.append(process_track(track, split, hdf5_dir, stats, fret_counts))
        if (i + 1) % 30 == 0:
            print(f"  processed {i + 1}/{len(track_ids)}")

    # --- Manifest ---
    manifest_path = os.path.join(args.out_dir, "manifest.csv")
    fields = ["track_id", "player_id", "style", "mode", "tempo", "split", "duration",
              "n_notes", "hdf5_path", "audio_mic_path", "jams_path"]
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # --- Basic validation ---
    players_by_split = defaultdict(set)
    for row in rows:
        players_by_split[row["split"]].add(row["player_id"])
    for a in players_by_split:
        for b in players_by_split:
            if a < b:
                assert not players_by_split[a] & players_by_split[b], f"player leak {a}/{b}"

    summary = {
        "val_player": args.val_player,
        "test_player": args.test_player,
        "clips_per_split": dict(Counter(r["split"] for r in rows)),
        "players_per_split": {k: sorted(v) for k, v in players_by_split.items()},
        "hours_per_split": {s: round(sum(r["duration"] for r in rows if r["split"] == s) / 3600, 3)
                            for s in players_by_split},
        "label_stats": dict(stats),
        "fret_histogram": {int(k): v for k, v in sorted(fret_counts.items())},
    }
    with open(os.path.join(args.out_dir, "stats.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nWrote {len(rows)} HDF5 files to {hdf5_dir}\nManifest: {manifest_path}")


if __name__ == "__main__":
    main()