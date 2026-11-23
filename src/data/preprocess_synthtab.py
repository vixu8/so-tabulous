"""Pack the SynthTab dataset into the same Kong-style HDF5 clips as GuitarSet,
so the shared loader (src/data/loader.py) and the AR-decoder tokenizer can read
it unchanged.

SynthTab is synthesised from Guitar Pro tabs (DadaGP), so unlike GuitarSet the
(string, fret) are given directly -- no pitch->string inference. Each tab is
rendered with several guitar synths; we treat every render as its own clip but
take the split from the *tab* (song) so renders of one song never cross splits.

Split: train/val only (SynthTab is training data; GuitarSet is the decoder's
zero-shot test). Keyed on the Guitar Pro file (`gp_file`), so every render AND
every guitar track of a song lands in the same split -- no leakage into val.
`split_for_tab` hashes that key, sending VAL_FRACTION of files to val. The dev
subset ships no split file; swap in SynthTab's official split for the full
dataset. There is no leave-one-player-out regrouping here (that is GuitarSet's
scheme): to build a dataset, filter the manifest and hand the rows straight to
GuitarSetDataset, e.g.

    import pandas as pd
    from loader import GuitarSetDataset, collate
    m = pd.read_csv("features/synthtab/manifest.csv")
    train = GuitarSetDataset(m[m.split == "train"], "features/synthtab",
                             mode="random", targets=("tokens",))

Tracks with a non-standard tuning (capo / drop / alternate) are skipped: they
can't be represented in standard-tuning fret space. See the SynthTab layout note
and load_raw_notes below.

HDF5 contents (identical schema to preprocessing.py):
  attrs:    split, duration, sample_rate, track_id, tab_id, synth
  waveform:        int16, 16 kHz mono
  midi_event:      bytes "note_on channel=0 note=.. velocity=.. time=0"   (Kong)
  midi_event_time: float32 seconds
  tab_notes:       float32 (N, 4) [string, fret, onset_s, offset_s], string 0 = low E

`midi_event*` are only needed for the Kong transcription target; the decoder uses
tab_notes alone. Pass --no_kong_events to skip them (smaller/faster) if you will
only train the decoder.

Usage:
  python src/data/preprocess_synthtab.py --synthtab_root <path> --out_dir features/synthtab
  python src/data/preprocess_synthtab.py --synthtab_root <path> --no_kong_events --limit 50
"""

import argparse
import csv
import functools
import glob
import hashlib
import json
import multiprocessing as mp
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass

import h5py
import librosa
import numpy as np

# Reuse GuitarSet's helpers so the two datasets stay byte-compatible.
from preprocessing import (MAX_FRET, OPEN_STRING_MIDI, SAMPLE_RATE,
                           float32_to_int16, to_kong_events)

N_STRINGS = len(OPEN_STRING_MIDI)
SPLITS = ("train", "val")          # SynthTab is training-only; GuitarSet is the test set
VAL_FRACTION = 0.05                # held-out val share, keyed on the Guitar Pro file


# --------------------------------------------------------------------------- #
# SynthTab layout
#
# <root>/<effect>/<synth>/<song>/<song>/<synth>_..._body.flac   (+ per-string .pkl)
# <root>/jams/<song>/<*.jams>                                   (labels, by song)
#
# where <effect> is one of acoustic / electric_clean / electric_distortion_di /
# electric_muted, <synth> is a guitar model (e.g. gibson_thumb, lespaul_...), and
# <song> is the DadaGP tab id shared across every render. Labels live under jams/
# keyed by <song>, NOT next to the audio.
#
# The JAMS has 6 `note_tab` annotations (one per string) + 1 `tempo` annotation.
# Each note: time/duration in MIDI ticks (PPQ below), value.fret, value.velocity.
# Each note_tab sandbox: {"string_index": 1..6, "open_tuning": <open MIDI>}, with
# string_index 1 = high e ... 6 = low E (Guitar Pro order; we flip to 0 = low E).
# --------------------------------------------------------------------------- #

SYNTHTAB_PPQ = 960          # ticks per beat in SynthTab's JAMS (matches string_*.mid)
AUDIO_PREFERENCE = ("mono_body", "stereo", "stere")  # which render of a clip to pack


@dataclass
class SynthTabTrack:
    """One rendered clip: a (song, effect/synth) pair and where its audio/labels are."""
    track_id: str          # filesystem-safe, unique per render
    tab_id: str            # the DadaGP song id (shared by all its renders)
    synth: str             # "<effect>/<synth>"
    split: str             # "train" | "val" | "test"
    audio_path: str        # rendered .flac
    label_path: str        # the song's .jams


class SkipTrack(Exception):
    """This track can't be represented (e.g. a non-standard tuning); skip it."""


def gp_file(tab_id):
    """Group key = the Guitar Pro file, dropping the '__<track> - <instrument>' tail.

    'Artist - Title - gp4__3 - Distortion Guitar__midi' -> 'Artist - Title - gp4'.
    So the several guitar tracks of one song file share a split (names without the
    gp pattern are their own group). SynthTab is training-only, so there is no test
    split; GuitarSet is the decoder's zero-shot test.
    """
    m = re.match(r"(.*- gp\d+)__\d+", tab_id)
    return m.group(1) if m else tab_id


def split_for_tab(tab_id):
    """Deterministic train/val split keyed on the Guitar Pro file, so every render
    AND every track of a song lands in the same split (no leakage into val).

    VAL_FRACTION of files go to val. The dev subset ships no split file; for the
    full dataset, swap in SynthTab's official split here (keep the gp_file keying).
    """
    h = int(hashlib.md5(gp_file(tab_id).encode("utf-8")).hexdigest(), 16) % 1000
    return "val" if h < round(VAL_FRACTION * 1000) else "train"


def _track_id(effect, synth, song):
    """Short, filesystem-safe, unique id (full song in the hash avoids collisions
    and keeps output paths short enough for Windows)."""
    digest = hashlib.sha1(f"{effect}__{synth}__{song}".encode("utf-8")).hexdigest()[:10]
    slug = re.sub(r"[^0-9A-Za-z]+", "_", f"{effect}_{synth}").strip("_")
    return f"{slug}_{digest}"


def _find_flac(song_dir):
    """Pick one .flac under a render's (double-nested) song dir, preferring the
    dry mono body over stereo mixes."""
    flacs = glob.glob(os.path.join(song_dir, "**", "*.flac"), recursive=True)
    for tag in AUDIO_PREFERENCE:
        for f in flacs:
            if tag in os.path.basename(f):
                return f
    return flacs[0] if flacs else None


def iter_synthtab_tracks(root, limit=None):
    """Yield a SynthTabTrack for every (song, effect/synth) render with both audio
    and labels present. Renders missing either are skipped with a message."""
    jams_root = os.path.join(root, "jams")
    effects = sorted(d for d in os.listdir(root)
                     if d != "jams" and os.path.isdir(os.path.join(root, d)))
    n = 0
    for effect in effects:
        for synth in sorted(os.listdir(os.path.join(root, effect))):
            synth_dir = os.path.join(root, effect, synth)
            if not os.path.isdir(synth_dir):
                continue
            for song in sorted(os.listdir(synth_dir)):
                song_dir = os.path.join(synth_dir, song)
                if not os.path.isdir(song_dir):
                    continue
                flac = _find_flac(song_dir)
                jams = glob.glob(os.path.join(jams_root, song, "*.jams"))
                if flac is None or not jams:
                    print(f"SKIP {effect}/{synth}/{song}: "
                          f"{'no flac' if flac is None else 'no jams'}")
                    continue
                yield SynthTabTrack(track_id=_track_id(effect, synth, song),
                                    tab_id=song, synth=f"{effect}/{synth}",
                                    split=split_for_tab(song),
                                    audio_path=flac, label_path=jams[0])
                n += 1
                if limit is not None and n >= limit:
                    return


# Guitar Pro / DadaGP number strings from the high e (string 1) down to the low
# E (string 6); the repo wants 0 = low E ... 5 = high e. Flip once here so every
# downstream stage (pitch math in loader/Head C/metrics) sees the repo
# convention. Set to False only if your SynthTab source already numbers 0 = low E.
FLIP_STRING_ORDER = True


def _flip_string(string):
    """0-based source string index (0 = high e) -> repo index (0 = low E)."""
    return (N_STRINGS - 1 - string) if FLIP_STRING_ORDER else string


def _ticks_to_seconds(ticks, bpm):
    return ticks * 60.0 / (bpm * SYNTHTAB_PPQ)


def load_raw_notes(track):
    """Parse one render's JAMS into (string, fret, onset_s, offset_s) notes in the
    repo's 0 = low E convention.

    Raises SkipTrack for anything we can't represent in standard-tuning fret
    space: a missing tempo, a note_tab without a string_index, or any string whose
    open_tuning differs from standard (capo / drop / alternate tunings).
    """
    with open(track.label_path, encoding="utf-8") as f:
        jam = json.load(f)
    anns = jam["annotations"]

    bpm = next((a["data"][0]["value"] for a in anns
                if a["namespace"] == "tempo" and a["data"]), None)
    if not bpm:
        raise SkipTrack(f"{track.track_id}: no tempo annotation")

    notes = []
    for a in anns:
        if a["namespace"] != "note_tab":
            continue
        sb = a.get("sandbox", {})
        string_index = sb.get("string_index")        # 1 = high e ... 6 = low E
        if string_index is None:
            raise SkipTrack(f"{track.track_id}: note_tab without string_index")
        repo_string = _flip_string(int(string_index) - 1)   # -> 0 = low E
        if sb.get("open_tuning") != OPEN_STRING_MIDI[repo_string]:
            raise SkipTrack(f"{track.track_id}: non-standard tuning "
                            f"(string {string_index} open={sb.get('open_tuning')})")
        for ev in a["data"]:
            onset = _ticks_to_seconds(ev["time"], bpm)
            offset = _ticks_to_seconds(ev["time"] + ev["duration"], bpm)
            notes.append((repo_string, int(ev["value"]["fret"]), onset, offset))
    return notes


# --------------------------------------------------------------------------- #
# Cleaning (dataset-agnostic) -- mirrors preprocessing.clean_string_notes.
# --------------------------------------------------------------------------- #

def clean_tab_notes(raw_notes, stats):
    """Validate + de-overlap raw (string, fret, onset, offset) notes.

    Returns a list of (string, fret, onset, offset, pitch) sorted by (onset,
    string). Drops frets out of 0..MAX_FRET and strings out of 0..5, and cuts the
    earlier note when two notes overlap on the same string (same rule as
    GuitarSet), so the tokenizer's one-note-per-(frame, string) assumption holds.
    """
    by_string = defaultdict(list)
    for string, fret, onset, offset in raw_notes:
        string, fret = int(string), int(fret)
        stats["notes_total"] += 1
        if not 0 <= string < N_STRINGS:
            stats["dropped_string_out_of_range"] += 1
            continue
        if not 0 <= fret <= MAX_FRET:
            stats["dropped_fret_out_of_range"] += 1
            continue
        onset, offset = float(onset), float(offset)
        if offset <= onset:
            stats["dropped_non_positive_duration"] += 1
            continue
        pitch = OPEN_STRING_MIDI[string] + fret
        by_string[string].append([onset, offset, pitch, fret])

    cleaned = []
    for string, notes in by_string.items():
        notes.sort(key=lambda n: n[0])
        kept = []
        for note in notes:
            if kept and kept[-1][1] > note[0]:  # overlap on this string
                kept[-1][1] = note[0]
                stats["truncated_same_string_overlap"] += 1
                if kept[-1][1] <= kept[-1][0]:
                    kept.pop()
                    stats["dropped_zero_length_after_truncation"] += 1
            kept.append(note)
        cleaned += [(string, fret, on, off, pitch) for on, off, pitch, fret in kept]

    cleaned.sort(key=lambda n: (n[2], n[0]))  # by onset, then string
    return cleaned


def write_hdf5(path, waveform, midi_events, midi_event_times, tab_notes, meta):
    """Same datasets as preprocessing.write_hdf5; SynthTab-flavoured string attrs."""
    with h5py.File(path, "w") as hf:
        for key in ["split", "track_id", "tab_id", "synth"]:
            hf.attrs.create(key, data=meta[key].encode(), dtype="S100")
        hf.attrs.create("duration", data=meta["duration"], dtype=np.float32)
        hf.attrs.create("sample_rate", data=SAMPLE_RATE, dtype=np.int32)

        hf.create_dataset("waveform", data=float32_to_int16(waveform), dtype=np.int16)
        if midi_events is not None:
            hf.create_dataset("midi_event", data=[e.encode() for e in midi_events],
                              dtype="S100")
            hf.create_dataset("midi_event_time", data=midi_event_times, dtype=np.float32)
        tab_array = np.array([n[:4] for n in tab_notes], dtype=np.float32).reshape(-1, 4)
        hf.create_dataset("tab_notes", data=tab_array, dtype=np.float32)


def process_track(track, hdf5_dir, kong_events=True):
    """Pack one clip into HDF5.

    Returns (manifest_row, stats, fret_counts). Counters are built per call, not
    shared, so this runs unchanged in a worker process; the parent merges them.
    """
    if track.split not in SPLITS:
        raise ValueError(f"{track.track_id}: bad split {track.split!r}")
    stats, fret_counts = Counter(), Counter()
    try:
        raw = load_raw_notes(track)
    except SkipTrack as e:
        print(f"SKIP {e}")
        stats["skipped_tracks"] += 1
        return None, stats, fret_counts
    clean = clean_tab_notes(raw, stats)
    for note in clean:
        fret_counts[note[1]] += 1

    midi_events, midi_event_times = (None, None)
    if kong_events:
        midi_events, midi_event_times = to_kong_events(clean, stats)

    waveform, _ = librosa.load(track.audio_path, sr=SAMPLE_RATE, mono=True)
    duration = len(waveform) / SAMPLE_RATE
    last_offset = max((n[3] for n in clean), default=0.0)
    if last_offset > duration + 0.5:
        print(f"WARNING {track.track_id}: last note ends at {last_offset:.2f}s "
              f"but audio is {duration:.2f}s")
        stats["duration_mismatch_clips"] += 1

    meta = {"split": track.split, "track_id": track.track_id, "tab_id": track.tab_id,
            "synth": track.synth, "duration": duration}
    hdf5_path = os.path.join(hdf5_dir, f"{track.track_id}.h5")
    write_hdf5(hdf5_path, waveform, midi_events, midi_event_times, clean, meta)

    row = {**meta, "n_notes": len(clean),
           # POSIX separators so a manifest built on Windows loads on Linux (EC2).
           "hdf5_path": os.path.relpath(hdf5_path, os.path.dirname(hdf5_dir)).replace(os.sep, "/"),
           "audio_path": track.audio_path}
    return row, stats, fret_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--synthtab_root", required=True,
                        help="root of the downloaded SynthTab dataset")
    parser.add_argument("--out_dir", default="features/synthtab")
    parser.add_argument("--no_kong_events", action="store_true",
                        help="skip midi_event* (decoder-only; smaller files)")
    parser.add_argument("--limit", type=int, default=None,
                        help="process only the first N clips (smoke test)")
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel worker processes (clips are independent); "
                             "audio decode/resample is the bottleneck")
    args = parser.parse_args()

    hdf5_dir = os.path.join(args.out_dir, "hdf5s")
    os.makedirs(hdf5_dir, exist_ok=True)
    stats, fret_counts, rows = Counter(), Counter(), []

    worker = functools.partial(process_track, hdf5_dir=hdf5_dir,
                               kong_events=not args.no_kong_events)
    # Materialize so we know the total up front (just directory walks, no file
    # reads) -- lets the progress line show N/total and an ETA.
    tracks = list(iter_synthtab_tracks(args.synthtab_root, limit=args.limit))
    total = len(tracks)
    print(f"found {total} renders with audio+labels; processing with "
          f"{args.workers} worker(s)", flush=True)
    pool = mp.Pool(args.workers) if args.workers > 1 else None
    results = pool.imap_unordered(worker, tracks, chunksize=1) if pool else map(worker, tracks)
    t0 = time.time()
    try:
        for i, (row, st, fc) in enumerate(results):
            stats.update(st)          # Counter.update adds counts
            fret_counts.update(fc)
            if row is not None:       # None = skipped (e.g. non-standard tuning)
                rows.append(row)
            if (i + 1) % 50 == 0 or (i + 1) == total:
                el = time.time() - t0
                rate = (i + 1) / el if el else 0.0
                eta = (total - (i + 1)) / rate / 60 if rate else 0.0
                print(f"  {i + 1}/{total} clips ({len(rows)} written, "
                      f"{stats['skipped_tracks']} skipped) -- "
                      f"{rate:.1f} clips/s, ETA {eta:.1f} min", flush=True)
    finally:
        if pool:
            pool.close()
            pool.join()

    # --- Manifest ---
    manifest_path = os.path.join(args.out_dir, "manifest.csv")
    fields = ["track_id", "tab_id", "synth", "split", "duration", "n_notes",
              "hdf5_path", "audio_path"]
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # --- Validation: no Guitar Pro file may appear in more than one split ---
    splits_by_file = defaultdict(set)
    for row in rows:
        splits_by_file[gp_file(row["tab_id"])].add(row["split"])
    leaked = {t: sorted(s) for t, s in splits_by_file.items() if len(s) > 1}
    assert not leaked, f"song file(s) span multiple splits: {dict(list(leaked.items())[:3])}"

    summary = {
        "clips_per_split": dict(Counter(r["split"] for r in rows)),
        "gp_files_per_split": {s: len({gp_file(r["tab_id"]) for r in rows if r["split"] == s})
                               for s in SPLITS},
        "hours_per_split": {s: round(sum(r["duration"] for r in rows if r["split"] == s) / 3600, 3)
                            for s in SPLITS},
        "label_stats": dict(stats),
        "fret_histogram": {int(k): v for k, v in sorted(fret_counts.items())},
    }
    with open(os.path.join(args.out_dir, "stats.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nWrote {len(rows)} HDF5 files to {hdf5_dir}\nManifest: {manifest_path}")


if __name__ == "__main__":
    main()
