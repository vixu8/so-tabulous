"""Shared GuitarSet loader over the preprocessed HDF5 clips (features/v1).

One Dataset for every model (Head A, Head C, the AR decoder): each item is a
10 s audio crop plus the targets each model needs, all built from the same crop.

    splits = load_split("features/v1", test_player="05")     # train = the other 5 players
    train = GuitarSetDataset(splits["train"], "features/v1", mode="random",
                             targets=("string_per_pitch",))
    loader = DataLoader(train, batch_size=8, shuffle=True, num_workers=4,
                        collate_fn=collate)

Modes:
  - "random":  training. One random 10 s crop per clip per item
               (`crops_per_clip` items per clip per epoch).
  - "windows": val / test / inference. Every non-overlapping 10 s window of every
               clip, starting at 0, 10, 20 ... s (same as TabTokenizer.window_starts).
               The last window is zero-padded; `n_valid_frames` says how many
               frames are real.

Item (dict of numpy arrays / Python scalars; `collate` turns them into tensors):
  waveform        float32 (160000,)    16 kHz audio in [-1, 1], what KongEncoder takes
  tab             int64   (1001, 6)    0 = string silent, fret + 1 = that fret sounding
  tab_onset       bool    (1001, 6)    True at a note's onset frame
  track_id, start_s, n_valid_frames
Optional, via `targets=`:
  "kong"              Kong's TargetProcessor rolls, each float32 (1001, 88):
                      reg_onset_roll, reg_offset_roll, frame_roll, mask_roll,
                      onset_roll, offset_roll (velocity dropped: GuitarSet has none)
  "string_per_pitch"  int64 (1001, 88): string index sounding each piano pitch,
                      IGNORE_INDEX where silent (Head C's target)
  "tokens"            list[int]: TabTokenizer().encode(tab_notes, start_s) (decoder)

Frame rule (same as Kong's TargetProcessor, so `tab` and `frame_roll` line up):
frame = round((t - start_s) * 100); a note covers frames [onset, offset]
inclusive, clipped to the window. A note that started before the crop fills
frames from 0 but gets no onset. One difference from Kong: Kong masks out notes
that ring past the window end (mask_roll = 0); `tab` still marks them as sounding.

Kong's rolls are only meaningful where mask_roll == 1 (its loss uses the mask).
TargetProcessor has edge quirks that its mask mostly hides: in a clip's last
window it skips the final note-off and holds that note to the window end, and it
can miss notes that started before the crop. Measured on features/v1 train
(848 windows): `tab` and `frame_roll` agree on 1.35M sounding frames; only 1,459
unmasked frames are on in `frame_roll` but not `tab`, and 162 the other way.
Use `tab` / `tab_notes`, not Kong's rolls, as the reference for evaluation.

Splits come from the manifest's player_id, NOT the split stored in the files, so
one features/ folder serves all six leave-one-player-out folds.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from tokenizer import PAD_ID, TabTokenizer

# Keep in sync with preprocessing.py (tuning, MAX_FRET) and kong_encoder.py (audio grid).
OPEN_STRING_MIDI = [40, 45, 50, 55, 59, 64]  # standard tuning, low E -> high e
N_STRINGS = len(OPEN_STRING_MIDI)
MAX_FRET = 19
SAMPLE_RATE = 16_000
FRAMES_PER_SECOND = 100
HOP_SAMPLES = SAMPLE_RATE // FRAMES_PER_SECOND             # 160
SEGMENT_SECONDS = 10.0
SEGMENT_SAMPLES = int(SEGMENT_SECONDS * SAMPLE_RATE)       # 160000
N_FRAMES = SEGMENT_SAMPLES // HOP_SAMPLES + 1              # 1001, Kong's T for 10 s
BEGIN_NOTE = 21                                            # Kong pitch index 0 = MIDI 21 (A0)
N_PITCHES = 88
IGNORE_INDEX = -100                                        # torch cross-entropy default

KONG_ROLLS = ("reg_onset_roll", "reg_offset_roll", "frame_roll", "mask_roll",
              "onset_roll", "offset_roll")
TARGETS = ("kong", "string_per_pitch", "tokens")
MODES = ("random", "windows")
DEFAULT_KONG_REPO = (Path(__file__).resolve().parents[1]
                     / "kong_wrapper" / "third_party" / "piano_transcription")


# ----- splits -----

def load_split(root, test_player):
    """Manifest rows per split, by player: {"train", "test"} -> DataFrame.

    Leave-one-player-out with no validation player (same as TabCNN): the test
    player's clips are "test", the other 5 players' clips are "train".
    """
    manifest = pd.read_csv(Path(root) / "manifest.csv", dtype={"player_id": str})
    test = str(test_player).zfill(2)
    players = set(manifest.player_id)
    if test not in players:
        raise ValueError(f"player {test!r} not in manifest (players: {sorted(players)})")
    splits = {
        "train": manifest[manifest.player_id != test],
        "test": manifest[manifest.player_id == test],
    }
    assert not set(splits["train"].player_id) & set(splits["test"].player_id), "player leak"
    return {k: df.reset_index(drop=True) for k, df in splits.items()}


MANIFEST_COLUMNS = ("track_id", "hdf5_path")


def load_manifest(root):
    """Every manifest row of a features folder -> DataFrame (no split).

    Used for the SynthTab decoder (decoder2): it trains on every clip of a SynthTab
    features folder and is tested zero-shot on every GuitarSet clip. Any folder in
    the features/v1 layout works:
      manifest.csv  one row per clip; needs track_id and hdf5_path (relative to
                    root); duration and player_id are optional
      HDF5 files    waveform int16 (16 kHz mono) and tab_notes float32 (N, 4) =
                    [string, fret, onset_s, offset_s], string 0 = low E, standard
                    tuning, frets 0..MAX_FRET, sorted by onset then string
    (Assumed layout for the SynthTab subset until its preprocessing script lands.)
    """
    manifest = pd.read_csv(Path(root) / "manifest.csv", dtype={"player_id": str,
                                                               "track_id": str})
    missing = [c for c in MANIFEST_COLUMNS if c not in manifest.columns]
    if missing:
        raise ValueError(f"{Path(root) / 'manifest.csv'} lacks columns {missing}")
    if manifest.track_id.duplicated().any():
        raise ValueError(f"{Path(root) / 'manifest.csv'} has duplicate track_ids")
    return manifest.reset_index(drop=True)


# ----- target helpers (module level so heads / metrics can reuse them) -----

def notes_to_tab_grid(notes, start_s=0.0, n_frames=N_FRAMES):
    """`tab_notes` rows (absolute times) -> (tab int64 (T, 6), tab_onset bool (T, 6)).

    Notes are written in onset order, so if rounding makes two notes on one string
    share a frame, the later note wins.
    """
    tab = np.zeros((n_frames, N_STRINGS), dtype=np.int64)
    onset = np.zeros((n_frames, N_STRINGS), dtype=bool)
    notes = np.asarray(notes, dtype=np.float64).reshape(-1, 4)
    for string, fret, on, off in notes[np.argsort(notes[:, 2], kind="stable")]:
        bgn = int(round((on - start_s) * FRAMES_PER_SECOND))
        fin = int(round((off - start_s) * FRAMES_PER_SECOND))
        if fin < 0 or bgn > n_frames - 1:
            continue
        s = int(string)
        tab[max(bgn, 0):min(fin, n_frames - 1) + 1, s] = int(fret) + 1
        if bgn >= 0:
            onset[bgn, s] = True
    return tab, onset


def tab_to_string_per_pitch(tab):
    """(T, 6) tab classes -> (T, 88) string index per Kong pitch, IGNORE_INDEX if silent.

    A pitch sounding on two strings at once (unison) keeps the lower string index.
    """
    tab = np.asarray(tab)
    out = np.full((tab.shape[0], N_PITCHES), IGNORE_INDEX, dtype=np.int64)
    for s in reversed(range(N_STRINGS)):  # lower strings written last, so they win
        frames = np.nonzero(tab[:, s] > 0)[0]
        pitch = OPEN_STRING_MIDI[s] + tab[frames, s] - 1
        out[frames, pitch - BEGIN_NOTE] = s
    return out


_TARGET_PROCESSORS = {}


def _kong_target_processor(repo_path=DEFAULT_KONG_REPO):
    """Kong's own TargetProcessor (utils/utilities.py), imported without copying it."""
    repo = Path(repo_path).expanduser().resolve()
    if repo in _TARGET_PROCESSORS:
        return _TARGET_PROCESSORS[repo]
    utils_dir = repo / "utils"
    path = utils_dir / "utilities.py"
    if not path.is_file():
        raise FileNotFoundError(
            f"Kong's utilities.py not found at {path}. Clone bytedance/piano_transcription "
            "into src/kong_wrapper/third_party/ (see src/kong_wrapper/README.md).")
    # utilities.py does `from piano_vad import ...` and `import config` from its own folder.
    added = str(utils_dir) not in sys.path
    if added:
        sys.path.insert(0, str(utils_dir))
    try:
        spec = importlib.util.spec_from_file_location("_kong_utilities", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        raise ImportError(f"Missing {exc.name!r} for Kong's utilities.py; "
                          "install requirements/model.txt.") from exc
    finally:
        if added:
            sys.path.remove(str(utils_dir))
    processor = module.TargetProcessor(SEGMENT_SECONDS, FRAMES_PER_SECOND,
                                       BEGIN_NOTE, N_PITCHES)
    _TARGET_PROCESSORS[repo] = processor
    return processor


# ----- dataset -----

class GuitarSetDataset(torch.utils.data.Dataset):
    """10 s crops of GuitarSet clips with tab / Kong / token targets. See module docstring."""

    def __init__(self, rows, root, mode="random", targets=(), crops_per_clip=1,
                 kong_repo=DEFAULT_KONG_REPO):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        targets = tuple(targets)
        unknown = set(targets) - set(TARGETS)
        if unknown:
            raise ValueError(f"unknown targets {sorted(unknown)}; choose from {TARGETS}")
        if crops_per_clip < 1:
            raise ValueError("crops_per_clip must be >= 1")
        self.root = Path(root)
        self.mode = mode
        self.targets = targets
        self.crops_per_clip = int(crops_per_clip)
        self.kong_repo = kong_repo
        self.tokenizer = TabTokenizer(segment_seconds=SEGMENT_SECONDS,
                                      frames_per_second=FRAMES_PER_SECOND)
        if "kong" in targets:
            _kong_target_processor(kong_repo)  # fail now, not inside a worker

        rows = pd.DataFrame(rows)
        self.clips = []  # (track_id, hdf5 path, n_samples)
        for track_id, rel in zip(rows.track_id, rows.hdf5_path):
            path = self.root / rel
            with h5py.File(path, "r") as hf:
                n = int(hf["waveform"].shape[0])
            self.clips.append((str(track_id), path, n))

        # windows mode: (clip index, start sample) for every window
        self.windows = []
        if mode == "windows":
            for i, (_, _, n) in enumerate(self.clips):
                n_windows = max(1, int(np.ceil(n / SEGMENT_SAMPLES - 1e-9)))
                self.windows += [(i, k * SEGMENT_SAMPLES) for k in range(n_windows)]

    def __len__(self):
        if self.mode == "windows":
            return len(self.windows)
        return len(self.clips) * self.crops_per_clip

    def __getitem__(self, index):
        if self.mode == "windows":
            clip, start = self.windows[index]
        else:
            clip = index % len(self.clips)
            n = self.clips[clip][2]
            # torch RNG: reproducible under torch.manual_seed, distinct per DataLoader worker
            start = int(torch.randint(0, max(n - SEGMENT_SAMPLES, 0) + 1, ()).item())
        return self._load(clip, start)

    def _load(self, clip, start):
        track_id, path, n = self.clips[clip]
        start_s = start / SAMPLE_RATE
        with h5py.File(path, "r") as hf:
            audio = hf["waveform"][start:start + SEGMENT_SAMPLES]
            tab_notes = hf["tab_notes"][:]
            if "kong" in self.targets:
                events = [e.decode() for e in hf["midi_event"][:]]
                event_times = hf["midi_event_time"][:]

        waveform = np.zeros(SEGMENT_SAMPLES, dtype=np.float32)
        waveform[:len(audio)] = audio.astype(np.float32) / 32767.0
        tab, tab_onset = notes_to_tab_grid(tab_notes, start_s)
        item = {
            "waveform": waveform,
            "tab": tab,
            "tab_onset": tab_onset,
            "track_id": track_id,
            "start_s": start_s,
            "n_valid_frames": min(N_FRAMES, len(audio) // HOP_SAMPLES + 1),
        }
        if "string_per_pitch" in self.targets:
            item["string_per_pitch"] = tab_to_string_per_pitch(tab)
        if "tokens" in self.targets:
            item["tokens"] = self.tokenizer.encode(tab_notes, start_s)
        if "kong" in self.targets:
            rolls, _, _ = _kong_target_processor(self.kong_repo).process(
                start_s, event_times, events, extend_pedal=False)
            for key in KONG_ROLLS:
                item[key] = rolls[key].astype(np.float32)
        return item


def collate(batch):
    """List of items -> dict of batched tensors.

    Arrays are stacked; `tokens` are right-padded with PAD_ID and come with
    `token_lengths`; `track_id` and `start_s` stay Python lists.
    """
    out = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        if key in ("track_id", "start_s"):
            out[key] = values
        elif key == "tokens":
            lengths = [len(v) for v in values]
            padded = np.full((len(values), max(lengths)), PAD_ID, dtype=np.int64)
            for row, ids in zip(padded, values):
                row[:len(ids)] = ids
            out["tokens"] = torch.from_numpy(padded)
            out["token_lengths"] = torch.tensor(lengths, dtype=torch.int64)
        elif key == "n_valid_frames":
            out[key] = torch.tensor(values, dtype=torch.int64)
        else:
            out[key] = torch.from_numpy(np.stack(values))
    return out
