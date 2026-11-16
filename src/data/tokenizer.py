"""Tokenizer for the autoregressive tab decoder.

Each note becomes 4 tokens, ONSET DUR STRING FRET, and one window becomes

    BOS, [ONSET, DUR, STRING, FRET] x N, EOS

Notes are ordered by (onset frame, string), so the notes of a chord share one
ONSET value and are written from the low E string up.

Notes go in and come out in the same format as `tab_notes` from preprocessing.py:
    float (N, 4) = [string, fret, onset_s, offset_s], string 0 = low E.

Time:
  - Windows are `segment_seconds` long (default 4 s) with 10 ms frames, matching
    the Kong encoder's 100 frames/s.
  - ONSET_k: onset rounded to the nearest frame k in [0, n_onset_frames).
  - DUR_j: duration in frames, snapped to the nearest of 69 log-spaced bins
    (exact up to 19 frames, then at most 1 frame or ~3% error, well inside
    mir_eval's offset tolerance of max(50 ms, 20% of the duration)).
  - Vocab with the defaults: 3 specials + 400 ONSET + 69 DUR + 6 STRING + 20 FRET = 498.

Window rules (encode):
  - Window [start, start + W) keeps the notes whose onset lies inside it.
  - Offsets are clipped to the window end.
  - After quantization, two notes on the same string in the same frame are
    merged and the longer one is kept.

Known limitation: a note that starts in one window and rings into the next is not
re-emitted in the next window (there are no MT3-style "tie" tokens), so after
stitching, its offset stops at the window boundary. Onset-only Note F1 and Tab F1
are unaffected. If offset F1 matters later, add tie tokens or use overlapping
windows.

Numpy only (no torch), so the data loader, decoder and metrics code can all
import it.

Example:
    tok = TabTokenizer()
    ids = tok.encode(tab_notes, start_s=8.0)            # one 4 s window
    notes = tok.decode(ids, start_s=8.0)                # back to (N, 4), absolute times
    full = stitch([tok.decode(tok.encode(tab_notes, s), s)
                   for s in tok.window_starts(duration)])
"""

from dataclasses import asdict, dataclass

import numpy as np

# Keep in sync with preprocessing.py. It isn't imported because importing it pulls
# in mirdata, librosa and h5py.
N_STRINGS = 6  # len(preprocessing.OPEN_STRING_MIDI)
MAX_FRET = 19  # preprocessing.MAX_FRET

PAD_ID, BOS_ID, EOS_ID = 0, 1, 2
N_SPECIAL = 3

# Token types, as returned by TabTokenizer.token_type / token_types.
PAD, BOS, EOS, ONSET, DUR, STRING, FRET, INVALID = range(8)
TYPE_NAMES = ["PAD", "BOS", "EOS", "ONSET", "DUR", "STRING", "FRET", "INVALID"]
NOTE_PATTERN = (ONSET, DUR, STRING, FRET)


@dataclass(frozen=True)
class TabTokenizer:
    segment_seconds: float = 4.0
    frames_per_second: int = 100
    n_duration_bins: int = 100  # before de-duplicating bins that round to the same frame
    n_strings: int = N_STRINGS
    max_fret: int = MAX_FRET

    def __post_init__(self):
        n_onset = int(round(self.segment_seconds * self.frames_per_second))
        if n_onset < 1:
            raise ValueError("segment_seconds * frames_per_second must be >= 1")
        dur_frames = np.unique(
            np.round(np.geomspace(1, n_onset, self.n_duration_bins)).astype(np.int64))

        def put(name, value):  # frozen dataclass: derived attributes go in via object.__setattr__
            object.__setattr__(self, name, value)

        put("n_onset_frames", n_onset)
        put("window_seconds", n_onset / self.frames_per_second)
        put("duration_frames", dur_frames)  # bin j <-> duration_frames[j] frames
        put("onset_offset", N_SPECIAL)
        put("dur_offset", N_SPECIAL + n_onset)
        put("string_offset", self.dur_offset + len(dur_frames))
        put("fret_offset", self.string_offset + self.n_strings)
        put("vocab_size", self.fret_offset + self.max_fret + 1)

        types = np.empty(self.vocab_size, dtype=np.int64)
        types[[PAD_ID, BOS_ID, EOS_ID]] = [PAD, BOS, EOS]
        types[self.onset_offset:self.dur_offset] = ONSET
        types[self.dur_offset:self.string_offset] = DUR
        types[self.string_offset:self.fret_offset] = STRING
        types[self.fret_offset:] = FRET
        put("_types", types)
        put("_log_dur", np.log(dur_frames))

    # ----- vocabulary -----

    pad_id = PAD_ID
    bos_id = BOS_ID
    eos_id = EOS_ID

    @property
    def n_duration_tokens(self):
        return len(self.duration_frames)

    def token_types(self, ids):
        """Vectorized token type (PAD/BOS/.../FRET, INVALID if out of range)."""
        ids = np.asarray(ids, dtype=np.int64)
        out = np.full(ids.shape, INVALID, dtype=np.int64)
        ok = (ids >= 0) & (ids < self.vocab_size)
        out[ok] = self._types[ids[ok]]
        return out

    def token_type(self, i):
        return int(self.token_types(np.array([i]))[0])

    def id_to_str(self, i):
        i = int(i)
        t = self.token_type(i)
        if t == ONSET:
            return f"ONSET_{i - self.onset_offset}"
        if t == DUR:
            return f"DUR_{i - self.dur_offset}"
        if t == STRING:
            return f"STRING_{i - self.string_offset}"
        if t == FRET:
            return f"FRET_{i - self.fret_offset}"
        return TYPE_NAMES[t] if t != INVALID else f"INVALID_{i}"

    def str_to_id(self, s):
        if s in ("PAD", "BOS", "EOS"):
            return {"PAD": PAD_ID, "BOS": BOS_ID, "EOS": EOS_ID}[s]
        kind, _, value = s.partition("_")
        offsets = {"ONSET": (self.onset_offset, self.n_onset_frames),
                   "DUR": (self.dur_offset, self.n_duration_tokens),
                   "STRING": (self.string_offset, self.n_strings),
                   "FRET": (self.fret_offset, self.max_fret + 1)}
        if kind not in offsets or not value.isdigit() or int(value) >= offsets[kind][1]:
            raise ValueError(f"unknown token string {s!r}")
        return offsets[kind][0] + int(value)

    def describe(self, ids):
        """Space-separated token names, for debugging."""
        return " ".join(self.id_to_str(i) for i in np.asarray(ids).ravel())

    # ----- encoding -----

    def window_starts(self, duration_s):
        """Start times of non-overlapping windows that cover [0, duration_s)."""
        n = max(1, int(np.ceil(duration_s / self.window_seconds - 1e-9)))
        return np.arange(n) * self.window_seconds

    def notes_in_window(self, tab_notes, start_s=0.0):
        """Notes with onset in [start_s, start_s + W), offsets clipped to the window
        end, times relative to start_s. Same (M, 4) layout and order as the input."""
        notes = self._check_notes(tab_notes)
        end = start_s + self.window_seconds
        w = notes[(notes[:, 2] >= start_s) & (notes[:, 2] < end)].copy()
        w[:, 3] = np.minimum(w[:, 3], end)
        w[:, 2:] -= start_s
        return w

    def encode(self, tab_notes, start_s=0.0, return_stats=False):
        """Token ids (list of int, BOS ... EOS) for the window starting at start_s."""
        notes = self._check_notes(tab_notes)
        end = start_s + self.window_seconds
        w = notes[(notes[:, 2] >= start_s) & (notes[:, 2] < end)]
        fps = self.frames_per_second

        rel_on = (w[:, 2] - start_s) * fps
        rel_off = (np.minimum(w[:, 3], end) - start_s) * fps
        onset = np.clip(np.rint(rel_on), 0, self.n_onset_frames - 1).astype(np.int64)
        dur = np.maximum(1, np.rint(rel_off - rel_on)).astype(np.int64)
        string = w[:, 0].astype(np.int64)
        fret = w[:, 1].astype(np.int64)

        # Re-sort on the quantized onset (raw float order can swap within a frame),
        # longest first so the de-dup below keeps the longer note.
        order = np.lexsort((-dur, string, onset))
        onset, string, fret, dur = onset[order], string[order], fret[order], dur[order]
        first = np.ones(len(onset), dtype=bool)
        first[1:] = (onset[1:] != onset[:-1]) | (string[1:] != string[:-1])
        onset, string, fret, dur = onset[first], string[first], fret[first], dur[first]

        body = np.stack([onset + self.onset_offset,
                         self.duration_bin(dur) + self.dur_offset,
                         string + self.string_offset,
                         fret + self.fret_offset], axis=1).ravel()
        ids = [BOS_ID, *body.tolist(), EOS_ID]
        if not return_stats:
            return ids
        stats = {"n_notes": int(first.sum()),
                 "offset_clipped": int((w[:, 3] > end).sum()),
                 "same_frame_merged": int((~first).sum())}
        return ids, stats

    def duration_bin(self, frames):
        """Index of the nearest (in log space) duration bin for each duration in frames."""
        frames = np.maximum(np.asarray(frames, dtype=np.float64), 1.0)
        return np.abs(np.log(frames)[..., None] - self._log_dur).argmin(axis=-1)

    # ----- decoding -----

    def decode(self, ids, start_s=0.0, strict=False, return_stats=False):
        """Token ids -> float (N, 4) [string, fret, onset_s, offset_s], absolute times.

        A leading BOS is optional; decoding stops at EOS, PAD or the end. Groups that
        aren't exactly ONSET DUR STRING FRET raise if strict, otherwise they are
        skipped up to the next ONSET token and counted.
        """
        ids = np.asarray(ids).astype(np.int64).ravel()
        types = self.token_types(ids)
        i = 1 if len(ids) and ids[0] == BOS_ID else 0
        rows, malformed, ended_with_eos = [], 0, False
        while i < len(ids):
            if types[i] in (EOS, PAD):
                ended_with_eos = bool(types[i] == EOS)
                break
            if tuple(types[i:i + 4]) == NOTE_PATTERN:
                on, d, s, f = ids[i:i + 4]
                rows.append((s - self.string_offset, f - self.fret_offset,
                             on - self.onset_offset, self.duration_frames[d - self.dur_offset]))
                i += 4
                continue
            if strict:
                raise ValueError(f"malformed note group at position {i}: "
                                 f"{self.describe(ids[i:i + 4])}")
            malformed += 1
            i += 1
            while i < len(ids) and types[i] not in (ONSET, EOS, PAD):
                i += 1

        out = np.zeros((len(rows), 4), dtype=np.float64)
        if rows:
            r = np.array(rows, dtype=np.float64)
            fps = self.frames_per_second
            out[:, 0], out[:, 1] = r[:, 0], r[:, 1]
            out[:, 2] = start_s + r[:, 2] / fps
            out[:, 3] = np.minimum(out[:, 2] + r[:, 3] / fps, start_s + self.window_seconds)
        if not return_stats:
            return out
        return out, {"n_notes": len(rows), "malformed_groups": malformed,
                     "ended_with_eos": ended_with_eos}

    def next_token_mask(self, prefix_ids):
        """Boolean [vocab_size] mask of tokens allowed after prefix_ids (which starts
        with BOS), for constrained decoding:
          - at a note boundary: ONSET >= the previous onset, or EOS
          - then DUR, then STRING, then FRET
          - within one onset (a chord), strings must strictly increase
          - after EOS only PAD (so finished rows in a batch just pad)
        Raises ValueError if the prefix itself breaks the grammar.
        """
        ids = np.asarray(prefix_ids, dtype=np.int64).ravel()
        if len(ids) == 0 or ids[0] != BOS_ID:
            raise ValueError("prefix must start with BOS")
        body, mask = ids[1:], np.zeros(self.vocab_size, dtype=bool)
        types = self.token_types(body)
        finished = np.flatnonzero(types == EOS)
        if len(finished):  # BOS notes... EOS PAD*: only PAD may follow
            e = finished[0]
            if e % 4 == 0 and np.all(types[e + 1:] == PAD) and \
                    np.array_equal(types[:e], np.resize(NOTE_PATTERN, e)):
                mask[PAD_ID] = True
                return mask
            raise ValueError(f"prefix is not grammatical: {self.describe(ids)}")
        if not np.array_equal(types, np.resize(NOTE_PATTERN, len(body))):
            raise ValueError(f"prefix is not grammatical: {self.describe(ids)}")

        n_notes, pos = len(body) // 4, len(body) % 4
        if n_notes:
            prev_onset = body[4 * (n_notes - 1)] - self.onset_offset
            prev_string = body[4 * (n_notes - 1) + 2] - self.string_offset
        if pos == 0:
            lo = 0
            if n_notes:  # same onset is only possible if a higher string is still free
                lo = prev_onset if prev_string < self.n_strings - 1 else prev_onset + 1
            mask[self.onset_offset + lo:self.dur_offset] = True
            mask[EOS_ID] = True
        elif pos == 1:
            mask[self.dur_offset:self.string_offset] = True
        elif pos == 2:
            cur_onset = body[-2] - self.onset_offset
            lo = prev_string + 1 if n_notes and cur_onset == prev_onset else 0
            mask[self.string_offset + lo:self.fret_offset] = True
        else:
            mask[self.fret_offset:] = True
        return mask

    # ----- config -----

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    # ----- helpers -----

    def _check_notes(self, tab_notes):
        notes = np.asarray(tab_notes, dtype=np.float64)
        if notes.size == 0:
            return notes.reshape(0, 4)
        if notes.ndim != 2 or notes.shape[1] != 4:
            raise ValueError(f"tab_notes must be (N, 4), got {notes.shape}")
        s, f = notes[:, 0], notes[:, 1]
        if np.any(s != np.round(s)) or np.any(f != np.round(f)):
            raise ValueError("string and fret must be whole numbers")
        if s.min() < 0 or s.max() >= self.n_strings:
            raise ValueError(f"string out of range 0..{self.n_strings - 1}")
        if f.min() < 0 or f.max() > self.max_fret:
            raise ValueError(f"fret out of range 0..{self.max_fret}")
        if np.any(notes[:, 3] < notes[:, 2]):
            raise ValueError("offset before onset")
        return notes


def stitch(segments):
    """Join decoded windows (already in absolute time) into one note array, sorted
    by onset, then string."""
    arrs = [np.asarray(s, dtype=np.float64).reshape(-1, 4) for s in segments]
    if not arrs:
        return np.zeros((0, 4))
    notes = np.concatenate(arrs)
    return notes[np.lexsort((notes[:, 0], notes[:, 2]))]
