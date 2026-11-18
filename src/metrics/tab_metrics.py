"""Shared evaluation metrics for every model (Head A, Head C, TabCNN, the AR decoder).

Note-level metrics take notes in the `tab_notes` layout used everywhere:
(N, 4) rows of [string, fret, onset_s, offset_s], string 0 = low E, frets 0..19.

    m = note_metrics(ref_notes, est_notes)      # one clip
    m["note_onset"]["f1"], m["tab_onset"]["f1"]
    p = playability(est_notes)["rate"]
    summary = aggregate([{**note_metrics(r, e), "playability": playability(e)}
                         for r, e in clips])

Definitions:
  Note F1     mir_eval note matching on pitch (50 cents) + onset (+-50 ms).
              note_onset = onset only (main number); note_offset also checks
              the offset (mir_eval default: 20% of the note length, at least 50 ms).
              Unisons are not merged (only 222 of 62,476 notes in features/v1).
  Tab F1      same matching, but the string and fret must also be right.
  Playability fraction of sounding frames a hand could play: no string has more
              than one note, and fretted notes (fret > 0) fit in a `max_fret_span`
              fret window (default 4 -> max - min <= 3). Frames are half-open
              [round(on * fps), round(off * fps)), so a note ending exactly where
              the next one on that string starts doesn't count as an overlap.
              GuitarSet's own labels score 0.992 (features/v1, all 360 clips).
  Frame P/R/F frame_metrics: same definitions as src/tabcnn/metrics.py (pitch_* /
              tab_*), vectorized; for comparing with the TabCNN replication.

Report both aggregate() numbers: pooled over all notes, and the mean over clips
(the usual transcription convention).

Named tab_metrics.py, not metrics.py: src/tabcnn/metrics.py would clash under
the repo's flat sys.path imports.
"""

from __future__ import annotations

import mir_eval
import numpy as np
import torch

# Keep in sync with src/data/preprocessing.py and src/data/loader.py.
OPEN_STRING_MIDI = np.array([40, 45, 50, 55, 59, 64])  # standard tuning, low E -> high e
N_STRINGS = len(OPEN_STRING_MIDI)
MAX_FRET = 19
FRAMES_PER_SECOND = 100

N_GUITAR_PITCHES = 44          # MIDI 40..83, TabCNN's pitch vector
LOWEST_GUITAR_MIDI = 40
FRAMES_PER_WINDOW = 1000       # frame 1000 of a 10 s window = frame 0 of the next
MIN_DURATION = 0.01            # mir_eval rejects notes with offset <= onset
TAB_PITCH_STRIDE = 32          # fake-pitch spacing per string for Tab F1 (> MAX_FRET)


def _prf(tp, n_pred, n_gt):
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gt if n_gt else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f}


# ----- note level -----

def _as_notes(notes):
    notes = np.asarray(notes, dtype=np.float64).reshape(-1, 4)
    if len(notes) and ((notes[:, 1] < 0).any() or (notes[:, 1] > MAX_FRET).any()):
        raise ValueError(f"frets must be in 0..{MAX_FRET}")
    return notes


def _intervals(notes):
    on = notes[:, 2]
    return np.stack([on, np.maximum(notes[:, 3], on + MIN_DURATION)], axis=1)


def _midi_to_hz(midi):
    return 440.0 * 2.0 ** ((np.asarray(midi, dtype=np.float64) - 69) / 12)


def _pitch_hz(notes):
    return _midi_to_hz(OPEN_STRING_MIDI[notes[:, 0].astype(int)] + notes[:, 1])


def _tab_hz(notes):
    # Each (string, fret) gets its own fake pitch, >= 1 semitone from every other,
    # so the 50-cent pitch matcher only pairs exact string + fret.
    return _midi_to_hz(20 + TAB_PITCH_STRIDE * notes[:, 0] + notes[:, 1])


def _match(ref, est, to_hz, onset_tolerance, offset_ratio, offset_min_tolerance):
    n_ref, n_est = len(ref), len(est)
    if n_ref == 0 or n_est == 0:
        n_match = 0
    else:
        n_match = len(mir_eval.transcription.match_notes(
            _intervals(ref), to_hz(ref), _intervals(est), to_hz(est),
            onset_tolerance=onset_tolerance, offset_ratio=offset_ratio,
            offset_min_tolerance=offset_min_tolerance))
    return {**_prf(n_match, n_est, n_ref), "n_ref": n_ref, "n_est": n_est, "n_match": n_match}


def note_metrics(ref, est, onset_tolerance=0.05, offset_ratio=0.2, offset_min_tolerance=0.05):
    """Note F1 and Tab F1 (onset-only and with offset) for one clip.

    ref, est: (N, 4) [string, fret, onset_s, offset_s]. Returns
    {note_onset, note_offset, tab_onset, tab_offset}, each
    {precision, recall, f1, n_ref, n_est, n_match}.
    """
    ref, est = _as_notes(ref), _as_notes(est)
    out = {}
    for name, to_hz in (("note", _pitch_hz), ("tab", _tab_hz)):
        out[f"{name}_onset"] = _match(ref, est, to_hz, onset_tolerance, None,
                                      offset_min_tolerance)
        out[f"{name}_offset"] = _match(ref, est, to_hz, onset_tolerance, offset_ratio,
                                       offset_min_tolerance)
    return out


def playability(est, max_fret_span=4, fps=FRAMES_PER_SECOND):
    """Fraction of sounding frames a hand could play (see module docstring).

    Returns {rate, n_sounding_frames, n_playable_frames, n_same_string_frames,
    n_wide_span_frames}. A note shorter than one frame still occupies one frame.
    """
    notes = _as_notes(est)
    empty = {"rate": 0.0, "n_sounding_frames": 0, "n_playable_frames": 0,
             "n_same_string_frames": 0, "n_wide_span_frames": 0}
    if not len(notes):
        return empty
    a = np.maximum(np.round(notes[:, 2] * fps).astype(int), 0)
    b = np.maximum(np.round(notes[:, 3] * fps).astype(int), a + 1)
    n_frames = int(b.max())
    strings, frets = notes[:, 0].astype(int), notes[:, 1].astype(int)

    diff = np.zeros((n_frames + 1, N_STRINGS), dtype=np.int64)
    np.add.at(diff, (a, strings), 1)
    np.add.at(diff, (b, strings), -1)
    per_string = np.cumsum(diff, axis=0)[:n_frames]                 # active notes per string
    lo = np.full(n_frames, np.inf)
    hi = np.full(n_frames, -np.inf)
    for i in np.nonzero(frets > 0)[0]:
        lo[a[i]:b[i]] = np.minimum(lo[a[i]:b[i]], frets[i])
        hi[a[i]:b[i]] = np.maximum(hi[a[i]:b[i]], frets[i])

    sounding = per_string.sum(1) > 0
    same_string = (per_string > 1).any(1) & sounding
    wide = (hi - lo > max_fret_span - 1) & sounding                 # inf - inf never > x
    playable = sounding & ~same_string & ~wide
    n = int(sounding.sum())
    return {"rate": int(playable.sum()) / n if n else 0.0, "n_sounding_frames": n,
            "n_playable_frames": int(playable.sum()),
            "n_same_string_frames": int(same_string.sum()), "n_wide_span_frames": int(wide.sum())}


def aggregate(per_clip):
    """Combine per-clip results into pooled and mean-over-clips numbers.

    per_clip: list of dicts mapping a metric name to a note_metrics entry
    (has n_match / n_ref / n_est) or a playability result (has n_sounding_frames).
    Returns {name: {"pooled": {...}, "mean_f1" | "mean_rate": float, "n_clips": int}}.
    """
    out = {}
    for name in per_clip[0] if per_clip else []:
        rows = [clip[name] for clip in per_clip]
        if "n_match" in rows[0]:
            tp, n_est, n_ref = (sum(r[k] for r in rows) for k in ("n_match", "n_est", "n_ref"))
            out[name] = {"pooled": {**_prf(tp, n_est, n_ref), "n_ref": n_ref, "n_est": n_est,
                                    "n_match": tp},
                         "mean_f1": float(np.mean([r["f1"] for r in rows])), "n_clips": len(rows)}
        elif "n_sounding_frames" in rows[0]:
            n = sum(r["n_sounding_frames"] for r in rows)
            ok = sum(r["n_playable_frames"] for r in rows)
            out[name] = {"pooled": {"rate": ok / n if n else 0.0, "n_sounding_frames": n,
                                    "n_playable_frames": ok},
                         "mean_rate": float(np.mean([r["rate"] for r in rows])),
                         "n_clips": len(rows)}
    return out


# ----- frame grids -----

def tab_grid_to_notes(tab, start_s=0.0, onsets=None, fps=FRAMES_PER_SECOND):
    """(T, 6) tab classes (0 silent, fret + 1) -> (N, 4) tab_notes rows, sorted by onset, string.

    A run of one class on a string becomes a note from a / fps to b / fps
    (inverse of the loader's inclusive frame rule). With an `onsets` (T, 6) bool
    grid, a new note also starts wherever it is True inside a run.
    """
    tab = np.asarray(tab)
    rows = []
    for s in range(tab.shape[1]):
        c = tab[:, s]
        starts = np.r_[True, c[1:] != c[:-1]]
        if onsets is not None:
            starts |= np.asarray(onsets)[:, s].astype(bool)
        idx = np.nonzero(starts)[0]
        ends = np.r_[idx[1:], len(c)] - 1
        for a, b in zip(idx, ends):
            if c[a] > 0:
                rows.append((s, c[a] - 1, start_s + a / fps, start_s + b / fps))
    notes = np.asarray(rows, dtype=np.float32).reshape(-1, 4)
    return notes[np.lexsort((notes[:, 0], notes[:, 2]))]


def _pitch_hot(tab):
    """(..., 6) tab classes -> (..., 44) bool pitch set (unisons collapse, like tab2pitch)."""
    tab = torch.as_tensor(tab).long()
    hot = torch.zeros(*tab.shape[:-1], N_GUITAR_PITCHES + 1, dtype=torch.bool)
    open_midi = torch.as_tensor(OPEN_STRING_MIDI)
    idx = torch.where(tab > 0, open_midi + tab - 1 - LOWEST_GUITAR_MIDI,
                      torch.full_like(tab, N_GUITAR_PITCHES))   # silent -> spare column
    hot.scatter_(-1, idx, True)
    return hot[..., :N_GUITAR_PITCHES]


def frame_metrics(pred_tab, gt_tab, n_valid):
    """Frame-level pitch and tab P/R/F over windows (N, T, 6) of tab classes.

    Same definitions as src/tabcnn/metrics.py (pitch_* / tab_*). Only frames
    < min(n_valid, 1000) count: padding is skipped, and frame 1000 is the next
    window's frame 0.
    """
    pred_tab, gt_tab = torch.as_tensor(pred_tab).long(), torch.as_tensor(gt_tab).long()
    n_valid = torch.as_tensor(n_valid).clamp(max=FRAMES_PER_WINDOW)
    keep = torch.arange(gt_tab.shape[1])[None, :] < n_valid[:, None]
    pred_tab, gt_tab = pred_tab[keep], gt_tab[keep]                 # (frames, 6)
    tab_tp = int(((pred_tab == gt_tab) & (gt_tab > 0)).sum())
    pred_hot, gt_hot = _pitch_hot(pred_tab), _pitch_hot(gt_tab)
    pitch_tp = int((pred_hot & gt_hot).sum())
    return {
        "pitch": _prf(pitch_tp, int(pred_hot.sum()), int(gt_hot.sum())),
        "tab": _prf(tab_tp, int((pred_tab > 0).sum()), int((gt_tab > 0).sum())),
    }
