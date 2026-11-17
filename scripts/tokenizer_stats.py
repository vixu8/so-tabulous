"""Run the tab tokenizer over preprocessed GuitarSet HDF5s and report what it does
to the real data.

For each clip: slide non-overlapping windows, encode, decode, stitch, and compare
with the original tab_notes. Reports the sequence-length distribution (use p99/max
to set the decoder's max_len), label changes (clipped offsets, same-frame merges,
notes past the last window), and the oracle round-trip F1:
  - note_f1_onset:  pitch + onset (50 ms), should be ~1.0
  - note_f1_offset: pitch + onset + offset, a bit below 1 (window-boundary clipping)
  - tab_f1_onset:   string + fret + onset, should be ~1.0
F1 needs mir_eval (pip install mir_eval); without it, those numbers are skipped.

Usage:
  python scripts/tokenizer_stats.py --hdf5_dir features/v1/hdf5s
  python scripts/tokenizer_stats.py --hdf5_dir features/v1/hdf5s --split train --segment_seconds 4
"""

import argparse
import glob
import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "data"))
from tokenizer import TabTokenizer, stitch  # noqa: E402

OPEN_STRING_MIDI = np.array([40, 45, 50, 55, 59, 64])  # keep in sync with preprocessing.py


def midi_to_hz(m):
    return 440.0 * 2.0 ** ((np.asarray(m, dtype=np.float64) - 69) / 12)


def f1_scores(ref, est):
    import mir_eval

    def prf(ref_p, est_p, offset_ratio):
        return mir_eval.transcription.precision_recall_f1_overlap(
            ref[:, 2:4], ref_p, est[:, 2:4], est_p, offset_ratio=offset_ratio)[2]

    pitch = lambda n: midi_to_hz(OPEN_STRING_MIDI[n[:, 0].astype(int)] + n[:, 1])
    # Tab F1 via the same matcher: give each (string, fret) its own fake pitch, at
    # least a semitone from any other, so only the exact string+fret can match.
    tab_id = lambda n: midi_to_hz(20 + n[:, 0] * 20 + n[:, 1])
    return {"note_f1_onset": prf(pitch(ref), pitch(est), None),
            "note_f1_offset": prf(pitch(ref), pitch(est), 0.2),
            "tab_f1_onset": prf(tab_id(ref), tab_id(est), None)}


def pct(values, q):
    return float(np.percentile(values, q)) if len(values) else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hdf5_dir", required=True)
    parser.add_argument("--split", default=None, choices=["train", "validation", "test"],
                        help="only clips from this split (default: all)")
    parser.add_argument("--segment_seconds", type=float, default=10.0)
    parser.add_argument("--json_out", default=None, help="also write the summary here")
    args = parser.parse_args()

    try:
        import mir_eval  # noqa: F401
        have_mir_eval = True
    except ImportError:
        have_mir_eval = False
        print("mir_eval not installed: skipping round-trip F1 (pip install mir_eval)")

    tok = TabTokenizer(segment_seconds=args.segment_seconds)
    paths = sorted(glob.glob(os.path.join(args.hdf5_dir, "*.h5")))
    if not paths:
        raise SystemExit(f"no .h5 files in {args.hdf5_dir}")

    seq_lens, notes_per_window, scores = [], [], []
    totals = {"clips": 0, "windows": 0, "notes_in": 0, "notes_encoded": 0,
              "offset_clipped": 0, "same_frame_merged": 0, "notes_past_last_window": 0,
              "malformed_groups": 0}
    worst = (0, None)

    for path in paths:
        with h5py.File(path, "r") as hf:
            if args.split and hf.attrs["split"].decode() != args.split:
                continue
            duration = float(hf.attrs["duration"])
            tab = hf["tab_notes"][:].astype(np.float64)

        starts = tok.window_starts(duration)
        segments = []
        for start in starts:
            ids, st = tok.encode(tab, start_s=start, return_stats=True)
            dec, dst = tok.decode(ids, start_s=start, return_stats=True)
            segments.append(dec)
            seq_lens.append(len(ids))
            notes_per_window.append(st["n_notes"])
            if len(ids) > worst[0]:
                worst = (len(ids), f"{os.path.basename(path)} @ {start:.0f}s")
            for k in ["offset_clipped", "same_frame_merged"]:
                totals[k] += st[k]
            totals["notes_encoded"] += st["n_notes"]
            totals["malformed_groups"] += dst["malformed_groups"]
        totals["clips"] += 1
        totals["windows"] += len(starts)
        totals["notes_in"] += len(tab)
        totals["notes_past_last_window"] += int((tab[:, 2] >= starts[-1] + tok.window_seconds).sum())

        full = stitch(segments)
        if have_mir_eval and len(tab) and len(full):
            scores.append(f1_scores(tab, full))

    summary = {
        "tokenizer": tok.to_dict(),
        "vocab_size": tok.vocab_size,
        "totals": totals,
        "seq_len": {"mean": float(np.mean(seq_lens)), "p50": pct(seq_lens, 50),
                    "p95": pct(seq_lens, 95), "p99": pct(seq_lens, 99),
                    "max": int(max(seq_lens)), "max_at": worst[1]},
        "notes_per_window": {"mean": float(np.mean(notes_per_window)),
                             "p99": pct(notes_per_window, 99), "max": int(max(notes_per_window))},
    }
    if scores:
        summary["oracle_round_trip_f1_mean_over_clips"] = {
            k: round(float(np.mean([s[k] for s in scores])), 5) for k in scores[0]}

    print(json.dumps(summary, indent=2))
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
