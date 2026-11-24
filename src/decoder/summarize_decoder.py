"""Summarize decoder results: mean +- std over the six folds, or the zero-shot run.

    python src/decoder/summarize_decoder.py --arch decoder1 --stage stage2
    python src/decoder/summarize_decoder.py --arch decoder1 --stage stage1 --folds 05
    python src/decoder/summarize_decoder.py --arch decoder2 --stage stage2     # zero-shot

decoder1 (GuitarSet folds): same layout as src/headA/summarize_head_a.py: reads
<runs>/<stage>_test{T}/eval.json for every test player in scripts/run_folds.py's
PLAYERS (or just --folds) and reports each fold's test numbers plus the mean and
standard deviation over folds (the number to quote).
decoder2 (SynthTab, --zeroshot, the default for --arch decoder2): reads the single
<runs>/<stage>_zeroshot/eval.json (all GuitarSet clips).

One table per decoding in eval.json (greedy, beam4, ...). Columns: frame pitch F1 /
tab F1 / TDR (Table 2), note-level Note F1 / Tab F1 (with and without offsets;
note-level Tab F1 is TART's metric), playability, notes per window, windows that hit
max_len without EOS. Writes <runs>/<stage>_lopo.json (or _folds<...>.json /
_zeroshot_summary.json), keyed by decoding.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from run_folds import PLAYERS  # noqa: E402

# (name, getter on eval.json["test"][<decoding>]); same list as summarize_head_a.py, keep in sync.
NOTE_METRICS = [
    ("note_f1", lambda r: r["note_onset"]["pooled"]["f1"]),
    ("note_f1_offset", lambda r: r["note_offset"]["pooled"]["f1"]),
    ("tab_f1", lambda r: r["tab_onset"]["pooled"]["f1"]),
    ("tab_f1_offset", lambda r: r["tab_offset"]["pooled"]["f1"]),
    ("playability", lambda r: r["playability"]["pooled"]["rate"]),
    ("note_f1_clip_mean", lambda r: r["note_onset"]["mean_f1"]),
    ("tab_f1_clip_mean", lambda r: r["tab_onset"]["mean_f1"]),
    ("note_precision", lambda r: r["note_onset"]["pooled"]["precision"]),
    ("note_recall", lambda r: r["note_onset"]["pooled"]["recall"]),
]
FRAME_METRICS = [
    ("frame_pitch_f1", lambda r: r["frame"]["pitch"]["f1"]),
    ("frame_tab_f1", lambda r: r["frame"]["tab"]["f1"]),
    ("tdr", lambda r: r["frame"]["tdr"]),
]


def decodings(run_dir):
    return list(json.loads((Path(run_dir) / "eval.json").read_text())["test"])


def fold_row(run_dir, decoding="greedy"):
    ev = json.loads((Path(run_dir) / "eval.json").read_text())
    test = ev["test"][decoding]
    row = {name: get(test) for name, get in NOTE_METRICS}
    if "frame" in test:
        row.update({name: get(test) for name, get in FRAME_METRICS})
    stats = ev["decode_stats"]
    stats = stats if "windows" in stats else stats[decoding]     # older runs: one decoding
    row["hit_max_len"] = float(stats["hit_max_len"])
    row["notes_per_window"] = stats["notes"] / max(stats["windows"], 1)
    return row


def run_dirs(runs, stage, test_players=None, zeroshot=False):
    if zeroshot:
        return {"zeroshot": Path(runs) / f"{stage}_zeroshot"}
    return {f"test{t}": Path(runs) / f"{stage}_test{t}" for t in PLAYERS
            if test_players is None or t in test_players}


def summarize(runs, stage="stage1", test_players=None, decoding="greedy", zeroshot=False):
    folds = {}
    for name, run_dir in run_dirs(runs, stage, test_players, zeroshot).items():
        if not (run_dir / "eval.json").is_file():
            raise FileNotFoundError(f"missing {run_dir / 'eval.json'}; run "
                                    + ("the zero-shot run" if zeroshot
                                       else "all folds first (scripts/run_folds.py)"))
        folds[name] = fold_row(run_dir, decoding)
    keys = list(next(iter(folds.values())))
    mean = {k: float(np.mean([f[k] for f in folds.values()])) for k in keys}
    std = {k: (float(np.std([f[k] for f in folds.values()], ddof=1)) if len(folds) > 1 else 0.0)
           for k in keys}
    return {"folds": folds, "mean": mean, "std": std}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--arch", default="decoder1", help="decoder1 or decoder2 (sets --runs)")
    p.add_argument("--runs", default=None, help="default runs/<arch>")
    p.add_argument("--stage", default="stage1", help="run folder prefix, e.g. stage2")
    p.add_argument("--folds", nargs="+", default=None, help="only these test players")
    p.add_argument("--zeroshot", action="store_true", default=None,
                   help="read <stage>_zeroshot (default for decoder2)")
    a = p.parse_args()
    runs = Path(a.runs or REPO / "runs" / a.arch)
    zeroshot = a.zeroshot if a.zeroshot is not None else a.arch == "decoder2"
    folds = [f.zfill(2) for f in a.folds] if a.folds else None
    first = next(iter(run_dirs(runs, a.stage, folds, zeroshot).values()))
    out = {}
    for decoding in decodings(first):
        s = summarize(runs, a.stage, folds, decoding, zeroshot)
        out[decoding] = s
        cols = [c for c in ("frame_pitch_f1", "frame_tab_f1", "tdr") if c in s["mean"]]
        head = {"frame_pitch_f1": "fr pitch", "frame_tab_f1": "fr tab", "tdr": "TDR"}
        cols += ["note_f1", "note_f1_offset", "tab_f1", "tab_f1_offset", "playability",
                 "notes_per_window", "hit_max_len"]
        head.update({"note_f1": "note", "note_f1_offset": "note+off", "tab_f1": "tab",
                     "tab_f1_offset": "tab+off", "playability": "playable",
                     "notes_per_window": "notes/win", "hit_max_len": "max_len"})
        print(f"\n{decoding}")
        print(f"{'fold':<14}" + "".join(f"{head[c]:>11}" for c in cols))
        for name, f in s["folds"].items():
            print(f"{name:<14}" + "".join(f"{f[c]:11.3f}" for c in cols))
        if len(s["folds"]) > 1:
            print(f"{'mean':<14}" + "".join(f"{s['mean'][c]:11.3f}" for c in cols))
            print(f"{'std':<14}" + "".join(f"{s['std'][c]:11.3f}" for c in cols))
    suffix = ("_zeroshot_summary" if zeroshot
              else "_lopo" if folds is None else "_folds" + "-".join(folds))
    path = runs / f"{a.stage}{suffix}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
