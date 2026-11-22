"""Average decoder results over the six leave-one-player-out folds.

    python src/decoder/summarize_decoder.py --runs runs/decoder --stage stage1
    python src/decoder/summarize_decoder.py --stage stage2 --folds 05

Same layout as src/headA/summarize_head_a.py: reads <runs>/<stage>_test{T}/eval.json
(note level, from eval_decoder.py) for every test player in scripts/run_folds.py's
PLAYERS (or just --folds), and reports each fold's test numbers plus the mean and
standard deviation over folds (the number to quote). The last columns are decoding
stats: windows that hit max_len without EOS. Writes <runs>/<stage>_lopo.json (or
<stage>_folds<...>.json).
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

# (name, getter on eval.json["test"]["greedy"]); same list as summarize_head_a.py, keep in sync.
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


def fold_row(run_dir):
    ev = json.loads((run_dir / "eval.json").read_text())
    row = {name: get(ev["test"]["greedy"]) for name, get in NOTE_METRICS}
    stats = ev["decode_stats"]
    row["hit_max_len"] = float(stats["hit_max_len"])
    row["notes_per_window"] = stats["notes"] / max(stats["windows"], 1)
    return row


def summarize(runs, stage="stage1", test_players=None):
    folds = {}
    for test in PLAYERS:
        if test_players is not None and test not in test_players:
            continue
        run_dir = Path(runs) / f"{stage}_test{test}"
        if not (run_dir / "eval.json").is_file():
            raise FileNotFoundError(f"missing {run_dir / 'eval.json'}; run all folds first "
                                    "(scripts/run_folds.py)")
        folds[f"test{test}"] = fold_row(run_dir)
    keys = list(next(iter(folds.values())))
    mean = {k: float(np.mean([f[k] for f in folds.values()])) for k in keys}
    std = {k: (float(np.std([f[k] for f in folds.values()], ddof=1)) if len(folds) > 1 else 0.0)
           for k in keys}
    return {"folds": folds, "mean": mean, "std": std}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--runs", default=str(REPO / "runs" / "decoder"))
    p.add_argument("--stage", default="stage1", help="run folder prefix, e.g. stage2")
    p.add_argument("--folds", nargs="+", default=None, help="only these test players")
    a = p.parse_args()
    folds = [f.zfill(2) for f in a.folds] if a.folds else None
    s = summarize(a.runs, a.stage, folds)

    cols = ["note_f1", "note_f1_offset", "tab_f1", "tab_f1_offset", "playability",
            "notes_per_window", "hit_max_len"]
    head = ["note", "note+off", "tab", "tab+off", "playable", "notes/win", "max_len"]
    print(f"{'fold':<14}" + "".join(f"{h:>11}" for h in head))
    for name, f in s["folds"].items():
        print(f"{name:<14}" + "".join(f"{f[c]:11.3f}" for c in cols))
    print(f"{'mean':<14}" + "".join(f"{s['mean'][c]:11.3f}" for c in cols))
    print(f"{'std':<14}" + "".join(f"{s['std'][c]:11.3f}" for c in cols))
    suffix = "_lopo" if folds is None else "_folds" + "-".join(folds)
    out = Path(a.runs) / f"{a.stage}{suffix}.json"
    out.write_text(json.dumps(s, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
