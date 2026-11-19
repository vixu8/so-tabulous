"""Average Head A results over the six leave-one-player-out folds.

    python src/headA/summarize_head_a.py --runs runs/head_a --stage stage1
    python src/headA/summarize_head_a.py --stage stage2 --folds 05

Same as src/headC/summarize_head_c.py, for Head A runs: reads
<runs>/<stage>_test{T}/eval.json (note level, from eval_head_a.py) and
metrics.json (frame level, from train_head_a.py) for every test player in
scripts/run_folds.py's PLAYERS (or just --folds), and reports each fold's test
numbers plus the mean and standard deviation over folds (the number to quote).
Settings are fixed (sounding threshold 0.5, min_frames 1); there is no validation
player. The "argmax tab" column is Tab F1 at argmax (KongHeadA.predict), for
reference. Writes <runs>/<stage>_lopo.json (or <stage>_folds<...>.json).
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

# (name, getter on eval.json["test"]["fixed"]); same list as summarize_head_c.py, keep in sync.
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
    tr = json.loads((run_dir / "metrics.json").read_text())
    row = {name: get(ev["test"]["fixed"]) for name, get in NOTE_METRICS}
    row.update({
        "argmax_tab_f1": ev["test"]["argmax"]["tab_onset"]["pooled"]["f1"],
        "frame_tab_f1": tr["test"]["frame"][tr["threshold"]]["tab"]["f1"],
    })
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
    p.add_argument("--runs", default=str(REPO / "runs" / "head_a"))
    p.add_argument("--stage", default="stage1", help="run folder prefix, e.g. stage2")
    p.add_argument("--folds", nargs="+", default=None, help="only these test players")
    a = p.parse_args()
    folds = [f.zfill(2) for f in a.folds] if a.folds else None
    s = summarize(a.runs, a.stage, folds)

    cols = ["note_f1", "note_f1_offset", "tab_f1", "tab_f1_offset", "playability",
            "frame_tab_f1", "argmax_tab_f1"]
    head = ["note", "note+off", "tab", "tab+off", "playable", "frame tab", "argmax tab"]
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
