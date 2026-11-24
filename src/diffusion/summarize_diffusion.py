"""Aggregate explicitly requested completed folds without selecting test settings."""

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(runs, folds, tag="stage1"):
    records = {}
    settings = {}
    for fold in folds:
        fold = str(fold).zfill(2)
        path = Path(runs) / f"{tag}_test{fold}" / "eval.json"
        evaluation = json.loads(path.read_text())
        if settings and settings != evaluation["decodings"]:
            raise ValueError("folds use different decoding settings")
        settings = evaluation["decodings"]
        for name, score in evaluation["test"].items():
            row = {k + "_f1": score[k]["pooled"]["f1"] for k in (
                "note_onset", "note_offset", "tab_onset", "tab_offset")}
            row.update(frame_pitch_f1=score["frame"]["pitch"]["f1"],
                       frame_tab_f1=score["frame"]["tab"]["f1"], tdr=score["frame"]["tdr"],
                       playability=score["playability"]["pooled"]["rate"],
                       seconds_per_window=evaluation["decode_stats"][name]["seconds_per_window"])
            records.setdefault(name, {})[fold] = row
    result = {}
    for name, rows in records.items():
        keys = next(iter(rows.values()))
        result[name] = {"folds": rows,
            "mean": {k: float(np.mean([r[k] for r in rows.values()])) for k in keys},
            "std": {k: float(np.std([r[k] for r in rows.values()], ddof=1)) if len(rows) > 1 else 0.0
                    for k in keys}}
    return {"decodings": settings, "results": result}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", default="runs/diffusion")
    p.add_argument("--folds", nargs="+", required=True, help="explicit folds, e.g. 05 or 00 01 02 03 04 05")
    p.add_argument("--tag", default="stage1")
    args = p.parse_args()
    result = summarize(args.runs, args.folds, args.tag)
    path = Path(args.runs) / f"{args.tag}_folds{'_'.join(args.folds)}.json"
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
