"""Head A note-level evaluation: per-frame fret classes -> notes -> metrics.

    python src/headA/eval_head_a.py --run_dir runs/head_a/stage1_val04_test05

Reads best.pt from train_head_a.py (its saved args give the data root, the
probability cache and the players) and, for every clip of the validation and
test players:
  1. stitches the cached 10 s windows of class probabilities back into the whole
     clip (1000 frames per window, n_valid in the last one; same as Head C),
  2. turns them into a tab grid (head_a.predict_tab_grid: argmax, or "a string
     sounds when 1 - p(silent) >= threshold"),
  3. turns each run of one fret on one string into a note
     (tab_metrics.tab_grid_to_notes, no onsets: Head A has no onset output, so a
     fret re-plucked without a gap merges into one note) and drops notes shorter
     than min_frames,
  4. scores Note F1 / Tab F1 / playability with src/metrics/tab_metrics.py
     (the same metrics and players as Head C and the decoder).

The sounding threshold and min_frames are swept on the validation player and
picked by pooled onset Tab F1. The test player is scored once with that setting
and once with the plain argmax at min_frames 1 (the `default`, Head A's own
predict), for reference.

Writes eval.json and test_notes.npz into --run_dir. The cache
(<run_dir>/probs_cache) is built by train_head_a.py from the run's best model and
deleted afterwards unless --keep_cache; keep it to re-run this script.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
for _sub in (HERE, HERE.parent / "data", HERE.parent / "metrics"):
    if str(_sub) not in sys.path:
        sys.path.insert(0, str(_sub))
from head_a import predict_tab_grid  # noqa: E402
from loader import load_split  # noqa: E402
from tab_metrics import aggregate, note_metrics, playability, tab_grid_to_notes  # noqa: E402

FRAMES_PER_SECOND = 100
FRAMES_PER_WINDOW = 1000            # frame 1000 of a window = frame 0 of the next
SOUND_THRESHOLDS = [None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]   # None = argmax
MIN_FRAMES = [1, 3, 5]
DEFAULT = (None, 1)                 # KongHeadA.predict, every run kept


def setting_key(setting):
    sound, min_frames = setting
    return f"sound={'argmax' if sound is None else sound},min_frames={min_frames}"


# ----- clip inputs -----

def stitch_windows(x, n_valid):
    """(W, T, ...) non-overlapping windows -> one clip (1000 * (W - 1) + n_valid[-1], ...).

    Same as eval_head_c.stitch_windows; keep in sync.
    """
    x = np.asarray(x) if not torch.is_tensor(x) else x
    parts = [x[i, :FRAMES_PER_WINDOW] for i in range(len(x) - 1)]
    parts.append(x[-1, :int(n_valid[-1])])
    return torch.cat(parts) if torch.is_tensor(x) else np.concatenate(parts)


def reference_notes(root, rows):
    """{track_id: tab_notes} from the clips' HDF5 files. Same as eval_head_c; keep in sync."""
    refs = {}
    for track_id, rel in zip(rows.track_id, rows.hdf5_path):
        with h5py.File(Path(root) / rel, "r") as hf:
            refs[track_id] = hf["tab_notes"][:]
    return refs


def clip_probs(cache_dir, track_id):
    """Cached class probabilities for one whole clip: float16 (T, 6, 21)."""
    with np.load(Path(cache_dir) / f"{track_id}.npz") as z:
        return stitch_windows(z["probs"], z["n_valid"])


# ----- notes and scoring -----

def clip_notes(probs, sound=None, min_frames=1):
    """(T, 6, 21) class probabilities -> (N, 4) tab_notes rows (times from the clip start)."""
    grid = predict_tab_grid(torch.as_tensor(np.asarray(probs, dtype=np.float32)), sound).numpy()
    notes = tab_grid_to_notes(grid)
    n_frames = np.rint((notes[:, 3] - notes[:, 2]) * FRAMES_PER_SECOND) + 1
    return notes[n_frames >= min_frames]


def _score_clip(task):
    """Worker: notes for every setting -> metrics. Top level for spawn."""
    track_id, probs, ref, sounds, min_frames = task
    results = {}
    for sound in sounds:
        for m in min_frames:
            est = clip_notes(probs, sound, m)
            results[(sound, m)] = {"metrics": {**note_metrics(ref, est),
                                               "playability": playability(est)},
                                   "est": est}
    return track_id, results


def make_pool(workers):
    """A spawn pool for score_clips (None for workers <= 1). Reuse it across calls."""
    return mp.get_context("spawn").Pool(workers) if workers > 1 else None


def score_clips(clips, refs, sounds=SOUND_THRESHOLDS, min_frames=MIN_FRAMES, pool=None,
                log=print):
    """Score whole clips for every (sound threshold, min_frames) setting.

    clips: {track_id: probs (T, 6, 21)}. Returns ({setting: aggregate},
    {setting: {track_id: est notes}}).
    """
    tasks = [(t, probs, refs[t], list(sounds), list(min_frames)) for t, probs in clips.items()]
    t0 = time.time()
    done = pool.map(_score_clip, tasks, chunksize=1) if pool else [_score_clip(t) for t in tasks]
    settings = [(s, m) for s in sounds for m in min_frames]
    log(f"  {len(tasks)} clips x {len(settings)} settings in {time.time() - t0:.0f} s")
    summary = {k: aggregate([res[k]["metrics"] for _, res in done]) for k in settings}
    notes = {k: {t: res[k]["est"] for t, res in done} for k in settings}
    return summary, notes


def best_setting(summary):
    return max(summary, key=lambda k: summary[k]["tab_onset"]["pooled"]["f1"])


def evaluate_split(rows, root, cache_dir, sounds, min_frames, workers, log=print):
    """Score every clip in `rows` (probabilities from `cache_dir`) for every setting."""
    refs = reference_notes(root, rows)
    clips = {t: clip_probs(cache_dir, t) for t in rows.track_id}
    pool = make_pool(workers)
    try:
        summary, notes = score_clips(clips, refs, sounds, min_frames, pool, log)
    finally:
        if pool:
            pool.close()
            pool.join()
    return summary, notes, refs


# ----- main -----

def _row(name, agg):
    g = lambda k: agg[k]["pooled"]["f1"]  # noqa: E731
    return (f"{name:<28} {g('note_onset'):.3f}   {g('note_offset'):.3f}    {g('tab_onset'):.3f}"
            f"   {g('tab_offset'):.3f}    {agg['playability']['pooled']['rate']:.3f}")


def run(run_dir, workers=None, sounds=SOUND_THRESHOLDS, min_frames=MIN_FRAMES, log=print):
    run_dir = Path(run_dir)
    args = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)["args"]
    workers = max(1, (os.cpu_count() or 2) - 1) if workers is None else workers
    splits = load_split(args["root"], args["val_player"], args["test_player"])
    sounds = [None] + sorted({s for s in sounds if s is not None})
    min_frames = sorted(set(min_frames) | {DEFAULT[1]})

    log(f"validation player {args['val_player']}: sweeping {len(sounds)} sounding thresholds "
        f"x {len(min_frames)} min_frames")
    val, _, _ = evaluate_split(splits["validation"], args["root"], args["cache_dir"],
                               sounds, min_frames, workers, log)
    sweep = {setting_key(k): {"note_onset_f1": a["note_onset"]["pooled"]["f1"],
                              "tab_onset_f1": a["tab_onset"]["pooled"]["f1"],
                              "playability": a["playability"]["pooled"]["rate"]}
             for k, a in val.items()}
    best = best_setting(val)
    log("val pooled onset Tab F1 (rows: sounding threshold, columns: min_frames)")
    log("          " + "".join(f"{m:>8}" for m in min_frames))
    for s in sounds:
        log(f"  {'argmax' if s is None else s:<8}"
            + "".join(f"{val[(s, m)]['tab_onset']['pooled']['f1']:8.3f}" for m in min_frames))
    log(f"chosen: {setting_key(best)} (default {setting_key(DEFAULT)})")

    log(f"test player {args['test_player']}")
    test_sounds = [s for s in sounds if s in {best[0], DEFAULT[0]}]
    test_min = sorted({best[1], DEFAULT[1]})
    test, test_notes, refs = evaluate_split(splits["test"], args["root"], args["cache_dir"],
                                            test_sounds, test_min, workers, log)

    results = {
        "thresholds": {"chosen": {"sound": best[0], "min_frames": best[1]},
                       "default": {"sound": DEFAULT[0], "min_frames": DEFAULT[1]}},
        "selection": "pooled tab_onset F1 on the validation player",
        "sweep_validation": sweep,
        "validation": {"chosen": val[best], "default": val[DEFAULT]},
        "test": {"chosen": test[best], "default": test[DEFAULT]},
        "run_args": args,
    }
    (run_dir / "eval.json").write_text(json.dumps(results, indent=2))
    np.savez(run_dir / "test_notes.npz",
             **{f"{t}__est": n for t, n in test_notes[best].items()},
             **{f"{t}__ref": n for t, n in refs.items()})

    log("\n" + f"{'pooled F1 / playability':<28} note    note+off  tab     tab+off  playable")
    log(_row("val  @ chosen", val[best]))
    log(_row("val  @ default", val[DEFAULT]))
    log(_row("test @ chosen", test[best]))
    log(_row("test @ default", test[DEFAULT]))
    mean = test[best]
    log(f"test @ chosen, mean over clips: note F1 {mean['note_onset']['mean_f1']:.3f}, "
        f"tab F1 {mean['tab_onset']['mean_f1']:.3f}")
    log(f"saved {run_dir / 'eval.json'} and test_notes.npz")
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run_dir", required=True,
                   help="folder with best.pt (and probs_cache/) from train_head_a.py --keep_cache")
    p.add_argument("--workers", type=int, default=None, help="default: CPU count - 1")
    p.add_argument("--sound_thresholds", type=float, nargs="+",
                   default=[s for s in SOUND_THRESHOLDS if s is not None],
                   help="argmax is always included")
    p.add_argument("--min_frames", type=int, nargs="+", default=MIN_FRAMES)
    a = p.parse_args()
    run(a.run_dir, a.workers, a.sound_thresholds, a.min_frames)


if __name__ == "__main__":
    main()
