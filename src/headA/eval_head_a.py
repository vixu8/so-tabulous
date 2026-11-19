"""Head A note-level evaluation: per-frame fret classes -> notes -> metrics.

    python src/headA/eval_head_a.py --run_dir runs/head_a/stage1_test05

Reads final.pt from train_head_a.py (its saved args give the data root, the
probability cache and the test player) and, for every clip of the test player:
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

There is no validation player (team convention, same as TabCNN), so the setting
is fixed in advance: a string sounds when 1 - p(silent) >= 0.5, min_frames 1
(`fixed`, the headline number). The plain argmax at min_frames 1 (`argmax`,
Head A's own predict) is scored too, for reference only: after little training
argmax is almost all silence.

Writes eval.json and test_notes.npz into --run_dir. The cache
(<run_dir>/probs_cache) is built by train_head_a.py from the run's final model and
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
FIXED = (0.5, 1)                    # (sounding threshold, min_frames): the headline setting
ARGMAX = (None, 1)                  # KongHeadA.predict, every run kept (reference)
SETTINGS = {"fixed": FIXED, "argmax": ARGMAX}


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


def score_clips(clips, refs, sounds=(FIXED[0], ARGMAX[0]), min_frames=(1,), pool=None,
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


def run(run_dir, workers=None, log=print):
    run_dir = Path(run_dir)
    args = torch.load(run_dir / "final.pt", map_location="cpu", weights_only=False)["args"]
    workers = max(1, (os.cpu_count() or 2) - 1) if workers is None else workers
    splits = load_split(args["root"], args["test_player"])
    sounds = sorted({v[0] for v in SETTINGS.values()}, key=lambda x: -1 if x is None else x)
    min_frames = sorted({v[1] for v in SETTINGS.values()})

    log(f"test player {args['test_player']}: fixed {setting_key(FIXED)}, "
        f"reference {setting_key(ARGMAX)}")
    test, test_notes, refs = evaluate_split(splits["test"], args["root"], args["cache_dir"],
                                            sounds, min_frames, workers, log)
    results = {
        "thresholds": {name: {"sound": v[0], "min_frames": v[1]} for name, v in SETTINGS.items()},
        "selection": "fixed in advance (no validation player)",
        "test": {name: test[v] for name, v in SETTINGS.items()},
        "run_args": args,
    }
    (run_dir / "eval.json").write_text(json.dumps(results, indent=2))
    np.savez(run_dir / "test_notes.npz",
             **{f"{t}__est": n for t, n in test_notes[FIXED].items()},
             **{f"{t}__ref": n for t, n in refs.items()})

    log("\n" + f"{'pooled F1 / playability':<28} note    note+off  tab     tab+off  playable")
    log(_row("test @ fixed", test[FIXED]))
    log(_row("test @ argmax", test[ARGMAX]))
    log(f"test @ fixed, mean over clips: note F1 {test[FIXED]['note_onset']['mean_f1']:.3f}, "
        f"tab F1 {test[FIXED]['tab_onset']['mean_f1']:.3f}")
    log(f"saved {run_dir / 'eval.json'} and test_notes.npz")
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run_dir", required=True,
                   help="folder with final.pt (and probs_cache/) from train_head_a.py --keep_cache")
    p.add_argument("--workers", type=int, default=None, help="default: CPU count - 1")
    a = p.parse_args()
    run(a.run_dir, a.workers)


if __name__ == "__main__":
    main()
