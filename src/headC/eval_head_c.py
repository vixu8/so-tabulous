"""Head C note-level evaluation: Kong's post-processor -> notes -> strings -> metrics.

    python src/headC/eval_head_c.py --run_dir runs/head_c/stage1_val04_test05

Reads best.pt from train_head_c.py (its saved args give the data root, the Kong
output cache and the players) and, for every clip of the validation and test
players:
  1. stitches the cached 10 s windows back into the whole clip (1000 frames per
     window, n_valid in the last one; same non-overlapping windows as the decoder),
  2. turns Kong's outputs into notes with Kong's own RegressionPostProcessor
     (onset peaks, offsets, frame activity), on the 44 guitar pitches only,
  3. gives each note a string with head_c.assign_strings,
  4. scores Note F1 / Tab F1 / playability with src/metrics/tab_metrics.py.

Kong's onset peaks are weak on guitar (median 0.22 at true onsets), so the onset
and frame thresholds are swept on the validation player and picked by pooled
onset Tab F1. The test player is scored once with that pair and once with Kong's
own defaults (onset 0.3, offset 0.3, frame 0.1), for reference.

Writes eval.json and test_notes.npz into --run_dir.

Kong's outputs come from the run's own cache (<run_dir>/kong_cache, best.pt's
args.cache_dir), which train_head_c.py builds from the run's best model (frozen
Kong in stage 1, LoRA'd Kong in stage 2) and deletes afterwards unless
--keep_cache. To re-run this script later, train with --keep_cache.
"""

from __future__ import annotations

import argparse
import importlib.util
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
from head_c import HeadC, as_piano, assign_strings  # noqa: E402
from loader import load_split  # noqa: E402
from tab_metrics import aggregate, note_metrics, playability  # noqa: E402

FRAMES_PER_SECOND = 100
FRAMES_PER_WINDOW = 1000            # frame 1000 of a window = frame 0 of the next
GUITAR_PITCHES = slice(19, 63)      # Kong indices of MIDI 40..83
GUITAR_INDEX_OFFSET = 19            # sliced index i is Kong index i + 19
KONG_DEFAULTS = {"onset": 0.3, "offset": 0.3, "frame": 0.1}   # pytorch/inference.py
ONSET_THRESHOLDS = [0.02, 0.05, 0.1, 0.2, 0.3]
FRAME_THRESHOLDS = [0.01, 0.05, 0.1]
DEFAULT_KONG_REPO = HERE.parent / "kong_wrapper" / "third_party" / "piano_transcription"

_UTILITIES = {}


def _kong_utilities(repo_path=DEFAULT_KONG_REPO):
    """Kong's utils/utilities.py as a module (imported, not copied).

    Same import as loader._kong_target_processor; keep in sync. Duplicated so
    loader.py stays untouched; merge into one helper later.
    """
    repo = Path(repo_path).expanduser().resolve()
    if repo in _UTILITIES:
        return _UTILITIES[repo]
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
        spec = importlib.util.spec_from_file_location("_kong_utilities_eval", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        raise ImportError(f"Missing {exc.name!r} for Kong's utilities.py; "
                          "install requirements/model.txt.") from exc
    finally:
        if added:
            sys.path.remove(str(utils_dir))
    _UTILITIES[repo] = module
    return module


# ----- clip inputs -----

def stitch_windows(x, n_valid):
    """(W, T, ...) non-overlapping windows -> one clip (1000 * (W - 1) + n_valid[-1], ...)."""
    x = np.asarray(x) if not torch.is_tensor(x) else x
    parts = [x[i, :FRAMES_PER_WINDOW] for i in range(len(x) - 1)]
    parts.append(x[-1, :int(n_valid[-1])])
    return torch.cat(parts) if torch.is_tensor(x) else np.concatenate(parts)


@torch.no_grad()
def clip_inputs(cache_dir, track_id, head):
    """Cached Kong outputs + head logits for one whole clip.

    Returns piano44 float32 (4, T, 44) [frame, reg_onset, reg_offset, velocity]
    on the guitar pitches, and string logits float32 (T, 88, 6).
    """
    with np.load(Path(cache_dir) / f"{track_id}.npz") as z:
        piano, n_valid = z["piano"], z["n_valid"]                 # (W, 4, 1001, 88) fp16
    head.eval()
    logits = head(as_piano(torch.from_numpy(piano)))              # (W, 1001, 88, 6)
    logits = stitch_windows(logits, n_valid).numpy()
    piano = stitch_windows(piano.transpose(0, 2, 1, 3), n_valid)  # (T, 4, 88)
    piano44 = piano[:, :, GUITAR_PITCHES].transpose(1, 0, 2).astype(np.float32)
    return np.ascontiguousarray(piano44), logits


# ----- notes -----

def detect_notes(piano44, onset_thresholds, frame_thresholds,
                 offset_threshold=KONG_DEFAULTS["offset"], kong_repo=DEFAULT_KONG_REPO):
    """Kong's RegressionPostProcessor on the guitar pitches, for every threshold pair.

    Same steps as RegressionPostProcessor.output_dict_to_note_pedal_arrays, but
    the offset peaks are found once and the onset peaks once per onset
    threshold, then reused for every frame threshold.
    Returns {(onset_t, frame_t): (N, 3) float32 [onset_s, offset_s, midi]}.
    """
    utils = _kong_utilities(kong_repo)
    frame, reg_onset, reg_offset, velocity = (np.asarray(a, dtype=np.float32) for a in piano44)
    pp = utils.RegressionPostProcessor(
        FRAMES_PER_SECOND, classes_num=frame.shape[1], onset_threshold=None,
        offset_threshold=offset_threshold, frame_threshold=None, pedal_offset_threshold=0.2)
    offset_bin, offset_shift = pp.get_binarized_output_from_regression(
        reg_offset, threshold=offset_threshold, neighbour=4)
    out = {}
    for onset_t in onset_thresholds:
        onset_bin, onset_shift = pp.get_binarized_output_from_regression(
            reg_onset, threshold=onset_t, neighbour=2)
        output_dict = {"frame_output": frame, "velocity_output": velocity,
                       "onset_output": onset_bin, "onset_shift_output": onset_shift,
                       "offset_output": offset_bin, "offset_shift_output": offset_shift}
        for frame_t in frame_thresholds:
            if not onset_bin.any():   # Kong's code can't handle zero notes
                notes = np.zeros((0, 3), dtype=np.float32)
            else:
                pp.frame_threshold = frame_t
                notes = pp.output_dict_to_detected_notes(output_dict)[:, :3].copy()
                notes[:, 2] += GUITAR_INDEX_OFFSET
            out[(onset_t, frame_t)] = notes
    return out


def _score_clip(task):
    """Worker: notes for every threshold pair -> tab notes -> metrics. Top level for spawn."""
    track_id, piano44, logits, ref, onset_ts, frame_ts, kong_repo = task
    logits = torch.from_numpy(logits)
    results = {}
    for pair, notes in detect_notes(piano44, onset_ts, frame_ts, kong_repo=kong_repo).items():
        est = assign_strings(notes, logits)
        results[pair] = {"metrics": {**note_metrics(ref, est), "playability": playability(est)},
                         "est": est}
    return track_id, results


def reference_notes(root, rows):
    """{track_id: tab_notes} from the clips' HDF5 files."""
    refs = {}
    for track_id, rel in zip(rows.track_id, rows.hdf5_path):
        with h5py.File(Path(root) / rel, "r") as hf:
            refs[track_id] = hf["tab_notes"][:]
    return refs


def make_pool(workers):
    """A spawn pool for score_clips (None for workers <= 1). Reuse it across calls."""
    return mp.get_context("spawn").Pool(workers) if workers > 1 else None


def score_clips(clips, refs, onset_ts, frame_ts, kong_repo=DEFAULT_KONG_REPO, pool=None,
                log=print):
    """Score whole clips for every threshold pair.

    clips: {track_id: (piano44 (4, T, 44), logits (T, 88, 6))} as numpy float32.
    Returns ({pair: aggregate}, {pair: {track_id: est notes}}).
    """
    tasks = [(t, piano44, logits, refs[t], list(onset_ts), list(frame_ts), str(kong_repo))
             for t, (piano44, logits) in clips.items()]
    t0 = time.time()
    done = pool.map(_score_clip, tasks, chunksize=1) if pool else [_score_clip(t) for t in tasks]
    log(f"  {len(tasks)} clips x {len(onset_ts) * len(frame_ts)} threshold pairs "
        f"in {time.time() - t0:.0f} s")
    pairs = [(o, f) for o in onset_ts for f in frame_ts]
    summary = {p: aggregate([res[p]["metrics"] for _, res in done]) for p in pairs}
    notes = {p: {t: res[p]["est"] for t, res in done} for p in pairs}
    return summary, notes


def evaluate_split(head, rows, root, cache_dir, onset_ts, frame_ts, kong_repo, workers, log=print):
    """Score every clip in `rows` (Kong outputs from `cache_dir`) for every threshold pair.

    Returns ({pair: aggregate}, {pair: {track_id: est notes}}, {track_id: ref notes}).
    """
    refs = reference_notes(root, rows)
    clips = {t: clip_inputs(cache_dir, t, head) for t in rows.track_id}
    pool = make_pool(workers)
    try:
        summary, notes = score_clips(clips, refs, onset_ts, frame_ts, kong_repo, pool, log)
    finally:
        if pool:
            pool.close()
            pool.join()
    return summary, notes, refs


# ----- main -----

def pair_key(pair):
    return f"onset={pair[0]},frame={pair[1]}"


def _row(name, agg):
    g = lambda k: agg[k]["pooled"]["f1"]
    return (f"{name:<28} {g('note_onset'):.3f}   {g('note_offset'):.3f}    {g('tab_onset'):.3f}"
            f"   {g('tab_offset'):.3f}    {agg['playability']['pooled']['rate']:.3f}")


def run(run_dir, kong_repo=DEFAULT_KONG_REPO, workers=None, onset_ts=ONSET_THRESHOLDS,
        frame_ts=FRAME_THRESHOLDS, log=print):
    run_dir = Path(run_dir)
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    args = ckpt["args"]
    head = HeadC(hidden=ckpt["hidden"], dropout=ckpt["dropout"])
    head.load_state_dict(ckpt["head"])
    workers = max(1, (os.cpu_count() or 2) - 1) if workers is None else workers
    splits = load_split(args["root"], args["val_player"], args["test_player"])
    onset_ts = sorted(set(onset_ts) | {KONG_DEFAULTS["onset"]})
    frame_ts = sorted(set(frame_ts) | {KONG_DEFAULTS["frame"]})
    default = (KONG_DEFAULTS["onset"], KONG_DEFAULTS["frame"])

    log(f"validation player {args['val_player']}: sweeping {len(onset_ts)} onset x "
        f"{len(frame_ts)} frame thresholds")
    val, _, _ = evaluate_split(head, splits["validation"], args["root"], args["cache_dir"],
                               onset_ts, frame_ts, kong_repo, workers, log)
    sweep = {pair_key(p): {"note_onset_f1": a["note_onset"]["pooled"]["f1"],
                           "tab_onset_f1": a["tab_onset"]["pooled"]["f1"],
                           "playability": a["playability"]["pooled"]["rate"]}
             for p, a in val.items()}
    best = max(val, key=lambda p: val[p]["tab_onset"]["pooled"]["f1"])
    log("val pooled onset Tab F1 (rows: onset threshold, columns: frame threshold)")
    log("          " + "".join(f"{f:>8}" for f in frame_ts))
    for o in onset_ts:
        log(f"  {o:<8}" + "".join(f"{val[(o, f)]['tab_onset']['pooled']['f1']:8.3f}"
                                   for f in frame_ts))
    log(f"chosen: {pair_key(best)} (Kong default {pair_key(default)})")

    log(f"test player {args['test_player']}")
    test_pairs_o, test_pairs_f = sorted({best[0], default[0]}), sorted({best[1], default[1]})
    test, test_notes, refs = evaluate_split(head, splits["test"], args["root"], args["cache_dir"],
                                            test_pairs_o, test_pairs_f, kong_repo, workers, log)

    results = {
        "thresholds": {"chosen": {"onset": best[0], "frame": best[1],
                                  "offset": KONG_DEFAULTS["offset"]},
                       "kong_default": KONG_DEFAULTS},
        "selection": "pooled tab_onset F1 on the validation player",
        "sweep_validation": sweep,
        "validation": {"chosen": val[best], "kong_default": val[default]},
        "test": {"chosen": test[best], "kong_default": test[default]},
        "run_args": args,
    }
    (run_dir / "eval.json").write_text(json.dumps(results, indent=2))
    np.savez(run_dir / "test_notes.npz",
             **{f"{t}__est": n for t, n in test_notes[best].items()},
             **{f"{t}__ref": n for t, n in refs.items()})

    log("\n" + f"{'pooled F1 / playability':<28} note    note+off  tab     tab+off  playable")
    log(_row(f"val  @ chosen", val[best]))
    log(_row(f"val  @ Kong default", val[default]))
    log(_row(f"test @ chosen", test[best]))
    log(_row(f"test @ Kong default", test[default]))
    mean = test[best]
    log(f"test @ chosen, mean over clips: note F1 {mean['note_onset']['mean_f1']:.3f}, "
        f"tab F1 {mean['tab_onset']['mean_f1']:.3f}")
    log(f"saved {run_dir / 'eval.json'} and test_notes.npz")
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run_dir", required=True, help="folder with best.pt from train_head_c.py")
    p.add_argument("--kong_repo", default=str(DEFAULT_KONG_REPO))
    p.add_argument("--workers", type=int, default=None, help="default: CPU count - 1")
    p.add_argument("--onset_thresholds", type=float, nargs="+", default=ONSET_THRESHOLDS)
    p.add_argument("--frame_thresholds", type=float, nargs="+", default=FRAME_THRESHOLDS)
    a = p.parse_args()
    run(a.run_dir, a.kong_repo, a.workers, a.onset_thresholds, a.frame_thresholds)


if __name__ == "__main__":
    main()
