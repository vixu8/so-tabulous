"""Decoder evaluation: greedy and/or beam-search decoding -> notes -> metrics.

    python src/decoder/eval_decoder.py --run_dir runs/decoder1/stage2_test05
    python src/decoder/eval_decoder.py --run_dir runs/decoder2/stage2_zeroshot --decoding beam

Reads final.pt from train_decoder.py (its saved args give the data root, the Kong
checkpoint, the data source and the test player), rebuilds the model (decoder1 or
decoder2: the architecture is in final.pt's config) and scores the test clips:
  - decoder1 (--data guitarset): the fold's test player,
  - decoder2 (--data synthtab): every GuitarSet clip, zero-shot.
For every test clip and every requested decoding:
  1. runs the model over the clip's non-overlapping 10 s windows (the loader's
     "windows" mode, same windows as the tokenizer), batched across clips,
  2. decodes under the grammar mask (tab_decoder.KongTabDecoder.generate): greedy,
     or beam search (size 4 by default, length penalty 1.0); optional
     --max_chord_span (TART: 5) -> tokens -> notes with TabTokenizer.decode;
     offsets are clipped to the window's real audio (n_valid frames),
  3. joins the windows with tokenizer.stitch (a note ringing across a window
     boundary is cut there: the tokenizer has no tie tokens),
  4. scores with src/metrics/tab_metrics.py:
       - frame level (Table 2, same as TabCNN / Head A / Head C): the decoded notes
         are drawn on each window's 1001-frame tab grid (loader.notes_to_tab_grid,
         the same grid the loader builds the reference `tab` with) -> frame_metrics:
         pitch P/R/F, tab P/R/F, TDR, pooled over windows,
       - note level: Note F1 / Tab F1 (onset +-50 ms, pitch +-50 cents, string and
         fret; TART's note-event Tab F1) with and without offsets, and playability.

Decoding has no thresholds, so nothing is tuned (there is no validation player).
Writes eval.json (per decoding: settings, decoding stats, test pooled +
mean-over-clips metrics) and test_notes_<decoding>.npz (`<track>__est` /
`<track>__ref`, for error analysis) into --run_dir. Decoding names: "greedy",
"beam<size>".
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
for _sub in (HERE, HERE.parent / "data", HERE.parent / "kong_wrapper", HERE.parent / "metrics"):
    if str(_sub) not in sys.path:
        sys.path.insert(0, str(_sub))
from loader import GuitarSetDataset, collate, load_manifest, load_split, notes_to_tab_grid  # noqa: E402
from lora import TunableKongEncoder, load_lora_state_dict  # noqa: E402
from tab_decoder import KongTabDecoder  # noqa: E402
from tab_metrics import aggregate, frame_metrics, note_metrics, playability  # noqa: E402
from tokenizer import TabTokenizer, stitch  # noqa: E402

FRAMES_PER_SECOND = 100
DECODING_DEFAULTS = {"decoding": ["greedy", "beam"], "beam_size": 4, "length_penalty": 1.0,
                     "max_chord_span": None}


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast(device, amp):
    """bf16 autocast on CUDA when --amp; a no-op otherwise."""
    if amp and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def reference_notes(root, rows):
    """{track_id: tab_notes} from the clips' HDF5 files. Same as eval_head_a; keep in sync."""
    refs = {}
    for track_id, rel in zip(rows.track_id, rows.hdf5_path):
        with h5py.File(Path(root) / rel, "r") as hf:
            refs[track_id] = hf["tab_notes"][:]
    return refs


def load_model(final_path, device, checkpoint=None, kong_repo=None):
    """Rebuild KongTabDecoder from train_decoder.py's final.pt (frozen Kong + LoRA if stage 2)."""
    ckpt = torch.load(final_path, map_location="cpu", weights_only=False)
    if "adapter" in ckpt:
        raise ValueError(f"{final_path} was trained with the fusion adapter, which the decoder "
                         "no longer uses; retrain it")
    args = ckpt["args"]
    encoder = TunableKongEncoder.from_checkpoint(checkpoint or args["checkpoint"],
                                                 repo_path=kong_repo or args["kong_repo"],
                                                 use_adapter=False)
    model = KongTabDecoder(encoder, TabTokenizer.from_dict(ckpt["tokenizer"]), **ckpt["config"])
    model.decoder.load_state_dict(ckpt["decoder"])
    if ckpt.get("lora"):
        model.set_stage(2, **ckpt["lora_config"])
        load_lora_state_dict(model.encoder.note_model, ckpt["lora"])
    return model.to(device).eval(), ckpt


def decoding_name(method, beam_size=4):
    return "greedy" if method == "greedy" else f"beam{beam_size}"


def test_rows(args):
    """Test clips from a run's args: the fold's test player, or all of GuitarSet
    (zero-shot) for a SynthTab-trained run."""
    if args.get("data", "guitarset") == "synthtab":
        return load_manifest(args["root"])
    return load_split(args["root"], args["test_player"])["test"]


@torch.no_grad()
def decode_clips(model, rows, root, device, batch_size=16, amp=False, log=print,
                 method="greedy", beam_size=4, length_penalty=1.0, max_chord_span=None):
    """Decode every window of every clip in `rows` -> ({track_id: notes (N, 4)}, stats,
    windows [(track_id, start_s, n_valid)])."""
    model.eval()
    tok = model.tokenizer
    ds = GuitarSetDataset(rows, root, mode="windows")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    parts, stats, windows = defaultdict(list), Counter(), []
    t0 = time.time()
    for batch in loader:
        n_valid = batch["n_valid_frames"]
        with autocast(device, amp):
            ids = model.generate(batch["waveform"].to(device), n_valid.to(device),
                                 method=method, beam_size=beam_size,
                                 length_penalty=length_penalty, max_chord_span=max_chord_span)
        for row, track, start_s, nv in zip(ids.cpu().numpy(), batch["track_id"],
                                           batch["start_s"], n_valid.tolist()):
            notes, st = tok.decode(row, start_s, return_stats=True)
            notes[:, 3] = np.minimum(notes[:, 3], start_s + (nv - 1) / FRAMES_PER_SECOND)
            parts[track].append(notes)
            windows.append((track, start_s, nv))
            stats["windows"] += 1
            stats["notes"] += st["n_notes"]
            stats["malformed_groups"] += st["malformed_groups"]
            stats["hit_max_len"] += int(not st["ended_with_eos"])
            stats["tokens"] += int((row != 0).sum())
        log(f"  {decoding_name(method, beam_size)}: decoded {stats['windows']}/{len(ds)} "
            f"windows, {time.time() - t0:.0f} s")
    stats["seconds"] = round(time.time() - t0, 1)
    return {t: stitch(p) for t, p in parts.items()}, dict(stats), windows


def window_frame_metrics(est, refs, windows):
    """Frame pitch / tab P/R/F + TDR, pooled over every test window (Table 2 metrics).

    Both sides are drawn on the window's 1001-frame grid with notes_to_tab_grid, the
    rule the loader uses for its `tab` target (what Head A / TabCNN are scored on).
    """
    pred = np.stack([notes_to_tab_grid(est[t], s)[0] for t, s, _ in windows])
    gt = np.stack([notes_to_tab_grid(refs[t], s)[0] for t, s, _ in windows])
    return frame_metrics(pred, gt, [nv for _, _, nv in windows])


def score(est, refs, windows=None):
    """Per-clip note metrics + playability -> tab_metrics.aggregate (pooled + mean over
    clips); plus pooled frame metrics when `windows` (from decode_clips) is given."""
    out = aggregate([{**note_metrics(refs[t], est[t]), "playability": playability(est[t])}
                     for t in refs])
    if windows is not None:
        out["frame"] = window_frame_metrics(est, refs, windows)
    return out


def _row(name, agg):
    g = lambda k: agg[k]["pooled"]["f1"]  # noqa: E731
    line = (f"{name:<22} {g('note_onset'):.3f}   {g('note_offset'):.3f}    {g('tab_onset'):.3f}"
            f"   {g('tab_offset'):.3f}    {agg['playability']['pooled']['rate']:.3f}")
    if "frame" in agg:
        fr = agg["frame"]
        line += f"     {fr['pitch']['f1']:.3f}    {fr['tab']['f1']:.3f}    {fr['tdr']:.3f}"
    return line


def decoding_settings(args):
    """Decoding options from a run's args, defaults for runs saved before they existed."""
    return {k: v if args.get(k) is None else args[k] for k, v in DECODING_DEFAULTS.items()}


def evaluate(model, run_dir, args, device, batch_size=16, amp=False, log=print):
    """Decode + score the test clips with each requested decoding; write eval.json and
    test_notes_<decoding>.npz into run_dir."""
    run_dir = Path(run_dir)
    opts = decoding_settings(args)
    rows = test_rows(args)
    refs = reference_notes(args["root"], rows)
    what = ("all GuitarSet clips, zero-shot" if args.get("data") == "synthtab"
            else f"test player {args['test_player']}")
    log(f"{what}: {len(rows)} clips; decoding {', '.join(opts['decoding'])} (beam size "
        f"{opts['beam_size']}, length penalty {opts['length_penalty']}, max chord span "
        f"{opts['max_chord_span']})")
    results = {"selection": "last epoch (no validation player)", "decodings": {},
               "decode_stats": {}, "test": {}, "run_args": args}
    for method in opts["decoding"]:
        name = decoding_name(method, opts["beam_size"])
        est, stats, windows = decode_clips(model, rows, args["root"], device, batch_size, amp,
                                           log, method, opts["beam_size"],
                                           opts["length_penalty"], opts["max_chord_span"])
        results["decodings"][name] = {
            "method": method, "grammar_masked": True, "max_chord_span": opts["max_chord_span"],
            **({"beam_size": opts["beam_size"], "length_penalty": opts["length_penalty"]}
               if method == "beam" else {})}
        results["decode_stats"][name] = stats
        results["test"][name] = score(est, refs, windows)
        np.savez(run_dir / f"test_notes_{name}.npz",
                 **{f"{t}__est": n for t, n in est.items()},
                 **{f"{t}__ref": n for t, n in refs.items()})
    (run_dir / "eval.json").write_text(json.dumps(results, indent=2))

    log("\n" + f"{'pooled F1 / playability':<22} note    note+off  tab     tab+off  playable"
        "  frame: pitch    tab      TDR")
    for name, test in results["test"].items():
        log(_row(f"test @ {name}", test))
    for name, test in results["test"].items():
        st = results["decode_stats"][name]
        log(f"{name}: mean over clips note F1 {test['note_onset']['mean_f1']:.3f}, tab F1 "
            f"{test['tab_onset']['mean_f1']:.3f}; {st['notes']} notes in {st['windows']} "
            f"windows, {st['hit_max_len']} hit max_len, {st['malformed_groups']} malformed "
            f"groups, {st['seconds']:.0f} s")
    log(f"saved {run_dir / 'eval.json'} and test_notes_<decoding>.npz")
    return results


def run(run_dir, device="auto", batch_size=16, amp=False, checkpoint=None, kong_repo=None,
        log=print, **decoding):
    """Standalone eval of a finished run; `decoding` (decoding, beam_size, length_penalty,
    max_chord_span) overrides the run's saved settings when not None."""
    device = pick_device(device)
    model, ckpt = load_model(Path(run_dir) / "final.pt", device, checkpoint, kong_repo)
    args = {**ckpt["args"], **{k: v for k, v in decoding.items() if v is not None}}
    return evaluate(model, run_dir, args, device, batch_size, amp, log)


def add_decoding_args(p, defaults=True):
    """--decoding / --beam_size / --length_penalty / --max_chord_span. With
    defaults=False (standalone eval) they default to None = the run's saved setting."""
    d = DECODING_DEFAULTS if defaults else dict.fromkeys(DECODING_DEFAULTS)
    p.add_argument("--decoding", nargs="+", choices=("greedy", "beam"), default=d["decoding"],
                   help="decodings to score (default: greedy and beam)")
    p.add_argument("--beam_size", type=int, default=d["beam_size"], help="beam width (TART: 4)")
    p.add_argument("--length_penalty", type=float, default=d["length_penalty"],
                   help="beam: hypotheses ranked by sum(log p) / length ** this")
    p.add_argument("--max_chord_span", type=int, default=None,
                   help="mask chords whose fretted notes span more than this many frets "
                        "(TART: 5); default off")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run_dir", required=True, help="folder with final.pt from train_decoder.py")
    p.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    p.add_argument("--decode_batch", type=int, default=16, help="windows decoded at once")
    p.add_argument("--amp", action="store_true", help="bf16 autocast (CUDA only)")
    p.add_argument("--checkpoint", default=None, help="Kong checkpoint (default: final.pt's)")
    p.add_argument("--kong_repo", default=None, help="Kong repo (default: final.pt's)")
    add_decoding_args(p, defaults=False)
    a = p.parse_args()
    run(a.run_dir, a.device, a.decode_batch, a.amp, a.checkpoint, a.kong_repo,
        decoding=a.decoding, beam_size=a.beam_size, length_penalty=a.length_penalty,
        max_chord_span=a.max_chord_span)


if __name__ == "__main__":
    main()
