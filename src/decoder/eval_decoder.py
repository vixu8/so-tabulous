"""Decoder note-level evaluation: greedy decoding -> notes -> metrics, on the test player.

    python src/decoder/eval_decoder.py --run_dir runs/decoder/stage2_test05

Reads final.pt from train_decoder.py (its saved args give the data root, the Kong
checkpoint and the test player), rebuilds the model and, for every clip of the
test player:
  1. runs the model over the clip's non-overlapping 10 s windows (the loader's
     "windows" mode, same windows as the tokenizer), batched across clips,
  2. greedy, grammar-masked decoding (tab_decoder.KongTabDecoder.generate) ->
     tokens -> notes with TabTokenizer.decode; offsets are clipped to the window's
     real audio (n_valid frames),
  3. joins the windows with tokenizer.stitch (a note ringing across a window
     boundary is cut there: the tokenizer has no tie tokens),
  4. scores Note F1 / Tab F1 / playability with src/metrics/tab_metrics.py (the
     same metrics and players as Head A, Head C and TabCNN).

Greedy decoding has no thresholds, so nothing is tuned (there is no validation
player). Writes eval.json (test pooled + mean-over-clips metrics, decoding stats)
and test_notes.npz (`<track>__est` / `<track>__ref`, same keys as the heads, for
error analysis) into --run_dir.
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
from loader import GuitarSetDataset, collate, load_split  # noqa: E402
from lora import TunableKongEncoder, load_lora_state_dict  # noqa: E402
from tab_decoder import KongTabDecoder  # noqa: E402
from tab_metrics import aggregate, note_metrics, playability  # noqa: E402
from tokenizer import TabTokenizer, stitch  # noqa: E402

FRAMES_PER_SECOND = 100


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
    args = ckpt["args"]
    encoder = TunableKongEncoder.from_checkpoint(checkpoint or args["checkpoint"],
                                                 repo_path=kong_repo or args["kong_repo"],
                                                 use_adapter=True)
    model = KongTabDecoder(encoder, TabTokenizer.from_dict(ckpt["tokenizer"]), **ckpt["config"])
    model.encoder.fusion_adapter.load_state_dict(ckpt["adapter"])
    model.decoder.load_state_dict(ckpt["decoder"])
    if ckpt.get("lora"):
        model.set_stage(2, **ckpt["lora_config"])
        load_lora_state_dict(model.encoder.note_model, ckpt["lora"])
    return model.to(device).eval(), ckpt


@torch.no_grad()
def decode_clips(model, rows, root, device, batch_size=16, amp=False, log=print):
    """Greedy-decode every window of every clip in `rows` -> ({track_id: notes (N, 4)}, stats)."""
    model.eval()
    tok = model.tokenizer
    ds = GuitarSetDataset(rows, root, mode="windows")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    parts, stats = defaultdict(list), Counter()
    t0 = time.time()
    for batch in loader:
        n_valid = batch["n_valid_frames"]
        with autocast(device, amp):
            ids = model.generate(batch["waveform"].to(device), n_valid.to(device))
        for row, track, start_s, nv in zip(ids.cpu().numpy(), batch["track_id"],
                                           batch["start_s"], n_valid.tolist()):
            notes, st = tok.decode(row, start_s, return_stats=True)
            notes[:, 3] = np.minimum(notes[:, 3], start_s + (nv - 1) / FRAMES_PER_SECOND)
            parts[track].append(notes)
            stats["windows"] += 1
            stats["notes"] += st["n_notes"]
            stats["malformed_groups"] += st["malformed_groups"]
            stats["hit_max_len"] += int(not st["ended_with_eos"])
            stats["tokens"] += int((row != 0).sum())
        log(f"  decoded {stats['windows']}/{len(ds)} windows, {time.time() - t0:.0f} s")
    return {t: stitch(p) for t, p in parts.items()}, dict(stats)


def score(est, refs):
    """Per-clip note metrics + playability -> tab_metrics.aggregate (pooled + mean over clips)."""
    return aggregate([{**note_metrics(refs[t], est[t]), "playability": playability(est[t])}
                      for t in refs])


def _row(name, agg):
    g = lambda k: agg[k]["pooled"]["f1"]  # noqa: E731
    return (f"{name:<28} {g('note_onset'):.3f}   {g('note_offset'):.3f}    {g('tab_onset'):.3f}"
            f"   {g('tab_offset'):.3f}    {agg['playability']['pooled']['rate']:.3f}")


def evaluate(model, run_dir, args, device, batch_size=16, amp=False, log=print):
    """Decode + score the test player; write eval.json and test_notes.npz into run_dir."""
    run_dir = Path(run_dir)
    rows = load_split(args["root"], args["test_player"])["test"]
    refs = reference_notes(args["root"], rows)
    log(f"test player {args['test_player']}: greedy decoding {len(rows)} clips")
    est, stats = decode_clips(model, rows, args["root"], device, batch_size, amp, log)
    test = score(est, refs)
    results = {"decoding": "greedy, grammar-masked (no thresholds)",
               "selection": "last epoch (no validation player)",
               "decode_stats": stats, "test": {"greedy": test}, "run_args": args}
    (run_dir / "eval.json").write_text(json.dumps(results, indent=2))
    np.savez(run_dir / "test_notes.npz",
             **{f"{t}__est": n for t, n in est.items()},
             **{f"{t}__ref": n for t, n in refs.items()})

    log("\n" + f"{'pooled F1 / playability':<28} note    note+off  tab     tab+off  playable")
    log(_row("test @ greedy", test))
    log(f"test, mean over clips: note F1 {test['note_onset']['mean_f1']:.3f}, "
        f"tab F1 {test['tab_onset']['mean_f1']:.3f}")
    log(f"decoding: {stats['notes']} notes in {stats['windows']} windows, "
        f"{stats['hit_max_len']} hit max_len, {stats['malformed_groups']} malformed groups")
    log(f"saved {run_dir / 'eval.json'} and test_notes.npz")
    return results


def run(run_dir, device="auto", batch_size=16, amp=False, checkpoint=None, kong_repo=None,
        log=print):
    device = pick_device(device)
    model, ckpt = load_model(Path(run_dir) / "final.pt", device, checkpoint, kong_repo)
    return evaluate(model, run_dir, ckpt["args"], device, batch_size, amp, log)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run_dir", required=True, help="folder with final.pt from train_decoder.py")
    p.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    p.add_argument("--decode_batch", type=int, default=16, help="windows decoded at once")
    p.add_argument("--amp", action="store_true", help="bf16 autocast (CUDA only)")
    p.add_argument("--checkpoint", default=None, help="Kong checkpoint (default: final.pt's)")
    p.add_argument("--kong_repo", default=None, help="Kong repo (default: final.pt's)")
    a = p.parse_args()
    run(a.run_dir, a.device, a.decode_batch, a.amp, a.checkpoint, a.kong_repo)


if __name__ == "__main__":
    main()
