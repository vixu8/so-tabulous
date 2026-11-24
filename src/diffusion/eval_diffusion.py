"""Score a finished diffusion fold, comparing one-pass and respaced sampling.

python src/diffusion/eval_diffusion.py --run_dir runs/diffusion/stage1_test05
Thresholds and sampling settings must be fixed before evaluating the test player.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
for sub in (HERE, HERE.parent / "data", HERE.parent / "kong_wrapper", HERE.parent / "metrics"):
    if str(sub) not in sys.path:
        sys.path.insert(0, str(sub))
from loader import GuitarSetDataset, collate, load_split  # noqa: E402
from tab_metrics import aggregate, frame_metrics, note_metrics, playability, tab_grid_to_notes  # noqa: E402
from lora import TunableKongEncoder  # noqa: E402
from kong_tab_diffusion import KongTabDiffusion  # noqa: E402
from train_diffusion import autocast, pick_device, restore  # noqa: E402


def grid_to_notes(tab, onset_probs, n_valid, start_s=0.0, threshold=0.5):
    """Shared half-open note conversion, clipped to the actual audio window.

    The loader counts an endpoint frame (n_valid = samples//160 + 1).
    Frame 1000 belongs to the next window; never draw it twice. Notes crossing
    a boundary are cut there, matching the AR baseline's untied windows.
    """
    length = min(int(n_valid), 1000, len(tab))
    if length <= 0:
        return np.empty((0, 4), dtype=np.float32)
    notes = tab_grid_to_notes(np.asarray(tab)[:length], start_s=start_s,
                              onsets=np.asarray(onset_probs)[:length] >= threshold)
    notes[:, 3] = np.minimum(notes[:, 3], start_s + max(0, int(n_valid) - 1) / 100)
    return notes[notes[:, 3] > notes[:, 2]]


def load_model(path, device, checkpoint=None, kong_repo=None):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    args = saved["args"]
    encoder = TunableKongEncoder.from_checkpoint(checkpoint or args["checkpoint"],
                                                 repo_path=kong_repo or args["kong_repo"],
                                                 use_adapter=args["use_adapter"])
    model = KongTabDiffusion(encoder, saved["timesteps"], saved["schedule"], **saved["config"])
    restore(model, saved["state"])
    return model.to(device).eval(), saved


@torch.no_grad()
def evaluate(model, run_dir, args, device, log=print):
    run_dir = Path(run_dir)
    model.eval()
    rows = load_split(args["root"], args["test_player"])["test"]
    references = {}
    for track, relative in zip(rows.track_id, rows.hdf5_path):
        with h5py.File(Path(args["root"]) / str(relative).replace("\\", "/"), "r") as hf:
            references[str(track)] = hf["tab_notes"][:]
    dataset = GuitarSetDataset(rows, args["root"], mode="windows")
    loader = DataLoader(dataset, batch_size=args.get("decode_batch", 8), collate_fn=collate)
    methods = [("one_pass", 1)] + [("diffusion", s) for s in args.get("inference_steps", [25])]
    results = {"selection": "last epoch; no test-player tuning", "run_args": args,
               "test": {}, "decode_stats": {}, "decodings": {}}
    for method, steps in methods:
        name = "one_pass" if method == "one_pass" else f"diffusion{steps}"
        generator = torch.Generator(device=device).manual_seed(args.get("seed", 0))
        parts = defaultdict(list)
        predicted, targets, lengths = [], [], []
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        for batch in loader:
            with autocast(device, args.get("amp", False)):
                output = model.generate(batch["waveform"].to(device), batch["n_valid_frames"].to(device),
                                        steps=steps, method=method, generator=generator,
                                        temperature=args.get("temperature", 1.0))
            tab, onsets = output["tab"].cpu(), output["onset_probs"].float().cpu()
            predicted.append(tab)
            targets.append(batch["tab"])
            lengths.append(batch["n_valid_frames"])
            for grid, onset, track, start, valid in zip(tab.numpy(), onsets.numpy(),
                    batch["track_id"], batch["start_s"], batch["n_valid_frames"].tolist()):
                parts[track].append(grid_to_notes(grid, onset, valid, start,
                                                  args.get("onset_threshold", 0.5)))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
        estimates = {}
        for track in references:
            notes = np.concatenate(parts[track]) if parts[track] else np.empty((0, 4), dtype=np.float32)
            estimates[track] = notes[np.lexsort((notes[:, 0], notes[:, 2]))]
        scored = aggregate([{**note_metrics(references[t], estimates[t]),
                             "playability": playability(estimates[t])} for t in references])
        scored["frame"] = frame_metrics(torch.cat(predicted), torch.cat(targets), torch.cat(lengths))
        results["test"][name] = scored
        results["decode_stats"][name] = {"seconds": seconds, "windows": len(dataset),
            "seconds_per_window": seconds / max(len(dataset), 1),
            "notes": sum(len(n) for n in estimates.values())}
        results["decodings"][name] = {"method": method, "steps": steps,
            "temperature": args.get("temperature", 1.0), "onset_threshold": args.get("onset_threshold", 0.5)}
        np.savez(run_dir / f"test_notes_{name}.npz",
                 **{f"{t}__est": n for t, n in estimates.items()},
                 **{f"{t}__ref": n for t, n in references.items()})
        log(f"{name}: tab F1 {scored['tab_onset']['pooled']['f1']:.4f}; "
            f"frame tab F1 {scored['frame']['tab']['f1']:.4f}; {seconds:.1f}s")
    (run_dir / "eval.json").write_text(json.dumps(results, indent=2))
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--checkpoint")
    p.add_argument("--kong_repo")
    p.add_argument("--root", help="override the saved data location")
    p.add_argument("--inference_steps", type=int, nargs="+")
    p.add_argument("--decode_batch", type=int)
    a = p.parse_args()
    device = pick_device(a.device)
    model, saved = load_model(Path(a.run_dir) / "final.pt", device, a.checkpoint, a.kong_repo)
    overrides = {k: getattr(a, k) for k in ("root", "inference_steps", "decode_batch") if getattr(a, k) is not None}
    evaluate(model, a.run_dir, {**saved["args"], **overrides}, device)


if __name__ == "__main__":
    main()
