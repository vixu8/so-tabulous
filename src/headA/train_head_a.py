"""Head A trainer, both stages, on one leave-one-player-out fold (5 train players + 1 test).

    python src/headA/train_head_a.py --stage 1 --test_player 05 --tabcnn_checkpoint checkpoints/tabcnn/5/weights.pt
    python src/headA/train_head_a.py --stage 2 --test_player 05

Model: Kong (TunableKongEncoder) + the fusion adapter -> `features` -> TabCNN's
dense head -> 21 classes per string per frame (src/headA/head_a.py).

There is no validation player (team convention, same as TabCNN): each stage trains
for a fixed number of epochs and keeps the last one; the sounding threshold is
fixed in advance (0.5), never picked on the test player.

Stage 1: Kong frozen (run live on random 10 s crops); the adapter, a fresh input
projection, and both pretrained TabCNN classifier layers train. The checkpoint
must exclude the same test player. Defaults: AdamW lr 1e-3, weight decay 1e-4,
30 epochs.

Stage 2: starts from the fold's stage 1 adapter + head
(<out_dir>/stage1_test{T}/final.pt) and a fresh rank-8 LoRA on every Kong
Linear / GRU weight (src/kong_wrapper/lora.py). Kong runs with dropout on and
BatchNorm frozen; the adapter and head train fully. Defaults: lr 1e-4, no weight
decay, 20 epochs.

Loss (both stages): the tab loss (6 x cross-entropy over the 21 classes, padded
frames ignored). No Kong loss: Head A reads `features`, not Kong's piano outputs.

After training, the final model's class probabilities for the test player are
cached in the run folder (float16), frame-level metrics are computed on them
(sounding threshold 0.5, plus argmax; comparable with TabCNN), and eval_head_a.run
scores them at note level. The cache is deleted unless --keep_cache.

For EC2 / long runs: everything is logged to <run_dir>/train.log; <run_dir>/last.pt
is saved every epoch, and --resume continues from it (a finished run, one with
eval.json, is skipped).

Writes <out_dir>/<tag>_test{T}/ (tag defaults to stage1 / stage2): final.pt
(adapter + head [+ LoRA A/B] + config), last.pt, metrics.json, eval.json,
test_notes.npz, train.log.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
for _sub in (HERE, HERE.parent / "data", HERE.parent / "kong_wrapper", HERE.parent / "metrics"):
    if str(_sub) not in sys.path:
        sys.path.insert(0, str(_sub))
import eval_head_a as ev  # noqa: E402
from head_a import KongHeadA, TabCNNHead, load_tabcnn_head, predict_tab_grid, tab_loss  # noqa: E402
from loader import GuitarSetDataset, collate, load_split  # noqa: E402
from lora import TunableKongEncoder, load_lora_state_dict, lora_parameters, lora_state_dict  # noqa: E402
from tab_metrics import frame_metrics  # noqa: E402

REPO = HERE.parents[1]
DEFAULT_KONG_REPO = HERE.parent / "kong_wrapper" / "third_party" / "piano_transcription"
STAGE_DEFAULTS = {1: {"lr": 1e-3, "weight_decay": 1e-4, "epochs": 30},
                  2: {"lr": 1e-4, "weight_decay": 0.0, "epochs": 20}}
FRAME_SOUNDS = (ev.FIXED[0], ev.ARGMAX[0])  # frame-level metrics: fixed 0.5, plus argmax


# ----- model -----

def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def stage1_dir(args):
    return Path(args.stage1_dir or Path(args.out_dir)
                / f"stage1_test{args.test_player}")


def build_model(args, device):
    """Stage 1: frozen Kong + fresh adapter and head. Stage 2: LoRA'd Kong + stage 1's."""
    torch.manual_seed(args.seed)                   # same adapter / head init every run
    if args.stage == 1 and not args.tabcnn_checkpoint:
        raise ValueError("Stage 1 requires --tabcnn_checkpoint pointing to the matching "
                         "held-out player's TabCNN weights.pt")
    encoder = TunableKongEncoder.from_checkpoint(args.checkpoint, repo_path=args.kong_repo,
                                                 use_adapter=True)             # starts frozen
    model = KongHeadA(encoder)
    if args.stage == 1:
        model.head = load_tabcnn_head(args.tabcnn_checkpoint, encoder.feature_dim)
        model.set_stage(1)
        return model.to(device)
    stage1 = torch.load(stage1_dir(args) / "final.pt", map_location="cpu", weights_only=False)
    model.head = TabCNNHead(encoder.feature_dim, stage1["head"]["dense.0.weight"].shape[1])
    model.encoder.fusion_adapter.load_state_dict(stage1["adapter"])
    model.head.load_state_dict(stage1["head"])
    model.set_stage(2, rank=args.rank, alpha=args.alpha)
    return model.to(device)


def trainable(model):
    """Parameter groups the optimizer updates: adapter + head, plus LoRA A/B once attached."""
    lora = lora_parameters(model.encoder.note_model)
    return ([{"params": lora}] if lora else []) + [
        {"params": [*model.encoder.fusion_adapter.parameters(), *model.head.parameters()]}]


def _cpu_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def snapshot(model):
    return {"lora": lora_state_dict(model.encoder.note_model),
            "adapter": _cpu_state(model.encoder.fusion_adapter),
            "head": _cpu_state(model.head)}


def restore(model, state):
    if state["lora"]:
        load_lora_state_dict(model.encoder.note_model, state["lora"])
    model.encoder.fusion_adapter.load_state_dict(state["adapter"])
    model.head.load_state_dict(state["head"])


# ----- training -----

def loss_fn(logits, batch):
    loss = tab_loss(logits, batch["tab"], batch["n_valid_frames"])
    return loss, {"tab": loss}


def batch_to(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


@torch.no_grad()
def clip_windows(model, row, root, device):
    """Run the model over one clip's 10 s windows -> (batch, logits [W, 1001, 6, 21])."""
    ds = GuitarSetDataset(pd.DataFrame([row._asdict()]), root, mode="windows")
    batch = collate([ds[k] for k in range(len(ds))])
    b = batch_to(batch, device)
    return b, model(b["waveform"])


class Logger:
    def __init__(self, path):
        self.path = Path(path)

    def __call__(self, *parts):
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        with open(self.path, "a") as f:
            f.write(line + "\n")


def run(args, log=None):
    device = pick_device(args.device)
    run_dir = Path(args.out_dir) / f"{args.tag}_test{args.test_player}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log = log or Logger(run_dir / "train.log")
    if args.resume and (run_dir / "eval.json").exists():
        log(f"{run_dir} already finished (eval.json exists); skipping")
        return json.loads((run_dir / "eval.json").read_text())

    np.random.seed(args.seed)
    model = build_model(args, device)
    source = (f"fresh adapter + TabCNN classifier from {args.tabcnn_checkpoint}" if args.stage == 1
              else f"stage 1 adapter + head from {stage1_dir(args)}")
    log(f"stage {args.stage}; device {device}; run {run_dir}; {source}")
    torch.manual_seed(args.seed)
    splits = load_split(args.root, args.test_player)
    train_ds = GuitarSetDataset(splits["train"], args.root, mode="random",
                                crops_per_clip=args.crops_per_clip)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
                        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
                        pin_memory=device.type == "cuda", drop_last=True)
    opt = torch.optim.AdamW(trainable(model), lr=args.lr, weight_decay=args.weight_decay)
    n_lora = sum(p.numel() for p in lora_parameters(model.encoder.note_model))
    log(f"train {len(splits['train'])} clips x {args.crops_per_clip} crops = "
        f"{len(train_ds)} per epoch ({len(loader)} steps); LoRA {n_lora:,} + adapter "
        f"{sum(p.numel() for p in model.encoder.fusion_adapter.parameters()):,} + head "
        f"{sum(p.numel() for p in model.head.parameters()):,} trainable; lr {args.lr}, "
        f"weight decay {args.weight_decay}, {args.epochs} epochs (last one kept)")

    history, start = [], 1
    last_path = run_dir / "last.pt"
    if args.resume and last_path.exists():
        last = torch.load(last_path, map_location="cpu", weights_only=False)
        restore(model, last["state"])
        opt.load_state_dict(last["optimizer"])
        torch.set_rng_state(last["torch_rng"])
        history, start = last["history"], last["epoch"] + 1
        log(f"resumed after epoch {last['epoch']}")

    for epoch in range(start, args.epochs + 1):
        t0 = time.time()
        model.train()
        sums, steps = defaultdict(float), 0
        for batch in loader:
            b = batch_to(batch, device)
            total, parts = loss_fn(model(b["waveform"]), b)
            opt.zero_grad(set_to_none=True)
            total.backward()
            opt.step()
            for k, v in {**parts, "total": total}.items():
                sums[k] += v.item()
            steps += 1
        train_parts = {k: v / max(steps, 1) for k, v in sums.items()}
        history.append({"epoch": epoch, "train_loss": train_parts,
                        "seconds": time.time() - t0})
        log(f"epoch {epoch:3d}/{args.epochs} | train {train_parts.get('total', float('nan')):.4f}"
            f" | {history[-1]['seconds']:.0f} s")
        torch.save({"epoch": epoch, "state": snapshot(model), "optimizer": opt.state_dict(),
                    "torch_rng": torch.get_rng_state(), "history": history}, last_path)
    return finish(args, model, run_dir, history, device, log)


# ----- after training: cache, frame-level metrics, note-level eval -----

@torch.no_grad()
def build_probs_cache(model, rows, root, cache_dir, device, log=print):
    """<cache_dir>/<track>.npz: probs float16 (W, 1001, 6, 21), tab int8 (W, 1001, 6), n_valid."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    t0 = time.time()
    for i, row in enumerate(rows.itertuples(index=False)):
        b, logits = clip_windows(model, row, root, device)
        np.savez(cache_dir / f"{row.track_id}.npz",
                 probs=logits.float().softmax(-1).half().cpu().numpy(),
                 tab=b["tab"].cpu().numpy().astype(np.int8),
                 n_valid=b["n_valid_frames"].cpu().numpy())
        if (i + 1) % 20 == 0 or i + 1 == len(rows):
            log(f"  cached {i + 1}/{len(rows)} clips, {time.time() - t0:.0f} s")


def frame_eval(cache_dir, track_ids, sounds):
    """Frame-level pitch / tab P/R/F (tab_metrics.frame_metrics) for each sounding threshold."""
    probs, tab, n_valid = [], [], []
    for t in track_ids:
        with np.load(Path(cache_dir) / f"{t}.npz") as z:
            probs.append(z["probs"])
            tab.append(z["tab"])
            n_valid.append(z["n_valid"])
    probs = torch.from_numpy(np.concatenate(probs)).float()
    tab, n_valid = torch.from_numpy(np.concatenate(tab)), torch.from_numpy(np.concatenate(n_valid))
    return {sound_name(s): frame_metrics(predict_tab_grid(probs, s), tab, n_valid)
            for s in sounds}


def sound_name(sound):
    return "argmax" if sound is None else str(sound)


def _summary(name, res, key):
    fm = res[key]
    return (f"{name:>10}: frame @ {key}: pitch P/R/F {fm['pitch']['precision']:.3f}/"
            f"{fm['pitch']['recall']:.3f}/{fm['pitch']['f1']:.3f}, tab P/R/F "
            f"{fm['tab']['precision']:.3f}/{fm['tab']['recall']:.3f}/{fm['tab']['f1']:.3f}")


def finish(args, model, run_dir, history, device, log):
    """Cache the final model's probabilities for the test player, then score them."""
    splits = load_split(args.root, args.test_player)
    cache_dir = run_dir / "probs_cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)              # always from this run's final state
    log(f"Caching class probabilities for the test player into {cache_dir} ...")
    build_probs_cache(model, splits["test"], args.root, cache_dir, device, log)
    threshold = sound_name(ev.FIXED[0])
    test = {"frame": frame_eval(cache_dir, splits["test"].track_id, FRAME_SOUNDS)}

    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config["cache_dir"] = str(cache_dir)       # eval_head_a reads probabilities from here
    state = snapshot(model)
    ckpt = {"adapter": state["adapter"], "head": state["head"], "threshold": threshold,
            "epochs": args.epochs, "args": config, "stage": args.stage}
    if args.stage == 2:
        ckpt.update(lora=state["lora"], lora_config={"rank": args.rank, "alpha": args.alpha})
    torch.save(ckpt, run_dir / "final.pt")
    (run_dir / "metrics.json").write_text(json.dumps(
        {"stage": args.stage, "epochs": args.epochs, "threshold": threshold,
         "history": history, "test": test, "args": config}, indent=2))
    log(_summary("test", test["frame"], threshold))
    log(_summary("test", test["frame"], "argmax"))

    results = ev.run(run_dir, args.workers, log=log)
    if not args.keep_cache:
        shutil.rmtree(cache_dir)
    return results


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--stage", type=int, choices=(1, 2), required=True)
    p.add_argument("--tabcnn_checkpoint", help="stage 1: TabCNN weights.pt for the same held-out player")
    p.add_argument("--root", default=str(REPO / "features" / "v1"))
    p.add_argument("--test_player", default="05")
    p.add_argument("--tag", default=None, help="run folder prefix (default stage1 / stage2)")
    p.add_argument("--out_dir", default=str(REPO / "runs" / "head_a"))
    p.add_argument("--stage1_dir", default=None,
                   help="stage 2: folder with stage 1's final.pt (default "
                        "<out_dir>/stage1_test{T})")
    p.add_argument("--checkpoint",
                   default=str(REPO / "src" / "kong_wrapper" / "checkpoints" / "kong_note_pedal.pth"))
    p.add_argument("--kong_repo", default=str(DEFAULT_KONG_REPO))
    p.add_argument("--rank", type=int, default=8, help="stage 2: LoRA rank")
    p.add_argument("--alpha", type=float, default=16.0, help="stage 2: LoRA alpha")
    p.add_argument("--lr", type=float, default=None, help="default: 1e-3 stage 1, 1e-4 stage 2")
    p.add_argument("--weight_decay", type=float, default=None,
                   help="default: 1e-4 stage 1, 0 stage 2")
    p.add_argument("--epochs", type=int, default=None, help="default: 30 stage 1, 20 stage 2")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--crops_per_clip", type=int, default=2, help="random 10 s crops per clip per epoch")
    p.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    p.add_argument("--workers", type=int, default=None,
                   help="note-scoring processes (default: CPU count - 1)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    p.add_argument("--resume", action="store_true", help="continue from <run_dir>/last.pt")
    p.add_argument("--keep_cache", action="store_true", help="keep <run_dir>/probs_cache")
    args = p.parse_args(argv)
    args.test_player = args.test_player.zfill(2)
    for k, v in STAGE_DEFAULTS[args.stage].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    args.tag = args.tag or f"stage{args.stage}"
    if args.workers is None:
        args.workers = max(1, (os.cpu_count() or 2) - 1)
    return args


if __name__ == "__main__":
    run(parse_args())
