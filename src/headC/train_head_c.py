"""Head C trainer, both stages, on one leave-one-player-out fold (5 train players + 1 test).

    python src/headC/train_head_c.py --stage 1 --test_player 05
    python src/headC/train_head_c.py --stage 2 --test_player 05 [--lam 0.2]

There is no validation player (team convention, same as TabCNN): each stage trains
for a fixed number of epochs and keeps the last one; every threshold is fixed in
advance (Kong's own defaults), never picked on the test player.

Stage 1: Kong frozen (TunableKongEncoder in "frozen" mode, run live on random 10 s
crops), a fresh head trains on the string loss. Defaults: AdamW lr 1e-3, weight
decay 1e-4, 30 epochs.

Stage 2: starts from the fold's stage 1 head (<out_dir>/stage1_test{T}/final.pt)
and a fresh rank-8 LoRA on every Kong Linear / GRU weight (src/kong_wrapper/lora.py).
Kong runs with dropout on and BatchNorm frozen; the head trains fully. Loss = Kong's
own masked BCE on onsets + offsets + frames (pytorch/losses.py `bce`, velocity
dropped: GuitarSet has none) + lam * string loss. Kong's loss keeps its 88 outputs
meaning pitches and teaches it guitar onsets. At the start of stage 2, Kong's loss
is ~0.09 and the string loss ~0.44, so lam = 0.2 (fixed, not swept) roughly balances
them. Defaults: lr 1e-4, no weight decay, 20 epochs.

After training, the final model's Kong outputs for the test player are cached in
the run folder, frame-level metrics are computed on them (Kong frame threshold
0.5), and eval_head_c.run scores them at note level with Kong's default
thresholds. The cache is deleted unless --keep_cache.

For EC2 / long runs: everything is logged to <run_dir>/train.log; <run_dir>/last.pt
is saved every epoch, and --resume continues from it (a finished run, one with
eval.json, is skipped).

Writes <out_dir>/<tag>_test{T}/ (tag defaults to stage1 / stage2): final.pt
(head [+ LoRA A/B] + config), last.pt, metrics.json, eval.json, test_notes.npz,
train.log.
"""

from __future__ import annotations

import argparse
import importlib.util
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
import eval_head_c as ev  # noqa: E402
from head_c import (IGNORE_INDEX, PIANO_KEYS, KongHeadC, as_piano,  # noqa: E402
                    predict_tab_grid, string_loss)
from loader import GuitarSetDataset, collate, load_split  # noqa: E402
from lora import TunableKongEncoder, load_lora_state_dict, lora_parameters, lora_state_dict  # noqa: E402
from tab_metrics import frame_metrics  # noqa: E402

CACHE_VERSION = 1
DEFAULT_THRESHOLD = 0.5                    # Kong frame_output threshold for frame-level metrics
REPO = HERE.parents[1]
KONG_TERMS = {"onset": ("reg_onset_output", "reg_onset_roll"),
              "offset": ("reg_offset_output", "reg_offset_roll"),
              "frame": ("frame_output", "frame_roll")}
TARGETS = {1: ("string_per_pitch",), 2: ("kong", "string_per_pitch")}
STAGE_DEFAULTS = {1: {"lr": 1e-3, "weight_decay": 1e-4, "epochs": 30},
                  2: {"lr": 1e-4, "weight_decay": 0.0, "epochs": 20}}


# ----- Kong output cache (the final model's outputs for the test player, for evaluation) -----

def cache_meta(root, checkpoint):
    checkpoint = Path(checkpoint)
    return {
        "version": CACHE_VERSION,
        "root": str(Path(root).resolve()),
        "checkpoint": checkpoint.name,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "piano_keys": list(PIANO_KEYS),
    }


def build_cache(root, cache_dir, meta, make_encoder, device, log=print, track_ids=None):
    """Cache Kong's piano outputs + targets for every clip (or just `track_ids`) lacking a file.

    make_encoder() is called only if at least one clip still needs Kong.
    Raises if cache_dir was built with a different checkpoint / data root.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "meta.json"
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        if old != meta:
            raise ValueError(f"{cache_dir} was built with {old}, not {meta}; "
                             "delete it or pass another --cache_dir")
    else:
        meta_path.write_text(json.dumps(meta, indent=2))

    manifest = pd.read_csv(Path(root) / "manifest.csv", dtype={"player_id": str})
    if track_ids is not None:
        manifest = manifest[manifest.track_id.isin(set(track_ids))]
    todo = manifest[[not (cache_dir / f"{t}.npz").exists() for t in manifest.track_id]]
    if todo.empty:
        return 0
    log(f"Caching Kong outputs for {len(todo)} clips into {cache_dir} ...")
    encoder = make_encoder().to(device).eval()
    t0 = time.time()
    for i, row in enumerate(todo.itertuples(index=False)):
        ds = GuitarSetDataset(pd.DataFrame([row._asdict()]), root, mode="windows",
                              targets=("string_per_pitch",))
        batch = collate([ds[k] for k in range(len(ds))])
        with torch.no_grad():
            piano = encoder(batch["waveform"].to(device))["piano"]
        arrays = {
            "piano": torch.stack([piano[k] for k in PIANO_KEYS], 1).half().cpu().numpy(),
            "spp": batch["string_per_pitch"].numpy().astype(np.int8),
            "tab": batch["tab"].numpy().astype(np.int8),
            "n_valid": batch["n_valid_frames"].numpy(),
        }
        tmp = cache_dir / f"{row.track_id}.tmp.npz"
        np.savez(tmp, **arrays)
        os.replace(tmp, cache_dir / f"{row.track_id}.npz")  # no half-written files
        if (i + 1) % 20 == 0 or i + 1 == len(todo):
            log(f"  {i + 1}/{len(todo)} clips, {time.time() - t0:.0f} s")
    return len(todo)


def load_windows(cache_dir, track_ids):
    """Concatenate cached windows -> dict of tensors (spp is IGNORE_INDEX past n_valid)."""
    parts = {"piano": [], "spp": [], "tab": [], "n_valid": []}
    for t in track_ids:
        with np.load(Path(cache_dir) / f"{t}.npz") as z:
            for k in parts:
                parts[k].append(z[k])
    data = {k: torch.from_numpy(np.concatenate(v)) for k, v in parts.items()}
    frames = torch.arange(data["spp"].shape[1])
    past_end = frames[None, :] >= data["n_valid"][:, None]
    data["spp"][past_end] = IGNORE_INDEX
    return data


def to_device(data, device):
    return {k: v.to(device) for k, v in data.items()}


@torch.no_grad()
def evaluate(head, data, thresholds=(DEFAULT_THRESHOLD,), batch_size=8):
    """String loss + accuracy on ground-truth pitches, and frame metrics per threshold."""
    head.eval()
    loss_sum, n_rows, n_correct = 0.0, 0, 0
    preds = {t: [] for t in thresholds}
    for a in range(0, len(data["piano"]), batch_size):
        piano = as_piano(data["piano"][a:a + batch_size])
        spp = data["spp"][a:a + batch_size].long()
        logits = head(piano)
        keep = spp != IGNORE_INDEX
        n = int(keep.sum())
        if n:
            loss_sum += string_loss(logits, spp).item() * n
            n_correct += int((logits.argmax(-1)[keep] == spp[keep]).sum())
            n_rows += n
        for t in thresholds:
            preds[t].append(predict_tab_grid(logits, piano["frame_output"], t).to(torch.int8).cpu())
    out = {
        "string_loss": loss_sum / max(n_rows, 1),
        "string_acc": n_correct / max(n_rows, 1),
        "frame": {},
    }
    gt, n_valid = data["tab"].cpu(), data["n_valid"].cpu()
    for t in thresholds:
        out["frame"][t] = frame_metrics(torch.cat(preds[t]), gt, n_valid)
    return out


# ----- losses -----

def kong_bce(repo_path=ev.DEFAULT_KONG_REPO):
    """Kong's masked BCE from pytorch/losses.py (imported, not copied)."""
    path = Path(repo_path) / "pytorch" / "losses.py"
    spec = importlib.util.spec_from_file_location("_kong_losses", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.bce


def _valid_frames(batch, n_frames, device):
    return (torch.arange(n_frames, device=device)[None, :]
            < batch["n_valid_frames"].to(device)[:, None])                    # (B, T)


def stage1_loss(out, batch):
    """(string loss, {string}) for a batch; frames >= n_valid don't count."""
    logits = out["string_logits"]
    valid = _valid_frames(batch, logits.shape[1], logits.device)
    spp = batch["string_per_pitch"].masked_fill(~valid[..., None], IGNORE_INDEX)
    loss = string_loss(logits, spp)
    return loss, {"string": loss}


def stage2_loss(out, batch, lam, bce):
    """(total, {onset, offset, frame, string}) for a batch; frames >= n_valid don't count."""
    logits = out["string_logits"]
    valid = _valid_frames(batch, logits.shape[1], logits.device)
    mask = batch["mask_roll"] * valid[..., None]
    parts = {}
    for name, (output_key, roll_key) in KONG_TERMS.items():
        parts[name] = (bce(out["piano"][output_key], batch[roll_key], mask) if mask.sum() > 0
                       else out["piano"][output_key].sum() * 0.0)
    spp = batch["string_per_pitch"].masked_fill(~valid[..., None], IGNORE_INDEX)
    parts["string"] = string_loss(logits, spp)
    total = parts["onset"] + parts["offset"] + parts["frame"] + lam * parts["string"]
    return total, parts


def make_loss(args):
    if args.stage == 1:
        return stage1_loss
    bce = kong_bce(args.kong_repo)
    return lambda out, batch: stage2_loss(out, batch, args.lam, bce)


# ----- training -----

def batch_to(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Logger:
    def __init__(self, path):
        self.path = Path(path)

    def __call__(self, *parts):
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        with open(self.path, "a") as f:
            f.write(line + "\n")


def stage1_dir(args):
    return Path(args.stage1_dir or Path(args.out_dir)
                / f"stage1_test{args.test_player}")


def build_model(args, device):
    """Stage 1: frozen Kong + fresh head. Stage 2: LoRA'd Kong + the fold's stage 1 head."""
    encoder = TunableKongEncoder.from_checkpoint(args.checkpoint, repo_path=args.kong_repo,
                                                 use_adapter=False)            # starts frozen
    if args.stage == 1:
        torch.manual_seed(args.seed)               # same head init whatever ran before
        return KongHeadC(encoder, hidden=args.hidden, dropout=args.dropout).to(device), None
    stage1 = torch.load(stage1_dir(args) / "final.pt", map_location="cpu", weights_only=False)
    encoder.set_finetune("lora", rank=args.rank, alpha=args.alpha)
    model = KongHeadC(encoder, hidden=stage1["hidden"], dropout=stage1["dropout"])
    model.head.load_state_dict(stage1["head"])
    return model.to(device), stage1


def trainable(model):
    """Parameter groups the optimizer updates: the head, plus LoRA A/B once attached."""
    lora = lora_parameters(model.encoder.note_model)
    return ([{"params": lora}] if lora else []) + [{"params": list(model.head.parameters())}]


def snapshot(model):
    return {"lora": lora_state_dict(model.encoder.note_model),
            "head": {k: v.detach().cpu().clone() for k, v in model.head.state_dict().items()}}


def restore(model, state):
    if state["lora"]:
        load_lora_state_dict(model.encoder.note_model, state["lora"])
    model.head.load_state_dict(state["head"])


def run(args, log=None):
    device = pick_device(args.device)
    run_dir = Path(args.out_dir) / f"{args.tag}_test{args.test_player}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log = log or Logger(run_dir / "train.log")
    if args.resume and (run_dir / "eval.json").exists():
        log(f"{run_dir} already finished (eval.json exists); skipping")
        return json.loads((run_dir / "eval.json").read_text())

    np.random.seed(args.seed)
    model, _ = build_model(args, device)
    source = "fresh head" if args.stage == 1 else f"stage 1 head from {stage1_dir(args)}"
    log(f"stage {args.stage}; device {device}; run {run_dir}; {source}"
        + (f"; lam {args.lam}" if args.stage == 2 else ""))
    torch.manual_seed(args.seed)
    loss_fn = make_loss(args)
    splits = load_split(args.root, args.test_player)
    train_ds = GuitarSetDataset(splits["train"], args.root, mode="random",
                                targets=TARGETS[args.stage], crops_per_clip=args.crops_per_clip,
                                kong_repo=args.kong_repo)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
                        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
                        pin_memory=device.type == "cuda", drop_last=True)
    opt = torch.optim.AdamW(trainable(model), lr=args.lr, weight_decay=args.weight_decay)
    n_lora = sum(p.numel() for p in lora_parameters(model.encoder.note_model))
    log(f"train {len(splits['train'])} clips x {args.crops_per_clip} crops = "
        f"{len(train_ds)} per epoch ({len(loader)} steps); LoRA {n_lora:,} + head "
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
        tl = train_parts
        kong = (f" (kong {tl['onset'] + tl['offset'] + tl['frame']:.4f}, string "
                f"{tl['string']:.4f})" if "onset" in tl else "")
        log(f"epoch {epoch:3d}/{args.epochs} | train {tl.get('total', float('nan')):.4f}{kong} | "
            f"{history[-1]['seconds']:.0f} s")
        torch.save({"epoch": epoch, "state": snapshot(model), "optimizer": opt.state_dict(),
                    "torch_rng": torch.get_rng_state(), "history": history}, last_path)
    return finish(args, model, run_dir, history, device, log)


def _summary(name, res, threshold):
    fm = res["frame"][threshold]
    return (f"{name:>10}: string acc {res['string_acc']:.4f} (loss {res['string_loss']:.4f}) | "
            f"frame @ {threshold}: pitch P/R/F {fm['pitch']['precision']:.3f}/"
            f"{fm['pitch']['recall']:.3f}/{fm['pitch']['f1']:.3f}, tab P/R/F "
            f"{fm['tab']['precision']:.3f}/{fm['tab']['recall']:.3f}/{fm['tab']['f1']:.3f}")


def finish(args, model, run_dir, history, device, log):
    """Cache the final model's Kong outputs for the test player, then score them."""
    splits = load_split(args.root, args.test_player)
    cache_dir = run_dir / "kong_cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)              # always from this run's final state
    meta = {**cache_meta(args.root, args.checkpoint), "run": run_dir.name, "epochs": args.epochs}
    build_cache(args.root, cache_dir, meta, lambda: model.encoder, device, log=log,
                track_ids=list(splits["test"].track_id))
    test = evaluate(model.head, to_device(load_windows(cache_dir, splits["test"].track_id), device),
                    (DEFAULT_THRESHOLD,), args.batch_size)

    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config["cache_dir"] = str(cache_dir)       # eval_head_c reads Kong outputs from here
    state = snapshot(model)
    ckpt = {"head": state["head"], "hidden": model.head.mlp[0].out_features,
            "dropout": model.head.mlp[2].p, "threshold": DEFAULT_THRESHOLD,
            "epochs": args.epochs, "args": config, "cache_meta": meta, "stage": args.stage}
    if args.stage == 2:
        ckpt.update(lora=state["lora"], lora_config={"rank": args.rank, "alpha": args.alpha})
    torch.save(ckpt, run_dir / "final.pt")
    (run_dir / "metrics.json").write_text(json.dumps(
        {"stage": args.stage, "epochs": args.epochs, "threshold": DEFAULT_THRESHOLD,
         "history": history, "test": test, "args": config, "cache_meta": meta}, indent=2))
    log(_summary("test", test, DEFAULT_THRESHOLD))

    results = ev.run(run_dir, args.kong_repo, args.workers, log=log)
    if not args.keep_cache:
        shutil.rmtree(cache_dir)
    return results


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--stage", type=int, choices=(1, 2), required=True)
    p.add_argument("--root", default=str(REPO / "features" / "v1"))
    p.add_argument("--test_player", default="05")
    p.add_argument("--tag", default=None, help="run folder prefix (default stage1 / stage2)")
    p.add_argument("--out_dir", default=str(REPO / "runs" / "head_c"))
    p.add_argument("--stage1_dir", default=None,
                   help="stage 2: folder with stage 1's final.pt (default "
                        "<out_dir>/stage1_test{T})")
    p.add_argument("--checkpoint",
                   default=str(REPO / "src" / "kong_wrapper" / "checkpoints" / "kong_note_pedal.pth"))
    p.add_argument("--kong_repo", default=str(ev.DEFAULT_KONG_REPO))
    p.add_argument("--lam", type=float, default=0.2, help="stage 2: weight of the string loss")
    p.add_argument("--hidden", type=int, default=256, help="stage 1: head hidden size")
    p.add_argument("--dropout", type=float, default=0.1, help="stage 1: head dropout")
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
                   help="post-processing processes (default: CPU count - 1)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    p.add_argument("--resume", action="store_true", help="continue from <run_dir>/last.pt")
    p.add_argument("--keep_cache", action="store_true", help="keep <run_dir>/kong_cache")
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
