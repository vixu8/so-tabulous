"""Head C stage 2: LoRA on Kong + the head, trained together on one leave-one-player-out fold.

    python src/headC/train_head_c_stage2.py --val_player 04 --test_player 05 [--lam 0.2]

Starts from the fold's stage 1 head (runs/head_c/stage1_val{V}_test{T}/best.pt) and a
fresh rank-8 LoRA on every Kong Linear / GRU weight (src/kong_wrapper/lora.py). Kong
runs live (dropout on, BatchNorm frozen); the head trains fully.

Loss = Kong's own masked BCE on onsets + offsets + frames (pytorch/losses.py `bce`,
velocity dropped: GuitarSet has none) + lam * Head C's string loss. Kong's loss keeps
its 88 outputs meaning pitches and teaches it guitar onsets. At the start of stage 2,
Kong's loss is ~0.09 and the string loss ~0.44, so lam = 0.2 roughly balances them.

Each epoch is scored on the validation player with the full note-level pipeline
(Kong's post-processor -> strings -> Tab F1, best over eval_head_c's threshold grid);
the best epoch by validation Tab F1 is kept (early stopping), and the validation loss
is logged alongside. Epoch 0 is the starting point (= stage 1 with Kong live).

After training, the best model's Kong outputs for the validation and test players are
cached in the run folder, and the same evaluation as stage 1 runs on them
(train_head_c.evaluate for frame level, eval_head_c.run for note level), so
summarize_head_c.py --stage <tag> works unchanged.

For AWS / long runs: everything is logged to <run_dir>/train.log; <run_dir>/last.pt is
saved every epoch, and --resume continues from it (e.g. after a spot interruption).

Writes <out_dir>/<tag>_val{V}_test{T}/: best.pt (head + LoRA A/B + config), last.pt,
metrics.json, eval.json, test_notes.npz, train.log.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
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
from head_c import IGNORE_INDEX, PIANO_KEYS, KongHeadC, string_loss  # noqa: E402
from loader import GuitarSetDataset, collate, load_split  # noqa: E402
from lora import TunableKongEncoder, load_lora_state_dict, lora_parameters, lora_state_dict  # noqa: E402
from train_head_c import (THRESHOLDS, _summary, build_cache, cache_meta, evaluate,  # noqa: E402
                          load_windows, pick_device, to_device)

REPO = HERE.parents[1]
KONG_TERMS = {"onset": ("reg_onset_output", "reg_onset_roll"),
              "offset": ("reg_offset_output", "reg_offset_roll"),
              "frame": ("frame_output", "frame_roll")}
TARGETS = ("kong", "string_per_pitch")


def kong_bce(repo_path=ev.DEFAULT_KONG_REPO):
    """Kong's masked BCE from pytorch/losses.py (imported, not copied)."""
    path = Path(repo_path) / "pytorch" / "losses.py"
    spec = importlib.util.spec_from_file_location("_kong_losses", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.bce


def stage2_loss(out, batch, lam, bce):
    """(total, {onset, offset, frame, string}) for a batch; frames >= n_valid don't count."""
    logits = out["string_logits"]
    n_frames = logits.shape[1]
    valid = (torch.arange(n_frames, device=logits.device)[None, :]
             < batch["n_valid_frames"].to(logits.device)[:, None])          # (B, T)
    mask = batch["mask_roll"] * valid[..., None]
    parts = {}
    for name, (output_key, roll_key) in KONG_TERMS.items():
        parts[name] = (bce(out["piano"][output_key], batch[roll_key], mask) if mask.sum() > 0
                       else out["piano"][output_key].sum() * 0.0)
    spp = batch["string_per_pitch"].masked_fill(~valid[..., None], IGNORE_INDEX)
    parts["string"] = string_loss(logits, spp)
    total = parts["onset"] + parts["offset"] + parts["frame"] + lam * parts["string"]
    return total, parts


def batch_to(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


@torch.no_grad()
def validate(model, rows, root, refs, device, lam, bce, pool, kong_repo,
             onset_ts=ev.ONSET_THRESHOLDS, frame_ts=ev.FRAME_THRESHOLDS):
    """Validation loss + note-level Tab F1 (best threshold pair) with Kong running live."""
    model.eval()
    sums, n_windows, clips = defaultdict(float), 0, {}
    for row in rows.itertuples(index=False):
        ds = GuitarSetDataset(pd.DataFrame([row._asdict()]), root, mode="windows",
                              targets=TARGETS)
        batch = collate([ds[k] for k in range(len(ds))])
        b = batch_to(batch, device)
        out = model(b["waveform"])
        total, parts = stage2_loss(out, b, lam, bce)
        for k, v in {**parts, "total": total}.items():
            sums[k] += v.item() * len(ds)
        n_windows += len(ds)
        n_valid = batch["n_valid_frames"].numpy()
        piano = torch.stack([out["piano"][k] for k in PIANO_KEYS], 1).float().cpu().numpy()
        piano = ev.stitch_windows(piano.transpose(0, 2, 1, 3), n_valid)          # (T, 4, 88)
        piano44 = np.ascontiguousarray(piano[:, :, ev.GUITAR_PITCHES].transpose(1, 0, 2))
        logits = ev.stitch_windows(out["string_logits"].float().cpu(), n_valid).numpy()
        clips[row.track_id] = (piano44, logits)
    summary, _ = ev.score_clips(clips, refs, onset_ts, frame_ts, kong_repo, pool,
                                log=lambda *_: None)
    best = max(summary, key=lambda p: summary[p]["tab_onset"]["pooled"]["f1"])
    return {"loss": {k: v / n_windows for k, v in sums.items()},
            "tab_f1": summary[best]["tab_onset"]["pooled"]["f1"],
            "note_f1": summary[best]["note_onset"]["pooled"]["f1"],
            "playability": summary[best]["playability"]["pooled"]["rate"],
            "thresholds": list(best)}


class Logger:
    def __init__(self, path):
        self.path = Path(path)

    def __call__(self, *parts):
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        with open(self.path, "a") as f:
            f.write(line + "\n")


def build_model(args, stage1, device):
    encoder = TunableKongEncoder.from_checkpoint(args.checkpoint, repo_path=args.kong_repo,
                                                 use_adapter=False)
    encoder.set_finetune("lora", rank=args.rank, alpha=args.alpha)
    model = KongHeadC(encoder, hidden=stage1["hidden"], dropout=stage1["dropout"])
    model.head.load_state_dict(stage1["head"])
    return model.to(device)


def snapshot(model):
    return {"lora": lora_state_dict(model.encoder.note_model),
            "head": {k: v.detach().cpu().clone() for k, v in model.head.state_dict().items()}}


def restore(model, state):
    load_lora_state_dict(model.encoder.note_model, state["lora"])
    model.head.load_state_dict(state["head"])


def run(args, log=None):
    device = pick_device(args.device)
    run_dir = Path(args.out_dir) / f"{args.tag}_val{args.val_player}_test{args.test_player}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log = log or Logger(run_dir / "train.log")
    if args.resume and (run_dir / "eval.json").exists():
        log(f"{run_dir} already finished (eval.json exists); skipping")
        return json.loads((run_dir / "eval.json").read_text())
    stage1_dir = Path(args.stage1_dir or Path(args.out_dir)
                      / f"stage1_val{args.val_player}_test{args.test_player}")
    stage1 = torch.load(stage1_dir / "best.pt", map_location="cpu", weights_only=False)
    log(f"device {device}; run {run_dir}; stage 1 head from {stage1_dir}; lam {args.lam}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = build_model(args, stage1, device)
    bce = kong_bce(args.kong_repo)
    splits = load_split(args.root, args.val_player, args.test_player)
    val_refs = ev.reference_notes(args.root, splits["validation"])
    train_ds = GuitarSetDataset(splits["train"], args.root, mode="random", targets=TARGETS,
                                crops_per_clip=args.crops_per_clip, kong_repo=args.kong_repo)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
                        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
                        pin_memory=device.type == "cuda", drop_last=True)
    opt = torch.optim.AdamW([{"params": lora_parameters(model.encoder.note_model)},
                             {"params": model.head.parameters()}],
                            lr=args.lr, weight_decay=args.weight_decay)
    log(f"train {len(splits['train'])} clips x {args.crops_per_clip} crops = "
        f"{len(train_ds)} per epoch ({len(loader)} steps); LoRA "
        f"{sum(p.numel() for p in lora_parameters(model.encoder.note_model)):,} + head "
        f"{sum(p.numel() for p in model.head.parameters()):,} trainable")

    history, best_f1, best_epoch, bad, start = [], -1.0, 0, 0, 0
    best_state = None
    last_path = run_dir / "last.pt"
    if args.resume and last_path.exists():
        last = torch.load(last_path, map_location="cpu", weights_only=False)
        restore(model, last["state"])
        opt.load_state_dict(last["optimizer"])
        torch.set_rng_state(last["torch_rng"])
        history, best_f1, best_epoch, bad = (last["history"], last["best_f1"],
                                             last["best_epoch"], last["bad"])
        best_state, start = last["best_state"], last["epoch"] + 1
        log(f"resumed after epoch {last['epoch']} (best {best_epoch}: Tab F1 {best_f1:.4f})")

    pool = ev.make_pool(args.workers)
    try:
        for epoch in range(start, args.epochs + 1):
            if epoch > 0 and bad >= args.patience:
                break
            t0 = time.time()
            train_parts = {}
            if epoch > 0:                      # epoch 0 = starting point, no training
                model.train()
                sums, steps = defaultdict(float), 0
                for batch in loader:
                    b = batch_to(batch, device)
                    total, parts = stage2_loss(model(b["waveform"]), b, args.lam, bce)
                    opt.zero_grad(set_to_none=True)
                    total.backward()
                    opt.step()
                    for k, v in {**parts, "total": total}.items():
                        sums[k] += v.item()
                    steps += 1
                train_parts = {k: v / steps for k, v in sums.items()}
            t_train = time.time() - t0
            val = validate(model, splits["validation"], args.root, val_refs, device, args.lam,
                           bce, pool, args.kong_repo)
            entry = {"epoch": epoch, "train_loss": train_parts, "val_loss": val["loss"],
                     "val_tab_f1": val["tab_f1"], "val_note_f1": val["note_f1"],
                     "val_playability": val["playability"], "val_thresholds": val["thresholds"],
                     "seconds_train": t_train, "seconds_total": time.time() - t0}
            history.append(entry)
            if val["tab_f1"] > best_f1:
                best_f1, best_epoch, bad, best_state = val["tab_f1"], epoch, 0, snapshot(model)
            elif epoch > 0:
                bad += 1
            log(f"epoch {epoch:3d} | train {train_parts.get('total', float('nan')):.4f} | val loss "
                f"{val['loss']['total']:.4f} (kong {val['loss']['onset'] + val['loss']['offset'] + val['loss']['frame']:.4f}, "
                f"string {val['loss']['string']:.4f}) | val note F1 {val['note_f1']:.4f} "
                f"tab F1 {val['tab_f1']:.4f} @ {val['thresholds']} | "
                f"{t_train:.0f}+{entry['seconds_total'] - t_train:.0f} s"
                + ("  *best*" if best_epoch == epoch else ""))
            torch.save({"epoch": epoch, "state": snapshot(model), "optimizer": opt.state_dict(),
                        "torch_rng": torch.get_rng_state(), "history": history,
                        "best_f1": best_f1, "best_epoch": best_epoch, "bad": bad,
                        "best_state": best_state}, last_path)
        if bad >= args.patience:
            log(f"early stop: no val Tab F1 improvement for {args.patience} epochs")
    finally:
        if pool:
            pool.close()
            pool.join()
    log(f"best epoch {best_epoch} (val Tab F1 {best_f1:.4f})")
    restore(model, best_state)
    return finish(args, model, run_dir, best_epoch, history, device, log)


def finish(args, model, run_dir, best_epoch, history, device, log):
    """Cache the best model's Kong outputs for val/test, then run stage 1's evaluation on them."""
    splits = load_split(args.root, args.val_player, args.test_player)
    cache_dir = run_dir / "kong_cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)              # always from this run's best state
    meta = {**cache_meta(args.root, args.checkpoint), "run": run_dir.name, "best_epoch": best_epoch}
    track_ids = [*splits["validation"].track_id, *splits["test"].track_id]
    build_cache(args.root, cache_dir, meta, lambda: model.encoder, device, log=log,
                track_ids=track_ids)

    val_data = to_device(load_windows(cache_dir, splits["validation"].track_id), device)
    val = evaluate(model.head, val_data, THRESHOLDS, args.batch_size)
    threshold = max(THRESHOLDS, key=lambda t: val["frame"][t]["tab"]["f1"])
    del val_data
    test = evaluate(model.head, to_device(load_windows(cache_dir, splits["test"].track_id), device),
                    (threshold, 0.5), args.batch_size)

    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config["cache_dir"] = str(cache_dir)       # eval_head_c reads Kong outputs from here
    state = snapshot(model)
    torch.save({"head": state["head"], "lora": state["lora"], "hidden": model.head.mlp[0].out_features,
                "dropout": model.head.mlp[2].p, "threshold": threshold, "best_epoch": best_epoch,
                "lora_config": {"rank": args.rank, "alpha": args.alpha},
                "args": config, "cache_meta": meta, "stage": 2}, run_dir / "best.pt")
    (run_dir / "metrics.json").write_text(json.dumps(
        {"best_epoch": best_epoch, "threshold": threshold, "history": history,
         "validation": val, "test": test, "args": config, "cache_meta": meta}, indent=2))
    log(_summary("validation", val, threshold))
    log(_summary("test", test, threshold))

    results = ev.run(run_dir, args.kong_repo, args.workers, log=log)
    if not args.keep_cache:
        shutil.rmtree(cache_dir)
    return results


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--root", default=str(REPO / "features" / "v1"))
    p.add_argument("--val_player", default="04")
    p.add_argument("--test_player", default="05")
    p.add_argument("--lam", type=float, default=0.2, help="weight of the string loss")
    p.add_argument("--tag", default="stage2", help="run folder prefix, e.g. stage2_lam0.2")
    p.add_argument("--out_dir", default=str(REPO / "runs" / "head_c"))
    p.add_argument("--stage1_dir", default=None,
                   help="default: <out_dir>/stage1_val{V}_test{T}")
    p.add_argument("--checkpoint",
                   default=str(REPO / "src" / "kong_wrapper" / "checkpoints" / "kong_note_pedal.pth"))
    p.add_argument("--kong_repo", default=str(ev.DEFAULT_KONG_REPO))
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=16.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--crops_per_clip", type=int, default=2, help="random 10 s crops per clip per epoch")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    p.add_argument("--workers", type=int, default=None,
                   help="post-processing processes (default: CPU count - 1)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    p.add_argument("--resume", action="store_true", help="continue from <run_dir>/last.pt")
    p.add_argument("--keep_cache", action="store_true", help="keep <run_dir>/kong_cache")
    args = p.parse_args(argv)
    args.val_player, args.test_player = args.val_player.zfill(2), args.test_player.zfill(2)
    if args.workers is None:
        import os
        args.workers = max(1, (os.cpu_count() or 2) - 1)
    return args


if __name__ == "__main__":
    run(parse_args())
