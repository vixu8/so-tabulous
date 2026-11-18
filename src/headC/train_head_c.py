"""Head C stage 1: Kong frozen, train only the string head, on one leave-one-player-out fold.

    python src/headC/train_head_c.py --val_player 04 --test_player 05

Kong is frozen, so its piano outputs never change. The first run passes every
10 s window of all clips through Kong once and caches the outputs (float16) with
the targets in --cache_dir, one .npz per clip. The cache doesn't depend on the
fold, so later runs and other folds skip Kong entirely and train in seconds.
Trade-off: the head trains on the fixed windows (0, 10, 20 s), not random crops.

Model choice uses only the validation player: early stopping on its string loss,
then Kong's frame threshold for predict_tab_grid is swept on it. The test player
is evaluated once, with the chosen state and threshold.

Metrics here are frame-level (src/metrics/tab_metrics.frame_metrics: same
definitions as src/tabcnn/metrics.py, so comparable with the TabCNN replication). Note-level Note F1 / Tab F1 /
playability come with the eval step.

Writes <out_dir>/stage1_val{V}_test{T}/best.pt and metrics.json.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "data"))
sys.path.insert(0, str(HERE.parent / "kong_wrapper"))
sys.path.insert(0, str(HERE.parent / "metrics"))
from head_c import (IGNORE_INDEX, PIANO_KEYS, HeadC, predict_tab_grid,  # noqa: E402
                    string_loss)
from loader import GuitarSetDataset, collate, load_split  # noqa: E402
from tab_metrics import frame_metrics  # noqa: E402

CACHE_VERSION = 1
# Kong is under-confident on guitar: on val04/test05 the best was 0.1, so the grid goes lower.
THRESHOLDS = [0.01, 0.02, 0.03, 0.05, 0.07] + [round(0.1 * k, 1) for k in range(1, 10)]
DEFAULT_THRESHOLD = 0.5
REPO = HERE.parents[1]


# ----- Kong output cache -----

def cache_meta(root, checkpoint):
    checkpoint = Path(checkpoint)
    return {
        "version": CACHE_VERSION,
        "root": str(Path(root).resolve()),
        "checkpoint": checkpoint.name,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "piano_keys": list(PIANO_KEYS),
    }


def build_cache(root, cache_dir, meta, make_encoder, device, log=print):
    """Cache Kong's piano outputs + targets for every clip in the manifest that lacks a file.

    make_encoder() is called only if at least one clip still needs Kong.
    Raises if cache_dir was built with a different checkpoint / data root.
    """
    import pandas as pd

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


def as_piano(piano_batch):
    """[B, 4, T, 88] float16 cache slice -> the dict HeadC expects (float32)."""
    x = piano_batch.float()
    return {k: x[:, i] for i, k in enumerate(PIANO_KEYS)}


# ----- evaluation -----

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


# ----- training -----

def train(head, train_data, val_data, epochs=50, patience=5, lr=1e-3, weight_decay=1e-4,
          batch_size=8, seed=0, log=print):
    """AdamW on the head; early stopping on validation string loss. Returns (best state, history)."""
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    gen = torch.Generator().manual_seed(seed)
    n = len(train_data["piano"])

    def record(epoch, train_loss, seconds):
        val = evaluate(head, val_data, batch_size=batch_size)
        entry = {"epoch": epoch, "train_loss": train_loss, "val_string_loss": val["string_loss"],
                 "val_string_acc": val["string_acc"],
                 "val_frame_tab_f1@0.5": val["frame"][DEFAULT_THRESHOLD]["tab"]["f1"],
                 "seconds": seconds}
        log(f"epoch {epoch:3d} | train {train_loss if train_loss is not None else float('nan'):.4f}"
            f" | val loss {entry['val_string_loss']:.4f} acc {entry['val_string_acc']:.4f}"
            f" frame tab F1@0.5 {entry['val_frame_tab_f1@0.5']:.4f} | {seconds:.1f} s")
        return entry

    history = [record(0, None, 0.0)]                      # untrained baseline
    best_loss, best_state, best_epoch, bad = float("inf"), copy.deepcopy(head.state_dict()), 0, 0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        head.train()
        total, count = 0.0, 0
        for idx in torch.randperm(n, generator=gen).split(batch_size):
            idx = idx.to(train_data["piano"].device)
            spp = train_data["spp"][idx].long()
            loss = string_loss(head(as_piano(train_data["piano"][idx])), spp)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
            count += len(idx)
        history.append(record(epoch, total / count, time.time() - t0))
        if history[-1]["val_string_loss"] < best_loss:
            best_loss, best_epoch, bad = history[-1]["val_string_loss"], epoch, 0
            best_state = copy.deepcopy(head.state_dict())
        else:
            bad += 1
            if bad >= patience:
                log(f"early stop: no val improvement for {patience} epochs")
                break
    log(f"best epoch {best_epoch} (val string loss {best_loss:.4f})")
    return best_state, best_epoch, history


# ----- main -----

def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _summary(name, res, threshold):
    fm = res["frame"][threshold]
    return (f"{name:>10}: string acc {res['string_acc']:.4f} (loss {res['string_loss']:.4f}) | "
            f"frame @ {threshold}: pitch P/R/F {fm['pitch']['precision']:.3f}/"
            f"{fm['pitch']['recall']:.3f}/{fm['pitch']['f1']:.3f}, tab P/R/F "
            f"{fm['tab']['precision']:.3f}/{fm['tab']['recall']:.3f}/{fm['tab']['f1']:.3f}")


def run(args, make_encoder=None, log=print):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)
    log(f"device: {device}")
    if make_encoder is None:
        def make_encoder():
            from kong_encoder import KongEncoder
            return KongEncoder.from_checkpoint(args.checkpoint, use_adapter=False,
                                               backbone_frozen=True)
    meta = cache_meta(args.root, args.checkpoint)
    build_cache(args.root, args.cache_dir, meta, make_encoder, device, log=log)

    splits = load_split(args.root, args.val_player, args.test_player)
    data = {name: load_windows(args.cache_dir, df.track_id) for name, df in splits.items()}
    for name, d in data.items():
        log(f"{name}: {len(splits[name])} clips, {len(d['piano'])} windows")
    train_data, val_data = to_device(data["train"], device), to_device(data["validation"], device)

    torch.manual_seed(args.seed)  # same head init whether or not Kong was just built
    head = HeadC(hidden=args.hidden, dropout=args.dropout).to(device)
    best_state, best_epoch, history = train(
        head, train_data, val_data, epochs=args.epochs, patience=args.patience, lr=args.lr,
        weight_decay=args.weight_decay, batch_size=args.batch_size, seed=args.seed, log=log)
    head.load_state_dict(best_state)

    val = evaluate(head, val_data, THRESHOLDS, args.batch_size)
    threshold = max(THRESHOLDS, key=lambda t: val["frame"][t]["tab"]["f1"])
    log("val frame tab F1 by threshold: " + ", ".join(
        f"{t}: {val['frame'][t]['tab']['f1']:.3f}" for t in THRESHOLDS) + f" -> {threshold}")
    del train_data, val_data
    test = evaluate(head, to_device(data["test"], device), (threshold, DEFAULT_THRESHOLD),
                    args.batch_size)

    out_dir = Path(args.out_dir) / f"stage1_val{args.val_player}_test{args.test_player}"
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    torch.save({"head": {k: v.cpu() for k, v in best_state.items()},
                "hidden": args.hidden, "dropout": args.dropout, "threshold": threshold,
                "best_epoch": best_epoch, "args": config, "cache_meta": meta},
               out_dir / "best.pt")
    results = {"best_epoch": best_epoch, "threshold": threshold, "history": history,
               "validation": val, "test": test, "args": config, "cache_meta": meta}
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))

    log(_summary("validation", val, threshold))
    log(_summary("test", test, threshold))
    log(f"saved {out_dir / 'best.pt'} and metrics.json")
    return results


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--root", default=str(REPO / "features" / "v1"))
    p.add_argument("--val_player", default="04")
    p.add_argument("--test_player", default="05")
    p.add_argument("--checkpoint",
                   default=str(REPO / "src" / "kong_wrapper" / "checkpoints" / "kong_note_pedal.pth"))
    p.add_argument("--cache_dir", default=str(REPO / "features" / "cache" / "kong_piano_v1"))
    p.add_argument("--out_dir", default=str(REPO / "runs" / "head_c"))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    args = p.parse_args(argv)
    args.val_player, args.test_player = args.val_player.zfill(2), args.test_player.zfill(2)
    return args


if __name__ == "__main__":
    run(parse_args())
