"""AR decoder trainer, both stages, on one leave-one-player-out fold (5 train players + 1 test).

    python src/decoder/train_decoder.py --stage 1 --test_player 05
    python src/decoder/train_decoder.py --stage 2 --test_player 05

Model: Kong (TunableKongEncoder) + the fusion adapter -> `features` -> a Transformer
decoder that writes TabTokenizer tokens (src/decoder/tab_decoder.py). Defaults:
d_model 256, 4 layers, 4 heads, FFN 1024, dropout 0.1, max_len 1280.

There is no validation player (team convention, same as TabCNN): each stage trains
for a fixed number of epochs and keeps the last one. Decoding is greedy, so there
are no thresholds to pick either.

Stage 1: Kong frozen (run live on random 10 s crops); the adapter and a fresh
decoder train. Defaults: AdamW lr 3e-4, 50 epochs.

Stage 2: starts from the fold's stage 1 adapter + decoder
(<out_dir>/stage1_test{T}/final.pt) and a fresh rank-8 LoRA on every Kong
Linear / GRU weight (src/kong_wrapper/lora.py). Kong runs with dropout on and
BatchNorm frozen; the adapter and decoder train fully. Defaults: lr 1e-4, 25 epochs.

Both stages: next-token cross-entropy (teacher forcing, PAD ignored); AdamW with
weight decay 0.01 on weight matrices only; gradient clipping at 1.0; the learning
rate warms up linearly over --warmup_epochs (2), then follows a cosine to 0 at the
last epoch. --amp turns on bf16 autocast (CUDA only). There is no Kong loss: the
decoder reads `features`, not Kong's piano outputs (same as Head A).

After training, eval_decoder.evaluate decodes the test player and scores it at
note level.

For EC2 / long runs: everything is logged to <run_dir>/train.log; <run_dir>/last.pt
is saved every epoch, and --resume continues from it (a finished run, one with
eval.json, is skipped).

Writes <out_dir>/<tag>_test{T}/ (tag defaults to stage1 / stage2): final.pt
(adapter + decoder [+ LoRA A/B] + model config + tokenizer config), last.pt,
metrics.json, eval.json, test_notes.npz, train.log.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
for _sub in (HERE, HERE.parent / "data", HERE.parent / "kong_wrapper", HERE.parent / "metrics"):
    if str(_sub) not in sys.path:
        sys.path.insert(0, str(_sub))
import eval_decoder as ev  # noqa: E402
from loader import GuitarSetDataset, collate, load_split  # noqa: E402
from lora import TunableKongEncoder, load_lora_state_dict, lora_parameters, lora_state_dict  # noqa: E402
from tab_decoder import DEFAULT_CONFIG, KongTabDecoder, split_tokens, token_accuracy, token_loss  # noqa: E402
from tokenizer import TabTokenizer  # noqa: E402

REPO = HERE.parents[1]
DEFAULT_KONG_REPO = HERE.parent / "kong_wrapper" / "third_party" / "piano_transcription"
STAGE_DEFAULTS = {1: {"lr": 3e-4, "epochs": 50}, 2: {"lr": 1e-4, "epochs": 25}}
MODEL_ARGS = ("d_model", "n_layers", "n_heads", "d_ff", "dropout", "max_len")


# ----- model -----

def stage1_dir(args):
    return Path(args.stage1_dir or Path(args.out_dir) / f"stage1_test{args.test_player}")


def build_model(args, device):
    """Stage 1: frozen Kong + fresh adapter and decoder. Stage 2: LoRA'd Kong + stage 1's."""
    torch.manual_seed(args.seed)                   # same adapter / decoder init every run
    encoder = TunableKongEncoder.from_checkpoint(args.checkpoint, repo_path=args.kong_repo,
                                                 use_adapter=True)             # starts frozen
    if args.stage == 1:
        model = KongTabDecoder(encoder, TabTokenizer(),
                               **{k: getattr(args, k) for k in MODEL_ARGS})
        model.set_stage(1)
        return model.to(device)
    stage1 = torch.load(stage1_dir(args) / "final.pt", map_location="cpu", weights_only=False)
    model = KongTabDecoder(encoder, TabTokenizer.from_dict(stage1["tokenizer"]), **stage1["config"])
    model.encoder.fusion_adapter.load_state_dict(stage1["adapter"])
    model.decoder.load_state_dict(stage1["decoder"])
    model.set_stage(2, rank=args.rank, alpha=args.alpha)
    return model.to(device)


def param_groups(model, weight_decay):
    """LoRA A/B + adapter + decoder; weight decay on matrices (Linear weights, embeddings,
    LoRA A/B), none on biases or LayerNorm parameters."""
    params = [*lora_parameters(model.encoder.note_model),
              *model.encoder.fusion_adapter.parameters(), *model.decoder.parameters()]
    decay = [p for p in params if p.ndim >= 2]
    no_decay = [p for p in params if p.ndim < 2]
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def lr_lambda(warmup_steps, total_steps):
    """Linear warmup to the peak, then cosine to 0 at total_steps."""
    def f(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return f


def _cpu_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def snapshot(model):
    return {"lora": lora_state_dict(model.encoder.note_model),
            "adapter": _cpu_state(model.encoder.fusion_adapter),
            "decoder": _cpu_state(model.decoder)}


def restore(model, state):
    if state["lora"]:
        load_lora_state_dict(model.encoder.note_model, state["lora"])
    model.encoder.fusion_adapter.load_state_dict(state["adapter"])
    model.decoder.load_state_dict(state["decoder"])


# ----- training -----

def batch_to(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


class Logger:
    def __init__(self, path):
        self.path = Path(path)

    def __call__(self, *parts):
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        with open(self.path, "a") as f:
            f.write(line + "\n")


def run(args, log=None):
    device = ev.pick_device(args.device)
    run_dir = Path(args.out_dir) / f"{args.tag}_test{args.test_player}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log = log or Logger(run_dir / "train.log")
    if args.resume and (run_dir / "eval.json").exists():
        log(f"{run_dir} already finished (eval.json exists); skipping")
        return json.loads((run_dir / "eval.json").read_text())

    np.random.seed(args.seed)
    model = build_model(args, device)
    source = ("fresh adapter + decoder" if args.stage == 1
              else f"stage 1 adapter + decoder from {stage1_dir(args)}")
    log(f"stage {args.stage}; device {device}; run {run_dir}; {source}; "
        f"decoder {model.config}" + ("; bf16 autocast" if args.amp and device.type == "cuda" else ""))
    torch.manual_seed(args.seed)
    splits = load_split(args.root, args.test_player)
    train_ds = GuitarSetDataset(splits["train"], args.root, mode="random", targets=("tokens",),
                                crops_per_clip=args.crops_per_clip)
    if train_ds.tokenizer != model.tokenizer:
        raise ValueError(f"loader tokenizer {train_ds.tokenizer} != model's {model.tokenizer}")
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
                        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
                        pin_memory=device.type == "cuda", drop_last=True)
    groups = param_groups(model, args.weight_decay)
    params = [p for g in groups for p in g["params"]]
    opt = torch.optim.AdamW(groups, lr=args.lr)
    steps_per_epoch = len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda(args.warmup_epochs * steps_per_epoch, args.epochs * steps_per_epoch))
    n_lora = sum(p.numel() for p in lora_parameters(model.encoder.note_model))
    log(f"train {len(splits['train'])} clips x {args.crops_per_clip} crops = "
        f"{len(train_ds)} per epoch ({steps_per_epoch} steps); LoRA {n_lora:,} + adapter "
        f"{sum(p.numel() for p in model.encoder.fusion_adapter.parameters()):,} + decoder "
        f"{sum(p.numel() for p in model.decoder.parameters()):,} trainable; peak lr {args.lr}, "
        f"weight decay {args.weight_decay}, warmup {args.warmup_epochs} epochs then cosine, "
        f"{args.epochs} epochs (last one kept)")

    history, start = [], 1
    last_path = run_dir / "last.pt"
    if args.resume and last_path.exists():
        last = torch.load(last_path, map_location="cpu", weights_only=False)
        restore(model, last["state"])
        opt.load_state_dict(last["optimizer"])
        sched.load_state_dict(last["scheduler"])
        torch.set_rng_state(last["torch_rng"])
        history, start = last["history"], last["epoch"] + 1
        log(f"resumed after epoch {last['epoch']}")

    for epoch in range(start, args.epochs + 1):
        t0 = time.time()
        model.train()
        sums, steps = defaultdict(float), 0
        for batch in loader:
            b = batch_to(batch, device)
            tokens_in, targets = split_tokens(b["tokens"], model.max_len)
            with ev.autocast(device, args.amp):
                logits = model(b["waveform"], tokens_in, b["n_valid_frames"])
            loss = token_loss(logits, targets)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
            sched.step()
            sums["loss"] += loss.item()
            sums["accuracy"] += token_accuracy(logits, targets).item()
            sums["grad_norm"] += float(grad_norm)
            sums["truncated"] += int((b["token_lengths"] > model.max_len + 1).sum())
            steps += 1
        train = {k: v / max(steps, 1) for k, v in sums.items() if k != "truncated"}
        train["truncated_sequences"] = int(sums["truncated"])
        history.append({"epoch": epoch, "train": train, "lr_end": sched.get_last_lr()[0],
                        "seconds": time.time() - t0})
        log(f"epoch {epoch:3d}/{args.epochs} | loss {train.get('loss', float('nan')):.4f} | "
            f"token acc {train.get('accuracy', float('nan')):.3f} | grad norm "
            f"{train.get('grad_norm', float('nan')):.2f} | lr {history[-1]['lr_end']:.2e} | "
            f"{history[-1]['seconds']:.0f} s"
            + (f" | {train['truncated_sequences']} truncated" if train["truncated_sequences"] else ""))
        torch.save({"epoch": epoch, "state": snapshot(model), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "torch_rng": torch.get_rng_state(),
                    "history": history}, last_path)
    return finish(args, model, run_dir, history, device, log)


def finish(args, model, run_dir, history, device, log):
    """Save the final model, then decode + score the test player."""
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    state = snapshot(model)
    ckpt = {"adapter": state["adapter"], "decoder": state["decoder"], "config": model.config,
            "tokenizer": model.tokenizer.to_dict(), "epochs": args.epochs, "args": config,
            "stage": args.stage}
    if args.stage == 2:
        ckpt.update(lora=state["lora"], lora_config={"rank": args.rank, "alpha": args.alpha})
    torch.save(ckpt, run_dir / "final.pt")
    (run_dir / "metrics.json").write_text(json.dumps(
        {"stage": args.stage, "epochs": args.epochs, "config": model.config,
         "history": history, "args": config}, indent=2))
    return ev.evaluate(model, run_dir, config, device, args.decode_batch, args.amp, log)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--stage", type=int, choices=(1, 2), required=True)
    p.add_argument("--root", default=str(REPO / "features" / "v1"))
    p.add_argument("--test_player", default="05")
    p.add_argument("--tag", default=None, help="run folder prefix (default stage1 / stage2)")
    p.add_argument("--out_dir", default=str(REPO / "runs" / "decoder"))
    p.add_argument("--stage1_dir", default=None,
                   help="stage 2: folder with stage 1's final.pt (default "
                        "<out_dir>/stage1_test{T})")
    p.add_argument("--checkpoint",
                   default=str(REPO / "src" / "kong_wrapper" / "checkpoints" / "kong_note_pedal.pth"))
    p.add_argument("--kong_repo", default=str(DEFAULT_KONG_REPO))
    for k in MODEL_ARGS:
        p.add_argument(f"--{k}", type=type(DEFAULT_CONFIG[k]), default=DEFAULT_CONFIG[k],
                       help=f"stage 1: decoder {k} (stage 2 uses stage 1's)")
    p.add_argument("--rank", type=int, default=8, help="stage 2: LoRA rank")
    p.add_argument("--alpha", type=float, default=16.0, help="stage 2: LoRA alpha")
    p.add_argument("--lr", type=float, default=None, help="peak lr; default 3e-4 stage 1, 1e-4 stage 2")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=None, help="default: 50 stage 1, 25 stage 2")
    p.add_argument("--warmup_epochs", type=float, default=2.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--crops_per_clip", type=int, default=2, help="random 10 s crops per clip per epoch")
    p.add_argument("--decode_batch", type=int, default=16, help="test windows decoded at once")
    p.add_argument("--amp", action="store_true", help="bf16 autocast (CUDA only)")
    p.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    p.add_argument("--resume", action="store_true", help="continue from <run_dir>/last.pt")
    args = p.parse_args(argv)
    args.test_player = args.test_player.zfill(2)
    for k, v in STAGE_DEFAULTS[args.stage].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    args.tag = args.tag or f"stage{args.stage}"
    return args


if __name__ == "__main__":
    run(parse_args())
