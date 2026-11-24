"""AR decoder trainer, both stages: decoder1 on a GuitarSet fold, decoder2 on SynthTab.

    # decoder1: one leave-one-player-out fold (5 train players + 1 test)
    python src/decoder/train_decoder.py --arch decoder1 --stage 1 --test_player 05
    python src/decoder/train_decoder.py --arch decoder1 --stage 2 --test_player 05
    # decoder2: SynthTab only, then zero-shot on all of GuitarSet
    python src/decoder/train_decoder.py --arch decoder2 --stage 1 --train_root features/synthtab
    python src/decoder/train_decoder.py --arch decoder2 --stage 2 --train_root features/synthtab
    # pretraining ablation: decoder1 + randomly initialized Kong, one stage
    python src/decoder/train_decoder.py --arch decoder1 --kong_init random --stage 1 --test_player 05

Model: Kong (TunableKongEncoder, no fusion adapter) -> `features` (Kong's final GRU,
512-d per frame) -> a Transformer decoder that writes TabTokenizer tokens
(src/decoder/tab_decoder.py, ARCH_CONFIGS). Both: d_model 256, FFN 1024, dropout
0.1, max_len 1280.
  --arch decoder1  4 layers, 4 heads, GELU FFN. Defaults: --data guitarset,
                   batch 8, no gradient accumulation, --out_dir runs/decoder1.
  --arch decoder2  TART's transformer (AudioFret): 6 layers, 8 heads, gated-GELU
                   FFN. Defaults: --data synthtab, microbatch 4 x --grad_accum 4 =
                   effective batch 16 (TART's), --out_dir runs/decoder2.
--n_layers / --n_heads / --d_ff / --ffn ... override the arch's values (stage 1).

Data (--data):
  guitarset  train on the 5 other players of --root (features/v1), test on
             --test_player. Run folder <tag>_test{T}.
  synthtab   train on every clip of --train_root (a SynthTab features folder in
             the features/v1 layout, see loader.load_manifest), test zero-shot on
             every GuitarSet clip of --root. Run folder <tag>_zeroshot. The model
             never sees GuitarSet in training (TART was evaluated zero-shot): it
             raises if --train_root is --root or shares a track_id with it.

There is no validation player (team convention, same as TabCNN): each stage trains
for a fixed number of epochs and keeps the last one. Decoding (greedy / beam) has
no thresholds to pick either.

Stage 1: Kong frozen (run live on random 10 s crops); a fresh decoder trains.
Defaults: AdamW lr 3e-4, 50 epochs.

Stage 2: starts from the same run's stage 1 decoder
(<out_dir>/stage1_test{T}/final.pt, or stage1_zeroshot) and a fresh rank-8 LoRA on
every Kong Linear / GRU weight (src/kong_wrapper/lora.py). Kong runs with dropout
on and BatchNorm frozen; the decoder trains fully. Defaults: lr 1e-4, 25 epochs.

Both stages: next-token cross-entropy (teacher forcing, PAD ignored); AdamW with
weight decay 0.01 on weight matrices only; gradient clipping at 1.0; the learning
rate warms up linearly over --warmup_epochs (2), then follows a cosine to 0 at the
last epoch (both counted in optimizer steps). Gradient accumulation (--grad_accum
N): N microbatches of --batch_size per optimizer step, each loss divided by the
group's total target tokens, so the gradient is the token-mean over the whole
effective batch (the same as one batch of N x --batch_size). --amp turns on bf16
autocast (CUDA only). There is no Kong loss: the decoder reads `features`, not
Kong's piano outputs (same as Head A).

After training, eval_decoder.evaluate decodes the test clips with every
--decoding (default greedy and beam, --beam_size 4, --length_penalty 1.0,
optional --max_chord_span) and scores them at frame and note level.

For EC2 / long runs: everything is logged to <run_dir>/train.log. Each epoch logs
loss, token accuracy, grad norm, lr, the epoch's time, total time so far, an ETA
for the remaining epochs and (CUDA) peak GPU memory; every --log_every optimizer
steps (default 50, 0 = off) a progress line inside the epoch gives running loss /
token accuracy, steps/s and the time left in the epoch. <run_dir>/history.jsonl
(one JSON line per finished epoch) is rewritten every epoch, so curves can be
plotted while the run is going. <run_dir>/last.pt is saved every epoch, and
--resume continues from it (a finished run, one with eval.json, is skipped).

Random Kong (--kong_init random, the pretraining ablation): Kong is built without
the MAESTRO checkpoint (KongEncoder.from_scratch: Kong's own init, and Kong's
forward without its .detach()s, so the tab loss reaches every branch) and trains
together with a fresh decoder in ONE stage (--stage 1; --stage 2 raises), Kong in
TunableKongEncoder's "scratch" mode: every Kong weight and BatchNorm trains, the
STFT / log-mel frontend stays fixed. Same loss, optimizer and schedule as above,
one AdamW group over Kong + decoder. Defaults: lr 3e-4, 75 epochs (= decoder1's
50 + 25), --out_dir runs/<arch>_randkong, --tag scratch. final.pt / last.pt also
store the trained Kong state ("kong"); eval_decoder rebuilds it from there.
The default --kong_init pretrained changes nothing above.

Writes <out_dir>/<tag>_test{T}/ or <tag>_zeroshot/ (tag defaults to stage1 /
stage2): final.pt (decoder [+ LoRA A/B] + model config + tokenizer config),
last.pt, metrics.json, eval.json, test_notes_<decoding>.npz, train.log, history.jsonl.
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
from loader import (GuitarSetDataset, collate, load_manifest,  # noqa: E402
                    load_split, load_synthtab_split)
from lora import TunableKongEncoder, load_lora_state_dict, lora_parameters, lora_state_dict  # noqa: E402
from tab_decoder import (ARCH_CONFIGS, DEFAULT_CONFIG, FFN_TYPES, KongTabDecoder,  # noqa: E402
                         split_tokens, token_accuracy, token_loss)
from tokenizer import PAD_ID, TabTokenizer  # noqa: E402

REPO = HERE.parents[1]
DEFAULT_KONG_REPO = HERE.parent / "kong_wrapper" / "third_party" / "piano_transcription"
STAGE_DEFAULTS = {1: {"lr": 3e-4, "epochs": 50}, 2: {"lr": 1e-4, "epochs": 25}}
RANDOM_KONG_DEFAULTS = {"lr": 3e-4, "epochs": 75}       # one stage = decoder1's 50 + 25
KONG_INITS = ("pretrained", "random")
MODEL_ARGS = ("d_model", "n_layers", "n_heads", "d_ff", "dropout", "max_len", "ffn")
ARCH_TRAIN_DEFAULTS = {"decoder1": {"data": "guitarset", "batch_size": 8, "grad_accum": 1},
                       "decoder2": {"data": "synthtab", "batch_size": 4, "grad_accum": 4}}


# ----- runs and data -----

def run_suffix(args):
    return "zeroshot" if args.data == "synthtab" else f"test{args.test_player}"


def stage1_dir(args):
    return Path(args.stage1_dir or Path(args.out_dir) / f"stage1_{run_suffix(args)}")


def training_data(args):
    """(train rows, val rows or None, their root, test rows).

    GuitarSet fold: a player split, no val. SynthTab: split by the manifest's
    `split` column (train on `train`, validate on `val`); test is all of GuitarSet.
    val is None when there are no `val` rows (manifest with no split column).
    """
    if args.data == "guitarset":
        splits = load_split(args.root, args.test_player)
        return splits["train"], None, args.root, splits["test"]
    train_root, test_root = Path(args.train_root).resolve(), Path(args.root).resolve()
    if train_root == test_root:
        raise ValueError("--data synthtab: --train_root is the GuitarSet --root; the SynthTab "
                         "decoder must never train on GuitarSet")
    sp = load_synthtab_split(args.train_root)
    train, val, test = sp["train"], sp["val"], load_manifest(test_root)
    shared = (set(train.track_id) | set(val.track_id)) & set(test.track_id)
    if shared:
        raise ValueError(f"--data synthtab: {len(shared)} training clips are GuitarSet clips "
                         f"(e.g. {sorted(shared)[0]}); the SynthTab decoder must never see "
                         "GuitarSet")
    return train, (val if len(val) else None), args.train_root, test


# ----- model -----


def random_kong(args):
    return getattr(args, "kong_init", "pretrained") == "random"


def build_model(args, device):
    """Stage 1: frozen Kong + fresh decoder. Stage 2: LoRA'd Kong + stage 1's decoder.
    --kong_init random: randomly initialized Kong ("scratch" mode) + fresh decoder."""
    torch.manual_seed(args.seed)                   # same decoder init every run
    if random_kong(args):
        encoder = TunableKongEncoder.from_scratch(repo_path=args.kong_repo, use_adapter=False)
        encoder.set_finetune("scratch")
        model = KongTabDecoder(encoder, TabTokenizer(),
                               **{k: getattr(args, k) for k in MODEL_ARGS})
        return model.to(device)
    encoder = TunableKongEncoder.from_checkpoint(args.checkpoint, repo_path=args.kong_repo,
                                                 use_adapter=False)            # starts frozen
    if args.stage == 1:
        model = KongTabDecoder(encoder, TabTokenizer(),
                               **{k: getattr(args, k) for k in MODEL_ARGS})
        model.set_stage(1)
        return model.to(device)
    stage1 = torch.load(stage1_dir(args) / "final.pt", map_location="cpu", weights_only=False)
    s1 = stage1["args"]
    if (s1.get("arch", "decoder1"), s1.get("data", "guitarset")) != (args.arch, args.data):
        raise ValueError(f"stage 1 run {stage1_dir(args)} is {s1.get('arch', 'decoder1')} / "
                         f"{s1.get('data', 'guitarset')}, not {args.arch} / {args.data}")
    model = KongTabDecoder(encoder, TabTokenizer.from_dict(stage1["tokenizer"]), **stage1["config"])
    model.decoder.load_state_dict(stage1["decoder"])
    model.set_stage(2, rank=args.rank, alpha=args.alpha)
    return model.to(device)


def kong_parameters(model, kong_init="pretrained"):
    """Kong's trainable weights: LoRA A/B (none when frozen), or with a random Kong
    every Kong weight except the fixed frontend."""
    if kong_init == "random":
        return model.encoder.trainable_parameters()
    return lora_parameters(model.encoder.note_model)


def param_groups(model, weight_decay, kong_init="pretrained"):
    """Kong (LoRA A/B, or all of a random Kong) + decoder; weight decay on matrices
    (Linear / conv weights, GRU matrices, embeddings, LoRA A/B), none on biases,
    LayerNorm or BatchNorm parameters."""
    params = [*kong_parameters(model, kong_init), *model.decoder.parameters()]
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


def snapshot(model, kong_init="pretrained"):
    state = {"lora": lora_state_dict(model.encoder.note_model),
             "decoder": _cpu_state(model.decoder)}
    if kong_init == "random":                      # the whole trained Kong (+ BN stats)
        state["kong"] = _cpu_state(model.encoder.note_model)
    return state


def restore(model, state):
    if state.get("kong") is not None:
        model.encoder.note_model.load_state_dict(state["kong"], strict=True)
    if state["lora"]:
        load_lora_state_dict(model.encoder.note_model, state["lora"])
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


def fmt_time(seconds):
    """45 s, 12m05s, 3h02m."""
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds} s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"


def write_history(run_dir, history):
    """<run_dir>/history.jsonl: one line per finished epoch (rewritten whole, so a
    resumed run never leaves duplicate epochs)."""
    with open(Path(run_dir) / "history.jsonl", "w") as f:
        for h in history:
            f.write(json.dumps(h) + "\n")


def groups_of(loader, n):
    """Yield lists of n consecutive batches (the last group may be shorter)."""
    group = []
    for batch in loader:
        group.append(batch)
        if len(group) == n:
            yield group
            group = []
    if group:
        yield group


def train_step(model, group, opt, params, args, device):
    """One optimizer step over a group of microbatches (gradient accumulation).

    Each microbatch's summed token loss is divided by the group's total target
    tokens, so the gradient is the token-mean over the whole effective batch.
    Returns (loss, token accuracy, grad norm, truncated sequences) for the group.
    """
    group = [batch_to(b, device) for b in group]
    pairs = [split_tokens(b["tokens"], model.max_len) for b in group]
    n_tokens = sum(int((targets != PAD_ID).sum()) for _, targets in pairs)
    opt.zero_grad(set_to_none=True)
    loss_total, correct, truncated = 0.0, 0.0, 0
    for b, (tokens_in, targets) in zip(group, pairs):
        with ev.autocast(device, args.amp):
            logits = model(b["waveform"], tokens_in, b["n_valid_frames"])
        loss = token_loss(logits, targets, reduction="sum") / max(n_tokens, 1)
        loss.backward()
        loss_total += loss.item()
        correct += token_accuracy(logits, targets).item() * int((targets != PAD_ID).sum())
        truncated += int((b["token_lengths"] > model.max_len + 1).sum())
    grad_norm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
    opt.step()
    return loss_total, correct / max(n_tokens, 1), float(grad_norm), truncated


@torch.no_grad()
def validate(model, loader, args, device):
    """Teacher-forced token loss / accuracy over the val set (no generation).

    One forward pass per batch (same as a train step but no backward), so it is
    cheap enough to run every epoch. Token-weighted so clips of different length
    count by their real tokens. Returns None for an empty / missing val loader.
    """
    if loader is None:
        return None
    was_training = model.training
    model.eval()
    loss_sum, correct, n_tokens = 0.0, 0.0, 0
    for b in loader:
        b = batch_to(b, device)
        tokens_in, targets = split_tokens(b["tokens"], model.max_len)
        n = int((targets != PAD_ID).sum())
        if n == 0:
            continue
        with ev.autocast(device, args.amp):
            logits = model(b["waveform"], tokens_in, b["n_valid_frames"])
        loss_sum += token_loss(logits, targets, reduction="sum").item()
        correct += token_accuracy(logits, targets).item() * n
        n_tokens += n
    model.train(was_training)
    return {"loss": loss_sum / max(n_tokens, 1), "accuracy": correct / max(n_tokens, 1)}


def run(args, log=None):
    device = ev.pick_device(args.device)
    run_dir = Path(args.out_dir) / f"{args.tag}_{run_suffix(args)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log = log or Logger(run_dir / "train.log")
    t_start = time.time()
    log(f"started {time.strftime('%Y-%m-%d %H:%M:%S')}")
    if args.resume and (run_dir / "eval.json").exists():
        log(f"{run_dir} already finished (eval.json exists); skipping")
        return json.loads((run_dir / "eval.json").read_text())

    np.random.seed(args.seed)
    model = build_model(args, device)
    source = ("randomly initialized Kong (no MAESTRO weights, no .detach()) + fresh decoder, "
              "trained together" if random_kong(args)
              else "fresh decoder" if args.stage == 1
              else f"stage 1 decoder from {stage1_dir(args)}")
    log(f"{args.arch}, stage {args.stage}; device {device}; run {run_dir}; {source}; "
        f"decoder {model.config}" + ("; bf16 autocast" if args.amp and device.type == "cuda" else ""))
    torch.manual_seed(args.seed)
    train_rows, val_rows, train_root, test_rows = training_data(args)
    train_ds = GuitarSetDataset(train_rows, train_root, mode="random", targets=("tokens",),
                                crops_per_clip=args.crops_per_clip)
    if train_ds.tokenizer != model.tokenizer:
        raise ValueError(f"loader tokenizer {train_ds.tokenizer} != model's {model.tokenizer}")
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
                        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
                        pin_memory=device.type == "cuda", drop_last=True)
    val_loader = None
    if val_rows is not None:
        # Fixed, non-overlapping windows (not random crops): a reproducible val set,
        # the same clips and windows every epoch so val loss is comparable over time.
        val_ds = GuitarSetDataset(val_rows, train_root, mode="windows", targets=("tokens",))
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                collate_fn=collate, num_workers=args.num_workers,
                                pin_memory=device.type == "cuda")
    groups = param_groups(model, args.weight_decay, args.kong_init)
    params = [p for g in groups for p in g["params"]]
    opt = torch.optim.AdamW(groups, lr=args.lr)
    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)          # optimizer steps
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda(args.warmup_epochs * steps_per_epoch, args.epochs * steps_per_epoch))
    n_kong = sum(p.numel() for p in kong_parameters(model, args.kong_init))
    data = (f"{args.data}: train {len(train_rows)} clips from {train_root}, "
            + (f"val {len(val_rows)} clips, " if val_rows is not None else "no val, ")
            + "test "
            + (f"all {len(test_rows)} GuitarSet clips (zero-shot)" if args.data == "synthtab"
               else f"player {args.test_player} ({len(test_rows)} clips)"))
    log(data)
    log(f"{len(train_rows)} clips x {args.crops_per_clip} crops = {len(train_ds)} per epoch; "
        f"microbatch {args.batch_size} x {args.grad_accum} accumulation = effective batch "
        f"{args.batch_size * args.grad_accum} ({steps_per_epoch} optimizer steps per epoch); "
        f"{'Kong (random init)' if random_kong(args) else 'LoRA'} {n_kong:,} + decoder {sum(p.numel() for p in model.decoder.parameters()):,} "
        f"trainable; peak lr {args.lr}, weight decay {args.weight_decay}, warmup "
        f"{args.warmup_epochs} epochs then cosine, {args.epochs} epochs (last one kept)")

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

    t_train = time.time()
    for epoch in range(start, args.epochs + 1):
        t0 = time.time()
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        sums, steps = defaultdict(float), 0
        for group in groups_of(loader, args.grad_accum):
            loss, accuracy, grad_norm, truncated = train_step(model, group, opt, params, args,
                                                              device)
            sched.step()
            sums["loss"] += loss
            sums["accuracy"] += accuracy
            sums["grad_norm"] += grad_norm
            sums["truncated"] += truncated
            steps += 1
            if args.log_every and steps % args.log_every == 0 and steps < steps_per_epoch:
                elapsed = time.time() - t0
                log(f"  epoch {epoch} step {steps}/{steps_per_epoch} | loss "
                    f"{sums['loss'] / steps:.4f} | token acc {sums['accuracy'] / steps:.3f} | "
                    f"lr {sched.get_last_lr()[0]:.2e} | {steps / elapsed:.2f} steps/s | "
                    f"{fmt_time(elapsed)}, ~{fmt_time(elapsed / steps * (steps_per_epoch - steps))}"
                    " left in epoch")
        train = {k: v / max(steps, 1) for k, v in sums.items() if k != "truncated"}
        train["truncated_sequences"] = int(sums["truncated"])
        seconds = time.time() - t0
        done = epoch - start + 1                                # epochs run this session
        eta = (time.time() - t_train) / done * (args.epochs - epoch)
        record = {"epoch": epoch, "train": train, "lr_end": sched.get_last_lr()[0],
                  "seconds": seconds, "elapsed_seconds": time.time() - t_start,
                  "steps": steps}
        if device.type == "cuda":
            record["peak_mem_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
        val = validate(model, val_loader, args, device)
        if val is not None:
            record["val"] = val
        history.append(record)
        log(f"epoch {epoch:3d}/{args.epochs} | loss {train.get('loss', float('nan')):.4f} | "
            f"token acc {train.get('accuracy', float('nan')):.3f} | grad norm "
            f"{train.get('grad_norm', float('nan')):.2f} | lr {record['lr_end']:.2e} | "
            f"{fmt_time(seconds)} (total {fmt_time(record['elapsed_seconds'])}, ETA "
            f"{fmt_time(eta)})"
            + (f" | val loss {val['loss']:.4f} | val acc {val['accuracy']:.3f}" if val else "")
            + (f" | peak mem {record['peak_mem_gb']:.1f} GB" if "peak_mem_gb" in record else "")
            + (f" | {train['truncated_sequences']} truncated" if train["truncated_sequences"] else ""))
        write_history(run_dir, history)
        torch.save({"epoch": epoch, "state": snapshot(model, args.kong_init), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "torch_rng": torch.get_rng_state(),
                    "history": history}, last_path)
    if start <= args.epochs:
        log(f"training done: {args.epochs - start + 1} epochs in {fmt_time(time.time() - t_train)}")
    t_eval = time.time()
    result = finish(args, model, run_dir, history, device, log)
    log(f"eval took {fmt_time(time.time() - t_eval)}; run total {fmt_time(time.time() - t_start)} "
        f"(finished {time.strftime('%Y-%m-%d %H:%M:%S')})")
    return result


def finish(args, model, run_dir, history, device, log):
    """Save the final model, then decode + score the test player."""
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    state = snapshot(model, args.kong_init)
    ckpt = {"decoder": state["decoder"], "config": model.config,
            "tokenizer": model.tokenizer.to_dict(), "epochs": args.epochs, "args": config,
            "stage": args.stage, "arch": args.arch, "data": args.data,
            "kong_init": args.kong_init}
    if random_kong(args):
        ckpt["kong"] = state["kong"]
    if args.stage == 2:
        ckpt.update(lora=state["lora"], lora_config={"rank": args.rank, "alpha": args.alpha})
    torch.save(ckpt, run_dir / "final.pt")
    (run_dir / "metrics.json").write_text(json.dumps(
        {"arch": args.arch, "data": args.data, "stage": args.stage, "epochs": args.epochs,
         "kong_init": args.kong_init,
         "config": model.config,
         "batch": {"micro": args.batch_size, "grad_accum": args.grad_accum,
                   "effective": args.batch_size * args.grad_accum},
         "history": history, "args": config}, indent=2))
    return ev.evaluate(model, run_dir, config, device, args.decode_batch, args.amp, log)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--arch", choices=tuple(ARCH_CONFIGS), default="decoder1",
                   help="decoder1 (midway, GuitarSet) or decoder2 (TART-style, SynthTab)")
    p.add_argument("--stage", type=int, choices=(1, 2), required=True)
    p.add_argument("--data", choices=("guitarset", "synthtab"), default=None,
                   help="default: guitarset for decoder1, synthtab for decoder2")
    p.add_argument("--root", default=str(REPO / "features" / "v1"),
                   help="GuitarSet features (train + test for guitarset; test for synthtab)")
    p.add_argument("--train_root", default=str(REPO / "features" / "synthtab"),
                   help="--data synthtab: SynthTab features folder (features/v1 layout)")
    p.add_argument("--test_player", default="05", help="--data guitarset: the fold's test player")
    p.add_argument("--tag", default=None, help="run folder prefix (default stage1 / stage2)")
    p.add_argument("--out_dir", default=None, help="default runs/<arch>")
    p.add_argument("--stage1_dir", default=None,
                   help="stage 2: folder with stage 1's final.pt (default "
                        "<out_dir>/stage1_test{T} or stage1_zeroshot)")
    p.add_argument("--kong_init", choices=KONG_INITS, default="pretrained",
                   help="pretrained (MAESTRO checkpoint; frozen / LoRA stages) or random "
                        "(pretraining ablation: random Kong + decoder in one stage, --stage 1)")
    p.add_argument("--checkpoint",
                   default=str(REPO / "src" / "kong_wrapper" / "checkpoints" / "kong_note_pedal.pth"))
    p.add_argument("--kong_repo", default=str(DEFAULT_KONG_REPO))
    for k in MODEL_ARGS:
        p.add_argument(f"--{k}", type=type(DEFAULT_CONFIG[k]), default=None,
                       choices=FFN_TYPES if k == "ffn" else None,
                       help=f"stage 1: decoder {k} (default: the arch's; stage 2 uses stage 1's)")
    p.add_argument("--rank", type=int, default=8, help="stage 2: LoRA rank")
    p.add_argument("--alpha", type=float, default=16.0, help="stage 2: LoRA alpha")
    p.add_argument("--lr", type=float, default=None,
                   help="peak lr; default 3e-4 stage 1, 1e-4 stage 2, 3e-4 --kong_init random")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=None,
                   help="default: 50 stage 1, 25 stage 2, 75 --kong_init random")
    p.add_argument("--warmup_epochs", type=float, default=2.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=None,
                   help="microbatch size (default: 8 decoder1, 4 decoder2)")
    p.add_argument("--grad_accum", type=int, default=None,
                   help="microbatches per optimizer step (default: 1 decoder1, 4 decoder2)")
    p.add_argument("--crops_per_clip", type=int, default=2, help="random 10 s crops per clip per epoch")
    p.add_argument("--log_every", type=int, default=50,
                   help="progress line every N optimizer steps inside an epoch (0 = off)")
    p.add_argument("--decode_batch", type=int, default=16, help="test windows decoded at once")
    ev.add_decoding_args(p)
    p.add_argument("--amp", action="store_true", help="bf16 autocast (CUDA only)")
    p.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    p.add_argument("--resume", action="store_true", help="continue from <run_dir>/last.pt")
    args = p.parse_args(argv)
    args.test_player = args.test_player.zfill(2)
    if args.kong_init == "random":
        if args.stage != 1:
            raise ValueError("--kong_init random is one stage (Kong + decoder from scratch); "
                             "use --stage 1, there is no stage 2")
        for k, v in RANDOM_KONG_DEFAULTS.items():
            if getattr(args, k) is None:
                setattr(args, k, v)
    for k, v in STAGE_DEFAULTS[args.stage].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    for k, v in ARCH_TRAIN_DEFAULTS[args.arch].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    arch = {**DEFAULT_CONFIG, **ARCH_CONFIGS[args.arch]}
    for k in MODEL_ARGS:
        if getattr(args, k) is None:
            setattr(args, k, arch[k])
    if args.batch_size < 1 or args.grad_accum < 1:
        raise ValueError("--batch_size and --grad_accum must be >= 1")
    suffix = "_randkong" if args.kong_init == "random" else ""
    args.out_dir = args.out_dir or str(REPO / "runs" / f"{args.arch}{suffix}")
    args.tag = args.tag or ("scratch" if args.kong_init == "random" else f"stage{args.stage}")
    return args


if __name__ == "__main__":
    run(parse_args())
