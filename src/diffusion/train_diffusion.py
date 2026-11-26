"""Train a GuitarSet diffusion fold: frozen Kong (stage 1), then LoRA (stage 2).

python src/diffusion/train_diffusion.py --test_player 05 --amp
Use --resume for epoch-boundary recovery with optimizer, schedule and RNG state.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for sub in (HERE, HERE.parent / "data", HERE.parent / "kong_wrapper"):
    if str(sub) not in sys.path:
        sys.path.insert(0, str(sub))
from loader import GuitarSetDataset, collate, load_split  # noqa: E402
from lora import TunableKongEncoder, lora_state_dict, load_lora_state_dict  # noqa: E402
from kong_tab_diffusion import KongTabDiffusion  # noqa: E402


def pick_device(name):
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available()
                        else "cpu" if name == "auto" else name)


def autocast(device, enabled):
    from contextlib import nullcontext
    if enabled and device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def build_model(args, device):
    encoder = TunableKongEncoder.from_checkpoint(args.checkpoint, repo_path=args.kong_repo,
                                                 use_adapter=args.use_adapter)
    model = KongTabDiffusion(encoder, timesteps=args.timesteps, schedule=args.schedule,
                           hidden_dim=args.hidden_dim, embed_dim=args.embed_dim,
                           lstm_hidden=args.lstm_hidden, heads=args.heads,
                           window=args.window, dilations=args.dilations, d_ff=args.d_ff,
                           dropout=args.dropout).to(device)
    if getattr(args, "stage", 1) == 2:
        encoder.set_finetune("lora", rank=args.rank, alpha=args.alpha)
    return model


def snapshot(model):
    def cpu(module):
        return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
    adapter = getattr(model.encoder, "fusion_adapter", None)
    return {"denoiser": cpu(model.denoiser), "diffusion": cpu(model.diffusion),
            "adapter": cpu(adapter) if adapter is not None else None,
            "lora": lora_state_dict(model.encoder.note_model)}


def restore(model, state):
    if state.get("lora"):
        load_lora_state_dict(model.encoder.note_model, state["lora"])
    model.denoiser.load_state_dict(state["denoiser"])
    model.diffusion.load_state_dict(state["diffusion"])
    adapter = getattr(model.encoder, "fusion_adapter", None)
    if (adapter is None) != (state["adapter"] is None):
        raise ValueError("checkpoint adapter configuration differs")
    if adapter is not None:
        adapter.load_state_dict(state["adapter"])


class FeatureCache:
    """Content-addressed raw features for exact audio crops and frozen weights.

    Hashing actual waveform bytes prevents using features for a different crop.
    Namespace includes all encoder state tensors, not merely the checkpoint path.
    A trainable adapter/backbone forbids cached final encoder features.
    """
    def __init__(self, directory, encoder):
        if any(p.requires_grad for p in encoder.parameters()):
            raise ValueError("feature caching requires a fully frozen encoder, including the adapter")
        digest = hashlib.sha256()
        for key, value in encoder.state_dict().items():
            value = value.detach().cpu().contiguous()
            digest.update(f"{key}:{value.dtype}:{tuple(value.shape)}".encode())
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        self.directory = Path(directory) / digest.hexdigest()
        self.directory.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def get(self, model, waveform):
        if any(p.requires_grad for p in model.encoder.parameters()):
            raise ValueError("encoder became trainable; cached final features are invalid")
        results = [None] * len(waveform)
        missing, paths = [], []
        for i, audio in enumerate(waveform):
            raw = audio.detach().cpu().contiguous().numpy().tobytes()
            path = self.directory / (hashlib.sha256(raw).hexdigest() + ".pt")
            if path.exists():
                results[i] = torch.load(path, weights_only=True, map_location=waveform.device)
            else:
                missing.append(i)
                paths.append(path)
        if missing:
            # Cache creation is float32 regardless of training autocast settings.
            with torch.autocast(waveform.device.type, enabled=False):
                features = model.features(waveform[missing].float())
            for i, path, feature in zip(missing, paths, features):
                results[i] = feature
                temporary = path.with_suffix(".tmp")
                torch.save(feature.detach().cpu(), temporary)
                temporary.replace(path)
        return torch.stack(results)


def lr_lambda(warmup_steps, total_steps):
    def schedule(step):
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1)))
    return schedule


def loss_weights(args):
    return {k: getattr(args, k) for k in ("lambda_onset", "masked_weight", "unmasked_weight",
                                         "onset_pos_weight")}


def train_step(model, batch, optimizer, device, args, cache=None):
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    features = cache.get(model, batch["waveform"]) if cache else None
    with autocast(device, args.amp):
        losses = model.training_loss(batch, features=features, **loss_weights(args))
    if not bool(torch.isfinite(losses["loss"])):
        raise FloatingPointError("non-finite diffusion loss")
    losses["loss"].backward()
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
    optimizer.step()
    # Reading scalar losses synchronizes CUDA, making the logged elapsed time
    # include actual device execution rather than only CPU launch overhead.
    stats = {k: float(losses[k].detach()) for k in ("loss", "fret_loss", "onset_loss")}
    stats.update(grad_norm=float(norm), valid_positions=int(losses["correct"][1]))
    stats["counts"] = {k: [int(v) for v in losses[k]] for k in (
        "correct", "corrupted_correct", "active_correct", "silence_correct")}
    stats["step_seconds"] = time.perf_counter() - started
    return stats


def rng_state(generator):
    return {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
            "python": random.getstate(), "loader": generator.get_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state, generator):
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    generator.set_state(state["loader"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    run_dir = Path(args.out_dir) / f"{args.tag}_test{args.test_player}"
    run_dir.mkdir(parents=True, exist_ok=True)
    def log(message):
        print(message, flush=True)
        with (run_dir / "train.log").open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")
    if (run_dir / "last.pt").exists() and not args.resume:
        raise FileExistsError(f"{run_dir} already has a checkpoint; use --resume or a new --tag")
    initial = None
    if args.stage == 2:
        # A resumed stage 2 is self-contained apart from the pretrained Kong.
        source = (run_dir / "last.pt" if args.resume else
                  Path(args.stage1_dir or Path(args.out_dir) / f"stage1_test{args.test_player}") / "final.pt")
        initial = torch.load(source, map_location="cpu", weights_only=False)
        previous = initial["args"]
        if not args.resume and (previous.get("stage", 1) != 1 or
                                previous["test_player"] != args.test_player):
            raise ValueError("stage 2 requires a stage 1 checkpoint for the same test player")
        # Reconstruct the trained architecture and its exact corruption process.
        for key in ("hidden_dim", "embed_dim", "lstm_hidden", "heads", "window", "d_ff",
                    "dilations", "dropout", "timesteps", "schedule", "use_adapter"):
            setattr(args, key, previous[key])
        if not args.skip_eval and any(s < 1 or s > args.timesteps for s in args.inference_steps):
            raise ValueError("inference_steps must lie in the stage 1 model's 1..timesteps")
    config = vars(args).copy()
    splits = load_split(args.root, args.test_player)
    if len(splits["train"]) == 0:
        raise ValueError("training split is empty")
    dataset = GuitarSetDataset(splits["train"], args.root, mode=args.crop_mode,
                               crops_per_clip=args.crops_per_clip)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
                        num_workers=args.num_workers, worker_init_fn=seed_worker,
                        generator=generator, pin_memory=device.type == "cuda")
    model = build_model(args, device)
    if initial is not None and not args.resume:
        restore(model, initial["state"])
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": [p for p in params if p.ndim >= 2], "weight_decay": args.weight_decay},
        {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0}], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda(round(args.warmup_epochs * len(loader)), args.epochs * len(loader)))
    history, first_epoch = [], 1
    if args.resume:
        saved = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
        operational = {"resume", "device", "log_every", "skip_eval", "inference_steps", "decode_batch"}
        legacy_defaults = {"stage": 1, "stage1_dir": None, "rank": 8, "alpha": 16.0}
        changed = [k for k, v in config.items() if k not in operational and
                   saved["args"].get(k, legacy_defaults.get(k)) != v]
        if changed:
            raise ValueError(f"resume requires the same training configuration: changed {changed}")
        restore(model, saved["state"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        restore_rng(saved["rng"], generator)
        first_epoch, history = saved["epoch"] + 1, saved["history"]
    cache = FeatureCache(args.feature_cache, model.encoder) if args.feature_cache else None
    counts = {"total": sum(p.numel() for p in model.parameters()),
              "trainable": sum(p.numel() for p in params),
              "denoiser": sum(p.numel() for p in model.denoiser.parameters())}
    log(f"stage {args.stage}; device {device}; parameters {counts}; {len(dataset)} training crops/windows; "
        f"train players {sorted(set(splits['train'].player_id))}; test {args.test_player}")
    for epoch in range(first_epoch, args.epochs + 1):
        model.train()
        started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        totals = {k: 0.0 for k in ("loss", "fret_loss", "onset_loss", "grad_norm", "step_seconds")}
        accuracy = {k: [0, 0] for k in ("correct", "corrupted_correct", "active_correct", "silence_correct")}
        for step, batch in enumerate(loader, 1):
            stats = train_step(model, batch, optimizer, device, args, cache)
            scheduler.step()
            for k in totals:
                totals[k] += stats[k]
            for k, pair in stats["counts"].items():
                accuracy[k] = [a + b for a, b in zip(accuracy[k], pair)]
            if args.log_every and step % args.log_every == 0:
                log(f"epoch {epoch} step {step}/{len(loader)} loss {stats['loss']:.4f} "
                    f"step seconds {stats['step_seconds']:.3f}")
        train = {k: v / len(loader) for k, v in totals.items()}
        train.update({k.replace("correct", "accuracy"): a / b if b else None
                      for k, (a, b) in accuracy.items()})
        record = {"epoch": epoch, "train": train, "seconds": time.perf_counter() - started,
                  "lr_end": scheduler.get_last_lr()[0],
                  "peak_mem_gb": torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0}
        history.append(record)
        log(json.dumps(record))
        (run_dir / "history.jsonl").write_text("".join(json.dumps(r) + "\n" for r in history))
        saved = {"epoch": epoch, "state": snapshot(model), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "rng": rng_state(generator),
                 "history": history, "args": config}
        temporary = run_dir / "last.tmp"
        torch.save(saved, temporary)
        temporary.replace(run_dir / "last.pt")
    final = {"state": snapshot(model), "config": model.config, "timesteps": args.timesteps,
             "schedule": args.schedule, "args": config, "selection": "last epoch; no validation player"}
    torch.save(final, run_dir / "final.pt")
    (run_dir / "metrics.json").write_text(json.dumps(
        {"args": config, "parameters": counts, "history": history, "selection": final["selection"]}, indent=2))
    if not args.skip_eval:
        from eval_diffusion import evaluate
        return evaluate(model.eval(), run_dir, config, device, log=log)
    return history


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", type=int, choices=(1, 2), default=1)
    p.add_argument("--stage1_dir", default=None, help="stage 2: directory containing stage 1 final.pt")
    p.add_argument("--rank", type=int, default=8, help="stage 2: Kong LoRA rank")
    p.add_argument("--alpha", type=float, default=16.0, help="stage 2: Kong LoRA alpha")
    p.add_argument("--root", default=str(REPO / "features" / "v1"))
    p.add_argument("--checkpoint", default=str(REPO / "src/kong_wrapper/checkpoints/kong_note_pedal.pth"))
    p.add_argument("--kong_repo", default=str(REPO / "src/kong_wrapper/third_party/piano_transcription"))
    p.add_argument("--test_player", default="05")
    p.add_argument("--out_dir", default=str(REPO / "runs/diffusion"))
    p.add_argument("--tag", default=None, help="run prefix (default stage1 / stage2)")
    for name, default in {"batch_size": 8, "epochs": 25, "num_workers": 4, "seed": 0,
                          "timesteps": 100, "hidden_dim": 64, "embed_dim": 16,
                          "lstm_hidden": 32, "heads": 4, "window": 5, "d_ff": 256,
                          "crops_per_clip": 2, "log_every": 50, "decode_batch": 8}.items():
        p.add_argument(f"--{name}", type=int, default=default)
    for name, default in {"lr": None, "weight_decay": 0.01, "grad_clip": 1.0,
                          "warmup_epochs": 2.0, "dropout": 0.1, "lambda_onset": 0.5,
                          "masked_weight": 1.0, "unmasked_weight": 1.0,
                          "onset_pos_weight": 1.0, "onset_threshold": 0.5,
                          "temperature": 1.0}.items():
        p.add_argument(f"--{name}", type=float, default=default)
    p.add_argument("--dilations", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--schedule", choices=("linear", "cosine"), default="linear")
    p.add_argument("--inference_steps", type=int, nargs="+", default=[10, 25, 50, 100])
    p.add_argument("--crop_mode", choices=("random", "windows"), default="random")
    p.add_argument("--feature_cache", default=None, help="exact-crop cache; incompatible with trainable adapter")
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                   help="BF16 on supported CUDA devices (default on); --no-amp disables")
    for name in ("use_adapter", "resume", "skip_eval"):
        p.add_argument(f"--{name}", action="store_true")
    args = p.parse_args(argv)
    args.tag = args.tag or f"stage{args.stage}"
    args.lr = args.lr if args.lr is not None else (3e-4 if args.stage == 1 else 1e-4)
    args.test_player = args.test_player.zfill(2)
    positive = ("batch_size", "epochs", "timesteps", "crops_per_clip", "decode_batch",
                "hidden_dim", "embed_dim", "lstm_hidden", "heads", "window", "d_ff")
    if any(getattr(args, k) < 1 for k in positive) or args.num_workers < 0 or args.log_every < 0:
        p.error("invalid training dimensions/counts")
    if args.use_adapter and args.feature_cache:
        p.error("cannot cache final Kong features while training the fusion adapter")
    if args.stage == 2 and args.feature_cache:
        p.error("cannot cache features while training Kong LoRA in stage 2")
    if args.rank < 1 or not math.isfinite(args.alpha) or args.alpha <= 0:
        p.error("LoRA rank and alpha must be positive")
    if not 0 <= args.onset_threshold <= 1 or args.temperature <= 0:
        p.error("invalid onset threshold or sampling temperature")
    if min(args.lambda_onset, args.masked_weight, args.unmasked_weight, args.weight_decay,
           args.warmup_epochs) < 0 or min(args.onset_pos_weight, args.grad_clip, args.lr) <= 0:
        p.error("invalid optimization/loss weights")
    if args.masked_weight + args.unmasked_weight == 0:
        p.error("at least one reconstruction weight must be positive")
    if not args.skip_eval and any(s < 1 or s > args.timesteps for s in args.inference_steps):
        p.error("inference_steps must lie in 1..timesteps")
    return args


if __name__ == "__main__":
    run(parse_args())
