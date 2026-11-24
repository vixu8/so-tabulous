"""Overfit one real training window with frozen Kong features; never use test audio.

python scripts/overfit_diffusion_window.py --track_id 00_BN1-129-Eb_comp --device cuda --amp
This diagnostic saves its own checkpoint/metrics, separate from LOPO runs.
"""

import argparse
import json
from pathlib import Path
import sys

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src/diffusion"))
from train_diffusion import autocast, build_model, parse_args, pick_device, snapshot  # noqa: E402
from loader import GuitarSetDataset, collate, load_split  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--track_id", required=True)
    p.add_argument("--window", type=int, default=0)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--output", default="runs/diffusion/overfit")
    local, training = p.parse_known_args()
    args = parse_args(training)
    if args.use_adapter:
        p.error("this diagnostic caches final features; use the frozen no-adapter baseline")
    if local.steps < 1:
        p.error("steps must be positive")
    torch.manual_seed(args.seed)
    rows = load_split(args.root, args.test_player)["train"]
    rows = rows[rows.track_id == local.track_id]
    if len(rows) != 1:
        p.error("track_id must identify exactly one clip in the training players")
    dataset = GuitarSetDataset(rows, args.root, mode="windows")
    if not 0 <= local.window < len(dataset):
        p.error("window index out of range")
    device = pick_device(args.device)
    batch = {k: v.to(device) if torch.is_tensor(v) else v
             for k, v in collate([dataset[local.window]]).items()}
    model = build_model(args, device).train()
    with torch.no_grad():
        features = model.features(batch["waveform"]).detach()
    optimizer = torch.optim.AdamW(model.denoiser.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = []
    for step in range(1, local.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, args.amp):
            losses = model.training_loss(batch, features=features, lambda_onset=args.lambda_onset,
                onset_pos_weight=args.onset_pos_weight, masked_weight=args.masked_weight,
                unmasked_weight=args.unmasked_weight)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.denoiser.parameters(), args.grad_clip)
        optimizer.step()
        if step == 1 or step % 50 == 0 or step == local.steps:
            row = {"step": step, "loss": float(losses["loss"].detach())}
            for key in ("correct", "active_correct", "corrupted_correct"):
                a, b = losses[key]
                row[key.replace("correct", "accuracy")] = float(a / b) if int(b) else None
            history.append(row)
            print(json.dumps(row), flush=True)
    model.eval()
    output = model.generate(None, batch["n_valid_frames"], steps=min(25, args.timesteps),
                            features=features, generator=torch.Generator(device=device).manual_seed(args.seed))
    from tab_denoiser import valid_frames
    keep = valid_frames(batch["n_valid_frames"], batch["tab"].shape[1], device)[..., None].expand_as(batch["tab"])
    active = keep & (batch["tab"] > 0)
    correct = output["tab"] == batch["tab"]
    result = {"track_id": local.track_id, "window": local.window, "history": history,
              "sample_accuracy": float(correct[keep].float().mean()),
              "sample_active_accuracy": float(correct[active].float().mean()) if bool(active.any()) else None}
    destination = Path(local.output)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "overfit.json").write_text(json.dumps(result, indent=2))
    torch.save({"state": snapshot(model), "config": model.config, "timesteps": args.timesteps,
                "schedule": args.schedule, "args": vars(args)}, destination / "final.pt")
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
