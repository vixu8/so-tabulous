"""Diagnose decoder learning and free generation on one fixed GuitarSet window.

Run on the machine holding features/v1 and the Kong checkpoint. Uses the normal
decoder architecture, stage-1 adapter training, and existing generation grammar.
Frozen Kong features are cached once; the adapter remains trainable on each step.
This is a memorization diagnostic, not a generalization evaluation.
"""
import argparse
import json
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
for sub in ("decoder", "data", "kong_wrapper"):
    sys.path.insert(0, str(REPO / "src" / sub))

import torch
from loader import GuitarSetDataset, collate, load_split
from lora import TunableKongEncoder
from tab_decoder import KongTabDecoder, split_tokens, token_accuracy, token_loss
from tokenizer import EOS_ID, TYPE_NAMES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=REPO / "features/v1")
    p.add_argument("--checkpoint", type=Path, default=REPO / "src/kong_wrapper/checkpoints/kong_note_pedal.pth")
    p.add_argument("--kong_repo", type=Path, default=REPO / "src/kong_wrapper/third_party/piano_transcription")
    p.add_argument("--test_player", default="05")
    p.add_argument("--track_id", help="optional training track to inspect")
    p.add_argument("--window_index", type=int, help="index within the selected dataset")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--out_dir", type=Path, default=REPO / "runs/decoder_overfit")
    args = p.parse_args()
    if args.steps < 1 or args.eval_every < 1:
        p.error("steps and eval_every must be positive")
    if not (args.root / "manifest.csv").is_file():
        p.error(f"No manifest.csv under {args.root}; pass --root pointing to GuitarSet features")
    if (args.out_dir / "report.json").exists():
        p.error("output already contains report.json; choose a fresh --out_dir")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    rows = load_split(args.root, args.test_player.zfill(2))["train"]
    if args.track_id:
        rows = rows[rows.track_id == args.track_id]
    if rows.empty:
        p.error("no matching training clips")
    ds = GuitarSetDataset(rows, args.root, mode="windows", targets=("tokens",))
    indices = [args.window_index] if args.window_index is not None else range(len(ds))
    item = None
    for index in indices:
        candidate = ds[index]
        n_notes = (len(candidate["tokens"]) - 2) // 4
        # A full 10 s window, several notes, and no target truncation.
        if 3 <= n_notes and len(candidate["tokens"]) <= 1280 and candidate["n_valid_frames"] >= 1000:
            item = candidate
            break
    if item is None:
        p.error("no full nonempty window with >=3 notes and <=1280 tokens found")
    b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in collate([item]).items()}
    encoder = TunableKongEncoder.from_checkpoint(args.checkpoint, repo_path=args.kong_repo,
                                                use_adapter=True)
    model = KongTabDecoder(encoder, ds.tokenizer).to(device)
    model.set_stage(1)
    model.eval()
    with torch.no_grad():
        out = encoder(b["waveform"])
        pretrained = out["pretrained_features"].detach()
        branches = {k: v.detach() for k, v in out["branch_features"].items()}
    del out
    # Same features() path as stage 1, with only frozen Kong computation cached.
    def cached_features(waveform):
        return encoder.fusion_adapter(pretrained, branches)
    model.features = cached_features
    params = [*encoder.fusion_adapter.parameters(), *model.decoder.parameters()]
    groups = [{"params": [v for v in params if v.ndim >= 2], "weight_decay": 0.01},
              {"params": [v for v in params if v.ndim < 2], "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=args.lr)
    inputs, targets = split_tokens(b["tokens"], model.max_len)
    reference = item["tokens"]
    reference = [int(v) for v in reference]
    tok = model.tokenizer
    report = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "track_id": str(item["track_id"]), "start_s": float(item["start_s"]),
              "reference_tokens": reference,
              "reference_notes": tok.decode(reference, float(item["start_s"])).tolist(),
              "generation": "existing greedy grammar, EOS allowed immediately",
              "learning_rate": "constant; diagnostic has no epoch warmup", "history": []}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    for step in range(args.steps + 1):
        train_loss = None
        if step:
            model.train()
            logits = model.decoder(inputs, model.features(b["waveform"]), b["n_valid_frames"])
            loss = token_loss(logits, targets)
            train_loss = loss.item()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        if step % args.eval_every and step != args.steps:
            continue
        model.eval()
        with torch.no_grad():
            logits = model.decoder(inputs, model.features(b["waveform"]), b["n_valid_frames"])
            generated = model.generate(b["waveform"], b["n_valid_frames"])[0].tolist()
            notes, stats = tok.decode(generated, float(item["start_s"]), return_stats=True)
            row = {"step": step, "seconds": time.time() - start, "train_loss": train_loss,
                   "teacher_loss": token_loss(logits, targets).item(),
                   "teacher_accuracy": token_accuracy(logits, targets).item(),
                   "generated_tokens": generated, "generated_notes": notes.tolist(),
                   "first_generated_type": TYPE_NAMES[tok.token_type(generated[1])],
                   "immediate_eos": generated[1] == EOS_ID,
                   "exact_token_match": generated == reference,
                   "decode_stats": {k: int(v) for k, v in stats.items()}}
        report["history"].append(row)
        (args.out_dir / "report.json").write_text(json.dumps(report, indent=2))
        print(f"step {step}: loss {row['teacher_loss']:.4f}, acc {row['teacher_accuracy']:.3f}, "
              f"first {row['first_generated_type']}, notes {len(notes)}, "
              f"exact {row['exact_token_match']}", flush=True)
        if row["exact_token_match"] and row["teacher_accuracy"] >= 0.999:
            break
    torch.save({"adapter": encoder.fusion_adapter.state_dict(), "decoder": model.decoder.state_dict(),
                "config": model.config, "tokenizer": tok.to_dict(), "report": report},
               args.out_dir / "diagnostic.pt")


if __name__ == "__main__":
    main()
