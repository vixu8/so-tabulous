# so-tabulous
Generative AI final project

Head A stage 1 requires the trained TabCNN checkpoint for the same held-out
player (the TabCNN trainer saves fold 5 under `5/weights.pt` for player `05`):

```bash
python src/headA/train_head_a.py --stage 1 --test_player 05 --tabcnn_checkpoint checkpoints/tabcnn/5/weights.pt --device cuda --resume
python src/headA/train_head_a.py --stage 2 --test_player 05 --device cuda --resume
```

Both TabCNN classifier layers initialize from the checkpoint and remain
trainable. A fresh linear projection maps Kong's 512 features to the pretrained
classifier's input size. Stage 2 restores the projection and classifier from
stage 1. Use a new output directory for this experiment if previous randomly
initialized Head A runs exist; `--resume` restores those previous weights.

## Pretraining ablation: decoder1 on a randomly initialized Kong

Same structure as decoder1, but Kong and the decoder both start from random
weights (no MAESTRO checkpoint) and train together in one stage on each
GuitarSet fold, with the tab loss only. Kong's `.detach()` calls are removed
for this run (same outputs), so the loss reaches all of Kong's branches; the
STFT / log-mel frontend stays fixed and BatchNorm trains. Defaults: 75 epochs
(= decoder1's 50 + 25), peak LR 3e-4 for Kong and decoder, warmup 2 epochs then
cosine, batch 8.

```bash
python scripts/run_folds.py --cmd "python src/decoder/train_decoder.py --arch decoder1 --kong_init random --stage 1 --test_player {test} --amp"
python src/decoder/summarize_decoder.py --runs runs/decoder1_randkong --stage scratch
```

Runs go to `runs/decoder1_randkong/scratch_test{T}/`; `final.pt` stores the
trained Kong. On a 16 GB GPU, if batch 8 runs out of memory, add
`--batch_size 4 --grad_accum 2` (same effective batch). Without
`--kong_init random` nothing changes.
