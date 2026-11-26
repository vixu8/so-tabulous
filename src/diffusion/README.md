# Guitar tablature discrete diffusion baseline

This package implements the requested small D3RM-inspired **absorbing-mask
baseline**. It uses the existing GuitarSet loader, Kong wrapper, player splits,
and shared metrics. The AR decoder is independent and unchanged.

Read alongside [D3RM's paper](https://arxiv.org/abs/2501.05068),
[decoder](https://github.com/hanshounsu/d3rm/blob/main/transcription/decoder.py),
[AdaLN/attention](https://github.com/hanshounsu/d3rm/blob/main/transcription/diffusion/natten2d_diffusion.py),
and [categorical diffusion implementation](https://github.com/hanshounsu/d3rm/blob/main/transcription/diffusion/trainer.py).
The architecture draws on its pitchwise recurrence, neighborhood attention,
timestep conditioning, and reuse of acoustic features. This implementation
uses stringwise recurrence, concatenated Kong conditioning, separate onset
prediction, and masked reconstruction rather than D3RM's substitution process,
reverse KL objective, or training/inference transition mismatch. Revealed
tokens remain fixed in this baseline; it does not refine their fret assignments.

## Shapes and masking

| Operation | Output |
|---|---|
| Frozen Kong, no adapter by default | `[B,T,512]` |
| Learned acoustic projection | `[B,T,6,64]` |
| State embedding (22 inputs) + string embedding | `[B,T,6,16]` |
| Shared bidirectional LSTM, packed by valid length | `[B*6,T,64]` |
| Concatenate recurrence + acoustics, linear projection | `[B,T,6,64]` |
| Four AdaLN residual blocks | `[B,T,6,64]` |
| Fret logits (21 clean classes) | `[B,T,6,21]` |
| Onset logits | `[B,T,6]` |

`T=1001` for 160000-sample audio. Each block has 4-head temporal attention
over offsets `[-2,-1,0,1,2] * dilation`, followed by 4-head attention over six
strings, then a 256-unit GELU FFN. Dilations are `[1,2,4,8]`. Temporal keys and
values are gathered locally: scores have shape `[B*6,T,4,5]`, never `[T,T]`.
Relative biases distinguish the five signed temporal offsets.

Clean states are silence 0 and fret+1 classes 1..20, low E through high e.
MASK=21 exists only in noisy inputs. Padding is silence in outputs and ignored
in corruption, packed recurrence, local keys, losses, and metrics. Frame 1000
is also excluded, matching shared `frame_metrics` and preventing duplicated
window endpoints. No duration tokenizer, BOS, or EOS is used.

The loader labels offsets inclusively; shared `tab_grid_to_notes` uses
half-open runs. Neither shared convention is changed. Notes use onset
probabilities to split repeated attacks, end on silence/fret changes, and are
clipped at `(n_valid_frames-1)/100` seconds (at most 10 seconds per window).
Notes crossing windows are cut at the boundary, as in AR evaluation.
Frame scores use generated categorical grids directly; they are not redrawn
from note events (which could add a frame at an exclusive end time).

## Process and loss

Timesteps are integers `0..N`, with `N=100` by default. The default schedule is
`alpha_bar[t] = 1-t/N`; `--schedule cosine` uses `cos(pi*t/(2*N))**2`.
Endpoints are exactly 1 and 0. The one-step clean survival probability is
`alpha[t] = alpha_bar[t]/alpha_bar[t-1]`; once masked, a state stays masked.
Training samples `t` independently and uniformly, **including t=0 and t=N**,
then corrupts directly using `q(y_t|y_0)` without a forward loop.

For a reverse jump `t -> s`, a masked position reveals with probability
`(alpha_bar[s]-alpha_bar[t])/(1-alpha_bar[t])`, sampling its clean class from
the denoiser softmax. Unrevealed positions remain MASK. Rounded evenly spaced
integer inference timesteps include both endpoints; the final jump reveals
every remaining valid MASK. Requested steps must be between 1 and N.

Loss is weighted clean-state CE plus `lambda_onset * BCEWithLogits`, default
lambda 0.5. Masked/unmasked CE weights and onset positive weight are configurable.
Logs include overall, corrupted-position, active-state, and silence accuracy;
an accuracy with no eligible positions is JSON `null`, not a misleading zero.
Inference computes acoustics once per window, reuses them for all jumps, and
runs an additional t=0 clean-grid pass for onsets. The one-pass comparison uses
argmax at t=N and the same final onset pass. Sampling is categorical, not argmax.

## Run one fold

Install the repository's `requirements/dev.txt` dependencies and the appropriate
CUDA PyTorch build, and set up Kong per `src/kong_wrapper/README.md`.

```bash
python src/diffusion/train_diffusion.py --test_player 05 --amp
python src/diffusion/train_diffusion.py --test_player 05 --amp --resume
python src/diffusion/eval_diffusion.py --run_dir runs/diffusion/stage1_test05
python src/diffusion/summarize_diffusion.py --folds 05
```

Defaults: batch 8, 25 epochs, AdamW at 3e-4, matrix weight decay 0.01, gradient
clip 1, two warmup epochs then cosine decay, four loader workers, two random
crops per clip per epoch. BF16 is enabled when the CUDA device supports it;
`--no-amp` disables it, and CPU/unsupported devices use float32.
`--use_adapter` trains the existing residual fusion adapter while keeping Kong
frozen; leave it off for the current AR baseline's encoder configuration.
Architecture and loss settings are exposed in `--help`.

### Stage 2: adapt Kong with LoRA

Stage 1 remains the default (`--stage 1`): train the denoiser with frozen Kong.
Stage 2 loads stage 1's `final.pt`, continues training the full denoiser (and
fusion adapter if enabled), and trains only LoRA weights in Kong. Defaults:
25 additional epochs, LR 1e-4, rank 8, alpha 16. Override these with `--epochs`,
`--lr`, `--rank`, and `--alpha`. Kong's original weights and BatchNorm statistics
stay fixed. Use the same pretrained Kong checkpoint/source and GuitarSet root.

```bash
python src/diffusion/train_diffusion.py --stage 1 --test_player 05 --device cuda --amp
python src/diffusion/train_diffusion.py --stage 2 --test_player 05 --device cuda --amp
python src/diffusion/eval_diffusion.py --run_dir runs/diffusion/stage2_test05
python src/diffusion/summarize_diffusion.py --tag stage2 --folds 05
```

Stage 2 reads `runs/diffusion/stage1_test<player>/final.pt` by default;
`--stage1_dir <directory>` selects a custom run. The held-out player must match.
The denoiser architecture, adapter setting, diffusion schedule and timestep
count are inherited automatically. Outputs default to
`runs/diffusion/stage2_test<player>/`. Both `last.pt` and `final.pt` include
LoRA A/B weights; standalone evaluation restores them automatically. Older
stage 1 checkpoints remain loadable.

Resume stage 2 with the same command plus `--resume`. It restores its own
checkpoint without needing the original stage 1 directory. Feature caching
is forbidden in stage 2 because Kong's features change during training.

Run both stages sequentially for all six folds:

```bash
python scripts/run_folds.py \
  --cmd "python src/diffusion/train_diffusion.py --stage 1 --test_player {test} --device cuda --amp" \
  --cmd "python src/diffusion/train_diffusion.py --stage 2 --test_player {test} --device cuda --amp"
python src/diffusion/summarize_diffusion.py --tag stage2 --folds 00 01 02 03 04 05
```

Splits are the loader's current convention: five training players, one test
player, **no validation player**. Keep the last fixed epoch; do not tune epochs,
thresholds, sampling temperature, or step counts on the test player. Defaults
report one-pass and diffusion with 10, 25, 50, and 100 steps. `--skip_eval` only
trains/saves. Use `--inference_steps 10 25` to predeclare fewer comparisons.

Outputs under `runs/diffusion/<tag>_test<player>/`: `train.log`, `history.jsonl`,
`last.pt`, `final.pt`, `metrics.json`, `eval.json`, and per-method note NPZ files.
Metrics cover frame pitch/tab P/R/F and TDR, onset and offset-aware pitch/tab
note scores, playability, and runtime including audio feature extraction.
The summarizer computes fold means and sample standard deviations and rejects
mixed decoding settings. It requires explicit fold names; it launches no jobs.

`--resume` restores decoder/adapter, optimizer, scheduler, Python/NumPy/Torch/CUDA
RNGs and the DataLoader generator at epoch boundaries. Training settings must
match, including total epochs, so the learning-rate schedule is preserved.
Frozen Kong weights remain external to checkpoints, as in AR: retain the same
Kong weights/source when restoring, or override their file locations at eval.
Logs include parameter counts, epoch and mean step seconds, and peak CUDA memory.

Optional `--feature_cache runs/diffusion/features` caches **exact audio crops**,
keyed by waveform bytes within an encoder-state fingerprint. It is only allowed
when all encoder outputs are frozen, and rejects a trainable fusion adapter.
Random crops usually have few hits; explicitly choose `--crop_mode windows` to
reuse aligned fixed windows across epochs. The same loader still constructs
the matching tab/onset targets. No cached features are used with LoRA or any
trainable encoder. SynthTab and full D3RM diffusion are deferred.

## Sanity checks

```bash
python -m pytest tests/test_tab_diffusion.py tests/test_train_diffusion.py tests/test_tab_metrics.py -q
python scripts/overfit_diffusion_window.py --track_id <training-track-id> --device cuda --amp
```

The overfit utility takes one selected non-overlapping window from a **training**
clip, freezes/caches its acoustic features, and logs loss and active/corrupted
accuracy. `--steps`, `--window`, and trainer settings such as `--lr` are supported.
It saves to `runs/diffusion/overfit` by default and reports sampled accuracy.
Inspect active accuracy and repeated-note behavior; silence accuracy alone does
not establish successful overfitting.

Tests check schedules/posteriors statistically, locality against explicit
neighborhood means, padding invariance, shapes at T=1001, frozen/adapter
gradients, synthetic single-clip overfitting, cache alignment, seeded checkpoint
replay, and exact interrupted/resumed CPU training. A real Kong smoke test runs
when its source/weights are available. CUDA tests skip without a CUDA device;
real GuitarSet overfitting and A10G memory/throughput still require that environment.
