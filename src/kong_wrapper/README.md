# Kong Encoder + Optional Fusion Adapter

Shared audio encoder for our guitar tablature project. This package wraps the pretrained [Kong piano transcription model](https://github.com/bytedance/piano_transcription) without editing its code. It **does not** include the guitar heads, decoder, dataset loader, or training loop.

## How the features work

Kong has four pretrained CRNN branches (**frame, onset, offset, velocity**) and its own learned refinement/fusion GRUs (`reg_onset_gru` and `frame_gru`). **We keep all of them.** The wrapper reads their hidden features using forward hooks.

**Without the adapter (`use_adapter=False`):**

$$
H_{\text{out}} = H_{\text{Kong}}
$$

Here, `H_Kong` is the **512-dimensional output of Kong's original `frame_gru`**. This is *already a pretrained, fused representation*: Kong's own onset/frame refinement still runs. We are **not** taking just one of the four CRNNs or disabling Kong's fusion.

**With the adapter (`use_adapter=True`):**

$$
H_{\text{branches}} = [H_{\text{frame}}; H_{\text{onset}}; H_{\text{offset}}; H_{\text{velocity}}]
$$
$$
\boxed{H_{\text{out}} = H_{\text{Kong}} + \sigma(a)\,A(H_{\text{branches}})}
$$

- Each branch feature is 512-dimensional; concatenation gives **2048 dimensions**.
- `A` is a new trainable MLP: **LayerNorm → Linear(2048, 256) → GELU → Dropout → Linear(256, 512)**.
- `sigmoid(a)` is a learned scalar gate that scales the correction (initially about **0.12**).
- The adapter's last linear layer is **zero-initialized**, so initially `A(...) = 0` and **`H_out = H_Kong`** exactly.

The adapter **adds** a guitar-specific correction; it never replaces Kong's original fusion. It is **not pretrained** and must be trained jointly with a guitar head or decoder. Turn it off for the pretrained-only baseline.

## Files

| File | Role |
|---|---|
| `kong_encoder.py` | Load Kong's note weights; expose features and original piano predictions; freeze/unfreeze backbone |
| `fusion_adapter.py` | Optional residual fusion adapter |
| `lora.py` | `TunableKongEncoder` (frozen / LoRA / full fine-tuning modes) and LoRA helpers |
| `../../tests/test_kong_lora.py` | LoRA tests (toy model + real checkpoint) |
| `../../tests/test_kong_wrapper.py` | Fast unit tests (mock backbone; in `tests/` at the repo root) |
| `../../tests/test_kong_checkpoint.py` | Integration test with real Kong weights (in `tests/` at the repo root; skips if setup isn't done) |
| `../../requirements/model.txt`, `../../requirements/dev.txt` | Dependencies (in `requirements/` at the repo root) |

## Setup

Run from this package's directory (Python 3.10+ recommended). For GPU use, install the appropriate [PyTorch build](https://pytorch.org/get-started/locally/) first.

```bash
python -m pip install -r ../../requirements/dev.txt
mkdir -p third_party checkpoints

git clone --depth 1 https://github.com/bytedance/piano_transcription.git \
  third_party/piano_transcription

curl -L 'https://zenodo.org/record/4034264/files/CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1' \
  -o checkpoints/kong_note_pedal.pth
```

Only the **note-model weights** are loaded from the combined note+pedal checkpoint. The original Kong repository is kept under `third_party/`.

## Usage

Imports are flat: put `src/kong_wrapper` on `sys.path` (scripts in `src/headA`, `src/headC` and `tests/` do this at the top) and `from kong_encoder import KongEncoder`. `from kong_wrapper.kong_encoder import KongEncoder` with `src/` on `sys.path` also works, but don't mix the two styles in one process (you would get two different `KongEncoder` classes).

```python
import torch
from kong_encoder import KongEncoder

encoder = KongEncoder.from_checkpoint(
    'checkpoints/kong_note_pedal.pth',
    use_adapter=False,       # True to enable the trainable correction
    backbone_frozen=True,    # Kong frozen; adapter (if present) trainable
).to('cuda')

waveform = torch.randn(2, 32000, device='cuda')  # [B, samples], 2 s at 16 kHz
outputs = encoder(waveform)

features = outputs['features']                 # [B, 201, 512]
original = outputs['pretrained_features']      # [B, 201, 512]
branches = outputs['branch_features']          # 4 tensors, each [B, 201, 512]
piano = outputs['piano']                       # 4 original outputs, each [B, 201, 88]
```

`features` is what Head A or the Transformer decoder should consume. Head C can use `piano` for Kong's original pitch predictions. `pretrained_features` always gives the **unmodified** Kong fused representation, even when the adapter is enabled. `refined_onset_features` is also available.

**Input:** mono float waveform `[B, samples]`, normalized approximately to `[-1, 1]`, sampled at **16 kHz**. Pass raw audio, **not** a spectrogram. The wrapper does not resample. At 100 frames/s with centered STFT, 2 seconds gives 201 frames and 10 seconds gives 1001 frames.

## Training and tests

To use the adapter, set `use_adapter=True` and optimize it **with the downstream guitar head**. No separate adapter training is required:

```python
optimizer = torch.optim.AdamW(
    list(encoder.fusion_adapter.parameters()) + list(head.parameters()),
    lr=1e-4,
)
# Train using your head's tab loss on head(encoder(audio)['features']).
```

`backbone_frozen=True` keeps Kong's weights and batch-norm statistics fixed while the adapter remains trainable.

## Fine-tuning (frozen / LoRA / full)

Use `TunableKongEncoder` from `lora.py` for any run that updates Kong. It is a `KongEncoder` with three modes:

```python
from lora import TunableKongEncoder, lora_state_dict, merge_lora

encoder = TunableKongEncoder.from_checkpoint('checkpoints/kong_note_pedal.pth')  # "frozen"
encoder.set_finetune('lora', rank=8, alpha=16)   # or 'frozen' / 'full'
encoder.to('cuda').train()

optimizer = torch.optim.AdamW(
    [*encoder.trainable_parameters(), *head.parameters()],  # + adapter if used
    lr=1e-4,
)
# ... train; save only the LoRA weights (~2 MB) next to the head:
torch.save(lora_state_dict(encoder.note_model), 'lora.pt')
```

| Mode | What trains in Kong |
|---|---|
| `frozen` | nothing (`no_grad` forward) |
| `lora` | low-rank `W + (alpha/rank)·B·A` on all 10 Linear weights and every GRU `weight_ih` / `weight_hh` (biases untouched): 486,400 weights at rank 8, about 2% of Kong |
| `full` | every Kong weight (call `merge_lora(encoder.note_model)` first if LoRA is attached) |

- **LoRA starts as the identity:** B is zero, so a fresh LoRA leaves every output unchanged.
- **In `lora` and `full`, Kong follows `train()` / `eval()`**, so dropout is on while training, as in Kong's own training. Batch-norm always keeps its pretrained statistics, so a saved LoRA file (A and B only) fully describes the model, and small batches don't add noise. The adapter, if any, follows `train()` / `eval()` as usual.
- **Kong's backprop bug is fixed at load time:** `models.py:149` uses an in-place dropout that breaks `backward()` in train mode. `kong_encoder.py` gives Kong's module an out-of-place dropout instead. Kong's file isn't edited and outputs are unchanged. This also fixes plain `KongEncoder` with `set_backbone_frozen(False)`.
- **Change modes with `set_finetune`**, not `set_backbone_frozen`.
- **Heads, the decoder and the fusion adapter train fully** on top of a LoRA'd Kong. LoRA is only for the large pretrained backbone.
- **Other helpers:** `load_lora_state_dict` restores saved LoRA weights; `merge_lora` folds them into plain weights for inference.
- **Implementation:** LoRA uses `torch.nn.utils.parametrize`, not `peft`, because `peft` can't wrap `nn.GRU`.

Laptop timing (M3 Max, MPS, 10 s crops, one training step):

| Batch | LoRA | Full fine-tuning |
|---|---|---|
| 4 | 5.0 s, 3.8 GB | 5.9 s, 11.9 GB |
| 8 | 5.5 s, 6.5 GB | 8.2 s, 22.6 GB |

(train mode with dropout on)

Run these from the **repo root** (not this directory):

```bash
# Unit tests (no checkpoint required)
python -m pytest -q tests/test_kong_wrapper.py

# Real checkpoint test (uses checkpoints/ and third_party/ in this folder; runs on CPU)
python -m pytest -q tests/test_kong_checkpoint.py
```

The checkpoint test skips, with the reason, if the checkpoint, the cloned Kong repo, or a package from `requirements/model.txt` is missing; run pytest with `-rs` to see why.

The offline unit tests passed when this package was created; the real-checkpoint integration test still needs to run in your environment. Because feature capture uses hooks, **do not run concurrent forwards on one wrapper instance** or use `nn.DataParallel`.
