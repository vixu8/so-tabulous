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
| `tests/test_kong_wrapper.py` | Fast unit tests (mock backbone) |
| `scripts/smoke_test_checkpoint.py` | Integration test with real Kong weights |
| `requirements.txt`, `requirements-dev.txt` | Dependencies |

## Setup

Run from this package's directory (Python 3.10+ recommended). For GPU use, install the appropriate [PyTorch build](https://pytorch.org/get-started/locally/) first.

```bash
python -m pip install -r requirements-dev.txt
mkdir -p third_party checkpoints

git clone --depth 1 https://github.com/bytedance/piano_transcription.git \
  third_party/piano_transcription

curl -L 'https://zenodo.org/record/4034264/files/CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1' \
  -o checkpoints/kong_note_pedal.pth
```

Only the **note-model weights** are loaded from the combined note+pedal checkpoint. The original Kong repository is kept under `third_party/`.

## Usage

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

`backbone_frozen=True` keeps Kong's weights and batch-norm statistics fixed while the adapter remains trainable. For full fine-tuning, use `encoder.set_backbone_frozen(False)`. LoRA is **not** implemented in this package.

```bash
# Unit tests (no checkpoint required)
python -m pytest -q tests/

# Real checkpoint test
python scripts/smoke_test_checkpoint.py \
  --checkpoint checkpoints/kong_note_pedal.pth \
  --repo third_party/piano_transcription \
  --device cpu
```

The offline unit tests passed when this package was created; the real-checkpoint integration test still needs to run in your environment. Because feature capture uses hooks, **do not run concurrent forwards on one wrapper instance** or use `nn.DataParallel`.
