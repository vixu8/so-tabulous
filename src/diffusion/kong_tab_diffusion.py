"""Frozen Kong acoustic encoder + independently trained categorical denoiser."""

import torch
from torch import nn

try:
    from .categorical_diffusion import CategoricalDiffusion, diffusion_loss
    from .tab_denoiser import DEFAULT_CONFIG, MASK, TabDenoiser, valid_frames
except ImportError:
    from categorical_diffusion import CategoricalDiffusion, diffusion_loss
    from tab_denoiser import DEFAULT_CONFIG, MASK, TabDenoiser, valid_frames


class KongTabDiffusion(nn.Module):
    def __init__(self, encoder, timesteps=100, schedule="linear", **config):
        super().__init__()
        self.encoder = encoder
        if hasattr(encoder, "set_finetune"):
            encoder.set_finetune("frozen")
        elif hasattr(encoder, "set_backbone_frozen"):
            encoder.set_backbone_frozen(True)
        else:
            encoder.requires_grad_(False)
            encoder.eval()
        self.config = {**DEFAULT_CONFIG, "memory_dim": encoder.feature_dim, **config}
        self.denoiser = TabDenoiser(**self.config)
        self.diffusion = CategoricalDiffusion(timesteps, schedule)

    def train(self, mode=True):
        super().train(mode)
        if not any(p.requires_grad for p in self.encoder.parameters()):
            self.encoder.eval()
        return self

    def features(self, waveform):
        return self.encoder(waveform)["features"]

    def forward(self, waveform, noisy, t, n_valid, features=None):
        features = self.features(waveform) if features is None else features
        return self.denoiser(noisy, self.denoiser.project_acoustics(features), t, n_valid)

    def training_loss(self, batch, features=None, generator=None, **weights):
        clean, n_valid = batch["tab"], batch["n_valid_frames"]
        t = self.diffusion.sample_timesteps(clean.shape[0], clean.device, generator)
        noisy = self.diffusion.q_sample(clean, t, n_valid, generator)
        outputs = self(batch.get("waveform"), noisy, t, n_valid, features)
        return diffusion_loss(outputs, clean, noisy, batch["tab_onset"], n_valid, **weights)

    @torch.no_grad()
    def generate(self, waveform, n_valid, steps=25, generator=None, temperature=1.0,
                 method="diffusion", features=None):
        """Eval-mode sampling; raw encoder features and projection computed once.

        one_pass uses argmax of the highest-timestep prediction. Both methods
        compute onsets with a separate t=0 pass on their resulting clean grid.
        """
        if self.training:
            raise RuntimeError("call model.eval() before generation")
        if method not in ("diffusion", "one_pass"):
            raise ValueError("method must be diffusion or one_pass")
        features = self.features(waveform) if features is None else features
        acoustics = self.denoiser.project_acoustics(features)
        batch, frames = features.shape[:2]
        keep = valid_frames(n_valid, frames, features.device)[..., None]
        noisy = torch.full((batch, frames, 6), MASK, device=features.device, dtype=torch.long)
        noisy = noisy.masked_fill(~keep, 0)
        schedule = self.diffusion.inference_schedule(steps if method == "diffusion" else 1)
        for t, s in zip(schedule[:-1], schedule[1:]):
            times = torch.full((batch,), t, device=features.device, dtype=torch.long)
            outputs = self.denoiser(noisy, acoustics, times, n_valid)
            if method == "one_pass":
                noisy = outputs["fret_logits"].argmax(-1).masked_fill(~keep, 0)
            else:
                noisy = self.diffusion.reverse_step(noisy, outputs["fret_logits"], t, s,
                                                     n_valid, generator, temperature)
        clean_outputs = self.denoiser(noisy, acoustics, torch.zeros(batch, device=features.device,
                                                                    dtype=torch.long), n_valid)
        return {"tab": noisy, "onset_probs": clean_outputs["onset_logits"].sigmoid().masked_fill(~keep, 0)}
