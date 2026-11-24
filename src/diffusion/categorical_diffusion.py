"""Absorbing MASK diffusion with exact skipped-step posterior transitions.

alpha_bar[t] = 1 - t/N (linear default), t=0..N. A clean token survives
with probability alpha_bar[t]; MASK is absorbing. No substitutions or KL
objective are used. This deliberately differs from full D3RM.
"""

import torch
from torch import nn
from torch.nn import functional as F

try:
    from .tab_denoiser import MASK, N_CLASSES, valid_frames
except ImportError:
    from tab_denoiser import MASK, N_CLASSES, valid_frames


class CategoricalDiffusion(nn.Module):
    def __init__(self, timesteps=100, schedule="linear"):
        super().__init__()
        if timesteps < 1 or schedule not in ("linear", "cosine"):
            raise ValueError("timesteps must be positive; schedule must be linear or cosine")
        self.timesteps, self.schedule = timesteps, schedule
        progress = torch.linspace(0, 1, timesteps + 1)
        survival = 1 - progress if schedule == "linear" else (progress * torch.pi / 2).cos().square()
        survival[0], survival[-1] = 1, 0
        self.register_buffer("alpha_bar", survival)

    def sample_timesteps(self, batch, device, generator=None):
        # Include clean inputs for the final onset pass as well as fully masked
        # inputs; every example samples independently, including both endpoints.
        return torch.randint(self.timesteps + 1, (batch,), device=device, generator=generator)

    def q_sample(self, clean, t, n_valid, generator=None):
        if t.shape != (clean.shape[0],) or bool(((t < 0) | (t > self.timesteps)).any()):
            raise ValueError("invalid diffusion timesteps")
        keep = valid_frames(n_valid, clean.shape[1], clean.device)[..., None].expand_as(clean)
        if bool(((clean[keep] < 0) | (clean[keep] >= N_CLASSES)).any()):
            raise ValueError("clean labels must be in 0..20")
        masked = torch.rand(clean.shape, device=clean.device, generator=generator) >= self.alpha_bar[t, None, None]
        return clean.masked_fill(masked & keep, MASK).masked_fill(~keep, 0)

    def inference_schedule(self, steps):
        if not 1 <= steps <= self.timesteps:
            raise ValueError("inference steps must be in 1..training timesteps")
        return torch.linspace(self.timesteps, 0, steps + 1).round().long().tolist()

    def reveal_probability(self, t, s):
        if not 0 <= s < t <= self.timesteps:
            raise ValueError("reverse transition requires 0 <= s < t <= N")
        return ((self.alpha_bar[s] - self.alpha_bar[t]) / (1 - self.alpha_bar[t])).clamp(0, 1)

    def reverse_probabilities(self, noisy, logits, t, s):
        """Normalized [B,T,6,22] posterior, useful for diagnostics/tests."""
        reveal = self.reveal_probability(t, s)
        clean = logits.float().softmax(-1) * reveal
        masked_posterior = torch.cat((clean, torch.ones_like(clean[..., :1]) * (1 - reveal)), -1)
        unchanged = F.one_hot(noisy, N_CLASSES + 1).float()
        return torch.where((noisy == MASK)[..., None], masked_posterior, unchanged)

    def reverse_step(self, noisy, logits, t, s, n_valid, generator=None, temperature=1.0):
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        reveal = self.reveal_probability(t, s)
        selected = torch.rand(noisy.shape, device=noisy.device, generator=generator) < reveal
        selected &= noisy == MASK
        keep = valid_frames(n_valid, noisy.shape[1], noisy.device)[..., None]
        selected &= keep
        out = noisy.clone()
        # Sample only newly revealed positions, retaining previously revealed
        # classes exactly as required by an absorbing-state posterior.
        if bool(selected.any()):
            probs = (logits[selected].float() / temperature).softmax(-1)
            out[selected] = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
        return out.masked_fill(~keep, 0)


def diffusion_loss(outputs, clean, noisy, onsets, n_valid, lambda_onset=0.5,
                   masked_weight=1.0, unmasked_weight=1.0, onset_pos_weight=1.0):
    if min(lambda_onset, masked_weight, unmasked_weight) < 0 or onset_pos_weight <= 0:
        raise ValueError("loss weights must be nonnegative; onset_pos_weight must be positive")
    keep = valid_frames(n_valid, clean.shape[1], clean.device)[..., None].expand_as(clean)
    # Replace ignored targets before CE/BCE, so even sentinel padding labels do
    # not cause an out-of-range error or enter the loss through 0 * NaN.
    target = clean.masked_fill(~keep, 0)
    logits, onset_logits = outputs["fret_logits"].float(), outputs["onset_logits"].float()
    ce = F.cross_entropy(logits.movedim(-1, 1), target, reduction="none")
    corrupted = (noisy == MASK) & keep
    weights = torch.where(corrupted, masked_weight, unmasked_weight) * keep
    fret_loss = (ce * weights).sum() / weights.sum().clamp(min=torch.finfo(ce.dtype).eps)
    bce = F.binary_cross_entropy_with_logits(
        onset_logits, onsets.masked_fill(~keep, 0).float(), reduction="none",
        pos_weight=onset_logits.new_tensor(onset_pos_weight))
    onset_loss = bce[keep].sum() / keep.sum().clamp(min=1)
    prediction = logits.argmax(-1)
    correct = prediction == target
    active = keep & (target > 0)
    silent = keep & (target == 0)
    def counts(mask):
        return (correct & mask).sum().detach(), mask.sum().detach()
    return {"loss": fret_loss + lambda_onset * onset_loss, "fret_loss": fret_loss,
            "onset_loss": onset_loss, "correct": counts(keep),
            "corrupted_correct": counts(corrupted), "active_correct": counts(active),
            "silence_correct": counts(silent)}
