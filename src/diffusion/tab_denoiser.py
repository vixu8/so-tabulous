"""Stringwise BiLSTM and local temporal / global string denoising attention.

Inspired by D3RM's pitchwise LSTM, neighborhood attention and AdaLN; this is
an independent guitar implementation without NATTEN or piano-state tokens.
"""

import math

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

N_STRINGS, N_CLASSES, MASK = 6, 21, 21


def valid_frames(n_valid, frames, device):
    """Exclude padding and the shared frame-1000 window endpoint."""
    lengths = torch.as_tensor(n_valid, device=device, dtype=torch.long)
    if lengths.ndim != 1 or bool((lengths < 0).any()):
        raise ValueError("n_valid must be a nonnegative vector")
    return torch.arange(frames, device=device)[None] < lengths.clamp(max=1000)[:, None]


class TimestepEmbedding(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        if dim < 4 or dim % 2:
            raise ValueError("timestep dimension must be even and >= 4")
        self.register_buffer("frequencies", torch.exp(
            -math.log(10000) * torch.arange(dim // 2) / (dim // 2 - 1)))
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))

    def forward(self, t):
        angles = t.float()[:, None] * self.frequencies[None]
        return self.mlp(torch.cat((angles.sin(), angles.cos()), -1))


class AdaLN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.affine = nn.Linear(dim, 2 * dim)

    def forward(self, x, timestep):
        scale, shift = self.affine(timestep)[:, None, None].chunk(2, -1)
        return self.norm(x) * (1 + scale) + shift


class LocalTemporalAttention(nn.Module):
    """Gather K dilated keys per query; memory and work are O(B*T*K*D).

    Input [B*6,T,D]. Scores are [B*6,T,H,K], never [T,T]. Offsets keep
    separate learned biases for preceding and succeeding frames.
    """
    def __init__(self, dim=64, heads=4, window=5, dilation=1, dropout=0.1):
        super().__init__()
        if dim % heads or window < 1 or window % 2 != 1 or dilation < 1:
            raise ValueError("invalid attention dimension, window or dilation")
        self.heads, self.head_dim, self.window = heads, dim // heads, window
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.bias = nn.Parameter(torch.zeros(heads, window))
        self.register_buffer("offsets", torch.arange(-(window // 2), window // 2 + 1) * dilation)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, keep):
        batch, frames, dim = x.shape
        q, k, v = self.qkv(x).reshape(batch, frames, 3, self.heads, self.head_dim).unbind(2)
        positions = torch.arange(frames, device=x.device)[:, None] + self.offsets[None]
        inside = (positions >= 0) & (positions < frames)
        indices = positions.clamp(0, frames - 1)
        keys, values = k[:, indices], v[:, indices]  # [B*6,T,K,H,Dh]
        scores = (q[:, :, None] * keys).sum(-1).transpose(-1, -2) / math.sqrt(self.head_dim)
        scores = scores + self.bias[None, None]
        allowed = inside[None] & keep[:, indices]
        # Padded queries have no valid keys in some neighborhoods. A dummy self
        # key keeps softmax finite; their output is zeroed immediately afterward.
        allowed = allowed | (~keep[:, :, None] & (self.offsets == 0)[None, None])
        weights = self.drop(scores.float().masked_fill(~allowed[:, :, None], -torch.inf).softmax(-1))
        out = (weights.transpose(-1, -2)[..., None].to(values.dtype) * values).sum(2)
        return self.out(out.reshape(batch, frames, dim)).masked_fill(~keep[..., None], 0)


class DenoiserBlock(nn.Module):
    def __init__(self, dim=64, heads=4, window=5, dilation=1, dropout=0.1, d_ff=256):
        super().__init__()
        self.norms = nn.ModuleList(AdaLN(dim) for _ in range(3))
        self.temporal = LocalTemporalAttention(dim, heads, window, dilation, dropout)
        self.strings = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(dim, d_ff), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(d_ff, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, timestep, keep):
        batch, frames, strings, dim = x.shape
        h = self.norms[0](x, timestep).permute(0, 2, 1, 3).reshape(batch * strings, frames, dim)
        string_keep = keep.repeat_interleave(strings, 0)
        h = self.temporal(h, string_keep).reshape(batch, strings, frames, dim).transpose(1, 2)
        x = x + self.drop(h)
        h = self.norms[1](x, timestep).reshape(batch * frames, strings, dim)
        # No temporal mixing here; padded frames cannot affect valid frames.
        h = self.strings(h, h, h, need_weights=False)[0].reshape_as(x)
        x = x + self.drop(h)
        x = x + self.drop(self.ff(self.norms[2](x, timestep)))
        return x.masked_fill(~keep[:, :, None, None], 0)


DEFAULT_CONFIG = dict(memory_dim=512, hidden_dim=64, embed_dim=16, lstm_hidden=32,
                      heads=4, window=5, dilations=(1, 2, 4, 8), d_ff=256, dropout=0.1)


class TabDenoiser(nn.Module):
    def __init__(self, memory_dim=512, hidden_dim=64, embed_dim=16, lstm_hidden=32,
                 heads=4, window=5, dilations=(1, 2, 4, 8), d_ff=256, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.acoustic = nn.Linear(memory_dim, N_STRINGS * hidden_dim)
        self.state_embedding = nn.Embedding(N_CLASSES + 1, embed_dim)
        self.string_embedding = nn.Embedding(N_STRINGS, embed_dim)
        self.lstm = nn.LSTM(embed_dim, lstm_hidden, num_layers=1, bidirectional=True, batch_first=True)
        self.condition = nn.Linear(2 * lstm_hidden + hidden_dim, hidden_dim)
        self.time = TimestepEmbedding(hidden_dim)
        self.blocks = nn.ModuleList(DenoiserBlock(hidden_dim, heads, window, d, dropout, d_ff)
                                    for d in dilations)
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.fret = nn.Linear(hidden_dim, N_CLASSES)
        self.onset = nn.Linear(hidden_dim, 1)

    def project_acoustics(self, features):
        return self.acoustic(features).reshape(*features.shape[:2], N_STRINGS, self.hidden_dim)

    def forward(self, noisy, acoustics, t, n_valid):
        batch, frames, strings = noisy.shape
        if strings != N_STRINGS or acoustics.shape != (batch, frames, strings, self.hidden_dim):
            raise ValueError("expected noisy [B,T,6] and projected acoustics [B,T,6,H]")
        if t.shape != (batch,):
            raise ValueError("expected one timestep per example")
        keep = valid_frames(n_valid, frames, noisy.device)
        noisy = noisy.masked_fill(~keep[..., None], 0)
        embedded = self.state_embedding(noisy) + self.string_embedding.weight[None, None]
        sequences = embedded.transpose(1, 2).reshape(batch * strings, frames, -1)
        lengths = keep.sum(1).repeat_interleave(strings).cpu()
        packed = pack_padded_sequence(sequences, lengths.clamp(min=1), batch_first=True,
                                      enforce_sorted=False)
        recurrent, _ = self.lstm(packed)
        recurrent, _ = pad_packed_sequence(recurrent, batch_first=True, total_length=frames)
        recurrent = recurrent.reshape(batch, strings, frames, -1).transpose(1, 2)
        x = self.condition(torch.cat((recurrent, acoustics.masked_fill(~keep[:, :, None, None], 0)), -1))
        x = x.masked_fill(~keep[:, :, None, None], 0)
        timestep = self.time(t)
        for block in self.blocks:
            x = block(x, timestep, keep)
        x = self.final_norm(x)
        return {"fret_logits": self.fret(x).masked_fill(~keep[:, :, None, None], 0),
                "onset_logits": self.onset(x).squeeze(-1).masked_fill(~keep[:, :, None], 0)}
