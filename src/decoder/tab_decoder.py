"""Autoregressive tab decoder: Kong features -> TabTokenizer tokens, one note at a time.

    tokenizer = TabTokenizer()                                    # src/data/tokenizer.py
    encoder = TunableKongEncoder.from_checkpoint(ckpt, use_adapter=True)   # starts frozen
    model = KongTabDecoder(encoder, tokenizer)
    model.set_stage(1)                                            # or 2: LoRA on Kong
    tokens_in, targets = split_tokens(batch["tokens"], model.max_len)
    logits = model(batch["waveform"], tokens_in, batch["n_valid_frames"])
    loss = token_loss(logits, targets)
    ids = model.generate(batch["waveform"], batch["n_valid_frames"])   # greedy, grammar-masked

Architecture (one 10 s window):
  waveform [B, 160000] -> Kong + fusion adapter -> features [B, 1001, 512]
  memory  = LayerNorm(Linear 512 -> d_model + learned per-frame position)
  decoder = token embedding + learned position -> n_layers pre-LN blocks
            (causal self-attention -> cross-attention to memory -> GELU FFN)
            -> LayerNorm -> Linear d_model -> vocab
Frames >= n_valid (the zero-padded end of a clip's last window) are masked out of
cross-attention. The output layer starts at zero, so the untrained model predicts a
uniform distribution (loss = ln(vocab)).

Tokens are TabTokenizer's: BOS, [ONSET, DUR, STRING, FRET] x N, EOS, notes ordered
by onset frame then string. Generation is greedy with a KV cache, and every step is
masked by TokenGrammar, a batched torch copy of TabTokenizer.next_token_mask: the
ONSET -> DUR -> STRING -> FRET pattern, non-decreasing onsets, strictly increasing
strings within a chord, only PAD after EOS, plus no onsets in a window's padding.
So every generated sequence decodes without malformed groups.

Stage 1: Kong frozen; the adapter and the decoder train.
Stage 2: LoRA on Kong (src/kong_wrapper/lora.py); adapter and decoder train fully.

Takes the encoder and tokenizer as arguments, so this file imports neither Kong nor
the loader; it needs src/data on sys.path for tokenizer.py's special token ids.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tokenizer import BOS_ID, EOS_ID, PAD_ID

DEFAULT_CONFIG = {"d_model": 256, "n_layers": 4, "n_heads": 4, "d_ff": 1024, "dropout": 0.1,
                  "max_len": 1280, "memory_len": 1001}


# ----- model -----

class Attention(nn.Module):
    """Multi-head attention with separate query / key-value projections (for KV caching)."""

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads, self.d_head, self.p = n_heads, d_model // n_heads, dropout
        self.q = nn.Linear(d_model, d_model)
        self.kv = nn.Linear(d_model, 2 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def _heads(self, x):                                   # [B, L, D] -> [B, H, L, Dh]
        return x.view(x.shape[0], x.shape[1], self.n_heads, self.d_head).transpose(1, 2)

    def project_kv(self, x):
        k, v = self.kv(x).chunk(2, dim=-1)
        return self._heads(k), self._heads(v)

    def forward(self, x, k, v, mask=None, causal=False):
        q = self._heads(self.q(x))
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal,
                                           dropout_p=self.p if self.training else 0.0)
        return self.out(o.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1))


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.ln_self, self.ln_cross, self.ln_ff = (nn.LayerNorm(d_model) for _ in range(3))
        self.self_attn = Attention(d_model, n_heads, dropout)
        self.cross_attn = Attention(d_model, n_heads, dropout)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, cross_kv, memory_mask, cache=None):
        """cache: None (teacher forcing, causal over x) or a dict this call extends
        with the new keys / values (generation: x is the next token, or a prefix when
        the cache is empty)."""
        h = self.ln_self(x)
        k, v = self.self_attn.project_kv(h)
        if cache is not None:
            if "k" in cache:
                if x.shape[1] != 1:
                    raise ValueError("with a filled KV cache, feed one token at a time")
                k, v = torch.cat([cache["k"], k], 2), torch.cat([cache["v"], v], 2)
            cache["k"], cache["v"] = k, v
        causal = x.shape[1] > 1                    # one new token may see the whole cache
        x = x + self.drop(self.self_attn(h, k, v, causal=causal))
        x = x + self.drop(self.cross_attn(self.ln_cross(x), *cross_kv, mask=memory_mask))
        return x + self.drop(self.ff(self.ln_ff(x)))


class TabDecoder(nn.Module):
    """Transformer decoder over tab tokens, cross-attending to encoder features."""

    def __init__(self, vocab_size, memory_dim=512, d_model=256, n_layers=4, n_heads=4,
                 d_ff=1024, dropout=0.1, max_len=1280, memory_len=1001):
        super().__init__()
        self.vocab_size, self.max_len, self.memory_len = vocab_size, max_len, memory_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.mem_proj = nn.Linear(memory_dim, d_model)
        self.mem_pos = nn.Embedding(memory_len, d_model)
        self.mem_ln = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(DecoderBlock(d_model, n_heads, d_ff, dropout)
                                    for _ in range(n_layers))
        self.ln_out = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)
        self.apply(self._init)
        nn.init.zeros_(self.out.weight)              # untrained = uniform prediction
        nn.init.zeros_(self.out.bias)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def encode_memory(self, memory: Tensor, n_valid: Tensor | None = None):
        """features [B, T, memory_dim] -> (memory [B, T, d_model], mask [B, 1, 1, T] or None)."""
        T = memory.shape[1]
        if T > self.memory_len:
            raise ValueError(f"memory has {T} frames, more than memory_len {self.memory_len}")
        mem = self.mem_ln(self.mem_proj(memory) + self.mem_pos.weight[:T])
        mask = None
        if n_valid is not None:
            frames = torch.arange(T, device=memory.device)
            mask = (frames[None, :] < n_valid.to(memory.device)[:, None])[:, None, None, :]
        return mem, mask

    def cross_kv(self, mem):
        return [block.cross_attn.project_kv(mem) for block in self.blocks]

    def _decode(self, tokens, cross, memory_mask, start=0, caches=None):
        L = tokens.shape[1]
        if start + L > self.max_len:
            raise ValueError(f"positions up to {start + L} exceed max_len {self.max_len}")
        x = self.drop(self.tok_emb(tokens) + self.pos_emb.weight[start:start + L])
        for i, block in enumerate(self.blocks):
            x = block(x, cross[i], memory_mask, None if caches is None else caches[i])
        return self.out(self.ln_out(x))

    def forward(self, tokens: Tensor, memory: Tensor, n_valid: Tensor | None = None) -> Tensor:
        """Teacher forcing: tokens [B, L] (starting with BOS) -> logits [B, L, vocab]."""
        mem, mask = self.encode_memory(memory, n_valid)
        return self._decode(tokens, self.cross_kv(mem), mask)

    @torch.no_grad()
    def generate(self, memory, n_valid, tokenizer, max_len=None):
        """Greedy, grammar-masked decoding -> ids [B, <= max_len] (BOS first, PAD after EOS).

        Run in eval mode (dropout off). Stops when every row has emitted EOS, or at
        max_len tokens (then the row has no EOS; tokenizer.decode still keeps its
        complete notes).
        """
        max_len = min(max_len or self.max_len, self.max_len)
        B, device = memory.shape[0], memory.device
        mem, mask = self.encode_memory(memory, n_valid)
        cross = self.cross_kv(mem)
        caches = [{} for _ in self.blocks]
        grammar = TokenGrammar(tokenizer, B, device, n_valid)
        tokens = torch.full((B, 1), BOS_ID, dtype=torch.long, device=device)
        step_in = tokens
        for step in range(max_len - 1):
            logits = self._decode(step_in, cross, mask, start=step, caches=caches)[:, -1]
            logits = logits.float().masked_fill(~grammar.mask(), float("-inf"))
            step_in = logits.argmax(-1, keepdim=True)
            grammar.update(step_in[:, 0])
            tokens = torch.cat([tokens, step_in], 1)
            if bool(grammar.done.all()):
                break
        return tokens


class KongTabDecoder(nn.Module):
    """Kong encoder (+ fusion adapter) -> `features` -> TabDecoder."""

    def __init__(self, encoder, tokenizer, **config):
        super().__init__()
        unknown = set(config) - set(DEFAULT_CONFIG)
        if unknown:
            raise ValueError(f"unknown decoder config {sorted(unknown)}")
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.config = {**DEFAULT_CONFIG, **config}
        self.decoder = TabDecoder(tokenizer.vocab_size, memory_dim=encoder.feature_dim,
                                  **self.config)

    @property
    def max_len(self):
        return self.decoder.max_len

    def features(self, waveform):
        return self.encoder(waveform)["features"]

    def forward(self, waveform, tokens, n_valid=None):
        return self.decoder(tokens, self.features(waveform), n_valid)

    @torch.no_grad()
    def generate(self, waveform, n_valid=None, max_len=None):
        return self.decoder.generate(self.features(waveform), n_valid, self.tokenizer, max_len)

    def set_stage(self, stage, rank=8, alpha=16.0):
        """Stage 1: Kong frozen (adapter + decoder train). Stage 2: LoRA on Kong (+ adapter + decoder)."""
        if stage not in (1, 2):
            raise ValueError("stage must be 1 or 2")
        if not hasattr(self.encoder, "set_finetune"):
            raise TypeError("set_stage needs a TunableKongEncoder (src/kong_wrapper/lora.py)")
        if stage == 1:
            self.encoder.set_finetune("frozen")
        else:
            self.encoder.set_finetune("lora", rank=rank, alpha=alpha)


# ----- losses -----

def split_tokens(tokens: Tensor, max_len: int):
    """Padded token ids [B, L] -> (decoder input, next-token targets), each <= max_len long.

    Sequences longer than max_len + 1 are cut (their targets then end without EOS);
    at 10 s windows the longest GuitarSet window has 1218 tokens, under the default 1280.
    """
    tokens = tokens[:, :max_len + 1]
    return tokens[:, :-1], tokens[:, 1:]


def token_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Mean next-token cross-entropy, PAD targets ignored."""
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1),
                           ignore_index=PAD_ID)


@torch.no_grad()
def token_accuracy(logits: Tensor, targets: Tensor) -> Tensor:
    keep = targets != PAD_ID
    return (logits.argmax(-1)[keep] == targets[keep]).float().mean()


# ----- constrained decoding -----

class TokenGrammar:
    """Batched next-token masks, equal to TabTokenizer.next_token_mask row by row.

    Optional n_valid [B]: onsets must also lie before frame n_valid (no notes in a
    window's zero padding). Call mask() -> bool [B, vocab], then update(next_ids [B]).
    """

    def __init__(self, tokenizer, batch_size, device, n_valid=None):
        t = tokenizer
        ids = torch.arange(t.vocab_size, device=device)
        self.n_strings = t.n_strings
        self.onset_offset, self.string_offset = t.onset_offset, t.string_offset
        self.is_onset = (ids >= t.onset_offset) & (ids < t.dur_offset)
        self.is_dur = (ids >= t.dur_offset) & (ids < t.string_offset)
        self.is_string = (ids >= t.string_offset) & (ids < t.fret_offset)
        self.is_fret = ids >= t.fret_offset
        self.onset_value = ids - t.onset_offset
        self.string_value = ids - t.string_offset
        self.is_eos, self.is_pad = ids == EOS_ID, ids == PAD_ID

        def full(v):
            return torch.full((batch_size,), v, dtype=torch.long, device=device)
        self.pos = full(0)              # position within the note: 0 ONSET, 1 DUR, 2 STRING, 3 FRET
        self.prev_onset, self.prev_string = full(-1), full(-1)     # last complete note
        self.cur_onset, self.cur_string = full(0), full(0)         # note being written
        self.done = torch.zeros(batch_size, dtype=torch.bool, device=device)
        self.onset_end = full(t.n_onset_frames)
        if n_valid is not None:
            self.onset_end = torch.minimum(self.onset_end, n_valid.to(device).long())

    def mask(self) -> Tensor:
        has_prev = self.prev_onset >= 0
        onset_lo = torch.where(~has_prev, 0, torch.where(self.prev_string < self.n_strings - 1,
                                                         self.prev_onset, self.prev_onset + 1))
        at_onset = (self.is_onset & (self.onset_value >= onset_lo[:, None])
                    & (self.onset_value < self.onset_end[:, None])) | self.is_eos
        string_lo = torch.where(has_prev & (self.cur_onset == self.prev_onset),
                                self.prev_string + 1, 0)
        at_string = self.is_string & (self.string_value >= string_lo[:, None])
        pos = self.pos[:, None]
        m = torch.where(pos == 0, at_onset,
                        torch.where(pos == 1, self.is_dur,
                                    torch.where(pos == 2, at_string, self.is_fret)))
        return torch.where(self.done[:, None], self.is_pad, m)

    def update(self, next_ids: Tensor) -> None:
        active = ~self.done
        ends = active & (self.pos == 0) & (next_ids == EOS_ID)
        self.done = self.done | ends
        writing = active & ~ends
        self.cur_onset = torch.where(writing & (self.pos == 0),
                                     next_ids - self.onset_offset, self.cur_onset)
        self.cur_string = torch.where(writing & (self.pos == 2),
                                      next_ids - self.string_offset, self.cur_string)
        finished = writing & (self.pos == 3)
        self.prev_onset = torch.where(finished, self.cur_onset, self.prev_onset)
        self.prev_string = torch.where(finished, self.cur_string, self.prev_string)
        self.pos = torch.where(writing, (self.pos + 1) % 4, self.pos)
