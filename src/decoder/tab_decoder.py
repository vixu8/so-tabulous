"""Autoregressive tab decoder: Kong features -> TabTokenizer tokens, one note at a time.

    tokenizer = TabTokenizer()                                    # src/data/tokenizer.py
    encoder = TunableKongEncoder.from_checkpoint(ckpt)          # no adapter; starts frozen
    model = KongTabDecoder(encoder, tokenizer, **ARCH_CONFIGS["decoder1"])   # or "decoder2"
    model.set_stage(1)                                            # or 2: LoRA on Kong
    tokens_in, targets = split_tokens(batch["tokens"], model.max_len)
    logits = model(batch["waveform"], tokens_in, batch["n_valid_frames"])
    loss = token_loss(logits, targets)
    ids = model.generate(batch["waveform"], batch["n_valid_frames"])                 # greedy
    ids = model.generate(batch["waveform"], batch["n_valid_frames"], method="beam")  # beam 4

Architecture (one 10 s window):
  waveform [B, 160000] -> Kong (unchanged) -> features [B, 1001, 512]
            (Kong's final GRU, frame_gru; no fusion adapter, per the midway plan)
  memory  = LayerNorm(Linear 512 -> d_model + learned per-frame position)
  decoder = token embedding + learned position -> n_layers pre-LN blocks
            (causal self-attention -> cross-attention to memory -> FFN)
            -> LayerNorm -> Linear d_model -> vocab
Frames >= n_valid (the zero-padded end of a clip's last window) are masked out of
cross-attention. The output layer starts at zero, so the untrained model predicts a
uniform distribution (loss = ln(vocab)).

Two architectures (ARCH_CONFIGS), both d_model 256, d_ff 1024, no fusion adapter:
  decoder1  the midway-plan decoder (GuitarSet folds): 4 layers, 4 heads, GELU FFN
  decoder2  matches TART's transformer, AudioFret (arXiv 2609.11904, sec. 5.1:
            d_model 256, 6 layers, 8 heads, gated-GELU FFN); trained on SynthTab
            only, tested zero-shot on GuitarSet. Everything TART leaves unstated
            (norm, positions, biases) stays as in decoder1.
Gated-GELU FFN (T5 v1.1): Linear d_model -> 2 d_ff, gelu(a) * b, Linear d_ff -> d_model.

Tokens are TabTokenizer's: BOS, [ONSET, DUR, STRING, FRET] x N, EOS, notes ordered
by onset frame then string. Every decoding step is masked by TokenGrammar, a
batched torch copy of TabTokenizer.next_token_mask: the ONSET -> DUR -> STRING ->
FRET pattern, non-decreasing onsets, strictly increasing strings within a chord,
only PAD after EOS, plus no onsets in a window's padding. So every generated
sequence decodes without malformed groups. Optional (max_chord_span, off by
default; TART sec. 5.3 uses 5): within a chord, the fretted notes' max fret - min
fret may not exceed the span (open strings don't count).

Decoding, both with a KV cache:
  greedy  argmax under the grammar mask (TabDecoder.generate)
  beam    beam search, size 4 by default (TART sec. 5.3), Hugging Face style
          (TabDecoder.beam_search): scores are full-vocabulary log-probs with
          disallowed tokens removed; hypotheses that emit EOS leave the beam for a
          finished pool, ranked by sum(log p) / length ** length_penalty (length =
          generated tokens incl. EOS; default 1.0, Hugging Face's default).

Stage 1: Kong frozen; the decoder trains.
Stage 2: LoRA on Kong (src/kong_wrapper/lora.py); the decoder trains fully.

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
                  "max_len": 1280, "memory_len": 1001, "ffn": "gelu"}
ARCH_CONFIGS = {
    "decoder1": {"d_model": 256, "n_layers": 4, "n_heads": 4, "d_ff": 1024, "ffn": "gelu"},
    "decoder2": {"d_model": 256, "n_layers": 6, "n_heads": 8, "d_ff": 1024, "ffn": "geglu"},
}
FFN_TYPES = ("gelu", "geglu")


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


class GatedGELU(nn.Module):
    """Gated-GELU feed-forward (T5 v1.1 / TART): gelu(x W_a) * (x W_b) -> dropout -> W_out."""

    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.wi = nn.Linear(d_model, 2 * d_ff)
        self.drop = nn.Dropout(dropout)
        self.wo = nn.Linear(d_ff, d_model)

    def forward(self, x):
        a, b = self.wi(x).chunk(2, dim=-1)
        return self.wo(self.drop(F.gelu(a) * b))


def feed_forward(kind, d_model, d_ff, dropout):
    if kind == "gelu":
        return nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
                             nn.Linear(d_ff, d_model))
    if kind == "geglu":
        return GatedGELU(d_model, d_ff, dropout)
    raise ValueError(f"ffn must be one of {FFN_TYPES}, got {kind!r}")


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, ffn="gelu"):
        super().__init__()
        self.ln_self, self.ln_cross, self.ln_ff = (nn.LayerNorm(d_model) for _ in range(3))
        self.self_attn = Attention(d_model, n_heads, dropout)
        self.cross_attn = Attention(d_model, n_heads, dropout)
        self.ff = feed_forward(ffn, d_model, d_ff, dropout)
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
                 d_ff=1024, dropout=0.1, max_len=1280, memory_len=1001, ffn="gelu"):
        super().__init__()
        self.vocab_size, self.max_len, self.memory_len = vocab_size, max_len, memory_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.mem_proj = nn.Linear(memory_dim, d_model)
        self.mem_pos = nn.Embedding(memory_len, d_model)
        self.mem_ln = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(DecoderBlock(d_model, n_heads, d_ff, dropout, ffn)
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
    def generate(self, memory, n_valid, tokenizer, max_len=None, max_chord_span=None):
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
        grammar = TokenGrammar(tokenizer, B, device, n_valid, max_chord_span)
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

    @torch.no_grad()
    def beam_search(self, memory, n_valid, tokenizer, beam_size=4, length_penalty=1.0,
                    max_len=None, max_chord_span=None, return_scores=False):
        """Grammar-masked beam search -> ids [B, <= max_len] (BOS first, PAD-padded).

        Hugging Face style (see the module docstring): K live beams per row; each
        step takes the top 2K continuations of the K beams by summed log-prob
        (log_softmax over the full vocabulary, disallowed tokens removed). A
        continuation that is EOS and ranks in the top K goes to the row's finished
        pool, scored sum(log p) / n_generated ** length_penalty (n_generated counts
        EOS); the best K others stay live. A row stops once its pool holds K
        hypotheses and the best live beam, normalized at the current length, can't
        beat the pool's worst. At max_len, live beams join the pool without EOS.
        Returns each row's best pooled hypothesis (and its normalized scores [B] if
        return_scores). Run in eval mode.
        """
        if beam_size < 1:
            raise ValueError("beam_size must be >= 1")
        max_len = min(max_len or self.max_len, self.max_len)
        B, K, device = memory.shape[0], beam_size, memory.device
        neg_inf = float("-inf")
        mem, mask = self.encode_memory(memory, n_valid)
        mem = mem.repeat_interleave(K, 0)                       # rows b*K + k
        mask = None if mask is None else mask.repeat_interleave(K, 0)
        cross = self.cross_kv(mem)
        caches = [{} for _ in self.blocks]
        grammar = TokenGrammar(tokenizer, B * K, device,
                               None if n_valid is None else n_valid.repeat_interleave(K, 0),
                               max_chord_span)
        seqs = torch.full((B * K, 1), BOS_ID, dtype=torch.long, device=device)
        scores = torch.full((B, K), neg_inf, device=device)
        scores[:, 0] = 0.0                                      # start from one beam, not K copies
        pools = [[] for _ in range(B)]                          # (normalized, raw, ids) per row
        done = [False] * B
        identity = list(range(B * K))

        def norm(raw, n):
            return raw / (max(n, 1) ** length_penalty)

        def add(b, raw, ids, n):
            pools[b].append((norm(raw, n), raw, ids))
            pools[b].sort(key=lambda h: h[0], reverse=True)
            del pools[b][K:]

        step_in = seqs
        for step in range(max_len - 1):
            logits = self._decode(step_in, cross, mask, start=step, caches=caches)[:, -1]
            logp = F.log_softmax(logits.float(), -1).masked_fill(~grammar.mask(), neg_inf)
            cand = (scores.reshape(-1, 1) + logp).reshape(B, -1)              # [B, K * V]
            top_s, top_i = cand.topk(min(2 * K, cand.shape[1]), dim=1)
            top_s, top_i = top_s.tolist(), top_i.tolist()
            n_gen = step + 1                                    # tokens generated incl. this one
            src, new_tok, new_scores = [], [], []
            for b in range(B):
                live = []
                if not done[b]:
                    for rank, (s, i) in enumerate(zip(top_s[b], top_i[b])):
                        if s == neg_inf or len(live) == K:
                            break
                        beam, tok = divmod(i, self.vocab_size)
                        if tok == EOS_ID:
                            if rank < K:
                                prefix = seqs[b * K + beam].tolist()
                                add(b, s, prefix + [EOS_ID], n_gen)
                        else:
                            live.append((b * K + beam, tok, s))
                    best_live = live[0][2] if live else neg_inf
                    if (not live or (len(pools[b]) == K
                                     and pools[b][-1][0] >= norm(best_live, n_gen))):
                        done[b] = True
                if done[b]:
                    live = []
                live += [(b * K, PAD_ID, neg_inf)] * (K - len(live))   # dead fillers
                for row, tok, s in live:
                    src.append(row)
                    new_tok.append(tok)
                    new_scores.append(s)
            if all(done):
                break
            if src != identity:
                idx = torch.tensor(src, device=device)
                seqs = seqs.index_select(0, idx)
                for cache in caches:
                    cache["k"] = cache["k"].index_select(0, idx)
                    cache["v"] = cache["v"].index_select(0, idx)
                grammar.reorder(idx)
            step_in = torch.tensor(new_tok, device=device)[:, None]
            grammar.update(step_in[:, 0])
            seqs = torch.cat([seqs, step_in], 1)
            scores = torch.tensor(new_scores, device=device).reshape(B, K)
        # Rows still running at max_len: their live beams join the pool (no EOS).
        n_gen = seqs.shape[1] - 1
        flat = scores.reshape(-1).tolist()
        for b in range(B):
            if not done[b]:
                for k in range(K):
                    if flat[b * K + k] != neg_inf:
                        add(b, flat[b * K + k], seqs[b * K + k].tolist(), n_gen)
        best = [pools[b][0] if pools[b] else (neg_inf, neg_inf, [BOS_ID, EOS_ID])
                for b in range(B)]
        width = max(len(h[2]) for h in best)
        ids = torch.full((B, width), PAD_ID, dtype=torch.long, device=device)
        for b, h in enumerate(best):
            ids[b, :len(h[2])] = torch.tensor(h[2], device=device)
        if return_scores:
            return ids, torch.tensor([h[0] for h in best])
        return ids


class KongTabDecoder(nn.Module):
    """Kong encoder (no fusion adapter) -> `features` -> TabDecoder."""

    def __init__(self, encoder, tokenizer, **config):
        super().__init__()
        unknown = set(config) - set(DEFAULT_CONFIG)
        if unknown:
            raise ValueError(f"unknown decoder config {sorted(unknown)}")
        if getattr(encoder, "fusion_adapter", None) is not None:
            raise ValueError("the decoder reads Kong's frame_gru output directly; build "
                             "the encoder with use_adapter=False")
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
    def generate(self, waveform, n_valid=None, max_len=None, method="greedy", beam_size=4,
                 length_penalty=1.0, max_chord_span=None):
        """method "greedy" or "beam" (beam_size, length_penalty) -> ids [B, <= max_len]."""
        memory = self.features(waveform)
        if method == "greedy":
            return self.decoder.generate(memory, n_valid, self.tokenizer, max_len, max_chord_span)
        if method == "beam":
            return self.decoder.beam_search(memory, n_valid, self.tokenizer, beam_size,
                                            length_penalty, max_len, max_chord_span)
        raise ValueError(f"method must be 'greedy' or 'beam', got {method!r}")

    def set_stage(self, stage, rank=8, alpha=16.0):
        """Stage 1: Kong frozen (decoder trains). Stage 2: LoRA on Kong (+ decoder)."""
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


def token_loss(logits: Tensor, targets: Tensor, reduction: str = "mean") -> Tensor:
    """Next-token cross-entropy, PAD targets ignored; mean over tokens (or "sum")."""
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1),
                           ignore_index=PAD_ID, reduction=reduction)


@torch.no_grad()
def token_accuracy(logits: Tensor, targets: Tensor) -> Tensor:
    keep = targets != PAD_ID
    return (logits.argmax(-1)[keep] == targets[keep]).float().mean()


# ----- constrained decoding -----

class TokenGrammar:
    """Batched next-token masks, equal to TabTokenizer.next_token_mask row by row.

    Optional n_valid [B]: onsets must also lie before frame n_valid (no notes in a
    window's zero padding). Optional max_chord_span: among the fretted notes (fret
    > 0) sharing one onset, max fret - min fret <= max_chord_span (TART uses 5);
    fret 0 is always allowed, so the mask is never empty. Call mask() -> bool
    [B, vocab], then update(next_ids [B]); reorder(idx) gathers rows (beam search).
    """

    STATE = ("pos", "prev_onset", "prev_string", "cur_onset", "cur_string", "done",
             "onset_end", "chord_onset", "chord_lo", "chord_hi")

    def __init__(self, tokenizer, batch_size, device, n_valid=None, max_chord_span=None):
        t = tokenizer
        ids = torch.arange(t.vocab_size, device=device)
        self.n_strings = t.n_strings
        self.onset_offset, self.string_offset = t.onset_offset, t.string_offset
        self.fret_offset = t.fret_offset
        self.is_onset = (ids >= t.onset_offset) & (ids < t.dur_offset)
        self.is_dur = (ids >= t.dur_offset) & (ids < t.string_offset)
        self.is_string = (ids >= t.string_offset) & (ids < t.fret_offset)
        self.is_fret = ids >= t.fret_offset
        self.onset_value = ids - t.onset_offset
        self.string_value = ids - t.string_offset
        self.fret_value = ids - t.fret_offset
        self.is_eos, self.is_pad = ids == EOS_ID, ids == PAD_ID
        self.max_chord_span = max_chord_span

        def full(v):
            return torch.full((batch_size,), v, dtype=torch.long, device=device)
        self.pos = full(0)              # position within the note: 0 ONSET, 1 DUR, 2 STRING, 3 FRET
        self.prev_onset, self.prev_string = full(-1), full(-1)     # last complete note
        self.cur_onset, self.cur_string = full(0), full(0)         # note being written
        self.done = torch.zeros(batch_size, dtype=torch.bool, device=device)
        self.onset_end = full(t.n_onset_frames)
        if n_valid is not None:
            self.onset_end = torch.minimum(self.onset_end, n_valid.to(device).long())
        # Fretted-fret range of the complete notes at onset chord_onset (the current chord).
        self.chord_onset, self.chord_lo, self.chord_hi = full(-1), full(10 ** 6), full(-1)

    def reorder(self, idx: Tensor) -> None:
        for name in self.STATE:
            setattr(self, name, getattr(self, name).index_select(0, idx))

    def mask(self) -> Tensor:
        has_prev = self.prev_onset >= 0
        onset_lo = torch.where(~has_prev, 0, torch.where(self.prev_string < self.n_strings - 1,
                                                         self.prev_onset, self.prev_onset + 1))
        at_onset = (self.is_onset & (self.onset_value >= onset_lo[:, None])
                    & (self.onset_value < self.onset_end[:, None])) | self.is_eos
        string_lo = torch.where(has_prev & (self.cur_onset == self.prev_onset),
                                self.prev_string + 1, 0)
        at_string = self.is_string & (self.string_value >= string_lo[:, None])
        at_fret = self.is_fret
        if self.max_chord_span is not None:
            same = (self.cur_onset == self.chord_onset)[:, None]
            f = self.fret_value[None, :]
            lo = torch.minimum(self.chord_lo[:, None], f)
            hi = torch.maximum(self.chord_hi[:, None], f)
            at_fret = at_fret & ((f == 0) | ~same | (hi - lo <= self.max_chord_span))
        pos = self.pos[:, None]
        m = torch.where(pos == 0, at_onset,
                        torch.where(pos == 1, self.is_dur,
                                    torch.where(pos == 2, at_string, at_fret)))
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
        new_chord = finished & (self.cur_onset != self.chord_onset)
        self.chord_lo = torch.where(new_chord, 10 ** 6, self.chord_lo)
        self.chord_hi = torch.where(new_chord, -1, self.chord_hi)
        self.chord_onset = torch.where(finished, self.cur_onset, self.chord_onset)
        fret = next_ids - self.fret_offset
        fretted = finished & (fret > 0)
        self.chord_lo = torch.where(fretted, torch.minimum(self.chord_lo, fret), self.chord_lo)
        self.chord_hi = torch.where(fretted, torch.maximum(self.chord_hi, fret), self.chord_hi)
        self.prev_onset = torch.where(finished, self.cur_onset, self.prev_onset)
        self.prev_string = torch.where(finished, self.cur_string, self.prev_string)
        self.pos = torch.where(writing, (self.pos + 1) % 4, self.pos)
