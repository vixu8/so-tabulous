"""Tests for the autoregressive tab decoder (src/decoder/tab_decoder.py), CPU only.

A fake encoder stands in for Kong; a real-checkpoint end-to-end run is in
tests/test_train_decoder.py.
"""

import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch import nn

SRC = Path(__file__).resolve().parents[1] / "src"
for sub in ("decoder", "data"):
    sys.path.insert(0, str(SRC / sub))
from tab_decoder import (KongTabDecoder, TabDecoder, TokenGrammar, split_tokens,  # noqa: E402
                         token_accuracy, token_loss)
from tokenizer import BOS_ID, EOS_ID, PAD_ID, TabTokenizer  # noqa: E402

TOK = TabTokenizer()
SMALL = {"d_model": 32, "n_layers": 2, "n_heads": 4, "d_ff": 64, "dropout": 0.0, "max_len": 64}


class FakeEncoder(nn.Module):
    feature_dim = 512

    def __init__(self, frames=1001):
        super().__init__()
        self.frames = frames
        self.proj = nn.Linear(1, 512)

    def forward(self, waveform):
        x = waveform[:, :self.frames, None]
        return {"features": self.proj(x)}


def random_notes(rng, n):
    """n notes in a 10 s window, some sharing onsets (chords)."""
    onsets = np.sort(rng.choice(np.arange(0, 9.9, 0.05), size=n))
    strings = rng.integers(0, 6, n)
    notes = np.stack([strings, rng.integers(0, 20, n), onsets,
                      onsets + rng.uniform(0.01, 2.0, n)], 1)
    return notes


def make_model(seed=0, random_out=False, **cfg):
    torch.manual_seed(seed)
    model = KongTabDecoder(FakeEncoder(), TOK, **{**SMALL, **cfg}).eval()
    if random_out:   # untrained output is uniform; give it opinions so generation is nontrivial
        nn.init.normal_(model.decoder.out.weight, std=1.0)
    return model


def test_shapes_and_uniform_loss_at_init():
    model = make_model()
    ids = torch.tensor([TOK.encode(random_notes(np.random.default_rng(0), 5)) + [PAD_ID] * 3])
    tokens_in, targets = split_tokens(ids, model.max_len)
    logits = model(torch.randn(1, 2000), tokens_in)
    assert logits.shape == (1, tokens_in.shape[1], TOK.vocab_size) == (1, 24, 1104)
    assert token_loss(logits, targets).item() == pytest.approx(math.log(TOK.vocab_size), rel=1e-5)
    assert 0.0 <= token_accuracy(logits, targets).item() <= 1.0


def test_split_tokens_truncates_to_max_len():
    ids = torch.arange(100)[None]
    tokens_in, targets = split_tokens(ids, 64)
    assert tokens_in.shape == targets.shape == (1, 64)
    assert torch.equal(targets[0], torch.arange(1, 65))


def test_loss_ignores_pad_targets():
    logits = torch.randn(1, 3, TOK.vocab_size)
    targets = torch.tensor([[5, PAD_ID, PAD_ID]])
    assert token_loss(logits, targets).item() == pytest.approx(
        nn.functional.cross_entropy(logits[0, :1], targets[0, :1]).item(), rel=1e-6)


def test_causal_past_logits_ignore_future_tokens():
    model = make_model(random_out=True)
    wave = torch.randn(1, 2000)
    a = torch.randint(3, TOK.vocab_size, (1, 12))
    b = a.clone()
    b[0, 7:] = torch.randint(3, TOK.vocab_size, (5,))
    la, lb = model(wave, a), model(wave, b)
    torch.testing.assert_close(la[0, :7], lb[0, :7])
    assert not torch.allclose(la[0, 7:], lb[0, 7:])


def test_memory_mask_hides_padding_frames():
    model = make_model(random_out=True)
    tokens = torch.tensor([[BOS_ID, 10, 1010]])
    wave = torch.randn(1, 2000)
    wave2 = wave.clone()
    wave2[0, 600:] = 5.0                     # change only frames >= n_valid
    n_valid = torch.tensor([600])
    torch.testing.assert_close(model(wave, tokens, n_valid), model(wave2, tokens, n_valid))
    assert not torch.allclose(model(wave, tokens), model(wave2, tokens))


def test_kv_cache_matches_full_forward():
    model = make_model(random_out=True)
    dec = model.decoder
    memory = model.features(torch.randn(2, 2000))
    n_valid = torch.tensor([1001, 400])
    tokens = torch.randint(3, TOK.vocab_size, (2, 10))
    tokens[:, 0] = BOS_ID
    full = dec(tokens, memory, n_valid)
    mem, mask = dec.encode_memory(memory, n_valid)
    cross, caches = dec.cross_kv(mem), [{} for _ in dec.blocks]
    steps = [dec._decode(tokens[:, i:i + 1], cross, mask, start=i, caches=caches)
             for i in range(tokens.shape[1])]
    torch.testing.assert_close(torch.cat(steps, 1), full, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("seed", range(5))
def test_grammar_equals_tokenizer_mask_on_every_prefix(seed):
    rng = np.random.default_rng(seed)
    seqs = [TOK.encode(random_notes(rng, n)) for n in (0, 3, 12)]
    seqs.append(TOK.encode(np.array([[s, 0, 2.0, 3.0] for s in range(6)])))   # full chord
    width = max(map(len, seqs)) + 2
    padded = torch.tensor([s + [PAD_ID] * (width - len(s)) for s in seqs])
    g = TokenGrammar(TOK, len(seqs), "cpu")
    for i in range(1, width):
        m = g.mask()
        for r, s in enumerate(seqs):
            prefix = (s + [PAD_ID] * width)[:i]
            expected = TOK.next_token_mask(prefix)
            assert np.array_equal(m[r].numpy(), expected), (r, i, TOK.describe(prefix[-4:]))
        g.update(padded[:, i])
    assert g.done.all()


def test_grammar_onset_cap_from_n_valid():
    g = TokenGrammar(TOK, 2, "cpu", n_valid=torch.tensor([1001, 300]))
    onsets = g.mask()[:, TOK.onset_offset:TOK.dur_offset]
    assert onsets[0].sum() == 1000 and onsets[1].sum() == 300 and onsets[1, :300].all()
    # After a note at frame 299 on the top string, only EOS is left for row 1.
    for tok in (TOK.onset_offset + 299, TOK.dur_offset, TOK.string_offset + 5, TOK.fret_offset):
        g.update(torch.tensor([tok, tok]))
    m = g.mask()
    assert m[1].nonzero().flatten().tolist() == [EOS_ID]
    assert m[0, TOK.onset_offset + 300] and m[0, EOS_ID]


def test_generate_is_grammatical_and_stops():
    model = make_model(random_out=True, max_len=64)
    wave = torch.randn(3, 2000)
    ids = model.generate(wave, torch.tensor([1001, 1001, 200]))
    assert ids.shape[0] == 3 and ids.shape[1] <= 64 and (ids[:, 0] == BOS_ID).all()
    for row, n_valid in zip(ids.tolist(), (1001, 1001, 200)):
        if EOS_ID in row:
            e = row.index(EOS_ID)
            assert all(t == PAD_ID for t in row[e + 1:])
            notes = TOK.decode(row, strict=True)
        else:                                   # hit max_len: drop the unfinished group
            notes = TOK.decode(row[:1 + 4 * ((len(row) - 1) // 4)], strict=True)
        assert len(notes) == 0 or (np.round(notes[:, 2] * 100) < n_valid).all()
    # The untrained (uniform) model's first allowed token is EOS: an empty transcription.
    assert make_model().generate(wave[:1]).tolist() == [[BOS_ID, EOS_ID]]


def test_generate_matches_teacher_forced_argmax():
    model = make_model(random_out=True, max_len=40)
    wave = torch.randn(1, 2000)
    ids = model.generate(wave)
    logits = model(wave, ids[:, :-1])
    g = TokenGrammar(TOK, 1, "cpu")
    for i in range(ids.shape[1] - 1):
        step = logits[0, i].masked_fill(~g.mask()[0], float("-inf"))
        assert step.argmax().item() == ids[0, i + 1].item()
        g.update(ids[:, i + 1])


def test_set_stage_needs_tunable_encoder():
    with pytest.raises(TypeError):
        make_model().set_stage(1)
    with pytest.raises(ValueError):
        make_model().set_stage(3)


def test_rejects_unknown_config_and_long_memory():
    with pytest.raises(ValueError):
        KongTabDecoder(FakeEncoder(), TOK, width=3)
    dec = TabDecoder(TOK.vocab_size, memory_len=10, **{k: v for k, v in SMALL.items()})
    with pytest.raises(ValueError):
        dec(torch.tensor([[BOS_ID]]), torch.randn(1, 11, 512))
