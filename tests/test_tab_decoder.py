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
from tab_decoder import (ARCH_CONFIGS, GatedGELU, KongTabDecoder, TabDecoder,  # noqa: E402
                         TokenGrammar, split_tokens, token_accuracy, token_loss)
from tokenizer import BOS_ID, EOS_ID, PAD_ID, TabTokenizer  # noqa: E402

TOK = TabTokenizer()
SMALL = {"d_model": 32, "n_layers": 2, "n_heads": 4, "d_ff": 64, "dropout": 0.0, "max_len": 64}
# Small stand-ins for the two architectures (decoder2: more layers / heads, gated-GELU FFN).
ARCHS = {"decoder1": {}, "decoder2": {"n_layers": 3, "n_heads": 8, "ffn": "geglu"}}


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


@pytest.mark.parametrize("arch", ARCHS)
def test_shapes_and_uniform_loss_at_init(arch):
    model = make_model(**ARCHS[arch])
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


@pytest.mark.parametrize("arch", ARCHS)
def test_causal_past_logits_ignore_future_tokens(arch):
    model = make_model(random_out=True, **ARCHS[arch])
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


@pytest.mark.parametrize("arch", ARCHS)
def test_kv_cache_matches_full_forward(arch):
    model = make_model(random_out=True, **ARCHS[arch])
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


@pytest.mark.parametrize("arch", ARCHS)
def test_generate_matches_teacher_forced_argmax(arch):
    model = make_model(random_out=True, max_len=40, **ARCHS[arch])
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
    with_adapter = FakeEncoder()
    with_adapter.fusion_adapter = nn.Identity()          # midway plan: no fusion adapter
    with pytest.raises(ValueError, match="use_adapter=False"):
        KongTabDecoder(with_adapter, TOK)
    dec = TabDecoder(TOK.vocab_size, memory_len=10, **{k: v for k, v in SMALL.items()})
    with pytest.raises(ValueError):
        dec(torch.tensor([[BOS_ID]]), torch.randn(1, 11, 512))


# ----- decoder2 (TART-style) -----

def test_arch_configs_and_gated_gelu():
    assert ARCH_CONFIGS["decoder1"] == {"d_model": 256, "n_layers": 4, "n_heads": 4,
                                        "d_ff": 1024, "ffn": "gelu"}
    assert ARCH_CONFIGS["decoder2"] == {"d_model": 256, "n_layers": 6, "n_heads": 8,
                                        "d_ff": 1024, "ffn": "geglu"}
    d, f = 256, 1024
    ff = GatedGELU(d, f, 0.0)
    assert sum(p.numel() for p in ff.parameters()) == 3 * d * f + 2 * f + d
    x = torch.randn(2, 5, d)
    a, b = ff.wi(x).chunk(2, -1)
    torch.testing.assert_close(ff(x), ff.wo(nn.functional.gelu(a) * b))
    sizes = {k: sum(p.numel() for p in TabDecoder(TOK.vocab_size, **c).parameters())
             for k, c in ARCH_CONFIGS.items()}
    assert 5.4e6 < sizes["decoder1"] < 5.6e6 and 8.5e6 < sizes["decoder2"] < 9.5e6
    with pytest.raises(ValueError):
        TabDecoder(TOK.vocab_size, ffn="relu")


# ----- beam search -----

def seq_score(model, wave, ids, n_valid=None, length_penalty=1.0):
    """Teacher-forced sum of log p (full vocab) over the generated tokens / n ** lp,
    checking every token is grammatical."""
    ids = ids[:, :int((ids[0] != PAD_ID).sum())]
    logp = torch.log_softmax(model(wave, ids[:, :-1], n_valid).float(), -1)
    g = TokenGrammar(TOK, 1, "cpu", n_valid)
    total = 0.0
    for i in range(ids.shape[1] - 1):
        tok = ids[0, i + 1].item()
        assert g.mask()[0, tok], (i, TOK.id_to_str(tok))
        total += logp[0, i, tok].item()
        g.update(ids[:, i + 1])
    return total / (ids.shape[1] - 1) ** length_penalty


def check_grammatical(row, n_valid):
    row = [t for t in row]
    if EOS_ID in row:
        e = row.index(EOS_ID)
        assert all(t == PAD_ID for t in row[e + 1:])
        notes = TOK.decode(row, strict=True)
    else:
        row = [t for t in row if t != PAD_ID]
        notes = TOK.decode(row[:1 + 4 * ((len(row) - 1) // 4)], strict=True)
    assert len(notes) == 0 or (np.round(notes[:, 2] * 100) < n_valid).all()
    return notes


@pytest.mark.parametrize("arch", ARCHS)
def test_beam_search_is_grammatical_and_scores_match_teacher_forcing(arch):
    model = make_model(seed=1, random_out=True, max_len=48, **ARCHS[arch])
    wave = torch.randn(3, 2000)
    n_valid = torch.tensor([1001, 1001, 150])
    ids, scores = model.decoder.beam_search(model.features(wave), n_valid, TOK, beam_size=4,
                                            return_scores=True)
    assert ids.shape[0] == 3 and ids.shape[1] <= 48 and (ids[:, 0] == BOS_ID).all()
    for b in range(3):
        check_grammatical(ids[b].tolist(), n_valid[b].item())
        # The returned score is the sequence's real (cached, reordered) log-prob.
        expected = seq_score(model, wave[b:b + 1], ids[b:b + 1], n_valid[b:b + 1])
        assert scores[b].item() == pytest.approx(expected, rel=1e-4, abs=1e-4)
    # Deterministic, and each row decodes the same alone as in the batch.
    again = model.generate(wave, n_valid, method="beam")
    torch.testing.assert_close(again, ids)
    for b in range(3):
        alone = model.generate(wave[b:b + 1], n_valid[b:b + 1], method="beam")
        width = alone.shape[1]
        assert alone[0].tolist() == ids[b, :width].tolist()
        assert (ids[b, width:] == PAD_ID).all()


def test_beam_search_length_penalty_and_untrained_model():
    model = make_model(seed=2, random_out=True, max_len=40)
    wave = torch.randn(2, 2000)
    for lp in (0.0, 1.0, 2.0):
        ids, scores = model.decoder.beam_search(model.features(wave), None, TOK, beam_size=3,
                                                length_penalty=lp, return_scores=True)
        for b in range(2):
            assert scores[b].item() == pytest.approx(
                seq_score(model, wave[b:b + 1], ids[b:b + 1], length_penalty=lp), rel=1e-4)
    # The untrained (uniform) model: every hypothesis scores -ln(vocab) per token; the
    # output is still a valid sequence.
    ids = make_model().generate(wave[:1], method="beam")
    check_grammatical(ids[0].tolist(), 1001)
    with pytest.raises(ValueError):
        make_model().generate(wave[:1], method="sample")


def test_beam_size_one_matches_greedy_when_greedy_ends_with_eos():
    # With one beam and no length preference, beam search keeps exactly greedy's path,
    # except that it may continue past an EOS greedy took; check the shared prefix.
    model = make_model(seed=3, random_out=True, max_len=40)
    wave = torch.randn(1, 2000)
    greedy = model.generate(wave)[0].tolist()
    beam = model.generate(wave, method="beam", beam_size=1, length_penalty=0.0)[0].tolist()
    body = [t for t in greedy if t not in (EOS_ID, PAD_ID)]
    assert beam[:len(body)] == body


def test_grammar_reorder_gathers_rows():
    g = TokenGrammar(TOK, 3, "cpu", n_valid=torch.tensor([1001, 500, 200]))
    g.update(torch.tensor([TOK.onset_offset + 5, EOS_ID, TOK.onset_offset + 7]))
    g.reorder(torch.tensor([2, 2, 1]))
    assert g.pos.tolist() == [1, 1, 0] and g.done.tolist() == [False, False, True]
    assert g.onset_end.tolist() == [200, 200, 500] and g.cur_onset.tolist()[:2] == [7, 7]


# ----- chord fret-span constraint (TART: <= 5) -----

def reference_fret_mask(prefix, span):
    """Pure-Python fret mask after `prefix` (which ends at a STRING token)."""
    body = prefix[1:]
    notes = [body[i:i + 4] for i in range(0, len(body) - 3, 4)]
    onset = body[len(notes) * 4]                       # the note being written
    frets = [n[3] - TOK.fret_offset for n in notes if n[0] == onset]
    frets = [f for f in frets if f > 0]
    allowed = np.zeros(TOK.vocab_size, bool)
    for f in range(TOK.max_fret + 1):
        allowed[TOK.fret_offset + f] = f == 0 or max(frets + [f]) - min(frets + [f]) <= span
    return allowed


@pytest.mark.parametrize("span", [0, 2, 5])
def test_chord_span_mask_matches_reference(span):
    rng = np.random.default_rng(span)
    chords = [np.array([[s, f, 1.0 + 0.5 * c, 3.0] for s, f in
                        zip(sorted(rng.choice(6, 4, replace=False)), rng.integers(0, 20, 4))])
              for c in range(5)]
    seqs = [TOK.encode(np.concatenate(chords)), TOK.encode(random_notes(rng, 15))]
    width = max(map(len, seqs))
    padded = torch.tensor([s + [PAD_ID] * (width - len(s)) for s in seqs])
    g = TokenGrammar(TOK, len(seqs), "cpu", max_chord_span=span)
    checked = 0
    for i in range(1, width):
        m = g.mask()
        for r, s in enumerate(seqs):
            prefix = (s + [PAD_ID] * width)[:i]
            base = TOK.next_token_mask(prefix)
            if i < len(s) and (i - 1) % 4 == 3:            # FRET position
                expected = base & reference_fret_mask(prefix, span)
                checked += 1
            else:
                expected = base
            assert np.array_equal(m[r].numpy(), expected), (r, i)
        g.update(padded[:, i])
    assert checked > 0


@pytest.mark.parametrize("method", ["greedy", "beam"])
def test_generation_respects_chord_span(method):
    model = make_model(seed=4, random_out=True, max_len=64)
    with torch.no_grad():                    # only onsets 0-2 are likely -> chords of up to 6 notes
        model.decoder.out.bias[TOK.onset_offset:TOK.onset_offset + 3] = 40.0
    wave = torch.randn(2, 2000)

    def worst_span(ids):
        worst = 0
        for row in ids.tolist():
            notes = check_grammatical(row, 1001)
            for onset in np.unique(notes[:, 2]):
                frets = notes[(notes[:, 2] == onset) & (notes[:, 1] > 0), 1]
                worst = max(worst, frets.max() - frets.min() if len(frets) else 0)
        return worst
    assert worst_span(model.generate(wave, method="greedy")) > 2     # unconstrained: wide chords
    assert worst_span(model.generate(wave, method=method, max_chord_span=2)) <= 2
