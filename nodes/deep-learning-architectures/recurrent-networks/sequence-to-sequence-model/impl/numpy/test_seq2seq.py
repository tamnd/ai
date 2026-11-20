# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/sequence-to-sequence-model/ (ai://N149H2).
"""Tests of the NumPy reference: the shape contract, the shared fixture, the gradient, the factorization, the search."""

import itertools
import json
import pathlib

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from seq2seq import (
    EOS,
    beam_search,
    config,
    decoder_inputs,
    decoder_step,
    encode,
    flops,
    forward,
    greedy_decode,
    init,
    loss,
    loss_and_grads,
    nll,
    param_count,
    recurrent_weight_count,
    schedule,
    sequence_log_prob,
    state_size,
    time_lags,
    weight_count,
)

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]
KEYS = ("src_vocab", "tgt_vocab", "embed_size", "hidden_size", "n_layers")


def cfg_of(c, reverse=True):
    return config(*(c["config"][k] for k in KEYS), reverse=reverse)


def sentences(rng, n, length, vocab):
    out = rng.integers(1, vocab, (n, length))
    out[:, -1] = EOS
    return out


def fixture():
    params = dict(np.load(FIXTURES / "params-small.npz"))
    inp = dict(np.load(FIXTURES / "input-small.npz"))
    want = dict(np.load(FIXTURES / "expected-small.npz"))
    return params, inp, want


def test_default_config_is_the_example():
    assert config() == cfg_of(SHAPES[0])


@pytest.mark.parametrize("c", [c for c in SHAPES if c["kind"] != "named"], ids=lambda c: c["name"])
def test_shapes(c):
    k, blocks = c["config"], c["blocks"]
    cfg = cfg_of(c)
    rng = np.random.default_rng(0)
    params = init(rng, cfg)
    src, tgt = sentences(rng, k["B"], k["S"], k["src_vocab"]), sentences(rng, k["B"], k["T"], k["tgt_vocab"])
    logits, cache = forward(params, cfg, src, tgt)
    assert list(logits.shape) == blocks["softmax"]["outputs"]["logits"]
    assert list(cache["top"].shape) == blocks["decoder"]["outputs"]["top"]
    assert list(cache["v"][0].shape) == blocks["encoder"]["outputs"]["v_h"]
    assert list(cache["v"][1].shape) == blocks["encoder"]["outputs"]["v_c"]
    assert sum(v.size for v in params.values()) == c["model"]["params"]


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_counts(c):
    cfg, k = cfg_of(c), c["config"]
    assert param_count(cfg) == c["model"]["params"]
    assert weight_count(cfg) == c["model"]["weights"]
    assert recurrent_weight_count(cfg) == c["model"]["recurrent_weights"]
    assert state_size(cfg) == c["model"]["state"]
    assert flops(cfg, k["B"], k["S"], k["T"]) == c["model"]["flops_forward"]


def test_the_counts_of_sutskever_et_al():
    # Section 3.4: 384M parameters, of which 64M are pure recurrent connections, and 8000 real numbers for a sentence.
    (c,) = [c for c in SHAPES if c["name"] == "sutskever"]
    cfg = cfg_of(c)
    assert weight_count(cfg) == 384_000_000
    assert recurrent_weight_count(cfg) == 64_000_000
    assert recurrent_weight_count(cfg) // 2 == 32_000_000  # the encoder's half, and the decoder's
    assert state_size(cfg) == 8000
    assert param_count(cfg) - weight_count(cfg) == 2 * 4 * 4 * 1000 + 80_000  # the biases they leave out


def test_shared_fixture():
    params, inp, want = fixture()
    cfg = config(reverse=True)
    logits, cache = forward(params, cfg, inp["src"], inp["tgt"])
    np.testing.assert_allclose(logits, want["logits"], rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(cache["v"][0], want["v_h"], rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(cache["v"][1], want["v_c"], rtol=1e-12, atol=1e-13)
    L, grads = loss_and_grads(params, cfg, inp["src"], inp["tgt"])
    np.testing.assert_allclose(L, want["loss"], rtol=1e-12)
    for k, g in grads.items():
        np.testing.assert_allclose(g, want["d" + k], rtol=1e-10, atol=1e-12, err_msg=k)
    np.testing.assert_allclose(sequence_log_prob(params, cfg, inp["src"], inp["tgt"]), want["seq_logp"], rtol=1e-12)
    fwd = forward(params, config(reverse=False), inp["src"], inp["tgt"])[0]
    np.testing.assert_allclose(fwd, want["logits_forward"], rtol=1e-12, atol=1e-13)
    greedy = greedy_decode(params, cfg, inp["src"], 6)
    for i, y in enumerate(greedy):
        assert y == [int(t) for t in want["greedy"][i] if t >= 0]
    tokens, logp = beam_search(params, cfg, inp["src"][0], 3, 6)
    assert tokens == [int(t) for t in want["beam"] if t >= 0]
    np.testing.assert_allclose(logp, want["beam_logp"], rtol=1e-12)


def test_gradients_match_finite_differences():
    cfg = config()
    rng = np.random.default_rng(1)
    params = init(rng, cfg, scale=0.5)
    src, tgt = sentences(rng, 2, 5, 7), sentences(rng, 2, 4, 6)
    _, grads = loss_and_grads(params, cfg, src, tgt)
    eps = 1e-6
    for name, p in params.items():
        for idx in np.ndindex(p.shape):
            old = p[idx]
            p[idx] = old + eps
            up = loss(params, cfg, src, tgt)
            p[idx] = old - eps
            down = loss(params, cfg, src, tgt)
            p[idx] = old
            assert abs((up - down) / (2 * eps) - grads[name][idx]) < 1e-7, (name, idx)


def test_unused_embeddings_get_no_gradient():
    # Only the rows of the tokens that appear are looked up, so only they receive a gradient.
    cfg = config()
    rng = np.random.default_rng(2)
    params = init(rng, cfg)
    src, tgt = np.array([[1, 2, 0]]), np.array([[3, 0]])
    _, grads = loss_and_grads(params, cfg, src, tgt)
    assert np.all(grads["src_embed"][3:] == 0) and np.any(grads["src_embed"][:3] != 0)
    assert np.all(grads["tgt_embed"][[1, 2, 4, 5]] == 0) and np.any(grads["tgt_embed"][[0, 3]] != 0)


def test_decoder_inputs_are_the_target_shifted_by_one():
    tgt = np.array([[4, 5, 0], [1, 2, 0]])
    np.testing.assert_array_equal(decoder_inputs(tgt), [[EOS, 4, 5], [EOS, 1, 2]])


def test_loss_is_the_sum_of_the_steps():
    # eq-factor: log p(y | x) is the sum over the steps of log p(y_t | v, y_<t), and the loss is its negative mean.
    cfg = config()
    rng = np.random.default_rng(3)
    params = init(rng, cfg, scale=0.5)
    src, tgt = sentences(rng, 3, 4, 7), sentences(rng, 3, 5, 6)
    seq = sequence_log_prob(params, cfg, src, tgt)
    np.testing.assert_allclose(-seq.mean(), loss(params, cfg, src, tgt), rtol=1e-13)
    # The decoder run one step at a time from v gives the same conditionals as the teacher-forced pass.
    state, _ = encode(params, cfg, src)
    total, prev = np.zeros(3), np.full(3, EOS)
    for t in range(tgt.shape[1]):
        logp, state = decoder_step(params, prev, state)
        total += logp[np.arange(3), tgt[:, t]]
        prev = tgt[:, t]
    np.testing.assert_allclose(total, seq, rtol=1e-12)


def all_sequences(V, n):
    """Every target sequence of at most n tokens that ends in <EOS>, and every prefix of n tokens without one."""
    words = range(1, V)
    complete = [list(p) + [EOS] for t in range(n) for p in itertools.product(words, repeat=t)]
    open_ = [list(p) for p in itertools.product(words, repeat=n)]
    return complete, open_


def prefix_log_prob(params, cfg, src, y):
    """log p of the first len(y) tokens of a target being y, which need not end in <EOS>."""
    state, _ = encode(params, cfg, src[None])
    total, prev = 0.0, np.array([EOS])
    for tok in y:
        logp, state = decoder_step(params, prev, state)
        total += logp[0, tok]
        prev = np.array([tok])
    return total


@settings(max_examples=15, deadline=None)
@given(st.integers(0, 10_000), st.integers(2, 4), st.integers(1, 4))
def test_probabilities_of_all_outputs_sum_to_one(seed, V, n):
    # The sequences that end within n steps and the prefixes still open after n steps partition all outcomes.
    cfg = config(5, V, 2, 3, 1)
    rng = np.random.default_rng(seed)
    params, src = init(rng, cfg, scale=1.0), rng.integers(1, 5, 3)
    complete, open_ = all_sequences(V, n)
    p = sum(np.exp(sequence_log_prob(params, cfg, src[None], np.array([y]))[0]) for y in complete)
    p += sum(np.exp(prefix_log_prob(params, cfg, src, y)) for y in open_)
    np.testing.assert_allclose(p, 1.0, rtol=1e-12)


def most_likely(params, cfg, src, n):
    complete, _ = all_sequences(cfg["tgt_vocab"], n)
    scores = [sequence_log_prob(params, cfg, src[None], np.array([y]))[0] for y in complete]
    best = int(np.argmax(scores))
    return complete[best], scores[best]


@settings(max_examples=20, deadline=None)
@given(st.integers(0, 10_000))
def test_beam_of_one_is_greedy(seed):
    cfg = config(5, 4, 2, 3, 2)
    rng = np.random.default_rng(seed)
    params, src = init(rng, cfg, scale=1.0), rng.integers(1, 5, 4)
    tokens, logp = beam_search(params, cfg, src, 1, 5)
    assert tokens == greedy_decode(params, cfg, src[None], 5)[0]
    if tokens[-1] == EOS:
        np.testing.assert_allclose(logp, sequence_log_prob(params, cfg, src[None], np.array([tokens]))[0])


@settings(max_examples=20, deadline=None)
@given(st.integers(0, 10_000))
def test_a_beam_that_never_prunes_finds_the_most_likely_output(seed):
    # With 3 target tokens and 4 steps there are at most 2^3 open prefixes of 3 tokens, extended by 3 tokens each.
    cfg = config(5, 3, 3, 4, 1)
    rng = np.random.default_rng(seed)
    params, src = init(rng, cfg, scale=1.5), rng.integers(1, 5, 4)
    want, want_logp = most_likely(params, cfg, src, 4)
    tokens, logp = beam_search(params, cfg, src, 24, 4)
    assert tokens == want
    np.testing.assert_allclose(logp, want_logp, rtol=1e-12)
    for beam in (1, 2, 3):  # no beam finds anything more likely than the exhaustive search
        b_tokens, b_logp = beam_search(params, cfg, src, beam, 4)
        if b_tokens[-1] == EOS:
            assert b_logp <= want_logp + 1e-12


def test_greedy_can_miss_the_most_likely_output():
    # Seed 0: greedy commits to token 1 first, while ending at once with <EOS> is more likely than any sequence.
    cfg = config(5, 3, 3, 4, 1)
    rng = np.random.default_rng(0)
    params, src = init(rng, cfg, scale=1.5), rng.integers(1, 5, 4)
    greedy = greedy_decode(params, cfg, src[None], 4)[0]
    assert greedy == [1, 1, EOS]
    best, best_logp = most_likely(params, cfg, src, 4)
    assert best == [EOS]
    greedy_logp = sequence_log_prob(params, cfg, src[None], np.array([greedy]))[0]
    assert greedy_logp < best_logp - 0.5
    assert beam_search(params, cfg, src, 2, 4)[0] == [1, EOS]  # a beam of 2 never holds <EOS> at the first step
    assert beam_search(params, cfg, src, 3, 4)[0] == best


def test_longer_outputs_are_never_more_likely_than_their_prefixes():
    # Each extra step multiplies p by a factor of at most 1, which is why the most likely output is often short.
    cfg = config(5, 4, 2, 3, 1)
    rng = np.random.default_rng(4)
    params, src = init(rng, cfg, scale=1.0), rng.integers(1, 5, 3)
    y = [int(t) for t in rng.integers(1, 4, 6)]
    logps = [prefix_log_prob(params, cfg, src, y[:n]) for n in range(len(y) + 1)]
    assert all(a >= b for a, b in itertools.pairwise(logps))


def test_reversal_is_reading_the_reversed_source():
    cfg_rev, cfg_fwd = config(reverse=True), config(reverse=False)
    rng = np.random.default_rng(5)
    params = init(rng, cfg_rev, scale=0.5)
    src, tgt = sentences(rng, 2, 5, 7), sentences(rng, 2, 4, 6)
    a = forward(params, cfg_rev, src, tgt)[0]
    b = forward(params, cfg_fwd, src[:, ::-1], tgt)[0]
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, forward(params, cfg_fwd, src, tgt)[0])


@pytest.mark.parametrize("n", [1, 2, 5, 30])
def test_time_lags(n):
    # Section 3.3 of Sutskever et al.: reversing keeps the average distance and makes the smallest one 1.
    fwd, rev = time_lags(n, False), time_lags(n, True)
    assert fwd == [n] * n
    assert rev == [2 * i - 1 for i in range(1, n + 1)]
    assert sum(fwd) == sum(rev) == n * n
    assert min(rev) == 1 and max(rev) == 2 * n - 1


def test_the_source_reaches_the_decoder_only_through_v():
    # Two different sources with the same v give the same output distribution: v is the whole interface.
    cfg = config()
    rng = np.random.default_rng(6)
    params = init(rng, cfg, scale=0.5)
    src_a, src_b = sentences(rng, 1, 5, 7), sentences(rng, 1, 3, 7)
    tgt = sentences(rng, 1, 4, 6)
    v_a, _ = encode(params, cfg, src_a)
    v_b, _ = encode(params, cfg, src_b)
    assert not np.allclose(v_a[0], v_b[0])
    run = lambda v: [decoder_step(params, np.array([t]), v)[0] for t in decoder_inputs(tgt)[0]]  # noqa: E731
    assert not np.allclose(run(v_a)[0], run(v_b)[0])
    logits = forward(params, cfg, src_a, tgt)[0]
    state = v_a
    for t, tok in enumerate(decoder_inputs(tgt)[0]):
        logp, state = decoder_step(params, np.array([tok]), state)
        np.testing.assert_allclose(logp[0], logits[0, t] - np.log(np.exp(logits[0, t]).sum()), rtol=1e-12)


def test_teacher_forced_inputs_can_be_replaced():
    # Scheduled sampling feeds some of the model's own tokens: the logits then differ after the first replaced step.
    cfg = config()
    rng = np.random.default_rng(7)
    params = init(rng, cfg, scale=0.5)
    src, tgt = sentences(rng, 1, 5, 7), sentences(rng, 1, 4, 6)
    dec_in = decoder_inputs(tgt).copy()
    dec_in[0, 2] = (dec_in[0, 2] % 5) + 1
    a, b = forward(params, cfg, src, tgt)[0], forward(params, cfg, src, tgt, dec_in)[0]
    np.testing.assert_array_equal(a[:, :2], b[:, :2])
    assert not np.allclose(a[:, 2:], b[:, 2:])
    assert nll(b, tgt) != nll(a, tgt)


def test_schedules():
    # Bengio et al. (2015), Section 2.4: each starts at or near 1 and decays towards its floor.
    steps = np.arange(0, 5000, 250)
    lin = [schedule("linear", i, 1.0, 2e-4, 0.1) for i in steps]
    exp = [schedule("exponential", i, 0.999) for i in steps]
    sig = [schedule("inverse_sigmoid", i, 500.0) for i in steps]
    assert lin[0] == 1.0 and min(lin) == 0.1
    assert exp[0] == 1.0 and exp[-1] < 0.01
    np.testing.assert_allclose(sig[0], 500 / 501)
    for s in (lin, exp, sig):
        assert all(a >= b for a, b in itertools.pairwise(s))
        assert all(0 <= v <= 1 for v in s)
    # The inverse sigmoid is 1/2 at i = k log k.
    np.testing.assert_allclose(schedule("inverse_sigmoid", 500 * np.log(500), 500.0), 0.5)
    with pytest.raises(ValueError):
        schedule("exponential", 1, 1.5)
    with pytest.raises(ValueError):
        schedule("inverse_sigmoid", 1, 0.5)
    with pytest.raises(ValueError):
        schedule("cosine", 1, 1.0)


def test_config_rejects_bad_sizes():
    with pytest.raises(ValueError):
        config(hidden_size=0)
    with pytest.raises(ValueError):
        config(n_layers=0)
    with pytest.raises(ValueError):
        config(tgt_vocab=1)
    with pytest.raises(ValueError):
        beam_search(init(np.random.default_rng(0), config()), config(), np.array([1, 0]), 0, 3)


def flop_case(c):
    """The forward call of a config of shapes.json, for aik flopcount."""
    cfg, k = cfg_of(c), c["config"]
    rng = np.random.default_rng(0)
    src, tgt = sentences(rng, k["B"], k["S"], k["src_vocab"]), sentences(rng, k["B"], k["T"], k["tgt_vocab"])
    return forward, (init(rng, cfg), cfg, src, tgt)
