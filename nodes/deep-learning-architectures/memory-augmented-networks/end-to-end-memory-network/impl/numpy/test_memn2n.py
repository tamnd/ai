# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/end-to-end-memory-network/ (ai://VDRYRT).
"""Tests of the NumPy reference: the position encoding, the null word, one hop against its equations, both tyings
against a loop over the hops written from the paper, a one-hop model wired by hand that answers a toy task, the
pitfalls, the shapes and the shared fixture."""

import json
import pathlib

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from memn2n import (
    NULL,
    config,
    cross_entropy,
    encode,
    flops,
    forward,
    hop,
    init,
    param_count,
    position_encoding,
    softmax,
    zero_null,
)

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]
KEYS = ("vocab", "dim", "hops", "mem", "sent_len")

# The toy task of the tests: "<person> went <place>" sentences, the question "where is <person>", the answer the
# place of the most recent sentence about that person. Word 0 is the null word.
WHERE, IS, WENT = 1, 2, 3
PERSONS, PLACES = (4, 5, 6), (7, 8, 9, 10, 11)


def cfg_of(c, **kw):
    return config(**{k: c["config"][k] for k in KEYS}, **kw)


def toy_batch(rng, batch, n=5, j=4):
    story = np.zeros((batch, n, j), dtype=np.int64)
    query = np.zeros((batch, j), dtype=np.int64)
    answer = np.zeros(batch, dtype=np.int64)
    for b in range(batch):
        who = rng.choice(PERSONS, n)
        where = rng.choice(PLACES, n)
        story[b, :, 0], story[b, :, 1], story[b, :, 2] = who, WENT, where
        asked = who[rng.integers(n)]
        query[b, :3] = WHERE, IS, asked
        answer[b] = where[np.flatnonzero(who == asked)[0]]  # row 0 is the most recent sentence
    return story, query, answer


def test_position_encoding_is_the_formula():
    J, d = 5, 4
    lw = position_encoding(J, d)
    want = [[(1 - j / J) - (k / d) * (1 - 2 * j / J) for k in range(1, d + 1)] for j in range(1, J + 1)]
    np.testing.assert_allclose(lw, want)


def test_position_encoding_special_rows():
    # The word at j = J/2 weighs 1/2 in every dimension, and the last word weighs k/d.
    lw = position_encoding(4, 6)
    np.testing.assert_allclose(lw[1], np.full(6, 0.5))
    np.testing.assert_allclose(lw[-1], np.arange(1, 7) / 6)
    np.testing.assert_allclose(position_encoding(1, 3)[0], [1 / 3, 2 / 3, 1.0])


def test_position_encoding_sees_word_order_and_bag_of_words_does_not():
    rng = np.random.default_rng(0)
    E = rng.normal(size=(8, 5))
    E[NULL] = 0.0
    a, b = np.array([[3, 5, 6]]), np.array([[6, 5, 3]])
    np.testing.assert_allclose(encode(E, a), encode(E, b))
    lw = position_encoding(3, 5)
    assert not np.allclose(encode(E, a, lw), encode(E, b, lw))


def test_null_word_adds_nothing():
    rng = np.random.default_rng(1)
    cfg = config(vocab=9, dim=4, hops=2, mem=3, sent_len=6)
    params = init(rng, cfg)
    for name in ("E0", "E1", "E2"):
        assert not params[name][NULL].any()
    s = np.array([[4, 7, 0, 0, 0, 0]])
    np.testing.assert_allclose(encode(params["E1"], s)[0], params["E1"][4] + params["E1"][7])
    # With a null row that is not zero, the four pads would add four copies of it to the sentence.
    loose = params["E1"].copy()
    loose[NULL] = 1.0
    np.testing.assert_allclose(encode(loose, s) - encode(params["E1"], s), [[4.0] * 4])
    # An update moves the null row; zero_null puts it back.
    moved = {k: v + 1.0 for k, v in params.items()}
    assert not zero_null(moved, cfg)["E2"][NULL].any() and zero_null(moved, cfg)["T0"].all()


def test_hop_is_the_equations():
    rng = np.random.default_rng(2)
    u, m, c = rng.normal(size=(3, 4)), rng.normal(size=(3, 5, 4)), rng.normal(size=(3, 5, 4))
    o, p = hop(u, m, c)
    for b in range(3):
        s = np.array([u[b] @ m[b, i] for i in range(5)])
        want_p = np.exp(s) / np.exp(s).sum()
        np.testing.assert_allclose(p[b], want_p)
        np.testing.assert_allclose(o[b], sum(want_p[i] * c[b, i] for i in range(5)))
    o_lin, p_lin = hop(u, m, c, linear=True)
    np.testing.assert_allclose(p_lin, np.einsum("bd,bnd->bn", u, m))
    np.testing.assert_allclose(o_lin, np.einsum("bn,bnd->bd", p_lin, c))


@settings(max_examples=40, deadline=None)
@given(st.integers(0, 10_000), st.floats(0.01, 30.0))
def test_attention_is_a_distribution(seed, scale):
    rng = np.random.default_rng(seed)
    u, m = scale * rng.normal(size=(2, 3)), scale * rng.normal(size=(2, 6, 3))
    _, p = hop(u, m, m)
    assert np.all(p >= 0) and np.allclose(p.sum(-1), 1.0)


def test_a_shift_of_every_memory_vector_along_nothing_changes_nothing():
    # Adding the same vector to every m_i adds the same number to every score, which the softmax ignores.
    rng = np.random.default_rng(3)
    u, m, c = rng.normal(size=(2, 4)), rng.normal(size=(2, 5, 4)), rng.normal(size=(2, 5, 4))
    o1, p1 = hop(u, m, c)
    o2, p2 = hop(u, m + rng.normal(size=(2, 1, 4)), c)
    np.testing.assert_allclose(p1, p2)
    np.testing.assert_allclose(o1, o2)


def by_hand(params, story, query, cfg, linear=False):
    """The model written out from Sections 2.1, 2.2 and 4.1 with explicit A^k, C^k, B, W, T_A^k and T_C^k."""
    K, J, d = cfg["hops"], cfg["sent_len"], cfg["dim"]
    lw = position_encoding(J, d) if cfg["encoding"] == "pe" else np.ones((J, d))
    if cfg["tying"] == "adjacent":
        A = [params[f"E{k}"] for k in range(K)]
        C = [params[f"E{k + 1}"] for k in range(K)]
        TA = [params[f"T{k}"] for k in range(K)]
        TC = [params[f"T{k + 1}"] for k in range(K)]
        Bm, W, H = params["E0"], params[f"E{K}"], np.eye(d)
    else:
        A, C = [params["A"]] * K, [params["C"]] * K
        TA, TC = [params["TA"]] * K, [params["TC"]] * K
        Bm, W, H = params["B"], params["W"], params["H"]
    out = []
    for b in range(story.shape[0]):
        u = sum(lw[j] * Bm[query[b, j]] for j in range(J))
        for k in range(K):
            m = np.array([sum(lw[j] * A[k][story[b, i, j]] for j in range(J)) + TA[k][i] for i in range(cfg["mem"])])
            c = np.array([sum(lw[j] * C[k][story[b, i, j]] for j in range(J)) + TC[k][i] for i in range(cfg["mem"])])
            s = m @ u
            p = s if linear else np.exp(s - s.max()) / np.exp(s - s.max()).sum()
            u = H @ u + p @ c
        out.append(W @ u)
    return np.array(out)


@pytest.mark.parametrize("tying", ["adjacent", "layerwise"])
@pytest.mark.parametrize("encoding", ["bow", "pe"])
@pytest.mark.parametrize("linear", [False, True])
def test_forward_is_the_model_of_the_paper(tying, encoding, linear):
    rng = np.random.default_rng(4)
    cfg = config(vocab=12, dim=5, hops=3, mem=5, sent_len=4, tying=tying, encoding=encoding)
    params = init(rng, cfg, sigma=0.5)
    story, query, _ = toy_batch(rng, 3)
    np.testing.assert_allclose(forward(params, story, query, cfg, linear=linear),
                               by_hand(params, story, query, cfg, linear=linear), rtol=1e-12, atol=1e-12)


def test_adjacent_tying_shares_the_matrices_between_hops():
    # Changing E1 changes both the output vectors of hop 1 and the memory vectors of hop 2.
    rng = np.random.default_rng(5)
    cfg = config(vocab=12, dim=5, hops=2, mem=5, sent_len=4, encoding="bow", temporal=False)
    params = init(rng, cfg, sigma=0.5)
    assert sorted(params) == ["E0", "E1", "E2"] and param_count(cfg) == 3 * 12 * 5
    story, query, _ = toy_batch(rng, 2)
    _, attn = forward(params, story, query, cfg, trace=True)
    bumped = dict(params, E1=params["E1"] * 1.5)
    _, attn2 = forward(bumped, story, query, cfg, trace=True)
    np.testing.assert_allclose(attn[0], attn2[0])  # hop 1 reads its memory through E0
    assert not np.allclose(attn[1], attn2[1])  # hop 2 reads its memory through E1


def hand_wired(beta=10.0, gamma=5.0):
    """One hop, the bag of words, adjacent tying, d = 9: dimensions 0-2 are the persons, 3-7 the places, 8 a
    recency feature. E0 (= A^1 = B) maps a person to beta in its dimension and the word where to 1 in dimension 8;
    T0 (= T_A^1) gives memory i the value -gamma i in dimension 8; E1 (= C^1 = W^T) maps a place to its dimension."""
    cfg = config(vocab=12, dim=9, hops=1, mem=5, sent_len=4, encoding="bow")
    E0, E1, T0, T1 = np.zeros((12, 9)), np.zeros((12, 9)), np.zeros((5, 9)), np.zeros((5, 9))
    for i, w in enumerate(PERSONS):
        E0[w, i] = beta
    E0[WHERE, 8] = 1.0
    T0[:, 8] = -gamma * np.arange(5)
    for i, w in enumerate(PLACES):
        E1[w, 3 + i] = 1.0
    return {"E0": E0, "E1": E1, "T0": T0, "T1": T1}, cfg


def test_a_hand_wired_hop_answers_the_toy_task():
    params, cfg = hand_wired()
    story, query, answer = toy_batch(np.random.default_rng(6), 200)
    logits, attn = forward(params, story, query, cfg, trace=True)
    np.testing.assert_array_equal(logits.argmax(-1), answer)
    who = query[:, 2]
    first = np.array([np.flatnonzero(story[b, :, 0] == who[b])[0] for b in range(200)])
    assert np.all(attn[0][np.arange(200), first] > 0.99)


def test_without_the_temporal_rows_the_hop_cannot_tell_old_from_new():
    params, cfg = hand_wired()
    params["T0"] = np.zeros_like(params["T0"])
    story = np.array([[[4, WENT, 7, 0], [4, WENT, 8, 0], [5, WENT, 9, 0], [6, WENT, 10, 0], [6, WENT, 11, 0]]])
    query = np.array([[WHERE, IS, 4, 0]])
    _, attn = forward(params, story, query, cfg, trace=True)
    np.testing.assert_allclose(attn[0, 0, :2], [0.5, 0.5])  # both sentences about person 4 tie


def test_reversed_order_keeps_the_latest_sentence_at_row_zero():
    # Padding the story with older sentences moves no recent sentence: the temporal row of each stays the same.
    params, cfg = hand_wired()
    story = np.zeros((1, 5, 4), dtype=np.int64)
    story[0, 0] = 4, WENT, 9, 0
    story[0, 1] = 4, WENT, 7, 0
    query = np.array([[WHERE, IS, 4, 0]])
    longer = story.copy()
    longer[0, 2:] = [[5, WENT, 8, 0], [6, WENT, 10, 0], [4, WENT, 11, 0]]
    for s in (story, longer):
        assert forward(params, s, query, cfg).argmax() == 9


def test_param_count_is_the_arrays():
    for tying in ("adjacent", "layerwise"):
        for temporal in (True, False):
            cfg = config(vocab=10, dim=4, hops=3, mem=6, sent_len=5, tying=tying, temporal=temporal)
            assert param_count(cfg) == sum(v.size for v in init(np.random.default_rng(0), cfg).values())
    adj = config(vocab=177, dim=20, hops=3, mem=50, sent_len=7)
    assert param_count(adj) == 4 * (177 * 20 + 50 * 20) == 18_160
    assert param_count(adj) == param_count(config(vocab=177, dim=20, hops=3, mem=50, sent_len=1))


def test_cross_entropy_is_summed_not_averaged():
    rng = np.random.default_rng(7)
    logits, answer = rng.normal(size=(4, 6)), np.array([0, 3, 5, 1])
    want = -sum(np.log(softmax(logits[b])[answer[b]]) for b in range(4))
    assert cross_entropy(logits, answer) == pytest.approx(want, rel=1e-12)


def test_linear_start_is_linear_in_the_question():
    # Without the softmax every hop maps u to u + sum_i (u . m_i) c_i, a linear map, so the whole model before the
    # answer softmax is linear in u: doubling the question's embedding doubles the logits.
    rng = np.random.default_rng(8)
    cfg = config(vocab=12, dim=5, hops=2, mem=5, sent_len=4, encoding="bow", temporal=False)
    params = init(rng, cfg)
    story, _, _ = toy_batch(rng, 1)
    q_only = np.array([[WHERE, IS, 0, 0]])  # words that the story does not use, so E0 changes only u
    logits, attn = forward(params, story, q_only, cfg, linear=True, trace=True)
    assert not np.allclose(attn.sum(-1), 1.0)
    doubled = dict(params, E0=params["E0"].copy())
    doubled["E0"][[WHERE, IS]] *= 2.0
    np.testing.assert_allclose(forward(doubled, story, q_only, cfg, linear=True), 2.0 * logits, rtol=1e-12)
    assert not np.allclose(forward(doubled, story, q_only, cfg), 2.0 * forward(params, story, q_only, cfg))


def test_config_checks():
    with pytest.raises(ValueError):
        config(vocab=1)
    with pytest.raises(ValueError):
        config(tying="none")
    with pytest.raises(ValueError):
        forward(init(np.random.default_rng(0), config()), np.zeros((1, 4, 4), int), np.zeros((1, 4), int), config())


@pytest.mark.parametrize("c", SHAPES, ids=[c["name"] for c in SHAPES])
def test_shape_contract(c):
    cfg, k = cfg_of(c), c["config"]
    params = init(np.random.default_rng(0), cfg)
    story = np.zeros((k["B"], k["N"], k["J"]), dtype=np.int64)
    query = np.zeros((k["B"], k["J"]), dtype=np.int64)
    blocks = c["blocks"]
    lw = position_encoding(k["J"], k["D"])
    mem = encode(params["E0"], story, lw) + params["T0"]
    assert list(mem.shape) == blocks["embed"]["outputs"]["memory"]
    u = encode(params["E0"], query, lw)
    assert list(u.shape) == blocks["question"]["outputs"]["u"]
    o, p = hop(u, mem, mem)
    assert list((u + o).shape) == blocks["hop"]["outputs"]["u_next"]
    assert list(p.shape) == blocks["hop"]["outputs"]["attention"]
    logits, attn = forward(params, story, query, cfg, trace=True)
    assert list(logits.shape) == blocks["answer"]["outputs"]["logits"]
    assert attn.shape == (k["K"], k["B"], k["N"])
    assert param_count(cfg) == c["model"]["params"] == sum(v.size for v in params.values())
    assert flops(cfg, k["B"]) == c["model"]["flops_forward"]
    assert param_count(cfg_of(c, tying="layerwise")) == c["model"]["layerwise_params"]


def test_shared_fixture():
    params = dict(np.load(FIXTURES / "params-small.npz"))
    inp, want = np.load(FIXTURES / "input-small.npz"), np.load(FIXTURES / "expected-small.npz")
    cfg = cfg_of(SHAPES[0])
    logits, attn = forward(params, inp["story"], inp["query"], cfg, trace=True)
    np.testing.assert_allclose(logits, want["logits"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(attn, want["attention"], rtol=1e-12, atol=1e-12)
    assert cross_entropy(logits, inp["answer"]) == pytest.approx(float(want["loss"]), rel=1e-12)


def flop_case(c):
    """The forward call of a config of shapes.json, for aik flopcount."""
    cfg, k = cfg_of(c), c["config"]
    rng = np.random.default_rng(0)
    story = rng.integers(0, k["V"], (k["B"], k["N"], k["J"]))
    query = rng.integers(0, k["V"], (k["B"], k["J"]))
    return forward, (init(rng, cfg), story, query, cfg)
