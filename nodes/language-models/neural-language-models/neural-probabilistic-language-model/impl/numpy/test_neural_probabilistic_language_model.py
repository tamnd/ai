# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/neural-probabilistic-language-model/ (ai://NEW-neural-probabilistic-language-model).
"""Tests of the NumPy implementation: the shared fixture, the worked example of the page, every gradient against
finite differences, the parameter count and the output share of Bengio et al., normalisation, weight decay and the
learning-rate schedule, the rows touched by noise-contrastive estimation, the Huffman-tree softmax, learning the class
language better than a smoothed trigram, and the mixture bound."""

import itertools
import math
import pathlib
from collections import Counter

import numpy as np
import pytest

import neural_probabilistic_language_model as nl

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
V, M, HID, N, DECAY = 61, 8, 16, 3, 1e-4


@pytest.fixture(scope="module")
def data():
    return dict(np.load(FIXTURES / "input-small.npz")), dict(np.load(FIXTURES / "expected-small.npz"))


def loaded(inp, prefix, h, tie):
    m = nl.NPLM(V, M, h, N, direct=True, tie=tie)
    m.p = {k[len(prefix) + 1 :]: v.copy() for k, v in inp.items() if k.startswith(prefix + "_")}
    return m


def close(got, want):
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-15)


def test_shared_fixture(data):
    inp, want = data
    ctx, tgt = nl.windows(inp["train"], N, 0)
    cte, tte = nl.windows(inp["test"], N, 0)
    m = loaded(inp, "init", HID, False)
    close(m.logits(cte[:32]), want["logits0"])
    loss, g = m.loss_and_grads(ctx[:32], tgt[:32], DECAY)
    close(loss, want["loss0"])
    for k, v in g.items():
        close(v, want[f"grad0_{k}"])
    loss, g = m.nce_loss_and_grads(ctx[:32], tgt[:32], inp["noise"], inp["log_pn"], DECAY)
    close(loss, want["nce0"])
    for k, v in g.items():
        close(v, want[f"ncegrad0_{k}"])
    t = loaded(inp, "tinit", M, True)
    assert "U" not in t.p
    close(t.logits(cte[:32]), want["tlogits0"])
    loss, g = t.loss_and_grads(ctx[:32], tgt[:32], DECAY)
    close(loss, want["tloss0"])
    for k, v in g.items():
        close(v, want[f"tgrad0_{k}"])
    hist = m.train(ctx, tgt, 2, 0.3, 1e-3, DECAY, 16, orders=inp["orders"])
    np.testing.assert_allclose(hist, want["train_loss"], rtol=1e-10)
    for k, v in m.p.items():
        np.testing.assert_allclose(v, want[f"final_{k}"], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(m.token_logprobs(cte, tte), want["final_test_logprob"], rtol=1e-9)
    hs = nl.HierarchicalSoftmax(inp["hs_counts"], HID)
    hs.v, hs.c = inp["hs_v"].copy(), inp["hs_c"].copy()
    close(hs.all_log_probs(inp["hs_a"]), want["hs_all_logprob"])
    loss, g_a, g_v, g_c = hs.loss_and_grads(inp["hs_a"], inp["hs_tgt"])
    close(loss, want["hs_loss"])
    close(g_a, want["hs_grad_a"])
    close(g_v, want["hs_grad_v"])
    close(g_c, want["hs_grad_c"])


def worked_example():
    """Three words a, b, c, one word of context, m = 2, h = 1, no direct connections: tanh(ln 2) = 0.6, and U is
    chosen so that the logits are (0, ln 4, ln 3)."""
    m = nl.NPLM(3, 2, 1, 2, direct=False)
    m.p["C"] = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    m.p["H"] = np.array([[math.log(2), 0.0]])
    m.p["d"] = np.zeros(1)
    m.p["U"] = np.array([[0.0], [5 / 3 * math.log(4)], [5 / 3 * math.log(3)]])
    m.p["b"] = np.zeros(3)
    return m


def test_worked_example():
    m = worked_example()
    ctx, tgt = np.array([[0]]), np.array([1])  # after a, predict b
    _, _, a = m.hidden(ctx)
    assert a[0, 0] == pytest.approx(0.6, rel=1e-15)
    np.testing.assert_allclose(m.logits(ctx)[0], [0.0, math.log(4), math.log(3)], atol=1e-15)
    np.testing.assert_allclose(np.exp(m.log_probs(ctx))[0], [1 / 8, 1 / 2, 3 / 8], rtol=1e-14)
    loss, g = m.loss_and_grads(ctx, tgt)
    assert loss == pytest.approx(math.log(2), rel=1e-14)
    np.testing.assert_allclose(g["b"], [1 / 8, -1 / 2, 3 / 8], rtol=1e-14)
    g_a = -(1 / 2) * 5 / 3 * math.log(4) + (3 / 8) * 5 / 3 * math.log(3)
    np.testing.assert_allclose(g["U"][:, 0], 0.6 * np.array([1 / 8, -1 / 2, 3 / 8]), rtol=1e-14)
    assert g["d"][0] == pytest.approx(0.64 * g_a, rel=1e-13)
    np.testing.assert_allclose(g["H"], [[0.64 * g_a, 0.0]], rtol=1e-13, atol=1e-16)
    np.testing.assert_allclose(g["C"], [[0.64 * g_a * math.log(2), 0.0], [0, 0], [0, 0]], rtol=1e-13, atol=1e-16)
    assert m.size() == nl.n_params(3, 2, 1, 2, direct=False) == 15


def numeric_grad(f, arr, eps=1e-6):
    num = np.zeros_like(arr)
    for idx in np.ndindex(arr.shape):
        old = arr[idx]
        arr[idx] = old + eps
        hi = f()
        arr[idx] = old - eps
        lo = f()
        arr[idx] = old
        num[idx] = (hi - lo) / (2 * eps)
    return num


@pytest.mark.parametrize(("direct", "tie"), [(True, False), (False, False), (True, True), (False, True)])
def test_gradients(direct, tie):
    rng = np.random.default_rng(7)
    m = nl.NPLM(11, 4, 4 if tie else 6, 4, direct=direct, tie=tie, seed=3, scale=0.5)
    for v in m.p.values():
        v += rng.normal(0, 0.3, v.shape)
    ctx, tgt = rng.integers(0, 11, (7, 3)), rng.integers(0, 11, 7)
    noise, log_pn = rng.integers(0, 11, (7, 3)), np.log(rng.dirichlet(np.ones(11)))

    def ml():
        return m.loss_and_grads(ctx, tgt, 0.03)

    def nce():
        return m.nce_loss_and_grads(ctx, tgt, noise, log_pn, 0.03)

    for f in (ml, nce):
        _, g = f()
        assert set(g) == set(m.p)
        for k, v in m.p.items():
            np.testing.assert_allclose(g[k], numeric_grad(lambda f=f: f()[0], v), atol=2e-8)


def test_parameter_count():
    for V_, m_, h_, n_ in ((17964, 100, 60, 6), (16383, 60, 50, 5), (16383, 30, 100, 5), (7, 3, 2, 2)):
        assert (
            nl.NPLM(V_, m_, h_, n_).size()
            == nl.n_params(V_, m_, h_, n_)
            == V_ * (1 + n_ * m_ + h_) + h_ * (1 + (n_ - 1) * m_)
        )
        assert nl.NPLM(V_, m_, h_, n_, direct=False).size() == nl.n_params(V_, m_, h_, n_, direct=False)
    assert nl.NPLM(9, 5, 5, 3, tie=True).size() == nl.n_params(9, 5, 5, 3, tie=True)
    # The AP News network: 11,904,264 parameters, 95 MB as float64, and 99.7% of the multiply-adds in the outputs.
    assert nl.n_params(17964, 100, 60, 6) == 11_904_264
    assert round(100 * nl.output_fraction(17964, 100, 60, 6), 1) == 99.7
    with pytest.raises(ValueError):
        nl.NPLM(9, 5, 6, 3, tie=True)


def test_softmax_and_windows():
    rng = np.random.default_rng(1)
    m = nl.NPLM(23, 5, 7, 4, seed=2, scale=3.0)
    ctx = rng.integers(0, 23, (40, 3))
    p = np.exp(m.log_probs(ctx))
    np.testing.assert_allclose(p.sum(axis=1), 1.0, rtol=1e-14)
    assert p.min() > 0
    big = nl.log_softmax(np.array([[1000.0, 1001.0, 999.0]]))
    np.testing.assert_allclose(np.exp(big).sum(), 1.0)
    c, t = nl.windows(np.array([5, 6, 7]), 3, 0)
    np.testing.assert_array_equal(c, [[0, 0], [0, 5], [5, 6]])
    np.testing.assert_array_equal(t, [5, 6, 7])
    # x lists the newest context word first.
    _, x, _ = m.hidden(np.array([[1, 2, 3]]))
    np.testing.assert_array_equal(x[0], np.concatenate([m.p["C"][3], m.p["C"][2], m.p["C"][1]]))


def test_decay_and_schedule():
    rng = np.random.default_rng(2)
    m = nl.NPLM(9, 3, 4, 3, seed=1)
    ctx, tgt = rng.integers(0, 9, (5, 2)), rng.integers(0, 9, 5)
    l0, g0 = m.loss_and_grads(ctx, tgt, 0.0)
    l1, g1 = m.loss_and_grads(ctx, tgt, 0.5)
    for k in ("b", "d"):
        np.testing.assert_array_equal(g0[k], g1[k])
    for k in nl.WEIGHTS:
        np.testing.assert_allclose(g1[k] - g0[k], 0.5 * m.p[k], rtol=1e-12, atol=1e-15)
    assert l1 - l0 == pytest.approx(m.penalty(0.5))
    for t in (0, 1, 10, 10**8):
        m.updates = t
        assert m.learning_rate(1e-3, 1e-8) == 1e-3 / (1 + 1e-8 * t)


def test_nce_rows():
    rng = np.random.default_rng(3)
    m = nl.NPLM(30, 4, 5, 3, seed=4)
    ctx, tgt, noise = rng.integers(0, 30, (6, 2)), rng.integers(0, 30, 6), rng.integers(0, 30, (6, 4))
    _, g = m.nce_loss_and_grads(ctx, tgt, noise, np.full(30, -math.log(30)))
    touched = np.zeros(30, dtype=bool)
    touched[tgt] = touched[noise.ravel()] = True
    for k in ("U", "b", "W"):
        assert np.all(g[k][~touched] == 0) and np.any(g[k][touched] != 0)
    # A word scored exactly as the noise would be scored has posterior 1 / 2 of being data.
    k = 4
    delta = np.log(1 / 30) - (np.log(k) + np.log(1 / (30 * k)))
    assert nl.sigmoid(np.array([delta]))[0] == pytest.approx(0.5)


def test_hierarchical_softmax():
    rng = np.random.default_rng(4)
    for V_ in (2, 3, 5, 8, 33, 100):
        _, code, mask = nl.huffman(np.ones(V_))
        depth = mask.sum(axis=1)
        assert depth.min() >= math.floor(math.log2(V_)) and depth.max() <= math.ceil(math.log2(V_))
        assert np.sum(2.0**-depth) == pytest.approx(1.0)  # a complete prefix code
        assert len({tuple(c[mk > 0]) for c, mk in zip(code, mask, strict=True)}) == V_
    counts = rng.zipf(1.5, 200).astype(float)
    hs = nl.HierarchicalSoftmax(counts, 6, seed=1, scale=1.0)
    q = counts / counts.sum()
    entropy = -(q * np.log2(q)).sum()
    mean_depth = (q * hs.depth()).sum()
    assert entropy <= mean_depth < entropy + 1
    a = np.tanh(rng.normal(size=(9, 6)))
    lp = hs.all_log_probs(a)
    np.testing.assert_allclose(np.exp(lp).sum(axis=1), 1.0, rtol=1e-12)
    tgt = rng.integers(0, 200, 9)
    np.testing.assert_allclose(hs.log_prob(a, tgt), lp[np.arange(9), tgt], rtol=1e-12)
    small = nl.HierarchicalSoftmax(rng.integers(1, 9, 7), 3, seed=2, scale=1.0)
    a, tgt = rng.normal(size=(4, 3)), rng.integers(0, 7, 4)
    _, g_a, g_v, g_c = small.loss_and_grads(a, tgt)
    for arr, g in ((a, g_a), (small.v, g_v), (small.c, g_c)):
        np.testing.assert_allclose(g, numeric_grad(lambda: small.loss_and_grads(a, tgt)[0], arr), atol=1e-8)
    with pytest.raises(ValueError):
        nl.huffman(np.ones(1))


def class_language(n, seed, source=2003):
    """n tokens of a class language like the fixture's, from the transition rows drawn with the source seed."""
    trans = np.random.default_rng(source).dirichlet(np.full(7, 0.3), size=(7, 7))
    rng = np.random.default_rng(seed)
    out, lp, u, v = [], [], 6, 6
    for _ in range(n):
        c = int(rng.choice(7, p=trans[u, v]))
        out.append(0 if c == 6 else 1 + 10 * c + int(rng.integers(10)))
        lp.append(math.log(trans[u, v, c] / (1 if c == 6 else 10)))
        u, v = v, c
    return np.array(out), np.array(lp)


def trigram(train, test, D=0.75):
    """Interpolated absolute discounting down to an add-one unigram."""
    s = np.concatenate([[0, 0], train])
    c3, c2, c1 = (
        Counter(zip(s[:-2], s[1:-1], s[2:], strict=True)),
        Counter(itertools.pairwise(s[1:])),
        Counter(s[2:].tolist()),
    )
    h2, n2, h1, n1 = Counter(), Counter(), Counter(), Counter()
    for (a, b, _), c in c3.items():
        h2[(a, b)] += c
        n2[(a, b)] += 1
    for (b, _), c in c2.items():
        h1[b] += c
        n1[b] += 1
    t = np.concatenate([[0, 0], test])
    out = []
    for i in range(2, len(t)):
        a, b, w = t[i - 2], t[i - 1], t[i]
        p = (c1[w] + 1) / (len(train) + V)
        if h1[b]:
            p = (max(c2[(b, w)] - D, 0) + D * n1[b] * p) / h1[b]
        if h2[(a, b)]:
            p = (max(c3[(a, b, w)] - D, 0) + D * n2[(a, b)] * p) / h2[(a, b)]
        out.append(math.log(p))
    return np.array(out)


def test_learns_word_classes():
    train, _ = class_language(20000, 11)
    test, true_lp = class_language(4000, 12)
    ctx, tgt = nl.windows(train, N, 0)
    cte, tte = nl.windows(test, N, 0)
    m = nl.NPLM(V, M, HID, N, seed=0)
    hist = m.train(ctx, tgt, 6, 0.3, 0.0, 1e-5, 16, np.random.default_rng(0))
    assert hist[-1] < hist[0]
    nn_lp, ng_lp = m.token_logprobs(cte, tte), trigram(train, test)
    ppl_nn, ppl_ng, ppl_true = nl.perplexity(nn_lp), nl.perplexity(ng_lp), nl.perplexity(true_lp)
    assert ppl_true < ppl_nn < 0.8 * ppl_ng
    # The nearest feature vector of almost every word belongs to a word of the same class.
    C = m.p["C"][1:] / np.linalg.norm(m.p["C"][1:], axis=1, keepdims=True)
    sim = C @ C.T
    np.fill_diagonal(sim, -2)
    assert np.mean(sim.argmax(axis=1) // 10 == np.arange(60) // 10) >= 0.9
    # Jensen: the half-half mixture's perplexity is at most the geometric mean of the two perplexities.
    assert nl.perplexity(nl.mix(nn_lp, ng_lp)) <= math.sqrt(ppl_nn * ppl_ng)


def test_mixture():
    rng = np.random.default_rng(5)
    for _ in range(50):
        a, b = np.log(rng.uniform(1e-4, 1, 100)), np.log(rng.uniform(1e-4, 1, 100))
        w = rng.uniform(0.01, 0.99)
        mixed = nl.mix(a, b, w)
        np.testing.assert_allclose(np.exp(mixed), w * np.exp(a) + (1 - w) * np.exp(b), rtol=1e-12)
        assert nl.perplexity(mixed) <= nl.perplexity(a) ** w * nl.perplexity(b) ** (1 - w) * (1 + 1e-12)
