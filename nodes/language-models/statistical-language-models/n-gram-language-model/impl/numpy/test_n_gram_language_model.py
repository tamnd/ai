# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/n-gram-language-model/ (ai://NEW-n-gram-language-model).
"""Tests of the NumPy implementation: the shared fixture for every order and model, the worked example of the page,
normalisation for every context and any discounts in range, the closed-form discounts, Stupid Backoff, the suffix
array against a plain sort, the infinity-gram, and sampling."""

import pathlib

import numpy as np
import pytest

import n_gram_language_model as ng

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
CONTEXTS = [[], ["w3"], ["w0", "w5"], ["w9", "w9", "w9"], ["zzz", "w1", "w2"]]
TOY = [["the", "cat", "sat"], ["the", "dog", "sat"], ["the", "cat", "ran"]]


def corpus(prefix):
    inp = np.load(FIXTURES / "input-small.npz")
    ids, lens = inp[f"{prefix}_ids"], inp[f"{prefix}_len"]
    ends = np.cumsum(lens)
    return [[f"w{i}" for i in ids[e - n : e]] for e, n in zip(ends, lens, strict=True)]


@pytest.fixture(scope="module")
def data():
    return corpus("train"), corpus("test"), dict(np.load(FIXTURES / "expected-small.npz"))


def test_shared_fixture(data):
    train, test, want = data
    for order in (1, 2, 3, 4):
        for kind in ("interpolated", "modified"):
            m = ng.KneserNey(train, order, kind)
            np.testing.assert_allclose(m.token_logprobs(test), want[f"{kind}{order}_logprob"], rtol=1e-12)
            got = np.array([m.discounts[n] for n in range(1, order + 1)])
            np.testing.assert_allclose(got, want[f"{kind}{order}_discounts"], rtol=1e-14)
    sb = ng.StupidBackoff(train, 3)
    win = ng.windows([sb.vocab.encode(s, 3) for s in test], 3)
    np.testing.assert_allclose(sb.score(win[:, -1], win[:, :-1]), want["sb3_score"], rtol=1e-14)
    ig = ng.InfiniGram(train)
    for j, ctx in enumerate(CONTEXTS):
        ids, p, n_eff = ig.distribution(ctx)
        words = np.array([int(ig.vocab.itos[i][1:]) for i in ids])
        order = np.argsort(words)
        assert n_eff == int(want[f"inf{j}_n"])
        np.testing.assert_array_equal(words[order], want[f"inf{j}_ids"])
        np.testing.assert_allclose(p[order], want[f"inf{j}_prob"], rtol=1e-14)
    vocab = ng.Vocab(train)
    _, c = ng.count(ng.windows([vocab.encode(s, 3) for s in train], 3))
    assert (c == 1).sum() / c.sum() == pytest.approx(float(want["gt_leftover3"]), rel=1e-15)


def test_worked_example():
    m = ng.KneserNey(TOY, 2, "interpolated")
    s = m.vocab.stoi
    assert m.vocab.itos == ["<unk>", "<s>", "</s>", "cat", "dog", "ran", "sat", "the"]
    np.testing.assert_array_equal(m.support, [0, 2, 3, 4, 5, 6, 7])  # BOS is never predicted
    np.testing.assert_allclose(m.discounts[2], [0, 5 / 9, 5 / 9, 5 / 9])
    np.testing.assert_allclose(m.discounts[1], [0, 1 / 2, 1 / 2, 1 / 2])
    ids, p = m.distribution(np.array([s["the"]]))
    want = {"cat": 793 / 1512, "dog": 289 / 1512, "the": 10 / 27 * 13 / 112, "<unk>": 10 / 27 * 3 / 56}
    for w, q in want.items():
        assert p[ids == s[w]][0] == pytest.approx(q, rel=1e-14)
    assert p.sum() == pytest.approx(1.0, abs=1e-15)
    want_lp = np.log([281 / 336, 289 / 1512, 65 / 1008, 583 / 1008])
    np.testing.assert_allclose(m.token_logprobs([["the", "dog", "ran"]]), want_lp, rtol=1e-14)
    assert m.perplexity([["the", "dog", "ran"]]) == pytest.approx(3.5987900799, rel=1e-10)
    # An unseen word gets the probability of UNK in every context.
    assert m.prob([s["dog"]], [[s["dog"]]])[0] == pytest.approx(5 / 9 * 13 / 112)
    unseen = m.token_logprobs([["the", "zebra"]])[1]
    assert unseen == pytest.approx(np.log(10 / 27 * 3 / 56))


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_normalisation(order):
    rng = np.random.default_rng(order)
    sents = [[f"x{i}" for i in rng.integers(0, 9, rng.integers(1, 9))] for _ in range(80)]
    for _ in range(3):
        discount = {n: np.array([0, 1, 2, 3]) * rng.random(4) for n in range(1, order + 1)}
        m = ng.KneserNey(sents, order, discount)
        contexts = [g[:-1] for n in m.grams for g in m.grams[n] if len(g) == order]
        contexts += [rng.integers(0, m.V, order - 1) for _ in range(30)]
        for ctx in contexts:
            ids, p = m.distribution(np.asarray(ctx, dtype=np.int64))
            assert p.sum() == pytest.approx(1.0, abs=1e-12) and p.min() > 0


def test_discounts():
    t = np.array([0, 40, 20, 12, 8])
    y = 40 / 80
    np.testing.assert_allclose(
        ng.closed_form_discounts(t), [0, 1 - 2 * y * 20 / 40, 2 - 3 * y * 12 / 20, 3 - 4 * y * 8 / 12]
    )
    np.testing.assert_allclose(ng.closed_form_discounts(t, modified=False), [0, y, y, y])
    for bad in ([0, 0, 0, 5, 5], [0, 4, 2, 0, 0], [0, 10, 1, 50, 0]):
        with pytest.raises(ValueError):
            ng.closed_form_discounts(np.array(bad))
    with pytest.raises(ValueError):
        ng.KneserNey([[f"w{i}" for i in range(3000)]], 6)


def test_stupid_backoff():
    sb = ng.StupidBackoff(TOY, 2)
    s = sb.vocab.stoi
    words = np.array([s[w] for w in ("the", "cat", "dog", "sat", "ran", "</s>")])
    sc = sb.score(words, np.full((6, 1), s["the"]))
    np.testing.assert_allclose(sc, [0.4 * 3 / 12, 2 / 3, 1 / 3, 0.4 * 2 / 12, 0.4 * 1 / 12, 0.4 * 3 / 12])
    assert sc.sum() == pytest.approx(1 + 0.4 * 9 / 12)  # the scores are not a distribution


def test_suffix_array():
    rng = np.random.default_rng(0)
    for n in (0, 1, 2, 7, 100, 513):
        toks = rng.integers(-1, 4, n)
        brute = sorted(range(n), key=lambda i: list(toks[i:]))
        np.testing.assert_array_equal(ng.suffix_array(toks), np.array(brute, dtype=np.int64))
    np.testing.assert_array_equal(ng.suffix_array(np.zeros(9, dtype=np.int64)), np.arange(8, -1, -1))


def test_infinigram():
    docs = [["a", "b", "c", "a", "b", "d"], ["x", "a", "b", "c"], ["b", "c", "a"]]
    ig = ng.InfiniGram(docs)
    s = ig.vocab.stoi
    assert ig.count([s["a"], s["b"]]) == 3 and ig.count([]) == 16
    ids, p, n = ig.distribution(["q", "x", "a", "b"])
    assert n == 4 and list(ids) == [s["c"]] and list(p) == [1.0]
    ids, p, n = ig.distribution(["z", "a", "b"])
    assert n == 3 and dict(zip(ids.tolist(), p.tolist(), strict=True)) == pytest.approx({s["c"]: 2 / 3, s["d"]: 1 / 3})
    ids, p, n = ig.distribution(["d"])
    assert n == 1 and p.sum() == pytest.approx(1.0) and p[ids == s["b"]][0] == pytest.approx(4 / 13)
    rng = np.random.default_rng(1)
    docs = [list(rng.choice(list("abcd"), rng.integers(1, 30))) for _ in range(20)]
    ig = ng.InfiniGram(docs)
    for _ in range(100):
        ctx = list(rng.choice(list("abcde"), rng.integers(0, 12)))
        ids, p, n = ig.distribution(ctx)
        assert p.sum() == pytest.approx(1.0, abs=1e-12) and n <= len(ctx) + 1


def test_sampling():
    m = ng.KneserNey(TOY, 2, "interpolated")
    rng = np.random.default_rng(4)
    draws = [m.sample(rng) for _ in range(2000)]
    ids, p = m.distribution(np.array([ng.BOS]))
    first = [d[0] if d else "</s>" for d in draws]
    for w in ("the", "cat", "sat"):
        assert first.count(w) / 2000 == pytest.approx(p[ids == m.vocab.stoi[w]][0], abs=0.04)
    assert all("<unk>" not in d and len(d) <= 50 for d in draws)
