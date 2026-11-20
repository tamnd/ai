# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/n-gram-language-model/ (ai://NEW-n-gram-language-model).
"""Tests of the plain Python reference: the shared fixture, the worked example of the page in exact fractions, an
independent trigram Kneser-Ney written from the equations of Heafield et al. (2013), normalisation for every context
and any discounts in range, the closed-form discounts, Kneser-Ney against maximum likelihood and add-k on held-out
text, the effect of order and data size, Stupid Backoff, the infinity-gram, Turing's leftover, and sampling."""

import ast
import itertools
import math
import pathlib
import random
import struct
import zipfile
from collections import Counter
from fractions import Fraction

import pytest

import n_gram_language_model as ng

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(
    d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir()
)
CONTEXTS = [[], ["w3"], ["w0", "w5"], ["w9", "w9", "w9"], ["zzz", "w1", "w2"]]
TOY = [["the", "cat", "sat"], ["the", "dog", "sat"], ["the", "cat", "ran"]]


def load_npz(path):
    """Read an .npz of little-endian int or float arrays with the standard library, as flat lists or scalars."""
    out = {}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            raw = z.read(name)
            hlen = (
                struct.unpack("<H", raw[8:10])[0]
                if raw[6] == 1
                else struct.unpack("<I", raw[8:12])[0]
            )
            start = 10 + hlen if raw[6] == 1 else 12 + hlen
            head = ast.literal_eval(raw[start - hlen : start].decode("latin1"))
            kind, size = head["descr"][1], int(head["descr"][2:])
            assert (
                head["descr"][0] in "<|" and not head["fortran_order"] and kind in "if"
            ), head
            n = math.prod(head["shape"])
            code = {
                ("i", 1): "b",
                ("i", 2): "h",
                ("i", 4): "i",
                ("i", 8): "q",
                ("f", 8): "d",
            }[kind, size]
            vals = list(struct.unpack(f"<{n}{code}", raw[start:]))
            out[name.removesuffix(".npy")] = vals if head["shape"] else vals[0]
    return out


def flat(x):
    return (
        [v for item in x for v in flat(item)] if isinstance(x, (list, tuple)) else [x]
    )


def mean(xs):
    return sum(xs) / len(xs)


def assert_close(got, want, rtol):
    got, want = flat(got), flat(want)
    assert len(got) == len(want)
    for a, b in zip(got, want, strict=True):
        assert abs(a - b) <= rtol * abs(b), (a, b)


def corpus(prefix):
    inp = load_npz(FIXTURES / "input-small.npz")
    ids, lens = inp[f"{prefix}_ids"], inp[f"{prefix}_len"]
    ends = itertools.accumulate(lens)
    return [[f"w{i}" for i in ids[e - n : e]] for e, n in zip(ends, lens, strict=True)]


@pytest.fixture(scope="module")
def data():
    return corpus("train"), corpus("test")


def token_logprobs(model, sentences):
    out = []
    for s in sentences:
        toks = ng.pad(s, model.order)
        out += [
            math.log(model.prob(toks[i], toks[i - model.order + 1 : i]))
            for i in range(model.order - 1, len(toks))
        ]
    return out


def test_shared_fixture(data):
    train, test = data
    want = load_npz(FIXTURES / "expected-small.npz")
    for order in (1, 2, 3):
        for kind in ("interpolated", "modified"):
            m = ng.KneserNey(train, order, kind)
            assert_close(
                token_logprobs(m, test), want[f"{kind}{order}_logprob"], rtol=1e-12
            )
            got = [m.discounts[n] for n in range(1, order + 1)]
            assert_close(got, want[f"{kind}{order}_discounts"], rtol=1e-14)
    sb = ng.StupidBackoff(train, 3)
    sc = []
    for s in test:
        toks = ng.pad(s, 3)
        sc += [sb.score(toks[i], toks[i - 2 : i]) for i in range(2, len(toks))]
    assert_close(sc, want["sb3_score"], rtol=1e-14)
    ig = ng.InfiniGram(train)
    for j, ctx in enumerate(CONTEXTS):
        dist, n_eff = ig.distribution(ctx)
        assert n_eff == int(want[f"inf{j}_n"])
        assert_close(
            [dist[f"w{i}"] for i in want[f"inf{j}_ids"]],
            want[f"inf{j}_prob"],
            rtol=1e-14,
        )
        assert len(dist) == len(want[f"inf{j}_ids"])
    assert ng.good_turing_leftover(ng.count_ngrams(train, 3)) == pytest.approx(
        float(want["gt_leftover3"]), rel=1e-15
    )


def test_worked_example():
    """The bigram of the page: interpolated Kneser-Ney on three sentences, checked in exact fractions."""
    m = ng.KneserNey(TOY, 2, "interpolated")
    F = Fraction
    assert m.entries[1] == {
        ("the",): 1,
        ("cat",): 1,
        ("dog",): 1,
        ("sat",): 2,
        ("ran",): 1,
        (ng.EOS,): 2,
    }
    assert m.entries[2][(ng.BOS, "the")] == 3 and m.entries[2][("the", "cat")] == 2
    assert dict(m.t[2]) == {1: 5, 2: 2, 3: 1} and dict(m.t[1]) == {1: 4, 2: 2}
    d2, d1 = F(5, 9), F(1, 2)
    assert m.discounts[2][1] == pytest.approx(float(d2)) and m.discounts[1][
        1
    ] == pytest.approx(float(d1))
    assert len(m.vocab) == 7
    uni = {
        w: (F(a) - d1) / 8 + F(3, 8) / 7
        for w, a in [("the", 1), ("cat", 1), ("dog", 1), ("sat", 2), ("ran", 1)]
    }
    uni[ng.EOS], uni[ng.UNK] = (F(2) - d1) / 8 + F(3, 8) / 7, F(3, 8) / 7
    for w, p in uni.items():
        assert m.prob(w) == pytest.approx(float(p), rel=1e-14)
    assert sum(uni.values()) == 1
    b_the = 2 * d2 / 3
    want = {
        "cat": (2 - d2) / 3 + b_the * uni["cat"],
        "dog": (1 - d2) / 3 + b_the * uni["dog"],
    }
    want.update({w: b_the * uni[w] for w in ("the", "sat", "ran", ng.EOS, ng.UNK)})
    for w, p in want.items():
        assert m.prob(w, ["the"]) == pytest.approx(float(p), rel=1e-14)
    assert sum(want.values()) == 1
    assert float(want["cat"]) == pytest.approx(0.524471, abs=5e-7)
    held = ["the", "dog", "ran"]
    p_the = (3 - d2) / 3 + d2 / 3 * uni["the"]
    p_ran = d2 * uni["ran"]
    p_end = (1 - d2) + d2 * uni[ng.EOS]
    lp = sum(math.log(p) for p in (p_the, want["dog"], p_ran, p_end))
    assert m.sentence_logprob(held) == (pytest.approx(lp, rel=1e-14), 4)
    assert m.perplexity([held]) == pytest.approx(math.exp(-lp / 4), rel=1e-14)
    assert m.perplexity([held]) == pytest.approx(3.599, abs=5e-4)
    assert ng.mle_prob(ng.count_ngrams(TOY, 2), "cat", ["the"]) == pytest.approx(2 / 3)
    assert ng.additive_prob(
        ng.count_ngrams(TOY, 2), 7, "cat", ["the"]
    ) == pytest.approx(0.3)


def independent_trigram(sentences, discount):
    """Trigram interpolated Kneser-Ney written straight from the equations, counting from the padded text."""
    tri, left = Counter(), {}
    for s in sentences:
        toks = ng.pad(s, 3)
        for i in range(2, len(toks)):
            tri[tuple(toks[i - 2 : i + 1])] += 1
    for x, y, z in tri:
        left.setdefault((y, z), set()).add(x)
    bi = {
        g: (sum(c for t, c in tri.items() if t[1:] == g) if g[0] == ng.BOS else len(v))
        for g, v in left.items()
    }
    left1 = {}
    for y, z in bi:
        left1.setdefault(z, set()).add(y)
    uni = {(z,): len(v) for z, v in left1.items()}
    vocab = {g[0] for g in uni} | {ng.UNK}

    def level(table, ctx, w, lower):
        rows = {g: a for g, a in table.items() if g[:-1] == ctx}
        if not rows:
            return lower
        total = sum(rows.values())
        u = max(rows.get(ctx + (w,), 0) - discount, 0) / total
        return u + discount * len(rows) / total * lower

    def prob(w, x, y):
        p1 = level(uni, (), w, 1 / len(vocab))
        p2 = level(bi, (y,), w, p1)
        return level(dict(tri), (x, y), w, p2)

    return prob, vocab


def test_independent_trigram():
    rng = random.Random(5)
    words = ["a", "b", "c", "d", "e", "f"]
    sents = [
        [rng.choice(words[: rng.randint(2, 6)]) for _ in range(rng.randint(1, 6))]
        for _ in range(60)
    ]
    disc = 0.7
    m = ng.KneserNey(sents, 3, {n: [0, disc, disc, disc] for n in (1, 2, 3)})
    prob, vocab = independent_trigram(sents, disc)
    assert set(m.vocab) == vocab
    for x in [ng.BOS, *words]:
        for y in [ng.BOS, *words] if x == ng.BOS else words:
            for w in [*words, ng.EOS, ng.UNK]:
                assert m.prob(w, [x, y]) == pytest.approx(prob(w, x, y), rel=1e-13)


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_normalisation(order):
    rng = random.Random(order)
    words = [f"x{i}" for i in range(9)]
    sents = [[rng.choice(words) for _ in range(rng.randint(1, 8))] for _ in range(80)]
    for _ in range(2):
        discount = {
            n: [0, rng.random(), 2 * rng.random(), 3 * rng.random()]
            for n in range(1, 5)
        }
        m = ng.KneserNey(sents, order, discount)
        contexts = {g[:-1] for n in m.entries for g in m.entries[n]} | {
            ("zz",) * (order - 1),
            ("x1", "q", "x2"),
        }
        for ctx in contexts:
            dist = m.distribution(ctx)
            assert sum(dist.values()) == pytest.approx(1.0, abs=1e-12)
            assert min(dist.values()) > 0


def test_discounts(data):
    train, _ = data
    m = ng.KneserNey(train, 3, "modified")
    for n in (1, 2, 3):
        t = m.t[n]
        y = t[1] / (t[1] + 2 * t[2])
        want = [0.0] + [k - (k + 1) * y * t[k + 1] / t[k] for k in (1, 2, 3)]
        assert m.discounts[n] == pytest.approx(want, rel=1e-14)
        assert all(0 < d < k for k, d in enumerate(m.discounts[n]) if k)
        assert ng.KneserNey(train, 3, "interpolated").discounts[n] == pytest.approx(
            [0, y, y, y]
        )
    with pytest.raises(ValueError):
        ng.closed_form_discounts({1: 0, 2: 0})
    with pytest.raises(ValueError):
        ng.closed_form_discounts({1: 4, 2: 2, 3: 0})


def test_kn_against_mle_and_additive(data):
    train, test = data
    counts = ng.count_ngrams(train, 3)
    kn = ng.KneserNey(train, 3, "modified")
    ctx_total = Counter()
    for g, c in counts.items():
        ctx_total[g[:-1]] += c
    tokens = [
        (tuple(t[i - 2 : i]), t[i])
        for s in test
        for t in [ng.pad(s, 3)]
        for i in range(2, len(t))
    ]
    unseen = sum(counts.get(h + (w,), 0) == 0 for h, w in tokens)
    assert (
        unseen > len(tokens) / 4
    )  # maximum likelihood gives each of these probability zero (or 0 / 0)
    V = len(kn.vocab)

    def additive_ppl(k):
        lp, n = 0.0, 0
        for s in test:
            toks = ng.pad(s, 3)
            for i in range(2, len(toks)):
                h = tuple(toks[i - 2 : i])
                lp += math.log(
                    (counts.get(h + (toks[i],), 0) + k) / (ctx_total[h] + k * V)
                )
                n += 1
        return math.exp(-lp / n)

    best_additive = min(additive_ppl(k) for k in (1.0, 0.1, 0.01, 0.001))
    assert additive_ppl(1.0) > 4 * kn.perplexity(test)
    assert best_additive > 2 * kn.perplexity(test)


def test_order_and_data(data):
    train, test = data
    want = load_npz(FIXTURES / "expected-small.npz")
    ppl = {
        (o, k): math.exp(-mean(want[f"{k}{o}_logprob"]))
        for o in (1, 2, 3, 4)
        for k in ("interpolated", "modified")
    }
    true = math.exp(-mean(load_npz(FIXTURES / "input-small.npz")["true_logprob"]))
    for k in ("interpolated", "modified"):
        assert ppl[(1, k)] > ppl[(2, k)] > ppl[(3, k)] > true
    for o in (1, 2, 3, 4):
        assert ppl[(o, "modified")] < ppl[(o, "interpolated")]
    small = ng.KneserNey(train[:600], 3, "modified").perplexity(test)
    assert small > ppl[(3, "modified")] * 1.3


def test_stupid_backoff():
    sb = ng.StupidBackoff(TOY, 2, alpha=0.4)
    assert sb.score("cat", ["the"]) == pytest.approx(2 / 3)
    assert sb.score("sat", ["the"]) == pytest.approx(0.4 * 2 / 12)
    total = sum(
        sb.score(w, ["the"]) for w in ["the", "cat", "dog", "sat", "ran", ng.EOS]
    )
    assert total == pytest.approx(
        1 + 0.4 * (1 - 3 / 12)
    )  # the scores are not a distribution
    sb3 = ng.StupidBackoff(TOY, 3, alpha=0.4)
    assert sb3.score("ran", ["dog", "sat"]) == pytest.approx(0.4 * 0.4 * 1 / 12)


def test_infinigram():
    docs = [["a", "b", "c", "a", "b", "d"], ["x", "a", "b", "c"], ["b", "c", "a"]]
    ig = ng.InfiniGram(docs)
    assert (
        ig.count(["a", "b"]) == 3
        and ig.count(["a", "b", "c"]) == 2
        and ig.count([]) == 16
    )
    dist, n = ig.distribution(["q", "x", "a", "b"])
    assert n == 4 and dist == {
        "c": 1.0
    }  # a sparse estimate from the longest suffix that is followed by a token
    dist, n = ig.distribution(["z", "a", "b"])
    assert n == 3 and dist == pytest.approx({"c": 2 / 3, "d": 1 / 3})
    dist, n = ig.distribution(
        ["b", "c"]
    )  # "b c" ends a document once: only the two continued occurrences count
    assert n == 3 and dist == {"a": 1.0}
    dist, n = ig.distribution(
        ["d"]
    )  # "d" is only ever last in its document, so the estimate backs off to unigrams
    assert (
        n == 1
        and sum(dist.values()) == pytest.approx(1.0)
        and dist["b"] == pytest.approx(4 / 13)
    )
    rng = random.Random(1)
    docs = [[rng.choice("abcd") for _ in range(rng.randint(1, 30))] for _ in range(20)]
    ig = ng.InfiniGram(docs)
    flat = [w for d in docs for w in d + [None]]
    brute = sorted(
        range(len(flat)),
        key=lambda i: [(0, "") if w is None else (1, w) for w in flat[i:]],
    )
    assert ig.sa == brute
    for _ in range(200):
        ctx = [rng.choice("abcde") for _ in range(rng.randint(0, 12))]
        dist, n = ig.distribution(ctx)
        assert sum(dist.values()) == pytest.approx(1.0, abs=1e-12)
        s = ctx[len(ctx) - n + 1 :]
        assert ig.next_counts(s) and all(
            not ig.next_counts(ctx[i:]) for i in range(len(ctx) - n + 1)
        )


def test_good_turing_leave_one_out():
    rng = random.Random(2)
    toks = [f"t{min(int(rng.paretovariate(1.0)), 500)}" for _ in range(3000)]
    counts = Counter((t,) for t in toks)
    novel = sum(
        1 for t in toks if counts[(t,)] == 1
    )  # removing token i leaves its type unseen iff it occurred once
    assert ng.good_turing_leftover(counts) == novel / len(toks)


def test_sampling():
    m = ng.KneserNey(TOY, 2, "interpolated")
    rng = random.Random(4)
    draws = [m.sample(rng) for _ in range(4000)]
    p = m.distribution([ng.BOS])
    freq = Counter(s[0] if s else ng.EOS for s in draws)
    for w in ("the", "cat", "sat"):
        assert freq[w] / 4000 == pytest.approx(p[w], abs=0.03)
    assert all(ng.UNK not in s and len(s) <= 50 for s in draws)
