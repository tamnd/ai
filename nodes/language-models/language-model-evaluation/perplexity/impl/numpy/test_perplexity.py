# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/perplexity/ (ai://NEW-perplexity).
"""Tests of perplexity.py: the vectors, the worked example of the page, the derivations, the edge cases and the errors."""

import json
import math
import pathlib

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from perplexity import AGGREGATES, BASES, bits_per_byte, score, windows

HERE = pathlib.Path(__file__).resolve().parent
VECTORS = next(d for d in (HERE / "vectors", HERE.parents[1] / "vectors") if d.is_dir())
L = math.log


def near(x):
    return pytest.approx(x, rel=1e-9, abs=1e-12)


def rows(name):
    with open(VECTORS / name, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.mark.parametrize("v", rows("edge.jsonl"), ids=lambda v: v["id"])
def test_vectors(v):
    got = score(v["predictions"], v["references"], **v["options"])
    assert got["perplexity"] == near(v["want"]["perplexity"])
    assert got["cross_entropy"] == near(v["want"]["cross_entropy"])
    assert got["per_sequence"] == near(v["want"]["per_sequence"])


def test_worked_example():
    a, b = [L(1 / 2), L(1 / 4), L(1 / 8)], [L(1 / 2), L(1 / 2)]
    tok = score([a, b], [3, 2], base="2")
    assert tok["cross_entropy"] == near(1.6)
    assert tok["perplexity"] == near(2**1.6) == near(3.0314331330207964)
    assert tok["per_sequence"] == near([4.0, 2.0])
    assert score([a, b], [3, 2], aggregate="mean-of-sequences")["perplexity"] == near(
        3.0
    )
    byte = score([a, b], [12, 8], base="2")
    assert byte["cross_entropy"] == near(0.4)
    assert byte["perplexity"] == near(1.3195079107728942)
    word = score([a, b], [2, 2], base="2")
    assert word["perplexity"] == near(4.0)
    assert word["perplexity"] == near(
        tok["perplexity"] ** (5 / 4)
    )  # PPL_W = PPL_T ** (N_T / N_W)
    assert score([a, b], [2, 2], aggregate="mean-of-sequences")["perplexity"] == near(
        5.0
    )


probs = st.lists(st.floats(1e-6, 1.0), min_size=1, max_size=40)


@settings(max_examples=200, deadline=None)
@given(p=probs)
def test_geometric_mean_and_bounds(p):
    """PPL is the geometric mean of 1/p, so 1/mean(p) <= PPL <= mean(1/p) and 1/max(p) <= PPL <= 1/min(p)."""
    p = np.array(p)
    ppl = score([np.log(p)], [len(p)])["perplexity"]
    assert ppl == pytest.approx(float(np.prod((1 / p) ** (1 / len(p)))), rel=1e-9)
    assert 1 / p.mean() <= ppl * (1 + 1e-12)
    assert ppl <= (1 / p).mean() * (1 + 1e-12)
    assert 1 / p.max() <= ppl * (1 + 1e-12) and ppl <= (1 / p.min()) * (1 + 1e-12)
    assert ppl >= 1 - 1e-12


@settings(max_examples=100, deadline=None)
@given(p=probs)
def test_base_invariance(p):
    lp = np.log(p)
    e, two = score([lp], [len(p)], base="e"), score([lp], [len(p)], base="2")
    assert e["perplexity"] == near(two["perplexity"])
    assert math.exp(e["cross_entropy"]) == pytest.approx(e["perplexity"], rel=1e-9)
    assert 2 ** two["cross_entropy"] == pytest.approx(e["perplexity"], rel=1e-9)


@pytest.mark.parametrize("v", [2, 10, 50257, 10**6])
def test_uniform_model_gives_vocabulary_size(v):
    assert score([[-math.log(v)] * 7], [7])["perplexity"] == pytest.approx(v, rel=1e-9)


@settings(max_examples=100, deadline=None)
@given(
    seqs=st.lists(
        st.lists(st.floats(1e-4, 1.0), min_size=3, max_size=3), min_size=1, max_size=8
    )
)
def test_equal_lengths_corpus_is_geometric_mean(seqs):
    """With equal unit counts the corpus value is the geometric mean of the per-sequence values, <= their mean."""
    pred = [np.log(s) for s in seqs]
    corpus = score(pred, [3] * len(seqs))
    mean = score(pred, [3] * len(seqs), aggregate="mean-of-sequences")
    per = np.array(corpus["per_sequence"])
    assert corpus["perplexity"] == pytest.approx(
        float(np.exp(np.log(per).mean())), rel=1e-9
    )
    assert corpus["perplexity"] <= mean["perplexity"] * (1 + 1e-12)


@settings(max_examples=100, deadline=None)
@given(
    seqs=st.lists(
        st.lists(st.floats(1e-4, 1.0), min_size=1, max_size=6), min_size=1, max_size=6
    ),
    data=st.data(),
)
def test_corpus_is_unit_weighted_and_between_extremes(seqs, data):
    units = [data.draw(st.integers(1, 50)) for _ in seqs]
    got = score([np.log(s) for s in seqs], units)
    per = np.array(got["per_sequence"])
    w = np.array(units) / sum(units)
    assert math.log(got["perplexity"]) == pytest.approx(
        float((w * np.log(per)).sum()), rel=1e-9, abs=1e-12
    )
    assert per.min() * (1 - 1e-12) <= got["perplexity"] <= per.max() * (1 + 1e-12)


@settings(max_examples=100, deadline=None)
@given(p=probs, words=st.integers(1, 40), nbytes=st.integers(1, 400))
def test_unit_conversion(p, words, nbytes):
    """PPL over U units is PPL_T ** (N_T / N_U), and bits per byte is (N_T / N_B) * loss / ln 2."""
    lp, n = np.log(p), len(p)
    tok = score([lp], [n])
    word = score([lp], [words])
    assert word["perplexity"] == pytest.approx(
        tok["perplexity"] ** (n / words), rel=1e-8
    )
    bpb = score([lp], [nbytes], base="2")["cross_entropy"]
    assert bits_per_byte(tok["cross_entropy"], n, nbytes) == pytest.approx(
        bpb, rel=1e-9, abs=1e-12
    )


def test_pile_conversion():
    """Gao et al. (2020): BPB = (L_T / L_B) * loss / ln 2 with L_T / L_B = 0.29335 GPT-2 tokens per byte."""
    assert bits_per_byte(2.0, 29335, 100000) == pytest.approx(
        0.29335 * 2.0 / math.log(2), rel=1e-12
    )


@settings(max_examples=200, deadline=None)
@given(n=st.integers(1, 300), w=st.integers(1, 64), data=st.data())
def test_windows(n, w, data):
    s = data.draw(st.integers(1, w))
    ws = windows(n, w, s)
    scored = [i for _, end, first in ws for i in range(first, end)]
    assert scored == list(range(n))  # every token scored exactly once, in order
    assert len(ws) == (1 if n <= w else 1 + math.ceil((n - w) / s))
    for k, (begin, end, first) in enumerate(ws):
        assert 0 <= begin <= first <= end <= n and end - begin <= w
        if k > 0:
            assert (
                first - begin >= w - s
            )  # the context inside the window before the first scored token


def test_zero_probability_gives_infinity():
    got = score([[L(0.5), -math.inf]], [2])
    assert got["perplexity"] == math.inf and got["per_sequence"] == [math.inf]


def test_order_does_not_matter():
    a, b = [L(0.3), L(0.2)], [L(0.9)]
    assert score([a, b], [2, 1])["perplexity"] == near(
        score([b, a], [1, 2])["perplexity"]
    )


@pytest.mark.parametrize(
    ("pred", "ref", "opts", "err"),
    [
        ([[0.1]], [1], {}, ValueError),
        ([[math.nan]], [1], {}, ValueError),
        ([[L(0.5)]], [0], {}, ValueError),
        ([[L(0.5)]], [1.0], {}, TypeError),
        ([[L(0.5)]], [True], {}, TypeError),
        ([[L(0.5)]], [1, 2], {}, ValueError),
        ([], [], {}, ValueError),
        ([[L(0.5)]], [1], {"aggregate": "median"}, ValueError),
        ([[L(0.5)]], [1], {"base": "10"}, ValueError),
    ],
)
def test_errors(pred, ref, opts, err):
    with pytest.raises(err):
        score(pred, ref, **opts)


def test_window_errors():
    for args in [(10, 0, 1), (10, 4, 0), (10, 4, 5), (0, 4, 2)]:
        with pytest.raises(ValueError):
            windows(*args)


def test_options_are_listed():
    assert AGGREGATES == ("corpus", "mean-of-sequences") and BASES == ("e", "2")
