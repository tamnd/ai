# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/bleu/ (ai://V3JG9E).
"""Tests of bleu.py: the sacreBLEU vectors, the worked example of the page, the edge cases and the properties."""

import json
import math
import pathlib

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bleu import TOKENIZERS, score, statistics

HERE = pathlib.Path(__file__).resolve().parent
VECTORS = next(d for d in (HERE / "vectors", HERE.parents[1] / "vectors") if d.is_dir())


def near(x):
    return pytest.approx(x, rel=1e-9, abs=1e-9)


def rows(name):
    with open(VECTORS / name, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def check(got, want):
    assert got["score"] == near(want["score"])
    assert got["precisions"] == near(want["precisions"])
    assert got["bp"] == near(want["bp"])
    assert got["sys_len"] == want["sys_len"]
    assert got["ref_len"] == want["ref_len"]


@pytest.mark.parametrize("v", rows("official.jsonl") + rows("edge.jsonl"), ids=lambda v: v["id"])
def test_vectors(v):
    check(score(v["predictions"], v["references"], **v["options"]), v["want"])


def test_worked_example():
    preds, refs = ["the cat sat on the mat"], [["the cat sat on the red mat"]]
    assert statistics(preds, refs) == (6, 7, [6, 4, 3, 2], [6, 5, 4, 3])
    got = score(preds, refs)
    assert got["precisions"] == near([100.0, 80.0, 75.0, 200.0 / 3.0])
    assert got["bp"] == near(math.exp(-1.0 / 6.0))
    assert got["score"] == near(100.0 * math.exp(-1.0 / 6.0) * 0.4**0.25)
    assert round(got["score"], 2) == 67.32
    assert got["score"] == near(rows("official.jsonl")[0]["want"]["score"])


def test_clipping():
    preds = ["the the the the the the the"]
    refs = [["The cat is on the mat.", "There is a cat on the mat."]]
    _, _, correct, total = statistics(preds, refs, lowercase=True)
    assert (correct[0], total[0]) == (2, 7)  # the word occurs at most twice in one reference
    assert score(preds, refs, lowercase=True)["precisions"][0] == near(200.0 / 7.0)
    assert statistics(preds, refs)[2][0] == 1  # case-sensitive: only the lowercase "the" of each reference


def test_brevity_penalty_is_corpus_level():
    # One segment is two tokens short and the other two tokens long: the corpus lengths are equal, so BP is 1.
    preds = ["a b c d", "a b c d e f g h"]
    refs = [["a b c d e f"], ["a b c d e f"]]
    got = score(preds, refs)
    assert (got["sys_len"], got["ref_len"], got["bp"]) == (12, 12, 1.0)
    assert score(preds[:1], refs[:1])["bp"] == near(math.exp(1.0 - 6.0 / 4.0))


def test_closest_reference_length():
    refs = [["one two three", "one two three four five six seven"]]
    assert statistics(["one two three four five"], refs)[1] == 3  # a tie, and the shorter wins
    assert statistics(["one two three four five six"], refs)[1] == 7
    assert statistics(["one"], refs)[1] == 3


def test_smoothing():
    preds, refs = ["the cat ran on a mat"], [["the cat sat on the mat"]]
    _, _, correct, total = statistics(preds, refs)
    assert (correct, total) == ([4, 1, 0, 0], [6, 5, 4, 3])
    p = score(preds, refs)["precisions"]
    assert p == near([400.0 / 6.0, 20.0, 100.0 / (2 * 4), 100.0 / (4 * 3)])
    p = score(preds, refs, smooth_method="floor")["precisions"]
    assert p == near([400.0 / 6.0, 20.0, 10.0 / 4, 10.0 / 3])
    p = score(preds, refs, smooth_method="add-k", smooth_value=0.5)["precisions"]
    assert p == near([400.0 / 6.0, 150.0 / 5.5, 50.0 / 4.5, 50.0 / 3.5])
    got = score(preds, refs, smooth_method="none")
    assert got["score"] == 0.0
    assert got["precisions"] == near([400.0 / 6.0, 20.0, 0.0, 0.0])


def test_no_match_and_empty():
    assert score(["a b c d"], [["e f g h"]], smooth_method="add-k")["score"] == 0.0
    got = score([""], [["the cat"]])
    assert (got["score"], got["bp"], got["sys_len"], got["ref_len"]) == (0.0, 0.0, 0, 2)


def test_effective_order():
    got = score(["the cat sat"], [["the cat sat on the mat"]], effective_order=True)
    assert got["score"] == near(100.0 * math.exp(1.0 - 2.0))
    assert score(["the cat sat"], [["the cat sat"]])["score"] == 0.0  # no 4-gram, no effective order
    assert score(["the cat sat"], [["the cat sat"]], effective_order=True)["score"] == near(100.0)


def test_max_order():
    got = score(["the cat is on the mat"], [["the cat sat on the mat"]], max_ngram_order=2)
    assert len(got["precisions"]) == 2
    assert got["score"] == near(100.0 * math.sqrt(5.0 / 6.0 * 3.0 / 5.0))


def test_tokenizers():
    t13a, intl = TOKENIZERS["13a"], TOKENIZERS["intl"]
    line = "He said &quot;no&quot; &amp; left -- at 3-4 pm, with 1,000.5 votes."
    assert t13a(line) == 'He said " no " & left -- at 3 - 4 pm , with 1,000.5 votes .'
    assert t13a("a<skipped>b well-\nknown") == "ab wellknown"
    assert intl("Der Preis stieg um 3,5 % \u2013 ein Rekord!") == "Der Preis stieg um 3,5 % \u2013 ein Rekord !"
    assert intl("It ended in 1999.") == "It ended in 1999."
    assert intl("Cost: $5, or €4.50!") == "Cost : $ 5 , or € 4.50!"
    assert intl("«Bonjour», dit-il.") == "« Bonjour » , dit - il ."
    assert TOKENIZERS["char"]("猫が座") == "猫 が 座"
    assert TOKENIZERS["none"]("a.b") == "a.b"


@pytest.mark.parametrize(
    ("preds", "refs", "opts"),
    [
        (["a"], [[]], {}),
        (["a", "b"], [["a"]], {}),
        (["a", "b"], [["a"], ["b", "c"]], {}),
        (["a"], [["a"]], {"tokenize": "spm"}),
        (["a"], [["a"]], {"smooth_method": "add-one"}),
        (["a"], [["a"]], {"max_ngram_order": 0}),
    ],
)
def test_bad_input(preds, refs, opts):
    with pytest.raises(ValueError):
        score(preds, refs, **opts)


words = st.lists(st.sampled_from(["a", "b", "c", "d", "e"]), min_size=1, max_size=12).map(" ".join)
corpus = st.lists(st.tuples(words, st.lists(words, min_size=2, max_size=2)), min_size=1, max_size=5)


@settings(max_examples=200, deadline=None)
@given(corpus, st.sampled_from(["none", "floor", "add-k", "exp"]), st.booleans())
def test_range_and_bp(segs, smooth, eff):
    preds, refs = [h for h, _ in segs], [r for _, r in segs]
    got = score(preds, refs, smooth_method=smooth, effective_order=eff)
    assert 0.0 <= got["score"] <= 100.0 + 1e-9
    assert all(0.0 <= p <= 100.0 + 1e-9 for p in got["precisions"])
    if got["sys_len"] >= got["ref_len"]:
        assert got["bp"] == 1.0
    else:
        assert 0.0 < got["bp"] < 1.0


@settings(max_examples=200, deadline=None)
@given(corpus, st.randoms(use_true_random=False))
def test_segment_order(segs, rnd):
    shuffled = list(segs)
    rnd.shuffle(shuffled)
    a = score([h for h, _ in segs], [r for _, r in segs])
    b = score([h for h, _ in shuffled], [r for _, r in shuffled])
    check(a, b)


@settings(max_examples=200, deadline=None)
@given(st.lists(st.lists(words, min_size=2, max_size=2), min_size=1, max_size=5))
def test_copy_of_a_reference(refs):
    preds = [r[0] for r in refs]
    assert score(preds, refs, effective_order=True)["score"] == near(100.0)
    if all(len(p.split()) >= 4 for p in preds):
        assert score(preds, refs)["score"] == near(100.0)


@settings(max_examples=200, deadline=None)
@given(corpus)
def test_statistics_add_over_segments(segs):
    total = statistics([h for h, _ in segs], [r for _, r in segs])
    parts = [statistics([h], [r]) for h, r in segs]
    assert total[0] == sum(p[0] for p in parts)
    assert total[1] == sum(p[1] for p in parts)
    assert total[2] == [sum(p[2][n] for p in parts) for n in range(4)]
    assert total[3] == [sum(p[3][n] for p in parts) for n in range(4)]
