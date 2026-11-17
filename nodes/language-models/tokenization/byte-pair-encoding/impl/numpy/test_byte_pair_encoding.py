# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/byte-pair-encoding/ (ai://NEW-byte-pair-encoding).
"""Tests of the NumPy version: the low/lower/newest/widest example of Sennrich et al., agreement with an
independent regex version of their Algorithm 1 on random tables, the token-count and vocabulary identities of the
derivation merge-count, encoding of the training words, priority encoding, and BPE-dropout at p = 0, p = 1 and in
between."""

import itertools
import random
import re

import pytest

import numpy as np

from byte_pair_encoding import (
    END,
    Symbols,
    bpe_dropout_encode,
    encode,
    encode_priority,
    merge_ids,
    pair_counts,
    train_bpe,
)


def initial_symbols(word):
    return tuple(word) + (END,)


def merge_symbols(symbols, pair):
    table = Symbols()
    arr = np.array([table.id(s) for s in symbols], dtype=np.int64)
    out = merge_ids(arr, table.id(pair[0]), table.id(pair[1]), table.id(pair[0] + pair[1]))
    return tuple(table.names[i] for i in out)

TABLE = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
MERGES = [
    ("e", "s"),
    ("es", "t"),
    ("est", END),
    ("l", "o"),
    ("lo", "w"),
    ("n", "e"),
    ("ne", "w"),
    ("new", "est" + END),
    ("low", END),
    ("w", "i"),
]


def regex_bpe(table, num_merges):
    """Algorithm 1 written independently: space-separated strings and a regular expression for the merge."""
    corpus = {" ".join(word) + " " + END: f for word, f in table.items()}
    learned = []
    for _ in range(num_merges):
        stats = {}
        for text, f in corpus.items():
            parts = text.split()
            for left, right in zip(parts, parts[1:]):
                stats[left, right] = stats.get((left, right), 0) + f
        if not stats:
            break
        top = max(stats, key=stats.get)
        learned.append(top)
        pattern = re.compile(r"(?<!\S)" + re.escape(" ".join(top)) + r"(?!\S)")
        corpus = {pattern.sub("".join(top), text): f for text, f in corpus.items()}
    return learned, {tuple(text.split()): f for text, f in corpus.items()}


def random_table(rng, alphabet="abc", max_len=6, size=6):
    table = {}
    while len(table) < size:
        word = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, max_len)))
        table.setdefault(word, rng.randint(1, 5))
    return table


def tokens(vocab):
    return sum(len(symbols) * f for symbols, f in vocab.items())


def counts_of(vocab):
    table = Symbols()
    words = [np.array([table.id(s) for s in symbols], dtype=np.int64) for symbols in vocab]
    size = len(table.names)
    uniq, counts, first = pair_counts(words, list(vocab.values()), size)
    order = np.argsort(first, kind="stable")
    return {(table.names[uniq[i] // size], table.names[uniq[i] % size]): int(counts[i]) for i in order}


def test_symbols_word():
    table = Symbols()
    assert table.word("low").tolist() == [0, 1, 2, 3]
    assert table.word("ow").tolist() == [1, 2, 3]
    assert table.names == ["l", "o", "w", END] and table.id("w") == 2
    for bad in ["", None, 3]:
        with pytest.raises(ValueError):
            table.word(bad)


def test_pair_counts_empty():
    uniq, counts, first = pair_counts([np.array([0])], [4], 1)
    assert uniq.size == counts.size == first.size == 0
    assert pair_counts([], [], 1)[0].size == 0


def test_pair_counts_of_the_example():
    vocab = {initial_symbols(w): f for w, f in TABLE.items()}
    counts = counts_of(vocab)
    assert counts[("e", "s")] == counts[("s", "t")] == counts[("t", END)] == 9
    assert counts[("w", "e")] == 8
    assert counts[("l", "o")] == counts[("o", "w")] == 7
    assert counts[("n", "e")] == counts[("e", "w")] == 6
    assert counts[("w", END)] == 5
    assert counts[("w", "i")] == counts[("i", "d")] == counts[("d", "e")] == 3
    assert counts[("e", "r")] == counts[("r", END)] == 2
    assert list(counts)[:3] == [("l", "o"), ("o", "w"), ("w", END)]
    assert sum(counts.values()) == tokens(vocab) - sum(vocab.values())


def test_overlapping_pairs_count_twice_but_merge_once():
    vocab = {initial_symbols("aaa"): 2}
    assert counts_of(vocab)[("a", "a")] == 4
    assert merge_symbols(("a", "a", "a", END), ("a", "a")) == ("aa", "a", END)
    assert merge_symbols(("a", "a", "a", "a"), ("a", "a")) == ("aa", "aa")
    assert merge_symbols(("a", "b"), ("b", "a")) == ("a", "b")
    assert merge_symbols(("x", "a"), ("x", "a")) == ("xa",)
    assert merge_symbols((), ("a", "b")) == ()


def test_sennrich_example_merges_and_segmentation():
    merges, vocab = train_bpe(TABLE, 10)
    assert merges == MERGES
    assert vocab == {
        ("low" + END,): 5,
        ("low", "e", "r", END): 2,
        ("newest" + END,): 6,
        ("wi", "d", "est" + END): 3,
    }
    assert tokens(vocab) == 5 + 2 * 4 + 6 + 3 * 3


def test_figure_1_segments_lower():
    figure = [("r", END), ("l", "o"), ("lo", "w"), ("e", "r" + END)]
    assert encode("lower", figure) == ["low", "er" + END]
    assert encode_priority("lower", figure) == ["low", "er" + END]


def test_ties_follow_table_order():
    assert train_bpe({"ab": 1, "cd": 1}, 1)[0] == [("a", "b")]
    assert train_bpe({"cd": 1, "ab": 1}, 1)[0] == [("c", "d")]
    assert train_bpe({"ab": 1, "cd": 2}, 1)[0] == [("c", "d")]


def test_matches_regex_algorithm_on_random_tables():
    rng = random.Random(0)
    for _ in range(150):
        table = random_table(rng)
        k = rng.randint(0, 12)
        assert train_bpe(table, k) == regex_bpe(table, k)


def test_stops_when_every_word_is_one_symbol():
    merges, vocab = train_bpe({"ab": 3}, 10)
    assert merges == [("a", "b"), ("ab", END)]
    assert vocab == {("ab" + END,): 3}
    assert train_bpe({"a": 1}, 0) == ([], {("a", END): 1})


def test_bad_arguments():
    with pytest.raises(ValueError):
        train_bpe(TABLE, -1)
    for bad in [0, -2]:
        with pytest.raises(ValueError):
            train_bpe({"ab": bad}, 1)
    with pytest.raises(ValueError):
        train_bpe({"": 1}, 1)
    with pytest.raises(ValueError):
        encode("", MERGES)
    for p in [-0.1, 1.1]:
        with pytest.raises(ValueError):
            bpe_dropout_encode("low", MERGES, p, 0)


def test_each_merge_removes_its_non_overlapping_occurrences():
    rng = random.Random(1)
    for _ in range(100):
        table = random_table(rng)
        vocab = {initial_symbols(w): f for w, f in table.items()}
        before = tokens(vocab)
        merges, _ = train_bpe(table, 12)
        for pair in merges:
            removed = sum(f * (len(s) - len(merge_symbols(s, pair))) for s, f in vocab.items())
            assert removed >= 1
            assert removed <= counts_of(vocab)[pair]
            vocab = {merge_symbols(s, pair): f for s, f in vocab.items()}
            assert tokens(vocab) == before - removed
            before -= removed


def test_vocabulary_is_initial_symbols_plus_merges():
    merges, vocab = train_bpe(TABLE, 10)
    initial = {s for w in TABLE for s in initial_symbols(w)}
    created = {a + b for a, b in merges}
    assert len(initial) == 11 and len(created) == 10 and not initial & created
    assert len(initial | created) == len(initial) + len(merges)
    alive = {s for symbols in vocab for s in symbols}
    assert alive <= initial | created
    assert "es" not in alive and "new" not in alive


def test_encode_reproduces_training_segmentation():
    rng = random.Random(2)
    for _ in range(100):
        table = random_table(rng)
        merges, vocab = train_bpe(table, rng.randint(0, 12))
        for word in table:
            segmented = tuple(encode(word, merges))
            assert segmented in vocab
            assert encode_priority(word, merges) == list(segmented)
            assert "".join(segmented) == word + END


def test_encode_unseen_word():
    assert encode("lowest", MERGES) == ["low", "est" + END]
    assert encode("wider", MERGES) == ["wi", "d", "e", "r", END]
    assert encode("xyz", MERGES) == ["x", "y", "z", END]
    assert encode("low", []) == ["l", "o", "w", END]


def test_priority_differs_when_the_conditions_fail():
    table = [("b", "c"), ("a", "b"), ("ab", "c"), ("abc", "d"), ("a", "bc")]
    assert encode("abcd", table) == ["abc", "d", END]
    assert encode_priority("abcd", table) == ["abcd", END]
    order = [("ab", "c"), ("a", "b")]
    assert encode("abc", order) == ["ab", "c", END]
    assert encode_priority("abc", order) == ["abc", END]


def test_priority_prefers_earlier_merge_then_leftmost():
    assert encode_priority("abc", [("b", "c"), ("a", "b")]) == ["a", "bc", END]
    assert encode_priority("abc", [("a", "b"), ("b", "c")]) == ["ab", "c", END]
    assert encode_priority("aaa", [("a", "a"), ("a", "a")]) == ["aa", "a", END]
    assert encode_priority("abab", [("a", "b"), ("ab", "ab")]) == ["abab", END]


def test_dropout_zero_is_bpe_and_one_is_characters():
    for word in ["lowest", "newer", "widest", "low"]:
        for seed in range(5):
            assert bpe_dropout_encode(word, MERGES, 0.0, seed) == encode_priority(word, MERGES)
            assert bpe_dropout_encode(word, MERGES, 1.0, seed) == list(initial_symbols(word))


def test_dropout_is_seeded_and_lossless():
    a = [bpe_dropout_encode("newest", MERGES, 0.3, np.random.default_rng(s)) for s in range(50)]
    b = [bpe_dropout_encode("newest", MERGES, 0.3, s) for s in range(50)]
    assert a == b
    assert all("".join(t) == "newest" + END for t in a)
    assert len({tuple(t) for t in a}) > 3
    shared = np.random.default_rng(9)
    first = [bpe_dropout_encode("newest", MERGES, 0.3, shared) for _ in range(20)]
    assert len({tuple(t) for t in first}) > 1


def test_dropout_rate_of_a_single_merge():
    rng = np.random.default_rng(3)
    for p in [0.1, 0.5, 0.8]:
        n = 4000
        merged = sum(bpe_dropout_encode("ab", [("a", "b")], p, rng) == ["ab", END] for _ in range(n))
        assert abs(merged / n - (1 - p)) < 0.03


def test_dropout_finer_on_average_as_p_grows():
    rng = np.random.default_rng(4)
    mean = []
    for p in [0.0, 0.1, 0.5, 0.9]:
        mean.append(sum(len(bpe_dropout_encode("newest", MERGES, p, rng)) for _ in range(400)) / 400)
    assert mean[0] == 1 and all(x < y for x, y in itertools.pairwise(mean))
