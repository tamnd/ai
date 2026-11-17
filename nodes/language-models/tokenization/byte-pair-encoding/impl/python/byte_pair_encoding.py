# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/byte-pair-encoding/ (ai://NEW-byte-pair-encoding).
"""Byte-pair encoding in plain Python: training from a word-frequency table, encoding, and BPE-dropout.

A word is split into its characters plus the end-of-word symbol END. train_bpe follows Algorithm 1 of Sennrich et al.:
count every adjacent pair over all words, weighted by word frequency (overlapping occurrences each count), merge the
most frequent pair, repeat. Ties go to the pair seen first when the words are scanned in the order of the table and
each word left to right; for an ordered table this is what max() over Algorithm 1's dict returns. A merge replaces the
non-overlapping occurrences of the pair from left to right, as Algorithm 1's regular expression does.

encode applies the merges in the order they were learned. encode_priority repeatedly merges the adjacent pair with the
earliest merge (ties to the leftmost occurrence), the procedure of Provilkov et al.; bpe_dropout_encode is the same loop
in which each candidate occurrence is dropped with probability p at every step, using a seeded random.Random."""

import random

END = "</w>"


def initial_symbols(word):
    """The characters of a nonempty word followed by END."""
    if not isinstance(word, str) or not word:
        raise ValueError("a word must be a nonempty string")
    return tuple(word) + (END,)


def pair_counts(vocab):
    """Frequency-weighted counts of adjacent pairs, in first-seen order. vocab maps symbol tuples to counts."""
    counts = {}
    for symbols, freq in vocab.items():
        for pair in zip(symbols, symbols[1:]):
            counts[pair] = counts.get(pair, 0) + freq
    return counts


def merge_symbols(symbols, pair):
    """Replace the non-overlapping occurrences of pair, scanning left to right, by the concatenated symbol."""
    a, b = pair
    out = []
    i = 0
    while i < len(symbols):
        if i + 1 < len(symbols) and symbols[i] == a and symbols[i + 1] == b:
            out.append(a + b)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return tuple(out)


def train_bpe(word_freqs, num_merges):
    """Learn up to num_merges merges from an ordered {word: frequency} table.

    Returns (merges, vocab): the list of merged pairs in order and the final {symbol tuple: frequency} segmentation of
    the training words. Training stops early when every word is a single symbol."""
    if num_merges < 0:
        raise ValueError("num_merges must be nonnegative")
    vocab = {}
    for word, freq in word_freqs.items():
        if freq <= 0:
            raise ValueError("word frequencies must be positive")
        vocab[initial_symbols(word)] = freq
    merges = []
    for _ in range(num_merges):
        counts = pair_counts(vocab)
        if not counts:
            break
        best = max(counts, key=counts.get)
        merges.append(best)
        vocab = {merge_symbols(symbols, best): freq for symbols, freq in vocab.items()}
    return merges, vocab


def encode(word, merges):
    """Segment a word by applying the merges in the order they were learned."""
    symbols = initial_symbols(word)
    for pair in merges:
        symbols = merge_symbols(symbols, pair)
    return list(symbols)


def _ranks(merges):
    ranks = {}
    for rank, pair in enumerate(merges):
        ranks.setdefault(pair, rank)
    return ranks


def _segment(word, merges, keep):
    ranks = _ranks(merges)
    symbols = list(initial_symbols(word))
    while True:
        best = None
        for i in range(len(symbols) - 1):
            rank = ranks.get((symbols[i], symbols[i + 1]))
            if rank is None or not keep():
                continue
            if best is None or rank < best[0]:
                best = (rank, i)
        if best is None:
            return symbols
        i = best[1]
        symbols[i : i + 2] = [symbols[i] + symbols[i + 1]]


def encode_priority(word, merges):
    """Merge the adjacent pair with the earliest merge, leftmost on ties, until no pair is in the table."""
    return _segment(word, merges, lambda: True)


def bpe_dropout_encode(word, merges, p, rng):
    """BPE-dropout: as encode_priority, but at each step every candidate occurrence is dropped with probability p.

    rng is a random.Random (or an int seed); one rng.random() is drawn per candidate occurrence per step, in left to
    right order, and the occurrence is kept when the draw is at least p. p = 0 is encode_priority, p = 1 characters."""
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    if not isinstance(rng, random.Random):
        rng = random.Random(rng)
    return _segment(word, merges, lambda: rng.random() >= p)
