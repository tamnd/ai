# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/byte-pair-encoding/ (ai://ZRQW82).
"""Byte-pair encoding in NumPy: symbols are integer ids, each word an int64 array, and pair statistics are computed
with np.unique and np.bincount over the whole table at once.

train_bpe matches Algorithm 1 of Sennrich et al. and the plain Python reference: pair counts are weighted by word
frequency with overlapping occurrences each counted, the most frequent pair is merged, and ties go to the pair whose
first occurrence comes earliest when the words are scanned in table order and each word left to right. A merge
replaces non-overlapping occurrences from left to right. encode applies the merges in learned order; encode_priority
merges the pair with the earliest merge, leftmost on ties; bpe_dropout_encode draws one uniform from a
numpy Generator for every adjacent position at every step and drops a candidate when its draw is below p."""

import numpy as np

END = "</w>"


class Symbols:
    """A two-way map between symbol strings and integer ids."""

    def __init__(self):
        self.names = []
        self.ids = {}

    def id(self, name):
        if name not in self.ids:
            self.ids[name] = len(self.names)
            self.names.append(name)
        return self.ids[name]

    def word(self, word):
        if not isinstance(word, str) or not word:
            raise ValueError("a word must be a nonempty string")
        return np.array([self.id(c) for c in word] + [self.id(END)], dtype=np.int64)


def pair_counts(words, freqs, size):
    """Unique pair keys left*size+right, their weighted counts, and the flat index of each key's first occurrence."""
    keys = [w[:-1] * size + w[1:] for w in words]
    weights = [np.full(w.size - 1, f, dtype=np.int64) for w, f in zip(words, freqs, strict=True)]
    keys = np.concatenate(keys) if keys else np.zeros(0, dtype=np.int64)
    weights = np.concatenate(weights) if weights else np.zeros(0, dtype=np.int64)
    uniq, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    counts = np.bincount(inverse.ravel(), weights=weights, minlength=uniq.size)
    return uniq, counts, first


def merge_ids(arr, a, b, new):
    """Replace the non-overlapping occurrences of (a, b), scanning left to right, by new."""
    hits = np.flatnonzero((arr[:-1] == a) & (arr[1:] == b))
    take = []
    for i in hits:
        if not take or i > take[-1] + 1:
            take.append(i)
    if not take:
        return arr
    take = np.array(take, dtype=np.int64)
    out = arr.copy()
    out[take] = new
    return np.delete(out, take + 1)


def train_bpe(word_freqs, num_merges):
    """Learn up to num_merges merges from an ordered {word: frequency} table.

    Returns (merges, vocab) in the same form as the plain Python reference: a list of string pairs and a
    {symbol tuple: frequency} dict of the final segmentation of the training words."""
    if num_merges < 0:
        raise ValueError("num_merges must be nonnegative")
    table = Symbols()
    words, freqs = [], []
    for word, freq in word_freqs.items():
        if freq <= 0:
            raise ValueError("word frequencies must be positive")
        words.append(table.word(word))
        freqs.append(freq)
    merges = []
    for _ in range(num_merges):
        size = len(table.names)
        uniq, counts, first = pair_counts(words, freqs, size)
        if uniq.size == 0:
            break
        top = np.flatnonzero(counts == counts.max())
        key = uniq[top[np.argmin(first[top])]]
        a, b = divmod(int(key), size)
        pair = (table.names[a], table.names[b])
        new = table.id(pair[0] + pair[1])
        merges.append(pair)
        words = [merge_ids(w, a, b, new) for w in words]
    vocab = {tuple(table.names[i] for i in w): f for w, f in zip(words, freqs, strict=True)}
    return merges, vocab


def _encode_table(word, merges):
    table = Symbols()
    arr = table.word(word)
    ids = [(table.id(a), table.id(b), table.id(a + b)) for a, b in merges]
    return table, arr, ids


def encode(word, merges):
    """Segment a word by applying the merges in the order they were learned."""
    table, arr, ids = _encode_table(word, merges)
    for a, b, new in ids:
        arr = merge_ids(arr, a, b, new)
    return [table.names[i] for i in arr]


def bpe_dropout_encode(word, merges, p, rng):
    """BPE-dropout with a numpy Generator (or an int seed). p = 0 is encode_priority, p = 1 gives characters."""
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    rng = np.random.default_rng(rng)
    table, arr, ids = _encode_table(word, merges)
    size = len(table.names)
    rank = {}
    for r, (a, b, new) in enumerate(ids):
        rank.setdefault(a * size + b, (r, new))
    while arr.size > 1:
        keys = arr[:-1] * size + arr[1:]
        ranks = np.array([rank.get(int(k), (np.inf, -1))[0] for k in keys], dtype=np.float64)
        ranks[rng.random(keys.size) < p] = np.inf
        i = int(np.argmin(ranks))
        if not np.isfinite(ranks[i]):
            break
        arr = np.concatenate([arr[:i], [rank[int(keys[i])][1]], arr[i + 2 :]])
    return [table.names[i] for i in arr]


def encode_priority(word, merges):
    """Merge the adjacent pair with the earliest merge, leftmost on ties, until no pair is in the table."""
    return bpe_dropout_encode(word, merges, 0.0, 0)
