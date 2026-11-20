# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/byte-pair-encoding/ (ai://ZRQW82).
"""Numeric check of encode-reproduces: replaying the learned merges on a training word gives its training
segmentation, priority encoding equals sequential encoding when every merge result is new, and the two
counterexamples where the conditions fail."""

import itertools
import random

END = "</w>"


def merge(symbols, pair):
    out, i = [], 0
    while i < len(symbols):
        if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
            out.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return tuple(out)


def train(table, k):
    vocab = {tuple(w) + (END,): f for w, f in table.items()}
    merges = []
    for _ in range(k):
        counts = {}
        for symbols, f in vocab.items():
            for pair in itertools.pairwise(symbols):
                counts[pair] = counts.get(pair, 0) + f
        if not counts:
            break
        pair = max(counts, key=counts.get)
        merges.append(pair)
        vocab = {merge(s, pair): f for s, f in vocab.items()}
    return merges, vocab


def sequential(word, merges):
    symbols = tuple(word) + (END,)
    for pair in merges:
        symbols = merge(symbols, pair)
    return symbols


def priority(word, merges):
    rank = {}
    for i, pair in enumerate(merges):
        rank.setdefault(pair, i)
    symbols = list(word) + [END]
    while True:
        best = None
        for i in range(len(symbols) - 1):
            r = rank.get((symbols[i], symbols[i + 1]))
            if r is not None and (best is None or r < best[0]):
                best = (r, i)
        if best is None:
            return tuple(symbols)
        i = best[1]
        symbols[i : i + 2] = [symbols[i] + symbols[i + 1]]


rng = random.Random(11)
checked = 0
for _ in range(3000):
    table = {}
    while len(table) < 6:
        table.setdefault("".join(rng.choice("abcd") for _ in range(rng.randint(1, 8))), rng.randint(1, 5))
    items = list(table.items())
    rng.shuffle(items)
    table = dict(items)
    merges, vocab = train(table, rng.randint(0, 20))
    for word, f in table.items():
        assert vocab.get(sequential(word, merges), 0) >= f
        assert priority(word, merges) == sequential(word, merges)
    created = [a + b for a, b in merges]
    if len(set(created)) == len(created):
        for _ in range(5):
            word = "".join(rng.choice("abcd") for _ in range(rng.randint(1, 10)))
            assert priority(word, merges) == sequential(word, merges), (merges, word)
            checked += 1
assert checked == 15000

table = [("b", "c"), ("a", "b"), ("ab", "c"), ("abc", "d"), ("a", "bc")]
assert sequential("abcd", table) == ("abc", "d", END) and priority("abcd", table) == ("abcd", END)
order = [("ab", "c"), ("a", "b")]
assert sequential("abc", order) == ("ab", "c", END) and priority("abc", order) == ("abc", END)
print("encode-reproduces: ok")
