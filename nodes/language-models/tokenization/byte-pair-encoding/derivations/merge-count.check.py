# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/byte-pair-encoding/ (ai://ZRQW82).
"""Numeric check of merge-count: each merge lowers the token count by the number of non-overlapping occurrences
it replaces, between 1 and the pair count, and k merges add k symbols to the initial ones."""

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


def stats(vocab):
    counts = {}
    for symbols, f in vocab.items():
        for pair in zip(symbols, symbols[1:]):
            counts[pair] = counts.get(pair, 0) + f
    return counts


def tokens(vocab):
    return sum(len(s) * f for s, f in vocab.items())


def run(table, k):
    vocab = {tuple(w) + (END,): f for w, f in table.items()}
    initial = {s for symbols in vocab for s in symbols}
    start, steps, created = tokens(vocab), [], []
    for _ in range(k):
        counts = stats(vocab)
        if not counts:
            break
        pair = max(counts, key=counts.get)
        new = {merge(s, pair): f for s, f in vocab.items()}
        drop = tokens(vocab) - tokens(new)
        replaced = sum(f * (len(s) - len(merge(s, pair))) for s, f in vocab.items())
        assert drop == replaced and 1 <= drop <= counts[pair], (pair, drop, counts[pair])
        steps.append((pair, counts[pair], drop))
        created.append(pair[0] + pair[1])
        vocab = new
    assert tokens(vocab) == start - sum(d for _, _, d in steps)
    assert len(set(created)) == len(created) and not set(created) & initial
    alive = {s for symbols in vocab for s in symbols}
    assert alive <= initial | set(created)
    return initial, start, steps, vocab, alive


initial, start, steps, vocab, alive = run({"low": 5, "lower": 2, "newest": 6, "widest": 3}, 10)
assert len(initial) == 11 and start == 95
assert [d for _, _, d in steps] == [9, 9, 9, 7, 7, 6, 6, 6, 5, 3]
assert tokens(vocab) == 28 and len(initial) + len(steps) == 21 and len(alive) == 9 and "es" not in alive

_, _, steps, _, _ = run({"aaa": 4}, 1)
assert steps == [(("a", "a"), 8, 4)]

rng = random.Random(7)
worst = 0
for _ in range(2000):
    table = {}
    while len(table) < 5:
        table.setdefault("".join(rng.choice("ab") for _ in range(rng.randint(1, 7))), rng.randint(1, 4))
    _, _, steps, _, _ = run(table, 15)
    worst = max([worst] + [count - drop for _, count, drop in steps])
assert worst > 0
print("merge-count: ok")
