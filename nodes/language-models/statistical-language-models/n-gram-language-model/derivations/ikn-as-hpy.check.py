# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/n-gram-language-model/ (ai://NEW-n-gram-language-model).
"""Exact check of ikn-as-hpy: the predictive probability of Teh's hierarchical Pitman-Yor language model (Eq. 11-12)
with strengths theta = 0, one discount per order, and the seating t_uw = min(1, c_uw) of Eq. 15-16 equals interpolated
Kneser-Ney with continuation counts at every lower order, built separately from the n-gram counts. KenLM's rule of
keeping raw counts for n-grams that start with <s> is not part of this correspondence, and changes the numbers."""

import random
from fractions import Fraction

BOS, EOS = "<s>", "</s>"


def ngram_counts(sents, N):
    c = {}
    for s in sents:
        toks = [BOS] * (N - 1) + s + [EOS]
        for i in range(N - 1, len(toks)):
            g = tuple(toks[i - N + 1 : i + 1])
            c[g] = c.get(g, 0) + 1
    return c


def hpy(sents, N, d, V):
    """Restaurant counts c[u][w]: customers from the data at depth N - 1, tables of the children below (Eq. 15-16)."""
    c = {}
    for g, n in ngram_counts(sents, N).items():
        c.setdefault(g[:-1], {})[g[-1]] = n
    for depth in range(N - 1, 0, -1):
        for u in [u for u in c if len(u) == depth]:
            for w, n in c[u].items():
                parent = c.setdefault(u[1:], {})
                parent[w] = parent.get(w, 0) + min(1, n)

    def prob(w, u):
        if len(u) == 0 and () not in c:
            return Fraction(1, V)
        lower = Fraction(1, V) if len(u) == 0 else prob(w, u[1:])
        if u not in c:
            return lower
        cu, tu = sum(c[u].values()), sum(1 for n in c[u].values() if n > 0)
        cuw = c[u].get(w, 0)
        D = d[len(u)]
        return (cuw - D * min(1, cuw)) / Fraction(cu) + D * tu / Fraction(cu) * lower

    return prob


def ikn(sents, N, d, V, bos_rule):
    """Interpolated Kneser-Ney from the counts: raw counts at order N, continuation counts below (optionally the
    raw count for an n-gram that starts with <s>, as in KenLM)."""
    top = ngram_counts(sents, N)
    a, raw = {N: dict(top)}, dict(top)
    for n in range(N - 1, 0, -1):
        cont, r = {}, {}
        for g in a[n + 1]:
            cont[g[1:]] = cont.get(g[1:], 0) + 1
            r[g[1:]] = r.get(g[1:], 0) + raw[g]
        raw = r
        a[n] = {g: (r[g] if bos_rule and g[0] == BOS else cont[g]) for g in cont}
    table = {}
    for e in a.values():
        for g, x in e.items():
            table.setdefault(g[:-1], {})[g[-1]] = x

    def prob(w, h):
        p = Fraction(1, V)
        for i in range(len(h), -1, -1):
            u = h[i:]
            if u in table:
                row, D = table[u], d[len(u)]
                tot = sum(row.values())
                p = max(row.get(w, 0) - D, 0) / Fraction(tot) + D * len(row) / Fraction(tot) * p
        return p

    return prob


def main():
    rng = random.Random(5)
    checked, differ = 0, 0
    for _ in range(40):
        N = rng.randint(1, 4)
        words = [f"x{i}" for i in range(rng.randint(2, 6))]
        sents = [[rng.choice(words) for _ in range(rng.randint(0, 6))] for _ in range(rng.randint(1, 25))]
        d = {n: Fraction(rng.randint(1, 10), 10) for n in range(N)}  # discounts in (0, 1], one per context length
        V = len(words) + 2  # the words, EOS and one unknown word
        teh, kn, kenlm = hpy(sents, N, d, V), ikn(sents, N, d, V, False), ikn(sents, N, d, V, True)
        for _ in range(30):
            h = tuple(rng.choice(words + [BOS]) for _ in range(N - 1))
            for w in words + [EOS, "unk"]:
                assert teh(w, h) == kn(w, h)
                differ += teh(w, h) != kenlm(w, h)
                checked += 1
    assert differ > 0
    print(f"ikn-as-hpy: {checked} probabilities agree exactly; KenLM's <s> rule changes {differ} of them")


if __name__ == "__main__":
    main()
