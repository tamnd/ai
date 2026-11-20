# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/n-gram-language-model/ (ai://NEW-n-gram-language-model).
"""Exact check of kn-normalisation: with adjusted counts, discounts 0 <= D_n(k) <= k, the pseudo probabilities u, the
backoffs b and the recursion p(w | h) = u(w | h) + b(h) p(w | h') ending in the uniform distribution, p(. | h) sums to
exactly one for every context h, seen or not, and is positive when every D_n(k) with k >= 1 is."""

import random
from fractions import Fraction

BOS, EOS, UNK = "<s>", "</s>", "<unk>"


def build(sents, order, disc):
    top = {}
    for s in sents:
        toks = [BOS] * (order - 1) + s + [EOS]
        for i in range(order - 1, len(toks)):
            g = tuple(toks[i - order + 1 : i + 1])
            top[g] = top.get(g, 0) + 1
    adj, raw = {order: dict(top)}, dict(top)
    for n in range(order - 1, 0, -1):
        left, r = {}, {}
        for g in adj[n + 1]:
            left[g[1:]] = left.get(g[1:], 0) + 1
            r[g[1:]] = r.get(g[1:], 0) + raw[g]
        raw = r
        adj[n] = {g: (r[g] if g[0] == BOS else left[g]) for g in left}
    vocab = sorted({g[0] for g in adj[1]} | {UNK})
    total, gamma = {}, {}
    for n, e in adj.items():
        for g, a in e.items():
            total[g[:-1]] = total.get(g[:-1], 0) + a
            gamma[g[:-1]] = gamma.get(g[:-1], 0) + disc[n][min(a, 3)]
    u = {g: (a - disc[n][min(a, 3)]) / total[g[:-1]] for n, e in adj.items() for g, a in e.items()}
    return adj, vocab, total, gamma, u


def prob(model, w, ctx):
    _, vocab, total, gamma, u = model
    p = Fraction(1, len(vocab))
    for i in range(len(ctx), -1, -1):
        h = ctx[i:]
        if h in total:
            p = u.get(h + (w,), Fraction(0)) + gamma[h] / total[h] * p
    return p


def main():
    rng = random.Random(0)
    contexts_checked = 0
    for trial in range(60):
        order = rng.randint(1, 4)
        words = [f"x{i}" for i in range(rng.randint(2, 8))]
        sents = [[rng.choice(words) for _ in range(rng.randint(0, 7))] for _ in range(rng.randint(1, 40))]
        disc = {n: [Fraction(0)] + [Fraction(rng.randint(0, 12 * k), 12) for k in (1, 2, 3)] for n in range(1, 5)}
        if trial % 3 == 0:  # discounts strictly inside (0, k]
            disc = {n: [Fraction(0)] + [Fraction(rng.randint(1, 12 * k), 12) for k in (1, 2, 3)] for n in range(1, 5)}
        model = build(sents, order, disc)
        adj, vocab = model[0], model[1]
        ctxs = {g[:-1] for e in adj.values() for g in e}
        ctxs |= {tuple(rng.choice(words + [BOS, "unseen"]) for _ in range(order - 1)) for _ in range(20)}
        ctxs = {c[len(c) - order + 1 :] if order > 1 else () for c in ctxs}
        positive = all(d > 0 for n in disc for d in disc[n][1:])
        for c in ctxs:
            ps = [prob(model, w, c) for w in vocab]
            assert sum(ps) == 1, (trial, c, sum(ps))
            assert min(ps) >= 0
            if positive:
                assert min(ps) > 0
            contexts_checked += 1
            # Each level: the u terms of a seen context sum to 1 - b(h).
            _, _, total, gamma, u = model
            if c in total:
                assert sum(u.get(c + (w,), 0) for w in vocab) == 1 - gamma[c] / total[c]
    # A discount above its count breaks it: D_1(1) = 1.5 gives a negative u.
    m = build([["a", "b"], ["a"]], 2, {n: [Fraction(0), Fraction(3, 2), Fraction(1), Fraction(1)] for n in (1, 2)})
    assert min(prob(m, w, ("a",)) for w in m[1]) < 0
    print(f"kn-normalisation: {contexts_checked} contexts in 60 random models sum to exactly 1")


if __name__ == "__main__":
    main()
