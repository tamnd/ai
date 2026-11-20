# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/n-gram-language-model/ (ai://NEW-n-gram-language-model).
"""Exact check of kn-continuation: for an interpolated bigram with absolute discount 0 < D <= 1, the lower-order
distribution q that makes the model keep the observed marginal of the predicted word,
sum_v c(v .) p(w | v) = c(. w), is q(w) = N1+(. w) / N1+(. .), the share of distinct bigram types that end in w;
the unigram relative frequency c(. w) / N does not keep it, and for D > 1 continuation counts do not either."""

import random
from fractions import Fraction


def bigrams(sents):
    c = {}
    for s in sents:
        toks = ["<s>"] + s + ["</s>"]
        for v, w in zip(toks, toks[1:], strict=False):
            c[(v, w)] = c.get((v, w), 0) + 1
    return c


def model(c, D, q):
    ctx, types = {}, {}
    for (v, _), n in c.items():
        ctx[v] = ctx.get(v, 0) + n
        types[v] = types.get(v, 0) + 1

    def p(w, v):
        return Fraction(max(c.get((v, w), 0) - D, 0), ctx[v]) + D * types[v] / Fraction(ctx[v]) * q[w]

    return p, ctx


def marginal_gap(c, D, q):
    p, ctx = model(c, D, q)
    words = {w for _, w in c}
    target = {w: sum(n for (_, x), n in c.items() if x == w) for w in words}
    return max(abs(sum(ctx[v] * p(w, v) for v in ctx) - target[w]) for w in words)


def continuation(c):
    words = {w for _, w in c}
    return {w: Fraction(sum(1 for (_, x) in c if x == w), len(c)) for w in words}


def relative(c):
    N = sum(c.values())
    words = {w for _, w in c}
    return {w: Fraction(sum(n for (_, x), n in c.items() if x == w), N) for w in words}


def solve(c, D):
    """q from the marginal constraint without assuming D <= 1."""
    p_types = len(c)
    words = {w for _, w in c}
    return {
        w: (sum(n for (_, x), n in c.items() if x == w) - sum(max(n - D, 0) for (_, x), n in c.items() if x == w))
        / (D * p_types)
        for w in words
    }


def main():
    rng = random.Random(1)
    for _ in range(200):
        words = [f"x{i}" for i in range(rng.randint(2, 9))]
        sents = [[rng.choice(words) for _ in range(rng.randint(0, 8))] for _ in range(rng.randint(1, 30))]
        c = bigrams(sents)
        D = Fraction(rng.randint(1, 20), 20)  # 0 < D <= 1
        q = continuation(c)
        assert sum(q.values()) == 1
        assert marginal_gap(c, D, q) == 0
        assert solve(c, D) == q  # the solution is unique and equals the continuation share
        D2 = Fraction(rng.randint(21, 40), 20)  # 1 < D <= 2
        if any(n == 1 for n in c.values()):
            assert solve(c, D2) != continuation(c)
    # San Francisco: frequent, but always after San. The relative frequency breaks the marginal; continuation keeps it.
    sents = [["san", "francisco"]] * 30 + [["on", w] for w in ("tuesday", "monday", "friday", "sunday")] * 3
    sents += [[w, "tuesday"] for w in ("by", "next", "last", "each", "every")]
    c = bigrams(sents)
    q, r = continuation(c), relative(c)
    D = Fraction(3, 4)
    assert marginal_gap(c, D, q) == 0 and marginal_gap(c, D, r) > 0
    assert r["francisco"] > r["tuesday"] and q["francisco"] < q["tuesday"]
    p_q, _ = model(c, D, q)
    p_r, _ = model(c, D, r)
    print(
        f"kn-continuation: 200 random corpora keep the marginal exactly; p(francisco | on) = "
        f"{float(p_q('francisco', 'on')):.4f} with continuation counts, {float(p_r('francisco', 'on')):.4f} with "
        f"relative frequencies; p(tuesday | on) = {float(p_q('tuesday', 'on')):.4f}, {float(p_r('tuesday', 'on')):.4f}"
    )


if __name__ == "__main__":
    main()
