# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/n-gram-language-model/ (ai://NEW-n-gram-language-model).
"""Exact check of infinigram-normalisation: choosing the suffix s before looking at the next word and dividing by the
number of occurrences of s that are followed by a token gives a distribution; dividing by cnt(s) instead loses the
occurrences of s at the end of a document; backing off separately for each word (the naive backoff) and Stupid
Backoff give scores that need not sum to one."""

import random
from fractions import Fraction


def occurrences(docs, s):
    """(followed, at_end): the next tokens of the occurrences of s, and how many occurrences end a document."""
    nxt, end = [], 0
    m = len(s)
    if m == 0:  # the empty string: every token follows it once, and cnt is the number of tokens
        return [w for d in docs for w in d], 0
    for d in docs:
        for i in range(len(d) - m + 1):
            if tuple(d[i : i + m]) == s:
                if i + m < len(d):
                    nxt.append(d[i + m])
                else:
                    end += 1
    return nxt, end


def infinigram(docs, ctx):
    for start in range(len(ctx) + 1):
        nxt, _ = occurrences(docs, tuple(ctx[start:]))
        if nxt:
            return {w: Fraction(nxt.count(w), len(nxt)) for w in set(nxt)}, tuple(ctx[start:])
    raise ValueError("empty data")


def liu_denominator(docs, ctx):
    """Liu et al.'s rule as written: the longest suffix with cnt(s) > 0, divided by cnt(s)."""
    for start in range(len(ctx) + 1):
        s = tuple(ctx[start:])
        nxt, end = occurrences(docs, s)
        if nxt or end:
            return {w: Fraction(nxt.count(w), len(nxt) + end) for w in set(nxt)}
    raise ValueError("empty data")


def naive_backoff(docs, ctx, w):
    for start in range(len(ctx) + 1):
        nxt, _ = occurrences(docs, tuple(ctx[start:]))
        if nxt.count(w) > 0:
            return Fraction(nxt.count(w), len(nxt))
    return Fraction(0)


def main():
    rng = random.Random(11)
    over = under = short = 0
    for _ in range(300):
        letters = "abcd"[: rng.randint(2, 4)]
        docs = [[rng.choice(letters) for _ in range(rng.randint(1, 12))] for _ in range(rng.randint(1, 6))]
        vocab = sorted({w for d in docs for w in d})
        ctx = [rng.choice(letters + "e") for _ in range(rng.randint(0, 8))]
        dist, s = infinigram(docs, ctx)
        assert sum(dist.values()) == 1
        # the chosen suffix is the longest one followed by a token, and it does not depend on the next word
        assert all(not occurrences(docs, tuple(ctx[i:]))[0] for i in range(len(ctx) - len(s)))
        liu = liu_denominator(docs, ctx)
        short += sum(liu.values()) < 1
        tot = sum(naive_backoff(docs, ctx, w) for w in vocab)
        over += tot > 1
        under += tot < 1
    assert short > 0 and over > 0
    # Stupid Backoff on "a b", "a c", "b b" with an end marker: after "a", b and c already take 1/2 each, and the
    # words reached by backing off add alpha f(w) / N on top.
    seen = {"b": Fraction(1, 2), "c": Fraction(1, 2)}
    unigram = {"a": 2, "</s>": 3}
    N = 9
    sb = {**seen, **{w: Fraction(2, 5) * Fraction(f, N) for w, f in unigram.items()}}
    assert sum(sb.values()) == Fraction(11, 9)
    print(
        f"infinigram-normalisation: 300 random cases sum to exactly 1; dividing by cnt(s) loses mass in {short}, "
        f"naive backoff sums above 1 in {over} and below 1 in {under}; Stupid Backoff example sums to 11/9"
    )


if __name__ == "__main__":
    main()
