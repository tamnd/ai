# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/chain-of-thought-prompting/ (ai://NEW-chain-of-thought-prompting).
# Check of derivations/majority-vote.md: the binomial formula of the two-answer vote against a sum over all outcomes,
# the exact step P_{m+2} - P_m, the Hoeffding bounds, the equality of even and odd sample counts under fair ties, the
# first-seen tie rule against uniform ties, the plurality limit over three answers in exact fractions and by
# simulation, the benchmark example where voting lowers accuracy, and the numbers of the page.
import itertools
from fractions import Fraction as F
from math import comb, exp, factorial

import numpy as np


def vote_binary(p, m):
    """Exact probability that the majority of m (odd) two-answer samples is correct."""
    return sum(comb(m, j) * p**j * (1 - p) ** (m - j) for j in range(m // 2 + 1, m + 1))


def vote_binary_fair_ties(p, m):
    """Exact accuracy for any m, ties broken by a fair coin."""
    win = sum(comb(m, j) * p**j * (1 - p) ** (m - j) for j in range(m // 2 + 1, m + 1))
    if m % 2 == 0:
        win += F(1, 2) * comb(m, m // 2) * p ** (m // 2) * (1 - p) ** (m // 2)
    return win


def brute_binary(p, m):
    """Sum over all 2^m outcomes, a fair coin on ties."""
    total = F(0)
    for out in itertools.product((0, 1), repeat=m):
        k = sum(out)
        pr = p**k * (1 - p) ** (m - k)
        if 2 * k > m:
            total += pr
        elif 2 * k == m:
            total += pr / 2
    return total


def plurality_exact(q, m):
    """Exact probability that the plurality of m samples over len(q) answers is answer 0, fair ties."""
    kk = len(q)
    total = F(0)
    for counts in itertools.product(range(m + 1), repeat=kk - 1):
        rest = m - sum(counts)
        if rest < 0:
            continue
        n = (rest,) + counts
        top = max(n)
        if n[0] != top:
            continue
        ties = sum(1 for x in n if x == top)
        coef = factorial(m)
        for x in n:
            coef //= factorial(x)
        pr = F(coef)
        for qa, x in zip(q, n, strict=True):
            pr *= qa**x
        total += pr / ties
    return total


def first_seen_vs_uniform(q, m):
    """For every count vector, the chance that the first-seen rule picks each tied answer, over all orders."""
    kk = len(q)
    first = {}
    for seq in itertools.product(range(kk), repeat=m):
        pr = F(1)
        for a in seq:
            pr *= q[a]
        n = [seq.count(a) for a in range(kk)]
        top = max(n)
        tied = [a for a in range(kk) if n[a] == top]
        if len(tied) < 2:
            continue
        pick = next(a for a in seq if a in tied)
        key = tuple(n)
        first.setdefault(key, {a: F(0) for a in tied})
        first[key][pick] += pr
    for key, d in first.items():
        vals = list(d.values())
        assert all(v == vals[0] for v in vals), (key, d)
    return len(first)


def main():
    ps = [F(1, 10), F(1, 4), F(2, 5), F(1, 2), F(3, 5), F(2, 3), F(3, 4), F(9, 10)]
    for p in ps:
        for m in range(1, 12):
            assert vote_binary_fair_ties(p, m) == brute_binary(p, m), (p, m)
    # Step identity and sign.
    for i in range(1, 21):
        p = F(i, 21)
        for m in range(1, 42, 2):
            h = (m + 1) // 2
            step = vote_binary(p, m + 2) - vote_binary(p, m)
            assert step == comb(m, h) * p**h * (1 - p) ** h * (2 * p - 1)
            assert (step > 0) == (p > F(1, 2)) and (step < 0) == (p < F(1, 2))
        assert vote_binary(p, 1) == p
    # Hoeffding bounds of item 3.
    for p in [0.55, 0.6, 0.7, 0.9]:
        for m in range(1, 202, 2):
            err = 1 - float(vote_binary(F(p).limit_denominator(1000), m))
            assert err <= exp(-2 * m * (p - 0.5) ** 2) + 1e-15
            acc = float(vote_binary(F(1 - p).limit_denominator(1000), m))
            assert acc <= exp(-2 * m * (p - 0.5) ** 2) + 1e-15
    # Even m equals the odd m below it with fair ties.
    for p in ps:
        for k in range(1, 15):
            assert vote_binary_fair_ties(p, 2 * k) == vote_binary(p, 2 * k - 1)
    # First-seen ties are uniform ties.
    tied_vectors = 0
    for q in [(F(1, 2), F(1, 3), F(1, 6)), (F(1, 5), F(3, 10), F(1, 2))]:
        for m in range(2, 8):
            tied_vectors += first_seen_vs_uniform(q, m)
    # Plurality over three answers: exact, bound and simulation.
    q = (F(2, 5), F(7, 20), F(1, 4))  # the correct answer is the mode with 0.4 < 1/2
    prev = None
    for m in (1, 5, 11, 21, 41, 61):
        acc = plurality_exact(q, m)
        bound = 2 * exp(-m * (0.4 - 0.35) ** 2 / 2)
        assert 1 - float(acc) <= bound + 1e-15
        if prev is not None:
            assert acc > prev
        prev = acc
    q_bad = (F(3, 10), F(1, 2), F(1, 5))  # a wrong answer is the mode
    for m in (11, 21, 41, 61):
        assert float(plurality_exact(q_bad, m)) <= exp(-m * (0.5 - 0.3) ** 2 / 2)
    rng = np.random.default_rng(20261115)
    m, trials = 21, 20000
    draws = rng.choice(3, size=(trials, m), p=[float(x) for x in q])
    counts = np.stack([(draws == a).sum(axis=1) for a in range(3)], axis=1)
    top = counts.max(axis=1)
    ties = (counts == top[:, None]).sum(axis=1)
    sim = float(np.mean((counts[:, 0] == top) / ties))
    exact = float(plurality_exact(q, m))
    assert abs(sim - exact) < 4 * np.sqrt(exact * (1 - exact) / trials), (sim, exact)
    # The benchmark example of item 7.
    single = (F(9, 10) + F(3, 10)) / 2
    assert single == F(3, 5)
    a1 = vote_binary(F(9, 10), 201)
    a2 = plurality_exact((F(3, 10), F(1, 2), F(1, 5)), 151)
    assert 1 - a1 < F(1, 10**40) and a2 < F(1, 300)
    limit = (a1 + a2) / 2
    assert abs(float(limit) - 0.5) < 0.01 and limit < single
    # Numbers of the page.
    assert vote_binary(F(3, 5), 3) == F(81, 125) == F(648, 1000)
    assert vote_binary(F(3, 5), 5) == F(68256, 100000)
    assert vote_binary(F(2, 5), 3) == F(352, 1000)
    assert vote_binary(F(3, 5), 21) == F(393687311953653, 476837158203125)
    assert round(float(vote_binary(F(3, 5), 21)), 4) == 0.8256
    print(
        f"ok: binomial vote matches enumeration, step identity exact, bounds hold, even = odd with fair ties, "
        f"first-seen ties uniform on {tied_vectors} tied count vectors, plurality exact vs simulation "
        f"{exact:.4f} vs {sim:.4f}, benchmark example {float(single)} -> {float(limit):.4f}"
    )


if __name__ == "__main__":
    main()
