# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/ai-safety-via-debate/ (ai://N45PC9).
# Check of derivations/bisection-debate.md: the bisection debates of Irving et al. (2018, Section 2.2) for counting
# primes and for a deterministic simulation, solved exactly by minimax over every claim in a range, the number of
# rounds and judge operations, the logarithmic score of Michael et al. (2023, Section 2.1), and the numbers of the
# worked example.
import math
from functools import lru_cache


def is_prime(k):
    if k < 2:
        return False
    return all(k % d for d in range(2, math.isqrt(k) + 1))


def count(a, b):
    return sum(is_prime(k) for k in range(a, b))


SLACK = 3  # claims searched: every integer from -SLACK to (b - a) + SLACK


@lru_cache(maxsize=None)
def count_win(a, b, c):
    """True when Alice, having claimed c primes in [a, b), wins against every reply of Bob."""
    if b - a == 1:
        return c == int(is_prime(a))  # the judge checks one number for primality
    m = (a + b) // 2
    for cl in range(-SLACK, (m - a) + SLACK + 1):
        cr = c - cl  # the judge subtracts: the right half gets the rest of the claim
        if count_win(a, m, cl) and count_win(m, b, cr):  # Bob may challenge either half
            return True
    return False


def play_count(n, claim, alice):
    """Play the counting debate on [0, 2^n) with a given Alice split rule and Bob's rule: challenge a wrong half."""
    a, b, c, rounds, subtractions = 0, 2 ** n, claim, 0, 0
    while b - a > 1:
        m = (a + b) // 2
        cl = alice(a, m, b, c)
        cr = c - cl
        subtractions += 1
        rounds += 1
        if cl != count(a, m):
            b, c = m, cl
        elif cr != count(m, b):
            a, c = m, cr
        else:
            b, c = m, cl  # both halves are true; Bob challenges one anyway and the claim stays true
    return int(is_prime(a)) == c, rounds, subtractions, a


def step(s, mod):
    return (s * s + 1) % mod  # a deterministic step on a small state space


@lru_cache(maxsize=None)
def sim_win(t0, t1, x, y, mod):
    """True when Alice, having claimed state y at time t1 from the agreed state x at t0, wins against every reply."""
    if t1 - t0 == 1:
        return step(x, mod) == y  # the judge checks one step
    tm = (t0 + t1) // 2
    return any(sim_win(t0, tm, x, z, mod) and sim_win(tm, t1, z, y, mod) for z in range(mod))


def run(x, k, mod):
    for _ in range(k):
        x = step(x, mod)
    return x


def main():
    # 1. Counting: Alice wins exactly when her claim is the true count, for every claim in the searched range.
    for n in range(0, 7):
        x = 2 ** n
        truth = count(0, x)
        for c in range(-SLACK, x + SLACK + 1):
            assert count_win(0, x, c) == (c == truth), (n, c)
    # every dyadic interval, not only [0, 2^n)
    for n in range(0, 5):
        size = 2 ** n
        for a in range(0, 64, size):
            for c in range(-SLACK, size + SLACK + 1):
                assert count_win(a, a + size, c) == (c == count(a, a + size))
    # 2. Rounds and judge work: n rounds, n subtractions and one primality test, whatever Alice does.
    import random
    rng = random.Random(20261116)
    for n in range(1, 11):
        x = 2 ** n
        truth = count(0, x)
        honest = lambda a, m, b, c: count(a, m)
        won, rounds, subs, _ = play_count(n, truth, honest)
        assert won and rounds == n and subs == n == round(math.log2(x))
        for _ in range(200):
            lie = truth + rng.choice([-3, -2, -1, 1, 2, 3])
            liar = lambda a, m, b, c: rng.randint(-2, (m - a) + 2)
            won, rounds, subs, _ = play_count(n, lie, liar)
            assert not won and rounds == n and subs == n
    # 3. Simulation: Alice wins exactly when her claimed final state is the true one; 2^k steps take k rounds.
    for mod in (5, 7, 11):
        for k in range(0, 4):
            t = 2 ** k
            for x in range(mod):
                truth = run(x, t, mod)
                for y in range(mod):
                    assert sim_win(0, t, x, y, mod) == (y == truth), (mod, k, x, y)
    # 4. The judge's logarithmic score: the expected score q log2 p + (1 - q) log2 (1 - p) peaks at p = q only.
    grid = [i / 1000 for i in range(1, 1000)]
    for q in (0.1, 0.25, 0.5, 0.7, 0.9):
        expected = [q * math.log2(p) + (1 - q) * math.log2(1 - p) for p in grid]
        best = grid[max(range(len(grid)), key=expected.__getitem__)]
        assert abs(best - q) < 1e-9, (q, best)
    # 5. The worked example.
    assert count(0, 16) == 6 and [k for k in range(16) if is_prime(k)] == [2, 3, 5, 7, 11, 13]
    assert (count(0, 8), count(8, 16)) == (4, 2)
    assert (count(8, 12), count(12, 16)) == (1, 1)
    assert (count(12, 14), count(14, 16)) == (1, 0)
    assert (count(14, 15), count(15, 16)) == (0, 0) and not is_prime(15)
    script = {(0, 16): 4, (8, 16): 1, (12, 16): 1, (14, 16): 0}  # Alice's claimed left halves after claiming 7
    won, rounds, subs, leaf = play_count(4, 7, lambda a, m, b, c: script[(a, b)])
    assert not won and rounds == 4 and subs == 4 and leaf == 15
    assert not count_win(0, 16, 7) and count_win(0, 16, 6)
    assert math.isclose(math.log2(0.5) - 0.05 * 0, -1.0)
    assert abs((math.log2(0.9) - 0.05 * 2) - (-0.252)) < 5e-4
    assert abs((math.log2(0.6) - 0.05 * 1) - (-0.787)) < 5e-4
    print("ok")


if __name__ == "__main__":
    main()
