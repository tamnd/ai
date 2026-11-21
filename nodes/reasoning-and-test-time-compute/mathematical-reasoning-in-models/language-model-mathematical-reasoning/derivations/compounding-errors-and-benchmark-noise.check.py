# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/language-model-mathematical-reasoning/ (ai://NEW-language-model-mathematical-reasoning).
"""Checks every number of compounding-errors-and-benchmark-noise.md."""

import math

import numpy as np


def close(x, y, tol=5e-5):
    assert abs(x - y) < tol, (x, y)


# 1. compounding
p = 0.95
for t, v in [(2, 0.9025), (4, 0.8145), (8, 0.6634), (16, 0.4401)]:
    close(p**t, v)
for t, v in [(2, 0.0451), (4, 0.0407), (8, 0.0332)]:
    close(p**t * (1 - p), v)
gemma, phi, mini = [84.4, 79.1, 68.1, 41.8], [87.6, 82.1, 64.8, 44.8], [92.5, 91.7, 81.1, 72.4]
ratios = lambda a: [round(a[i + 1] / a[i], 3) for i in range(3)]  # noqa: E731
assert ratios(gemma) == [0.937, 0.861, 0.614]
assert [round(gemma[i] - gemma[i + 1], 1) for i in range(3)] == [5.3, 11.0, 26.3]
assert ratios(phi) == [0.937, 0.789, 0.691]
assert ratios(mini) == [0.991, 0.884, 0.893]
for a in (gemma, phi):
    r = ratios(a)
    assert r[0] > r[1] > r[2]
assert not ratios(mini)[1] > ratios(mini)[2]

# 2. digits of a product
rng = np.random.default_rng(0)
a, b = rng.integers(0, 10**6, 100_000), rng.integers(0, 10**6, 100_000)
for j in (1, 2, 3):
    assert np.array_equal((a * b) % 10**j, ((a % 10**j) * (b % 10**j)) % 10**j)
close(1 - (1 - 1e-4) ** 200_000, 1 - 2.1e-9, 1e-10)
close(1 - (1 - 1e-6) ** 200_000, 0.1813)
draws = rng.integers(0, 10**6, 200_000)
close(len(set(draws.tolist())) / 10**6, 0.1813, 2e-3)
for n, v in [(2, 0.9606), (3, 0.9135), (5, 0.7778)]:
    close(0.99 ** (n * n), v)

# 3. spread across sets
close(100 * math.sqrt(0.791 * 0.209 / 100), 4.07, 5e-3)
close(0.030**2 * 100**2, 9.0, 1e-9)
ps = rng.uniform(0, 1, 100)
assert math.sqrt(np.sum(ps * (1 - ps))) / 100 <= math.sqrt(ps.mean() * (1 - ps.mean()) / 100)
sims = [np.mean(rng.random(100) < ps) for _ in range(20_000)]
close(float(np.std(sims)), math.sqrt(np.sum(ps * (1 - ps))) / 100, 1e-3)

# 4. gap between two benchmarks
se = math.sqrt(0.16 / 1205 + 0.16 / 1319)
close(se, 0.0159, 1e-4)
close(8 / (100 * se), 5.02, 5e-3)
close(200 * se, 3.19, 5e-3)


# 5. voting and pass@k
def vote(q, w, k):
    r, acc = 1 - q - w, 0.0
    for c in range(1, k + 1):
        for s in range(k - c + 1):
            u = k - c - s
            pr = math.comb(k, c) * math.comb(k - c, s) * q**c * w**s * r**u
            top = max([s] + [1] * u)
            acc += pr if c > top else pr / (1 + (s == top) + u * (top == 1)) if c == top else 0
    return acc


for (q, w), vals in {
    (0.4, 0.0): [0.4, 0.7149, 0.9951, 1.0],
    (0.4, 0.2): [0.4, 0.6426, 0.8455, 0.9543],
    (0.4, 0.5): [0.4, 0.4039, 0.3411, 0.2493],
}.items():
    for k, v in zip((1, 5, 15, 41), vals, strict=True):
        close(vote(q, w, k), v)


def passk(n, c, k):
    return 1 - math.comb(n - c, k) / math.comb(n, k)


close(passk(256, 10, 1), 0.0391)
close(passk(256, 10, 64), 0.9470)
close(passk(256, 1, 64), 0.25, 1e-12)
close(passk(256, 1, 256), 1.0, 1e-12)
print("ok")
