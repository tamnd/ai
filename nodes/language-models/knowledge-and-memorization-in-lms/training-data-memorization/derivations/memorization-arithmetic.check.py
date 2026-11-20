# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/training-data-memorization/ (ai://NEW-training-data-memorization).
# Check of derivations/memorization-arithmetic.md: the power law E(d) = c d^s gives E(10)/E(1) = 10^s and generations
# per copy rising with d when s > 1; a log-linear extractable fraction gains b log10 2 per doubling and must stop at
# F = 1; and the singleton term of Feldman's bound for Zipf and uniform frequencies over n = 50,000 examples.
import bisect
import itertools
import math
import random

rng = random.Random(11)

# item 1
for _ in range(1000):
    c = math.exp(rng.uniform(-5.0, 5.0))
    s = rng.uniform(0.5, 4.0)
    e = [c * d**s for d in range(1, 101)]
    assert math.isclose(e[9] / e[0], 10**s, rel_tol=1e-12)
    per_copy = [e[d - 1] / d for d in range(1, 101)]
    pairs = list(itertools.pairwise(per_copy))
    if s > 1:
        assert all(b > a for a, b in pairs)
    elif s < 1:
        assert all(b < a for a, b in pairs)
assert math.isclose(math.log10(1000), 3.0)
assert math.isclose(10 ** (3 - 1), 100.0)

# item 2
b = 0.19
step = b * math.log10(2)
assert abs(step - 0.0572) < 5e-5
for _ in range(1000):
    a = rng.uniform(-1.5, 0.5)
    n_par = 10 ** rng.uniform(6, 12)
    f = a + b * math.log10(n_par)
    assert math.isclose(a + b * math.log10(2 * n_par) - f, step, rel_tol=1e-9)
    limit = (1 - a) / b
    assert a + b * (limit + 1e-9) > 1 >= a + b * (limit - 1e-9)


# item 3
def singleton_terms(alpha, n):
    w = [x * (1 - x) ** (n - 1) for x in alpha]
    sigma = math.fsum(w)
    tau1 = math.fsum(x * wi for x, wi in zip(alpha, w, strict=True)) / sigma
    return tau1, sigma


n = 50_000
harm = math.fsum(1 / i for i in range(1, n + 1))
zipf = [1 / (i * harm) for i in range(1, n + 1)]
assert math.isclose(math.fsum(zipf), 1.0, rel_tol=1e-12)
tau1, sigma = singleton_terms(zipf, n)
excess = 0.9 * tau1 * n * sigma
print(
    f"Zipf: H = {harm:.4f}, tau1 n = {tau1 * n:.4f}, min alpha n = {n / (n * harm):.4f}, sigma = {sigma:.4f}"
)
print(f"Zipf: added error >= {excess:.4f}")
assert abs(harm - 11.397) < 5e-4
assert abs(tau1 * n - 0.472) < 5e-4
assert abs(1 / harm - 0.088) < 5e-4
assert abs(sigma - 0.170) < 5e-4
assert abs(excess - 0.072) < 5e-4

for k, want_sigma, want_excess in ((5000, 4.54e-5, 4.1e-4), (1000, 1.9e-22, None)):
    uni = [1 / k] * k
    t, sg = singleton_terms(uni, n)
    assert math.isclose(t, 1 / k, rel_tol=1e-12)
    assert math.isclose(sg, (1 - 1 / k) ** (n - 1), rel_tol=1e-9)
    print(f"uniform K = {k}: sigma = {sg:.3g}, added error >= {0.9 * t * n * sg:.3g}")
    assert abs(sg / want_sigma - 1) < 0.01
    if want_excess is not None:
        assert abs(0.9 * t * n * sg / want_excess - 1) < 0.01

# simulation of the singleton fraction under the Zipf law
cum = list(itertools.accumulate(zipf))
fracs = []
for _ in range(200):
    counts = [0] * n
    for _ in range(n):
        counts[min(bisect.bisect_left(cum, rng.random()), n - 1)] += 1
    fracs.append(sum(1 for x in counts if x == 1) / n)
mean = math.fsum(fracs) / len(fracs)
print(f"simulated singleton fraction {mean:.4f} against sigma {sigma:.4f}")
assert abs(mean - sigma) < 0.005
print("memorization-arithmetic: ok")
