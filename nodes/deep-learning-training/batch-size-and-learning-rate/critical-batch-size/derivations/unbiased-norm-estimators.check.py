# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/critical-batch-size/ (ai://X29NFT).
# Check of derivations/unbiased-norm-estimators.md. Per-example gradients come from a finite, skewed population; a
# batch draws from it with replacement. With B_small < B_big,
#   G2 = (B_big |G_big|^2 - B_small |G_small|^2) / (B_big - B_small),
#   S  = (|G_small|^2 - |G_big|^2) / (1 / B_small - 1 / B_big)
# must average to |G|^2 and tr(Sigma), whether the small batch is a part of the big one (one worker of a data-parallel
# step) or drawn apart from it. The per-example form, B_small = 1 averaged over the n examples of a batch, must equal
# the Bessel-corrected sample variance. The ratio S / G2 is checked to be biased while the ratio of the means is not.
import numpy as np

rng = np.random.default_rng(11)
d, N = 5, 40
pop = rng.gamma(0.5, 2.0, size=(N, d)) - 0.6  # skewed per-example gradients
G = pop.mean(axis=0)
trS = np.sum((pop - G) ** 2) / N  # the population variance, for sampling with replacement
print(f"|G|^2 = {G @ G:.4f}, tr(Sigma) = {trS:.4f}, B_simple = {trS / (G @ G):.3f}")


def estimates(sq_small, sq_big, b_small, b_big):
    g2 = (b_big * sq_big - b_small * sq_small) / (b_big - b_small)
    s = (sq_small - sq_big) / (1.0 / b_small - 1.0 / b_big)
    return g2, s


M = 400_000
for b_small, workers in [(1, 8), (4, 8), (16, 2)]:
    b_big = b_small * workers
    idx = rng.integers(0, N, size=(M, b_big))
    per = pop[idx]  # (M, b_big, d)
    big = per.mean(axis=1)
    small = per[:, :b_small].mean(axis=1)  # the first worker's part of the same step
    apart = pop[rng.integers(0, N, size=(M, b_small))].mean(axis=1)  # a small batch drawn apart
    for name, sm in [("nested", small), ("apart", apart)]:
        g2, s = estimates(np.sum(sm**2, 1), np.sum(big**2, 1), b_small, b_big)
        z_g = (g2.mean() - G @ G) / (g2.std() / np.sqrt(M))
        z_s = (s.mean() - trS) / (s.std() / np.sqrt(M))
        print(f"B_small={b_small:2d} B_big={b_big:3d} {name:6s}  E[G2]={g2.mean():.4f} (z={z_g:+.2f})  "
              f"E[S]={s.mean():.4f} (z={z_s:+.2f})")
        assert abs(z_g) < 4.5 and abs(z_s) < 4.5

# B_small = 1 averaged over the n examples of one batch is the Bessel-corrected sample variance, exactly
n = 12
x = pop[rng.integers(0, N, size=n)]
mean = x.mean(axis=0)
g2, s = estimates(np.mean(np.sum(x**2, 1)), mean @ mean, 1, n)
assert np.isclose(s, np.sum((x - mean) ** 2) / (n - 1))
assert np.isclose(g2, mean @ mean - s / n)

# the ratio of the estimates is biased; the ratio of their means is not
b_small, b_big = 8, 64
idx = rng.integers(0, N, size=(M, b_big))
per = pop[idx]
g2, s = estimates(np.sum(per[:, :b_small].mean(1) ** 2, 1), np.sum(per.mean(1) ** 2, 1), b_small, b_big)
ok = g2 > 0
ratios = s[ok] / g2[ok]
print(f"P(G2 > 0) = {ok.mean():.3f}; median of S/G2 = {np.median(ratios):.3f}; "
      f"ratio of means = {s.mean() / g2.mean():.3f}; true = {trS / (G @ G):.3f}")
assert abs(s.mean() / g2.mean() / (trS / (G @ G)) - 1) < 0.03
assert abs(np.median(ratios) / (trS / (G @ G)) - 1) > 0.05
print("ok")
