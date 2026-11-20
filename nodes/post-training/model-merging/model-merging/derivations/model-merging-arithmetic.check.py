# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/model-merging/ (ai://NEW-model-merging).
"""Checks the statements of model-merging-arithmetic.md numerically."""

import math

import numpy as np

rng = np.random.default_rng(0)

# 1: task arithmetic contains the uniform soup and WiSE-FT.
pre = rng.normal(size=7)
fts = pre + rng.normal(size=(4, 7)) * 0.1
taus = fts - pre
n = len(fts)
for lam in (0.1, 0.25, 0.4, 1.0):
    assert np.allclose(pre + lam * taus.sum(0), (1 - n * lam) * pre + lam * fts.sum(0))
assert np.allclose(pre + taus.sum(0) / n, fts.mean(0))
for a in (0.0, 0.3, 0.5, 1.0):
    assert np.allclose(pre + a * taus[0], (1 - a) * pre + a * fts[0])

# 2: Fisher merging is a per-coordinate weighted average, and the average when the Fishers are equal.
thetas = rng.normal(size=(3, 6))
F = rng.random((3, 6)) + 0.01
lam = np.array([0.2, 0.5, 0.3])
merged = (lam[:, None] * F * thetas).sum(0) / (lam[:, None] * F).sum(0)


def objective(t):
    return -0.5 * np.sum(lam[:, None] * F * (t - thetas) ** 2)


h = 1e-6
for j in range(6):
    e = np.zeros(6)
    e[j] = h
    assert abs((objective(merged + e) - objective(merged - e)) / (2 * h)) < 1e-8
for _ in range(100):
    assert objective(merged + rng.normal(size=6) * 0.1) < objective(merged)
f = rng.random(6) + 0.1
eq = (lam[:, None] * f * thetas).sum(0) / (lam[:, None] * f).sum(0)
assert np.allclose(eq, (lam[:, None] * thetas).sum(0))
ex = (np.array([3, 1]) * [1, 0] + np.array([1, 1]) * [0, 1]) / np.array([4, 2])
assert np.allclose(ex, [0.75, 0.5])

# 3: weight averaging equals output ensembling for a linear head, not through tanh.
phi = rng.normal(size=(10, 5))
Ws, bs = rng.normal(size=(3, 4, 5)), rng.normal(size=(3, 4))
w = np.array([0.5, 0.3, 0.2])
avg_out = sum(w[i] * (phi @ Ws[i].T + bs[i]) for i in range(3))
out_avg = phi @ np.tensordot(w, Ws, 1).T + w @ bs
assert np.allclose(avg_out, out_avg)
assert abs(math.tanh(2.0) - 0.964028) < 1e-6 and abs(0.5 * math.tanh(4.0) - 0.499665) < 1e-6

# 4: DARE's mean and variance.
for p, want in [(0.9, 9.0), (0.99, 99.0)]:
    assert abs(p / (1 - p) - want) < 1e-9
    delta = 0.002
    mask = rng.random(2_000_000) >= p
    x = mask * delta / (1 - p)
    se = math.sqrt(delta**2 * p / (1 - p) / x.size)
    assert abs(x.mean() - delta) < 5 * se
    assert abs(x.var() / (delta**2 * p / (1 - p)) - 1) < 0.05
    assert abs((mask * delta).mean() - (1 - p) * delta) < 5 * se

# 5: the SWA running mean is exact.
snaps = rng.normal(size=(20, 8))
avg = snaps[0].copy()
for k in range(1, len(snaps)):
    avg = (avg * k + snaps[k]) / (k + 1)
    assert np.allclose(avg, snaps[: k + 1].mean(0))

# 6: the disjoint mean is at least the plain mean in the elected direction.
for _ in range(2000):
    a = rng.normal(size=int(rng.integers(1, 8)))
    a[rng.random(a.size) < 0.3] = 0.0
    s = np.sign(a.sum())
    agree = a[np.sign(a) == s]
    if s == 0 or agree.size == 0:
        continue
    dm = agree.mean()
    assert s * dm >= s * a.mean() - 1e-12


# 7: the loss barrier.
def barrier(loss, A, B, points=101):
    ls = [loss((1 - t) * A + t * B) for t in np.linspace(0, 1, points)]
    return max(ls) - 0.5 * (ls[0] + ls[-1])


for _ in range(200):
    A, B = rng.normal(size=3), rng.normal(size=3)
    c = rng.normal(size=3)
    k = rng.normal(size=3)

    def nonconvex(z, c=c, k=k):
        return float(np.sin(z @ k) ** 2 + 0.1 * z @ z)

    def convex(z, c=c):
        return float(np.sum((z - c) ** 2))

    for loss in (nonconvex, convex):
        bar = barrier(loss, A, B)
        assert bar >= 0.5 * abs(loss(A) - loss(B)) - 1e-12
    assert abs(barrier(convex, A, B) - 0.5 * abs(convex(A) - convex(B))) < 1e-12
cA = np.array([1.0, 0.0])
assert barrier(lambda z: float((z - cA) @ (z - cA)), np.array([0.0, 0.0]), np.array([2.0, 0.0])) == 0.0

# 8: permutations.
W1, W2 = rng.normal(size=(6, 3)), rng.normal(size=(2, 6))
P = np.eye(6)[rng.permutation(6)]
xs = rng.normal(size=(3, 50))


def net(a, b):
    return b @ np.tanh(a @ xs)


assert np.allclose(net(P @ W1, W2 @ P.T), net(W1, W2))
mixed = net((W1 + P @ W1) / 2, (W2 + W2 @ P.T) / 2)
assert not np.allclose(mixed, net(W1, W2))
assert np.allclose(net((W1 + P.T @ (P @ W1)) / 2, (W2 + (W2 @ P.T) @ P) / 2), net(W1, W2))

# 9: the TIES worked example.
T = np.array([[0.5, -0.2, 0.05, 0.8, -0.6], [-0.45, -0.4, 0.02, 0.6, 0.1], [0.4, 0.1, -0.01, -0.05, -0.7]])
trimmed = np.zeros_like(T)
for i, t in enumerate(T):
    keep = np.argsort(-np.abs(t), kind="stable")[:3]
    trimmed[i, keep] = t[keep]
assert np.allclose(trimmed, [[0.5, 0, 0, 0.8, -0.6], [-0.45, -0.4, 0, 0.6, 0], [0.4, 0.1, 0, 0, -0.7]])
assert np.allclose(trimmed.sum(0), [0.45, -0.3, 0, 1.4, -1.3])
sign = np.sign(trimmed.sum(0))
agree = (np.sign(trimmed) == sign) & (trimmed != 0)
tm = np.where(agree.any(0), (trimmed * agree).sum(0) / np.maximum(agree.sum(0), 1), 0.0)
assert np.allclose(tm, [0.45, -0.4, 0, 0.7, -0.65])
assert np.allclose(T.mean(0), [0.15, -0.5 / 3, 0.02, 1.35 / 3, -0.4])
assert np.allclose(np.round(T.mean(0), 3), [0.15, -0.167, 0.02, 0.45, -0.4])

print("model-merging-arithmetic: ok")
