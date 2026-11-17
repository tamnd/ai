# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/maximal-update-parametrization/ (ai://4VJECR).
# Check of derivations/feature-update.md: an update aligned with its input moves W x by n times its entry size (law
# of large numbers), a random matrix only by sqrt(n) times (central limit); after one SGD step SP moves the output by
# Theta(n eta), the last hidden preactivations by Theta(sqrt(n) eta) and the first by Theta(eta / sqrt(n)); after one
# Adam step SP moves the hidden preactivations by Theta(n eta); under muP all of them move by Theta(eta).
import numpy as np

rng = np.random.default_rng(0)
widths = np.array([64, 256, 1024, 4096])


def slope(vals):
    return float(np.polyfit(np.log(widths), np.log(vals), 1)[0])


# 1. aligned rank-one update versus an independent random matrix with entries of the same size
aligned, rand = [], []
for n in widths:
    x = rng.standard_normal(n)
    a = rng.standard_normal(n)
    eta = 0.01
    dW = -eta * np.outer(a, x)
    assert np.allclose(dW @ x, -eta * (x @ x) * a)
    aligned.append(np.sqrt(np.mean((dW @ x) ** 2)))
    R = eta * rng.choice([-1.0, 1.0], size=(n, n)) * np.abs(a)[:, None]
    rand.append(np.sqrt(np.mean((R @ x) ** 2)))
assert abs(slope(aligned) - 1.0) < 0.1 and abs(slope(rand) - 0.5) < 0.1

# 2. the readout alone: f = v . x, SGD on 0.5 (f - y)^2 changes f by -lr chi |x|^2
for n in widths:
    x = rng.standard_normal(n)
    v = rng.standard_normal(n) / np.sqrt(n)
    y = 1.0
    chi = v @ x - y
    for lr in (0.1, 0.1 / n):
        v1 = v - lr * chi * x
        assert np.isclose(v1 @ x - v @ x, -lr * chi * (x @ x))


# 3. a two-hidden-layer tanh network after one step, SP against muP (Table 3 with base width n0)
def one_step(n, param, opt, seed, n0=64, d=16, B=8):
    rng_data = np.random.default_rng(100)
    X = rng_data.standard_normal((B, d))
    y = np.ones((B, 1))
    g = np.random.default_rng(seed)
    r = n / n0
    U = g.standard_normal((n, d)) / np.sqrt(d)
    W = g.standard_normal((n, n)) / np.sqrt(n)
    V = g.standard_normal((1, n)) / np.sqrt(n * (r if param == "mup" else 1.0))
    if opt == "sgd":
        eta = 0.05
        lrU, lrW, lrV = (eta * r, eta, eta / r) if param == "mup" else (eta, eta, eta)
    else:
        eta = 0.003
        lrU, lrW, lrV = (eta, eta / r, eta / r) if param == "mup" else (eta, eta, eta)

    def fwd(U, W, V):
        h1 = X @ U.T
        x1 = np.tanh(h1)
        h2 = x1 @ W.T
        x2 = np.tanh(h2)
        return h1, x1, h2, x2, x2 @ V.T

    h1, x1, h2, x2, f = fwd(U, W, V)
    df = (f - y) / B
    dh2 = (df @ V) * (1 - x2**2)
    dh1 = (dh2 @ W) * (1 - x1**2)
    gV, gW, gU = df.T @ x2, dh2.T @ x1, dh1.T @ X
    if opt == "adam":  # the first Adam step with zero epsilon is the sign of the gradient
        gV, gW, gU = np.sign(gV), np.sign(gW), np.sign(gU)
    H1, _, H2, _, F = fwd(U - lrU * gU, W - lrW * gW, V - lrV * gV)
    rms = lambda z: float(np.sqrt(np.mean(z**2)))  # noqa: E731
    return rms(H1 - h1), rms(H2 - h2), rms(F - f)


res = {}
for param in ("sp", "mup"):
    for opt in ("sgd", "adam"):
        vals = np.array([np.mean([one_step(n, param, opt, s) for s in range(3)], axis=0) for n in widths])
        res[param, opt] = [slope(vals[:, j]) for j in range(3)]

s_in, s_hid, s_out = res["sp", "sgd"]
assert abs(s_in + 0.5) < 0.15 and abs(s_hid - 0.5) < 0.15 and abs(s_out - 1.0) < 0.15
assert abs(res["sp", "adam"][1] - 1.0) < 0.15
for opt in ("sgd", "adam"):
    assert all(abs(s) < 0.1 for s in res["mup", opt])
print("ok")
