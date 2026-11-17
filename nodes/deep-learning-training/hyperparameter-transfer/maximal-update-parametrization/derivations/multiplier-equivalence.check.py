# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/maximal-update-parametrization/ (ai://NEW-maximal-update-parametrization).
# Check of derivations/multiplier-equivalence.md: a layer A W x with W ~ N(0, B^2) and learning rate C computes the
# same function at every step after (A, B, C) -> (A t, B / t, C / t^2) under SGD and (A t, B / t, C / t) under Adam
# with epsilon 0; so the readout of Table 3 (A = 1, B^2 = 1/n^2, C = eta/n) equals the readout with multiplier 1/n,
# B = 1 and C = eta n (SGD) or C = eta (Adam) of Table 8.
import numpy as np

rng = np.random.default_rng(2)
d, n, B, steps = 5, 32, 6, 6
X = rng.standard_normal((B, d))
Y = rng.standard_normal((B, 1))
Z = {k: rng.standard_normal(s) for k, s in (("U", (n, d)), ("W", (n, n)), ("V", (1, n)))}


def train(A, Bs, C, opt):
    P = {k: Bs[k] * Z[k] for k in Z}
    m = {k: np.zeros_like(v) for k, v in P.items()}
    s = {k: np.zeros_like(v) for k, v in P.items()}
    outs = []
    for t in range(1, steps + 1):
        h1 = A["U"] * X @ P["U"].T
        x1 = np.tanh(h1)
        h2 = A["W"] * x1 @ P["W"].T
        x2 = np.tanh(h2)
        f = A["V"] * x2 @ P["V"].T
        outs.append(f.copy())
        df = (f - Y) / B
        dx2 = A["V"] * df @ P["V"]
        dh2 = dx2 * (1 - x2**2)
        dx1 = A["W"] * dh2 @ P["W"]
        dh1 = dx1 * (1 - x1**2)
        g = {"V": A["V"] * df.T @ x2, "W": A["W"] * dh2.T @ x1, "U": A["U"] * dh1.T @ X}
        for k in P:
            if opt == "sgd":
                P[k] = P[k] - C[k] * g[k]
            else:
                m[k] = 0.9 * m[k] + 0.1 * g[k]
                s[k] = 0.999 * s[k] + 0.001 * g[k] ** 2
                P[k] = P[k] - C[k] * (m[k] / (1 - 0.9**t)) / np.sqrt(s[k] / (1 - 0.999**t))
    return np.array(outs)


A0 = {"U": 1.0, "W": 1.0, "V": 1.0}
B0 = {"U": 1 / np.sqrt(d), "W": 1 / np.sqrt(n), "V": 1 / n}
for opt, C0, power in (
    ("sgd", {"U": 0.05 * n, "W": 0.05, "V": 0.05 / n}, 2),
    ("adam", {"U": 0.01, "W": 0.01 / n, "V": 0.01 / n}, 1),
):
    base = train(A0, B0, C0, opt)
    for _ in range(5):
        th = {k: float(np.exp(rng.uniform(-2, 2))) for k in Z}
        A1 = {k: A0[k] * th[k] for k in Z}
        B1 = {k: B0[k] / th[k] for k in Z}
        C1 = {k: C0[k] / th[k] ** power for k in Z}
        assert np.allclose(train(A1, B1, C1, opt), base, rtol=1e-8, atol=1e-10)
    # the wrong power changes the trajectory
    th = {"U": 1.0, "W": 1.0, "V": 4.0}
    wrong = train(
        {k: A0[k] * th[k] for k in Z},
        {k: B0[k] / th[k] for k in Z},
        {k: C0[k] / th[k] ** (3 - power) for k in Z},
        opt,
    )
    assert not np.allclose(wrong[1:], base[1:], rtol=1e-3)
    # Table 3 readout to Table 8 readout: t = 1/n
    t8 = {"U": 1.0, "W": 1.0, "V": 1.0 / n}
    A8 = {k: A0[k] * t8[k] for k in Z}
    B8 = {k: B0[k] / t8[k] for k in Z}
    C8 = {k: C0[k] / t8[k] ** power for k in Z}
    assert np.isclose(A8["V"], 1 / n) and np.isclose(B8["V"], 1.0)
    assert np.isclose(C8["V"], 0.05 * n if opt == "sgd" else 0.01)
    assert np.allclose(train(A8, B8, C8, opt), base, rtol=1e-8, atol=1e-10)
print("ok")
