# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/edge-of-stability/ (ai://NEW-edge-of-stability).
# Check of derivations/quadratic-stability.md: on 0.5 x^T H x each eigen-coordinate is multiplied by 1 - eta lambda_i;
# gradient descent converges iff eta lambda_1 < 2; Cohen's 20 and 1 example at eta = 0.09 and 0.11; and the momentum
# thresholds (2 + 2 beta) / eta for Polyak and (2 + 2 beta) / (eta (1 + 2 beta)) for Nesterov.
import numpy as np

rng = np.random.default_rng(0)
for d in (2, 5, 20):
    A = rng.normal(size=(d, d))
    H = A @ A.T + 0.1 * np.eye(d)
    lam, Q = np.linalg.eigh(H)
    x0 = rng.normal(size=d)
    for scale, converges in ((0.99, True), (1.01, False)):
        eta = scale * 2 / lam[-1]
        x = x0.copy()
        for t in range(1, 301):
            x = x - eta * H @ x
            assert np.allclose(Q.T @ x, (1 - eta * lam) ** t * (Q.T @ x0), rtol=1e-8, atol=1e-10)
        top = abs(Q[:, -1] @ x)
        assert (top < abs(Q[:, -1] @ x0)) == converges

# the boundary eta lambda = 2 keeps the size and flips the sign
x = np.array([1.0])
for t in range(7):
    x = x - (2 / 3.0) * 3.0 * x
    assert x[0] == (-1) ** (t + 1)

# Cohen et al. (2021), Figure 2: eigenvalues 20 and 1
assert np.isclose(1 - 0.09 * 20, -0.8) and np.isclose(1 - 0.11 * 20, -1.2)
assert np.isclose(1 - 0.09 * 1, 0.91) and np.isclose(1 - 0.11 * 1, 0.89)
assert abs(0.8**100 - 2.037e-10) < 5e-14 and abs(1.2**100 / 8.282e7 - 1) < 1e-4
H = np.diag([20.0, 1.0])
for eta, grows in ((0.09, False), (0.11, True)):
    x = np.array([1.0, 1.0])
    for _ in range(100):
        x = x - eta * H @ x
    assert (abs(x[0]) > 1) == grows and abs(x[1]) < 1e-4


def run(lam, eta, beta, nesterov, steps):
    prev = x = 1.0
    for _ in range(steps):
        if nesterov:
            y = x + beta * (x - prev)
            new = y - eta * lam * y
        else:
            new = x - eta * lam * x + beta * (x - prev)
        prev, x = x, new
    return x


for beta in (0.0, 0.2, 0.5, 0.9):
    for nesterov in (False, True):
        eta = 0.01
        mss = (2 + 2 * beta) / (eta * (1 + 2 * beta)) if nesterov else (2 + 2 * beta) / eta
        assert abs(run(0.98 * mss, eta, beta, nesterov, 5000)) < 1e-3
        assert abs(run(1.02 * mss, eta, beta, nesterov, 400)) > 1e3
        for s in (0.98, 1.02):
            el = s * mss * eta
            b, c = ((1 - el) * (1 + beta), (1 - el) * beta) if nesterov else (1 + beta - el, beta)
            rho = max(abs(np.roots([1, -b, c])))
            assert (rho < 1) == (s < 1)
assert np.isclose((2 + 2 * 0.9) / 0.01, 380.0) and np.isclose((2 + 2 * 0.9) / (0.01 * 2.8), 135.714, atol=1e-3)
print("quadratic-stability: ok")
