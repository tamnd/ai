# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/neural-tangent-kernel/ (ai://3SJCNH).
# Check of derivations/kernel-flow.md: gradient descent on the linearized two-layer ReLU network with the loss
# (1/(2n)) |f_lin(X) - y|^2 gives the residual (I - eta K / n)^k r0 exactly, which tends to the flow exp(-eta K t / n) r0
# as the step shrinks at fixed time t = eta k, and each eigen-component decays at the rate eta lambda_i / n.
import numpy as np

rng = np.random.default_rng(0)
n, d, m = 8, 3, 512
X = rng.standard_normal((n, d))
X /= np.linalg.norm(X, axis=1, keepdims=True)
y = np.sin(2.0 * X[:, 0]) + X[:, 1]
W, a = rng.standard_normal((m, d)), rng.standard_normal(m)
U = X @ W.T
J = np.concatenate(
    [(((U > 0) * a)[:, :, None] * X[:, None, :]).reshape(n, -1), np.maximum(U, 0)],
    axis=1,
) / np.sqrt(m)
f0 = np.maximum(U, 0) @ a / np.sqrt(m)
K = J @ J.T
lam, V = np.linalg.eigh(K)
r0 = f0 - y
print("eigenvalues of the empirical NTK:", np.round(lam, 4))


def gd(eta, k):
    omega = np.zeros(J.shape[1])
    rs = [r0]
    for _ in range(k):
        omega -= eta * J.T @ (f0 + J @ omega - y) / n
        rs.append(f0 + J @ omega - y)
    return np.array(rs)


# 1. The discrete map is exact: r_k = (I - eta K / n)^k r0.
eta, k = 1.0, 200
rs = gd(eta, k)
exact = V @ ((1 - eta * lam / n) ** k * (V.T @ r0))
e1 = np.abs(rs[-1] - exact).max()
print(f"max |r_k - (I - eta K/n)^k r0| at eta={eta}, k={k}: {e1:.2e}")
assert e1 < 1e-10

# 2. The flow: at fixed time T = eta k, the gap to exp(-K T / n) r0 shrinks in proportion to the step.
T = 100.0
flow = V @ (np.exp(-lam * T / n) * (V.T @ r0))
gaps = []
for eta in (1.0, 0.5, 0.25, 0.125):
    g = np.abs(gd(eta, round(T / eta))[-1] - flow).max()
    gaps.append(g)
    print(f"eta={eta:<6} steps={round(T / eta):<5} max gap to the flow: {g:.3e}")
ratios = [gaps[i] / gaps[i + 1] for i in range(3)]
print("ratios of successive gaps (first order gives 2):", np.round(ratios, 3))
assert all(1.8 < q < 2.2 for q in ratios)

# 3. Each eigen-component decays at the rate eta lambda_i / n: fit log|v_i . r_k| against k.
eta, k = 0.05, 400
rs = gd(eta, k)
for i in range(n):
    c = rs @ V[:, i]
    rate = -np.polyfit(np.arange(k + 1), np.log(np.abs(c)), 1)[0]
    pred = eta * lam[i] / n
    print(
        f"lambda={lam[i]:.4f} fitted rate={rate:.6f} eta*lambda/n={pred:.6f} -log(1-eta*lambda/n)={-np.log(1 - pred):.6f}"
    )
    assert abs(rate - (-np.log(1 - pred))) < 1e-9
    assert abs(rate - pred) / pred < eta * lam[i] / n
print("ok")
