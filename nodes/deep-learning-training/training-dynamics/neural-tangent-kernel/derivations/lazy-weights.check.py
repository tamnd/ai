# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/neural-tangent-kernel/ (ai://3SJCNH).
# Check of derivations/lazy-weights.md: for the two-layer ReLU network in the NTK parametrization, trained by gradient
# descent on the loss (1/(2n)) |f(X) - y|^2, the relative change |theta_T - theta_0| / |theta_0| falls like 1/sqrt(m),
# the linearized model obeys |omega| <= |r0| / sqrt(lambda_min), and at a fixed width a scale factor alpha on the
# centred model divides the displacement by alpha.
import numpy as np

rng = np.random.default_rng(2)
n, d = 6, 3
X = rng.standard_normal((n, d))
X /= np.linalg.norm(X, axis=1, keepdims=True)
y = np.sin(3.0 * X[:, 0])


def parts(W, a):
    m = a.shape[0]
    U = X @ W.T
    f = np.maximum(U, 0) @ a / np.sqrt(m)
    J = np.concatenate(
        [(((U > 0) * a)[:, :, None] * X[:, None, :]).reshape(n, -1), np.maximum(U, 0)],
        axis=1,
    )
    return f, J / np.sqrt(m)


def train(m, seed, eta=1.0, steps=300, alpha=None):
    """Gradient descent from a fresh draw. With alpha, the model is alpha (f - f_0) and the loss is divided by alpha^2."""
    r = np.random.default_rng(seed)
    W, a = r.standard_normal((m, d)), r.standard_normal(m)
    th0 = np.concatenate([W.ravel(), a])
    f_init = parts(W, a)[0]
    th = th0.copy()
    for _ in range(steps):
        f, J = parts(th[: m * d].reshape(m, d), th[m * d :])
        if alpha is None:
            th -= eta * J.T @ (f - y) / n
        else:
            # The gradient of (1/alpha^2) (1/(2n)) |alpha (f - f_0) - y|^2 is J^T (alpha (f - f_0) - y) / (n alpha).
            th -= eta * J.T @ (alpha * (f - f_init) - y) / (n * alpha)
    f, _ = parts(th[: m * d].reshape(m, d), th[m * d :])
    out = f if alpha is None else alpha * (f - f_init)
    return np.linalg.norm(th - th0), np.linalg.norm(th0), np.abs(out - y).max()


widths = [2**k for k in range(6, 14)]
rel, absd = [], []
for m in widths:
    v = [train(m, 10 + s) for s in range(5)]
    rel.append(np.mean([dt / t0 for dt, t0, _ in v]))
    absd.append(np.mean([dt for dt, _, _ in v]))
    print(
        f"m={m:<5} |dtheta|={absd[-1]:.4f} relative change={rel[-1]:.5f} sqrt(m)*relative={np.sqrt(m) * rel[-1]:.4f} "
        f"max train residual={max(e for _, _, e in v):.1e}"
    )
slope = np.polyfit(np.log(widths), np.log(rel), 1)[0]
print(f"fitted slope of log relative change against log m: {slope:.4f} (the derivation gives -1/2)")
assert abs(slope + 0.5) < 0.08

# The bound on the linearized model, at every step of gradient descent with eta <= n / lambda_max.
m = 512
W, a = rng.standard_normal((m, d)), rng.standard_normal(m)
f0, J = parts(W, a)
K = J @ J.T
lam = np.linalg.eigvalsh(K)
r0 = f0 - y
bound = np.linalg.norm(r0) / np.sqrt(lam[0])
eta = n / lam[-1]
omega = np.zeros(J.shape[1])
worst = 0.0
for _ in range(2000):
    omega -= eta * J.T @ (f0 + J @ omega - y) / n
    worst = max(worst, np.linalg.norm(omega))
exact = np.sqrt(r0 @ np.linalg.solve(K, r0))
print(
    f"linearized, m={m}: max_k |omega_k|={worst:.4f}, sqrt(r0' K^-1 r0)={exact:.4f}, |r0|/sqrt(lambda_min)={bound:.4f}"
)
assert worst <= exact + 1e-9 and exact <= bound + 1e-12

# Scale, not width: at m = 256, the centred model alpha (f - f_0) moves its weights by about 1/alpha.
alphas = [4.0, 16.0, 64.0]
dis = [np.mean([train(256, 50 + s, alpha=al)[0] for s in range(3)]) for al in alphas]
for al, dv in zip(alphas, dis):
    print(f"alpha={al:<5} |dtheta|={dv:.5f} alpha*|dtheta|={al * dv:.4f}")
s_alpha = np.polyfit(np.log(alphas), np.log(dis), 1)[0]
print(f"fitted slope of log |dtheta| against log alpha: {s_alpha:.4f} (Chizat et al. give O(1/alpha))")
assert abs(s_alpha + 1.0) < 0.1
print("ok")
