# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/sharpness-aware-minimization/ (ai://8BNWST).
# Check of derivations/sam-quadratic.md: on L(w) = w.Hw / 2 the SAM gradient is H(w + rho Hw / ||Hw||), which is the
# gradient of L + rho ||Hw||; in the eigenbasis it scales the i-th component of the gradient by 1 + rho lambda_i / ||Hw||,
# so the sharp components shrink faster than under plain gradient descent; with H positive definite it is never zero,
# and in one dimension a fixed step ends in the 2-cycle +-eta lambda rho / (2 - eta lambda); on the valley
# (ab - 1)^2 / 2, whose minima have Hessian eigenvalues 0 and a^2 + b^2, SAM moves toward the flattest minimum a = b.
import numpy as np

rng = np.random.default_rng(1)


def sam_grad(H, w, rho):
    g = H @ w
    return H @ (w + rho * g / np.linalg.norm(g))


# (i) closed form, and equality with the gradient of L + rho ||Hw|| by central differences
for _ in range(200):
    d = int(rng.integers(2, 6))
    Q, _r = np.linalg.qr(rng.normal(size=(d, d)))
    lam = rng.uniform(0.05, 5.0, size=d)
    H = Q @ np.diag(lam) @ Q.T
    w = rng.normal(size=d)
    rho = float(rng.uniform(0.01, 1.0))
    s = sam_grad(H, w, rho)
    assert np.allclose(s, H @ w + rho * H @ H @ w / np.linalg.norm(H @ w))

    def F(v):
        return 0.5 * v @ H @ v + rho * np.linalg.norm(H @ v)

    h = 1e-6
    fd = np.array([(F(w + h * e) - F(w - h * e)) / (2 * h) for e in np.eye(d)])
    assert np.allclose(fd, s, rtol=1e-5, atol=1e-6)
    # (ii) eigen-components: (Q^T s)_i = lambda_i c_i (1 + rho lambda_i / ||Hw||) with c = Q^T w
    c = Q.T @ w
    assert np.allclose(Q.T @ s, lam * c * (1 + rho * lam / np.linalg.norm(H @ w)))
    # (iii) never zero for positive definite H
    assert np.linalg.norm(s) > 0
    # and at least as long as the plain gradient along every eigendirection
    assert np.all(np.abs(Q.T @ s) >= np.abs(lam * c) - 1e-12)

# (ii) the sharp component shrinks faster: H = diag(10, 0.1), the same small step for GD and SAM (eta lambda_max = 0.1)
H = np.diag([10.0, 0.1])
eta, rho = 0.01, 0.05
w_gd = w_sam = np.array([1.0, 1.0])
for t in range(300):
    w_gd = w_gd - eta * (H @ w_gd)
    w_sam = w_sam - eta * sam_grad(H, w_sam, rho)
    assert abs(w_sam[0] / w_sam[1]) < abs(w_gd[0] / w_gd[1])
# the direction of the iterate turns toward the flat eigenvector e_2 faster under SAM
assert abs(w_sam[0]) / np.linalg.norm(w_sam) < abs(w_gd[0]) / np.linalg.norm(w_gd)

# (iv) one dimension: w <- w - eta lam (w + rho sign w) ends in a 2-cycle of amplitude eta lam rho / (2 - eta lam)
for eta_lam in [0.1, 0.5, 0.9, 1.5]:
    rho = 0.2
    w = 3.0
    for _ in range(4000):
        w = w - eta_lam * (w + rho * np.sign(w))
    amp = eta_lam * rho / (2 - eta_lam)
    assert abs(abs(w) - amp) < 1e-9, (eta_lam, w, amp)
    w2 = w - eta_lam * (w + rho * np.sign(w))
    assert abs(w2 + w) < 1e-9  # it alternates sign


# (v) the valley L(a, b) = (ab - 1)^2 / 2
def vgrad(x):
    a, b = x
    r = a * b - 1
    return r * np.array([b, a])


def vhess(x):
    a, b = x
    r = a * b - 1
    return np.outer([b, a], [b, a]) + r * np.array([[0.0, 1.0], [1.0, 0.0]])


for a in [0.25, 0.5, 1.0, 2.0, 4.0]:
    ev = np.linalg.eigvalsh(vhess(np.array([a, 1 / a])))
    assert abs(ev[0]) < 1e-12 and abs(ev[1] - (a * a + 1 / (a * a))) < 1e-9
    assert ev[1] >= 2 - 1e-12

x_gd = x_sam = np.array([3.0, 0.4])
eta, rho = 0.02, 0.05
for _ in range(20000):
    x_gd = x_gd - eta * vgrad(x_gd)
    g = vgrad(x_sam)
    x_sam = x_sam - eta * vgrad(x_sam + rho * g / (np.linalg.norm(g) + 1e-12))
top_gd = np.linalg.eigvalsh(vhess(x_gd))[-1]
top_sam = np.linalg.eigvalsh(vhess(x_sam))[-1]
assert abs(x_gd[0] * x_gd[1] - 1) < 1e-6 and top_gd > 8.5  # GD stops at the first minimum it reaches
assert abs(x_sam[0] * x_sam[1] - 1) < 0.05 and top_sam < 2.5, (x_sam, top_sam)  # SAM drifts to a = b
print("ok")
