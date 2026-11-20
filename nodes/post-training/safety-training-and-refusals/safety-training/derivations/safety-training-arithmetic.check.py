# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/safety-training/ (ai://NEW-safety-training).
"""Checks every step of safety-training-arithmetic.md numerically."""

import base64
import math

import numpy as np

rng = np.random.default_rng(0)

# 1. Safety data fraction.
N = 20_000
for n, pct in [(100, 0.50), (500, 2.44), (1000, 4.76), (2000, 9.09)]:
    assert round(100 * n / (N + n), 2) == pct, n
n3 = 0.03 * N / 0.97
assert abs(n3 - 618.56) < 0.01 and round(n3) == 619
assert abs(619 / (N + 619) - 0.03) < 1e-3

# 2. Cross-entropy = KL + teacher entropy; minimized at the teacher.
for _ in range(100):
    p = rng.dirichlet(np.ones(6))
    q = rng.dirichlet(np.ones(6))
    ce = -np.sum(p * np.log(q))
    kl = np.sum(p * np.log(p / q))
    ent = -np.sum(p * np.log(p))
    assert abs(ce - (kl + ent)) < 1e-12 and kl >= 0
    assert -np.sum(p * np.log(p)) <= ce + 1e-12  # q = p gives the smallest cross-entropy

# 3. Projection properties and the difference of means.
d = 7
for _ in range(50):
    r = rng.normal(size=d)
    rh = r / np.linalg.norm(r)
    P = np.eye(d) - np.outer(rh, rh)
    x = rng.normal(size=d)
    assert np.allclose(P, P.T) and np.allclose(P @ P, P)
    assert abs(rh @ (P @ x)) < 1e-12
    y = P @ rng.normal(size=d)
    assert np.allclose(P @ y, y)
    assert abs(x @ x - ((P @ x) @ (P @ x) + (rh @ x) ** 2)) < 1e-10
    assert np.linalg.norm(P @ x) <= np.linalg.norm(x) + 1e-12
H = rng.normal(loc=1.0, size=(40, d))
B = rng.normal(size=(30, d))
mu, nu = H.mean(0), B.mean(0)
r = mu - nu
rh = r / np.linalg.norm(r)
assert abs(rh @ r - np.linalg.norm(r)) < 1e-12
for _ in range(1000):
    u = rng.normal(size=d)
    u /= np.linalg.norm(u)
    assert u @ r <= np.linalg.norm(r) + 1e-12
assert np.allclose((B + r).mean(0), mu)

# 4. Weight orthogonalization equals ablation of every residual activation.
V, T, h = 11, 5, 9
E, Pos = rng.normal(size=(V, d)), rng.normal(size=(T, d))
layers = [(rng.normal(size=(d, h)), rng.normal(size=h), rng.normal(size=(h, d)), rng.normal(size=d)) for _ in range(3)]
U = rng.normal(size=(d, V))


def ln(z):
    return (z - z.mean(-1, keepdims=True)) / np.sqrt(z.var(-1, keepdims=True) + 1e-5)


def forward(E, Pos, layers, proj=None):
    toks = np.array([3, 1, 4, 1, 5])
    x = E[toks] + Pos
    f = (lambda z: z @ proj.T) if proj is not None else (lambda z: z)
    x = f(x)
    for W_in, b_in, W_out, b_out in layers:
        hid = np.maximum(ln(x) @ W_in + b_in, 0)
        x = f(x + hid @ W_out + b_out)
    return ln(x) @ U, x


P = np.eye(d) - np.outer(rh, rh)
logits_a, xa = forward(E, Pos, layers, proj=P)
layers_b = [(W_in, b_in, W_out @ P, b_out @ P) for W_in, b_in, W_out, b_out in layers]  # rows write to the stream
logits_b, xb = forward(E @ P, Pos @ P, layers_b)
assert np.allclose(logits_a, logits_b) and np.allclose(xa, xb)
assert np.max(np.abs(xb @ rh)) < 1e-10
logits_0, _ = forward(E, Pos, layers)
assert not np.allclose(logits_0, logits_b)

# 5. Base64 length.
q = "What tools do I need to cut down a stop sign?"
assert len(q.encode()) == 45
enc = base64.b64encode(q.encode()).decode()
assert enc == "V2hhdCB0b29scyBkbyBJIG5lZWQgdG8gY3V0IGRvd24gYSBzdG9wIHNpZ24/" and len(enc) == 60
for n in range(0, 100):
    assert len(base64.b64encode(bytes(n))) == 4 * math.ceil(n / 3)

# 6. Robustness gamma.
assert abs(60 / 1.8 - 33.33) < 0.01 and abs(60 / 0.45 - 133.33) < 0.01 and 1.8 / 0.45 == 4

# 7. Counts.
assert 0.38 * 250 == 95 and abs(0.216 * 250 - 54) < 1e-9 and 95 + 54 == 149 and 149 / 250 == 0.596
assert 1351 + 623 == 1974

# 8. GCG: first-order estimate and candidate counts.
Vt, dt = 13, 4
Et = rng.normal(size=(Vt, dt))
w = rng.normal(size=dt)


def loss(onehot):
    return float(np.log1p(np.exp(-(onehot @ Et) @ w)))


x0 = 2
e = np.eye(Vt)[x0]
s = (e @ Et) @ w
g = Et @ w * (-1 / (1 + np.exp(s)))  # gradient of log(1 + exp(-s)) with respect to the one-hot vector
num = np.array([(loss(e + 1e-6 * np.eye(Vt)[v]) - loss(e - 1e-6 * np.eye(Vt)[v])) / 2e-6 for v in range(Vt)])
assert np.allclose(g, num, atol=1e-6)
est = loss(e) + g - g[x0]
assert list(np.argsort(g)[:3]) == list(np.argsort(est)[:3])  # Top-k(-g): the k smallest g
assert 20 * (32_000 - 1) == 639_980 and 256 * 20 == 5_120 and 512 / 5_120 == 0.1
assert abs(512 / 639_980 - 0.0008) < 1e-5

print("safety-training-arithmetic: ok")
