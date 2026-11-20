# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/normalizing-flow/ (ai://NEW-normalizing-flow).
"""The normalizing flow of the page in NumPy: a stack of affine coupling layers (eq-coupling) with alternating masks
on vectors of d_data dimensions, a standard normal base density, and the exact log-likelihood of eq-change-of-variables.
Every backward pass is written by hand. Also the planar flow of Rezende and Mohamed (2015) with the invertibility fix
of eq-planar, and one masked autoregressive layer (eq-autoregressive) whose density needs one pass and whose sampling
needs d_data passes.

The direction x -> z is f, the one that evaluates densities; sampling runs f^-1 from z ~ N(0, I). A coupling layer's
scale and shift come from one tanh hidden layer of the kept coordinates, so the scale is bounded, and they are masked
to 0 on the kept coordinates, so y = x * exp(s) + t leaves those unchanged.
"""

import math

import numpy as np

PARTS = ("W1", "b1", "Ws", "bs", "Wt", "bt")


def config(d_data=3, d_hidden=5, n_layers=3):
    if min(d_data, d_hidden, n_layers) < 1 or d_data < 2:
        raise ValueError("d_data must be at least 2 and every size positive")
    return {"d_data": int(d_data), "d_hidden": int(d_hidden), "n_layers": int(n_layers)}


def masks(cfg):
    """(n_layers, D): 1 on the coordinates a layer keeps, 0 on those it transforms; layer k keeps the indices of
    parity k, so consecutive layers transform complementary halves."""
    i = np.arange(cfg["d_data"])
    return np.stack([((i + k) % 2 == 0).astype(float) for k in range(cfg["n_layers"])])


def param_shapes(cfg):
    D, H = cfg["d_data"], cfg["d_hidden"]
    one = {"W1": (D, H), "b1": (H,), "Ws": (H, D), "bs": (D,), "Wt": (H, D), "bt": (D,)}
    return {f"c{k}_{n}": s for k in range(cfg["n_layers"]) for n, s in one.items()}


def param_count(cfg):
    return sum(math.prod(s) for s in param_shapes(cfg).values())


def flops_forward(cfg, batch):
    """eq-flops: the three products of each coupling layer, a multiply-add as 2."""
    return 6 * batch * cfg["n_layers"] * cfg["d_data"] * cfg["d_hidden"]


def init(rng, cfg):
    """W1 random, everything else 0, so that every layer starts as the identity (the zero init of Glow)."""
    p = {k: np.zeros(s) for k, s in param_shapes(cfg).items()}
    for k in range(cfg["n_layers"]):
        p[f"c{k}_W1"] = rng.standard_normal((cfg["d_data"], cfg["d_hidden"])) / math.sqrt(cfg["d_data"])
    return p


def _layer(p, k, b, x):
    q = {n: p[f"c{k}_{n}"] for n in PARTS}
    xm = x * b
    h = np.tanh(xm @ q["W1"] + q["b1"])
    s = (h @ q["Ws"] + q["bs"]) * (1 - b)
    t = (h @ q["Wt"] + q["bt"]) * (1 - b)
    return q, xm, h, s, t


def forward(p, cfg, x):
    """f: data x (B, D) -> z (B, D), the log-determinant (B,) of eq-logdet, and the caches of the layers."""
    ld, caches = np.zeros(len(x)), []
    for k, b in enumerate(masks(cfg)):
        q, xm, h, s, t = _layer(p, k, b, x)
        es = np.exp(s)
        caches.append((x, xm, h, es))
        x = x * es + t
        ld = ld + s.sum(1)
    return x, ld, caches


def inverse(p, cfg, z):
    """f^-1: z -> x, the layers in reverse, each x = (y - t) * exp(-s) with s and t from the kept coordinates."""
    for k, b in reversed(list(enumerate(masks(cfg)))):
        _, _, _, s, t = _layer(p, k, b, z)
        z = (z - t) * np.exp(-s)
    return z


def log_normal(z):
    return -0.5 * np.sum(z**2, -1) - 0.5 * z.shape[-1] * math.log(2 * math.pi)


def log_prob(p, cfg, x):
    """eq-change-of-variables: log p(x) = log N(f(x); 0, I) + log |det df/dx|, one value per row."""
    z, ld, _ = forward(p, cfg, x)
    return log_normal(z) + ld


def nll_and_grads(p, cfg, x):
    """The mean negative log-likelihood in nats, its gradients in the parameters and in x."""
    B = len(x)
    z, ld, caches = forward(p, cfg, x)
    loss = -np.mean(log_normal(z) + ld)
    g, gy, gld = {}, z / B, -1.0 / B
    for k, b in reversed(list(enumerate(masks(cfg)))):
        x, xm, h, es = caches[k]
        q = {n: p[f"c{k}_{n}"] for n in PARTS}
        gs = (gy * x * es + gld) * (1 - b)
        gt = gy * (1 - b)
        g[f"c{k}_Ws"], g[f"c{k}_bs"] = h.T @ gs, gs.sum(0)
        g[f"c{k}_Wt"], g[f"c{k}_bt"] = h.T @ gt, gt.sum(0)
        ga = (gs @ q["Ws"].T + gt @ q["Wt"].T) * (1 - h**2)
        g[f"c{k}_W1"], g[f"c{k}_b1"] = xm.T @ ga, ga.sum(0)
        gy = gy * es + (ga @ q["W1"].T) * b
    return float(loss), g, gy


def sample(p, cfg, n, rng, temperature=1.0):
    """x = f^-1(z) with z ~ N(0, T^2 I); T < 1 is the reduced temperature of Glow."""
    return inverse(p, cfg, temperature * rng.standard_normal((n, cfg["d_data"])))


def dequantize(pixels, rng, levels=256):
    """Integer pixels in [0, levels) -> continuous values in [0, 1): (pixel + u) / levels, u ~ U[0, 1)."""
    return (pixels + rng.random(np.shape(pixels))) / levels


def bits_per_dim(nll_nats, d, levels=256):
    """eq-bpd: the negative log-likelihood of data scaled by 1/levels, in bits per dimension of the integer data."""
    return (nll_nats + d * math.log(levels)) / (d * math.log(2))


def u_hat(w, u):
    """eq-planar: u with its component along w changed so that w.u >= -1, which makes the tanh planar map invertible."""
    wu = w @ u
    return u + (np.logaddexp(0.0, wu) - 1.0 - wu) * w / (w @ w)


def planar(z, w, u, b):
    """The planar flow f(z) = z + u tanh(w.z + b) and log |det df/dz| = log |1 + u.psi(z)|, psi = tanh'(w.z + b) w."""
    a = np.tanh(z @ w + b)
    return z + np.outer(a, u), np.log(np.abs(1.0 + (1.0 - a**2) * (w @ u)))


def ar_mask(d):
    """M[i, j] = 1 when input i may feed output j, i < j: output j sees x_1..x_{j-1} only."""
    return np.triu(np.ones((d, d)), 1)


def ar_params(x, q):
    m = ar_mask(x.shape[-1])
    return x @ (q["Wm"] * m) + q["bm"], x @ (q["Wa"] * m) + q["ba"]


def ar_density(q, x):
    """eq-autoregressive, the density direction: u = (x - mu(x)) exp(-alpha(x)) in one pass, log det = -sum alpha."""
    mu, al = ar_params(x, q)
    return (x - mu) * np.exp(-al), -al.sum(-1)


def ar_sample(q, u, passes=None):
    """The sampling direction: x_i = u_i exp(alpha_i) + mu_i from x_1..x_{i-1}, one pass per coordinate."""
    x = np.zeros_like(u)
    for _ in range(u.shape[-1] if passes is None else passes):
        mu, al = ar_params(x, q)
        x = u * np.exp(al) + mu
    return x


def adam(p, g, st, lr, b1=0.9, b2=0.999, eps=1e-8):
    st["t"] = st.get("t", 0) + 1
    for k in g:
        st[k + "m"] = b1 * st.get(k + "m", 0.0) + (1 - b1) * g[k]
        st[k + "v"] = b2 * st.get(k + "v", 0.0) + (1 - b2) * g[k] ** 2
        mh, vh = st[k + "m"] / (1 - b1 ** st["t"]), st[k + "v"] / (1 - b2 ** st["t"])
        p[k] = p[k] - lr * mh / (np.sqrt(vh) + eps)


def train(p, cfg, sample_data, steps, rng, batch=256, lr=3e-3):
    """Maximum likelihood with Adam on minibatches; returns the losses."""
    st, trace = {}, []
    for _ in range(steps):
        loss, g, _ = nll_and_grads(p, cfg, sample_data(batch, rng))
        adam(p, g, st, lr)
        trace.append(loss)
    return trace
