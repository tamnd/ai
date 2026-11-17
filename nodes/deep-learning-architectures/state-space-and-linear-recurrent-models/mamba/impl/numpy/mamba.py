# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/mamba/ (ai://NEW-mamba).
"""Mamba of Gu and Dao (2023), the NumPy reference: a language model of Mamba blocks, forward, backward and counts.

Tokens are embedded to D numbers and pass through L residual blocks, r <- r + mixer(rmsnorm(r)), then a final
RMSNorm and the transposed embedding give the next-token scores (the head is tied to the embedding). The mixer is
the Mamba block (eq-block): an input projection to two branches of width I = E D, x and z; a causal depthwise
convolution of width K and SiLU on x; the selective SSM (eq-selection, eq-scan) on x, whose step Delta and matrices
B and C are computed from x at every position; the skip D x; the gate SiLU(z); and an output projection back to D.
The state matrix is diagonal, A = -exp(A_log), one row of N entries per channel. disc="simple" discretizes as the
reference code of the authors does, A_bar = exp(Delta A) and B_bar = Delta B; disc="zoh" uses zero-order hold for
both, B_bar = (exp(Delta A) - 1) / A * B, as in eq. 4 of the paper. Everything is float64.
"""

import math

import numpy as np

DISCS = ("simple", "zoh")


def config(vocab=11, d_model=8, n_layers=2, d_state=4, d_conv=3, expand=2, dt_rank=None, disc="simple", eps=1e-5):
    """Return the config; the defaults are the example config of shapes.yaml. dt_rank=None gives ceil(D / 16)."""
    if min(vocab, d_model, n_layers, d_state, d_conv, expand) < 1:
        raise ValueError("the sizes must be at least 1")
    if disc not in DISCS:
        raise ValueError(f"disc must be one of {DISCS}")
    R = math.ceil(d_model / 16) if dt_rank is None else dt_rank
    return {"vocab": vocab, "d_model": d_model, "n_layers": n_layers, "d_state": d_state, "d_conv": d_conv,
            "expand": expand, "d_inner": expand * d_model, "dt_rank": R, "disc": disc, "eps": eps}  # fmt: skip


def block_shapes(cfg):
    D, I, N, K, R = cfg["d_model"], cfg["d_inner"], cfg["d_state"], cfg["d_conv"], cfg["dt_rank"]
    return {"norm": (D,), "in_proj": (D, 2 * I), "conv_w": (I, K), "conv_b": (I,), "x_proj": (I, R + 2 * N),
            "dt_w": (R, I), "dt_b": (I,), "A_log": (I, N), "D": (I,), "out_proj": (I, D)}  # fmt: skip


def param_shapes(cfg):
    shapes = {"embed": (cfg["vocab"], cfg["d_model"]), "norm_f": (cfg["d_model"],)}
    for l in range(cfg["n_layers"]):
        shapes |= {f"l{l}_{k}": s for k, s in block_shapes(cfg).items()}
    return shapes


def param_count(cfg):
    """eq-params, from the formula rather than the shapes, so that the two can be checked against each other."""
    D, I, N, K, R, V = (cfg[k] for k in ("d_model", "d_inner", "d_state", "d_conv", "dt_rank", "vocab"))
    block = 3 * I * D + I * (K + 1) + I * (R + 2 * N) + (R + 1) * I + I * N + I + D
    return V * D + D + cfg["n_layers"] * block


def flops(cfg, B, T):
    """eq-flops: the forward FLOPs on B sequences of T tokens, a multiply-add as 2: the matrix products, the
    convolution, the two multiply-adds of the scan per state entry (the update and the read-out by C) and the head."""
    D, I, N, K, R, V = (cfg[k] for k in ("d_model", "d_inner", "d_state", "d_conv", "dt_rank", "vocab"))
    block = 2 * D * 2 * I + 2 * I * K + 2 * I * (R + 2 * N) + 2 * R * I + 4 * I * N + 2 * I * D
    return B * T * (cfg["n_layers"] * block + 2 * D * V)


def init(rng, cfg, scale=0.1, dt_min=1e-3, dt_max=1e-1):
    """Random weights N(0, scale^2); A_log = log(1..N) (S4D-Real, A_n = -(n + 1)); D = 1; and the bias of Delta set
    so that softplus(dt_b) is log-uniform in [dt_min, dt_max], as in the reference code."""
    p = {k: rng.normal(0.0, scale, s) for k, s in param_shapes(cfg).items()}
    I, N = cfg["d_inner"], cfg["d_state"]
    p["norm_f"] = np.ones(cfg["d_model"])
    for l in range(cfg["n_layers"]):
        dt = np.exp(rng.uniform(np.log(dt_min), np.log(dt_max), I))
        p[f"l{l}_dt_b"] = dt + np.log(-np.expm1(-dt))  # the inverse of softplus
        p[f"l{l}_A_log"] = np.log(np.tile(np.arange(1.0, N + 1), (I, 1)))
        p[f"l{l}_D"] = np.ones(I)
        p[f"l{l}_norm"] = np.ones(cfg["d_model"])
    return p


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def softplus(x):
    return np.logaddexp(0.0, x)


def silu(x):
    return x * sigmoid(x)


def silu_grad(x):
    s = sigmoid(x)
    return s * (1.0 + x * (1.0 - s))


def rmsnorm(x, w, eps):
    r = 1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + eps)
    return x * r * w, r


def rmsnorm_backward(dy, x, w, r):
    xh = x * r
    g = dy * w
    return r * (g - xh * (g * xh).mean(-1, keepdims=True)), (dy * xh).reshape(-1, x.shape[-1]).sum(0)


def causal_conv(x, w, b):
    """Depthwise causal convolution over the positions of (B, T, I): out[t] = b + sum_k w[:, k] x[t - K + 1 + k]."""
    K, T = w.shape[1], x.shape[1]
    xp = np.concatenate([np.zeros((x.shape[0], K - 1, x.shape[2])), x], 1)
    return b + sum(w[:, k] * xp[:, k : k + T] for k in range(K))


def discretize(dt, A, Bm, disc):
    """eq-disc: A_bar (B, T, I, N) and B_bar (B, T, I, N) from Delta (B, T, I), A (I, N) and B (B, T, N)."""
    dA = np.exp(dt[..., None] * A)
    if disc == "simple":
        return dA, dt[..., None] * Bm[:, :, None, :]
    return dA, (dA - 1.0) / A * Bm[:, :, None, :]


def selective_scan(dA, dB, x, C, h0=None):
    """eq-scan: h_t = A_bar_t h_{t-1} + B_bar_t x_t and y_t = C_t h_t, sequentially; returns y and all the states."""
    Bsz, T, I, N = dA.shape
    h = np.zeros((Bsz, I, N)) if h0 is None else h0
    hs = np.empty_like(dA)
    for t in range(T):
        h = dA[:, t] * h + dB[:, t] * x[:, t, :, None]
        hs[:, t] = h
    return np.einsum("btin,btn->bti", hs, C), hs


def mixer(p, cfg, u, pre=""):
    """The Mamba block on (B, T, D) inputs; returns its output and a cache for the backward pass."""
    I, N, R = cfg["d_inner"], cfg["d_state"], cfg["dt_rank"]
    xz = u @ p[pre + "in_proj"]
    x, z = xz[..., :I], xz[..., I:]
    xc = causal_conv(x, p[pre + "conv_w"], p[pre + "conv_b"])
    xa = silu(xc)
    dbl = xa @ p[pre + "x_proj"]
    dtl, Bm, C = dbl[..., :R], dbl[..., R : R + N], dbl[..., R + N :]
    dtp = dtl @ p[pre + "dt_w"] + p[pre + "dt_b"]
    dt = softplus(dtp)
    A = -np.exp(p[pre + "A_log"])
    dA, dB = discretize(dt, A, Bm, cfg["disc"])
    ys, hs = selective_scan(dA, dB, xa, C)
    y = ys + p[pre + "D"] * xa
    g = silu(z)
    out = (y * g) @ p[pre + "out_proj"]
    cache = dict(u=u, x=x, z=z, xc=xc, xa=xa, dtl=dtl, Bm=Bm, C=C, dtp=dtp, dt=dt, A=A, dA=dA, dB=dB, hs=hs, y=y, g=g)
    return out, cache


def mixer_backward(dout, p, cfg, c, pre=""):
    """The gradients of the block with respect to its parameters and its input u."""
    I, N, R, K = cfg["d_inner"], cfg["d_state"], cfg["dt_rank"], cfg["d_conv"]
    grads = {"out_proj": np.einsum("bti,btd->id", c["y"] * c["g"], dout)}
    dyg = dout @ p[pre + "out_proj"].T
    dy, dz = dyg * c["g"], dyg * c["y"] * silu_grad(c["z"])
    grads["D"] = (dy * c["xa"]).sum((0, 1))
    dxa = dy * p[pre + "D"]
    hs, dA, dB, xa, C = c["hs"], c["dA"], c["dB"], c["xa"], c["C"]
    T = xa.shape[1]
    ddA, ddB, dC = np.zeros_like(dA), np.zeros_like(dB), np.zeros_like(C)
    gh = np.zeros_like(hs[:, 0])
    for t in reversed(range(T)):  # the scan run backwards: gh is dLoss/dh_t
        gh = gh + dy[:, t, :, None] * C[:, t, None, :]
        dC[:, t] = np.einsum("bi,bin->bn", dy[:, t], hs[:, t])
        ddA[:, t] = gh * (hs[:, t - 1] if t > 0 else 0.0)
        ddB[:, t] = gh * xa[:, t, :, None]
        dxa[:, t] += (gh * dB[:, t]).sum(-1)
        gh = gh * dA[:, t]
    dt, A, Bm = c["dt"], c["A"], c["Bm"]
    ddt = (ddA * dA * A).sum(-1)
    dAm = (ddA * dA * dt[..., None]).sum((0, 1))
    if cfg["disc"] == "simple":
        ddt += (ddB * Bm[:, :, None, :]).sum(-1)
        dBm = (ddB * dt[..., None]).sum(2)
    else:
        q = (dA - 1.0) / A
        ddt += (ddB * dA * Bm[:, :, None, :]).sum(-1)
        dAm += (ddB * Bm[:, :, None, :] * (dt[..., None] * dA * A - (dA - 1.0)) / A**2).sum((0, 1))
        dBm = (ddB * q).sum(2)
    grads["A_log"] = dAm * A
    ddtp = ddt * sigmoid(c["dtp"])
    grads["dt_w"] = np.einsum("btr,bti->ri", c["dtl"], ddtp)
    grads["dt_b"] = ddtp.sum((0, 1))
    ddbl = np.concatenate([ddtp @ p[pre + "dt_w"].T, dBm, dC], -1)
    grads["x_proj"] = np.einsum("bti,btk->ik", xa, ddbl)
    dxa = dxa + ddbl @ p[pre + "x_proj"].T
    dxc = dxa * silu_grad(c["xc"])
    grads["conv_b"] = dxc.sum((0, 1))
    xp = np.concatenate([np.zeros((xa.shape[0], K - 1, I)), c["x"]], 1)
    grads["conv_w"] = np.stack([(dxc * xp[:, k : k + T]).sum((0, 1)) for k in range(K)], 1)
    dxp = np.zeros_like(xp)
    for k in range(K):
        dxp[:, k : k + T] += dxc * p[pre + "conv_w"][:, k]
    dxz = np.concatenate([dxp[:, K - 1 :], dz], -1)
    grads["in_proj"] = np.einsum("btd,btj->dj", c["u"], dxz)
    return grads, dxz @ p[pre + "in_proj"].T


def forward(p, cfg, tokens):
    """The (B, T, V) next-token scores of a (B, T) array of token ids, and a cache for the backward pass."""
    r = p["embed"][tokens]
    layers = []
    for l in range(cfg["n_layers"]):
        u, s = rmsnorm(r, p[f"l{l}_norm"], cfg["eps"])
        out, c = mixer(p, cfg, u, f"l{l}_")
        layers.append({"r": r, "s": s, "mixer": c})
        r = r + out
    hf, sf = rmsnorm(r, p["norm_f"], cfg["eps"])
    return hf @ p["embed"].T, {"tokens": tokens, "layers": layers, "r": r, "sf": sf, "hf": hf}


def nll(logits, targets):
    """eq-loss: the mean cross-entropy of the next tokens."""
    m = logits.max(-1, keepdims=True)
    logp = logits - m - np.log(np.exp(logits - m).sum(-1, keepdims=True))
    return float(-np.take_along_axis(logp, targets[..., None], -1).mean())


def lm_loss(p, cfg, tokens):
    """The loss of predicting tokens[:, 1:] from tokens[:, :-1]."""
    return nll(forward(p, cfg, tokens[:, :-1])[0], tokens[:, 1:])


def backward(p, cfg, cache, logits, targets):
    """The gradients of eq-loss with respect to every parameter."""
    e = np.exp(logits - logits.max(-1, keepdims=True))
    dl = e / e.sum(-1, keepdims=True)
    np.put_along_axis(dl, targets[..., None], np.take_along_axis(dl, targets[..., None], -1) - 1.0, -1)
    dl /= targets.size
    V, D = p["embed"].shape
    grads = {"embed": dl.reshape(-1, V).T @ cache["hf"].reshape(-1, D)}
    dr, grads["norm_f"] = rmsnorm_backward(dl @ p["embed"], cache["r"], p["norm_f"], cache["sf"])
    for l in reversed(range(cfg["n_layers"])):
        c = cache["layers"][l]
        g, du = mixer_backward(dr, p, cfg, c["mixer"], f"l{l}_")
        grads |= {f"l{l}_{k}": v for k, v in g.items()}
        dr_n, grads[f"l{l}_norm"] = rmsnorm_backward(du, c["r"], p[f"l{l}_norm"], c["s"])
        dr = dr + dr_n
    np.add.at(grads["embed"], cache["tokens"], dr)
    return grads


def init_state(cfg, B):
    """The recurrent state of each layer: the last K - 1 inputs of the convolution and the (I, N) SSM state."""
    I = cfg["d_inner"]
    return [(np.zeros((B, cfg["d_conv"] - 1, I)), np.zeros((B, I, cfg["d_state"]))) for _ in range(cfg["n_layers"])]


def step(p, cfg, token, state):
    """One token of generation: (B,) ids and the state give the (B, V) scores and the next state, in O(1) time
    and memory in the length; running it over a sequence gives the scores of forward."""
    I, N, R, K = cfg["d_inner"], cfg["d_state"], cfg["dt_rank"], cfg["d_conv"]
    r, new = p["embed"][token], []
    for l, (cs, h) in enumerate(state):
        q = f"l{l}_"
        u, _ = rmsnorm(r, p[q + "norm"], cfg["eps"])
        xz = u @ p[q + "in_proj"]
        x, z = xz[:, :I], xz[:, I:]
        win = np.concatenate([cs, x[:, None]], 1)
        xa = silu(p[q + "conv_b"] + np.einsum("bki,ik->bi", win, p[q + "conv_w"]))
        dbl = xa @ p[q + "x_proj"]
        dt = softplus(dbl[:, :R] @ p[q + "dt_w"] + p[q + "dt_b"])
        dA, dB = discretize(dt[:, None], -np.exp(p[q + "A_log"]), dbl[:, None, R : R + N], cfg["disc"])
        h = dA[:, 0] * h + dB[:, 0] * xa[..., None]
        y = np.einsum("bin,bn->bi", h, dbl[:, R + N :]) + p[q + "D"] * xa
        r = r + (y * silu(z)) @ p[q + "out_proj"]
        new.append((win[:, 1:] if K > 1 else cs, h))
    hf, _ = rmsnorm(r, p["norm_f"], cfg["eps"])
    return hf @ p["embed"].T, new
