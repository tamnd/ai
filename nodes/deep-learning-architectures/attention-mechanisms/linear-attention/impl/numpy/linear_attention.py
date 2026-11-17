# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/linear-attention/ (ai://NEW-linear-attention).
"""Linear attention of Katharopoulos, Vyas, Pappas and Fleuret (2020), forward and backward, the NumPy reference.

The single-head functions take arrays whose last two axes are (N, D) for queries and keys and (N, M) for values, with
any leading batch axes. kernel_attention is eq-general on the page, quadratic in N; linear_attention is eq-linear, the
same numbers by associativity; causal_linear_attention is eq-causal, run as the recurrence eq-rnn with the states
S_i and z_i, and causal_numerator_backward is Algorithm 1 of the paper, the gradient of the numerator in constant
memory. The layer (init, forward, backward) is h heads of it between projections W_q, W_k, W_v and W_o, with no biases;
backward is checked against finite differences in the tests. step is one token of the layer as an RNN. dpfp and
fast_weight_rule are two variants of Schlag, Irie and Schmidhuber (2021): a feature map that enlarges the state, and
the delta rule, which overwrites what the state holds under a key instead of adding to it.
"""

import numpy as np


def config(d_model=8, n_heads=2, d_k=None, d_v=None, feature="elu", causal=True):
    d_k = d_model // n_heads if d_k is None else int(d_k)
    d_v = d_model // n_heads if d_v is None else int(d_v)
    if feature not in FEATURES:
        raise ValueError(f"unknown feature map {feature!r}")
    return {"d_model": int(d_model), "n_heads": int(n_heads), "d_k": d_k, "d_v": d_v, "feature": feature,
            "causal": bool(causal)}  # fmt: skip


def init(rng, cfg):
    F, H, D, M = cfg["d_model"], cfg["n_heads"], cfg["d_k"], cfg["d_v"]
    return {
        "W_q": rng.normal(0.0, F**-0.5, (F, H * D)),
        "W_k": rng.normal(0.0, F**-0.5, (F, H * D)),
        "W_v": rng.normal(0.0, F**-0.5, (F, H * M)),
        "W_o": rng.normal(0.0, (H * M) ** -0.5, (H * M, F)),
    }


# Feature maps: phi(x) applied to the last axis, and the derivative needed by backward (None if not elementwise).


def elu_plus_one(x):
    """eq-elu: elu(x) + 1, which is x + 1 for x > 0 and exp(x) otherwise; always positive."""
    return np.where(x > 0, x + 1.0, np.exp(np.minimum(x, 0.0)))


def elu_plus_one_grad(x):
    return np.where(x > 0, 1.0, np.exp(np.minimum(x, 0.0)))


def relu(x):
    return np.maximum(x, 0.0)


def relu_grad(x):
    return (x > 0).astype(float)


def poly2(x, c=1.0):
    """The explicit feature map of the kernel (q . k + c)^2: (c, sqrt(2c) x, all x_a x_b), of size 1 + D + D^2."""
    outer = (x[..., :, None] * x[..., None, :]).reshape(*x.shape[:-1], -1)
    return np.concatenate([np.full(x.shape[:-1] + (1,), c), np.sqrt(2 * c) * x, outer], -1)


def dpfp(x, nu=1):
    """The deterministic parameter-free projection of Schlag, Irie and Schmidhuber (2021), their Eq. 37 as coded in
    their Appendix C: r = (relu(x), relu(-x)) of size 2D, times itself rolled by 1..nu places; size 2 D nu."""
    r = np.concatenate([relu(x), relu(-x)], -1)
    rolled = np.concatenate([np.roll(r, j, -1) for j in range(1, nu + 1)], -1)
    return np.concatenate([r] * nu, -1) * rolled


FEATURES = {"elu": (elu_plus_one, elu_plus_one_grad), "relu": (relu, relu_grad), "poly2": (poly2, None), "dpfp": (dpfp, None)}


# Single-head attention.


def softmax_attention(Q, K, V, causal=False):
    """eq-softmax: softmax(Q K^T / sqrt(D)) V, row by row, with the future masked if causal."""
    s = Q @ np.swapaxes(K, -1, -2) / np.sqrt(Q.shape[-1])
    if causal:
        s = np.where(np.tril(np.ones(s.shape[-2:], bool)), s, -np.inf)
    w = np.exp(s - s.max(-1, keepdims=True))
    return (w / w.sum(-1, keepdims=True)) @ V


def kernel_attention(Q, K, V, sim, causal=False):
    """eq-general: V'_i = sum_j sim(Q_i, K_j) V_j / sum_j sim(Q_i, K_j), from the (N, N) matrix of similarities."""
    A = sim(Q, K)
    if causal:
        A = A * np.tril(np.ones(A.shape[-2:]))
    return (A @ V) / A.sum(-1, keepdims=True)


def linear_attention(fQ, fK, V):
    """eq-linear: phi(Q) (phi(K)^T V) over phi(Q) sum_j phi(K_j); fQ and fK are the mapped queries and keys."""
    KV = np.swapaxes(fK, -1, -2) @ V  # (C, M), once for all the queries
    z = fK.sum(-2)  # (C,)
    return (fQ @ KV) / (fQ @ z[..., None])


def causal_linear_attention(fQ, fK, V):
    """eq-causal by the recurrence eq-rnn: S_i = S_{i-1} + phi(K_i) V_i^T, z_i = z_{i-1} + phi(K_i). Returns the
    output and the numerators and denominators, which are all backward needs besides the inputs."""
    N = fQ.shape[-2]
    S = np.zeros(fK.shape[:-2] + (fK.shape[-1], V.shape[-1]))
    z = np.zeros(fK.shape[:-2] + (fK.shape[-1],))
    num, den = np.zeros(fQ.shape[:-1] + (V.shape[-1],)), np.zeros(fQ.shape[:-1])
    for i in range(N):
        S = S + fK[..., i, :, None] * V[..., i, None, :]
        z = z + fK[..., i, :]
        num[..., i, :] = np.einsum("...c,...cm->...m", fQ[..., i, :], S)
        den[..., i] = np.einsum("...c,...c->...", fQ[..., i, :], z)
    return num / den[..., None], num, den


def causal_numerator_backward(fQ, fK, V, G):
    """Algorithm 1, backward: from G = dL/dnum, the gradients of the causal numerator with respect to phi(Q), phi(K)
    and V, with one (C, M) running sum forward and one backward and no S_i stored."""
    N = fQ.shape[-2]
    dQ, dK, dV = np.zeros_like(fQ), np.zeros_like(fK), np.zeros_like(V)
    S = np.zeros(fK.shape[:-2] + (fK.shape[-1], V.shape[-1]))
    for i in range(N):  # eq-grad-q: dL/dphi(Q_i) = G_i S_i^T
        S = S + fK[..., i, :, None] * V[..., i, None, :]
        dQ[..., i, :] = np.einsum("...m,...cm->...c", G[..., i, :], S)
    S = np.zeros_like(S)
    for i in reversed(range(N)):  # eq-grad-kv: R_i = sum_{j >= i} phi(Q_j) G_j^T
        S = S + fQ[..., i, :, None] * G[..., i, None, :]
        dV[..., i, :] = np.einsum("...cm,...c->...m", S, fK[..., i, :])
        dK[..., i, :] = np.einsum("...cm,...m->...c", S, V[..., i, :])
    return dQ, dK, dV


def sum_normalise(f):
    """Schlag et al.'s Eq. 29: divide a positive feature vector by the sum of its components."""
    return f / f.sum(-1, keepdims=True)


def fast_weight_rule(fQ, fK, V, beta=None):
    """The unnormalised recurrence of a fast weight memory S (C, M), read out as y_i = phi(Q_i)^T S_i.
    beta=None is the sum rule of linear attention, S_i = S_{i-1} + phi(K_i) V_i^T; with beta, a (..., N) array in
    [0, 1], it is the delta rule of Schlag et al.'s Eq. 20 to 25: retrieve vbar_i = S_{i-1}^T phi(K_i) and write
    S_i = S_{i-1} + beta_i phi(K_i) (V_i - vbar_i)^T, which replaces the value stored under phi(K_i)."""
    S = np.zeros(fK.shape[:-2] + (fK.shape[-1], V.shape[-1]))
    out = np.zeros(fQ.shape[:-1] + (V.shape[-1],))
    for i in range(fQ.shape[-2]):
        k = fK[..., i, :]
        if beta is None:
            S = S + k[..., :, None] * V[..., i, None, :]
        else:
            vbar = np.einsum("...c,...cm->...m", k, S)
            S = S + beta[..., i, None, None] * k[..., :, None] * (V[..., i, :] - vbar)[..., None, :]
        out[..., i, :] = np.einsum("...c,...cm->...m", fQ[..., i, :], S)
    return out, S


# The layer.


def split(x, H):
    """(B, N, H*D) -> (B, H, N, D)."""
    B, N, HD = x.shape
    return x.reshape(B, N, H, HD // H).transpose(0, 2, 1, 3)


def merge(x):
    """(B, H, N, D) -> (B, N, H*D)."""
    B, H, N, D = x.shape
    return x.transpose(0, 2, 1, 3).reshape(B, N, H * D)


def forward(params, cfg, x):
    """The attention layer on x of shape (B, N, F): h heads of linear attention, concatenated and projected by W_o."""
    H = cfg["n_heads"]
    phi = FEATURES[cfg["feature"]][0]
    q, k, v = (split(x @ params[w], H) for w in ("W_q", "W_k", "W_v"))
    fQ, fK = phi(q), phi(k)
    if cfg["causal"]:
        att, num, den = causal_linear_attention(fQ, fK, v)
    else:
        KV = np.swapaxes(fK, -1, -2) @ v
        num, den = fQ @ KV, fQ @ fK.sum(-2)[..., None]
        den = den[..., 0]
        att = num / den[..., None]
    a = merge(att)
    cache = {"x": x, "q": q, "k": k, "v": v, "fQ": fQ, "fK": fK, "num": num, "den": den, "a": a}
    return a @ params["W_o"], cache


def backward(params, cfg, cache, dout):
    """The gradients of a scalar loss with respect to the parameters and x, from dout = dL/d(output)."""
    H = cfg["n_heads"]
    x, q, k, v, fQ, fK, num, den, a = (cache[n] for n in ("x", "q", "k", "v", "fQ", "fK", "num", "den", "a"))
    grads = {"W_o": np.einsum("bnf,bng->fg", a, dout)}
    datt = split(dout @ params["W_o"].T, H)
    dnum = datt / den[..., None]
    dden = -(datt * num).sum(-1) / den**2  # (B, H, N)
    if cfg["causal"]:
        dfQ, dfK, dv = causal_numerator_backward(fQ, fK, v, dnum)
        z = np.cumsum(fK, -2)  # z_i, only to keep this readable; a constant-memory version runs one more sum
        dfQ = dfQ + dden[..., None] * z
        dfK = dfK + np.flip(np.cumsum(np.flip(dden[..., None] * fQ, -2), -2), -2)
    else:
        dfQ = dnum @ (np.swapaxes(v, -1, -2) @ fK) + dden[..., None] * fK.sum(-2, keepdims=True)
        dfK = v @ (np.swapaxes(dnum, -1, -2) @ fQ) + (dden[..., None] * fQ).sum(-2, keepdims=True)
        dv = fK @ (np.swapaxes(fQ, -1, -2) @ dnum)
    phi_grad = FEATURES[cfg["feature"]][1]
    if phi_grad is None:
        raise NotImplementedError("backward needs an elementwise feature map")
    dq, dk = merge(dfQ * phi_grad(q)), merge(dfK * phi_grad(k))
    dv = merge(dv)
    for w, d in (("W_q", dq), ("W_k", dk), ("W_v", dv)):
        grads[w] = np.einsum("bnf,bnd->fd", x, d)
    dx = dq @ params["W_q"].T + dk @ params["W_k"].T + dv @ params["W_v"].T
    return grads, dx


def initial_state(cfg, B=1):
    H, D, M = cfg["n_heads"], cfg["d_k"], cfg["d_v"]
    C = 1 + D + D * D if cfg["feature"] == "poly2" else D
    return np.zeros((B, H, C, M)), np.zeros((B, H, C))


def step(params, cfg, state, x_i):
    """eq-rnn for one token x_i of shape (B, F): update (s, z) and return the output of the layer and the new state.
    The state has H (C M + C) numbers per sequence, whatever the position."""
    H = cfg["n_heads"]
    phi = FEATURES[cfg["feature"]][0]
    s, z = state
    q, k, v = ((x_i @ params[w]).reshape(x_i.shape[0], H, -1) for w in ("W_q", "W_k", "W_v"))
    fq, fk = phi(q), phi(k)
    s = s + fk[..., :, None] * v[..., None, :]
    z = z + fk
    att = np.einsum("bhc,bhcm->bhm", fq, s) / np.einsum("bhc,bhc->bh", fq, z)[..., None]
    return att.reshape(x_i.shape[0], -1) @ params["W_o"], (s, z)


def flops(N, D, M, C=None, kind="linear"):
    """FLOPs of the matrix products of one head of non-causal self-attention on N positions, 2 per multiply-add,
    given Q, K, V and the feature maps already applied; the sums of the keys and of the weights count as products
    with a vector of ones, and the exp and the divisions are left out, being N^2 and N M at most.
    softmax: Q K^T, the weights times V and their row sums. linear: phi(K)^T V, z = phi(K)^T 1, then phi(Q) times
    both."""
    C = D if C is None else C
    if kind == "softmax":
        return 2 * N * N * D + 2 * N * N * M + 2 * N * N
    return 4 * N * C * M + 4 * N * C


def loss(out, target):
    """Half the mean squared error, the loss of the fixtures."""
    return 0.5 * np.mean((out - target) ** 2)
