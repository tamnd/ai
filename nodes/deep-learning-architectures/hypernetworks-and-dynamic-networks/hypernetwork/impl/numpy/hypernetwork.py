# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/hypernetwork/ (ai://NEW-hypernetwork).
"""Hypernetworks of Ha, Dai and Le (2016), the NumPy reference, forward and backward.

Two kinds, as in the paper. The static hypernetwork (eq-static on the page, the paper's eq. 2) is a two-layer linear
network that turns a learned embedding z_j into a basic kernel of N_out filters of N_in channels, f x f each; a layer
whose kernel is m times wider in both directions is tiled from m^2 basic kernels (eq-tile, the paper's eq. 3). The
main network of this page is the one of shapes.yaml: an ordinary stem convolution, D convolutions whose kernels the
hypernetwork generates, a global average pool and a linear layer to the classes; forward, backward and the counts.

The dynamic hypernetwork is the HyperRNN: a small recurrent cell reads [h_{t-1}; x_t] and gives at each step the
embeddings z_h, z_x, z_b of the main RNN's weights. hyper_rnn_full is the general form (eq-full, the paper's eq. 5),
which builds both weight matrices from N_z slices at every step; hyper_rnn is the memory-efficient form (eq-scaled,
the paper's eq. 8), which scales the rows of fixed matrices by the weight scaling vectors d_h and d_x, with its
gradient through time by hand. hyper_lstm is the HyperLSTM of the paper's Appendix A.2.2, eqs. 10 to 13 without the
dropout, and init_lstm its initialization of Appendix A.2.3. The embeddings are computed from the hyper cell's new
state hhat_t, as the text of Section 3.2 says, not from hhat_{t-1} as eqs. 6 and 11 are printed.
"""

import math

import numpy as np

# ---------------------------------------------------------------------------------------------------------------------
# The static hypernetwork and the main convolutional network


def config(c_in=1, n_basic=2, mult=2, fsize=3, depth=2, n_z=2, size=5, classes=3):
    """The config of shapes.yaml; the defaults are its example. The paper fixes d, the hidden size of the
    hypernetwork, equal to N_z, and so does this page."""
    cfg = {"c_in": c_in, "n_basic": n_basic, "mult": mult, "fsize": fsize, "depth": depth, "n_z": n_z, "size": size,
           "classes": classes}  # fmt: skip
    for k, v in cfg.items():
        if int(v) != v or v < 1:
            raise ValueError(f"{k} must be a positive integer, not {v!r}")
    if fsize % 2 == 0:
        raise ValueError("fsize must be odd, so that a padding of (fsize - 1) / 2 keeps the size")
    cfg = {k: int(v) for k, v in cfg.items()}
    cfg["d"] = cfg["n_z"]
    return cfg


def width(cfg):
    """The channels of the main network: mult basic blocks of n_basic."""
    return cfg["mult"] * cfg["n_basic"]


def n_embeddings(cfg):
    """D m^2: each generated layer is tiled from m x m basic kernels, each from its own embedding."""
    return cfg["depth"] * cfg["mult"] ** 2


def hyper_param_count(n_z, d, n_in, n_out, f, n_emb):
    """The paper's count for the two-layer hypernetwork with its embeddings: N_z n_emb + d (N_z + 1) N_in
    + f N_out f (d + 1) (eq-hyper-params; the paper writes D for n_emb, one embedding per layer)."""
    return n_z * n_emb + d * (n_z + 1) * n_in + f * n_out * f * (d + 1)


def one_layer_param_count(n_z, n_in, n_out, f):
    """A one-layer linear hypernetwork from z to the whole kernel, the paper's N_z N_in f N_out f, without biases."""
    return n_z * n_in * f * n_out * f


def param_shapes(cfg):
    N, M, f, Nz, d = cfg["n_basic"], width(cfg), cfg["fsize"], cfg["n_z"], cfg["d"]
    shapes = {
        "stem_w": (M, cfg["c_in"], f, f),
        "stem_b": (M,),
        "z": (n_embeddings(cfg), Nz),
        "W_in": (N, d, Nz),  # the paper's W_i, one d x N_z matrix for each input channel i of a basic kernel
        "B_in": (N, d),  # the paper's B_i
        "W_out": (f, N * f, d),
        "B_out": (f, N * f),
    }
    for j in range(cfg["depth"]):
        shapes[f"conv{j}_b"] = (M,)
    shapes |= {"head_w": (M, cfg["classes"]), "head_b": (cfg["classes"],)}
    return shapes


def param_count(cfg):
    """eq-params from the formulas, so that it can be checked against param_shapes."""
    N, M, f, K = cfg["n_basic"], width(cfg), cfg["fsize"], cfg["classes"]
    stem = f * f * cfg["c_in"] * M + M
    hyper = hyper_param_count(cfg["n_z"], cfg["d"], N, N, f, n_embeddings(cfg))
    return stem + hyper + cfg["depth"] * M + M * K + K


def flops(cfg, B):
    """eq-flops: the multiply-adds of the products of one forward pass over B images, each counted as 2. The kernels
    are generated once per pass, whatever B."""
    N, M, f, S, Nz, d = cfg["n_basic"], width(cfg), cfg["fsize"], cfg["size"], cfg["n_z"], cfg["d"]
    stem = 2 * B * S * S * f * f * cfg["c_in"] * M
    hyper = 2 * n_embeddings(cfg) * N * d * (Nz + f * f * N)
    convs = cfg["depth"] * 2 * B * S * S * f * f * M * M
    return stem + hyper + convs + 2 * B * M * cfg["classes"]


def init(rng, cfg, scale=None):
    """Random parameters. The kernels of the stem have the He scale; the hypernetwork is scaled so that its kernels
    start with about the same scale, sqrt(2 / (f^2 M)); the embeddings are N(0, 1)."""
    N, M, f, Nz, d = cfg["n_basic"], width(cfg), cfg["fsize"], cfg["n_z"], cfg["d"]
    target = math.sqrt(2.0 / (f * f * M)) if scale is None else scale
    p = {
        "stem_w": rng.normal(0.0, math.sqrt(2.0 / (f * f * cfg["c_in"])), (M, cfg["c_in"], f, f)),
        "stem_b": np.zeros(M),
        "z": rng.normal(size=(n_embeddings(cfg), Nz)),
        "W_in": rng.normal(0.0, Nz**-0.5, (N, d, Nz)),
        "B_in": np.zeros((N, d)),
        "W_out": rng.normal(0.0, target * d**-0.5, (f, N * f, d)),
        "B_out": np.zeros((f, N * f)),
    }
    for j in range(cfg["depth"]):
        p[f"conv{j}_b"] = np.zeros(M)
    p |= {"head_w": rng.normal(0.0, M**-0.5, (M, cfg["classes"])), "head_b": np.zeros(cfg["classes"])}
    return p


def generate(p, z):
    """eq-static for E embeddings at once: z (E, N_z) gives the hidden vectors a (E, N_in, d) and the kernels in the
    paper's layout, K (E, N_in, f, N_out f), where K[e, i] is the slice K_i of the paper, f x N_out f."""
    a = np.einsum("idz,ez->eid", p["W_in"], z) + p["B_in"]  # eq: eq-static  a_i = W_i z + B_i
    K = np.einsum("uwd,eid->eiuw", p["W_out"], a) + p["B_out"]  # eq: eq-static  K_i = <W_out, a_i> + B_out
    return a, K


def to_filters(K, f):
    """The paper's kernel matrix K^j, N_in f x N_out f, with entry (i f + u, o f + v) the weight of input channel i,
    output channel o and filter position (u, v), as the (N_out, N_in, f, f) array of a convolution."""
    E, Nin, _, w = K.shape
    return K.reshape(E, Nin, f, w // f, f).transpose(0, 3, 1, 2, 4)  # (E, N_out, N_in, f, f)


def from_filters(w, f):
    """The inverse of to_filters."""
    E, Nout, Nin = w.shape[:3]
    return w.transpose(0, 2, 3, 1, 4).reshape(E, Nin, f, Nout * f)


def kernel_matrix(K):
    """K^j as the paper writes it: the slices K_1, ..., K_{N_in} stacked, (N_in f) x (N_out f)."""
    E, Nin, f, w = K.shape
    return K.reshape(E, Nin * f, w)


def tile(basic, m_in, m_out=None):
    """eq-tile: m_in m_out basic filters (m_in m_out, N, N, f, f), ordered row by row over (input block, output
    block) as in the paper's eq. 3, give one kernel (m_out N, m_in N, f, f). Block (r, c) maps the input channels
    r N to (r + 1) N - 1 to the output channels c N to (c + 1) N - 1."""
    m_out = m_in if m_out is None else m_out
    E, N, _, f, _ = basic.shape
    if E != m_in * m_out:
        raise ValueError(f"{E} basic kernels do not tile an {m_in} x {m_out} grid")
    g = basic.reshape(m_in, m_out, N, N, f, f)  # (r, c, out, in, u, v)
    return g.transpose(1, 2, 0, 3, 4, 5).reshape(m_out * N, m_in * N, f, f)


def untile(w, m_in, m_out=None):
    """The inverse of tile: (m_out N, m_in N, f, f) to (m_in m_out, N, N, f, f)."""
    m_out = m_in if m_out is None else m_out
    N, f = w.shape[1] // m_in, w.shape[2]
    return w.reshape(m_out, N, m_in, N, f, f).transpose(2, 0, 1, 3, 4, 5).reshape(m_in * m_out, N, N, f, f)


def kernels(p, cfg):
    """The D kernels of the main network, each (M, M, f, f), and the cache of the generation."""
    m, f = cfg["mult"], cfg["fsize"]
    a, K = generate(p, p["z"])
    basic = to_filters(K, f)  # (D m^2, N, N, f, f)
    ws = [tile(basic[j * m * m : (j + 1) * m * m], m) for j in range(cfg["depth"])]
    return ws, {"a": a}


def im2col(x, k, pad):
    """The windows of x (B, C, H, W) for stride 1: (B, C, H', W', k, k), a view of the zero-padded input."""
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    return np.lib.stride_tricks.sliding_window_view(xp, (k, k), axis=(2, 3))


def conv2d(x, w, b):
    """A stride-1 convolution with the padding that keeps the size: x (B, C, S, S), w (D, C, f, f)."""
    cols = im2col(x, w.shape[2], w.shape[2] // 2)  # (B, C, S, S, f, f)
    return np.einsum("bchwuv,dcuv->bdhw", cols, w) + b[None, :, None, None]


def conv2d_backward(x, w, dy):
    """(dx, dw, db) of conv2d given dy (B, D, S, S)."""
    k = w.shape[2]
    pad = k // 2
    cols = im2col(x, k, pad)
    dw = np.einsum("bchwuv,bdhw->dcuv", cols, dy)
    db = dy.sum(axis=(0, 2, 3))
    dcols = np.einsum("bdhw,dcuv->bchwuv", dy, w)
    B, C, H, W = x.shape
    dxp = np.zeros((B, C, H + 2 * pad, W + 2 * pad))
    for u in range(k):
        for v in range(k):
            dxp[:, :, u : u + H, v : v + W] += dcols[:, :, :, :, u, v]
    return dxp[:, :, pad : pad + H, pad : pad + W], dw, db


def forward(p, x, cfg):
    """Logits (B, K) of the images x (B, C_in, S, S) and the cache for backward."""
    ws, gen = kernels(p, cfg)
    pre = [conv2d(x, p["stem_w"], p["stem_b"])]
    hs = [np.maximum(pre[0], 0.0)]
    for j, w in enumerate(ws):
        pre.append(conv2d(hs[-1], w, p[f"conv{j}_b"]))
        hs.append(np.maximum(pre[-1], 0.0))
    pooled = hs[-1].mean(axis=(2, 3))  # (B, M)
    logits = pooled @ p["head_w"] + p["head_b"]
    return logits, {"x": x, "ws": ws, "pre": pre, "hs": hs, "pooled": pooled, **gen}


def log_softmax(s):
    s = s - s.max(-1, keepdims=True)
    return s - np.log(np.exp(s).sum(-1, keepdims=True))


def loss(logits, y):
    """eq-loss: the mean cross-entropy of the labels y (B,)."""
    return -log_softmax(logits)[np.arange(len(y)), y].mean()


def loss_grad(logits, y):
    g = np.exp(log_softmax(logits))
    g[np.arange(len(y)), y] -= 1.0
    return g / len(y)


def generate_backward(p, z, a, dK):
    """Gradients of eq-static, given dK (E, N_in, f, N_out f): dW_in, dB_in, dW_out, dB_out and dz."""
    da = np.einsum("eiuw,uwd->eid", dK, p["W_out"])
    return {
        "W_out": np.einsum("eiuw,eid->uwd", dK, a),
        "B_out": dK.sum(axis=(0, 1)),
        "W_in": np.einsum("eid,ez->idz", da, z),
        "B_in": da.sum(axis=0),
        "z": np.einsum("eid,idz->ez", da, p["W_in"]),
    }


def backward(p, cfg, cache, dlogits):
    """The gradient of every parameter and of the input, given dlogits (B, K). The gradients of the D generated
    kernels are untiled and sent back through eq-static to the shared hypernetwork and to each embedding."""
    m, f, S = cfg["mult"], cfg["fsize"], cfg["size"]
    g = {"head_w": cache["pooled"].T @ dlogits, "head_b": dlogits.sum(0)}
    dh = np.broadcast_to((dlogits @ p["head_w"].T)[:, :, None, None] / (S * S), cache["hs"][-1].shape)
    dws = [None] * cfg["depth"]
    for j in reversed(range(cfg["depth"])):
        dpre = dh * (cache["pre"][j + 1] > 0)
        dh, dws[j], g[f"conv{j}_b"] = conv2d_backward(cache["hs"][j], cache["ws"][j], dpre)
    dpre = dh * (cache["pre"][0] > 0)
    dx, g["stem_w"], g["stem_b"] = conv2d_backward(cache["x"], p["stem_w"], dpre)
    dbasic = np.concatenate([untile(dw, m) for dw in dws])  # (D m^2, N, N, f, f)
    g |= generate_backward(p, p["z"], cache["a"], from_filters(dbasic, f))
    return g, dx


# ---------------------------------------------------------------------------------------------------------------------
# The dynamic hypernetwork: HyperRNN


def config_rnn(n_x=3, n_h=5, n_hat=4, n_z=2):
    cfg = {"n_x": n_x, "n_h": n_h, "n_hat": n_hat, "n_z": n_z}
    for k, v in cfg.items():
        if int(v) != v or v < 1:
            raise ValueError(f"{k} must be a positive integer, not {v!r}")
    return {k: int(v) for k, v in cfg.items()}


def rnn_param_shapes(cfg, full=False):
    """The parameters of the HyperRNN. The hyper cell (eq-hyper-cell, the paper's eq. 6) is shared by both forms;
    full=True gives the tensors of eq-full in place of the fixed matrices and the scaling projections of eq-scaled."""
    Nx, Nh, Nhat, Nz = cfg["n_x"], cfg["n_h"], cfg["n_hat"], cfg["n_z"]
    if full:
        main = {"W_hz": (Nh, Nh, Nz), "W_xz": (Nh, Nx, Nz), "W_bz": (Nh, Nz), "b_0": (Nh,)}
    else:
        main = {"W_h": (Nh, Nh), "W_x": (Nh, Nx), "W_hz": (Nh, Nz), "W_xz": (Nh, Nz), "W_bz": (Nh, Nz), "b_0": (Nh,)}
    hyper = {"W_hhat": (Nhat, Nhat), "W_xhat": (Nhat, Nh + Nx), "b_hat": (Nhat,), "W_hh": (Nz, Nhat), "b_hh": (Nz,),
             "W_hx": (Nz, Nhat), "b_hx": (Nz,), "W_hb": (Nz, Nhat)}  # fmt: skip
    return main | hyper


def rnn_param_count(cfg, full=False):
    return sum(math.prod(s) for s in rnn_param_shapes(cfg, full).values())


def init_rnn(rng, cfg, full=False, scale=0.5):
    return {k: rng.normal(0.0, scale / math.sqrt(s[-1] if len(s) > 1 else 4.0), s)
            for k, s in rnn_param_shapes(cfg, full).items()}  # fmt: skip


def hyper_cell(p, hhat, h, x):
    """eq-hyper-cell: the hyper cell reads xhat = [h_{t-1}; x_t] and gives its new state and the three embeddings."""
    xhat = np.concatenate([h, x], -1)
    hhat = np.tanh(hhat @ p["W_hhat"].T + xhat @ p["W_xhat"].T + p["b_hat"])
    z_h = hhat @ p["W_hh"].T + p["b_hh"]
    z_x = hhat @ p["W_hx"].T + p["b_hx"]
    z_b = hhat @ p["W_hb"].T
    return xhat, hhat, z_h, z_x, z_b


def hyper_rnn_full(p, x):
    """eq-full over a sequence x (B, T, N_x): at every step W_h(z_h) = <W_hz, z_h> and W_x(z_x) = <W_xz, z_x> are
    built for each sequence of the batch, B N_h (N_h + N_x) numbers. Returns the states (B, T, N_h) and the
    generated matrices (B, T, N_h, N_h) and (B, T, N_h, N_x)."""
    B, T, _ = x.shape
    Nh, Nhat = p["W_hz"].shape[0], p["W_hhat"].shape[0]
    h, hhat = np.zeros((B, Nh)), np.zeros((B, Nhat))
    hs, Whs, Wxs = [], [], []
    for t in range(T):
        _, hhat, z_h, z_x, z_b = hyper_cell(p, hhat, h, x[:, t])
        Wh = np.einsum("ijz,bz->bij", p["W_hz"], z_h)  # eq: eq-full
        Wx = np.einsum("ijz,bz->bij", p["W_xz"], z_x)
        b = z_b @ p["W_bz"].T + p["b_0"]
        h = np.tanh(np.einsum("bij,bj->bi", Wh, h) + np.einsum("bij,bj->bi", Wx, x[:, t]) + b)
        hs.append(h)
        Whs.append(Wh)
        Wxs.append(Wx)
    return np.stack(hs, 1), np.stack(Whs, 1), np.stack(Wxs, 1)


def full_from_scaled(p):
    """The tensors of eq-full that give the same network as the parameters p of eq-scaled: slice k of W_hz is
    diag(W_hz[:, k]) W_h, and the same for W_x (the derivation weight-scaling)."""
    q = {k: v for k, v in p.items() if k not in ("W_h", "W_x")}
    q["W_hz"] = p["W_hz"][:, None, :] * p["W_h"][:, :, None]
    q["W_xz"] = p["W_xz"][:, None, :] * p["W_x"][:, :, None]
    return q


def hyper_rnn(p, x, return_cache=False):
    """eq-scaled over a sequence x (B, T, N_x): h_t = tanh(d_h * (W_h h_{t-1}) + d_x * (W_x x_t) + b), with
    d_h = W_hz z_h, d_x = W_xz z_x and b = W_bz z_b + b_0. Returns the states (B, T, N_h)."""
    B, T, _ = x.shape
    Nh, Nhat = p["W_h"].shape[0], p["W_hhat"].shape[0]
    h, hhat = np.zeros((B, Nh)), np.zeros((B, Nhat))
    steps, hs = [], []
    for t in range(T):
        xhat, hhat_new, z_h, z_x, z_b = hyper_cell(p, hhat, h, x[:, t])
        d_h, d_x = z_h @ p["W_hz"].T, z_x @ p["W_xz"].T  # eq: eq-scaled  the weight scaling vectors
        b = z_b @ p["W_bz"].T + p["b_0"]
        u, v = h @ p["W_h"].T, x[:, t] @ p["W_x"].T
        h_new = np.tanh(d_h * u + d_x * v + b)
        steps.append({"h": h, "hhat": hhat, "xhat": xhat, "hhat_new": hhat_new, "z_h": z_h, "z_x": z_x, "z_b": z_b,
                      "d_h": d_h, "d_x": d_x, "u": u, "v": v, "h_new": h_new})  # fmt: skip
        h, hhat = h_new, hhat_new
        hs.append(h)
    out = np.stack(hs, 1)
    return (out, steps) if return_cache else out


def hyper_rnn_backward(p, x, steps, dout):
    """Backpropagation through time of hyper_rnn, given dout = dL/dh_t (B, T, N_h): the gradients of all the
    parameters and dx (B, T, N_x). The gradient reaches h_{t-1} by two paths, through W_h in the main RNN and through
    xhat in the hyper cell."""
    g = {k: np.zeros_like(v) for k, v in p.items()}
    dx = np.zeros_like(x)
    Nh = p["W_h"].shape[0]
    carry_h, carry_hat = np.zeros_like(dout[:, 0]), np.zeros((x.shape[0], p["W_hhat"].shape[0]))
    for t in reversed(range(x.shape[1])):
        s = steps[t]
        da = (dout[:, t] + carry_h) * (1.0 - s["h_new"] ** 2)
        g["W_h"] += (da * s["d_h"]).T @ s["h"]
        g["W_x"] += (da * s["d_x"]).T @ x[:, t]
        g["b_0"] += da.sum(0)
        dh_prev = (da * s["d_h"]) @ p["W_h"]
        dx[:, t] += (da * s["d_x"]) @ p["W_x"]
        dd_h, dd_x = da * s["u"], da * s["v"]
        g["W_hz"] += dd_h.T @ s["z_h"]
        g["W_xz"] += dd_x.T @ s["z_x"]
        g["W_bz"] += da.T @ s["z_b"]
        dz_h, dz_x, dz_b = dd_h @ p["W_hz"], dd_x @ p["W_xz"], da @ p["W_bz"]
        hn = s["hhat_new"]
        g["W_hh"] += dz_h.T @ hn
        g["b_hh"] += dz_h.sum(0)
        g["W_hx"] += dz_x.T @ hn
        g["b_hx"] += dz_x.sum(0)
        g["W_hb"] += dz_b.T @ hn
        dhn = dz_h @ p["W_hh"] + dz_x @ p["W_hx"] + dz_b @ p["W_hb"] + carry_hat
        dahat = dhn * (1.0 - hn**2)
        g["W_hhat"] += dahat.T @ s["hhat"]
        g["W_xhat"] += dahat.T @ s["xhat"]
        g["b_hat"] += dahat.sum(0)
        dxhat = dahat @ p["W_xhat"]
        carry_h = dh_prev + dxhat[:, :Nh]
        dx[:, t] += dxhat[:, Nh:]
        carry_hat = dahat @ p["W_hhat"]
    return g, dx


def rnn_loss(out, target):
    return 0.5 * np.mean((out - target) ** 2)


# ---------------------------------------------------------------------------------------------------------------------
# HyperLSTM, the paper's Appendix A.2.2 and A.2.3

GATES = ("i", "g", "f", "o")


def lstm_param_shapes(cfg):
    """Every array has a leading axis of 4, the gates i, g, f, o in that order."""
    Nx, Nh, Nhat, Nz = cfg["n_x"], cfg["n_h"], cfg["n_hat"], cfg["n_z"]
    return {
        "W_h": (4, Nh, Nh), "W_x": (4, Nh, Nx), "W_hz": (4, Nh, Nz), "W_xz": (4, Nh, Nz), "W_bz": (4, Nh, Nz),
        "b_0": (4, Nh),
        "W_hhat": (4, Nhat, Nhat), "W_xhat": (4, Nhat, Nh + Nx), "b_hat": (4, Nhat),
        "W_hh": (4, Nz, Nhat), "b_hh": (4, Nz), "W_hx": (4, Nz, Nhat), "b_hx": (4, Nz), "W_hb": (4, Nz, Nhat),
    }  # fmt: skip


def orthogonal(rng, shape):
    """A matrix with orthonormal rows or columns, whichever are fewer, from the QR decomposition of a Gaussian one."""
    a = rng.normal(size=(max(shape), min(shape)))
    q, r = np.linalg.qr(a)
    q = q * np.sign(np.diag(r))
    return q if shape[0] >= shape[1] else q.T


def init_lstm(rng, cfg):
    """Appendix A.2.3: orthogonal weights and zero biases in the hyper cell; zero weights and unit biases for the
    projections to z_h and z_x, N(0, 0.01^2) for that to z_b; orthogonal W_h and W_x, zero b_0 and W_bz; W_hz and W_xz
    the constant 0.1 / N_z, so that every weight scaling vector starts at 0.1."""
    s, Nz = lstm_param_shapes(cfg), cfg["n_z"]
    p = {}
    for k in ("W_h", "W_x", "W_hhat", "W_xhat"):
        p[k] = np.stack([orthogonal(rng, s[k][1:]) for _ in range(4)])
    p |= {"b_hat": np.zeros(s["b_hat"]), "W_hh": np.zeros(s["W_hh"]), "b_hh": np.ones(s["b_hh"]),
          "W_hx": np.zeros(s["W_hx"]), "b_hx": np.ones(s["b_hx"]), "W_hb": rng.normal(0.0, 0.01, s["W_hb"]),
          "b_0": np.zeros(s["b_0"]), "W_bz": np.zeros(s["W_bz"]), "W_hz": np.full(s["W_hz"], 0.1 / Nz),
          "W_xz": np.full(s["W_xz"], 0.1 / Nz)}  # fmt: skip
    return p


def layer_norm(a, eps=1e-5):
    """Layer normalization over the last axis, without a learned gain and bias."""
    mu = a.mean(-1, keepdims=True)
    return (a - mu) / np.sqrt(((a - mu) ** 2).mean(-1, keepdims=True) + eps)


def sigmoid(a):
    return 1.0 / (1.0 + np.exp(-a))


def hyper_lstm(p, x, ln=True, return_scales=False):
    """The HyperLSTM over x (B, T, N_x), the paper's eqs. 10 to 13 without the dropout; with ln=False every LN is
    the identity. Returns the main states h (B, T, N_h) and, with return_scales, the vectors d_h of the four gates
    (B, T, 4, N_h)."""
    norm = layer_norm if ln else (lambda a: a)
    B, T, _ = x.shape
    Nh, Nhat = p["W_h"].shape[1], p["W_hhat"].shape[1]
    h, c, hhat, chat = np.zeros((B, Nh)), np.zeros((B, Nh)), np.zeros((B, Nhat)), np.zeros((B, Nhat))
    hs, ds = [], []
    for t in range(T):
        xhat = np.concatenate([h, x[:, t]], -1)
        gh = norm(np.einsum("ykj,bj->ybk", p["W_hhat"], hhat) + np.einsum("ykj,bj->ybk", p["W_xhat"], xhat)
                  + p["b_hat"][:, None])  # fmt: skip
        chat = sigmoid(gh[2]) * chat + sigmoid(gh[0]) * np.tanh(gh[1])  # eq. 10
        hhat = sigmoid(gh[3]) * np.tanh(norm(chat))
        z_h = np.einsum("yzk,bk->ybz", p["W_hh"], hhat) + p["b_hh"][:, None]  # eq. 11
        z_x = np.einsum("yzk,bk->ybz", p["W_hx"], hhat) + p["b_hx"][:, None]
        z_b = np.einsum("yzk,bk->ybz", p["W_hb"], hhat)
        d_h = np.einsum("ynz,ybz->ybn", p["W_hz"], z_h)  # eq. 12
        d_x = np.einsum("ynz,ybz->ybn", p["W_xz"], z_x)
        bias = np.einsum("ynz,ybz->ybn", p["W_bz"], z_b) + p["b_0"][:, None]
        gates = norm(d_h * np.einsum("ynj,bj->ybn", p["W_h"], h) + d_x * np.einsum("ynj,bj->ybn", p["W_x"], x[:, t])
                     + bias)  # fmt: skip
        c = sigmoid(gates[2]) * c + sigmoid(gates[0]) * np.tanh(gates[1])  # eq. 13
        h = sigmoid(gates[3]) * np.tanh(norm(c))
        hs.append(h)
        ds.append(d_h.transpose(1, 0, 2))
    out = np.stack(hs, 1)
    return (out, np.stack(ds, 1)) if return_scales else out


# ---------------------------------------------------------------------------------------------------------------------
# The worked example of the page


def worked_example():
    """One basic kernel of 1 input and 2 output channels of size 1 x 1, N_z = d = 1: the kernel is the affine function
    W_out (W_1 z + B_1) + B_out of the single embedding z."""
    p = {"W_in": np.array([[[2.0]]]), "B_in": np.array([[1.0]]), "W_out": np.array([[[1.0], [-1.0]]]),
         "B_out": np.array([[0.5, 0.0]])}  # fmt: skip
    return {z: generate(p, np.array([[z]]))[1][0, 0, 0] for z in (0.0, 1.0, -0.5)}
