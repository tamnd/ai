# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/neural-turing-machine/ (ai://P3DWR1).
"""A Neural Turing Machine with a feedforward controller, the NumPy reference.

The memory is a (B, N, M) array: N locations of width M for each sequence of the batch. Each head turns the
outputs of the controller into a weighting over the locations in four stages, content (eq-content, eq-cosine),
interpolation (eq-interpolate), shift (eq-shift) and sharpening (eq-sharpen). A read head returns the weighted
sum of the rows (eq-read); a write head erases and then adds (eq-erase, eq-add). At each step the write heads act
first, so a read sees what was written at the same step. The memory starts at a small constant and every head
at location 0, so no parameter depends on N.
"""

import numpy as np

EPS = 1e-8  # in the denominator of the cosine, for rows or keys of zero length
TINY = 1e-30  # the floor of a squared length before its square root, whose derivative at 0 is infinite
MEMORY_INIT = 1e-6
NAMES = ("Wc", "bc", "Wh", "bh", "Wy", "by")


def config(x_dim=4, y_dim=3, hidden=8, n_loc=6, width=4, n_read=1, n_write=1, shift=1):
    """x_dim inputs, y_dim outputs, a controller of `hidden` tanh units, N = n_loc rows of width M = `width`,
    n_read and n_write heads, and shifts from -shift to shift."""
    if min(x_dim, y_dim, hidden, n_loc, width, n_read) < 1 or n_write < 0 or shift < 0 or 2 * shift + 1 > n_loc:
        raise ValueError("the sizes must be positive, n_write >= 0, and 2 shift + 1 <= n_loc")
    return {
        "x_dim": x_dim,
        "y_dim": y_dim,
        "hidden": hidden,
        "n_loc": n_loc,
        "width": width,
        "n_read": n_read,
        "n_write": n_write,
        "shift": shift,
    }


def head_sizes(cfg):
    """Outputs of the controller per read head (key, beta, g, s, gamma) and per write head (also erase, add)."""
    read = cfg["width"] + 3 + 2 * cfg["shift"] + 1
    return read, read + 2 * cfg["width"]


def n_head_outputs(cfg):
    r, w = head_sizes(cfg)
    return cfg["n_read"] * r + cfg["n_write"] * w


def param_count(cfg):
    x, y, h, m, nr = cfg["x_dim"], cfg["y_dim"], cfg["hidden"], cfg["width"], cfg["n_read"]
    return h * (x + nr * m) + h + n_head_outputs(cfg) * (h + 1) + y * (h + nr * m) + y  # eq: eq-params


def flops(cfg, batch, steps):
    """FLOPs of forward over `steps` steps, a multiply-add counted as 2: the three linear maps of the controller,
    then for each head the cosine products and the shift, and for each read head the weighted sum."""
    x, y, h, m, nr = cfg["x_dim"], cfg["y_dim"], cfg["hidden"], cfg["width"], cfg["n_read"]
    n, k, heads = cfg["n_loc"], 2 * cfg["shift"] + 1, cfg["n_read"] + cfg["n_write"]
    linear = h * (x + nr * m) + n_head_outputs(cfg) * h + y * (h + nr * m)
    return 2 * batch * steps * (linear + heads * n * (m + k) + nr * n * m)  # eq: eq-flops


def init(rng, cfg, scale=1.0):
    x, y, h, m, nr = cfg["x_dim"], cfg["y_dim"], cfg["hidden"], cfg["width"], cfg["n_read"]
    p, z = n_head_outputs(cfg), x + nr * m
    return {
        "Wc": scale * rng.normal(size=(h, z)) / np.sqrt(z),
        "bc": np.zeros(h),
        "Wh": scale * rng.normal(size=(p, h)) / np.sqrt(h),
        "bh": np.zeros(p),
        "Wy": scale * rng.normal(size=(y, h + nr * m)) / np.sqrt(h + nr * m),
        "by": np.zeros(y),
    }


def softplus(z):
    return np.logaddexp(0.0, z)


def sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def safe_norm(v):
    return np.sqrt(np.maximum((v * v).sum(-1), TINY))


def cosine(memory, key):
    """K(k, M(i)) for every row: memory (B, N, M), key (B, M) to (B, N)."""
    dots = np.einsum("bnm,bm->bn", memory, key)
    return dots / (safe_norm(memory) * safe_norm(key)[:, None] + EPS)


def content_weights(memory, key, beta):
    return softmax(beta[:, None] * cosine(memory, key))  # eq: eq-content


def interpolate(wc, w_prev, g):
    return g[:, None] * wc + (1.0 - g[:, None]) * w_prev  # eq: eq-interpolate


def shift(w, s):
    """The circular convolution w~(i) = sum_j w(j) s(i - j), indices mod N; s[:, k] is the weight of shift k - S."""
    big_s = (s.shape[-1] - 1) // 2
    rolled = np.stack([np.roll(w, k, axis=-1) for k in range(-big_s, big_s + 1)], axis=1)  # rolled[:, k, i] = w(i - k)
    return np.einsum("bkn,bk->bn", rolled, s)  # eq: eq-shift


def sharpen(w, gamma):
    p = w ** gamma[:, None]
    return p / p.sum(-1, keepdims=True)  # eq: eq-sharpen


def address(memory, w_prev, head):
    """The four stages of figure 2 of Graves et al.: content, interpolation, shift, sharpening."""
    wc = content_weights(memory, head["key"], head["beta"])
    return sharpen(shift(interpolate(wc, w_prev, head["g"]), head["s"]), head["gamma"])


def read(memory, w):
    return np.einsum("bn,bnm->bm", w, memory)  # eq: eq-read


def write(memory, ws, erases, adds):
    """All erases, then all adds: the order of the heads does not matter within either."""
    keep = np.ones_like(memory)
    for w, e in zip(ws, erases, strict=True):
        keep = keep * (1.0 - w[:, :, None] * e[:, None, :])  # eq: eq-erase
    out = memory * keep
    for w, a in zip(ws, adds, strict=True):
        out = out + w[:, :, None] * a[:, None, :]  # eq: eq-add
    return out


def split_heads(o, cfg):
    """The raw outputs (B, P) to the head parameters, read heads first: key and add by tanh, beta >= 0 by
    softplus, g and erase in (0, 1) by the sigmoid, s by a softmax, gamma >= 1 by 1 + softplus."""
    m, k = cfg["width"], 2 * cfg["shift"] + 1
    heads, i = [], 0
    for j in range(cfg["n_read"] + cfg["n_write"]):
        hd = {
            "key": np.tanh(o[:, i : i + m]),
            "beta": softplus(o[:, i + m]),
            "g": sigmoid(o[:, i + m + 1]),
            "s": softmax(o[:, i + m + 2 : i + m + 2 + k]),
            "gamma": 1.0 + softplus(o[:, i + m + 2 + k]),
        }
        i += m + 3 + k
        if j >= cfg["n_read"]:
            hd["erase"], hd["add"] = sigmoid(o[:, i : i + m]), np.tanh(o[:, i + m : i + 2 * m])
            i += 2 * m
        heads.append(hd)
    return heads


def initial_state(cfg, batch):
    """The memory at MEMORY_INIT, every head on location 0, the read vectors at zero."""
    w0 = np.zeros((batch, cfg["n_loc"]))
    w0[:, 0] = 1.0
    memory = np.full((batch, cfg["n_loc"], cfg["width"]), MEMORY_INIT)
    heads = cfg["n_read"] + cfg["n_write"]
    return memory, [w0.copy() for _ in range(heads)], [np.zeros((batch, cfg["width"])) for _ in range(cfg["n_read"])]


def step(params, state, x_t, cfg):
    """One step: the controller, the writes, the reads, the output logits (B, Y)."""
    memory, ws, reads = state
    hid = np.tanh(np.concatenate([x_t] + reads, axis=-1) @ params["Wc"].T + params["bc"])
    heads = split_heads(hid @ params["Wh"].T + params["bh"], cfg)
    nr = cfg["n_read"]
    w_write = [address(memory, ws[nr + j], hd) for j, hd in enumerate(heads[nr:])]
    memory = write(memory, w_write, [hd["erase"] for hd in heads[nr:]], [hd["add"] for hd in heads[nr:]])
    w_read = [address(memory, ws[j], hd) for j, hd in enumerate(heads[:nr])]
    reads = [read(memory, w) for w in w_read]
    y = np.concatenate([hid] + reads, axis=-1) @ params["Wy"].T + params["by"]
    return (memory, w_read + w_write, reads), y


def forward(params, x, cfg, trace=False):
    """x (B, T, X) to logits (B, T, Y); with trace, also the weightings (T, heads, B, N) and the last memory."""
    state = initial_state(cfg, x.shape[0])
    ys, ws = [], []
    for t in range(x.shape[1]):
        state, y = step(params, state, x[:, t], cfg)
        ys.append(y)
        ws.append(np.stack(state[1]))
    logits = np.stack(ys, axis=1)
    return (logits, np.stack(ws), state[0]) if trace else logits


def bits_per_sequence(logits, targets, mask):
    """The cross-entropy of the sigmoid outputs in bits, summed over the masked steps and the bits, mean over B."""
    nats = np.logaddexp(0.0, logits) - targets * logits
    return float((nats * mask[:, :, None]).sum() / logits.shape[0] / np.log(2.0))
