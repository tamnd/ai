# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/neural-turing-machine/ (ai://P3DWR1).
"""A Neural Turing Machine with a feedforward controller, the PyTorch reference.

It follows the NumPy reference step for step, on a dict of tensors, so that autograd gives the gradients. The
squared lengths in the cosine have a floor before the square root: its derivative at 0 is infinite, and a row of
zeros in the memory would make every gradient NaN.
"""

import math

import torch

EPS = 1e-8
TINY = 1e-30
MEMORY_INIT = 1e-6
NAMES = ("Wc", "bc", "Wh", "bh", "Wy", "by")


def config(x_dim=4, y_dim=3, hidden=8, n_loc=6, width=4, n_read=1, n_write=1, shift=1):
    if min(x_dim, y_dim, hidden, n_loc, width, n_read) < 1 or n_write < 0 or shift < 0 or 2 * shift + 1 > n_loc:
        raise ValueError("the sizes must be positive, n_write >= 0, and 2 shift + 1 <= n_loc")
    return {"x_dim": x_dim, "y_dim": y_dim, "hidden": hidden, "n_loc": n_loc, "width": width,
            "n_read": n_read, "n_write": n_write, "shift": shift}


def n_head_outputs(cfg):
    read = cfg["width"] + 3 + 2 * cfg["shift"] + 1
    return cfg["n_read"] * read + cfg["n_write"] * (read + 2 * cfg["width"])


def param_count(cfg):
    x, y, h, m, nr = cfg["x_dim"], cfg["y_dim"], cfg["hidden"], cfg["width"], cfg["n_read"]
    return h * (x + nr * m) + h + n_head_outputs(cfg) * (h + 1) + y * (h + nr * m) + y


def init(gen, cfg, scale=1.0, dtype=torch.float64):
    x, y, h, m, nr = cfg["x_dim"], cfg["y_dim"], cfg["hidden"], cfg["width"], cfg["n_read"]
    p, z = n_head_outputs(cfg), x + nr * m

    def normal(*shape):
        return scale * torch.randn(*shape, generator=gen, dtype=dtype) / math.sqrt(shape[1])

    params = {"Wc": normal(h, z), "bc": torch.zeros(h, dtype=dtype), "Wh": normal(p, h),
              "bh": torch.zeros(p, dtype=dtype), "Wy": normal(y, h + nr * m), "by": torch.zeros(y, dtype=dtype)}
    return {k: v.requires_grad_() for k, v in params.items()}


def safe_norm(v):
    return torch.sqrt(torch.clamp((v * v).sum(-1), min=TINY))


def cosine(memory, key):
    dots = torch.einsum("bnm,bm->bn", memory, key)
    return dots / (safe_norm(memory) * safe_norm(key)[:, None] + EPS)


def content_weights(memory, key, beta):
    return torch.softmax(beta[:, None] * cosine(memory, key), dim=-1)


def interpolate(wc, w_prev, g):
    return g[:, None] * wc + (1.0 - g[:, None]) * w_prev


def shift(w, s):
    big_s = (s.shape[-1] - 1) // 2
    rolled = torch.stack([torch.roll(w, k, dims=-1) for k in range(-big_s, big_s + 1)], dim=1)
    return torch.einsum("bkn,bk->bn", rolled, s)


def sharpen(w, gamma):
    p = w ** gamma[:, None]
    return p / p.sum(-1, keepdim=True)


def address(memory, w_prev, head):
    wc = content_weights(memory, head["key"], head["beta"])
    return sharpen(shift(interpolate(wc, w_prev, head["g"]), head["s"]), head["gamma"])


def read(memory, w):
    return torch.einsum("bn,bnm->bm", w, memory)


def write(memory, ws, erases, adds):
    keep = torch.ones_like(memory)
    for w, e in zip(ws, erases):
        keep = keep * (1.0 - w[:, :, None] * e[:, None, :])
    out = memory * keep
    for w, a in zip(ws, adds):
        out = out + w[:, :, None] * a[:, None, :]
    return out


def split_heads(o, cfg):
    m, k = cfg["width"], 2 * cfg["shift"] + 1
    sp = torch.nn.functional.softplus
    heads, i = [], 0
    for j in range(cfg["n_read"] + cfg["n_write"]):
        hd = {"key": torch.tanh(o[:, i:i + m]), "beta": sp(o[:, i + m]), "g": torch.sigmoid(o[:, i + m + 1]),
              "s": torch.softmax(o[:, i + m + 2:i + m + 2 + k], dim=-1), "gamma": 1.0 + sp(o[:, i + m + 2 + k])}
        i += m + 3 + k
        if j >= cfg["n_read"]:
            hd["erase"], hd["add"] = torch.sigmoid(o[:, i:i + m]), torch.tanh(o[:, i + m:i + 2 * m])
            i += 2 * m
        heads.append(hd)
    return heads


def initial_state(cfg, batch, dtype=torch.float64):
    w0 = torch.zeros(batch, cfg["n_loc"], dtype=dtype)
    w0[:, 0] = 1.0
    memory = torch.full((batch, cfg["n_loc"], cfg["width"]), MEMORY_INIT, dtype=dtype)
    reads = [torch.zeros(batch, cfg["width"], dtype=dtype) for _ in range(cfg["n_read"])]
    return memory, [w0.clone() for _ in range(cfg["n_read"] + cfg["n_write"])], reads


def step(params, state, x_t, cfg):
    memory, ws, reads = state
    hid = torch.tanh(torch.cat([x_t] + reads, dim=-1) @ params["Wc"].T + params["bc"])
    heads = split_heads(hid @ params["Wh"].T + params["bh"], cfg)
    nr = cfg["n_read"]
    w_write = [address(memory, ws[nr + j], hd) for j, hd in enumerate(heads[nr:])]
    memory = write(memory, w_write, [hd["erase"] for hd in heads[nr:]], [hd["add"] for hd in heads[nr:]])
    w_read = [address(memory, ws[j], hd) for j, hd in enumerate(heads[:nr])]
    reads = [read(memory, w) for w in w_read]
    y = torch.cat([hid] + reads, dim=-1) @ params["Wy"].T + params["by"]
    return (memory, w_read + w_write, reads), y


def forward(params, x, cfg, trace=False):
    state = initial_state(cfg, x.shape[0], x.dtype)
    ys, ws = [], []
    for t in range(x.shape[1]):
        state, y = step(params, state, x[:, t], cfg)
        ys.append(y)
        ws.append(torch.stack(state[1]))
    logits = torch.stack(ys, dim=1)
    return (logits, torch.stack(ws), state[0]) if trace else logits


def bits_per_sequence(logits, targets, mask):
    nats = torch.nn.functional.softplus(logits) - targets * logits
    return (nats * mask[:, :, None]).sum() / logits.shape[0] / math.log(2.0)
