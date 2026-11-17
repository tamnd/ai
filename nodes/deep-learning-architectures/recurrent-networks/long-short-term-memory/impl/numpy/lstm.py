# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/long-short-term-memory/ (ai://YVW8D3).
"""One LSTM layer with a forget gate and no peepholes, the NumPy reference.

It computes eq-gates, eq-cell and eq-hidden of the page over a sequence, and backpropagates through time.
The four gates are stacked in the order input, forget, cell candidate, output, the order of torch.nn.LSTM,
so that W[:, k * H:(k + 1) * H] holds the input weights of gate k.
"""

import numpy as np

GATES = ("i", "f", "g", "o")


def config(input_size=3, hidden_size=4):
    """Return the config of the layer."""
    if min(input_size, hidden_size) < 1:
        raise ValueError("the sizes must be at least 1")
    return {"input_size": input_size, "hidden_size": hidden_size}


def init(rng, cfg, forget_bias=0.0):
    """Return the parameters: input weights W, recurrent weights U and one bias b, for the four gates.

    The weights are uniform in [-1/sqrt(H), 1/sqrt(H)], as in torch.nn.LSTM. forget_bias is added to the
    bias of the forget gate, which starts the cells near remembering when it is positive.
    """
    n_in, H = cfg["input_size"], cfg["hidden_size"]
    k = H**-0.5
    W = rng.uniform(-k, k, (n_in, 4 * H))  # (I, 4 * H)
    U = rng.uniform(-k, k, (H, 4 * H))  # (H, 4 * H)
    b = rng.uniform(-k, k, 4 * H)  # (4 * H,)
    b[H : 2 * H] += forget_bias
    return {"W": W, "U": U, "b": b}


def sigmoid(z):
    """The logistic function, without overflow for large negative z."""
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def cell(params, x, h, c):
    """One step: return (h_new, c_new, gates) for x (B, I), h (B, H) and c (B, H)."""
    H = h.shape[-1]
    z = x @ params["W"] + h @ params["U"] + params["b"]  # (B, 4 * H)
    i = sigmoid(z[..., :H])  # (B, H)
    f = sigmoid(z[..., H : 2 * H])  # (B, H)
    g = np.tanh(z[..., 2 * H : 3 * H])  # (B, H)
    o = sigmoid(z[..., 3 * H :])  # (B, H)
    c_new = f * c + i * g  # (B, H): the additive update of eq-cell
    h_new = o * np.tanh(c_new)  # (B, H)
    return h_new, c_new, {"i": i, "f": f, "g": g, "o": o}


def forward(params, x, h0=None, c0=None):
    """Return (h, cache) for a batch of sequences x (B, T, I).

    h (B, T, H) is the hidden state at every step. The cache holds the gates, the cell states and the
    initial state, for backward and for the shape tests; cache["h_T"] and cache["c_T"] are the final state.
    """
    B, T, _ = x.shape
    H = params["U"].shape[0]
    h_t = np.zeros((B, H)) if h0 is None else h0  # (B, H)
    c_t = np.zeros((B, H)) if c0 is None else c0  # (B, H)
    hs, cs = np.empty((B, T, H)), np.empty((B, T, H))  # (B, T, H), (B, T, H)
    gates = {name: np.empty((B, T, H)) for name in GATES}  # (B, T, H) each
    for t in range(T):
        h_t, c_t, gs = cell(params, x[:, t], h_t, c_t)
        hs[:, t], cs[:, t] = h_t, c_t
        for name in GATES:
            gates[name][:, t] = gs[name]
    cache = {
        "x": x,
        "h0": np.zeros((B, H)) if h0 is None else h0,
        "c0": np.zeros((B, H)) if c0 is None else c0,
    }
    cache |= {"h": hs, "c": cs, "h_T": h_t, "c_T": c_t} | gates
    return hs, cache


def backward(params, cache, dh, dh_T=None, dc_T=None):
    """Backpropagation through time.

    dh (B, T, H) is the gradient of the loss with respect to the hidden state of each step, and dh_T and dc_T
    are extra gradients for the final state. Return (dx, dparams, dh0, dc0).
    """
    x, hs, cs = cache["x"], cache["h"], cache["c"]
    B, T, _ = x.shape
    H = params["U"].shape[0]
    dW, dU, db = (
        np.zeros_like(params["W"]),
        np.zeros_like(params["U"]),
        np.zeros_like(params["b"]),
    )
    dx = np.zeros_like(x)  # (B, T, I)
    dh_next = np.zeros((B, H)) if dh_T is None else dh_T.copy()  # (B, H)
    dc_next = np.zeros((B, H)) if dc_T is None else dc_T.copy()  # (B, H)
    for t in reversed(range(T)):
        i, f, g, o = (cache[name][:, t] for name in GATES)  # (B, H) each
        c_prev = cs[:, t - 1] if t > 0 else cache["c0"]  # (B, H)
        h_prev = hs[:, t - 1] if t > 0 else cache["h0"]  # (B, H)
        tc = np.tanh(cs[:, t])  # (B, H)
        dh_t = dh[:, t] + dh_next  # (B, H)
        dc = dc_next + dh_t * o * (1.0 - tc**2)  # (B, H)
        dz = np.concatenate(
            [
                dc * g * i * (1.0 - i),
                dc * c_prev * f * (1.0 - f),
                dc * i * (1.0 - g**2),
                dh_t * tc * o * (1.0 - o),
            ],
            axis=-1,
        )  # (B, 4 * H)
        dW += x[:, t].T @ dz
        dU += h_prev.T @ dz
        db += dz.sum(axis=0)
        dx[:, t] = dz @ params["W"].T
        dh_next = dz @ params["U"].T  # (B, H)
        dc_next = dc * f  # (B, H): the cell path of eq-cell-jacobian, a product with f only
    return dx, {"W": dW, "U": dU, "b": db}, dh_next, dc_next


def param_count(cfg):
    """The number of parameters, from the sizes alone."""
    n_in, H = cfg["input_size"], cfg["hidden_size"]
    return 4 * H * n_in + 4 * H * H + 4 * H


def flops(cfg, B, T):
    """FLOPs of forward with a multiply-add as 2, without the activations and the elementwise products."""
    n_in, H = cfg["input_size"], cfg["hidden_size"]
    return T * (2 * B * n_in * 4 * H + 2 * B * H * 4 * H)
