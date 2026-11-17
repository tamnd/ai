# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/gated-recurrent-unit/ (ai://NEW-gated-recurrent-unit).
"""One GRU layer, the NumPy reference.

It computes eq-gates, eq-candidate and eq-update of the page over a sequence, and backpropagates through time.
The three blocks are stacked in the order reset, update, candidate, the order of torch.nn.GRU, so that
W[:, k * H:(k + 1) * H] holds the input weights of block k. There are two biases, b_i on the input side and b_h on
the recurrent side, as in torch.nn.GRU.

reset="after" applies the reset gate to the recurrent projection of the candidate, r * (h U_n + b_hn), as
torch.nn.GRU and cuDNN do. reset="before" applies it to the state, (r * h) U_n + b_hn, as in eq. (8) of Cho et al.
(2014) and in the equations of Chung et al. (2014). The update keeps the old state when z = 1.
"""

import numpy as np

GATES = ("r", "z", "n")
RESETS = ("after", "before")


def config(input_size=3, hidden_size=4):
    """Return the config of the layer."""
    if min(input_size, hidden_size) < 1:
        raise ValueError("the sizes must be at least 1")
    return {"input_size": input_size, "hidden_size": hidden_size}


def init(rng, cfg):
    """Return the parameters W (I, 3H), U (H, 3H), b_i (3H,) and b_h (3H,), uniform in [-1/sqrt(H), 1/sqrt(H)]."""
    n_in, H = cfg["input_size"], cfg["hidden_size"]
    k = H**-0.5
    return {
        "W": rng.uniform(-k, k, (n_in, 3 * H)),  # (I, 3 * H)
        "U": rng.uniform(-k, k, (H, 3 * H)),  # (H, 3 * H)
        "b_i": rng.uniform(-k, k, 3 * H),  # (3 * H,)
        "b_h": rng.uniform(-k, k, 3 * H),  # (3 * H,)
    }


def sigmoid(z):
    """The logistic function, without overflow for large negative z."""
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def cell(params, x, h, reset="after"):
    """One step: return (h_new, step) for x (B, I) and h (B, H).

    step holds the gates r, z, n (B, H) each, the input projection a (B, 3H) and the recurrent projection m (B, 3H),
    whose last block is h U_n + b_hn with the reset after the matrix and (r * h) U_n + b_hn with it before.
    """
    if reset not in RESETS:
        raise ValueError(f"reset must be one of {RESETS}")
    H = h.shape[-1]
    W, U, b_i, b_h = params["W"], params["U"], params["b_i"], params["b_h"]
    a = x @ W + b_i  # (B, 3 * H)
    m_rz = h @ U[:, : 2 * H] + b_h[: 2 * H]  # (B, 2 * H)
    r = sigmoid(a[..., :H] + m_rz[..., :H])  # (B, H)
    z = sigmoid(a[..., H : 2 * H] + m_rz[..., H:])  # (B, H)
    if reset == "after":
        m_n = h @ U[:, 2 * H :] + b_h[2 * H :]  # (B, H)
        n = np.tanh(a[..., 2 * H :] + r * m_n)  # (B, H): eq-candidate, reset after the matrix
    else:
        m_n = (r * h) @ U[:, 2 * H :] + b_h[2 * H :]  # (B, H)
        n = np.tanh(a[..., 2 * H :] + m_n)  # (B, H): eq-candidate-cho, reset before the matrix
    h_new = (1.0 - z) * n + z * h  # (B, H): eq-update
    m = np.concatenate([m_rz, m_n], axis=-1)  # (B, 3 * H)
    return h_new, {"r": r, "z": z, "n": n, "a": a, "m": m}


def forward(params, x, h0=None, reset="after"):
    """Return (h, cache) for a batch of sequences x (B, T, I).

    h (B, T, H) is the state at every step. The cache holds the gates, the projections a and m (B, T, 3H) and the
    initial state, for backward and for the shape tests; cache["h_T"] is the final state.
    """
    B, T, _ = x.shape
    H = params["U"].shape[0]
    h_t = np.zeros((B, H)) if h0 is None else h0  # (B, H)
    cache = {"x": x, "h0": h_t, "reset": reset}
    hs = np.empty((B, T, H))  # (B, T, H)
    steps = {name: np.empty((B, T, H)) for name in GATES}  # (B, T, H) each
    steps |= {"a": np.empty((B, T, 3 * H)), "m": np.empty((B, T, 3 * H))}  # (B, T, 3 * H) each
    for t in range(T):
        h_t, step = cell(params, x[:, t], h_t, reset)
        hs[:, t] = h_t
        for name, v in step.items():
            steps[name][:, t] = v
    cache |= steps | {"h": hs, "h_T": h_t}
    return hs, cache


def backward(params, cache, dh, dh_T=None):
    """Backpropagation through time.

    dh (B, T, H) is the gradient of the loss with respect to the state of each step, and dh_T an extra gradient for
    the final state. Return (dx, dparams, dh0).
    """
    x, hs, reset = cache["x"], cache["h"], cache["reset"]
    B, T, _ = x.shape
    W, U = params["W"], params["U"]
    H = U.shape[0]
    grads = {k: np.zeros_like(v) for k, v in params.items()}
    dx = np.zeros_like(x)  # (B, T, I)
    dh_next = np.zeros((B, H)) if dh_T is None else dh_T.copy()  # (B, H)
    for t in reversed(range(T)):
        r, z, n = (cache[name][:, t] for name in GATES)  # (B, H) each
        m_n = cache["m"][:, t, 2 * H :]  # (B, H)
        h_prev = hs[:, t - 1] if t > 0 else cache["h0"]  # (B, H)
        g = dh[:, t] + dh_next  # (B, H): the gradient of h_t
        dz = g * (h_prev - n)  # (B, H)
        dan = g * (1.0 - z) * (1.0 - n**2)  # (B, H): the pre-activation of the candidate
        dh_prev = g * z  # (B, H): the direct path of eq-update
        if reset == "after":
            dr = dan * m_n  # (B, H)
            dm_n = dan * r  # (B, H)
            grads["U"][:, 2 * H :] += h_prev.T @ dm_n
            dh_prev += dm_n @ U[:, 2 * H :].T
        else:
            drh = dan @ U[:, 2 * H :].T  # (B, H): the gradient of r * h_{t-1}
            dr = drh * h_prev  # (B, H)
            dm_n = dan  # (B, H)
            grads["U"][:, 2 * H :] += (r * h_prev).T @ dm_n
            dh_prev += drh * r
        grads["b_h"][2 * H :] += dm_n.sum(axis=0)
        dar = dr * r * (1.0 - r)  # (B, H)
        daz = dz * z * (1.0 - z)  # (B, H)
        drz = np.concatenate([dar, daz], axis=-1)  # (B, 2 * H): the pre-activations of r and z
        grads["U"][:, : 2 * H] += h_prev.T @ drz
        grads["b_h"][: 2 * H] += drz.sum(axis=0)
        dh_prev += drz @ U[:, : 2 * H].T
        da = np.concatenate([drz, dan], axis=-1)  # (B, 3 * H)
        grads["W"] += x[:, t].T @ da
        grads["b_i"] += da.sum(axis=0)
        dx[:, t] = da @ W.T
        dh_next = dh_prev
    return dx, grads, dh_next


def param_count(cfg):
    """The number of parameters, from the sizes alone."""
    n_in, H = cfg["input_size"], cfg["hidden_size"]
    return 3 * H * n_in + 3 * H * H + 6 * H


def flops(cfg, B, T):
    """FLOPs of forward with a multiply-add as 2, without the activations and the elementwise products."""
    n_in, H = cfg["input_size"], cfg["hidden_size"]
    return T * (2 * B * n_in * 3 * H + 2 * B * H * 3 * H)


def lstm_param_count(cfg, peepholes=False):
    """The parameters of an LSTM layer of the same sizes with one bias per gate, and optionally 3 diagonal peepholes."""
    n_in, H = cfg["input_size"], cfg["hidden_size"]
    return 4 * H * (n_in + H + 1) + (3 * H if peepholes else 0)


def tanh_param_count(cfg):
    """The parameters of a tanh recurrent layer h_t = tanh(x_t W + h_{t-1} U + b) of the same sizes."""
    n_in, H = cfg["input_size"], cfg["hidden_size"]
    return H * (n_in + H + 1)
