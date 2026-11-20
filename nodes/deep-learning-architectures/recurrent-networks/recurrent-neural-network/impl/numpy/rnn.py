# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/recurrent-neural-network/ (ai://FKJKTR).
"""One simple (Elman) recurrent layer, the NumPy reference.

It computes eq-step of the page, h_t = f(x_t W + b_i + h_{t-1} U + b_h), over a sequence, and backpropagates
through time by hand. The parameters are W (I, H), U (H, H) and the two biases b_i and b_h of torch.nn.RNN, whose
weight_ih_l0 and weight_hh_l0 are W.T and U.T. The nonlinearity f is tanh or relu, as in torch.nn.RNN.
"""

import numpy as np

NONLINEARITIES = ("tanh", "relu")


def config(input_size=3, hidden_size=4, nonlinearity="tanh"):
    """Return the config of the layer."""
    if min(input_size, hidden_size) < 1:
        raise ValueError("the sizes must be at least 1")
    if nonlinearity not in NONLINEARITIES:
        raise ValueError(f"nonlinearity must be one of {NONLINEARITIES}")
    return {"input_size": input_size, "hidden_size": hidden_size, "nonlinearity": nonlinearity}


def init(rng, cfg):
    """Return W (I, H), U (H, H), b_i (H,) and b_h (H,), uniform in [-1/sqrt(H), 1/sqrt(H)] as in torch.nn.RNN."""
    n_in, H = cfg["input_size"], cfg["hidden_size"]
    k = H**-0.5
    return {
        "W": rng.uniform(-k, k, (n_in, H)),  # (I, H)
        "U": rng.uniform(-k, k, (H, H)),  # (H, H)
        "b_i": rng.uniform(-k, k, H),  # (H,)
        "b_h": rng.uniform(-k, k, H),  # (H,)
    }


def irnn_init(rng, cfg, scale=1.0, std=0.001):
    """The IRNN of Le et al. (2015): U = scale * I, zero biases, input weights normal with standard deviation std."""
    n_in, H = cfg["input_size"], cfg["hidden_size"]
    return {
        "W": rng.normal(0.0, std, (n_in, H)),  # (I, H)
        "U": scale * np.eye(H),  # (H, H)
        "b_i": np.zeros(H),  # (H,)
        "b_h": np.zeros(H),  # (H,)
    }


def act(a, nonlinearity):
    """The nonlinearity f of eq-step."""
    if nonlinearity == "tanh":
        return np.tanh(a)
    if nonlinearity == "relu":
        return np.maximum(a, 0.0)
    raise ValueError(f"nonlinearity must be one of {NONLINEARITIES}")


def act_grad(a, h, nonlinearity):
    """f'(a), from the pre-activation a and the output h = f(a); the derivative of relu at 0 is taken as 0."""
    if nonlinearity == "tanh":
        return 1.0 - h * h
    if nonlinearity == "relu":
        return (a > 0).astype(a.dtype)
    raise ValueError(f"nonlinearity must be one of {NONLINEARITIES}")


def cell(params, x, h, nonlinearity="tanh"):
    """One step of eq-step: return (h_new, a) for x (B, I) and h (B, H), where a (B, H) is the pre-activation."""
    a = x @ params["W"] + params["b_i"] + h @ params["U"] + params["b_h"]  # (B, H)
    return act(a, nonlinearity), a


def forward(params, x, h0=None, nonlinearity="tanh"):
    """Return (h, cache) for x (B, T, I): h (B, T, H) is the state of every step; cache holds what backward needs."""
    B, T, _ = x.shape
    H = params["U"].shape[0]
    h_prev = np.zeros((B, H)) if h0 is None else h0  # (B, H)
    hs, As = np.zeros((B, T, H)), np.zeros((B, T, H))
    for t in range(T):
        h_prev, a = cell(params, x[:, t], h_prev, nonlinearity)
        hs[:, t], As[:, t] = h_prev, a
    cache = {"x": x, "h0": np.zeros((B, H)) if h0 is None else h0, "h": hs, "a": As, "h_T": hs[:, -1]}
    cache["nonlinearity"] = nonlinearity
    return hs, cache


def backward(params, cache, dh, dh_T=None):
    """Backpropagation through time.

    dh (B, T, H) is the gradient of the loss with respect to every state, dh_T (B, H) an extra gradient on the last
    one. Return (dx, grads, dh0): the gradients with respect to the inputs, the parameters and the initial state.
    """
    x, hs, As, nl = cache["x"], cache["h"], cache["a"], cache["nonlinearity"]
    B, T, _ = x.shape
    W, U = params["W"], params["U"]
    grads = {k: np.zeros_like(v) for k, v in params.items()}
    dx = np.zeros_like(x)
    carry = np.zeros_like(hs[:, 0]) if dh_T is None else dh_T.copy()  # (B, H): dL/dh_t from later steps
    for t in reversed(range(T)):
        g = dh[:, t] + carry  # (B, H): the whole dL/dh_t
        da = g * act_grad(As[:, t], hs[:, t], nl)  # (B, H)
        h_prev = cache["h0"] if t == 0 else hs[:, t - 1]
        grads["W"] += x[:, t].T @ da  # (I, H): the shared weights collect one term per step
        grads["U"] += h_prev.T @ da  # (H, H)
        grads["b_i"] += da.sum(0)
        grads["b_h"] += da.sum(0)
        dx[:, t] = da @ W.T  # (B, I)
        carry = da @ U.T  # (B, H): dL/dh_{t-1} through the step
    return dx, grads, carry


def step_jacobian(params, a, h, nonlinearity="tanh"):
    """The Jacobian dh_t/dh_{t-1} (H, H) of one step for one sequence: diag(f'(a_t)) U^T, row i for unit i of h_t."""
    return act_grad(a, h, nonlinearity)[:, None] * params["U"].T


def param_count(cfg):
    """The parameters of one layer in the form of torch.nn.RNN: H I + H^2 + 2 H."""
    n_in, H = cfg["input_size"], cfg["hidden_size"]
    return H * n_in + H * H + 2 * H


def one_bias_param_count(cfg):
    """The same layer with a single bias, H I + H^2 + H; the two biases of torch.nn.RNN only enter as a sum."""
    n_in, H = cfg["input_size"], cfg["hidden_size"]
    return H * n_in + H * H + H


def flops(cfg, B, T):
    """FLOPs of a forward pass, a multiply-add as 2, the two products only: 2 B T H (I + H)."""
    n_in, H = cfg["input_size"], cfg["hidden_size"]
    return 2 * B * T * H * (n_in + H)
