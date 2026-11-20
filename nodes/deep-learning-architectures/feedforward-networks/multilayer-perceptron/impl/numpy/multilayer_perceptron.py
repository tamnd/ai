# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/multilayer-perceptron/ (ai://YA59M7).
"""The multilayer perceptron, the NumPy reference: forward, backward and one step of SGD.

It computes eq-layer, eq-output, eq-loss, eq-delta, eq-grads and eq-sgd of the page, and initializes with
eq-he or eq-glorot. A network has n_hidden hidden layers of width d_hidden and a linear output layer. Each
weight matrix W_l has the shape (n_out, n_in), as on the page, and a batch is a matrix with one example per
row, so a layer computes h @ W_l.T + b_l.
"""

import math

import numpy as np

_erf = np.frompyfunc(math.erf, 1, 1)


def _gelu(z):
    return z * 0.5 * (1.0 + _erf(z / math.sqrt(2.0)).astype(z.dtype))


def _gelu_grad(z):
    phi = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return 0.5 * (1.0 + _erf(z / math.sqrt(2.0)).astype(z.dtype)) + z * phi


def _sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


# Each activation is a pair: the function and its derivative, both as functions of the pre-activation z.
ACTIVATIONS = {
    "relu": (lambda z: np.maximum(z, 0.0), lambda z: (z > 0).astype(z.dtype)),
    "tanh": (np.tanh, lambda z: 1.0 - np.tanh(z) ** 2),
    "sigmoid": (_sigmoid, lambda z: _sigmoid(z) * (1.0 - _sigmoid(z))),
    "gelu": (_gelu, _gelu_grad),
    "identity": (lambda z: z, np.ones_like),
}


def config(d_in=2, d_hidden=4, d_out=1, n_hidden=1, activation="relu"):
    """Return the config: the input, hidden and output widths, the number of hidden layers and the activation."""
    if min(d_in, d_hidden, d_out, n_hidden) < 1 or activation not in ACTIVATIONS:
        raise ValueError("the widths and n_hidden must be at least 1, and the activation one of ACTIVATIONS")
    return {"d_in": d_in, "d_hidden": d_hidden, "d_out": d_out, "n_hidden": n_hidden, "activation": activation}


def widths(cfg):
    """The widths n_0, ..., n_{L+1} of the layers, from the input to the output."""
    return [cfg["d_in"]] + [cfg["d_hidden"]] * cfg["n_hidden"] + [cfg["d_out"]]


def param_count(cfg):
    """The number of weights and biases: the sum over the layers of n_l n_{l+1} + n_{l+1}."""
    n = widths(cfg)
    return sum(a * b + b for a, b in zip(n[:-1], n[1:], strict=True))


def flops(cfg, B):
    """The FLOPs of one forward pass over a batch of B examples, a multiply-add counted as 2; biases and activations are left out."""
    n = widths(cfg)
    return 2 * B * sum(a * b for a, b in zip(n[:-1], n[1:], strict=True))


def init(rng, cfg, scheme="he", dtype=np.float64):
    """Draw the weights with eq-he (Gaussian, variance 2 / n_in) or eq-glorot (uniform, variance 2 / (n_in + n_out)); zero biases."""
    n = widths(cfg)
    params = {"W": [], "b": []}
    for n_in, n_out in zip(n[:-1], n[1:], strict=True):
        if scheme == "he":  # eq: eq-he
            W = rng.normal(0.0, math.sqrt(2.0 / n_in), size=(n_out, n_in))
        elif scheme == "glorot":  # eq: eq-glorot
            a = math.sqrt(6.0 / (n_in + n_out))
            W = rng.uniform(-a, a, size=(n_out, n_in))
        else:
            raise ValueError("scheme is he or glorot")
        params["W"].append(W.astype(dtype))
        params["b"].append(np.zeros(n_out, dtype=dtype))
    return params


def forward(params, x, cfg):
    """Map x of shape (B, I) to y of shape (B, O). The cache keeps the input and pre-activation of each layer."""
    act, _ = ACTIVATIONS[cfg["activation"]]
    W, b = params["W"], params["b"]
    h = x  # (B, I)
    cache = {"h": [], "z": [], "activation": cfg["activation"]}
    for l in range(len(W) - 1):  # noqa: E741
        z = h @ W[l].T + b[l]  # (B, H)  eq: eq-layer
        cache["h"].append(h)
        cache["z"].append(z)
        h = act(z)  # (B, H)
    y = h @ W[-1].T + b[-1]  # (B, O)  eq: eq-output
    cache["h"].append(h)
    return y, cache


def backward(params, cache, dy):
    """Given dy = dL/dy of shape (B, O), return dL/dx of shape (B, I) and the gradients of every W_l and b_l."""
    _, act_grad = ACTIVATIONS[cache["activation"]]
    W = params["W"]
    L = len(W)
    dW, db = [None] * L, [None] * L
    delta = dy  # (B, O), the error of the output layer
    for l in reversed(range(L)):  # noqa: E741
        dW[l] = delta.T @ cache["h"][l]  # eq: eq-grads
        db[l] = delta.sum(axis=0)
        dh = delta @ W[l]  # (B, n_l)
        if l > 0:
            delta = dh * act_grad(cache["z"][l - 1])  # eq: eq-delta
    return dh, {"W": dW, "b": db}


def mse_loss(y, t):
    """The loss of eq-loss, half the mean over the batch of the squared error, and its gradient for y."""
    B = y.shape[0]
    r = y - t  # (B, O)
    return 0.5 * float(np.sum(r * r)) / B, r / B  # eq: eq-loss


def sgd_step(params, grads, lr):
    """One step of gradient descent, eq-sgd; it returns new parameters and leaves the old ones alone."""
    return {k: [p - lr * g for p, g in zip(params[k], grads[k], strict=True)] for k in ("W", "b")}  # eq: eq-sgd


def xor_params(dtype=np.float64):
    """The 2-2-1 ReLU network of the worked example, which computes XOR on {0, 1}^2."""
    return {
        "W": [np.array([[1.0, 1.0], [1.0, 1.0]], dtype=dtype), np.array([[1.0, -2.0]], dtype=dtype)],
        "b": [np.array([0.0, -1.0], dtype=dtype), np.array([0.0], dtype=dtype)],
    }


def train(params, x, t, cfg, lr, steps):
    """Full-batch gradient descent on eq-loss; returns the final parameters and the loss before each step."""
    losses = []
    for _ in range(steps):
        y, cache = forward(params, x, cfg)
        loss, dy = mse_loss(y, t)
        losses.append(loss)
        _, grads = backward(params, cache, dy)
        params = sgd_step(params, grads, lr)
    return params, losses
