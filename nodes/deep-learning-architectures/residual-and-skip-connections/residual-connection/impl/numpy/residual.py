# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/residual-connection/ (ai://NEW-residual-connection).
"""A stack of residual MLP blocks, the NumPy reference.

Each block computes x + F(x), with the branch F(x) = W2 act(W1 x + b1) + b2 (eq-residual, eq-branch).
With residual=False the same blocks form a plain stack, x -> F(x).
The backward pass is written by hand (eq-backward); the tests check it against finite differences.
"""

import itertools

import numpy as np


def act(z, name):
    """The activation of the branch: relu or tanh."""
    if name == "tanh":
        return np.tanh(z)
    return np.maximum(z, 0.0)


def act_grad(z, name):
    """The derivative of the activation at z."""
    if name == "tanh":
        return 1.0 - np.tanh(z) ** 2
    return (z > 0.0).astype(np.float64)


def config(d, n, hidden=None, activation="relu", residual=True):
    """Return the configuration: width d, n blocks, branch width hidden, the activation, and the shortcut on or off."""
    if activation not in ("relu", "tanh"):
        raise ValueError(f"unknown activation {activation}")
    return {"d": d, "n": n, "hidden": hidden or d, "activation": activation, "residual": residual}


def init(rng, cfg, gain=1.0, zero_last=False):
    """Draw the weights of the n blocks.

    W1 has the He scale sqrt(2 / d) and W2 the scale gain / sqrt(hidden); the biases start at zero.
    zero_last sets the last layer of every branch to zero, so each block starts as the identity (Fixup rule 1).
    """
    d, m = cfg["d"], cfg["hidden"]
    params = []
    for _ in range(cfg["n"]):
        w1 = rng.normal(0.0, np.sqrt(2.0 / d), size=(m, d))
        w2 = rng.normal(0.0, gain / np.sqrt(m), size=(d, m))
        if zero_last:
            w2 = np.zeros((d, m))
        params.append({"W1": w1, "b1": np.zeros(m), "W2": w2, "b2": np.zeros(d)})
    return params


def branch(p, x, activation):
    """The residual branch on a batch x of shape (B, d). Return F(x) and the pre-activation z and activation h."""
    z = x @ p["W1"].T + p["b1"]  # eq: eq-branch
    h = act(z, activation)
    return h @ p["W2"].T + p["b2"], z, h


def forward(params, x, cfg):
    """Run the stack on x of shape (B, d). Return the output and the cache of the backward pass."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    cache = []
    for p in params:
        f, z, h = branch(p, x, cfg["activation"])
        cache.append((x, z, h))
        x = x + f if cfg["residual"] else f  # eq: eq-residual
    return x, cache


def backward(params, cache, dy, cfg):
    """Return (dx, grads) for the output gradient dy, block by block from the last one."""
    dx = np.atleast_2d(np.asarray(dy, dtype=np.float64))
    grads = []
    for p, (x, z, h) in zip(reversed(params), reversed(cache)):
        dz = (dx @ p["W2"]) * act_grad(z, cfg["activation"])
        grads.append({"W1": dz.T @ x, "b1": dz.sum(axis=0), "W2": dx.T @ h, "b2": dx.sum(axis=0)})
        through = dz @ p["W1"]
        dx = dx + through if cfg["residual"] else through  # eq: eq-backward, the shortcut passes dx unchanged
    return dx, grads[::-1]


def block_jacobian(p, x, activation):
    """The Jacobian J_F = W2 diag(act'(W1 x + b1)) W1 of one branch at one input vector x."""
    z = p["W1"] @ x + p["b1"]
    return p["W2"] @ (act_grad(z, activation)[:, None] * p["W1"])


def jacobian(params, x, cfg):
    """The Jacobian of the whole stack at one input vector x: the ordered product of the (I + J_i), or of the J_i when plain."""
    x = np.asarray(x, dtype=np.float64)
    eye = np.eye(x.size)
    total = eye
    for p in params:
        j = block_jacobian(p, x, cfg["activation"])
        total = ((eye + j) if cfg["residual"] else j) @ total  # eq: eq-jacobian
        f = branch(p, x[None, :], cfg["activation"])[0][0]
        x = x + f if cfg["residual"] else f
    return total


def path_terms(mats):
    """Expand the product of (I + A_i), with A_1 applied first, into one term per subset of the A_i (eq-paths).

    Return a dict from the tuple of picks (1 if the path goes through A_i) to the ordered product of the picked A_i.
    """
    d = mats[0].shape[0]
    terms = {}
    for picks in itertools.product((0, 1), repeat=len(mats)):
        term = np.eye(d)
        for a, pick in zip(mats, picks):
            if pick:
                term = a @ term
        terms[picks] = term
    return terms


def highway(p, x, activation, t=None, c=None, gate=None):
    """A highway layer y = H(x) T(x) + x C(x) (eq-highway), with H the branch of p.

    Give t and c to fix the gate values; otherwise T = sigmoid(x WT^T + bT) from gate, and C = 1 - T as in Srivastava et al.
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    h = branch(p, x, activation)[0]
    if t is None:
        t = 1.0 / (1.0 + np.exp(-(x @ gate["WT"].T + gate["bT"])))
    if c is None:
        c = 1.0 - t
    return h * t + x * c
