# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/grokking/ (ai://NEW-grokking).
"""Grokking in NumPy: the modular-addition task, the Fourier algorithm that solves it, and a small model to train on it.

modular_addition_split builds the table a + b mod p and splits its p * p rows at random into a training and a test
set; fourier_logits computes the logit of each answer c as sum_k alpha_k cos(w_k (a + b - c)) with w_k = 2 pi k / p,
going through the trig identities of eq-sum-angle and eq-read-off the way the reverse-engineered transformer does;
decay_only applies the decoupled weight decay of AdamW with zero gradient, theta <- (1 - lr * wd) theta (eq-shrink);
init_mlp, forward, loss_and_grads and adamw_step are an embedding plus one-hidden-layer ReLU network with a
hand-written backward pass and the AdamW update, small enough to train on a CPU for a tiny p; demo returns the numbers
of the worked example of the page."""

import numpy as np

KEY_FREQUENCIES = (14, 35, 41, 42, 52)  # the five key frequencies of the mainline p = 113 model


def modular_addition_split(p, train_fraction, rng):
    """All p * p pairs (a, b) with label (a + b) mod p, split at random.

    Returns (train_x, train_y, test_x, test_y); x has shape (n, 2) of ints, y shape (n,). The training set has
    round(train_fraction * p * p) rows and the test set the rest."""
    if p < 2:
        raise ValueError("p must be at least 2")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be strictly between 0 and 1")
    a, b = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
    x = np.stack([a.ravel(), b.ravel()], axis=1)
    y = (x[:, 0] + x[:, 1]) % p
    order = rng.permutation(p * p)
    n_train = int(round(train_fraction * p * p))
    tr, te = order[:n_train], order[n_train:]
    return x[tr], y[tr], x[te], y[te]


def fourier_logits(a, b, p, freqs=KEY_FREQUENCIES, alpha=None):
    """Logits of shape (n, p): logit[c] = sum_k alpha_k cos(w_k (a + b - c)), w_k = 2 pi k / p.

    Computed in three steps: embed a and b as cos and sin of w_k a and w_k b; compose them into cos and sin of
    w_k (a + b) with the angle-sum identities (eq-sum-angle); read off cos(w_k (a + b - c)) against cos(w_k c) and
    sin(w_k c) (eq-read-off)."""
    a = np.atleast_1d(np.asarray(a))
    b = np.atleast_1d(np.asarray(b))
    w = 2.0 * np.pi * np.asarray(freqs, dtype=np.float64) / p  # (K,)
    alpha = np.ones_like(w) if alpha is None else np.asarray(alpha, dtype=np.float64)
    ca, sa = np.cos(np.outer(a, w)), np.sin(np.outer(a, w))  # (n, K)
    cb, sb = np.cos(np.outer(b, w)), np.sin(np.outer(b, w))
    cab = ca * cb - sa * sb  # cos(w (a + b)), eq-sum-angle
    sab = sa * cb + ca * sb  # sin(w (a + b))
    c = np.arange(p)
    cc, sc = np.cos(np.outer(w, c)), np.sin(np.outer(w, c))  # (K, p)
    return (cab * alpha) @ cc + (sab * alpha) @ sc  # eq-read-off, summed over k


def accuracy(logits, labels):
    """The fraction of rows whose largest logit is at the label."""
    return float(np.mean(np.argmax(logits, axis=1) == np.asarray(labels)))


def decay_only(theta, lr, wd, steps):
    """theta after `steps` AdamW updates with zero gradient: each step multiplies by (1 - lr * wd) (eq-shrink)."""
    theta = np.asarray(theta, dtype=np.float64)
    for _ in range(steps):
        theta = theta * (1.0 - lr * wd)
    return theta


def init_mlp(rng, p, d=16, hidden=64, scale=1.0):
    """Embedding E (p, d) shared by both inputs, W1 (2d, hidden), W2 (hidden, p); scale multiplies every weight."""
    return {
        "E": scale * rng.normal(0.0, 1.0, (p, d)),
        "W1": scale * rng.normal(0.0, 1.0 / np.sqrt(2 * d), (2 * d, hidden)),
        "W2": scale * rng.normal(0.0, 1.0 / np.sqrt(hidden), (hidden, p)),
    }


def forward(params, x):
    """Logits (n, p) of the pairs x (n, 2): h = relu([E[a], E[b]] W1), logits = h W2."""
    z = np.concatenate([params["E"][x[:, 0]], params["E"][x[:, 1]]], axis=1)
    h = np.maximum(z @ params["W1"], 0.0)
    return h @ params["W2"]


def loss_and_grads(params, x, y):
    """Mean softmax cross-entropy of the labels y and its gradient with respect to each parameter."""
    E, W1, W2 = params["E"], params["W1"], params["W2"]
    n, d = x.shape[0], E.shape[1]
    z = np.concatenate([E[x[:, 0]], E[x[:, 1]]], axis=1)
    pre = z @ W1
    h = np.maximum(pre, 0.0)
    logits = h @ W2
    m = logits.max(axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(logits - m).sum(axis=1))
    loss = float(np.mean(lse - logits[np.arange(n), y]))
    g = np.exp(logits - lse[:, None])
    g[np.arange(n), y] -= 1.0
    g /= n  # d loss / d logits
    gW2 = h.T @ g
    gpre = (g @ W2.T) * (pre > 0)
    gW1 = z.T @ gpre
    gz = gpre @ W1.T
    gE = np.zeros_like(E)
    np.add.at(gE, x[:, 0], gz[:, :d])
    np.add.at(gE, x[:, 1], gz[:, d:])
    return loss, {"E": gE, "W1": gW1, "W2": gW2}


def adamw_init(params):
    """Zero first and second moments and step count 0."""
    return {
        "t": 0,
        "m": {k: np.zeros_like(v) for k, v in params.items()},
        "v": {k: np.zeros_like(v) for k, v in params.items()},
    }


def adamw_step(params, grads, state, lr=1e-3, wd=1.0, betas=(0.9, 0.98), eps=1e-8):
    """One AdamW update in the order of torch.optim.AdamW: decay theta by (1 - lr * wd), then the Adam step.

    The defaults are the hyperparameters of the original grokking runs (learning rate 1e-3, weight decay 1,
    beta2 = 0.98)."""
    b1, b2 = betas
    t = state["t"] + 1
    new_p, new_m, new_v = {}, {}, {}
    for k, theta in params.items():
        g = grads[k]
        m = b1 * state["m"][k] + (1.0 - b1) * g
        v = b2 * state["v"][k] + (1.0 - b2) * g * g
        mhat = m / (1.0 - b1**t)
        vhat = v / (1.0 - b2**t)
        new_p[k] = theta * (1.0 - lr * wd) - lr * mhat / (np.sqrt(vhat) + eps)
        new_m[k], new_v[k] = m, v
    return new_p, {"t": t, "m": new_m, "v": new_v}


def weight_norm(params):
    """The l2 norm of all parameters together."""
    return float(np.sqrt(sum(np.sum(v * v) for v in params.values())))


def demo(rng):
    """The worked example: the Fourier algorithm at p = 113 with the five key frequencies, and pure weight decay."""
    p = 113
    a, b = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
    a, b = a.ravel(), b.ravel()
    logits = fourier_logits(a, b, p)
    labels = (a + b) % p
    correct = logits[np.arange(p * p), labels]
    wrong = logits.copy()
    wrong[np.arange(p * p), labels] = -np.inf
    all_k = fourier_logits(a, b, p, freqs=range(1, (p - 1) // 2 + 1))
    tx, ty, sx, sy = modular_addition_split(p, 0.3, rng)
    return {
        "p": p,
        "pairs": p * p,
        "train": len(ty),
        "test": len(sy),
        "accuracy": accuracy(logits, labels),
        "correct_logit": float(correct.max()),
        "largest_wrong_logit": float(wrong.max()),
        "all_freq_wrong_logit": float(np.delete(all_k[0], 0).max()),
        "decay_1000_steps": float(decay_only(1.0, 1e-3, 1.0, 1000)),
    }
