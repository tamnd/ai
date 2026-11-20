# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/fine-tuning/ (ai://NEW-fine-tuning).
"""The arithmetic of fine-tuning a pre-trained model, in NumPy.

Fine-tuning starts from pre-trained weights, usually adds a new head, and trains on labelled data for the target task.
This file has:

* a two-layer linear network y = v^T B x, trained by full fine-tuning, by linear probing (B frozen), and by linear
  probing then fine-tuning (LP-FT), to show how fine-tuning changes features only along the training inputs;
* ULMFiT's schedules: slanted triangular learning rates, discriminative per-layer rates and gradual unfreezing;
* BERT's linear warmup and decay, and Adam with and without bias correction;
* the elastic weight consolidation penalty and the auxiliary language-modelling objective.
"""

import math

import numpy as np

# A two-layer linear network: features B x (k of them), prediction v^T B x.


def predict(v, B, X):
    """Predictions for the rows of X (n, d), with B (k, d) and v (k,)."""
    return X @ B.T @ v


def loss(v, B, X, Y):
    r = predict(v, B, X) - Y
    return 0.5 * float(r @ r) / len(Y)


def grads(v, B, X, Y):
    """Gradients of loss: v gets B X^T r / n, and B gets v r^T X / n, a matrix whose rows lie in the span of the
    training inputs."""
    r = predict(v, B, X) - Y
    return B @ X.T @ r / len(Y), np.outer(v, r @ X) / len(Y)


def fine_tune(v, B, X, Y, lr=0.01, steps=20_000, train_features=True):
    """Gradient descent on the head v and, if train_features, on the feature extractor B too."""
    v, B = v.astype(float).copy(), B.astype(float).copy()
    for _ in range(steps):
        gv, gB = grads(v, B, X, Y)
        v -= lr * gv
        if train_features:
            B -= lr * gB
    return v, B


def linear_probe(B, X, Y):
    """The minimum-norm least-squares head on frozen features B x."""
    return np.linalg.lstsq(X @ B.T, Y, rcond=None)[0]


def lp_ft(B, X, Y, lr=0.01, steps=20_000):
    """Linear probing, then full fine-tuning from the probed head."""
    return fine_tune(linear_probe(B, X, Y), B, X, Y, lr, steps)


# ULMFiT's schedules.


def stlr(t, T, cut_frac=0.1, ratio=32, eta_max=0.01):
    """Slanted triangular learning rate at iteration t of T: a short linear rise from eta_max / ratio to eta_max over
    the first cut_frac of training, then a long linear decay back to eta_max / ratio."""
    cut = math.floor(T * cut_frac)
    p = t / cut if t < cut else 1 - (t - cut) / (cut * (1 / cut_frac - 1))
    return eta_max * (1 + p * (ratio - 1)) / ratio


def discriminative_lrs(eta_top, n_layers, factor=2.6):
    """Learning rates from the lowest layer to the top one, each layer's rate the one above divided by factor."""
    return [eta_top / factor ** (n_layers - 1 - layer) for layer in range(n_layers)]


def gradual_unfreezing(n_layers, n_epochs):
    """For each epoch, the indices of trainable layers (0 is the lowest): the top layer first, one more per epoch."""
    return [list(range(max(n_layers - 1 - e, 0), n_layers)) for e in range(n_epochs)]


# BERT-style schedules and Adam.


def warmup_linear(t, T, peak, warmup_frac=0.1):
    """Linear warmup from 0 to peak over the first warmup_frac of T steps, then linear decay to 0 at T."""
    w = warmup_frac * T
    return peak * t / w if t < w else peak * max(T - t, 0) / (T - w)


def bias_correction_ratio(t, beta1=0.9, beta2=0.999):
    """How many times larger step t of Adam is without bias correction, for a constant gradient: the uncorrected
    step is (1 - beta1^t) / sqrt(1 - beta2^t) times lr, the corrected one lr."""
    return (1 - beta1**t) / math.sqrt(1 - beta2**t)


def adam(grad_fn, theta, lr, steps, beta1=0.9, beta2=0.999, eps=1e-8, bias_correction=True, weight_decay=0.0):
    """Adam (AdamW with decoupled weight decay if weight_decay > 0). Without bias correction this is BERTAdam.
    Returns the list of iterates, starting with theta."""
    theta = np.asarray(theta, dtype=float).copy()
    m, s = np.zeros_like(theta), np.zeros_like(theta)
    out = [theta.copy()]
    for t in range(1, steps + 1):
        g = grad_fn(theta)
        m = beta1 * m + (1 - beta1) * g
        s = beta2 * s + (1 - beta2) * g * g
        mh, sh = (m / (1 - beta1**t), s / (1 - beta2**t)) if bias_correction else (m, s)
        theta = theta - lr * (mh / (np.sqrt(sh) + eps) + weight_decay * theta)
        out.append(theta.copy())
    return out


# Limiting forgetting.


def ewc_penalty(theta, theta_old, fisher, lam):
    """(lam / 2) sum_i F_i (theta_i - theta_old_i)^2."""
    d = np.asarray(theta) - np.asarray(theta_old)
    return 0.5 * lam * float(np.sum(np.asarray(fisher) * d * d))


def ewc_grad(theta, theta_old, fisher, lam):
    return lam * np.asarray(fisher) * (np.asarray(theta) - np.asarray(theta_old))


def ewc_quadratic_optimum(a, b, theta_old, fisher, lam):
    """Minimizer of (a/2)(theta - b)^2 + (lam/2) F (theta - theta_old)^2, coordinate by coordinate: a new-task loss
    with curvature a and minimum b, pulled towards the old value with strength lam * F."""
    a, f = np.asarray(a, dtype=float), lam * np.asarray(fisher, dtype=float)
    return (a * np.asarray(b) + f * np.asarray(theta_old)) / (a + f)


def auxiliary_objective(task_loss, lm_loss, lam=0.5):
    """GPT's fine-tuning objective L3 = L2 + lam * L1, written as losses to minimize."""
    return task_loss + lam * lm_loss
