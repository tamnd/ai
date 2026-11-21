# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/contrastive-language-image-pretraining/ (ai://NEW-contrastive-language-image-pretraining).
"""Contrastive and sigmoid image-text losses with their gradients, a linear dual encoder trained on paired
synthetic data, and zero-shot classification from class captions.

The data are two views of a shared latent: an "image" x = A z + noise and a "caption" y = B z + noise. Two linear
encoders map each view to a unit vector. The symmetric softmax loss of CLIP and the pairwise sigmoid loss of SigLIP
are written with explicit gradients, including the gradient through the L2 normalization and through the
log-parameterized logit scale, so that plain gradient descent trains the encoders.
"""

import numpy as np

LOG_INIT_SCALE = float(np.log(1 / 0.07))  # CLIP's initial temperature 0.07
LOG_MAX_SCALE = float(np.log(100.0))  # CLIP clips the logit scale at 100


def normalize(e):
    n = np.linalg.norm(e, axis=1, keepdims=True)
    return e / n, n


def normalize_backward(u, n, du):
    """Gradient through u = e / |e|: (du - u (u . du)) / |e|."""
    return (du - u * np.sum(u * du, axis=1, keepdims=True)) / n


def log_softmax(z, axis):
    z = z - z.max(axis=axis, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=axis, keepdims=True))


def clip_loss(u, v, log_s):
    """Symmetric cross-entropy over the N x N logits s u v^T. Returns the loss and its gradients."""
    n = len(u)
    s = np.exp(log_s)
    sim = u @ v.T
    z = s * sim
    lp_rows = log_softmax(z, axis=1)  # each image picks a caption
    lp_cols = log_softmax(z, axis=0)  # each caption picks an image
    loss = -(np.trace(lp_rows) + np.trace(lp_cols)) / (2 * n)
    eye = np.eye(n)
    dz = ((np.exp(lp_rows) - eye) + (np.exp(lp_cols) - eye)) / (2 * n)
    du = s * dz @ v
    dv = s * dz.T @ u
    dlog_s = float(np.sum(dz * sim) * s)
    return float(loss), du, dv, dlog_s


def log_sigmoid(x):
    return -np.logaddexp(0.0, -x)


def sigmoid_loss(u, v, log_t, b):
    """SigLIP loss -(1/N) sum_ij log sigmoid(y_ij (t u_i . v_j + b)), with y = +1 on the diagonal, -1 elsewhere."""
    n = len(u)
    t = np.exp(log_t)
    sim = u @ v.T
    z = t * sim + b
    y = 2 * np.eye(n) - 1
    loss = -np.sum(log_sigmoid(y * z)) / n
    dz = -y * (1 - 1 / (1 + np.exp(-y * z))) / n  # d/dz of -log sigmoid(y z) is -y sigmoid(-y z)
    du = t * dz @ v
    dv = t * dz.T @ u
    return float(loss), du, dv, float(np.sum(dz * sim) * t), float(np.sum(dz))


def sigmoid_loss_chunked(u, v, log_t, b, chunks):
    """The same loss computed block by block, as when text embeddings are passed between devices."""
    t, n = np.exp(log_t), len(u)
    total = 0.0
    for i in np.array_split(np.arange(n), chunks):
        for j in np.array_split(np.arange(n), chunks):
            y = np.where(i[:, None] == j[None, :], 1.0, -1.0)
            total -= np.sum(log_sigmoid(y * (t * u[i] @ v[j].T + b)))
    return total / n


def make_pairs(n, rng, k=8, dx=32, dy=24, noise=0.3, a=None, b=None):
    """n paired views of a k-dimensional latent; the mixing matrices are returned so new data share them."""
    a = rng.standard_normal((k, dx)) if a is None else a
    b = rng.standard_normal((k, dy)) if b is None else b
    z = rng.standard_normal((n, k))
    x = z @ a + noise * rng.standard_normal((n, dx))
    y = z @ b + noise * rng.standard_normal((n, dy))
    return x, y, z, a, b


class DualEncoder:
    def __init__(self, dx, dy, d=16, rng=None, log_s=LOG_INIT_SCALE, loss="clip"):
        rng = rng or np.random.default_rng(0)
        self.wi = rng.standard_normal((dx, d)) / np.sqrt(dx)
        self.wt = rng.standard_normal((dy, d)) / np.sqrt(dy)
        self.log_s, self.b, self.loss = log_s, -10.0, loss
        if loss == "sigmoid":
            self.log_s = np.log(10.0)

    def embed_images(self, x):
        return normalize(x @ self.wi)[0]

    def embed_texts(self, y):
        return normalize(y @ self.wt)[0]

    def step(self, x, y, lr, scalar_lr=0.01, max_log_s=LOG_MAX_SCALE):
        """One gradient step; the scale and bias get a smaller rate, or they collapse to the label prior."""
        u, nu = normalize(x @ self.wi)
        v, nv = normalize(y @ self.wt)
        if self.loss == "clip":
            loss, du, dv, dls = clip_loss(u, v, self.log_s)
            db = 0.0
        else:
            loss, du, dv, dls, db = sigmoid_loss(u, v, self.log_s, self.b)
        self.wi -= lr * x.T @ normalize_backward(u, nu, du)
        self.wt -= lr * y.T @ normalize_backward(v, nv, dv)
        self.log_s = min(self.log_s - scalar_lr * dls, max_log_s)
        self.b -= scalar_lr * db
        return loss


def train(model, x, y, steps=300, batch=128, lr=0.5, rng=None):
    rng = rng or np.random.default_rng(1)
    losses = []
    for _ in range(steps):
        idx = rng.choice(len(x), batch, replace=False)
        losses.append(model.step(x[idx], y[idx], lr))
    return losses


def retrieval_accuracy(model, x, y):
    """Share of images whose own caption has the largest cosine among all captions."""
    sim = model.embed_images(x) @ model.embed_texts(y).T
    return float(np.mean(sim.argmax(1) == np.arange(len(x))))


def zero_shot_classify(model, x, class_texts):
    """class_texts: (K, P, dy) captions per class; P prompts are averaged after normalization (an ensemble)."""
    k, p, dy = class_texts.shape
    e = model.embed_texts(class_texts.reshape(k * p, dy)).reshape(k, p, -1).mean(1)
    w = e / np.linalg.norm(e, axis=1, keepdims=True)
    return (model.embed_images(x) @ w.T).argmax(1)


def loss_floor(s, n, neg_cos):
    """The softmax loss when positives have cosine 1 and all negatives cosine neg_cos."""
    return float(np.log1p((n - 1) * np.exp(-s * (1 - neg_cos))))


def modality_gap(model, x, y):
    """Distance between the mean image embedding and the mean text embedding, as in Liang et al."""
    return float(np.linalg.norm(model.embed_images(x).mean(0) - model.embed_texts(y).mean(0)))
