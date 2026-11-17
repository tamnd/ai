# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/maximal-update-parametrization/ (ai://NEW-maximal-update-parametrization).
"""The maximal update parametrization (muP) of an MLP in NumPy, with the standard parametrization (SP) for contrast.

layer_hparams maps a base width, a target width and the base hyperparameters to the initialization std and the
learning rate of each layer, by eq-mup-sgd and eq-mup-adam (Table 3 of Yang et al. 2022, with the base-width
constants of their Eq. 4); attention_logits is eq-attn; MLP is a two-hidden-layer tanh network whose initialization
and per-layer updates follow layer_hparams; feature_change measures the per-coordinate change of the last hidden
preactivations and of the output after one training step."""

import numpy as np

LAYERS = ("input", "hidden", "output", "bias")
OPTIMIZERS = ("sgd", "adam")
PARAMS = ("mup", "sp")


def layer_hparams(width, base_width, lr, d_in, sigma=1.0, optimizer="adam", param="mup"):
    """Initialization std and learning rate of each layer kind of an MLP of the given width.

    lr and sigma are the base hyperparameters, tuned at base_width. At width == base_width muP and SP agree."""
    if width <= 0 or base_width <= 0 or d_in <= 0:
        raise ValueError("widths and the input dimension must be positive")
    if optimizer not in OPTIMIZERS:
        raise ValueError(f"optimizer must be one of {OPTIMIZERS}")
    if param not in PARAMS:
        raise ValueError(f"param must be one of {PARAMS}")
    r = width / base_width
    std = {
        "input": sigma / np.sqrt(d_in),
        "hidden": sigma / np.sqrt(width),
        "output": sigma / np.sqrt(width),
        "bias": 0.0,
    }
    rate = {k: float(lr) for k in LAYERS}
    if param == "mup":
        # eq: mup-init -- the readout variance is 1/(n * n/n0) instead of 1/n
        std["output"] = sigma / np.sqrt(width * r)
        if optimizer == "sgd":
            # eq: mup-sgd -- input weights and biases lr * n/n0, hidden lr, output lr * n0/n
            rate["input"] = lr * r
            rate["bias"] = lr * r
            rate["output"] = lr / r
        else:
            # eq: mup-adam -- input weights and biases lr, hidden and output lr * n0/n
            rate["hidden"] = lr / r
            rate["output"] = lr / r
    return {k: {"std": float(std[k]), "lr": float(rate[k])} for k in LAYERS}


def attention_logits(q, k, base_d_head=None, alpha=1.0, param="mup"):
    """Attention logits alpha * sqrt(d0) / d * q k^T under muP, alpha / sqrt(d) * q k^T under SP.

    q has shape (..., Tq, d) and k has shape (..., Tk, d); d0 defaults to d, where both rules agree."""
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    d = q.shape[-1]
    if k.shape[-1] != d:
        raise ValueError("queries and keys must have the same dimension")
    if param == "sp":
        scale = alpha / np.sqrt(d)
    elif param == "mup":
        d0 = d if base_d_head is None else base_d_head
        # eq: attn -- 1/d attention, equal to 1/sqrt(d) at the base head dimension
        scale = alpha * np.sqrt(d0) / d
    else:
        raise ValueError(f"param must be one of {PARAMS}")
    return scale * (q @ np.swapaxes(k, -1, -2))


class MLP:
    """f(x) = V tanh(W tanh(U x + b1) + b2), trained on the loss 0.5 * mean over the batch of |f - y|^2."""

    KIND = {"U": "input", "b1": "bias", "W": "hidden", "b2": "bias", "V": "output"}

    def __init__(
        self,
        d_in,
        width,
        d_out,
        base_width,
        lr,
        sigma=1.0,
        optimizer="adam",
        param="mup",
        seed=0,
        betas=(0.9, 0.999),
        eps=1e-12,
    ):
        self.optimizer = optimizer
        self.hp = layer_hparams(width, base_width, lr, d_in, sigma, optimizer, param)
        rng = np.random.default_rng(seed)
        shapes = {
            "U": (width, d_in),
            "b1": (width,),
            "W": (width, width),
            "b2": (width,),
            "V": (d_out, width),
        }
        self.params = {
            name: rng.standard_normal(shape) * self.hp[self.KIND[name]]["std"] for name, shape in shapes.items()
        }
        self.lr = {name: self.hp[kind]["lr"] for name, kind in self.KIND.items()}
        self.betas = betas
        self.eps = eps
        self.t = 0
        self.m = {name: np.zeros_like(p) for name, p in self.params.items()}
        self.v = {name: np.zeros_like(p) for name, p in self.params.items()}

    def forward(self, X):
        p = self.params
        h1 = X @ p["U"].T + p["b1"]
        x1 = np.tanh(h1)
        h2 = x1 @ p["W"].T + p["b2"]
        x2 = np.tanh(h2)
        f = x2 @ p["V"].T
        return f, (h1, x1, h2, x2)

    def loss(self, X, y):
        f, _ = self.forward(X)
        return 0.5 * float(np.mean(np.sum((f - y) ** 2, axis=-1)))

    def grads(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        f, (h1, x1, h2, x2) = self.forward(X)
        df = (f - y) / X.shape[0]
        p = self.params
        dx2 = df @ p["V"]
        dh2 = dx2 * (1.0 - x2**2)
        dx1 = dh2 @ p["W"]
        dh1 = dx1 * (1.0 - x1**2)
        return {
            "V": df.T @ x2,
            "W": dh2.T @ x1,
            "b2": dh2.sum(0),
            "U": dh1.T @ X,
            "b1": dh1.sum(0),
        }

    def step(self, X, y):
        """One step of SGD or Adam with the per-layer learning rates; returns the loss before the step."""
        loss = self.loss(X, y)
        g = self.grads(X, y)
        self.t += 1
        b1, b2 = self.betas
        for name, p in self.params.items():
            if self.optimizer == "sgd":
                p -= self.lr[name] * g[name]
            else:
                self.m[name] = b1 * self.m[name] + (1 - b1) * g[name]
                self.v[name] = b2 * self.v[name] + (1 - b2) * g[name] ** 2
                mhat = self.m[name] / (1 - b1**self.t)
                vhat = self.v[name] / (1 - b2**self.t)
                p -= self.lr[name] * mhat / (np.sqrt(vhat) + self.eps)
        return loss


def feature_change(net, X, y):
    """RMS per-coordinate change of the last hidden preactivations and of the output after one step on (X, y)."""
    f0, (_, _, h0, _) = net.forward(X)
    net.step(X, y)
    f1, (_, _, h1, _) = net.forward(X)
    return float(np.sqrt(np.mean((h1 - h0) ** 2))), float(np.sqrt(np.mean((f1 - f0) ** 2)))
