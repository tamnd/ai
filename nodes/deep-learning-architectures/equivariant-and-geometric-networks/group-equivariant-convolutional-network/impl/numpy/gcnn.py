# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/group-equivariant-convolutional-network/ (ai://NEW-group-equivariant-convolutional-network).
"""The group equivariant convolutional network for the group p4, the NumPy reference.

p4 is the group of translations of the pixel grid and rotations by multiples of 90 degrees. A feature map on p4 is
an array (B, C, 4, n, n): for each channel, one planar map per rotation r. The rotation by 90 degrees of such a map,
rotate_p4, rotates every plane and moves plane r to r + 1 (eq-p4-action). The lifting layer correlates a planar
image with the four rotated copies of each filter (eq-lift); the group correlation correlates a p4 map with filters
that are themselves p4 maps, rotated and cycled (eq-gconv). Both are computed as Cohen and Welling do, by building
the augmented filter bank and calling one planar correlation (eq-bank). Rotations are about the centre of the
image, which is a symmetry of the grid for any n, and the planar correlation pads with zeros (or wraps around), so
both commute with rot90 exactly. The model: lifting layer, ReLU, group correlation, ReLU, max over the four
rotations (coset pooling, eq-rotpool), mean over the plane, linear classifier; its logits are invariant to
rotations of the input by multiples of 90 degrees and, with mode="circular", to cyclic translations.
"""

import numpy as np

NAMES = ("W1", "b1", "W2", "b2", "Wc", "bc")


def config(c_in=1, c1=4, c2=4, k=3, n_classes=3, mode="zero"):
    if min(c_in, c1, c2, n_classes) < 1 or k < 1 or k % 2 == 0:
        raise ValueError("the channel counts and n_classes must be at least 1, and k odd")
    if mode not in ("zero", "circular"):
        raise ValueError("mode is zero or circular")
    return {"c_in": c_in, "c1": c1, "c2": c2, "k": k, "n_classes": n_classes, "mode": mode}


def param_count(cfg):
    c_in, c1, c2, k, m = cfg["c_in"], cfg["c1"], cfg["c2"], cfg["k"], cfg["n_classes"]
    return c1 * c_in * k * k + c1 + 4 * c2 * c1 * k * k + c2 + m * c2 + m  # eq: eq-params


def flops(cfg, B, n):
    """FLOPs of the forward pass on a batch of B images of n by n, a multiply-add counted as 2."""
    c_in, c1, c2, k, m = cfg["c_in"], cfg["c1"], cfg["c2"], cfg["k"], cfg["n_classes"]
    return 2 * B * n * n * k * k * (4 * c1 * c_in + 16 * c2 * c1) + 2 * B * c2 * m  # eq: eq-flops


def init(rng, cfg, scale=1.0):
    c_in, c1, c2, k, m = cfg["c_in"], cfg["c1"], cfg["c2"], cfg["k"], cfg["n_classes"]
    return {"W1": scale * rng.normal(size=(c1, c_in, k, k)) / np.sqrt(c_in * k * k), "b1": np.zeros(c1),
            "W2": scale * rng.normal(size=(c2, c1, 4, k, k)) / np.sqrt(4 * c1 * k * k), "b2": np.zeros(c2),
            "Wc": scale * rng.normal(size=(m, c2)) / np.sqrt(c2), "bc": np.zeros(m)}


# The group action on images and on p4 feature maps.

def rotate(x, r=1):
    """The image x (..., n, n) rotated by r times 90 degrees about its centre."""
    return np.rot90(x, r, axes=(-2, -1))


def rotate_p4(f, r=1):
    """A p4 map f (B, C, 4, n, n) rotated by r times 90 degrees: each plane rotates and plane s moves to s + r."""
    return np.roll(rotate(f, r), r, axis=2)  # eq: eq-p4-action


def shift(x, a, b, mode):
    """y[..., u, v] = x[..., u - a, v - b], with zeros or wrapping around where u - a or v - b leaves the grid."""
    if mode == "circular":
        return np.roll(x, (a, b), axis=(-2, -1))
    n, m = x.shape[-2:]
    y = np.zeros_like(x)
    y[..., max(a, 0):n + min(a, 0), max(b, 0):m + min(b, 0)] = x[..., max(-a, 0):n + min(-a, 0), max(-b, 0):m + min(-b, 0)]
    return y


# The planar correlation and its gradients.

def corr2d(x, w, mode="zero"):
    """out[b, o, u, v] = sum over c, i, j of x[b, c, u + i - p, v + j - p] w[o, c, i, j], with p = k // 2."""
    k = w.shape[-1]
    p = k // 2
    out = 0.0
    for i in range(k):
        for j in range(k):
            out = out + np.einsum("bcuv,oc->bouv", shift(x, p - i, p - j, mode), w[:, :, i, j])
    return out


def corr2d_backward(x, w, dout, mode="zero"):
    k = w.shape[-1]
    p = k // 2
    dx, dw = np.zeros_like(x), np.zeros_like(w)
    for i in range(k):
        for j in range(k):
            dw[:, :, i, j] = np.einsum("bouv,bcuv->oc", dout, shift(x, p - i, p - j, mode))
            dx += shift(np.einsum("bouv,oc->bcuv", dout, w[:, :, i, j]), i - p, j - p, mode)
    return dx, dw


# The two equivariant layers, as one planar correlation with the augmented filter bank.

def lift_bank(W):
    """W (O, C, k, k) to the bank (4 O, C, k, k) whose filter o * 4 + r is W[o] rotated r times."""
    O, C, k, _ = W.shape
    return np.stack([rotate(W, r) for r in range(4)], axis=1).reshape(4 * O, C, k, k)  # eq: eq-lift


def gconv_bank(W):
    """W (O, C, 4, k, k) to the bank (4 O, 4 C, k, k): filter (o, r) on input plane (c, s) is W[o, c, s - r] rotated r times."""
    O, C, _, k, _ = W.shape
    bank = np.empty((O, 4, C, 4, k, k), dtype=W.dtype)
    for r in range(4):
        for s in range(4):
            bank[:, r, :, s] = rotate(W[:, :, (s - r) % 4], r)  # eq: eq-gconv
    return bank.reshape(4 * O, 4 * C, k, k)  # eq: eq-bank


def lift(x, W, mode="zero"):
    """The lifting correlation of x (B, C, n, n) with W (O, C, k, k): a p4 map (B, O, 4, n, n)."""
    B, _, n, _ = x.shape
    return corr2d(x, lift_bank(W), mode).reshape(B, W.shape[0], 4, n, n)


def gconv(f, W, mode="zero"):
    """The p4 correlation of f (B, C, 4, n, n) with W (O, C, 4, k, k): a p4 map (B, O, 4, n, n)."""
    B, C, _, n, _ = f.shape
    return corr2d(f.reshape(B, 4 * C, n, n), gconv_bank(W), mode).reshape(B, W.shape[0], 4, n, n)


def lift_backward(x, W, dout, mode="zero"):
    B, O = dout.shape[:2]
    n = x.shape[-1]
    dx, dbank = corr2d_backward(x, lift_bank(W), dout.reshape(B, 4 * O, n, n), mode)
    dbank = dbank.reshape(O, 4, *W.shape[1:])
    return dx, sum(rotate(dbank[:, r], -r) for r in range(4))


def gconv_backward(f, W, dout, mode="zero"):
    B, C, _, n, _ = f.shape
    O, k = W.shape[0], W.shape[-1]
    df, dbank = corr2d_backward(f.reshape(B, 4 * C, n, n), gconv_bank(W), dout.reshape(B, 4 * O, n, n), mode)
    dbank = dbank.reshape(O, 4, C, 4, k, k)
    dW = np.zeros_like(W)
    for r in range(4):
        for s in range(4):
            dW[:, :, (s - r) % 4] += rotate(dbank[:, r, :, s], -r)
    return df.reshape(f.shape), dW


def gconv_direct(f, W, mode="circular"):
    """The p4 correlation by its definition, [f * psi](t, r) = sum over (y, s) of f(y, s) psi(r^-1 (y - t), s - r).

    Slow, for the tests: the sum runs over the group elements h = (y, s) and the filter is read at g^-1 h."""
    B, C, _, n, _ = f.shape
    O, k = W.shape[0], W.shape[-1]
    p = k // 2
    out = np.zeros((B, O, 4, n, n))
    for r in range(4):
        for s in range(4):
            Wr = rotate(W[:, :, (s - r) % 4], r)  # psi(r^-1 z, s - r) as an array over the offset z
            for i in range(k):
                for j in range(k):
                    out[:, :, r] += np.einsum("bcuv,oc->bouv", shift(f[:, :, s], p - i, p - j, mode), Wr[:, :, i, j])
    return out


# The model.

def forward(params, x, cfg, cache=False):
    """Logits (B, n_classes) of the images x (B, c_in, n, n)."""
    mode = cfg["mode"]
    h1 = np.maximum(lift(x, params["W1"], mode) + params["b1"][:, None, None, None], 0.0)  # (B, c1, 4, n, n)
    h2 = np.maximum(gconv(h1, params["W2"], mode) + params["b2"][:, None, None, None], 0.0)  # (B, c2, 4, n, n)
    pooled = h2.max(axis=2)  # (B, c2, n, n), eq: eq-rotpool
    g = pooled.mean(axis=(2, 3))  # (B, c2)
    logits = g @ params["Wc"].T + params["bc"]  # (B, n_classes)
    if cache:
        return logits, (x, h1, h2, pooled, g)
    return logits


def cross_entropy(logits, labels):
    z = logits - logits.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    B = len(labels)
    grad = np.exp(logp)
    grad[np.arange(B), labels] -= 1.0
    return -logp[np.arange(B), labels].mean(), grad / B


def backward(params, cache, dlogits, cfg):
    """The gradients of the parameters and of the input, given dL/dlogits."""
    mode = cfg["mode"]
    x, h1, h2, pooled, g = cache
    n = x.shape[-1]
    grads = {"Wc": dlogits.T @ g, "bc": dlogits.sum(0)}
    dg = dlogits @ params["Wc"]
    dpooled = np.broadcast_to(dg[:, :, None, None] / (n * n), pooled.shape)
    winner = h2.argmax(axis=2)  # the rotation that wins the max, ties to the first
    dh2 = (np.arange(4)[None, None, :, None, None] == winner[:, :, None]) * dpooled[:, :, None]
    dz2 = dh2 * (h2 > 0)
    grads["b2"] = dz2.sum(axis=(0, 2, 3, 4))
    dh1, grads["W2"] = gconv_backward(h1, params["W2"], dz2, mode)
    dz1 = dh1 * (h1 > 0)
    grads["b1"] = dz1.sum(axis=(0, 2, 3, 4))
    dx, grads["W1"] = lift_backward(x, params["W1"], dz1, mode)
    return grads, dx


def loss_and_grads(params, x, labels, cfg):
    logits, cache = forward(params, x, cfg, cache=True)
    loss, dlogits = cross_entropy(logits, labels)
    grads, dx = backward(params, cache, dlogits, cfg)
    return loss, grads, dx


def train(params, x, labels, cfg, lr=0.1, steps=100):
    params = {k: v.copy() for k, v in params.items()}
    losses = []
    for _ in range(steps):
        loss, grads, _ = loss_and_grads(params, x, labels, cfg)
        for k in NAMES:
            params[k] -= lr * grads[k]
        losses.append(loss)
    return params, losses
