# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/group-equivariant-convolutional-network/ (ai://1REXCC).
"""The group equivariant convolutional network for the group p4, the PyTorch reference.

The same model as the NumPy reference: a lifting layer, ReLU, a p4 group correlation, ReLU, the maximum over the 4
rotations, the mean over the plane and a linear classifier. Both equivariant layers build the augmented filter bank
from the stored weights and call conv2d once (Cohen and Welling, section 7), so the cost is that of a planar
convolution with 4 times as many output channels; autograd differentiates through the bank, which sums the gradient
of the 4 copies of each weight. conv2d is a cross-correlation, as on the page.
"""

import torch
import torch.nn.functional as F

NAMES = ("W1", "b1", "W2", "b2", "Wc", "bc")


def config(c_in=1, c1=4, c2=4, k=3, n_classes=3, mode="zero"):
    if min(c_in, c1, c2, n_classes) < 1 or k < 1 or k % 2 == 0:
        raise ValueError("the channel counts and n_classes must be at least 1, and k odd")
    if mode not in ("zero", "circular"):
        raise ValueError("mode is zero or circular")
    return {"c_in": c_in, "c1": c1, "c2": c2, "k": k, "n_classes": n_classes, "mode": mode}


def param_count(cfg):
    c_in, c1, c2, k, m = cfg["c_in"], cfg["c1"], cfg["c2"], cfg["k"], cfg["n_classes"]
    return c1 * c_in * k * k + c1 + 4 * c2 * c1 * k * k + c2 + m * c2 + m


def init(gen, cfg, dtype=torch.float64):
    c_in, c1, c2, k, m = cfg["c_in"], cfg["c1"], cfg["c2"], cfg["k"], cfg["n_classes"]
    r = lambda *s: torch.randn(*s, generator=gen, dtype=dtype)  # noqa: E731
    return {
        "W1": r(c1, c_in, k, k) / (c_in * k * k) ** 0.5,
        "b1": torch.zeros(c1, dtype=dtype),
        "W2": r(c2, c1, 4, k, k) / (4 * c1 * k * k) ** 0.5,
        "b2": torch.zeros(c2, dtype=dtype),
        "Wc": r(m, c2) / c2**0.5,
        "bc": torch.zeros(m, dtype=dtype),
    }


def rotate(x, r=1):
    return torch.rot90(x, r, dims=(-2, -1))


def rotate_p4(f, r=1):
    """A p4 map (B, C, 4, n, n) rotated by r times 90 degrees: each plane rotates and plane s moves to s + r."""
    return torch.roll(rotate(f, r), r, dims=2)


def lift_bank(W):
    O, C, k, _ = W.shape  # noqa: E741
    return torch.stack([rotate(W, r) for r in range(4)], dim=1).reshape(4 * O, C, k, k)


def gconv_bank(W):
    """W (O, C, 4, k, k) to (4 O, 4 C, k, k): filter (o, r) on input plane (c, s) is W[o, c, s - r] turned r times.

    Cycling the rotation axis by r and turning the planes is a gather, so autograd routes each gradient back."""
    O, C, _, k, _ = W.shape  # noqa: E741
    rows = [rotate(torch.roll(W, r, dims=2), r) for r in range(4)]  # rows[r][:, :, s] = rot^r W[:, :, s - r]
    return torch.stack(rows, dim=1).reshape(4 * O, 4 * C, k, k)


def corr2d(x, w, mode):
    p = w.shape[-1] // 2
    if mode == "circular":
        return F.conv2d(F.pad(x, (p, p, p, p), mode="circular"), w)
    return F.conv2d(x, w, padding=p)


def lift(x, W, mode="zero"):
    B, _, n, _ = x.shape
    return corr2d(x, lift_bank(W), mode).reshape(B, W.shape[0], 4, n, n)


def gconv(f, W, mode="zero"):
    B, C, _, n, _ = f.shape
    return corr2d(f.reshape(B, 4 * C, n, n), gconv_bank(W), mode).reshape(B, W.shape[0], 4, n, n)


def forward(params, x, cfg):
    mode = cfg["mode"]
    h1 = torch.relu(lift(x, params["W1"], mode) + params["b1"][:, None, None, None])
    h2 = torch.relu(gconv(h1, params["W2"], mode) + params["b2"][:, None, None, None])
    g = h2.amax(dim=2).mean(dim=(2, 3))
    return g @ params["Wc"].T + params["bc"]


def loss_fn(params, x, labels, cfg):
    return F.cross_entropy(forward(params, x, cfg), labels)
