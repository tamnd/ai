# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/deep-sets/ (ai://NEW-deep-sets).
"""Deep Sets of Zaheer et al. (2017), the PyTorch version, with the parameters of the NumPy reference.

A batch of sets is a (B, N, D0) tensor with a (B, N) mask of the real elements. Each equivariant layer is
act(h Lambda - 1 pool(h) Gamma + 1 beta), the pool a sum, a mean or a maximum over the elements of each set that skips
the padding; interact=False drops Gamma. head="set" pools the set, applies one hidden layer of rho and a linear map
to K class scores; head="element" gives one score per element, -inf on the padding, for a softmax over the set.
"""

import torch
import torch.nn.functional as F

POOLS = ("sum", "mean", "max")
ACTS = ("tanh", "relu")
HEADS = ("set", "element")


def config(d_in=3, width=8, n_layers=2, n_classes=5, pool="max", readout="sum", act="tanh", interact=True,
           head="set"):  # fmt: skip
    """Return the config of the network; the defaults are the example config of shapes.yaml."""
    if min(d_in, width, n_layers) < 1:
        raise ValueError("the sizes must be at least 1")
    if head == "set" and n_classes < 1:
        raise ValueError("the set head needs at least one output; n_classes=1 gives a regression")
    if pool not in POOLS or readout not in POOLS:
        raise ValueError(f"pool and readout must be one of {POOLS}")
    if act not in ACTS or head not in HEADS:
        raise ValueError(f"act must be one of {ACTS} and head one of {HEADS}")
    return {"d_in": d_in, "width": width, "n_layers": n_layers, "n_classes": n_classes, "pool": pool,
            "readout": readout, "act": act, "interact": bool(interact), "head": head}  # fmt: skip


def param_shapes(cfg):
    w = [cfg["d_in"]] + [cfg["width"]] * cfg["n_layers"]
    shapes = {}
    for l in range(cfg["n_layers"]):
        shapes[f"eq{l}_Lambda"] = (w[l], w[l + 1])
        if cfg["interact"]:
            shapes[f"eq{l}_Gamma"] = (w[l], w[l + 1])
        shapes[f"eq{l}_beta"] = (w[l + 1],)
    H, K = cfg["width"], cfg["n_classes"]
    if cfg["head"] == "set":
        shapes |= {"rho_W": (H, H), "rho_b": (H,), "out_W": (H, K), "out_b": (K,)}
    else:
        shapes["out_Lambda"] = (H, 1)
        if cfg["interact"]:
            shapes["out_Gamma"] = (H, 1)
        shapes["out_beta"] = (1,)
    return shapes


def param_count(cfg):
    return sum(torch.Size(s).numel() for s in param_shapes(cfg).values())


def pool(h, mask, kind):
    """Pool (B, N, D) over the elements of each set to (B, D), feature by feature, skipping the padding."""
    m = mask[..., None]
    if kind == "max":
        return h.masked_fill(m == 0, float("-inf")).amax(1)
    s = (h * m).sum(1)
    return s / m.sum(1) if kind == "mean" else s


class DeepSets(torch.nn.Module):
    """Deep Sets, its parameters named as in the NumPy reference."""

    def __init__(self, cfg, scale=0.1, generator=None):
        super().__init__()
        self.cfg = cfg
        self.p = torch.nn.ParameterDict(
            {name: torch.nn.Parameter(torch.randn(shape, generator=generator) * scale)
             for name, shape in param_shapes(cfg).items()}  # fmt: skip
        )

    @classmethod
    def from_params(cls, params, cfg):
        model = cls(cfg)
        with torch.no_grad():
            for name, value in params.items():
                model.p[name].copy_(torch.as_tensor(value))
        return model

    def act(self, z):
        return torch.tanh(z) if self.cfg["act"] == "tanh" else F.relu(z)

    def equivariant(self, h, mask, prefix):
        """eq-equivariant before the activation."""
        z = h @ self.p[prefix + "Lambda"] + self.p[prefix + "beta"]
        if prefix + "Gamma" in self.p:
            z = z - (pool(h, mask, self.cfg["pool"]) @ self.p[prefix + "Gamma"])[:, None, :]
        return z

    def features(self, x, mask=None):
        """The output of the equivariant layers, (B, N, H), zero on the padding."""
        mask = torch.ones(x.shape[:2], dtype=x.dtype) if mask is None else mask.to(x.dtype)
        h = x * mask[..., None]
        for l in range(self.cfg["n_layers"]):
            h = self.act(self.equivariant(h, mask, f"eq{l}_")) * mask[..., None]
        return h, mask

    def forward(self, x, mask=None):
        h, mask = self.features(x, mask)
        if self.cfg["head"] == "set":
            s = pool(h, mask, self.cfg["readout"])
            return self.act(s @ self.p["rho_W"] + self.p["rho_b"]) @ self.p["out_W"] + self.p["out_b"]
        return self.equivariant(h, mask, "out_")[..., 0].masked_fill(mask == 0, float("-inf"))


def nll(logits, labels):
    """eq-loss: the cross-entropy of the labels (a class, or the position of the outlier), averaged over the batch."""
    return F.cross_entropy(logits, labels)
