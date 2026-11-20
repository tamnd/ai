# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/deep-sets/ (ai://S263A8).
"""Deep Sets of Zaheer et al. (2017), the NumPy reference: forward, backward and the counts.

A batch of sets is a (B, N, D0) array x with a (B, N) mask that is 1 on the elements and 0 on the padding, so that
sets of different sizes share one array. L permutation-equivariant layers (eq-equivariant) map each element to H
numbers: h' = act(h Lambda - 1 pool(h) Gamma + 1 beta), where pool sums, averages or takes the maximum over the
elements of each set, feature by feature, ignoring the padding. With interact=False the layers drop Gamma and act
on each element alone, which is the phi of the invariant model (eq-invariant). With head="set" a readout pool turns
the set into one vector, rho is one hidden layer and a linear map gives K class scores, the point-cloud classifier
of the paper's Appendix H; with head="element" a last equivariant layer without activation gives one score per
element and a softmax over the set picks one, the outlier detector of its Section 4.3. Everything is float64.
"""

import numpy as np

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
    return {
        "d_in": d_in,
        "width": width,
        "n_layers": n_layers,
        "n_classes": n_classes,
        "pool": pool,
        "readout": readout,
        "act": act,
        "interact": bool(interact),
        "head": head,
    }


def widths(cfg):
    """The widths D_0, ..., D_L of the equivariant layers."""
    return [cfg["d_in"]] + [cfg["width"]] * cfg["n_layers"]


def param_shapes(cfg):
    w = widths(cfg)
    shapes = {}
    for l in range(cfg["n_layers"]):
        shapes[f"eq{l}_Lambda"] = (w[l], w[l + 1])
        if cfg["interact"]:
            shapes[f"eq{l}_Gamma"] = (w[l], w[l + 1])
        shapes[f"eq{l}_beta"] = (w[l + 1],)
    H = cfg["width"]
    if cfg["head"] == "set":
        shapes |= {"rho_W": (H, H), "rho_b": (H,), "out_W": (H, cfg["n_classes"]), "out_b": (cfg["n_classes"],)}
    else:
        shapes["out_Lambda"] = (H, 1)
        if cfg["interact"]:
            shapes["out_Gamma"] = (H, 1)
        shapes["out_beta"] = (1,)
    return shapes


def param_count(cfg):
    """eq-params, from the formula rather than the shapes, so that the two can be checked against each other."""
    w, k = widths(cfg), 2 if cfg["interact"] else 1
    n = sum(k * w[l] * w[l + 1] + w[l + 1] for l in range(cfg["n_layers"]))
    H, K = cfg["width"], cfg["n_classes"]
    return n + (H * H + H + H * K + K if cfg["head"] == "set" else k * H + 1)


def flops(cfg, B, N):
    """eq-flops: the FLOPs of the matrix products of a forward pass over B sets of N elements, a multiply-add as 2."""
    w, H = widths(cfg), cfg["width"]
    n = 0
    for l in range(cfg["n_layers"]):
        n += 2 * B * N * w[l] * w[l + 1] + (2 * B * w[l] * w[l + 1] if cfg["interact"] else 0)
    if cfg["head"] == "set":
        return n + 2 * B * H * H + 2 * B * H * cfg["n_classes"]
    return n + 2 * B * N * H + (2 * B * H if cfg["interact"] else 0)


def init(rng, cfg, scale=0.1):
    """Random parameters, every one N(0, scale^2)."""
    return {name: rng.normal(0.0, scale, shape) for name, shape in param_shapes(cfg).items()}


def full_mask(x):
    return np.ones(x.shape[:2])


def pool(h, mask, kind):
    """Pool (B, N, D) over the elements of each set to (B, D), feature by feature, skipping the padding."""
    m = mask[..., None]
    if kind == "max":
        return np.where(m > 0, h, -np.inf).max(1)
    s = (h * m).sum(1)
    return s / m.sum(1) if kind == "mean" else s


def pool_backward(dp, h, mask, kind):
    """The gradient of pool with respect to h; for max it goes to the first maximizing element."""
    m = mask[..., None]
    if kind == "sum":
        return dp[:, None, :] * m
    if kind == "mean":
        return dp[:, None, :] * m / m.sum(1, keepdims=True)
    masked = np.where(m > 0, h, -np.inf)
    hit = np.zeros_like(h)
    idx = masked.argmax(1)
    b, d = np.meshgrid(np.arange(h.shape[0]), np.arange(h.shape[2]), indexing="ij")
    hit[b, idx, d] = 1.0
    return dp[:, None, :] * hit


def activate(z, act):
    return np.tanh(z) if act == "tanh" else np.maximum(z, 0.0)


def activate_grad(z, a, act):
    return 1.0 - a * a if act == "tanh" else (z > 0).astype(float)


def equivariant(h, mask, Lam, Gam, beta, kind):
    """eq-equivariant before the activation: h Lambda - 1 pool(h) Gamma + 1 beta, and the pool."""
    z = h @ Lam + beta
    p = None
    if Gam is not None:
        p = pool(h, mask, kind)
        z = z - (p @ Gam)[:, None, :]
    return z, p


def equivariant_backward(dz, h, mask, p, Lam, Gam, kind):
    """The gradients of eq-equivariant (before the activation) with respect to h and the parameters."""
    grads = {"Lambda": np.einsum("bnd,bne->de", h, dz), "beta": dz.sum((0, 1))}
    dh = dz @ Lam.T
    if Gam is not None:
        dq = -dz.sum(1)
        grads["Gamma"] = p.T @ dq
        dh = dh + pool_backward(dq @ Gam.T, h, mask, kind)
    return grads, dh


def forward(params, cfg, x, mask=None):
    """The scores: (B, K) class scores for head="set", (B, N) element scores (padding at -inf) for "element"."""
    mask = full_mask(x) if mask is None else np.asarray(mask, dtype=float)
    h, layers = x * mask[..., None], []
    for l in range(cfg["n_layers"]):
        Gam = params.get(f"eq{l}_Gamma")
        z, p = equivariant(h, mask, params[f"eq{l}_Lambda"], Gam, params[f"eq{l}_beta"], cfg["pool"])
        a = activate(z, cfg["act"]) * mask[..., None]
        layers.append({"h": h, "z": z, "p": p, "a": a})
        h = a
    cache = {"mask": mask, "layers": layers, "features": h}
    if cfg["head"] == "set":
        s = pool(h, mask, cfg["readout"])
        u = s @ params["rho_W"] + params["rho_b"]
        r = activate(u, cfg["act"])
        cache |= {"s": s, "u": u, "r": r}
        return r @ params["out_W"] + params["out_b"], cache
    z, p = equivariant(h, mask, params["out_Lambda"], params.get("out_Gamma"), params["out_beta"], cfg["pool"])
    cache |= {"z": z, "p": p}
    return np.where(mask > 0, z[..., 0], -np.inf), cache


def nll(logits, labels):
    """eq-loss: the cross-entropy of the labels (a class, or the position of the outlier), averaged over the batch."""
    m = logits.max(-1, keepdims=True)
    logp = logits - m - np.log(np.exp(logits - m).sum(-1, keepdims=True))
    return float(-logp[np.arange(len(labels)), labels].mean())


def loss(params, cfg, x, labels, mask=None):
    return nll(forward(params, cfg, x, mask)[0], labels)


def backward(params, cfg, cache, logits, labels, x):
    """The gradients of eq-loss with respect to the parameters and to the input x."""
    mask, B = cache["mask"], len(labels)
    e = np.exp(logits - logits.max(-1, keepdims=True))
    dlogits = e / e.sum(-1, keepdims=True)
    dlogits[np.arange(B), labels] -= 1.0
    dlogits /= B
    grads = {}
    h = cache["features"]
    if cfg["head"] == "set":
        grads["out_W"] = cache["r"].T @ dlogits
        grads["out_b"] = dlogits.sum(0)
        du = (dlogits @ params["out_W"].T) * activate_grad(cache["u"], cache["r"], cfg["act"])
        grads["rho_W"] = cache["s"].T @ du
        grads["rho_b"] = du.sum(0)
        dh = pool_backward(du @ params["rho_W"].T, h, mask, cfg["readout"])
    else:
        dz = (dlogits * mask)[..., None]
        g, dh = equivariant_backward(dz, h, mask, cache["p"], params["out_Lambda"], params.get("out_Gamma"),
                                     cfg["pool"])  # fmt: skip
        grads |= {"out_" + k: v for k, v in g.items()}
    for l in reversed(range(cfg["n_layers"])):
        c = cache["layers"][l]
        dz = dh * mask[..., None] * activate_grad(c["z"], c["a"], cfg["act"])
        g, dh = equivariant_backward(dz, c["h"], mask, c["p"], params[f"eq{l}_Lambda"], params.get(f"eq{l}_Gamma"),
                                     cfg["pool"])  # fmt: skip
        grads |= {f"eq{l}_" + k: v for k, v in g.items()}
    return grads, dh * mask[..., None]


def equivariant_matrix(lam, gam, M):
    """Lemma 3: the M x M weight matrix lambda I + gamma 1 1^T of a one-channel equivariant layer."""
    return lam * np.eye(M) + gam * np.ones((M, M))
