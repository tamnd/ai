# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/vision-transformer/ (ai://DV9G9W).
"""The Vision Transformer of Dosovitskiy et al. (2021), the NumPy reference: forward, backward and the counts.

An image (B, S, S, C) is cut into N = (S / P)^2 patches of P x P pixels in raster order, each flattened to P * P * C
numbers in the order (row, column, channel) and mapped to D numbers by one affine layer (eq-patches). A learned class
token is put in front, learned position embeddings are added, and L pre-norm encoder layers follow:
z' = z + MSA(LN(z)) (eq-attn) and z = z' + MLP(LN(z')) (eq-mlp), the MLP having an exact GELU between its two layers.
The class scores are an affine map of y = LN(z_L^0), the final state of the class token (eq-output); with
head_hidden > 0 the head is the pre-training MLP with one tanh hidden layer. pool="gap" drops the class token and
averages the normalized final states instead; pos="none" drops the position embeddings. Everything is float64.
"""

import math

import numpy as np

EPS = 1e-6  # of every layer normalization
POOLS = ("cls", "gap")
POSITIONS = ("learned", "none")
_erf = np.vectorize(math.erf, otypes=[float])


def config(image_size=8, patch_size=2, channels=3, d_model=8, n_heads=2, d_ff=16, n_layers=2, n_classes=5,
           head_hidden=0, pool="cls", pos="learned"):
    """Return the config of the network; the defaults are the example config of shapes.yaml."""
    if min(image_size, patch_size, channels, d_model, n_heads, d_ff, n_layers) < 1 or head_hidden < 0:
        raise ValueError("the sizes must be at least 1")
    if image_size % patch_size:
        raise ValueError("the patch size must divide the image size")
    if d_model % n_heads:
        raise ValueError("the number of heads must divide d_model")
    if n_classes < 2:
        raise ValueError("a classifier needs at least two classes")
    if pool not in POOLS or pos not in POSITIONS:
        raise ValueError(f"pool must be one of {POOLS} and pos one of {POSITIONS}")
    return {
        "image_size": image_size,
        "patch_size": patch_size,
        "channels": channels,
        "d_model": d_model,
        "n_heads": n_heads,
        "d_ff": d_ff,
        "n_layers": n_layers,
        "n_classes": n_classes,
        "head_hidden": head_hidden,
        "pool": pool,
        "pos": pos,
    }


def n_patches(cfg):
    return (cfg["image_size"] // cfg["patch_size"]) ** 2


def n_tokens(cfg):
    return n_patches(cfg) + (cfg["pool"] == "cls")


def param_shapes(cfg):
    """The name and shape of every parameter, in a fixed order."""
    P, C, D, F, K, Hh = (cfg[k] for k in ("patch_size", "channels", "d_model", "d_ff", "n_classes", "head_hidden"))
    shapes = {"embed_W": (P * P * C, D), "embed_b": (D,)}
    if cfg["pool"] == "cls":
        shapes["cls"] = (D,)
    if cfg["pos"] == "learned":
        shapes["pos"] = (n_tokens(cfg), D)
    for l in range(cfg["n_layers"]):
        shapes |= {
            f"l{l}_ln1_g": (D,), f"l{l}_ln1_b": (D,),
            f"l{l}_qkv_W": (D, 3 * D), f"l{l}_qkv_b": (3 * D,),
            f"l{l}_out_W": (D, D), f"l{l}_out_b": (D,),
            f"l{l}_ln2_g": (D,), f"l{l}_ln2_b": (D,),
            f"l{l}_fc1_W": (D, F), f"l{l}_fc1_b": (F,),
            f"l{l}_fc2_W": (F, D), f"l{l}_fc2_b": (D,),
        }  # fmt: skip
    shapes |= {"ln_g": (D,), "ln_b": (D,)}
    if Hh:
        shapes |= {"pre_W": (D, Hh), "pre_b": (Hh,), "head_W": (Hh, K), "head_b": (K,)}
    else:
        shapes |= {"head_W": (D, K), "head_b": (K,)}
    return shapes


def param_count(cfg):
    """eq-params, in closed form."""
    P, C, D, F, K, Hh, L = (cfg[k] for k in ("patch_size", "channels", "d_model", "d_ff", "n_classes", "head_hidden",
                                              "n_layers"))  # fmt: skip
    embed = P * P * C * D + D + D * (cfg["pool"] == "cls") + n_tokens(cfg) * D * (cfg["pos"] == "learned")
    layer = 4 * D * D + 2 * D * F + 9 * D + F
    head = 2 * D + (D * Hh + Hh + Hh * K + K if Hh else D * K + K)
    return embed + L * layer + head


def flops(cfg, B):
    """eq-flops: the FLOPs of the matrix products of a forward pass over B images, a multiply-add counted as 2."""
    P, C, D, F, K, Hh, L = (cfg[k] for k in ("patch_size", "channels", "d_model", "d_ff", "n_classes", "head_hidden",
                                              "n_layers"))  # fmt: skip
    N, n = n_patches(cfg), n_tokens(cfg)
    layer = 8 * n * D * D + 4 * n * n * D + 4 * n * D * F
    head = 2 * D * Hh + 2 * Hh * K if Hh else 2 * D * K
    return B * (2 * N * P * P * C * D + L * layer + head)


def init(rng, cfg, scale=0.1):
    """Random parameters: weights and biases N(0, scale^2), gains of the normalizations 1 + N(0, scale^2)."""
    params = {}
    for name, shape in param_shapes(cfg).items():
        params[name] = rng.normal(0.0, scale, shape) + (1.0 if name.endswith("_g") else 0.0)
    return params


def patchify(images, P):
    """(B, S, S, C) -> (B, N, P * P * C): the patches in raster order, each in the order (row, column, channel)."""
    B, S, S2, C = images.shape
    if S != S2 or S % P:
        raise ValueError("the images must be square, with a side that the patch size divides")
    G = S // P
    return images.reshape(B, G, P, G, P, C).transpose(0, 1, 3, 2, 4, 5).reshape(B, G * G, P * P * C)


def unpatchify(patches, P, C):
    """The inverse of patchify."""
    B, N, _ = patches.shape
    G = math.isqrt(N)
    return patches.reshape(B, G, G, P, P, C).transpose(0, 1, 3, 2, 4, 5).reshape(B, G * P, G * P, C)


def layer_norm(x, g, b):
    mu = x.mean(-1, keepdims=True)
    rstd = 1.0 / np.sqrt(((x - mu) ** 2).mean(-1, keepdims=True) + EPS)
    xhat = (x - mu) * rstd
    return xhat * g + b, (xhat, rstd, g)


def layer_norm_backward(dy, cache):
    xhat, rstd, g = cache
    axes = tuple(range(dy.ndim - 1))
    dg, db = (dy * xhat).sum(axes), dy.sum(axes)
    dxhat = dy * g
    dx = rstd * (dxhat - dxhat.mean(-1, keepdims=True) - xhat * (dxhat * xhat).mean(-1, keepdims=True))
    return dx, dg, db


def gelu(x):
    """The exact GELU, x Phi(x), with Phi the standard normal distribution function."""
    return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))


def gelu_grad(x):
    return 0.5 * (1.0 + _erf(x / math.sqrt(2.0))) + x * np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def softmax(s):
    e = np.exp(s - s.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def split_heads(x, H):
    B, n, D = x.shape
    return x.reshape(B, n, H, D // H).transpose(0, 2, 1, 3)


def merge_heads(x):
    B, H, n, d = x.shape
    return x.transpose(0, 2, 1, 3).reshape(B, n, H * d)


def msa(p, l, h, H):
    """eq-msa: multi-head self-attention of layer l on the normalized tokens h (B, n, D)."""
    D = h.shape[-1]
    qkv = h @ p[f"l{l}_qkv_W"] + p[f"l{l}_qkv_b"]
    q, k, v = (split_heads(qkv[..., i * D : (i + 1) * D], H) for i in range(3))
    a = softmax(q @ k.transpose(0, 1, 3, 2) / math.sqrt(D // H))
    o = merge_heads(a @ v)
    return o @ p[f"l{l}_out_W"] + p[f"l{l}_out_b"], {"h": h, "q": q, "k": k, "v": v, "weights": a, "o": o}


def msa_backward(p, l, dy, c):
    D = dy.shape[-1]
    d = D // c["q"].shape[1]
    g = {f"l{l}_out_W": np.einsum("bni,bnj->ij", c["o"], dy), f"l{l}_out_b": dy.sum((0, 1))}
    do = split_heads(dy @ p[f"l{l}_out_W"].T, c["q"].shape[1])
    a = c["weights"]
    da = do @ c["v"].transpose(0, 1, 3, 2)
    dv = a.transpose(0, 1, 3, 2) @ do
    ds = a * (da - (da * a).sum(-1, keepdims=True)) / math.sqrt(d)
    dq, dk = ds @ c["k"], ds.transpose(0, 1, 3, 2) @ c["q"]
    dqkv = np.concatenate([merge_heads(dq), merge_heads(dk), merge_heads(dv)], axis=-1)
    g[f"l{l}_qkv_W"] = np.einsum("bni,bnj->ij", c["h"], dqkv)
    g[f"l{l}_qkv_b"] = dqkv.sum((0, 1))
    return dqkv @ p[f"l{l}_qkv_W"].T, g


def embed(p, cfg, images):
    """eq-patches: the tokens z_0 (B, n, D) of a batch of images."""
    x = patchify(images, cfg["patch_size"])
    z = x @ p["embed_W"] + p["embed_b"]
    if cfg["pool"] == "cls":
        z = np.concatenate([np.broadcast_to(p["cls"], (z.shape[0], 1, z.shape[-1])), z], axis=1)
    if cfg["pos"] == "learned":
        z = z + p["pos"]
    return z, x


def forward(params, cfg, images):
    """The class scores (B, K) of a batch of images (B, S, S, C), and the cache of the backward pass."""
    p, H = params, cfg["n_heads"]
    z, x = embed(p, cfg, images)
    cache = {"patches": x, "tokens": z, "layers": []}
    for l in range(cfg["n_layers"]):
        h1, n1 = layer_norm(z, p[f"l{l}_ln1_g"], p[f"l{l}_ln1_b"])
        a, ca = msa(p, l, h1, H)
        z1 = z + a  # eq-attn
        h2, n2 = layer_norm(z1, p[f"l{l}_ln2_g"], p[f"l{l}_ln2_b"])
        u = h2 @ p[f"l{l}_fc1_W"] + p[f"l{l}_fc1_b"]
        gu = gelu(u)
        z = z1 + gu @ p[f"l{l}_fc2_W"] + p[f"l{l}_fc2_b"]  # eq-mlp
        cache["layers"].append({"n1": n1, "attn": ca, "n2": n2, "h2": h2, "u": u, "gu": gu})
    zn, nf = layer_norm(z, p["ln_g"], p["ln_b"])
    y = zn[:, 0] if cfg["pool"] == "cls" else zn.mean(1)  # eq-output
    cache |= {"final": z, "nf": nf, "y": y}
    if cfg["head_hidden"]:
        t = np.tanh(y @ p["pre_W"] + p["pre_b"])
        cache["t"] = t
        return t @ p["head_W"] + p["head_b"], cache
    return y @ p["head_W"] + p["head_b"], cache


def nll(logits, labels):
    """eq-loss: the cross-entropy of the labels, averaged over the batch."""
    m = logits.max(-1, keepdims=True)
    logp = logits - m - np.log(np.exp(logits - m).sum(-1, keepdims=True))
    return float(-logp[np.arange(len(labels)), labels].mean())


def loss(params, cfg, images, labels):
    return nll(forward(params, cfg, images)[0], labels)


def backward(params, cfg, cache, labels):
    """The gradient of eq-loss with respect to every parameter, and to the images."""
    p = params
    B = len(labels)
    logits = (cache["t"] if cfg["head_hidden"] else cache["y"]) @ p["head_W"] + p["head_b"]
    dlogits = softmax(logits)
    dlogits[np.arange(B), labels] -= 1.0
    dlogits /= B
    g = {}
    if cfg["head_hidden"]:
        t = cache["t"]
        g["head_W"], g["head_b"] = t.T @ dlogits, dlogits.sum(0)
        dpre = (dlogits @ p["head_W"].T) * (1.0 - t * t)
        g["pre_W"], g["pre_b"] = cache["y"].T @ dpre, dpre.sum(0)
        dy = dpre @ p["pre_W"].T
    else:
        g["head_W"], g["head_b"] = cache["y"].T @ dlogits, dlogits.sum(0)
        dy = dlogits @ p["head_W"].T
    n = cache["final"].shape[1]
    dzn = np.zeros_like(cache["final"])
    if cfg["pool"] == "cls":
        dzn[:, 0] = dy
    else:
        dzn += dy[:, None, :] / n
    dz, g["ln_g"], g["ln_b"] = layer_norm_backward(dzn, cache["nf"])
    for l in reversed(range(cfg["n_layers"])):
        c = cache["layers"][l]
        g[f"l{l}_fc2_W"] = np.einsum("bni,bnj->ij", c["gu"], dz)
        g[f"l{l}_fc2_b"] = dz.sum((0, 1))
        du = (dz @ p[f"l{l}_fc2_W"].T) * gelu_grad(c["u"])
        g[f"l{l}_fc1_W"] = np.einsum("bni,bnj->ij", c["h2"], du)
        g[f"l{l}_fc1_b"] = du.sum((0, 1))
        dh2, g[f"l{l}_ln2_g"], g[f"l{l}_ln2_b"] = layer_norm_backward(du @ p[f"l{l}_fc1_W"].T, c["n2"])
        dz1 = dz + dh2
        dh1, ga = msa_backward(p, l, dz1, c["attn"])
        g |= ga
        dx1, g[f"l{l}_ln1_g"], g[f"l{l}_ln1_b"] = layer_norm_backward(dh1, c["n1"])
        dz = dz1 + dx1
    if cfg["pos"] == "learned":
        g["pos"] = dz.sum(0)
    if cfg["pool"] == "cls":
        g["cls"] = dz[:, 0].sum(0)
        dz = dz[:, 1:]
    x = cache["patches"]
    g["embed_W"] = np.einsum("bni,bnj->ij", x, dz)
    g["embed_b"] = dz.sum((0, 1))
    dimages = unpatchify(dz @ p["embed_W"].T, cfg["patch_size"], cfg["channels"])
    return {k: g[k] for k in param_shapes(cfg)}, dimages


def resize_pos(pos, old_grid, new_grid, has_cls=True):
    """Section 3.2: the position embeddings of a G x G grid resized to a G' x G' grid by 2D bilinear interpolation
    of their location in the image (sample points at the centres of the new patches, the edges clamped); the
    embedding of the class token is kept."""
    head, grid = (pos[:1], pos[1:]) if has_cls else (pos[:0], pos)
    if grid.shape[0] != old_grid * old_grid:
        raise ValueError("the position embeddings do not match the old grid")
    w = interpolation_matrix(old_grid, new_grid)  # (G', G)
    D = grid.shape[-1]
    out = np.einsum("ir,js,rsd->ijd", w, w, grid.reshape(old_grid, old_grid, D))
    return np.concatenate([head, out.reshape(new_grid * new_grid, D)], axis=0)


def interpolation_matrix(n_in, n_out):
    """The (n_out, n_in) weights of linear interpolation with half-pixel centres, as torch's align_corners=False."""
    w = np.zeros((n_out, n_in))
    for i in range(n_out):
        src = max((i + 0.5) * n_in / n_out - 0.5, 0.0)
        i0 = min(int(src), n_in - 1)
        i1 = min(i0 + 1, n_in - 1)
        lam = src - i0
        w[i, i0] += 1.0 - lam
        w[i, i1] += lam
    return w


def embed_as_conv(W, P, C):
    """embed_W (P * P * C, D), flattened (row, column, channel), as the weight (D, C, P, P) of a convolution with a
    P x P kernel and stride P, the layout of torch.nn.Conv2d."""
    return W.reshape(P, P, C, -1).transpose(3, 2, 0, 1)


def embed_from_conv(w):
    """The inverse of embed_as_conv: a convolution weight (D, C, P, P) as embed_W (P * P * C, D)."""
    D, C, P, _ = w.shape
    return w.transpose(2, 3, 1, 0).reshape(P * P * C, D)
