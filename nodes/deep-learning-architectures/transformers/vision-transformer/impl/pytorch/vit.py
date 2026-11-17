# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/vision-transformer/ (ai://NEW-vision-transformer).
"""The Vision Transformer of Dosovitskiy et al. (2021), the PyTorch version, with the parameters of the NumPy
reference.

An image (B, S, S, C) is cut into (S / P)^2 patches of P x P pixels in raster order, each flattened in the order
(row, column, channel) and mapped to D numbers by one affine layer; a class token goes in front, learned position
embeddings are added, and L pre-norm encoder layers with an exact GELU follow. The class scores are an affine map of
the normalized final state of the class token, or of the average of the patches with pool="gap"; pos="none" drops
the position embeddings. encoder_layer() returns the torch.nn.TransformerEncoderLayer with the weights of a layer,
and embed_conv() the patch embedding as the weight and bias of a strided convolution.
"""

import math

import torch
import torch.nn.functional as F

EPS = 1e-6  # of every layer normalization
POOLS = ("cls", "gap")
POSITIONS = ("learned", "none")


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


def n_tokens(cfg):
    return (cfg["image_size"] // cfg["patch_size"]) ** 2 + (cfg["pool"] == "cls")


def param_shapes(cfg):
    """The name and shape of every parameter, in the order of the NumPy reference."""
    P, C, D, F_, K, Hh = (cfg[k] for k in ("patch_size", "channels", "d_model", "d_ff", "n_classes", "head_hidden"))
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
            f"l{l}_fc1_W": (D, F_), f"l{l}_fc1_b": (F_,),
            f"l{l}_fc2_W": (F_, D), f"l{l}_fc2_b": (D,),
        }  # fmt: skip
    shapes |= {"ln_g": (D,), "ln_b": (D,)}
    if Hh:
        shapes |= {"pre_W": (D, Hh), "pre_b": (Hh,), "head_W": (Hh, K), "head_b": (K,)}
    else:
        shapes |= {"head_W": (D, K), "head_b": (K,)}
    return shapes


def param_count(cfg):
    return sum(math.prod(s) for s in param_shapes(cfg).values())


def patchify(images, P):
    """(B, S, S, C) -> (B, N, P * P * C): the patches in raster order, each in the order (row, column, channel)."""
    B, S, _, C = images.shape
    G = S // P
    return images.reshape(B, G, P, G, P, C).permute(0, 1, 3, 2, 4, 5).reshape(B, G * G, P * P * C)


def unpatchify(patches, P, C):
    """The inverse of patchify."""
    B, N, _ = patches.shape
    G = math.isqrt(N)
    return patches.reshape(B, G, G, P, P, C).permute(0, 1, 3, 2, 4, 5).reshape(B, G * P, G * P, C)


def resize_pos(pos, old_grid, new_grid, has_cls=True):
    """The position embeddings of a G x G grid resized to G' x G' by bilinear interpolation, with
    torch.nn.functional.interpolate; the embedding of the class token is kept."""
    head, grid = (pos[:1], pos[1:]) if has_cls else (pos[:0], pos)
    D = grid.shape[-1]
    img = grid.reshape(1, old_grid, old_grid, D).permute(0, 3, 1, 2)
    out = F.interpolate(img, size=(new_grid, new_grid), mode="bilinear", align_corners=False)
    return torch.cat([head, out.permute(0, 2, 3, 1).reshape(new_grid * new_grid, D)], 0)


class ViT(torch.nn.Module):
    """The Vision Transformer, its parameters named as in the NumPy reference."""

    def __init__(self, cfg, scale=0.1, generator=None):
        super().__init__()
        self.cfg = cfg
        self.p = torch.nn.ParameterDict()
        for name, shape in param_shapes(cfg).items():
            w = torch.randn(shape, generator=generator) * scale
            self.p[name] = torch.nn.Parameter(w + 1.0 if name.endswith("_g") else w)

    @classmethod
    def from_params(cls, params, cfg):
        model = cls(cfg)
        with torch.no_grad():
            for name, value in params.items():
                model.p[name].copy_(torch.as_tensor(value))
        return model

    def embed(self, images):
        p, cfg = self.p, self.cfg
        z = patchify(images, cfg["patch_size"]) @ p["embed_W"] + p["embed_b"]
        if cfg["pool"] == "cls":
            z = torch.cat([p["cls"].expand(z.shape[0], 1, -1), z], 1)
        if cfg["pos"] == "learned":
            z = z + p["pos"]
        return z

    def layer(self, l, z):
        p, H = self.p, self.cfg["n_heads"]
        B, n, D = z.shape
        h = F.layer_norm(z, (D,), p[f"l{l}_ln1_g"], p[f"l{l}_ln1_b"], EPS)
        q, k, v = (h @ p[f"l{l}_qkv_W"] + p[f"l{l}_qkv_b"]).reshape(B, n, 3, H, D // H).permute(2, 0, 3, 1, 4)
        a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(D // H), -1)
        z = z + (a @ v).transpose(1, 2).reshape(B, n, D) @ p[f"l{l}_out_W"] + p[f"l{l}_out_b"]
        h = F.layer_norm(z, (D,), p[f"l{l}_ln2_g"], p[f"l{l}_ln2_b"], EPS)
        return z + F.gelu(h @ p[f"l{l}_fc1_W"] + p[f"l{l}_fc1_b"]) @ p[f"l{l}_fc2_W"] + p[f"l{l}_fc2_b"]

    def forward(self, images):
        p, cfg = self.p, self.cfg
        z = self.embed(images)
        for l in range(cfg["n_layers"]):
            z = self.layer(l, z)
        z = F.layer_norm(z, (cfg["d_model"],), p["ln_g"], p["ln_b"], EPS)
        y = z[:, 0] if cfg["pool"] == "cls" else z.mean(1)
        if cfg["head_hidden"]:
            y = torch.tanh(y @ p["pre_W"] + p["pre_b"])
        return y @ p["head_W"] + p["head_b"]

    def encoder_layer(self, l):
        """Layer l as a torch.nn.TransformerEncoderLayer: pre-norm, GELU, batch first, no dropout."""
        p, D = self.p, self.cfg["d_model"]
        t = torch.nn.TransformerEncoderLayer(D, self.cfg["n_heads"], self.cfg["d_ff"], dropout=0.0, activation="gelu",
                                             layer_norm_eps=EPS, batch_first=True, norm_first=True,
                                             dtype=p["embed_W"].dtype)  # fmt: skip
        with torch.no_grad():
            t.self_attn.in_proj_weight.copy_(p[f"l{l}_qkv_W"].T)
            t.self_attn.in_proj_bias.copy_(p[f"l{l}_qkv_b"])
            t.self_attn.out_proj.weight.copy_(p[f"l{l}_out_W"].T)
            t.self_attn.out_proj.bias.copy_(p[f"l{l}_out_b"])
            t.linear1.weight.copy_(p[f"l{l}_fc1_W"].T)
            t.linear1.bias.copy_(p[f"l{l}_fc1_b"])
            t.linear2.weight.copy_(p[f"l{l}_fc2_W"].T)
            t.linear2.bias.copy_(p[f"l{l}_fc2_b"])
            for i in (1, 2):
                getattr(t, f"norm{i}").weight.copy_(p[f"l{l}_ln{i}_g"])
                getattr(t, f"norm{i}").bias.copy_(p[f"l{l}_ln{i}_b"])
        return t

    def embed_conv(self):
        """The patch embedding as the weight (D, C, P, P) and bias (D,) of a convolution of stride P."""
        P, C = self.cfg["patch_size"], self.cfg["channels"]
        return self.p["embed_W"].reshape(P, P, C, -1).permute(3, 2, 0, 1), self.p["embed_b"]


def nll(logits, labels):
    """eq-loss: the cross-entropy of the labels, averaged over the batch."""
    return F.cross_entropy(logits, labels)
