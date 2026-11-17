# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/randaugment/ (ai://NEW-randaugment).
"""RandAugment and TrivialAugment in PyTorch.

The same operations and samplers as the NumPy reference, on uint8 tensors of shape (H, W) or (C, H, W) and with a
torch.Generator for every random draw. The image operations are identity, rotate (nearest-neighbour, about the
centre), translate_x, translate_y, posterize, solarize and cutout. magnitude_to_arg is eq-magnitude,
randaugment_sample is eq-randaugment and trivialaugment_sample is eq-trivialaugment.
"""

import math

import torch

# The ranges at the largest magnitude, from the RandAugment space of Table 8 of Mueller and Hutter (2021).
RANGES = {
    "identity": 0.0,
    "rotate": 30.0,
    "translate_x": 10.0,
    "translate_y": 10.0,
    "posterize": 4.0,
    "solarize": 256.0,
    "cutout": 0.2,
}
SIGNED = ("rotate", "translate_x", "translate_y")
OPS = tuple(RANGES)


def _as_image(img):
    if not isinstance(img, torch.Tensor) or img.dtype != torch.uint8 or img.dim() not in (2, 3):
        raise ValueError("an image is a uint8 tensor of shape (H, W) or (C, H, W)")
    return img


def identity(img):
    return _as_image(img).clone()


def rotate(img, degrees, fill=0):
    """Rotate counter-clockwise by `degrees` about the centre, with nearest-neighbour sampling."""
    img = _as_image(img)
    h, w = img.shape[-2:]
    ci, cj = (h - 1) / 2.0, (w - 1) / 2.0
    t = math.radians(degrees)
    i, j = torch.meshgrid(
        torch.arange(h, dtype=torch.float64),
        torch.arange(w, dtype=torch.float64),
        indexing="ij",
    )
    x, y = j - cj, ci - i
    # Each output pixel takes the input pixel that the rotation moves onto it: the inverse rotation.
    xs = math.cos(t) * x + math.sin(t) * y
    ys = -math.sin(t) * x + math.cos(t) * y
    si = torch.round(ci - ys).long()
    sj = torch.round(xs + cj).long()
    inside = (si >= 0) & (si < h) & (sj >= 0) & (sj < w)
    out = torch.full_like(img, fill)
    out[..., inside] = img[..., si[inside], sj[inside]]
    return out


def _shift(img, pixels, dim, fill):
    img = _as_image(img)
    p = int(pixels)
    out = torch.full_like(img, fill)
    n = img.shape[dim]
    if abs(p) >= n:
        return out
    src = img.narrow(dim, max(-p, 0), n - abs(p))
    out.narrow(dim, max(p, 0), n - abs(p)).copy_(src)
    return out


def translate_x(img, pixels, fill=0):
    """Move the content `pixels` columns to the right (left when negative); the vacated columns get `fill`."""
    return _shift(img, pixels, img.dim() - 1, fill)


def translate_y(img, pixels, fill=0):
    """Move the content `pixels` rows down (up when negative); the vacated rows get `fill`."""
    return _shift(img, pixels, img.dim() - 2, fill)


def posterize(img, bits):
    """Keep the `bits` most significant bits of each 8-bit value."""
    img = _as_image(img)
    bits = int(bits)
    if not 1 <= bits <= 8:
        raise ValueError(f"bits must be in 1..8, got {bits}")
    return img & ((0xFF << (8 - bits)) & 0xFF)


def solarize(img, threshold):
    """Invert every value at or above `threshold`: v -> 255 - v."""
    img = _as_image(img)
    # Compare in float64: a uint8 tensor compared with 256 would wrap the threshold to 0.
    return torch.where(img.double() >= threshold, 255 - img, img)


def cutout(img, size, center, fill=0):
    """Set a size x size square about `center` = (row, col) to `fill`, clipped at the border (DeVries and Taylor)."""
    img = _as_image(img)
    h, w = img.shape[-2:]
    r, c = center
    half = int(size) // 2
    r0, r1 = min(max(r - half, 0), h), min(max(r + half, 0), h)
    c0, c1 = min(max(c - half, 0), w), min(max(c + half, 0), w)
    out = img.clone()
    out[..., r0:r1, c0:c1] = fill
    return out


def magnitude_to_arg(op, m, m_max=10):
    """eq: eq-magnitude. The argument of `op` at magnitude m: RANGES[op] * m / m_max, then the form each op needs."""
    if op not in RANGES:
        raise ValueError(f"unknown operation {op}")
    if not 0 <= m <= m_max:
        raise ValueError(f"the magnitude must be in [0, {m_max}], got {m}")
    v = RANGES[op] * m / m_max
    if op == "posterize":
        return 8 - int(round(v))
    if op == "solarize":
        return 256.0 - v
    return v


def _randint(n, gen):
    return int(torch.randint(n, (1,), generator=gen))


def apply_op(img, op, m, gen, m_max=10):
    """Apply `op` at magnitude m. Rotations and translations get a random sign; cutout a random centre."""
    arg = magnitude_to_arg(op, m, m_max)
    if op in SIGNED and float(torch.rand((), generator=gen)) < 0.5:
        arg = -arg
    if op == "identity":
        return identity(img)
    if op == "rotate":
        return rotate(img, arg)
    if op == "translate_x":
        return translate_x(img, round(arg))
    if op == "translate_y":
        return translate_y(img, round(arg))
    if op == "posterize":
        return posterize(img, arg)
    if op == "solarize":
        return solarize(img, arg)
    h, w = img.shape[-2:]
    size = int(round(arg * min(h, w)))
    return cutout(img, size, (_randint(h, gen), _randint(w, gen)))


def randaugment_sample(ops, n, m, gen):
    """eq: eq-randaugment. N operations drawn uniformly with replacement, each paired with the magnitude M."""
    if n < 0:
        raise ValueError(f"N must be >= 0, got {n}")
    idx = torch.randint(len(ops), (n,), generator=gen)
    return [(ops[int(k)], m) for k in idx]


def randaugment(img, n, m, gen, ops=OPS, m_max=10):
    """Apply the N sampled operations in order, all at magnitude M."""
    out = _as_image(img)
    for op, mag in randaugment_sample(ops, n, m, gen):
        out = apply_op(out, op, mag, gen, m_max)
    return out


def trivialaugment_sample(ops, gen, m_max=30):
    """eq: eq-trivialaugment. One operation and one magnitude in {0, ..., m_max}, both uniform."""
    return ops[_randint(len(ops), gen)], _randint(m_max + 1, gen)


def trivialaugment(img, gen, ops=OPS, m_max=30):
    op, m = trivialaugment_sample(ops, gen, m_max)
    return apply_op(img, op, m, gen, m_max)
