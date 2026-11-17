# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/randaugment/ (ai://88YEPR).
"""RandAugment and TrivialAugment in NumPy.

The image operations act on uint8 arrays of shape (H, W) or (H, W, C): identity, rotate (nearest-neighbour, about the
centre), translate_x, translate_y, posterize, solarize and cutout. magnitude_to_arg maps an integer magnitude m in
{0, ..., m_max} linearly onto the range of each operation, as in eq-magnitude. randaugment_sample draws N operations
uniformly with replacement and pairs each with the one magnitude M, as in eq-randaugment. trivialaugment_sample draws
one operation and one magnitude uniformly, as in eq-trivialaugment. The counting functions are eq-aa-space,
eq-ra-grid, eq-hit and eq-cutout-area.
"""

import numpy as np

# The ranges at the largest magnitude, from the RandAugment space of Table 8 of Mueller and Hutter (2021).
# rotate in degrees, translate in pixels, posterize in bits removed (8 - 4 = 4 bits kept),
# solarize as the drop of the threshold from 256, cutout as the side over the shorter image side.
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
    img = np.asarray(img)
    if img.dtype != np.uint8 or img.ndim not in (2, 3):
        raise ValueError("an image is a uint8 array of shape (H, W) or (H, W, C)")
    return img


def identity(img):
    return _as_image(img).copy()


def rotate(img, degrees, fill=0):
    """Rotate counter-clockwise by `degrees` about the centre, with nearest-neighbour sampling."""
    img = _as_image(img)
    h, w = img.shape[:2]
    ci, cj = (h - 1) / 2.0, (w - 1) / 2.0
    t = np.deg2rad(degrees)
    i, j = np.mgrid[0:h, 0:w].astype(np.float64)
    x, y = j - cj, ci - i
    # Each output pixel takes the input pixel that the rotation moves onto it: the inverse rotation.
    xs = np.cos(t) * x + np.sin(t) * y
    ys = -np.sin(t) * x + np.cos(t) * y
    si = np.rint(ci - ys).astype(np.int64)
    sj = np.rint(xs + cj).astype(np.int64)
    inside = (si >= 0) & (si < h) & (sj >= 0) & (sj < w)
    out = np.full_like(img, fill)
    out[inside] = img[si[inside], sj[inside]]
    return out


def _shift(img, pixels, axis, fill):
    img = _as_image(img)
    p = int(pixels)
    out = np.full_like(img, fill)
    n = img.shape[axis]
    if abs(p) >= n:
        return out
    src = [slice(None)] * img.ndim
    dst = [slice(None)] * img.ndim
    src[axis] = slice(max(-p, 0), n - max(p, 0))
    dst[axis] = slice(max(p, 0), n - max(-p, 0))
    out[tuple(dst)] = img[tuple(src)]
    return out


def translate_x(img, pixels, fill=0):
    """Move the content `pixels` columns to the right (left when negative); the vacated columns get `fill`."""
    return _shift(img, pixels, 1, fill)


def translate_y(img, pixels, fill=0):
    """Move the content `pixels` rows down (up when negative); the vacated rows get `fill`."""
    return _shift(img, pixels, 0, fill)


def posterize(img, bits):
    """Keep the `bits` most significant bits of each 8-bit value."""
    img = _as_image(img)
    bits = int(bits)
    if not 1 <= bits <= 8:
        raise ValueError(f"bits must be in 1..8, got {bits}")
    return img & np.uint8((0xFF << (8 - bits)) & 0xFF)


def solarize(img, threshold):
    """Invert every value at or above `threshold`: v -> 255 - v."""
    img = _as_image(img)
    return np.where(img >= threshold, 255 - img, img).astype(np.uint8)


def cutout(img, size, center, fill=0):
    """Set a size x size square about `center` = (row, col) to `fill`, clipped at the border (DeVries and Taylor)."""
    img = _as_image(img)
    h, w = img.shape[:2]
    r, c = center
    half = int(size) // 2
    r0, r1 = min(max(r - half, 0), h), min(max(r + half, 0), h)
    c0, c1 = min(max(c - half, 0), w), min(max(c + half, 0), w)
    out = img.copy()
    out[r0:r1, c0:c1] = fill
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


def apply_op(img, op, m, rng, m_max=10):
    """Apply `op` at magnitude m. Rotations and translations get a random sign; cutout a random centre."""
    arg = magnitude_to_arg(op, m, m_max)
    if op in SIGNED and rng.random() < 0.5:
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
    h, w = np.asarray(img).shape[:2]
    size = int(round(arg * min(h, w)))
    return cutout(img, size, (int(rng.integers(h)), int(rng.integers(w))))


def randaugment_sample(ops, n, m, rng):
    """eq: eq-randaugment. N operations drawn uniformly with replacement, each paired with the magnitude M."""
    if n < 0:
        raise ValueError(f"N must be >= 0, got {n}")
    idx = rng.integers(len(ops), size=n)
    return [(ops[k], m) for k in idx]


def randaugment(img, n, m, rng, ops=OPS, m_max=10):
    """Apply the N sampled operations in order, all at magnitude M."""
    out = _as_image(img)
    for op, mag in randaugment_sample(ops, n, m, rng):
        out = apply_op(out, op, mag, rng, m_max)
    return out


def trivialaugment_sample(ops, rng, m_max=30):
    """eq: eq-trivialaugment. One operation and one magnitude in {0, ..., m_max}, both uniform."""
    return ops[int(rng.integers(len(ops)))], int(rng.integers(m_max + 1))


def trivialaugment(img, rng, ops=OPS, m_max=30):
    op, m = trivialaugment_sample(ops, rng, m_max)
    return apply_op(img, op, m, rng, m_max)


def autoaugment_space(n_ops=16, n_mag=10, n_prob=11, ops_per_sub=2, n_sub=5):
    """eq: eq-aa-space. The number of policies: (ops x magnitudes x probabilities) ** (operations per policy)."""
    return (n_ops * n_mag * n_prob) ** (ops_per_sub * n_sub)


def randaugment_grid(n_values, m_values):
    """eq: eq-ra-grid. The number of (N, M) settings of a grid search."""
    return len(set(n_values)) * len(set(m_values))


def prob_applied(k, n):
    """eq: eq-hit. The probability that one given operation of K is drawn at least once in N uniform draws."""
    return 1.0 - (1.0 - 1.0 / k) ** n


def cutout_expected_area(s, length):
    """eq: eq-cutout-area. The mean number of masked pixels of an s x s cutout, s even, 0 <= s <= 2L, on L x L."""
    if s % 2 or not 0 <= s <= 2 * length:
        raise ValueError("s must be even with 0 <= s <= 2L")
    return (s - s * s / (4.0 * length)) ** 2
