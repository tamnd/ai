# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/mixup/ (ai://NEW-mixup).
# Check of derivations/cutmix-area.md: the CutMix label weight 1 - (x2 - x1)(y2 - y1)/(WH) is the share of pixels from
# x_A, clipping only raises it above the sampled lam, and with uniform lam and centre its mean is 163/240.
import math

import numpy as np


def box(w, h, lam, cx, cy, rounding=True):
    rw, rh = w * math.sqrt(1.0 - lam), h * math.sqrt(1.0 - lam)
    edges = [max(cx - rw / 2, 0.0), min(cx + rw / 2, w), max(cy - rh / 2, 0.0), min(cy + rh / 2, h)]
    if rounding:
        edges = [int(round(v)) for v in edges]
    return edges


def main():
    rng = np.random.default_rng(20261116)

    # 1 and 2 on the pixel grid.
    for _ in range(20000):
        w, h = int(rng.integers(1, 65)), int(rng.integers(1, 49))
        lam = float(rng.uniform())
        cx, cy = float(rng.uniform(0, w)), float(rng.uniform(0, h))
        x1, x2, y1, y2 = box(w, h, lam, cx, cy)
        mask = np.ones((h, w))
        mask[y1:y2, x1:x2] = 0.0
        xa, xb = rng.normal(size=(h, w)), rng.normal(size=(h, w))
        mixed = mask * xa + (1 - mask) * xb
        adjusted = 1.0 - (x2 - x1) * (y2 - y1) / (w * h)
        assert int((mask == 0).sum()) == (x2 - x1) * (y2 - y1)
        assert abs(mask.mean() - adjusted) < 1e-12
        assert np.array_equal(mixed, np.where(mask == 1.0, xa, xb))
        # Without rounding the clipped box never exceeds the sampled area.
        fx1, fx2, fy1, fy2 = box(w, h, lam, cx, cy, rounding=False)
        assert 1.0 - (fx2 - fx1) * (fy2 - fy1) / (w * h) >= lam - 1e-12
        # Rounding moves each edge by at most half a pixel.
        slack = (0.5 * 2 * (fy2 - fy1 + 1) + 0.5 * 2 * (fx2 - fx1 + 1)) / (w * h)
        assert adjusted >= lam - slack - 1e-12
        # A box well inside the image keeps the sampled area.
        if fx1 > 0 and fx2 < w and fy1 > 0 and fy2 < h:
            assert abs(1.0 - (fx2 - fx1) * (fy2 - fy1) / (w * h) - lam) < 1e-12

    # 3. The mean of the adjusted lam, by Monte Carlo on a 224 x 224 image and by quadrature of step 3.
    n, w, h = 200_000, 224, 224
    lam = rng.uniform(size=n)
    cx, cy = rng.uniform(0, w, size=n), rng.uniform(0, h, size=n)
    s = np.sqrt(1.0 - lam)
    x1 = np.round(np.clip(cx - w * s / 2, 0, None))
    x2 = np.round(np.clip(cx + w * s / 2, None, w))
    y1 = np.round(np.clip(cy - h * s / 2, 0, None))
    y2 = np.round(np.clip(cy + h * s / 2, None, h))
    adjusted = 1.0 - (x2 - x1) * (y2 - y1) / (w * h)
    assert abs(adjusted.mean() - 163 / 240) < 2e-3, adjusted.mean()
    assert abs(lam.mean() - 0.5) < 5e-3
    m = 400_000
    u = (np.arange(m) + 0.5) / m
    assert abs(((np.sqrt(u) - u / 4) ** 2).mean() - 77 / 240) < 1e-9
    c = (np.arange(m) + 0.5) / m
    for side in [0.1, 0.5, 0.9, 1.0]:
        clipped = np.minimum(c + side / 2, 1) - np.maximum(c - side / 2, 0)
        assert abs(clipped.mean() - (side - side**2 / 4)) < 1e-9

    # The example of the page.
    x1, x2, y1, y2 = box(32, 32, 0.75, 4.0, 28.0)
    assert (x1, x2, y1, y2) == (0, 12, 20, 32)
    assert 1.0 - (x2 - x1) * (y2 - y1) / 1024 == 0.859375
    print("ok: the CutMix weight 1 - box/(WH) is the share of pixels from x_A, >= lam, with mean 163/240 for alpha = 1")


if __name__ == "__main__":
    main()
