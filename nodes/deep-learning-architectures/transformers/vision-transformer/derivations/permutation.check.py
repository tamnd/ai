# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/vision-transformer/ (ai://NEW-vision-transformer).
"""Numeric check of the derivation permutation: without position embeddings a ViT is blind to the order of patches."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import vit  # noqa: E402

rng = np.random.default_rng(0)
worst, smallest_gap = 0.0, np.inf
# widths of at least 4: on one feature a layer normalization outputs its bias alone, and on two only a sign
for trial in range(40):
    G, P, C, H, Dh, F, L, K = (int(v) for v in rng.integers([2, 1, 1, 2, 2, 1, 1, 2], [5, 4, 3, 4, 4, 9, 4, 6]))
    pool = ("cls", "gap")[trial % 2]
    kw = dict(image_size=G * P, patch_size=P, channels=C, d_model=H * Dh, n_heads=H, d_ff=F, n_layers=L, n_classes=K,
              head_hidden=int(rng.integers(0, 3)), pool=pool)  # fmt: skip
    images = rng.normal(size=(3, G * P, G * P, C))
    perm = rng.permutation(G * G)
    while (perm == np.arange(G * G)).all():
        perm = rng.permutation(G * G)
    moved = vit.unpatchify(vit.patchify(images, P)[:, perm], P, C)
    cfg = vit.config(**kw, pos="none")
    params = vit.init(rng, cfg, scale=0.5)
    a, b = vit.forward(params, cfg, images)[0], vit.forward(params, cfg, moved)[0]
    worst = max(worst, float(np.abs(a - b).max() / np.abs(a).max()))
    learned = vit.config(**kw, pos="learned")
    params = vit.init(rng, learned, scale=0.5)
    a, b = vit.forward(params, learned, images)[0], vit.forward(params, learned, moved)[0]
    smallest_gap = min(smallest_gap, float(np.abs(a - b).max()))
assert worst < 1e-11, worst
assert smallest_gap > 1e-4, smallest_gap
print("ok")
