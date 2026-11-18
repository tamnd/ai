# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/neural-turing-machine/ (ai://NEW-neural-turing-machine).
"""Numeric check of the derivation addressing-keeps-a-weighting."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import ntm  # noqa: E402

rng = np.random.default_rng(0)


def is_weighting(w):
    return np.all(w >= 0) and np.allclose(w.sum(-1), 1.0)


for n, m, s_max in ((4, 3, 1), (7, 2, 2), (5, 5, 2), (12, 4, 1)):
    cfg = ntm.config(n_loc=n, width=m, shift=s_max)
    for _ in range(50):
        mem = rng.normal(size=(3, n, m)) * rng.uniform(0.1, 3)
        w_prev = ntm.softmax(rng.normal(size=(3, n)) * 4)
        head = ntm.split_heads(rng.normal(size=(3, ntm.n_head_outputs(cfg))) * 3, cfg)[0]
        wc = ntm.content_weights(mem, head["key"], head["beta"])
        wg = ntm.interpolate(wc, w_prev, head["g"])
        ws = ntm.shift(wg, head["s"])
        w = ntm.sharpen(ws, head["gamma"])
        # Statement 1: every stage gives a weighting.
        assert all(is_weighting(v) for v in (wc, wg, ws, w))
        np.testing.assert_allclose(w, ntm.address(mem, w_prev, head))

# Statement 2: beta = 0 gives the uniform weighting, a large beta the uniform weighting on the best rows.
mem = rng.normal(size=(1, 6, 3))
key = rng.normal(size=(1, 3))
np.testing.assert_allclose(ntm.content_weights(mem, key, np.zeros(1)), 1 / 6)
mem[0, 1] = mem[0, 4] = 2.0 * key[0]  # two equal rows in the direction of the key: a tie at cosine 1
lim = ntm.content_weights(mem, key, np.array([1e5]))
np.testing.assert_allclose(lim[0, [1, 4]], 0.5, atol=1e-9)
assert lim[0, [0, 2, 3, 5]].max() < 1e-9
print("addressing-keeps-a-weighting: ok")
