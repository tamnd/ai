# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/group-normalization/ (ai://NEW-group-normalization).
# Check of derivations/group-moments.md: within each group the normalized values have mean 0 and biased variance
# var / (var + eps); the channels inside a group need not.
import numpy as np


def group_norm(x, g, eps):
    xg = x.reshape(x.shape[0], g, -1)
    return ((xg - xg.mean(-1, keepdims=True)) / np.sqrt(xg.var(-1, keepdims=True) + eps)).reshape(x.shape)


def main():
    rng = np.random.default_rng(20261116)
    count = 0
    for c, g in [(4, 1), (4, 2), (6, 3), (8, 8), (32, 4)]:
        for eps in [0.0, 1e-5, 0.1, 4.0]:
            x = rng.normal(size=(3, c, 2, 5)) * rng.uniform(0.1, 5) + rng.normal()
            xg = x.reshape(3, g, -1)
            var = xg.var(-1)
            h = group_norm(x, g, eps).reshape(3, g, -1)
            assert np.allclose(h.mean(-1), 0.0, atol=1e-12)
            assert np.allclose(h.var(-1), var / (var + eps), rtol=1e-10)
            if eps > 0:
                assert np.all(h.var(-1) < 1.0)
            count += 1
    # the worked example: the group of channels 0 and 1 is (0, 0, 4, 4); each channel keeps its own mean
    x = np.array([[[0.0, 0.0], [4.0, 4.0], [0.0, 6.0], [6.0, 8.0]]])
    h = group_norm(x, 2, 0.0)
    assert np.allclose(h[0, :2], [[-1.0, -1.0], [1.0, 1.0]])
    assert np.allclose(h[0, :2].mean(), 0.0) and np.allclose(h[0, :2].var(), 1.0)
    assert np.allclose(h[0, 0].mean(), -1.0) and np.allclose(h[0, 0].var(), 0.0)
    print(f"ok: mean 0 and variance var / (var + eps) in every group, {count} cases; channels keep their own means")


if __name__ == "__main__":
    main()
