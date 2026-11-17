# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/group-normalization/ (ai://NEW-group-normalization).
# Check of derivations/special-cases.md: group normalization with G = 1 is layer normalization over (C, H, W), with
# G = C it is instance normalization over (H, W), and batch normalization of one example is instance normalization.
import numpy as np


def group_norm(x, g, eps):
    n, c = x.shape[:2]
    xg = x.reshape(n, g, -1)
    return ((xg - xg.mean(-1, keepdims=True)) / np.sqrt(xg.var(-1, keepdims=True) + eps)).reshape(x.shape)


def layer_norm(x, eps):
    f = x.reshape(x.shape[0], -1)
    return ((f - f.mean(1, keepdims=True)) / np.sqrt(f.var(1, keepdims=True) + eps)).reshape(x.shape)


def instance_norm(x, eps):
    m, v = x.mean((2, 3), keepdims=True), x.var((2, 3), keepdims=True)
    return (x - m) / np.sqrt(v + eps)


def batch_norm(x, eps):
    m, v = x.mean((0, 2, 3), keepdims=True), x.var((0, 2, 3), keepdims=True)
    return (x - m) / np.sqrt(v + eps)


def sets(shape, kind, g=None):
    n, c, h, w = shape
    out = {}
    for i in np.ndindex(*shape):
        members = []
        for k in np.ndindex(*shape):
            if kind == "bn":
                inside = k[1] == i[1]
            elif kind == "ln":
                inside = k[0] == i[0]
            elif kind == "in":
                inside = k[0] == i[0] and k[1] == i[1]
            else:
                inside = k[0] == i[0] and k[1] // (c // g) == i[1] // (c // g)
            if inside:
                members.append(k)
        out[i] = frozenset(members)
    return out


def main():
    rng = np.random.default_rng(20261116)
    cases = 0
    for n in [1, 2, 5]:
        for c in [1, 2, 4, 6, 8]:
            for h, w in [(1, 1), (2, 3), (4, 4)]:
                for eps in [0.0, 1e-5, 0.3]:
                    x = rng.normal(size=(n, c, h, w)) * 3 + rng.normal()
                    if eps == 0.0 and h * w == 1:
                        continue  # one position per channel has zero variance, and IN divides 0 by 0
                    assert np.allclose(group_norm(x, 1, eps), layer_norm(x, eps), atol=1e-12)
                    assert np.allclose(group_norm(x, c, eps), instance_norm(x, eps), atol=1e-12)
                    assert np.allclose(batch_norm(x[:1], eps), instance_norm(x[:1], eps), atol=1e-12)
                    cases += 1
    for shape in [(2, 4, 1, 2), (1, 6, 2, 1)]:
        assert sets(shape, "gn", 1) == sets(shape, "ln")
        assert sets(shape, "gn", shape[1]) == sets(shape, "in")
        assert sets((1,) + shape[1:], "bn") == sets((1,) + shape[1:], "in")
        assert sets(shape, "gn", 2) != sets(shape, "ln") and sets(shape, "gn", 2) != sets(shape, "in")
    print(f"ok: G = 1 is LN, G = C is IN, BN of one example is IN, on {cases} random NCHW tensors; the sets agree")


if __name__ == "__main__":
    main()
