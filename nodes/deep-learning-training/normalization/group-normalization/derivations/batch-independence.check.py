# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/group-normalization/ (ai://QFEJMP).
# Check of derivations/batch-independence.md: the group normalization output of one example does not change when the
# batch size or the other examples change; the batch normalization output does.
import numpy as np


def group_norm(x, g, eps=1e-5):
    n = x.shape[0]
    xg = x.reshape(n, g, -1)
    return ((xg - xg.mean(-1, keepdims=True)) / np.sqrt(xg.var(-1, keepdims=True) + eps)).reshape(x.shape)


def batch_norm(x, eps=1e-5):
    axes = (0,) + tuple(range(2, x.ndim))
    return (x - x.mean(axes, keepdims=True)) / np.sqrt(x.var(axes, keepdims=True) + eps)


def main():
    rng = np.random.default_rng(20261116)
    moved = 0
    for n in [2, 3, 8]:
        for c, g in [(4, 1), (4, 2), (8, 4), (8, 8)]:
            x = rng.normal(size=(n, c, 3, 3))
            gamma, beta = rng.normal(size=(1, c, 1, 1)), rng.normal(size=(1, c, 1, 1))
            full = group_norm(x, g) * gamma + beta
            for k in range(n):
                alone = group_norm(x[k:k + 1], g) * gamma + beta
                assert np.allclose(full[k:k + 1], alone, atol=1e-12)
                other = rng.normal(size=x.shape) * 7 - 3
                other[k] = x[k]
                assert np.allclose((group_norm(other, g) * gamma + beta)[k], alone[0], atol=1e-12)
                if np.max(np.abs(batch_norm(other)[k] - batch_norm(x)[k])) > 1e-3:
                    moved += 1
    assert moved > 0
    # the counterexample of step 2, with eps = 0
    assert np.isclose(batch_norm(np.array([0.0, 2.0]).reshape(2, 1, 1), 0.0)[0, 0, 0], -1.0)
    out = batch_norm(np.array([0.0, 2.0, 4.0]).reshape(3, 1, 1), 0.0)[0, 0, 0]
    assert np.isclose(out, -np.sqrt(1.5)) and not np.isclose(out, -1.0)
    print(f"ok: GN of an example ignores the rest of the batch; BN of the same example moved in {moved} cases")


if __name__ == "__main__":
    main()
