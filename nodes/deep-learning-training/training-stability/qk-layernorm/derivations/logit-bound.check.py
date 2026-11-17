# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/qk-layernorm/ (ai://NEW-qk-layernorm).
# Check of derivations/logit-bound.md: the norm sqrt(d) of a LayerNorm'd vector, the bound on the logit,
# its tightness, the floor and ceiling on the attention weights, and the quadratic growth without the norm.
import numpy as np


def ln(u, g=None, eps=0.0):
    c = u - u.mean(axis=-1, keepdims=True)
    y = c / np.sqrt((c**2).mean(axis=-1, keepdims=True) + eps)
    return y if g is None else y * g


def main():
    rng = np.random.default_rng(20261116)
    worst = 0.0
    for d in [2, 4, 16, 64, 128]:
        for scale in [1e-3, 1.0, 1e3, 1e6]:
            x = rng.normal(size=(9, 32))
            wq, wk = scale * rng.normal(size=(32, d)), scale * rng.normal(size=(32, d))
            q, k = ln(x @ wq), ln(x @ wk)
            # Step 1: the norm is sqrt(d) whatever the scale of the weights.
            assert np.allclose(np.linalg.norm(q, axis=-1), np.sqrt(d), rtol=1e-10)
            z = q @ k.T / np.sqrt(d)
            worst = max(worst, np.abs(z).max() / np.sqrt(d))
            assert np.abs(z).max() <= np.sqrt(d) * (1 + 1e-12)
            # Step 4: with gains, the bound is sqrt(d) max|g_q| max|g_k|.
            gq, gk = rng.normal(size=d) * 3, rng.normal(size=d)
            zg = ln(x @ wq, gq) @ ln(x @ wk, gk).T / np.sqrt(d)
            assert np.abs(zg).max() <= np.sqrt(d) * np.abs(gq).max() * np.abs(gk).max() * (1 + 1e-12)
            # Step 5: scaling W_Q and W_K by c leaves the normalized logits alone and scales the raw ones by c^2.
            assert np.allclose(ln(x @ (7 * wq)) @ ln(x @ (7 * wk)).T / np.sqrt(d), z, atol=1e-9)
            raw = (x @ wq) @ (x @ wk).T / np.sqrt(d)
            assert np.allclose((x @ (7 * wq)) @ (x @ (7 * wk)).T / np.sqrt(d), 49 * raw, rtol=1e-9)
            # Step 6: tight when the key is the query, and for constant gains a and b.
            assert np.allclose(np.diag(q @ q.T) / np.sqrt(d), np.sqrt(d), rtol=1e-12)
            a, b = 1.7, -0.6
            assert np.allclose(
                np.diag((a * q) @ (b * q).T) / np.sqrt(d),
                a * b * np.sqrt(d),
                rtol=1e-12,
            )
            # Step 7: eps > 0 gives a norm below sqrt(d).
            norms = np.linalg.norm(ln(x @ wq, eps=1e-2), axis=-1)
            assert np.all(norms <= np.sqrt(d) * (1 + 1e-15))
            if scale <= 1.0:
                assert np.all(norms < np.sqrt(d))
            # Step 9: without centering, u / rms(u) also has norm sqrt(d).
            u = x @ wq
            r = u / np.sqrt(np.mean(u * u, axis=-1, keepdims=True))
            assert np.allclose(np.linalg.norm(r, axis=-1), np.sqrt(d), rtol=1e-12)
            assert np.allclose(
                (7 * u) / np.sqrt(np.mean(49 * u * u, axis=-1, keepdims=True)),
                r,
                rtol=1e-12,
            )
    # Step 8: the floor and the ceiling of the weights at the extreme logits.
    for d, n in [(4, 3), (64, 512), (128, 4096)]:
        bnd = np.sqrt(d)
        z = np.full(n, bnd)
        z[0] = -bnd
        p = np.exp(z - z.max()) / np.exp(z - z.max()).sum()
        assert np.isclose(p[0], 1 / (1 + (n - 1) * np.exp(2 * bnd)), rtol=1e-9)
        z2 = -z
        p2 = np.exp(z2 - z2.max()) / np.exp(z2 - z2.max()).sum()
        assert np.isclose(p2[0], 1 / (1 + (n - 1) * np.exp(-2 * bnd)), rtol=1e-9)
        print(f"d={d} n={n}: weight floor {p[0]:.3e}, weight ceiling {p2[0]:.6f}")
    print(f"largest |z| / sqrt(d) over all draws: {worst:.4f} (bound 1)")
    print("logit-bound: ok")


if __name__ == "__main__":
    main()
