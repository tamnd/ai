# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/mixed-precision-training/ (ai://NEW-mixed-precision-training).
# Check of derivations/loss-scaling.md: a loss scale S = 2^k moves the window of gradient magnitudes that FP16 keeps
# from (2^-25, 65520) to (2^-25 / S, 65520 / S); scaling by a power of two commutes with rounding and with the
# arithmetic of backpropagation while nothing overflows or leaves the normal range; unscaling in FP32 is exact; and a
# factor of 8 lifts the range [2^-27, 2^-24) that Micikevicius et al. found important into FP16.
import numpy as np


def f16(x):
    return np.asarray(x, dtype=np.float64).astype(np.float16).astype(np.float64)


def mlp_grads(x, w1, w2, upstream, dtype):
    """Gradients of a two-layer ReLU network y = relu(x w1) w2 for the upstream gradient dL/dy, all in dtype."""
    x, w1, w2, upstream = (a.astype(dtype) for a in (x, w1, w2, upstream))
    h = x @ w1
    a = np.maximum(h, 0)
    g_w2 = a.T @ upstream
    g_a = upstream @ w2.T
    g_h = g_a * (h > 0)
    g_w1 = x.T @ g_h
    return g_w1, g_w2


def main():
    np.seterr(over="ignore", under="ignore")
    rng = np.random.default_rng(20261116)

    # 1. The window of magnitudes kept, nonzero and finite, as a function of S = 2^k.
    assert f16(65519.99)[()] == 65504.0 and np.isinf(f16(65520.0))
    g = np.exp(rng.uniform(np.log(2.0**-40), np.log(2.0**20), 400000)) * rng.choice([-1.0, 1.0], 400000)
    for k in range(21):
        s = 2.0**k
        stored = f16(s * g)
        assert np.array_equal(stored != 0, np.abs(g) > 2.0**-25 / s)
        assert np.array_equal(np.isfinite(stored), np.abs(g) < 65520.0 / s)

    # 2. Inside the normal range, rounding commutes with the power-of-two scale, and the unscaled gradient has the
    # relative error of an FP16 normal, at most 2^-11, however small the gradient was.
    for k in (3, 7, 10, 15):
        s = 2.0**k
        sel = (np.abs(g) >= 2.0**-14) & (np.abs(s * g) <= 65504.0)
        assert np.array_equal(f16(s * g[sel]), s * f16(g[sel]))
        sel = (np.abs(s * g) >= 2.0**-14) & (np.abs(s * g) <= 65504.0)
        rel = np.abs(f16(s * g[sel]) / s - g[sel]) / np.abs(g[sel])
        assert rel.max() <= 2.0**-11

    # 3. Unscaling an FP16 gradient in FP32 by 1/S is exact for S = 2^k, k <= 102: the result stays normal in FP32.
    all16 = np.arange(0x0001, 0x7C00, dtype=np.uint16).view(np.float16)
    for k in (0, 1, 10, 24, 64, 102):
        inv = np.float32(2.0**-k)
        un = all16.astype(np.float32) * inv
        assert np.array_equal(un.astype(np.float64), all16.astype(np.float64) / 2.0**k)

    # 4. Backpropagation in FP16 with the upstream gradient multiplied by S = 2^k gives exactly S times the gradients,
    # as long as nothing overflows or becomes subnormal.
    x = rng.normal(size=(8, 5))
    w1 = rng.normal(size=(5, 6))
    w2 = rng.normal(size=(6, 3))
    d = rng.normal(size=(8, 3)) * 0.01
    base = mlp_grads(x, w1, w2, d, np.float16)
    for k in (1, 4, 8):
        scaled = mlp_grads(x, w1, w2, d * 2.0**k, np.float16)
        for b, sc in zip(base, scaled):
            nz = b != 0
            assert np.all(np.abs(b[nz]) >= 2.0**-14) and np.all(np.isfinite(sc))
            assert np.array_equal(sc.astype(np.float64), b.astype(np.float64) * 2.0**k)
    # With an upstream gradient of order 1e-7 FP16 loses the gradients; a scale of 2^12 keeps them, and after
    # unscaling in FP32 they are within FP16 precision of the float64 gradients.
    tiny = rng.normal(size=(8, 3)) * 1e-7
    ref = mlp_grads(x, w1, w2, tiny, np.float64)
    raw = mlp_grads(x, w1, w2, tiny, np.float16)
    s = 2.0**12
    rec = [a.astype(np.float32) * np.float32(1 / s) for a in mlp_grads(x, w1, w2, tiny * s, np.float16)]
    for r, a, c in zip(ref, raw, rec):
        err_raw = np.abs(a.astype(np.float64) - r).max() / np.abs(r).max()
        err_rec = np.abs(c.astype(np.float64) - r).max() / np.abs(r).max()
        assert err_raw > 0.05 and err_rec < 0.01, (err_raw, err_rec)

    # 5. Micikevicius et al.: scaling by 8 raises exponents by 3, so [2^-27, 2^-24) becomes [2^-24, 2^-21) and none
    # of it rounds to zero, while unscaled the part up to 2^-25 is lost.
    v = np.exp(rng.uniform(np.log(2.0**-27), np.log(2.0**-24), 100000))
    assert np.all(f16(8.0 * v) != 0) and np.all(f16(8.0 * v) >= 2.0**-24) and np.all(8.0 * v < 2.0**-21)
    assert np.mean(f16(v) == 0) > 0.6
    assert np.log2(8.0) == 3

    # 6. The largest power-of-two scale that cannot overflow for a known largest gradient: S = 2^floor(log2(65504/G)).
    for big in (1e-3, 0.5, 3.0, 700.0):
        s = 2.0 ** np.floor(np.log2(65504.0 / big))
        assert np.isfinite(f16(s * big)) and s * big <= 65504.0 and 2 * s * big > 65504.0
    print("ok: a loss scale 2^k shifts the FP16 window by k binades, exactly, and unscaling in FP32 is exact")


if __name__ == "__main__":
    main()
