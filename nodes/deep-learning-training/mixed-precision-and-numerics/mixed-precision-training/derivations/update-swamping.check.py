# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/mixed-precision-training/ (ai://EPEFQ9).
# Check of derivations/update-swamping.md: under round to nearest even, FP16 rounds to zero exactly the magnitudes up
# to 2^-25, and an update u added to a normal weight w is lost whenever |w|/|u| > 2^(m+2) and never when
# |w|/|u| < 2^(m+1), with m = 10 for FP16 and m = 7 for BF16. Sums are formed exactly in float64 and rounded once.
import numpy as np


def to_bf16(x):
    """Round float64 values to BF16 with round to nearest even, through float32 bits; x must be exact in float32."""
    b = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    b = (b + 0x7FFF + ((b >> 16) & 1)) >> 16 << 16
    return b.astype(np.uint32).view(np.float32).astype(np.float64)


def lost_fp16(w, u):
    return (w + u).astype(np.float16).astype(np.float64) == w


def main():
    np.seterr(over="ignore")
    rng = np.random.default_rng(20261116)

    # 1. Underflow: zero exactly up to 2^-25 (the tie at 2^-25 goes to the even zero), 2^-24 just above it.
    t = 2.0**-25
    assert np.float16(t) == 0 and np.float16(np.nextafter(t, 1.0)) == 2.0**-24
    assert np.float16(-t) == 0 and np.float16(1.5 * t) == 2.0**-24
    # 1.5 * 2^-24 ties between the subnormals 1 and 2 times 2^-24 and goes to the even one, 2^-23.
    assert np.float16(1.5 * 2.0**-24) == 2.0**-23 and np.float16(np.nextafter(1.5 * 2.0**-24, 0.0)) == 2.0**-24
    xs = np.exp(rng.uniform(np.log(2.0**-30), np.log(2.0**-20), 200000))
    assert np.array_equal(xs.astype(np.float16) == 0, xs <= t)
    # Subnormals keep an absolute spacing of 2^-24, so the relative error of a value near 2^-24 is up to 1/2,
    # against 2^-11 for normal numbers.
    sub = xs[(xs > t) & (xs < 2.0**-14)]
    err = np.abs(sub.astype(np.float16).astype(np.float64) - sub)
    assert err.max() <= 2.0**-25 and (err / sub).max() > 0.3
    nor = np.exp(rng.uniform(np.log(2.0**-14), np.log(60000.0), 200000))
    assert (np.abs(nor.astype(np.float16).astype(np.float64) - nor) / nor).max() <= 2.0**-11
    # Figure 2b of Micikevicius et al.: an exponent below -24 means a magnitude below 2^-24, of which those above
    # 2^-25 round up to 2^-24 instead of vanishing.
    assert np.float16(2.0**-24.5) == 2.0**-24 and np.float16(2.0**-25.5) == 0

    # 2. Swamping, FP16: every positive normal weight against updates of both signs at random ratios.
    bits = np.arange(0x0400, 0x7C00, dtype=np.uint16)  # positive normal FP16 values
    w_all = bits.view(np.float16).astype(np.float64)
    for sign in (1.0, -1.0):
        w = np.repeat(w_all, 40)
        ratio = 2.0 ** rng.uniform(9.0, 14.0, w.size)
        u = (sign * w / ratio).astype(np.float16).astype(np.float64)  # the update itself is an FP16 number
        keep = u != 0
        w, u = w[keep], u[keep]
        r = w / np.abs(u)
        lost = lost_fp16(w, u)
        assert lost[r > 2.0**12].all(), "a ratio above 4096 must always lose the update"
        assert not lost[r < 2.0**11].any(), "a ratio below 2048 must never lose the update"
        between = (r >= 2.0**11) & (r <= 2.0**12)
        assert lost[between].any() and not lost[between].all()
    # The two edges: at w = 1 an update of 2^-11 (ratio 2048) ties and rounds back to 1, and 2^-11 + 2^-21 does not;
    # at w = 2 - 2^-10, the largest weight of that binade, an update just under 2^-11 (ratio about 4094) is lost,
    # while 2^-11 itself ties and rounds to the even neighbour 2.
    assert lost_fp16(np.array([1.0]), np.array([2.0**-11]))[0]
    assert not lost_fp16(np.array([1.0]), np.array([2.0**-11 + 2.0**-21]))[0]
    w_top, u_top = 2.0 - 2.0**-10, 2.0**-11 - 2.0**-21
    assert np.float16(u_top) == u_top and w_top / u_top > 4094
    assert lost_fp16(np.array([w_top]), np.array([u_top]))[0]
    assert np.float16(w_top + 2.0**-11) == 2.0

    # 3. The same bounds for BF16, m = 7: 256 and 512.
    w = to_bf16(np.exp(rng.uniform(np.log(1e-3), np.log(1e3), 300000)))
    ratio = 2.0 ** rng.uniform(6.0, 11.0, w.size)
    u = to_bf16(w / ratio) * rng.choice([-1.0, 1.0], w.size)
    r = w / np.abs(u)
    lost = to_bf16(w + u) == w
    assert lost[r > 2.0**9].all() and not lost[r < 2.0**8].any()
    # The BF16 rounding here agrees with float32 bits: 1 + 2^-8 ties to 1, 1 + 3 * 2^-8 ties up to 1 + 2^-6.
    assert to_bf16(np.array([1 + 2.0**-8, 1 + 3 * 2.0**-8]))[0] == 1.0
    assert to_bf16(np.array([1 + 3 * 2.0**-8]))[0] == 1 + 2.0**-6

    # 4. FP32 master weights: 1000 updates of 2^-12 to w = 1 (ratio 4096) leave an FP16 weight at 1 and move an
    # FP32 weight to 1 + 1000 * 2^-12 = 1.244140625 exactly.
    w16, w32 = np.float16(1.0), np.float32(1.0)
    for _ in range(1000):
        w16 = np.float16(np.float64(w16) + 2.0**-12)
        w32 = np.float32(w32 + np.float32(2.0**-12))
    assert w16 == 1.0 and w32 == 1.244140625
    print("ok: FP16 rounds |x| <= 2^-25 to zero; updates are lost above ratio 2^(m+2) and kept below 2^(m+1)")


if __name__ == "__main__":
    main()
