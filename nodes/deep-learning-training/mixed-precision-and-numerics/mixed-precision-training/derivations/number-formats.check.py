# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/mixed-precision-training/ (ai://NEW-mixed-precision-training).
# Check of derivations/number-formats.md: the machine epsilon, smallest normal, smallest subnormal and largest finite
# value of a binary format with e exponent bits and m mantissa bits, for FP32, FP16, BF16 and the two FP8 encodings,
# found by decoding every bit pattern of the 8- and 16-bit formats and compared with numpy.finfo and with the tables of
# Kalamkar et al. (2019) and Micikevicius et al. (2022).
import math

import numpy as np


def formula(e, m):
    bias = 2 ** (e - 1) - 1
    e_min, e_max = 1 - bias, 2**e - 2 - bias
    return {
        "eps": 2.0**-m,
        "min_normal": 2.0**e_min,
        "min_subnormal": 2.0 ** (e_min - m),
        "max": (2.0 - 2.0**-m) * 2.0**e_max,
    }


def decode(bits, e, m, ieee=True):
    """The value of a sign-exponent-mantissa bit pattern, IEEE 754 style; with ieee=False the top exponent holds
    normal numbers and only the all-ones pattern is NaN, as in E4M3."""
    bias = 2 ** (e - 1) - 1
    sign = -1.0 if (bits >> (e + m)) & 1 else 1.0
    exp = (bits >> m) & (2**e - 1)
    man = bits & (2**m - 1)
    if exp == 2**e - 1:
        if ieee:
            return sign * math.inf if man == 0 else math.nan
        if man == 2**m - 1:
            return math.nan
    if exp == 0:
        return sign * man * 2.0 ** (1 - bias - m)
    return sign * (1 + man / 2**m) * 2.0 ** (exp - bias)


def enumerate_format(e, m, ieee=True):
    vals = np.array([decode(b, e, m, ieee) for b in range(2 ** (1 + e + m))])
    pos = vals[np.isfinite(vals) & (vals > 0)]
    normals = pos[pos >= 2.0 ** (2 - 2 ** (e - 1))]
    one = np.searchsorted(np.sort(pos), 1.0)
    srt = np.sort(pos)
    return {
        "eps": srt[one + 1] - 1.0,
        "min_normal": normals.min(),
        "min_subnormal": pos.min(),
        "max": pos.max(),
        "has_inf": bool(np.isinf(vals).any()),
    }


def binades(lo, hi):
    """The number of binades [2^k, 2^(k+1)) that hold a positive finite value between lo and hi."""
    return math.floor(math.log2(hi)) - math.floor(math.log2(lo)) + 1


def main():
    np.seterr(invalid="ignore")
    # FP16 and FP32 against numpy, and FP16 by decoding all 65536 patterns, also through numpy's own decoder.
    for e, m, dt in [(5, 10, np.float16), (8, 23, np.float32)]:
        f, fi = formula(e, m), np.finfo(dt)
        assert f["eps"] == fi.eps and f["min_normal"] == fi.smallest_normal and f["max"] == fi.max
        assert f["min_subnormal"] == fi.smallest_subnormal
    f16 = formula(5, 10)
    assert enumerate_format(5, 10) == {**f16, "has_inf": True}
    all16 = np.arange(2**16, dtype=np.uint16).view(np.float16).astype(np.float64)
    pos16 = all16[np.isfinite(all16) & (all16 > 0)]
    assert pos16.min() == 2.0**-24 and pos16.max() == 65504.0
    assert (f16["eps"], f16["min_normal"], f16["min_subnormal"]) == (
        2.0**-10,
        2.0**-14,
        2.0**-24,
    )

    # BF16: the top 16 bits of an FP32 number; decode all patterns through float32.
    bf = formula(8, 7)
    assert enumerate_format(8, 7) == {**bf, "has_inf": True}
    all_bf = (np.arange(2**16, dtype=np.uint32) << 16).view(np.float32).astype(np.float64)
    pos_bf = all_bf[np.isfinite(all_bf) & (all_bf > 0)]
    assert pos_bf.min() == 2.0**-133 == bf["min_subnormal"] and pos_bf.max() == bf["max"]
    assert bf["min_normal"] == np.finfo(np.float32).smallest_normal == 2.0**-126
    assert bf["eps"] == 2.0**-7
    assert abs(bf["max"] - 3.3895313892515355e38) < 1e24
    # Same exponent range as FP32: the largest values share the exponent 127 and differ in the mantissa, by about 2^-8.
    ratio = float(np.finfo(np.float32).max) / bf["max"]
    assert ratio == (2.0 - 2.0**-23) / (2.0 - 2.0**-7) and 2.0**-8 < ratio - 1.0 < 1.01 * 2.0**-8

    # Kalamkar et al., Table 1, to three significant figures. Their BF16 maximum, 3.38e38, is the true value
    # 3.3895e38 cut to three figures; rounded it is 3.39e38. Their FP16 maximum 6.55e4 is 65504 rounded.
    sig3 = lambda x: float(f"{x:.2e}")
    assert sig3(65504.0) == 6.55e4 and sig3(2.0**-14) == 6.10e-5 and sig3(2.0**-24) == 5.96e-8
    assert sig3(np.finfo(np.float32).max) == 3.40e38 and sig3(2.0**-126) == 1.18e-38 and sig3(2.0**-149) == 1.40e-45
    assert math.floor(2.0**-126 / 1e-40) / 100 == 1.17  # Kalamkar print 1.17e-38, cut rather than rounded
    assert sig3(bf["max"]) == 3.39e38 and math.floor(bf["max"] / 1e36) / 100 == 3.38

    # Precision in decimal digits: log10 of 2^(m+1).
    assert round((10 + 1) * math.log10(2), 2) == 3.31 and round((7 + 1) * math.log10(2), 2) == 2.41

    # FP16 normal exponents run from -14 to 15, the range Micikevicius et al. (2018) give; 40 binades in all.
    assert math.log2(f16["min_normal"]) == -14 and math.floor(math.log2(f16["max"])) == 15
    assert binades(f16["min_subnormal"], f16["max"]) == 40

    # FP8, Table 1 of Micikevicius et al. (2022).
    e5m2 = enumerate_format(5, 2)
    assert e5m2 == {**formula(5, 2), "has_inf": True}
    assert (e5m2["max"], e5m2["min_normal"], e5m2["min_subnormal"]) == (
        57344.0,
        2.0**-14,
        2.0**-16,
    )
    assert 57344.0 == 1.75 * 2**15
    e4m3 = enumerate_format(4, 3, ieee=False)
    assert not e4m3["has_inf"]
    assert (e4m3["max"], e4m3["min_normal"], e4m3["min_subnormal"]) == (
        448.0,
        2.0**-6,
        2.0**-9,
    )
    assert 448.0 == 1.75 * 2**8
    vals = np.array([decode(b, 4, 3, ieee=False) for b in range(256)])
    assert np.isnan(vals).sum() == 2  # S.1111.111 for both signs
    # Without reclaiming the top exponent E4M3 would stop at 240; reclaiming adds seven magnitudes and one binade.
    ieee_e4m3 = enumerate_format(4, 3, ieee=True)
    assert ieee_e4m3["max"] == 240.0
    extra = sorted(v for v in vals if np.isfinite(v) and v > 240.0)
    assert extra == [256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0]
    assert binades(2.0**-9, 240.0) == 17 and binades(2.0**-9, 448.0) == 18
    assert binades(2.0**-16, 57344.0) == 32
    # E5M2 is FP16 with the low 8 mantissa bits dropped: every E5M2 value is an FP16 value.
    e5 = np.array([decode(b, 5, 2) for b in range(256)])
    fin = e5[np.isfinite(e5)]
    assert np.array_equal(fin.astype(np.float16).astype(np.float64), fin)
    print("ok: eps, smallest normal and subnormal, and max of FP32, FP16, BF16, E5M2 and E4M3")


if __name__ == "__main__":
    main()
