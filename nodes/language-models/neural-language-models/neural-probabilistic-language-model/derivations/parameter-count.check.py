# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/neural-probabilistic-language-model/ (ai://NEW-neural-probabilistic-language-model).
"""Check of parameter-count: the free parameters of the neural probabilistic language model with and without direct
connections, the size of the AP News network, and the share of its multiply-adds spent in the output units."""


def sizes(V, m, h, n, direct=True):
    shapes = {"b": (V,), "d": (h,), "U": (V, h), "H": (h, (n - 1) * m), "C": (V, m)}
    if direct:
        shapes["W"] = (V, (n - 1) * m)
    total = 0
    for s in shapes.values():
        count = 1
        for k in s:
            count *= k
        total += count
    return total


def main():
    for V in (2, 7, 100, 16383, 17964):
        for m in (1, 30, 60, 100):
            for h in (0, 1, 50, 60, 100):
                for n in (2, 3, 5, 6):
                    assert sizes(V, m, h, n) == V * (1 + n * m + h) + h * (1 + (n - 1) * m)
                    assert sizes(V, m, h, n, False) == V * (1 + m + h) + h * (1 + (n - 1) * m)
    V, h, n, m = 17964, 60, 6, 100
    total = sizes(V, m, h, n)
    assert total == 11_904_264 and V * 661 == 11_874_204 and h * 501 == 30_060
    assert 8 * total == 95_234_112
    dominant = V * (n * m + h) / total
    assert round(100 * dominant, 1) == 99.6
    out = V * (1 + (n - 1) * m + h)
    hid = h * (1 + (n - 1) * m)
    feat = (n - 1) * m
    assert (out, hid, feat) == (10_077_804, 30_060, 500)
    share = out / (out + hid + feat)
    alt_out = V * (1 + n * m + h)
    alt = alt_out / (alt_out + h * (1 + n * m) + n * m)
    assert round(100 * share, 1) == 99.7 and round(100 * alt, 1) == 99.7
    print(
        f"parameter-count: AP News network {total:,} parameters, {8 * total / 1e6:.1f} MB as float64; "
        f"output share {share:.5f} ({alt:.5f} with nm in every term)"
    )


if __name__ == "__main__":
    main()
