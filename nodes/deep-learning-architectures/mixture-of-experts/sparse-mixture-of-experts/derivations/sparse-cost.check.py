# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/sparse-mixture-of-experts/ (ai://NEW-sparse-mixture-of-experts).
# Check of derivations/sparse-cost.md: a layer of N two-layer experts d -> h -> d with a d x N router has 2 N d h + d N
# weights and costs 4 k d h + 2 d N FLOPs per token when only the chosen experts run, 4 k c d h + 2 d N on average when
# every padded slot is computed, against 2 d h weights and 4 d h FLOPs for a dense layer; numbers for d = 768, h = 3072.
import math

import numpy as np


class Counter:
    def __init__(self):
        self.macs = 0

    def mm(self, a, b):
        self.macs += a.shape[0] * a.shape[1] * b.shape[1]
        return a @ b


def layer(d, h, n, rng):
    return {
        "router": rng.normal(size=(d, n)),
        "w_in": rng.normal(size=(n, d, h)),
        "w_out": rng.normal(size=(n, h, d)),
    }


def weights(w):
    return sum(v.size for v in w.values())


def forward(w, x, k, c, padded, counter):
    t = x.shape[0]
    n = w["router"].shape[1]
    logits = counter.mm(x, w["router"])
    choice = np.argsort(-logits, axis=-1)[:, :k]
    cap = int(math.floor(k * t * c / n + 1e-9))
    y = np.zeros_like(x)
    for e in range(n):
        rows = np.nonzero((choice == e).any(axis=-1))[0][:cap]
        batch = np.zeros((cap, x.shape[1])) if padded else x[rows]
        if padded:
            batch[: len(rows)] = x[rows]
        if batch.shape[0] == 0:
            continue
        out = counter.mm(np.maximum(counter.mm(batch, w["w_in"][e]), 0.0), w["w_out"][e])
        y[rows] += out[: len(rows)]
    return y


def main():
    rng = np.random.default_rng(20261116)

    for _ in range(200):
        d = int(rng.integers(2, 12))
        h = int(rng.integers(2, 20))
        n = int(rng.integers(2, 10))
        k = int(rng.integers(1, min(3, n) + 1))
        c = float(rng.choice([0.5, 1.0, 1.25, 2.0]))
        t = n * int(rng.integers(1, 6))
        w = layer(d, h, n, rng)
        assert weights(w) == 2 * n * d * h + d * n
        x = rng.normal(size=(t, d))
        cap = int(math.floor(k * t * c / n + 1e-9))

        cnt = Counter()
        forward(w, x, k, c, padded=True, counter=cnt)
        assert 2 * cnt.macs == 4 * n * cap * d * h + 2 * d * n * t
        if abs(k * t * c / n - round(k * t * c / n)) < 1e-9:
            assert 2 * cnt.macs == round(t * (4 * k * c * d * h + 2 * d * n))

        big = Counter()
        forward(w, x, k, float(n), padded=False, counter=big)
        assert 2 * big.macs == t * (4 * k * d * h + 2 * d * n)

    # Item 3: the weight-to-FLOP ratio.
    for n, k in ((8, 1), (64, 1), (64, 2), (2048, 1)):
        d, h = 1024, 4096
        ratio = (2 * n * d * h) / (4 * k * d * h)
        assert ratio == n / (2 * k)

    # Item 4: d = 768, h = 3072, 128 experts.
    d, h, n = 768, 3072, 128
    assert 2 * d * h == 4_718_592
    assert 2 * n * d * h + d * n == 604_078_080
    assert 4 * d * h + 2 * d * n == 9_633_792
    assert int(1.25 * 4 * d * h) + 2 * d * n == 11_993_088
    assert 2 * 4 * d * h + 2 * d * n == 19_070_976
    assert 4 * d * h == 9_437_184
    assert abs((2 * d * n) / (4 * d * h) - 1 / 48) < 1e-15

    # Item 5: computation per number moved is h.
    for d, h in ((16, 64), (768, 3072), (1024, 8192)):
        assert (2 * d * h) / (2 * d) == h
    print("ok: params 2Ndh + dN; FLOPs 4kdh + 2dN, 4kcdh + 2dN padded; ratio N/2k; worked numbers; compute/comm = h")


if __name__ == "__main__":
    main()
