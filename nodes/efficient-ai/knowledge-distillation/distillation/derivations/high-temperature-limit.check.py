# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/distillation/ (ai://NEW-distillation).
# Check of derivations/high-temperature-limit.md: as T grows, softmax(x/T) = 1/N + (x - mean x)/(N T) + O(1/T^2); the
# soft gradient (q - p)/T is O(1/T^2); T^2 times it tends to (centred z - centred v)/N with error O(1/T), which is the
# gradient of logit matching (1/2N) sum (z - v)^2 for zero-mean logits; and T^2 KL(p || q) tends to (1/2N) sum delta^2.
import numpy as np


def log_softmax(x):
    m = x.max()
    return x - m - np.log(np.exp(x - m).sum())


def scaled_grad(z, v, t):
    return t * (np.exp(log_softmax(z / t)) - np.exp(log_softmax(v / t)))


def scaled_kl(z, v, t):
    lp = log_softmax(v / t)
    return t * t * (np.exp(lp) * (lp - log_softmax(z / t))).sum()


def main():
    rng = np.random.default_rng(20261116)

    for _ in range(500):
        n = int(rng.integers(2, 31))
        v = rng.normal(scale=rng.uniform(0.5, 4.0), size=n)
        z = rng.normal(scale=rng.uniform(0.5, 4.0), size=n)
        delta = (z - z.mean()) - (v - v.mean())
        limit = delta / n
        kl_limit = (delta**2).sum() / (2 * n)

        # 1. The first-order expansion of the softmax, with a second-order remainder.
        for t in (1e2, 1e3):
            approx = 1.0 / n + (z - z.mean()) / (n * t)
            err = np.abs(np.exp(log_softmax(z / t)) - approx).max()
            assert err < 2.0 * (np.abs(z).max() / t) ** 2, (err, t)

        # 2 and 3. The unscaled gradient is O(1/T^2); the scaled one converges at rate 1/T.
        errs, kl_errs = [], []
        for t in (1e3, 1e4, 1e5):
            g = scaled_grad(z, v, t)
            assert np.abs(g / t**2).max() < 2 * np.abs(limit).max() / t**2 + 1e-15
            errs.append(np.abs(g - limit).max())
            # T^2 KL loses about T^2 * 1e-15 to cancellation, so it is checked at the two lower temperatures only.
            if t < 1e5:
                kl_errs.append(abs(scaled_kl(z, v, t) - kl_limit))
        # At least the rate 1/T: each tenfold T cuts the error by 5 or more (for N = 2 the softmax is odd about
        # the uniform point, so the second-order term vanishes and the rate is 1/T^2).
        for e in (errs, kl_errs):
            for a, b in zip(e, e[1:]):
                if a > 1e-6:
                    assert a / b > 5, e
        assert errs[0] < 2 * (np.abs(z).max() + np.abs(v).max()) ** 2 / 1e3, errs

        # 4. The limit is the gradient of (1/2N) sum delta^2, and of (1/2N) sum (z - v)^2 for zero-mean logits.
        def matching(x, v=v):
            d = (x - x.mean()) - (v - v.mean())
            return (d**2).sum() / (2 * n)

        h = 1e-6
        fd = np.array([(matching(z + h * e) - matching(z - h * e)) / (2 * h) for e in np.eye(n)])
        assert np.allclose(fd, limit, atol=1e-8)
        zc, vc = z - z.mean(), v - v.mean()
        assert np.allclose((zc - vc) / n, ((zc - zc.mean()) - (vc - vc.mean())) / n)
        c = rng.normal(scale=3.0)
        assert np.allclose(((v + c) - (v + c).mean()) - (v - v.mean()), 0.0, atol=1e-12)
        assert np.abs(scaled_grad(v + c, v, 1e3)).max() < 1e-12

    # The worked example of the page: both logit vectors have mean 0.
    v = np.array([3.0, 1.0, -4.0])
    z = np.array([1.0, 0.5, -1.5])
    assert abs(v.mean()) < 1e-15 and abs(z.mean()) < 1e-15
    assert np.allclose(scaled_grad(z, v, 4.0), [-0.592381, 0.097268, 0.495113], atol=1e-6)
    assert np.allclose(scaled_grad(z, v, 100.0), [-0.667294, -0.155354, 0.822648], atol=1e-6)
    assert np.allclose((z - v) / 3, [-0.666667, -0.166667, 0.833333], atol=1e-6)
    assert abs(scaled_kl(z, v, 4.0) - 1.095802) < 1e-6
    assert abs(scaled_kl(z, v, 100.0) - 1.732200) < 1e-6
    assert abs(((z - v) ** 2).sum() / 6 - 1.75) < 1e-15
    err = [np.abs(scaled_grad(z, v, t) - (z - v) / 3).max() * t for t in (1e2, 1e3, 1e4)]
    assert np.allclose(err, 1.13, rtol=0.05), err
    print("ok: T^2 dC/dz -> (centred z - centred v)/N at rate 1/T; T^2 KL -> (1/2N) sum delta^2; zero-mean gives logit matching")


if __name__ == "__main__":
    main()
