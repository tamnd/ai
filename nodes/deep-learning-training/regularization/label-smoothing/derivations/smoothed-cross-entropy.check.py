# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/label-smoothing/ (ai://NEW-label-smoothing).
# Check of derivations/smoothed-cross-entropy.md: the cross-entropy with the smoothed target (1 - eps) delta_y + eps u
# is (1 - eps) NLL + eps H(u, p); for uniform u it is (1 - eps) NLL + eps KL(u || p) + eps log K; -H(p) is
# KL(p || u) - log K; label dropout averages to the smoothed target; and the gradient in the logits is p - q'.
import numpy as np


def log_softmax(z):
    m = z.max()
    return z - m - np.log(np.exp(z - m).sum())


def smoothed_target(y, k, eps, u=None):
    u = np.full(k, 1.0 / k) if u is None else u
    q = eps * u
    q[y] += 1.0 - eps
    return q


def loss(z, q):
    return -(q * log_softmax(z)).sum()


def main():
    rng = np.random.default_rng(20261115)

    for _ in range(2000):
        k = int(rng.integers(2, 51))
        z = rng.normal(scale=rng.uniform(0.1, 8.0), size=k)
        y = int(rng.integers(k))
        eps = float(rng.uniform(0.0, 1.0))
        lp = log_softmax(z)
        p = np.exp(lp)
        nll = -lp[y]

        # 1. The pair of losses, for uniform and for a random u.
        u = rng.dirichlet(np.ones(k))
        assert abs(loss(z, smoothed_target(y, k, eps, u)) - ((1 - eps) * nll + eps * -(u * lp).sum())) < 1e-12
        q = smoothed_target(y, k, eps)
        h_up = -lp.mean()
        assert abs(loss(z, q) - ((1 - eps) * nll + eps * h_up)) < 1e-12

        # 2. The KL form with uniform u.
        kl_up = (np.log(1.0 / k) - lp).mean()
        assert abs(loss(z, q) - ((1 - eps) * nll + eps * kl_up + eps * np.log(k))) < 1e-12
        assert kl_up >= -1e-15

        # 3. The confidence penalty is the reverse KL up to log K, and is bounded by log K.
        ent = -(p * lp).sum()
        kl_pu = (p * (lp - np.log(1.0 / k))).sum()
        assert abs(-ent - (kl_pu - np.log(k))) < 1e-12
        assert -1e-12 <= kl_pu <= np.log(k) + 1e-12

        # 4. Label dropout: the expected NLL over the noisy label is the smoothed loss.
        prob = (1 - eps) * (np.arange(k) == y) + eps / k
        assert abs((prob * -lp).sum() - loss(z, q)) < 1e-12

        # 5. The gradient is p - q', in [-1, 1], summing to 0, against central differences.
        g = p - q
        assert np.all(np.abs(g) <= 1.0) and abs(g.sum()) < 1e-12
        if k <= 8:
            h = 1e-6
            fd = np.array([(loss(z + h * e, q) - loss(z - h * e, q)) / (2 * h) for e in np.eye(k)])
            assert np.allclose(fd, g, atol=1e-7), (fd, g)

    # 4 again, by sampling: keep the label with probability 1 - eps, else draw it uniformly.
    k, y, eps = 5, 2, 0.4
    z = np.array([0.3, -1.0, 2.0, 0.5, -0.2])
    lp = log_softmax(z)
    n = 200_000
    keep = rng.random(n) >= eps
    labels = np.where(keep, y, rng.integers(k, size=n))
    mc = -lp[labels].mean()
    assert abs(mc - loss(z, smoothed_target(y, k, eps))) < 1e-2, mc

    # The worked example of the page.
    z = np.array([2.0, 0.0, 0.0])
    q = smoothed_target(0, 3, 0.3)
    assert np.allclose(q, [0.8, 0.1, 0.1])
    assert abs(loss(z, q) - 0.639545) < 1e-6
    assert abs(-log_softmax(z)[0] - 0.239545) < 1e-6
    assert abs(-log_softmax(z).mean() - 1.572878) < 1e-6
    assert np.allclose(np.exp(log_softmax(z)) - q, [-0.013014, 0.006507, 0.006507], atol=1e-6)
    print("ok: smoothed CE = (1-eps) NLL + eps H(u,p) = (1-eps) NLL + eps KL(u||p) + eps log K; gradient p - q'")


if __name__ == "__main__":
    main()
