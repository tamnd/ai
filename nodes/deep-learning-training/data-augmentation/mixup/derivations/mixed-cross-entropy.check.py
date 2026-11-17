# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/mixup/ (ai://NEW-mixup).
# Check of derivations/mixed-cross-entropy.md: the cross-entropy against the mixed target lam e_i + (1 - lam) e_j is
# lam CE(y_i) + (1 - lam) CE(y_j); its gradient in the logits is p minus the mixed target; mixing a minibatch with a
# permutation of itself gives the same loss and weight gradient in the soft-target and two-label forms.
import numpy as np


def log_softmax(z):
    m = z.max(axis=-1, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(axis=-1, keepdims=True))


def soft_ce(z, q):
    return -(q * log_softmax(z)).sum(axis=-1)


def main():
    rng = np.random.default_rng(20261116)

    for _ in range(2000):
        k = int(rng.integers(2, 31))
        z = rng.normal(scale=rng.uniform(0.1, 6.0), size=k)
        yi, yj = (int(v) for v in rng.integers(k, size=2))
        lam = float(rng.uniform())
        e = np.eye(k)
        q = lam * e[yi] + (1 - lam) * e[yj]
        lp = log_softmax(z)

        # 1. Label mixing is loss mixing.
        assert abs(soft_ce(z, q) - (lam * -lp[yi] + (1 - lam) * -lp[yj])) < 1e-12

        # 2. The gradient is p - q, bounded and summing to zero.
        g = np.exp(lp) - q
        assert np.all(np.abs(g) <= 1.0) and abs(g.sum()) < 1e-12
        if k <= 8:
            h = 1e-6
            fd = np.array([(soft_ce(z + h * d, q) - soft_ce(z - h * d, q)) / (2 * h) for d in e])
            assert np.allclose(fd, g, atol=1e-7), (fd, g)

        # 4. Equal labels give the ordinary cross-entropy.
        assert abs(soft_ce(z, lam * e[yi] + (1 - lam) * e[yi]) + lp[yi]) < 1e-12

    # 3. A minibatch mixed through a permutation, for a linear model W x: same loss, same gradient in W.
    for _ in range(200):
        b, d, k = int(rng.integers(2, 17)), int(rng.integers(1, 6)), int(rng.integers(2, 7))
        x = rng.normal(size=(b, d))
        y = rng.integers(k, size=b)
        w = rng.normal(size=(d, k))
        lam = float(rng.beta(0.4, 0.4))
        perm = rng.permutation(b)
        xm = lam * x + (1 - lam) * x[perm]
        e = np.eye(k)
        q = lam * e[y] + (1 - lam) * e[y[perm]]
        z = xm @ w
        lp = log_softmax(z)
        soft = soft_ce(z, q).mean()
        two = lam * -lp[np.arange(b), y].mean() + (1 - lam) * -lp[np.arange(b), y[perm]].mean()
        assert abs(soft - two) < 1e-12
        grad_soft = xm.T @ (np.exp(lp) - q) / b
        p = np.exp(lp)
        grad_two = lam * xm.T @ (p - e[y]) / b + (1 - lam) * xm.T @ (p - e[y[perm]]) / b
        assert np.allclose(grad_soft, grad_two, atol=1e-12)

    # The worked example of the page.
    z = np.array([1.0, 0.5, -0.5])
    q = np.array([0.7, 0.0, 0.3])
    assert np.allclose(0.7 * np.array([2.0, 0.0]) + 0.3 * np.array([0.0, 4.0]), [1.4, 1.2])
    assert np.allclose(np.exp(log_softmax(z)), [0.546549, 0.331499, 0.121952], atol=1e-6)
    assert np.allclose(-log_softmax(z), [0.604131, 1.104131, 2.104131], atol=1e-6)
    assert abs(soft_ce(z, q) - 1.054131) < 1e-6
    assert np.allclose(np.exp(log_softmax(z)) - q, [-0.153451, 0.331499, -0.178048], atol=1e-6)
    print("ok: CE(lam e_i + (1 - lam) e_j) = lam CE(y_i) + (1 - lam) CE(y_j); gradient p - y~; batch forms agree")


if __name__ == "__main__":
    main()
