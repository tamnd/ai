# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/distillation/ (ai://JGNGHC).
# Check of derivations/soft-target-gradient.md: the soft cross-entropy C = -sum p log q with p = softmax(v/T) and
# q = softmax(z/T) has gradient (q - p)/T in the student logits z; the KL divergence differs from C by the teacher
# entropy and has the same gradient; the gradient sums to zero and the loss is shift invariant; and the combined loss
# alpha T^2 KL + (1 - alpha) CE has gradient alpha T (q - p) + (1 - alpha)(softmax(z) - onehot).
import numpy as np


def log_softmax(x):
    m = x.max()
    return x - m - np.log(np.exp(x - m).sum())


def soft_ce(z, v, t):
    return -(np.exp(log_softmax(v / t)) * log_softmax(z / t)).sum()


def soft_kl(z, v, t):
    lp = log_softmax(v / t)
    return (np.exp(lp) * (lp - log_softmax(z / t))).sum()


def combined(z, v, y, t, alpha):
    return alpha * t * t * soft_kl(z, v, t) - (1 - alpha) * log_softmax(z)[y]


def central(f, z, h=1e-5):
    return np.array([(f(z + h * e) - f(z - h * e)) / (2 * h) for e in np.eye(len(z))])


def main():
    rng = np.random.default_rng(20261116)

    for _ in range(2000):
        n = int(rng.integers(2, 41))
        scale = rng.uniform(0.1, 10.0)
        v = rng.normal(scale=scale, size=n)
        z = rng.normal(scale=scale, size=n)
        t = float(np.exp(rng.uniform(np.log(0.2), np.log(50.0))))
        alpha = float(rng.uniform())
        y = int(rng.integers(n))
        p = np.exp(log_softmax(v / t))
        q = np.exp(log_softmax(z / t))
        g = (q - p) / t

        # 2. C = KL + H(p).
        h = -(p * log_softmax(v / t)).sum()
        assert abs(soft_ce(z, v, t) - (soft_kl(z, v, t) + h)) < 1e-10
        assert soft_kl(z, v, t) >= -1e-12

        # 3. The gradient sums to zero; shifting z changes nothing.
        assert abs(g.sum()) < 1e-12
        c = rng.normal(scale=5.0)
        assert abs(soft_kl(z + c, v, t) - soft_kl(z, v, t)) < 1e-10

        if n <= 8:
            # 1. The gradient of C and of KL is (q - p)/T.
            for f in (lambda x: soft_ce(x, v, t), lambda x: soft_kl(x, v, t)):
                fd = central(f, z)
                assert np.allclose(fd, g, rtol=1e-6, atol=1e-9), (fd, g)
            # 5. The combined loss.
            full = alpha * t * (q - p) + (1 - alpha) * (np.exp(log_softmax(z)) - np.eye(n)[y])
            fd = central(lambda x: combined(x, v, y, t, alpha), z)
            assert np.allclose(fd, full, rtol=1e-6, atol=1e-8), (fd, full)

    # 4. T = 1 with a one-hot target is the ordinary cross-entropy gradient.
    z = np.array([0.2, -1.0, 2.5, 0.0])
    y = 2
    onehot = np.eye(4)[y]
    fd = central(lambda x: -log_softmax(x)[y], z)
    assert np.allclose(fd, np.exp(log_softmax(z)) - onehot, atol=1e-9)

    # The worked example of the page.
    v = np.array([3.0, 1.0, -4.0])
    z = np.array([1.0, 0.5, -1.5])
    for t, want in ((1.0, [-0.287889, 0.240081, 0.047808]), (4.0, [-0.037024, 0.006079, 0.030945])):
        g = (np.exp(log_softmax(z / t)) - np.exp(log_softmax(v / t))) / t
        assert np.allclose(g, want, atol=1e-6), (t, g)
        assert np.allclose(central(lambda x: soft_ce(x, v, t), z), want, atol=1e-6)
    assert abs(soft_kl(z, v, 1.0) - 0.213906) < 1e-6
    assert abs(soft_ce(z, v, 1.0) - 0.585469) < 1e-6
    assert abs(combined(z, v, 0, 4.0, 0.9) - 1.038612) < 1e-6
    assert abs(combined(z, v, 0, 4.0, 0.5) - 0.809855) < 1e-6
    print("ok: dC/dz = dKL/dz = (q - p)/T; C = KL + H(p); combined gradient alpha T (q - p) + (1 - alpha)(softmax(z) - y)")


if __name__ == "__main__":
    main()
