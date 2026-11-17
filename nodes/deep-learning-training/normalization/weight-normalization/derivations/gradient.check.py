# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/weight-normalization/ (ai://NEW-weight-normalization).
# Check of derivations/gradient.md: the gradients of a loss in g and v under w = g v / ||v||, against central finite
# differences; the projection form (g / ||v||) M_w grad_w; and grad_v . v = 0.
import numpy as np


def grads(v, g, grad_w):
    n = np.linalg.norm(v)
    grad_g = grad_w @ v / n
    return g / n * grad_w - g * grad_g / n**2 * v, grad_g


def fd(f, a, h=1e-6):
    out = np.zeros_like(a)
    for i in range(a.size):
        e = np.zeros_like(a)
        e[i] = h
        out[i] = (f(a + e) - f(a - e)) / (2 * h)
    return out


def main():
    rng = np.random.default_rng(20261116)
    cases = 0
    for k in (1, 2, 3, 5, 8):
        for _ in range(20):
            a, c = rng.normal(size=k), rng.normal(size=k)
            v, g = rng.normal(size=k) * rng.uniform(0.1, 10), rng.normal()

            def loss(v_, g_):
                w = g_ * v_ / np.linalg.norm(v_)
                return float(np.sum(np.tanh(a * w)) + 0.5 * np.sum((w - c) ** 2))

            w = g * v / np.linalg.norm(v)
            grad_w = a * (1 - np.tanh(a * w) ** 2) + (w - c)
            grad_v, grad_g = grads(v, g, grad_w)
            assert np.allclose(grad_v, fd(lambda x: loss(x, g), v), rtol=1e-6, atol=1e-8)
            assert np.isclose(grad_g, (loss(v, g + 1e-6) - loss(v, g - 1e-6)) / 2e-6, rtol=1e-6, atol=1e-8)
            m = np.eye(k) - np.outer(w, w) / (w @ w)
            assert np.allclose(grad_v, g / np.linalg.norm(v) * m @ grad_w)
            assert abs(grad_v @ v) < 1e-10 * max(1.0, np.linalg.norm(grad_v) * np.linalg.norm(v))
            cases += 1
    # the worked example: v = (3, 4), g = 2, grad_w = (1, 0)
    grad_v, grad_g = grads(np.array([3.0, 4.0]), 2.0, np.array([1.0, 0.0]))
    assert np.isclose(grad_g, 0.6) and np.allclose(grad_v, [0.256, -0.192])
    print(f"ok: eq-grad matches finite differences in {cases} cases, equals the projection form, and is orthogonal to v")


if __name__ == "__main__":
    main()
