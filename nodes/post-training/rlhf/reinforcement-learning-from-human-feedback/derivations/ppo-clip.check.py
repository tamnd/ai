# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/reinforcement-learning-from-human-feedback/ (ai://NEW-reinforcement-learning-from-human-feedback).
# Check of derivations/ppo-clip.md: the clipped surrogate is a lower bound on the unclipped one and equal to it inside
# [1 - eps, 1 + eps], it is flat beyond the interval on the side that would raise it, and at rho = 1 its gradient is
# the policy gradient A grad log pi.
import numpy as np


def clip_obj(rho, a, eps):
    return np.minimum(rho * a, np.clip(rho, 1 - eps, 1 + eps) * a)


def main():
    rng = np.random.default_rng(20261115)
    rho = np.exp(rng.normal(0, 0.6, 20000))
    a = rng.normal(0, 2, 20000)
    for eps in [0.0, 0.1, 0.2, 0.3]:
        c = clip_obj(rho, a, eps)
        # 1. A lower bound, with equality inside the interval.
        assert np.all(c <= rho * a + 1e-15)
        inside = (rho >= 1 - eps) & (rho <= 1 + eps)
        assert np.allclose(c[inside], (rho * a)[inside], rtol=1e-15)
        # 2. Flat beyond the interval on the side that raises the objective: (1 + eps) A for A > 0 and rho > 1 + eps,
        #    (1 - eps) A for A < 0 and rho < 1 - eps.
        up = (a > 0) & (rho > 1 + eps)
        down = (a < 0) & (rho < 1 - eps)
        assert np.allclose(c[up], (1 + eps) * a[up]) and np.allclose(c[down], (1 - eps) * a[down])
        # 3. On the other side the unclipped term is the minimum, so the gradient in rho is A.
        other = ((a > 0) & (rho < 1 - eps)) | ((a < 0) & (rho > 1 + eps))
        assert np.allclose(c[other], (rho * a)[other])
    # 4. The derivative in log pi: rho A inside, 0 where flat, by central differences.
    h, eps = 1e-6, 0.2
    for r0, a0, want in [(1.1, 1.0, 1.1), (1.5, 1.0, 0.0), (0.5, -1.0, 0.0), (0.5, 1.0, 0.5), (1.5, -1.0, -1.5)]:
        lp = np.log(r0)
        d = (clip_obj(np.exp(lp + h), a0, eps) - clip_obj(np.exp(lp - h), a0, eps)) / (2 * h)
        assert np.isclose(d, want, atol=1e-8), (r0, a0, d)
    # 5. At rho = 1 (the first minibatch after sampling), the gradient in log pi is A: that of A log pi.
    d = (clip_obj(np.exp(h), 0.7, eps) - clip_obj(np.exp(-h), 0.7, eps)) / (2 * h)
    assert np.isclose(d, 0.7, rtol=1e-8)
    # 6. The worked example of the page.
    assert np.isclose(clip_obj(1.5, 1.0, 0.2), 1.2) and np.isclose(clip_obj(0.5, 1.0, 0.2), 0.5)
    assert np.isclose(clip_obj(0.5, -1.0, 0.2), -0.8)
    print("ok: the lower bound, equality inside the interval, the flat sides, the gradient at rho = 1 and the worked "
          "example")


if __name__ == "__main__":
    main()
