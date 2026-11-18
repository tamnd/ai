# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/reinforcement-learning-from-human-feedback/ (ai://6XVY3X).
# Check of derivations/bradley-terry-gradient.md: the Bradley-Terry probability as a logistic function of the reward
# difference, the loss as the logistic loss, its gradient and curvature, the invariance to a shift of the reward, the
# softmax over two responses, and the gradient through a parametrized reward model.
import numpy as np


def log_sigmoid(x):
    return -np.logaddexp(0.0, -x)


def loss(rw, rl):
    return -log_sigmoid(rw - rl)


def main():
    rng = np.random.default_rng(20261115)
    # 1. exp(r_w) / (exp(r_w) + exp(r_l)) = sigma(r_w - r_l), and the loss is log(1 + exp(-Delta)).
    rw, rl = rng.normal(0, 3, 1000), rng.normal(0, 3, 1000)
    p = np.exp(rw) / (np.exp(rw) + np.exp(rl))
    assert np.allclose(p, 1 / (1 + np.exp(-(rw - rl))), rtol=1e-12)
    assert np.allclose(loss(rw, rl), np.log1p(np.exp(-(rw - rl))), rtol=1e-12)
    assert np.isclose(loss(0.4, 0.4), np.log(2), rtol=1e-15)
    # 2. The gradient: dL/dr_w = -sigma(-Delta) = -(1 - sigma(Delta)), dL/dr_l = +sigma(-Delta), by central differences.
    h = 1e-6
    d = rw - rl
    gw = (loss(rw + h, rl) - loss(rw - h, rl)) / (2 * h)
    gl = (loss(rw, rl + h) - loss(rw, rl - h)) / (2 * h)
    s = 1 / (1 + np.exp(d))
    assert np.allclose(gw, -s, atol=1e-8) and np.allclose(gl, s, atol=1e-8)
    assert np.all(np.abs(s) < 1)
    # The pair that the model orders wrong (Delta < 0) gets a weight above 1/2, the one it orders right below 1/2.
    assert np.all(s[d < 0] > 0.5) and np.all(s[d > 0] < 0.5)
    # 3. The second derivative in Delta is sigma(Delta) sigma(-Delta), positive and at most 1/4 (at Delta = 0).
    h2 = 1e-4
    curv = (loss(rw + h2, rl) - 2 * loss(rw, rl) + loss(rw - h2, rl)) / h2**2
    sd = 1 / (1 + np.exp(-d))
    assert np.allclose(curv, sd * (1 - sd), atol=1e-6)
    assert np.all(sd * (1 - sd) <= 0.25 + 1e-15)
    # 4. A shift c(x), the same for both responses of a prompt, changes nothing.
    c = rng.normal(0, 10, 1000)
    assert np.allclose(loss(rw + c, rl + c), loss(rw, rl), rtol=1e-9, atol=1e-12)
    # 5. The cross-entropy over two responses with the preferred one as the label is the same loss.
    two = np.stack([rw, rl], axis=1)
    ce = -(rw - np.logaddexp.reduce(two, axis=1))
    assert np.allclose(ce, loss(rw, rl), rtol=1e-12)
    # 6. Through a linear reward model r = theta . phi(x, y), the gradient is -sigma(-Delta) (phi_w - phi_l).
    theta = rng.normal(size=4)
    pw, pl = rng.normal(size=4), rng.normal(size=4)
    f = lambda th: loss(th @ pw, th @ pl)  # noqa: E731
    num = np.array([(f(theta + h * e) - f(theta - h * e)) / (2 * h) for e in np.eye(4)])
    dd = theta @ pw - theta @ pl
    assert np.allclose(num, -(1 / (1 + np.exp(dd))) * (pw - pl), atol=1e-8)
    print("ok: the Bradley-Terry probability, the logistic loss, its gradient and curvature, shift invariance, the "
          "two-way softmax and the gradient in the parameters")


if __name__ == "__main__":
    main()
