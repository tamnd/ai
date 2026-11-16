# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/direct-preference-optimization/ (ai://NEW-direct-preference-optimization).
# Check of derivations/dpo-gradient.md: for l = -log sigmoid(beta (a_w - a_l)) with a = log pi - log pi_ref, the
# derivatives in log pi(y_w) and log pi(y_l) are -beta sigmoid(r_l - r_w) and +beta sigmoid(r_l - r_w), with
# r = beta a; and through a softmax policy over a finite response set this is the chain rule of the log-probabilities.
import numpy as np


def sigmoid(x):
    return np.exp(-np.logaddexp(0.0, -x))


def loss(lw, ll, rw, rl, beta):
    return np.logaddexp(0.0, -beta * ((lw - rw) - (ll - rl)))


rng = np.random.default_rng(2)
for _ in range(2000):
    lw, ll, rw, rl = rng.normal(-20, 8, size=4)
    beta = float(rng.uniform(0.01, 2.0))
    hw, hl = beta * (lw - rw), beta * (ll - rl)
    gw = -beta * sigmoid(hl - hw)
    gl = beta * sigmoid(hl - hw)
    eps = 1e-6
    fw = (loss(lw + eps, ll, rw, rl, beta) - loss(lw - eps, ll, rw, rl, beta)) / (2 * eps)
    fl = (loss(lw, ll + eps, rw, rl, beta) - loss(lw, ll - eps, rw, rl, beta)) / (2 * eps)
    assert abs(gw - fw) < 1e-6 and abs(gl - fl) < 1e-6
    assert 0 < sigmoid(hl - hw) < 1 or abs(hl - hw) > 30

# Through a softmax policy pi = softmax(theta) over k responses: grad_theta l = -beta s (e_w - e_l) with the
# gradient of log pi(y) being e_y - pi, which equals -beta s (e_w - e_l) since the pi terms cancel.
for _ in range(500):
    k = int(rng.integers(2, 8))
    theta = rng.normal(0, 2, size=k)
    ref = rng.dirichlet(np.ones(k))
    w, v = rng.choice(k, size=2, replace=False)
    beta = float(rng.uniform(0.05, 2.0))

    def full(th, w=w, v=v, ref=ref, beta=beta):
        lp = th - th.max() - np.log(np.exp(th - th.max()).sum())
        return loss(lp[w], lp[v], np.log(ref[w]), np.log(ref[v]), beta)

    lp = theta - theta.max() - np.log(np.exp(theta - theta.max()).sum())
    s = sigmoid(beta * (lp[v] - np.log(ref[v])) - beta * (lp[w] - np.log(ref[w])))
    e = np.eye(k)
    analytic = -beta * s * (e[w] - e[v])
    fd = np.array([(full(theta + 1e-6 * e[m]) - full(theta - 1e-6 * e[m])) / 2e-6 for m in range(k)])
    assert np.allclose(analytic, fd, atol=1e-7)

# Worked example: log-probs (-12, -15) under the policy, (-13, -14) under the reference, beta = 0.1.
s = sigmoid(-0.1 - 0.1)
assert abs(s - 0.45016600268752216) < 1e-12
assert abs(-0.1 * s - (-0.04501660026875222)) < 1e-15
assert abs(loss(-12.0, -15.0, -13.0, -14.0, 0.1) - 0.5981388693815918) < 1e-12
print("ok")
