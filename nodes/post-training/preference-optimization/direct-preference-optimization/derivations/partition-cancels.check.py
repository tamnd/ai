# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/direct-preference-optimization/ (ai://CZW0BQ).
# Check of derivations/partition-cancels.md: substituting r = beta log(pi*/pi_ref) + beta log Z(x) into the
# Bradley-Terry model gives sigmoid(beta log(pi*(y1)/pi_ref(y1)) - beta log(pi*(y2)/pi_ref(y2))), free of Z; rewards
# that differ by a function of x give the same preferences and the same optimal policy; and the DPO loss is the
# negative log-likelihood of the preference data under this model.
import numpy as np


def logsumexp(a):
    m = a.max()
    return m + np.log(np.exp(a - m).sum())


def sigmoid(x):
    return np.exp(-np.logaddexp(0.0, -x))


rng = np.random.default_rng(1)
for _ in range(2000):
    k = int(rng.integers(2, 10))
    ref = rng.dirichlet(np.ones(k))
    r = rng.normal(0, 3, size=k)
    beta = float(rng.uniform(0.05, 5.0))
    star = np.exp(np.log(ref) + r / beta - logsumexp(np.log(ref) + r / beta))
    i, j = rng.choice(k, size=2, replace=False)
    bt = np.exp(r[i]) / (np.exp(r[i]) + np.exp(r[j]))
    implied = beta * np.log(star / ref)
    assert abs(sigmoid(implied[i] - implied[j]) - bt) < 1e-10
    # Adding any f(x) to the reward leaves both the preferences and the optimal policy unchanged.
    f = rng.normal(0, 50)
    r2 = r + f
    assert abs(np.exp(r2[i] - np.logaddexp(r2[i], r2[j])) - bt) < 1e-10
    star2 = np.exp(np.log(ref) + r2 / beta - logsumexp(np.log(ref) + r2 / beta))
    assert np.allclose(star2, star, atol=1e-12)
    # The DPO per-pair loss equals -log p(y_i > y_j) under the policy-defined Bradley-Terry model.
    loss = np.logaddexp(
        0.0,
        -(beta * (np.log(star[i]) - np.log(ref[i])) - beta * (np.log(star[j]) - np.log(ref[j]))),
    )
    assert abs(loss + np.log(bt)) < 1e-9

# Worked example of the page: the BT probability of y1 over y2 for r = (1, 0, -1), from the rewards and from pi*.
ref = np.array([0.5, 0.3, 0.2])
r = np.array([1.0, 0.0, -1.0])
star = ref * np.exp(r / 0.5)
star /= star.sum()
p_from_policy = sigmoid(0.5 * np.log(star[0] / ref[0]) - 0.5 * np.log(star[1] / ref[1]))
assert abs(p_from_policy - 0.7310585786300049) < 1e-12
print("ok")
