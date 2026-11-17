# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/direct-preference-optimization/ (ai://NEW-direct-preference-optimization).
# Check of derivations/kl-constrained-optimum.md: over a finite set of responses, the policy
# pi*(y) = pi_ref(y) exp(r(y)/beta) / Z maximizes E_pi[r] - beta KL(pi || pi_ref); the maximum is beta log Z; any
# other policy falls short by exactly beta KL(pi || pi*); and r = beta log(pi*/pi_ref) + beta log Z.
import numpy as np


def logsumexp(a):
    m = a.max()
    return m + np.log(np.exp(a - m).sum())


def objective(p, ref, r, beta):
    nz = p > 0
    return (p * r).sum() - beta * (p[nz] * np.log(p[nz] / ref[nz])).sum()


rng = np.random.default_rng(0)
for _ in range(2000):
    k = int(rng.integers(2, 12))
    ref = rng.dirichlet(np.ones(k) * rng.uniform(0.2, 3.0))
    ref = np.maximum(ref, 1e-12)
    ref /= ref.sum()
    r = rng.normal(0, rng.uniform(0.1, 5.0), size=k)
    beta = float(np.exp(rng.uniform(np.log(0.05), np.log(20.0))))
    log_z = logsumexp(np.log(ref) + r / beta)
    star = np.exp(np.log(ref) + r / beta - log_z)
    assert abs(star.sum() - 1) < 1e-12 and (star >= 0).all()
    best = objective(star, ref, r, beta)
    assert abs(best - beta * log_z) < 1e-9 * max(1.0, abs(best))
    # Reward recovered from the optimal policy, up to the prompt-only term beta log Z.
    assert np.allclose(beta * np.log(star / ref) + beta * log_z, r, atol=1e-8)
    for _ in range(5):
        q = rng.dirichlet(np.ones(k))
        gap = best - objective(q, ref, r, beta)
        kl = (q * np.log(q / star)).sum()
        assert gap >= -1e-10
        assert abs(gap - beta * kl) < 1e-8 * max(1.0, gap)
    # Small perturbations of pi* on the simplex never improve the objective.
    d = rng.normal(size=k)
    d -= d.mean()
    for eps in (1e-3, 1e-5):
        q = star + eps * d * star
        q /= q.sum()
        if (q > 0).all():
            assert objective(q, ref, r, beta) <= best + 1e-12

# The worked example of the page: pi_ref = (0.5, 0.3, 0.2), r = (1, 0, -1), beta = 0.5.
ref = np.array([0.5, 0.3, 0.2])
r = np.array([1.0, 0.0, -1.0])
u = ref * np.exp(r / 0.5)
assert abs(u.sum() - 4.021595106112647) < 1e-12
assert np.allclose(u / u.sum(), [0.918672, 0.074597, 0.006730], atol=1e-6)
assert abs(objective(u / u.sum(), ref, r, 0.5) - 0.5 * np.log(u.sum())) < 1e-12
# beta controls the distance from pi_ref.
for beta, want in (
    (10.0, [0.534647, 0.290261, 0.175093]),
    (1.0, [0.784399, 0.173139, 0.042463]),
):
    u = ref * np.exp(r / beta)
    assert np.allclose(u / u.sum(), want, atol=1e-6)
print("ok")
