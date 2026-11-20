# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/discrete-diffusion-model/ (ai://NEW-discrete-diffusion-model).
"""Numeric check of the derivation x0-parameterization-optimum."""

import itertools

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import discrete_diffusion as dd  # noqa: E402


def expected_bound(p0, Qs, denoise):
    prior = p0 @ dd.cumulative(Qs)[-1]
    total = 0.0
    for x0 in range(len(p0)):
        if p0[x0] > 0:
            LT, mids, L0 = dd.d3pm_bound(x0, Qs, denoise, prior)
            total += p0[x0] * (LT + sum(mids) + L0)
    return total


def joint_kl(p0, Qs, denoise):
    """KL(q(x_{0:T}) || p(x_{0:T})) by enumerating every trajectory."""
    n, T = len(p0), len(Qs)
    Qbar = dd.cumulative(Qs)
    prior = p0 @ Qbar[-1]
    total = 0.0
    for xs in itertools.product(range(n), repeat=T + 1):
        q = p0[xs[0]] * np.prod([Qs[t][xs[t], xs[t + 1]] for t in range(T)])
        if q == 0:
            continue
        lp = np.log(prior[xs[T]])
        for t in range(T, 0, -1):
            step = dd.posterior(dd.one_hot(xs[t], n), denoise(t, xs[t]), Qs[t - 1], Qbar[t - 1])
            lp += np.log(step[xs[t - 1]])
        total += q * (np.log(q) - lp)
    return total


rng = np.random.default_rng(1)
K, T = 3, 4
betas = [0.2, 0.3, 0.4, 0.5]
for kind in ("uniform", "absorbing"):
    if kind == "uniform":
        Qs = [dd.uniform_q(b, K) for b in betas]
        p0 = rng.dirichlet(np.ones(K))
    else:
        Qs = [dd.absorbing_q(b, K) for b in betas]
        p0 = np.append(rng.dirichlet(np.ones(K)), 0.0)
    H = dd.entropy(p0)
    best = expected_bound(p0, Qs, lambda t, xt: p0)
    assert abs(best - H) < 1e-12, (kind, best, H)
    assert abs(joint_kl(p0, Qs, lambda t, xt: p0)) < 1e-12
    post = expected_bound(p0, Qs, dd.posterior_mean(p0, Qs))
    if kind == "absorbing":
        assert abs(post - H) < 1e-12
    else:
        assert post > H + 1e-3, (post, H)
        assert abs(post - H - joint_kl(p0, Qs, dd.posterior_mean(p0, Qs))) < 1e-10
    for _ in range(5):
        table = {(t, xt): np.append(rng.dirichlet(np.ones(K)), [0.0] * (len(p0) - K)) for t in range(1, T + 1)
                 for xt in range(len(p0))}  # fmt: skip

        def rand(t, xt, table=table):
            return table[(t, xt)]

        b = expected_bound(p0, Qs, rand)
        assert b > H
        assert abs(b - H - joint_kl(p0, Qs, rand)) < 1e-10
