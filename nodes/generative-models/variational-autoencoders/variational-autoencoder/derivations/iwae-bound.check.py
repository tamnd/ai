# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/variational-autoencoder/ (ai://NEW-variational-autoencoder).
"""Numeric check of the derivation iwae-bound."""

import itertools
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import vae  # noqa: E402


def exact_bound(joint, q, k):
    """L_k by enumerating every k-tuple of a latent variable with len(q) values."""
    total = 0.0
    for t in itertools.product(range(len(q)), repeat=k):
        idx = np.array(t)
        total += np.prod(q[idx]) * np.log(np.mean(joint[idx] / q[idx]))
    return total


rng = np.random.default_rng(0)
m = 5
for _ in range(10):
    prior = rng.dirichlet(np.ones(m))
    lik = rng.uniform(0.01, 1.0, m)  # p(x | z) at the observed x
    joint = prior * lik
    log_px = np.log(joint.sum())
    q = rng.dirichlet(np.ones(m))
    bounds = [exact_bound(joint, q, k) for k in range(1, 6)]
    assert abs(bounds[0] - np.sum(q * np.log(joint / q))) < 1e-12  # L_1 is the ELBO
    assert all(a <= b + 1e-12 for a, b in zip(bounds, bounds[1:])) and bounds[-1] <= log_px  # statement 2
    assert bounds[-1] - bounds[0] > 0  # strictly tighter when q is not the posterior
    post = joint / joint.sum()
    assert all(abs(exact_bound(joint, post, k) - log_px) < 1e-12 for k in (1, 2, 3))

# L_k -> log p(x) with bounded weights
prior, lik, q = rng.dirichlet(np.ones(m)), rng.uniform(0.01, 1.0, m), rng.dirichlet(np.ones(m))
joint = prior * lik
draws = rng.choice(m, size=(2000, 2000), p=q)
lk = np.mean(np.log(np.mean(joint[draws] / q[draws], 1)))
assert 0 < np.log(joint.sum()) - lk < 0.01

# statement 3 on the reference: the iwae gradient is the normalized-weight average of the per-draw gradients
cfg = vae.config(5, 4, 3, "bernoulli")
p = vae.init(rng, cfg)
x = (rng.uniform(size=(1, 5)) > 0.5).astype(float)
eps = rng.standard_normal((6, 1, 3))
_, g = vae.loss(p, cfg, x, eps, "iwae")
out, _ = vae.forward(p, cfg, x, eps)
lw = vae.log_weights(cfg, x, out, eps)[:, 0]
wn = np.exp(lw - lw.max()) / np.exp(lw - lw.max()).sum()
per_draw = [vae.loss(p, cfg, x, eps[i : i + 1], "elbo_mc")[1] for i in range(6)]  # the gradients of -log w_i
for k in g:
    np.testing.assert_allclose(g[k], sum(w * d[k] for w, d in zip(wn, per_draw)), rtol=1e-10, atol=1e-13)
print("iwae-bound: ok")
