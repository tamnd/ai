# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/classifier-free-guidance/ (ai://NEW-classifier-free-guidance).
"""Numeric check of the derivation implicit-classifier."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import classifier_free_guidance as cg  # noqa: E402

rng = np.random.default_rng(2)
means, s2 = rng.standard_normal((3, 3)), 0.3
w = rng.uniform(0.2, 1.2, 3)
w /= w.sum()
z = rng.standard_normal((7, 3))
abar, c, h = cg.alpha_bars(1000), 1, 1e-6
for t in (10, 300, 800):
    a = np.full(len(z), abar[t])
    sig = np.sqrt(1 - a)[:, None]
    e_c, e_u = cg.exact_eps_cond(z, a, means, s2, c), cg.exact_eps(z, a, means, s2, w)
    g = cg.classifier_grad(z, a, means, s2, w, np.full(len(z), c))
    for s in (0.0, 2.5, 7.0):

        def logp(zz, a=a, s=s):
            return s * cg.log_marginal(zz, a, means[c : c + 1], s2, np.ones(1)) + (1 - s) * cg.log_marginal(
                zz, a, means, s2, w
            )

        fd = np.stack([(logp(z + h * np.eye(3)[i]) - logp(z - h * np.eye(3)[i])) / (2 * h) for i in range(3)], 1)
        guided = cg.guide(e_c, e_u, s)
        assert np.allclose(guided, -sig * fd, rtol=1e-6, atol=1e-7)
        assert np.allclose(guided, cg.classifier_guided_eps(e_u, g, a, s), atol=1e-12)
        assert np.allclose(guided, cg.classifier_guided_eps(e_c, g, a, s - 1.0), atol=1e-12)
    # Bayes: log p(c | z) = log p(z | c) + log w_c - log p(z)
    lc = cg.log_classifier(z, a, means, s2, w)[:, c]
    rhs = cg.log_marginal(z, a, means[c : c + 1], s2, np.ones(1)) + np.log(w[c]) - cg.log_marginal(z, a, means, s2, w)
    assert np.allclose(lc, rhs, atol=1e-12)
