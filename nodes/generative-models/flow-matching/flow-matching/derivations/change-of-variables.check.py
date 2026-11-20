# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/flow-matching/ (ai://NEW-flow-matching).
"""Numeric check of the derivation change-of-variables."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import flow_matching as fm  # noqa: E402

rng = np.random.default_rng(6)
means, s, w = np.array([[-2.0, 0.0], [2.0, 1.0]]), 0.5, np.array([0.4, 0.6])


def v(z, t):
    return fm.mixture_field(z, t, means, s, w)


def div(z, t, h=1e-5):
    return sum((v(z + h * e, t)[:, i] - v(z - h * e, t)[:, i]) / (2 * h) for i, e in enumerate(np.eye(2)))


x1 = means[rng.choice(2, 6, p=w)] + s * rng.standard_normal((6, 2))
ll = fm.log_likelihood(v, div, x1, 400, lambda z: fm.log_normal(z, 0.0, 1.0))
exact = fm.mixture_log_density(x1, np.ones(len(x1)), means, s, w)
assert np.allclose(ll, exact, atol=1e-4), (ll, exact)
# the network's divergence is the exact trace of its Jacobian
cfg = fm.config(d_data=3, d_hidden=7, d_time=4)
p = fm.init(rng, cfg)
p["W2"] = rng.standard_normal(p["W2"].shape)
x, t = rng.standard_normal((5, 3)), rng.uniform(0, 1, 5)
num = sum((fm.velocity(p, cfg, x + 1e-6 * e, t)[0][:, i] - fm.velocity(p, cfg, x - 1e-6 * e, t)[0][:, i]) / 2e-6
          for i, e in enumerate(np.eye(3)))  # fmt: skip
assert np.allclose(fm.divergence(p, cfg, x, t), num, atol=1e-7)
print("change-of-variables: ok")
