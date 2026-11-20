# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/flow-matching/ (ai://NEW-flow-matching).
"""Numeric check of the derivation cfm-gradient."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import flow_matching as fm  # noqa: E402

rng = np.random.default_rng(2)
means, s, w = np.array([[-2.0, 0.0], [2.0, 1.0]]), 0.4, np.array([0.3, 0.7])
for sm in (0.0, 0.1):
    cfg = fm.config(d_data=2, d_hidden=6, d_time=4, sigma_min=sm)
    n = 400_000
    x1 = means[rng.choice(2, n, p=w)] + s * rng.standard_normal((n, 2))
    x0 = rng.standard_normal((n, 2))
    t = rng.uniform(0, 1, n)
    xt = fm.path(cfg, x0, x1, t)
    u = fm.mixture_field(xt, t, means, s, w, sm)
    gaps = []
    for seed in (3, 4):
        p = fm.init(np.random.default_rng(seed), cfg)
        p["W2"] = np.random.default_rng(seed + 10).standard_normal(p["W2"].shape)
        lc, gc = fm.loss_and_grads(p, cfg, x0, x1, t)
        lf, gf = fm.regress(p, cfg, xt, t, u)
        for k in gc:
            assert np.allclose(gc[k], gf[k], atol=0.03 * (1 + np.abs(gf[k]).max())), (k, gc[k], gf[k])
        gaps.append(lc - lf)
    # the gap is a constant, E||u_t(x | x_1)||^2 - E||u_t(x)||^2, the same for both networks
    assert abs(gaps[0] - gaps[1]) < 0.02, gaps
    tgt = fm.target(cfg, x0, x1)
    assert abs(gaps[0] - (np.mean(np.sum(tgt**2, 1)) - np.mean(np.sum(u**2, 1)))) < 0.02
    assert gaps[0] > 0.1
print("cfm-gradient: ok")
