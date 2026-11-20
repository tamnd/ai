# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/consistency-model/ (ai://NEW-consistency-model).
"""Numeric check of the derivation sampling-cost."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import consistency_model as cm  # noqa: E402

means, s2, w = np.array([[1.0, -0.5], [-1.0, 0.5]]), 0.1, np.array([0.3, 0.7])
calls = {"n": 0}


def score_fn(x, t):
    calls["n"] += 1
    return cm.score(x, t, means, s2, w)


def edm_heun(x, N):
    """Karras and colleagues' sampler: Heun steps down the N levels, then one Euler step to sigma = 0."""
    ts = cm.karras_steps(N)[::-1]
    for a, b in zip(ts[:-1], ts[1:], strict=True):
        x = cm.teacher_step(x, a, b, score_fn)
    return cm.teacher_step(x, ts[-1], 0.0, score_fn, solver="euler")


rng = np.random.default_rng(3)
for N in (2, 18, 40):
    calls["n"] = 0
    edm_heun(cm.T_MAX * rng.standard_normal((5, 2)), N)
    assert calls["n"] == 2 * N - 1
assert 2 * 18 - 1 == 35

cfg = cm.config(2, 8, 4)
p = cm.init(rng, cfg)
fcalls = {"n": 0}


def f(x, t):
    fcalls["n"] += 1
    return cm.forward(p, cfg, x, np.full(len(x), t))


for taus in ((), (0.8,), (2.0, 0.8, 0.3)):
    fcalls["n"] = 0
    noise = rng.standard_normal((len(taus), 5, 2)) if taus else None
    cm.sample(f, cm.T_MAX * rng.standard_normal((5, 2)), taus, noise)
    assert fcalls["n"] == 1 + len(taus)

ts = cm.karras_steps(18)
for solver, want in (("heun", 2), ("euler", 1)):
    calls["n"] = 0
    cm.cd_pair(rng.standard_normal((5, 2)), rng.standard_normal((5, 2)), ts[3], ts[4], score_fn, solver)
    assert calls["n"] == want

for (D, H, E, B), (P, C) in (((2, 64, 8, 256), (834, None)), ((3072, 1024, 64, 1), (None, 12_713_984))):
    c = cm.config(D, H, E)
    assert cm.param_count(c) == D * H + E * H + H + H * D + D
    assert cm.flops_forward(c, B) == 2 * B * (D * H + E * H + H * D)
    assert P is None or cm.param_count(c) == P
    assert C is None or cm.flops_forward(c, B) == C
    assert sum(v.size for v in cm.init(rng, c).values()) == cm.param_count(c)
print("sampling-cost: ok")
