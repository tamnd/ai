# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/autoregressive-model/ (ai://NEW-autoregressive-model).
"""Numeric check of the derivation made-masks."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import autoregressive as ar  # noqa: E402

rng = np.random.default_rng(1)
for trial in range(40):
    D = int(rng.integers(2, 8))
    hidden = tuple(int(h) for h in rng.integers(1, 9, int(rng.integers(0, 4))))
    direct = not hidden or bool(rng.integers(0, 2))
    cfg = ar.config(D, hidden, direct)
    m = ar.degrees(cfg, rng.permutation(D) + 1, rng)
    mk = ar.masks(cfg, m)
    allowed = m[0][:, None] < m[0][None, :]  # [i, d]: input i comes before output d
    # path counts input -> output through the hidden layers, in the (in, out) layout of the code
    paths = np.zeros((D, D))
    if hidden:
        prod = np.eye(D)
        for i in range(1, len(hidden) + 1):
            prod = prod @ mk[f"W{i}"]
        paths = paths + prod @ mk["V"]
    if direct:
        paths = paths + mk["A"]
        assert np.array_equal(mk["A"] > 0, allowed)  # the direct mask links every allowed pair
    assert np.all(paths[~allowed] == 0), trial
    # each hidden unit depends only on inputs of degree at most its own
    reach = np.eye(D)
    for i in range(1, len(hidden) + 1):
        reach = reach @ mk[f"W{i}"]
        assert np.all(reach[m[0][:, None] > m[i][None, :]] == 0)
    # the Jacobian of the logits by central differences
    p = {k: rng.normal(0.0, 1.0, s) for k, s in ar.param_shapes(cfg).items()}
    x = rng.standard_normal((1, D))
    for i in range(D):
        e = np.zeros((1, D))
        e[0, i] = 1e-6
        col = (ar.logits(p, cfg, mk, x + e)[0] - ar.logits(p, cfg, mk, x - e)[0])[0] / 2e-6
        assert np.all(np.abs(col[~allowed[i]]) < 1e-8), (trial, i)

# a hidden unit of degree D could feed no output: its column of the output mask is all zero
cfg = ar.config(4, (5,), False)
m = [np.arange(1, 5), np.array([1, 2, 3, 4, 4])]
assert np.all(ar.masks(cfg, m)["V"][3:] == 0) and np.all(ar.masks(cfg, m)["V"][:3].sum(1) > 0)
print("made-masks: ok")
