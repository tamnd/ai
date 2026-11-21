# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/autoregressive-model/ (ai://NEW-autoregressive-model).
"""Numeric check of the derivation autoregressive-normalization."""

import itertools
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import autoregressive as ar  # noqa: E402

rng = np.random.default_rng(0)
D, K = 4, 3
# a table q_d(. | prefix) for every prefix, each row on the simplex
tables = [rng.dirichlet(np.ones(K), size=(K,) * d) for d in range(D)]


def joint(x):
    return float(np.prod([tables[d][tuple(x[:d])][x[d]] for d in range(D)]))


outcomes = list(itertools.product(range(K), repeat=D))
assert abs(sum(joint(x) for x in outcomes) - 1.0) < 1e-12
# the marginal of the first two variables is the product of the first two conditionals
for x in itertools.product(range(K), repeat=2):
    m = sum(joint(x + y) for y in itertools.product(range(K), repeat=D - 2))
    assert abs(m - tables[0][()][x[0]] * tables[1][(x[0],)][x[1]]) < 1e-12

# a MADE with random weights and a random ordering is normalized for every setting of its weights
cfg = ar.config(6, (7, 5), True)
p = {k: rng.normal(0.0, 1.5, s) for k, s in ar.param_shapes(cfg).items()}
m0 = rng.permutation(6) + 1
mk = ar.masks(cfg, ar.degrees(cfg, m0, rng))
assert abs(np.exp(ar.log_prob(p, cfg, mk, ar.all_binary(6))).sum() - 1.0) < 1e-12

# a WaveNet stack over ternary sequences of length 4
w = {k: rng.normal(0.0, 0.8, s) for k, s in ar.wavenet_shapes(3, 2, 2, (1, 2)).items()}
X = np.array(outcomes)
assert abs(np.exp(ar.wavenet_log_prob(w, X, (1, 2))).sum() - 1.0) < 1e-12

# ancestral sampling of a smaller MADE draws from its probabilities
cfg = ar.config(3, (6,))
p = {k: rng.normal(0.0, 1.2, s) for k, s in ar.param_shapes(cfg).items()}
m0 = np.array([3, 1, 2])
mk = ar.masks(cfg, ar.degrees(cfg, m0))
Xb = ar.all_binary(3)
prob = np.exp(ar.log_prob(p, cfg, mk, Xb))
n = 60000
s = ar.sample(p, cfg, mk, m0, rng, n)
freq = np.array([np.mean(np.all(s == row, 1)) for row in Xb])
assert np.all(np.abs(freq - prob) < 5 * np.sqrt(prob * (1 - prob) / n) + 1e-4)
print("autoregressive-normalization: ok")
