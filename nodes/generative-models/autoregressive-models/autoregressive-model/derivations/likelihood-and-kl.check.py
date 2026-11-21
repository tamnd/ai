# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/autoregressive-model/ (ai://NEW-autoregressive-model).
"""Numeric check of the derivation likelihood-and-kl."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import autoregressive as ar  # noqa: E402

rng = np.random.default_rng(3)
for _ in range(20):
    q, p = rng.dirichlet(np.ones(16)), rng.dirichlet(np.ones(16))
    ce, h, kl = -np.sum(q * np.log(p)), -np.sum(q * np.log(q)), np.sum(q * np.log(q / p))
    assert abs(ce - (h + kl)) < 1e-12 and kl > 0 and ce > h
    assert abs(-np.sum(q * np.log(q)) - h) < 1e-15  # p = q attains the bound

cfg = ar.config(5, (8, 8))
p = {k: rng.normal(0.0, 1.0, s) for k, s in ar.param_shapes(cfg).items()}
mks = ar.mask_list(cfg, rng, 5)
X = ar.all_binary(5)
ens = ar.ensemble_log_prob(p, cfg, mks, X)
members = np.array([ar.log_prob(p, cfg, mk, X) for mk in mks])
assert abs(np.exp(ens).sum() - 1.0) < 1e-12
assert np.all(ens >= members.mean(0) - 1e-12)
assert np.all(np.abs(ens - np.log(np.exp(members).mean(0))) < 1e-12)

# a trained MADE: the data cross-entropy splits into entropy plus KL
target = rng.dirichlet(np.ones(32) * 0.3)
data = X[rng.choice(32, 3000, p=target)]
cfg = ar.config(5, (24,))
p = ar.init(rng, cfg)
mk = ar.masks(cfg, ar.degrees(cfg))
ar.train(p, cfg, data, 400, rng, [mk])
lp = ar.log_prob(p, cfg, mk, X)
ce, h = -np.sum(target * lp), -np.sum(target * np.log(target))
kl = np.sum(target * (np.log(target) - lp))
assert abs(ce - (h + kl)) < 1e-10 and 0 < kl < 0.2
print("likelihood-and-kl: ok")
