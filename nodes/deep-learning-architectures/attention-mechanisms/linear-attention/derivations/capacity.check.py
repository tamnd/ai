# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/linear-attention/ (ai://NEW-linear-attention).
"""Numeric check of the derivation capacity: the rank bound on exact recall, the crosstalk of the sum rule, the zero
denominator of ReLU features, and the delta rule overwriting a value."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import linear_attention as L  # noqa: E402

rng = np.random.default_rng(3)
for C in (2, 4, 8):
    for n_keys in range(1, 3 * C):
        Phi = L.elu_plus_one(rng.normal(size=(n_keys, C)))  # rows are the mapped keys
        W = np.eye(n_keys)  # orthonormal values, M = L
        S, *_ = np.linalg.lstsq(Phi, W, rcond=None)  # the best state for recall, in least squares
        err = np.abs(Phi @ S - W).max()
        # 1. exact recall is possible for L <= C (generic keys) and impossible for L > C
        assert (err < 1e-8) == (n_keys <= C), (C, n_keys, err)
        if n_keys > C:
            assert np.linalg.matrix_rank(Phi @ S) <= C
    # 2. orthonormal keys: exact recall by the sum rule, and the crosstalk of non-orthogonal ones
    keys, vals = np.linalg.qr(rng.normal(size=(C, C)))[0].T, rng.normal(size=(C, 3))
    _, S = L.fast_weight_rule(keys, keys, vals)
    assert np.allclose(keys @ S, vals)
    keys2 = rng.normal(size=(C, C))
    _, S2 = L.fast_weight_rule(keys2, keys2, vals)
    G = keys2 @ keys2.T
    assert np.allclose(keys2 @ S2, G @ vals)
# 3. ReLU can zero a query, elu + 1 cannot
q = -np.abs(rng.normal(size=(1, 5)))
assert np.all(L.relu(q) == 0) and np.all(L.elu_plus_one(q) > 0)
x = np.linspace(-50, 50, 2001)
assert np.all(L.elu_plus_one(x) > 0)
# 4. the delta rule overwrites, the sum rule adds
for _ in range(10):
    k = rng.normal(size=6)
    k /= np.linalg.norm(k)
    v1, v2 = rng.normal(size=4), rng.normal(size=4)
    K, V = np.stack([k, k]), np.stack([v1, v2])
    assert np.allclose(k @ L.fast_weight_rule(K, K, V)[1], v1 + v2)
    assert np.allclose(k @ L.fast_weight_rule(K, K, V, np.ones(2))[1], v2)
print("capacity: ok")
