# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/hypernetwork/ (ai://NEW-hypernetwork).
"""Numeric check of the derivation weight-scaling: row scaling, eq-scaled as eq-full, the counts and the init."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import hypernetwork as H  # noqa: E402

rng = np.random.default_rng(0)

# 1. row scaling is an elementwise product after the matrix
for _ in range(20):
    W, delta, v = rng.normal(size=(6, 4)), rng.normal(size=6), rng.normal(size=4)
    assert np.allclose(np.diag(delta) @ W @ v, delta * (W @ v))

# 2. eq-scaled is eq-full with the slices diag(W_hz[:, k]) W_h
for Nx, Nh, Nhat, Nz in [(3, 5, 4, 2), (2, 7, 3, 4), (1, 3, 6, 1)]:
    cfg = H.config_rnn(Nx, Nh, Nhat, Nz)
    p = H.init_rnn(rng, cfg)
    x = rng.normal(size=(3, 6, Nx))
    q = H.full_from_scaled(p)
    for k in range(Nz):
        assert np.allclose(q["W_hz"][:, :, k], np.diag(p["W_hz"][:, k]) @ p["W_h"])
    hs_full, Whs, Wxs = H.hyper_rnn_full(q, x)
    assert np.allclose(H.hyper_rnn(p, x), hs_full, atol=1e-14)

    # 3. the memory per step and the counts of parameters
    assert Whs.shape[2:] == (Nh, Nh) and Wxs.shape[2:] == (Nh, Nx)
    hyper = Nhat * (Nhat + Nh + Nx + 1) + 3 * Nz * Nhat + 2 * Nz
    assert H.rnn_param_count(cfg, full=True) == Nz * Nh * (Nh + Nx) + Nz * Nh + Nh + hyper
    assert H.rnn_param_count(cfg) == Nh * (Nh + Nx) + 3 * Nz * Nh + Nh + hyper
    assert (Nh * (Nh + Nx)) / (2 * Nh) == (Nh + Nx) / 2

# 4. the initialization of Appendix A.2.3: every scaling vector is 0.1
for Nz in (1, 4, 16):
    cfg = H.config_rnn(5, 8, 6, Nz)
    p = H.init_lstm(rng, cfg)
    out, ds = H.hyper_lstm(p, rng.normal(size=(4, 7, 5)), return_scales=True)
    assert np.allclose(ds, 0.1, atol=1e-15)
    assert np.all(p["W_bz"] == 0) and np.all(p["b_0"] == 0)

print("weight-scaling: ok")
