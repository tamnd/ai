# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/discrete-diffusion-model/ (ai://NEW-discrete-diffusion-model).
"""Numeric check of the derivation parameter-and-flop-count."""

import itertools
import json

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import discrete_diffusion as dd  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent.parent
for c in json.loads((HERE / "fixtures" / "shapes.json").read_text())["configs"]:
    k = c["config"]
    cfg = dd.config(k["seq_len"], k["vocab"], k["hidden"])
    assert dd.param_count(cfg) == c["model"]["params"]
    assert dd.flops_forward(cfg, k["B"]) == c["model"]["flops_forward"]

for L, K, H, B in itertools.product((1, 3, 7), (2, 5), (1, 4, 9), (1, 6)):
    cfg = dd.config(L, K, H)
    P = L * (K + 1) * H + H + H * L * K + L * K
    assert dd.param_count(cfg) == P
    assert sum(int(np.prod(s)) for s in dd.param_shapes(cfg).values()) == P
    assert dd.flops_forward(cfg, B) == 2 * B * L * (K + 1) * H + 2 * B * H * L * K


class Counted(np.ndarray):
    macs = 0

    def __matmul__(self, other):
        a, b = np.asarray(self), np.asarray(other)
        Counted.macs += a.shape[0] * a.shape[1] * b.shape[1]
        return a @ b

    def __rmatmul__(self, other):
        a, b = np.asarray(other), np.asarray(self)
        Counted.macs += a.shape[0] * a.shape[1] * b.shape[1]
        return a @ b


rng = np.random.default_rng(7)
for L, K, H, B in ((3, 3, 5, 4), (4, 2, 7, 3)):
    cfg = dd.config(L, K, H)
    p = {n: (v.view(Counted) if n.startswith("W") else v) for n, v in dd.init(rng, cfg).items()}
    Counted.macs = 0
    dd.forward(p, rng.integers(0, K + 1, size=(B, L)), cfg)
    assert 2 * Counted.macs == dd.flops_forward(cfg, B), (Counted.macs, dd.flops_forward(cfg, B))

    calls = 0
    plain = dd.init(rng, cfg)
    den = dd.denoise_fn(plain, cfg)

    def counted(z, den=den):
        global calls
        calls += 1
        return den(z)

    dd.exact_nelbo(counted, rng.integers(0, K, size=L), K)
    assert calls == 2**L - 1
