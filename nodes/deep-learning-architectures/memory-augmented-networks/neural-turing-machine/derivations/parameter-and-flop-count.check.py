# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/neural-turing-machine/ (ai://NEW-neural-turing-machine).
"""Numeric check of the derivation parameter-and-flop-count."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import ntm  # noqa: E402

MACS = [0]
_einsum = np.einsum


def counting_einsum(spec, *ops):
    """Multiply-adds of an einsum: the product of the sizes of all the indices."""
    sizes = {}
    for term, op in zip(spec.split("->")[0].split(","), ops):
        sizes.update(zip(term, op.shape))
    MACS[0] += int(np.prod(list(sizes.values())))
    return _einsum(spec, *ops)


ntm.np.einsum = counting_einsum
rng = np.random.default_rng(3)
for x, y, h, n, m, nr, nw, s in ((4, 3, 8, 6, 4, 1, 1, 1), (2, 5, 7, 9, 3, 2, 1, 2), (3, 3, 5, 5, 2, 3, 2, 1),
                                 (9, 8, 100, 128, 20, 1, 1, 1)):
    cfg = ntm.config(x, y, h, n, m, nr, nw, s)
    params = ntm.init(rng, cfg)
    # Statement 1: the parameter count is the sum of the arrays.
    assert ntm.param_count(cfg) == sum(v.size for v in params.values())
    k = 2 * s + 1
    p_h = nr * (m + 3 + k) + nw * (3 * m + 3 + k)
    assert ntm.param_count(cfg) == h * (x + nr * m) + h + p_h * (h + 1) + y * (h + nr * m) + y
    # Statement 2: count the einsums of one forward pass (the cosine, the shift and the read); the three weight
    # matrices are each applied once per step to each sequence, B times their size.
    B, T = 2, 3
    MACS[0] = 0
    ntm.forward(params, rng.normal(size=(B, T, x)), cfg)
    assert MACS[0] == T * B * ((nr + nw) * n * (m + k) + nr * n * m)
    linear = sum(params[w].size for w in ("Wc", "Wh", "Wy"))
    assert linear == h * (x + nr * m) + p_h * h + y * (h + nr * m)
    assert ntm.flops(cfg, B, T) == 2 * (MACS[0] + T * B * linear)
    # Statement 3: N changes the FLOPs but not the parameters.
    big = ntm.config(x, y, h, 4 * n, m, nr, nw, s)
    assert ntm.param_count(big) == ntm.param_count(cfg)
    assert ntm.flops(big, B, T) - ntm.flops(cfg, B, T) == 2 * B * T * 3 * n * ((nr + nw) * (m + k) + nr * m)
    # Statement 4: a learned initial state would add N M + R M + (R + W) N.
    learned = n * m + nr * m + (nr + nw) * n
    state = ntm.initial_state(cfg, 1)
    assert learned == state[0].size + sum(r.size for r in state[2]) + sum(w.size for w in state[1])
print("parameter-and-flop-count: ok")
