# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/end-to-end-memory-network/ (ai://NEW-end-to-end-memory-network).
"""Numeric check of the derivation parameter-and-flop-count."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import memn2n  # noqa: E402

MACS = [0]
_einsum = np.einsum


def counting_einsum(spec, *ops):
    """Multiply-adds of an einsum: the product of the sizes of all the indices; ... stands for the batch axes."""
    ins = spec.split("->")[0].split(",")
    sizes = {}
    for term, op in zip(ins, ops):
        if term.startswith("..."):
            lead = op.ndim - (len(term) - 3)
            sizes.update({f"lead{i}": op.shape[i] for i in range(lead)})
            term, shape = term[3:], op.shape[lead:]
        else:
            shape = op.shape
        sizes.update(zip(term, shape))
    MACS[0] += int(np.prod(list(sizes.values())))
    return _einsum(spec, *ops)


memn2n.np.einsum = counting_einsum
rng = np.random.default_rng(5)
for v, d, k, n, j in ((12, 6, 3, 5, 4), (20, 3, 1, 7, 2), (9, 8, 4, 3, 6), (177, 20, 3, 50, 7)):
    for tying in ("adjacent", "layerwise"):
        cfg = memn2n.config(v, d, k, n, j, tying=tying)
        params = memn2n.init(rng, cfg)
        # Statement 1: the count is the arrays, and J does not enter it.
        assert memn2n.param_count(cfg) == sum(x.size for x in params.values())
        assert memn2n.param_count(cfg) == memn2n.param_count(memn2n.config(v, d, k, n, 1, tying=tying))
        if tying == "adjacent":
            assert memn2n.param_count(cfg) == (k + 1) * (v * d + n * d)
        else:
            assert memn2n.param_count(cfg) == 4 * v * d + d * d + 2 * n * d
        # Statement 4: count the einsums of a forward pass, add the matrix products (the answer, and H for each hop).
        B = 3
        MACS[0] = 0
        memn2n.forward(params, rng.integers(0, v, (B, n, j)), rng.integers(0, v, (B, j)), cfg)
        encodings = k + 1 if tying == "adjacent" else 2
        assert MACS[0] == B * (j * d * (encodings * n + 1) + 2 * k * n * d)
        matmul = B * v * d + (B * k * d * d if tying == "layerwise" else 0)
        assert memn2n.flops(cfg, B) == 2 * (MACS[0] + matmul)
        if tying == "adjacent":
            assert memn2n.flops(cfg, B) == B * (2 * j * d * ((k + 1) * n + 1) + 4 * k * n * d + 2 * v * d)

# Statement 2: no tying against adjacent tying, at the bAbI sizes.
v, d, k, n = 177, 20, 3, 50
untied = 2 * k * (v * d + n * d) + 2 * v * d
assert (memn2n.param_count(memn2n.config(v, d, k, n, 7)), untied) == (18_160, 34_320)
# Statement 3: the language model against a simple recurrent network at the Penn Treebank sizes.
v, d, n = 10_000, 150, 100
lm, rnn = 3 * v * d + d * d + 2 * n * d, 2 * v * d + d * d
assert (lm, rnn) == (4_552_500, 3_022_500) and round(lm / rnn, 3) == 1.506
# The story term beats the answer term at the bAbI sizes.
assert 7 * 4 * 50 > 177
print("parameter-and-flop-count: ok")
