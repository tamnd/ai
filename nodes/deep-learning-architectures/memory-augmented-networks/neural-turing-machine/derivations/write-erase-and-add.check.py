# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/neural-turing-machine/ (ai://NEW-neural-turing-machine).
"""Numeric check of the derivation write-erase-and-add."""

import itertools
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import ntm  # noqa: E402

rng = np.random.default_rng(1)
for heads in (1, 2, 3, 4):
    mem = rng.normal(size=(2, 6, 3))
    ws = [ntm.softmax(rng.normal(size=(2, 6)) * 3) for _ in range(heads)]
    es = [rng.uniform(size=(2, 3)) for _ in range(heads)]
    adds = [rng.normal(size=(2, 3)) for _ in range(heads)]
    closed = mem * np.prod([1 - w[:, :, None] * e[:, None, :] for w, e in zip(ws, es)], axis=0)
    closed = closed + sum(w[:, :, None] * a[:, None, :] for w, a in zip(ws, adds))
    out = ntm.write(mem, ws, es, adds)
    np.testing.assert_allclose(out, closed)
    # Statement 1: every order of the heads gives the same memory.
    for order in itertools.permutations(range(heads)):
        np.testing.assert_allclose(ntm.write(mem, [ws[i] for i in order], [es[i] for i in order],
                                             [adds[i] for i in order]), out)

# Statement 2: w = 1 and e = 1 replace the row; e < 1 in one column keeps part of it.
mem = rng.normal(size=(1, 4, 3))
a = rng.normal(size=(1, 3))
w = np.eye(4)[[2]]
np.testing.assert_allclose(ntm.write(mem, [w], [np.ones((1, 3))], [a])[0, 2], a[0])
part = ntm.write(mem, [w], [np.array([[1.0, 0.5, 1.0]])], [np.zeros((1, 3))])[0, 2]
np.testing.assert_allclose(part, [0.0, 0.5 * mem[0, 2, 1], 0.0])
# Statement 3: weight p on a row keeps the fraction 1 - p.
for p in (0.1, 0.5, 0.9):
    w = np.array([[p, 1 - p, 0.0, 0.0]])
    out = ntm.write(mem, [w], [np.ones((1, 3))], [np.zeros((1, 3))])
    np.testing.assert_allclose(out[0, 0], (1 - p) * mem[0, 0])
    np.testing.assert_allclose(out[0, 1], p * mem[0, 1])
print("write-erase-and-add: ok")
