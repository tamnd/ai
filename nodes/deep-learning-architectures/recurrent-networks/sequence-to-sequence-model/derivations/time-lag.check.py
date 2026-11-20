# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/sequence-to-sequence-model/ (ai://NEW-sequence-to-sequence-model).
"""Numeric check of the derivation time-lag: the distances in order and reversed, and what reversal computes."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import seq2seq  # noqa: E402

for n in range(1, 201):
    fwd, rev = seq2seq.time_lags(n, False), seq2seq.time_lags(n, True)
    assert fwd == [n] * n
    assert rev == [2 * i - 1 for i in range(1, n + 1)]
    assert sum(rev) == sum(fwd) == n * n
    assert min(rev) == 1 and all(a < b for a, b in zip(rev, rev[1:]))

rng = np.random.default_rng(0)
cfg_rev, cfg_fwd = seq2seq.config(reverse=True), seq2seq.config(reverse=False)
params = seq2seq.init(rng, cfg_rev, scale=0.5)
src, tgt = rng.integers(1, 7, (2, 6)), rng.integers(1, 6, (2, 4))
a = seq2seq.forward(params, cfg_rev, src, tgt)[0]
assert np.array_equal(a, seq2seq.forward(params, cfg_fwd, src[:, ::-1], tgt)[0])
print("ok")
