# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/neural-turing-machine/ (ai://NEW-neural-turing-machine).
"""Numeric check of the derivation shift-and-sharpen."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import ntm  # noqa: E402

rng = np.random.default_rng(2)
# Statement 1: a one-hot shift rotates.
w = ntm.softmax(rng.normal(size=(2, 9)))
for big_s in (1, 2, 4):
    for d in range(-big_s, big_s + 1):
        s = np.zeros((2, 2 * big_s + 1))
        s[:, d + big_s] = 1.0
        np.testing.assert_allclose(ntm.shift(w, s), np.roll(w, d, axis=-1))

# Statement 2: the gate at 0 and 1.
wc, wp = ntm.softmax(rng.normal(size=(2, 9))), ntm.softmax(rng.normal(size=(2, 9)))
np.testing.assert_array_equal(ntm.interpolate(wc, wp, np.ones(2)), wc)
np.testing.assert_array_equal(ntm.interpolate(wc, wp, np.zeros(2)), wp)

# Statement 3: repeated shifts add the mean and the variance of s.
n, start = 101, 50
pos = np.arange(n)
for s_vals in ([0.1, 0.8, 0.1], [0.0, 0.3, 0.7], [0.2, 0.1, 0.3, 0.1, 0.3]):
    s = np.array([s_vals])
    shifts = np.arange(len(s_vals)) - (len(s_vals) - 1) // 2
    mu, var = (s[0] * shifts).sum(), (s[0] * shifts**2).sum() - (s[0] * shifts).sum() ** 2
    w = np.eye(n)[[start]]
    for k in range(1, 21):
        w = ntm.shift(w, s)
        mean = (w[0] * pos).sum()
        assert abs(mean - (start + k * mu)) < 1e-9
        assert abs((w[0] * (pos - mean) ** 2).sum() - k * var) < 1e-9

# Statement 4: sharpening.
w = ntm.softmax(rng.normal(size=(3, 7)) * 2)
np.testing.assert_allclose(ntm.sharpen(w, np.ones(3)), w)
prev = w
for g in (1.5, 3.0, 10.0, 500.0):
    out = ntm.sharpen(w, np.full(3, g))
    if g <= 10:  # beyond, the small weights underflow to 0 and tie
        assert np.all(np.argsort(out, -1) == np.argsort(w, -1))
    assert np.all(out.max(-1) >= prev.max(-1) - 1e-12)
    prev = out
assert np.all(prev.max(-1) > 0.999)
tie = np.array([[0.4, 0.4, 0.2]])
np.testing.assert_allclose(ntm.sharpen(tie, np.array([500.0])), [[0.5, 0.5, 0.0]], atol=1e-12)
np.testing.assert_array_equal(ntm.sharpen(np.eye(5)[[3]], np.array([4.0])), np.eye(5)[[3]])
print("shift-and-sharpen: ok")
