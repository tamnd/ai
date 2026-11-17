# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/cosine-learning-rate-schedule/ (ai://FG7K47).
# Check of derivations/slope.md: the slope of eq-cosine is -(pi / 2T)(eta_max - eta_min) sin(pi t / T), zero at both ends
# and steepest at T / 2, where it is pi / 2 times the slope of a linear decay; near the ends it moves to second order.
import numpy as np

for hi, lo, T in [(0.1, 0.0, 100.0), (6e-4, 6e-5, 5000.0), (0.05, 0.0, 1.0)]:

    def eta(t):
        return lo + 0.5 * (hi - lo) * (1 + np.cos(np.pi * t / T))

    t = np.linspace(0.0, T, 10_001)
    h = T * 1e-6
    fd = (eta(t + h) - eta(t - h)) / (2 * h)
    exact = -np.pi / (2 * T) * (hi - lo) * np.sin(np.pi * t / T)
    assert np.allclose(fd, exact, rtol=1e-5, atol=1e-9 * (hi - lo) / T)
    # the steepest point is the middle, with |slope| = (pi / 2) (hi - lo) / T
    i = int(np.argmax(np.abs(fd)))
    assert abs(t[i] - T / 2) <= T / 10_000
    smax = np.pi / 2 * (hi - lo) / T
    assert abs(abs(fd[i]) - smax) < 1e-6 * smax
    assert abs(smax / ((hi - lo) / T) - np.pi / 2) < 1e-12
    # zero slope at both ends (one-sided differences shrink with the step)
    for end in (0.0, T):
        d = abs(eta(end + h) - eta(end)) / h if end == 0.0 else abs(eta(end) - eta(end - h)) / h
        assert d < 1e-5 * smax, (end, d)
    # second order at the ends: eta_max - eta(t) <= (pi^2 / 4)(hi - lo)(t / T)^2, and symmetric at the end
    u = t / T
    assert np.all(hi - eta(t) <= np.pi**2 / 4 * (hi - lo) * u**2 + 1e-15)
    assert np.all(eta(t) - lo <= np.pi**2 / 4 * (hi - lo) * (1 - u) ** 2 + 1e-15)
    # the cosine stays above the linear decay in the first half and below it in the second
    lin = hi - (hi - lo) * u
    assert np.all(eta(t[u < 0.5]) >= lin[u < 0.5] - 1e-15) and np.all(eta(t[u > 0.5]) <= lin[u > 0.5] + 1e-15)

# at integer steps the differences have the same shape: largest in the middle, smallest at the ends
T, hi, lo = 100, 0.1, 0.0
k = np.arange(T + 1)
d = np.diff(lo + 0.5 * (hi - lo) * (1 + np.cos(np.pi * k / T)))
assert np.argmax(np.abs(d)) in (T // 2 - 1, T // 2)
assert abs(d[0]) < abs(d[T // 2]) / 30 and abs(d[-1]) < abs(d[T // 2]) / 30
print("ok")
