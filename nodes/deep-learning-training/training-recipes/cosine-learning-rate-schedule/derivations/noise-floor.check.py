# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/cosine-learning-rate-schedule/ (ai://NEW-cosine-learning-rate-schedule).
# Check of derivations/noise-floor.md: on f(x) = h x^2 / 2 with additive gradient noise, v = E[x^2] follows
# v' = (1 - eta h)^2 v + eta^2 sigma^2, a constant rate settles at eta sigma^2 / (h (2 - eta h)), and decaying the rate
# from 0.1 over 1,000 steps ends far below that floor. This is the page's own toy run.
import numpy as np

h, s2, T = 1.0, 1.0, 1000


def floor(eta):
    return eta * s2 / (h * (2 - eta * h))


def run(etas, v0):
    v = v0
    for e in etas:
        v = (1 - e * h) ** 2 * v + e**2 * s2
    return v


# (ii) the fixed point, from above and from below
for eta in [0.5, 0.1, 0.01]:
    for v0 in [0.0, 1.0]:
        assert abs(run([eta] * 20_000, v0) - floor(eta)) < 1e-9
    assert abs(0.5 * h * floor(eta) / (eta * s2 / 4) - 1) < eta  # the loss floor is about eta sigma^2 / 4

t = np.arange(T)
schedules = {
    "constant": np.full(T, 0.1),
    "cosine": 0.05 * (1 + np.cos(np.pi * t / T)),
    "linear": 0.1 * (1 - t / T),
    "step": 0.1 * 0.1 ** ((t >= 500).astype(int) + (t >= 750).astype(int)),
    "wsd": np.where(t <= 800, 0.1, 0.1 * (1 - (t - 800) / 200)),
}
v0 = floor(0.1)
v = {k: run(e, v0) for k, e in schedules.items()}
expect = {"constant": 0.0526, "cosine": 0.0036, "linear": 0.0044, "step": 0.0034, "wsd": 0.0100}
for k in expect:
    assert abs(v[k] - expect[k]) < 5e-5, (k, v[k])
for k in ["cosine", "linear", "step", "wsd"]:
    assert v[k] * 5 < v["constant"], k

# the recursion against Monte Carlo: 20,000 independent runs of noisy SGD with the cosine schedule
rng = np.random.default_rng(0)
x = rng.normal(scale=np.sqrt(v0), size=20_000)
for e in schedules["cosine"]:
    x = x - e * (h * x + rng.normal(scale=np.sqrt(s2), size=x.size))
assert abs(np.mean(x**2) / v["cosine"] - 1) < 0.04, (np.mean(x**2), v["cosine"])
print("ok")
