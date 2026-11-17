# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/mode-connectivity/ (ai://NEW-mode-connectivity).
# Check of derivations/bezier-curve.md: the endpoints and midpoint of the quadratic Bezier curve, the bend factor
# 2 t (1 - t) against a finite difference, and the ring loss with barrier 1 on the segment and 1/16 on the curve.
import numpy as np


def bezier(w1, th, w2, t):
    return (1 - t) ** 2 * w1 + 2 * t * (1 - t) * th + t * t * w2


rng = np.random.default_rng(0)
for _ in range(20):
    w1, th, w2 = rng.normal(size=(3, 7))
    assert np.allclose(bezier(w1, th, w2, 0.0), w1)
    assert np.allclose(bezier(w1, th, w2, 1.0), w2)
    assert np.allclose(bezier(w1, th, w2, 0.5), (w1 + 2 * th + w2) / 4)
    assert np.allclose(bezier(w1, th, w2, 0.25), (9 * w1 + 6 * th + w2) / 16)
    t = rng.uniform()
    e = 1e-6 * np.eye(7)[0]
    fd = (bezier(w1, th + e, w2, t) - bezier(w1, th - e, w2, t)) / 2e-6
    assert np.allclose(fd, 2 * t * (1 - t) * np.eye(7)[0], atol=1e-8)


def ring(w):
    return (w @ w - 1) ** 2


a, b, th = np.array([-1.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 2.0])
ts = np.linspace(0, 1, 200001)
seg = np.array([ring((1 - t) * a + t * b) for t in ts[::100]])
assert abs(seg.max() - 1.0) < 1e-12 and abs(ts[::100][seg.argmax()] - 0.5) < 1e-12
curve = np.array([ring(bezier(a, th, b, t)) for t in ts])
s = ts * (1 - ts)
assert np.allclose(curve, (4 * s * (1 - 4 * s)) ** 2)
assert abs(curve.max() - 1 / 16) < 1e-9
tmax = ts[curve.argmax()]
assert min(abs(tmax - (1 - np.sqrt(0.5)) / 2), abs(tmax - (1 + np.sqrt(0.5)) / 2)) < 1e-5
assert abs(ring(bezier(a, th, b, 0.5))) < 1e-15
print("ok")
