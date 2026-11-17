# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/weak-to-strong-generalization/ (ai://G0ZGHC).
# Check of derivations/pgr.md: PGR = (w2s - weak) / (ceiling - weak) is 0 at the weak performance and 1 at the
# ceiling, undefined at a zero gap and unbounded near it, invariant under P -> aP + b with a != 0, outside [0, 1]
# when the student is worse than the supervisor or better than the ceiling, and 0 when the student copies the supervisor.
import numpy as np


def pgr(w, s, c):
    if c == w:
        raise ZeroDivisionError
    return (s - w) / (c - w)


rng = np.random.default_rng(3)
for _ in range(1000):
    w, c = rng.uniform(0, 1, size=2)
    if abs(c - w) < 1e-6:
        continue
    assert pgr(w, w, c) == 0.0
    assert abs(pgr(w, c, c) - 1.0) < 1e-12
    s = rng.uniform(0, 1)
    a = rng.choice([-1.0, 1.0]) * rng.uniform(0.01, 100)
    b = rng.normal() * 10
    assert abs(pgr(a * w + b, a * s + b, a * c + b) - pgr(w, s, c)) < 1e-6 * (
        1 + abs(pgr(w, s, c))
    )
    assert abs(pgr(1 - w, 1 - s, 1 - c) - pgr(w, s, c)) < 1e-9 * (1 + abs(pgr(w, s, c)))

try:
    pgr(0.7, 0.75, 0.7)
    raise AssertionError("expected an undefined PGR")
except ZeroDivisionError:
    pass
vals = [abs(pgr(0.7, 0.75, 0.7 + g)) for g in (1e-1, 1e-3, 1e-6)]
assert vals[0] < vals[1] < vals[2] and vals[2] > 1e4

assert pgr(0.6, 0.55, 0.8) < 0 and pgr(0.6, 0.85, 0.8) > 1

y = rng.integers(0, 2, size=200)
weak_pred = np.where(rng.uniform(size=200) < 0.7, y, 1 - y)
student_pred = weak_pred.copy()
ceiling_pred = y.copy()
acc = lambda pr: float(np.mean(pr == y))  # noqa: E731
assert pgr(acc(weak_pred), acc(student_pred), acc(ceiling_pred)) == 0.0
print("ok")
