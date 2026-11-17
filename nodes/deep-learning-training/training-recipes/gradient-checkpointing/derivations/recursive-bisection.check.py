# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/gradient-checkpointing/ (ai://C4TQW1).
# Check of derivations/recursive-bisection.md: a simulation of the bisection schedule keeps 2 + ceil(log2 n)
# activations and runs n + R(n) layer forwards, which is n + (n / 2) log2 n when n is a power of two.
import math


def simulate(n):
    state = {"live": 2, "peak": 2, "forwards": n, "backwards": []}

    def bisect(lo, hi):
        if hi - lo == 1:
            state["backwards"].append(lo)
            return
        mid = (lo + hi) // 2
        state["forwards"] += mid - lo
        state["live"] += 1
        state["peak"] = max(state["peak"], state["live"])
        bisect(mid, hi)
        bisect(lo, mid)
        state["live"] -= 1

    bisect(0, n)
    assert state["backwards"] == list(reversed(range(n)))  # each layer once, last to first
    return state["peak"], state["forwards"]


def recurrence(n):
    r = [0, 0]
    for t in range(2, n + 1):
        r.append(t // 2 + r[t - t // 2] + r[t // 2])
    return n + r[n]


for n in range(1, 513):
    peak, forwards = simulate(n)
    assert peak == 2 + math.ceil(math.log2(n)), (n, peak)
    assert forwards == recurrence(n), (n, forwards)
    assert forwards <= n + n * math.ceil(math.log2(n)) / 2 + n

for p in range(0, 17):
    n = 2**p
    assert simulate(n) == (2 + p, n + n * p // 2)

print("n = 9:", simulate(9), "n = 16:", simulate(16), "n = 1024:", simulate(1024))
print("ok")
