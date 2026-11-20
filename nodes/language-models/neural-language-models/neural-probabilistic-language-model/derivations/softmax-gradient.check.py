# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/neural-probabilistic-language-model/ (ai://NEW-neural-probabilistic-language-model).
"""Check of softmax-gradient: the derivative of -log softmax(y)_w is p_j - 1{j = w}, the entries sum to zero, and
the derivative of tanh is 1 - tanh^2."""

import math
import random


def nll(y, w):
    top = max(y)
    return -(y[w] - top) + math.log(sum(math.exp(v - top) for v in y))


def main():
    rng = random.Random(3)
    worst = 0.0
    for _ in range(200):
        V = rng.randint(2, 30)
        y = [rng.gauss(0, 3) for _ in range(V)]
        w = rng.randrange(V)
        top = max(y)
        z = sum(math.exp(v - top) for v in y)
        p = [math.exp(v - top) / z for v in y]
        grad = [p[j] - (j == w) for j in range(V)]
        assert abs(sum(grad)) < 1e-12 and grad[w] < 0 and all(g > 0 for j, g in enumerate(grad) if j != w)
        for j in range(V):
            eps = 1e-6
            hi, lo = list(y), list(y)
            hi[j] += eps
            lo[j] -= eps
            num = (nll(hi, w) - nll(lo, w)) / (2 * eps)
            worst = max(worst, abs(num - grad[j]))
    assert worst < 1e-8
    for _ in range(200):
        o = rng.gauss(0, 2)
        num = (math.tanh(o + 1e-6) - math.tanh(o - 1e-6)) / 2e-6
        assert abs(num - (1 - math.tanh(o) ** 2)) < 1e-9
    print(f"softmax-gradient: largest finite-difference error {worst:.2e} over 200 random logit vectors")


if __name__ == "__main__":
    main()
