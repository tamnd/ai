# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/convolutional-neural-network/ (ai://GNXZCP).
"""Numeric check of the derivation output-size: count the windows that fit, and evaluate the examples."""


def formula(n, k, s, p):
    return (n + 2 * p - k) // s + 1


def enumerate_windows(n, k, s, p):
    return sum(1 for start in range(0, n + 2 * p, s) if start + k <= n + 2 * p)


for n in range(1, 31):
    for k in range(1, 8):
        for s in range(1, 5):
            for p in range(0, 4):
                if k <= n + 2 * p:
                    assert formula(n, k, s, p) == enumerate_windows(n, k, s, p), (n, k, s, p)

for n in range(1, 40):
    for k in (1, 3, 5, 7):
        assert formula(n, k, 1, (k - 1) // 2) == n, (n, k)

sizes = [32]
for k, s in [(5, 1), (2, 2), (5, 1), (2, 2), (5, 1)]:
    sizes.append(formula(sizes[-1], k, s, 0))
assert sizes == [32, 28, 14, 10, 5, 1], sizes

assert formula(224, 11, 4, 0) == 54 and (224 - 11) % 4 == 1
assert formula(227, 11, 4, 0) == 55 and (227 - 11) % 4 == 0
assert formula(224, 11, 4, 2) == 55
print("output-size: ok")
