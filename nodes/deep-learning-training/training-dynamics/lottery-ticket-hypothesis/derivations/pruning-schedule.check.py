# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/lottery-ticket-hypothesis/ (ai://NEW-lottery-ticket-hypothesis).
# Check of derivations/pruning-schedule.md: pruning the fraction p of the survivors in each of n rounds leaves (1 - p)^n;
# the least n with (1 - p)^n <= s is ceil(ln s / ln(1 - p)); the per-round rate for a total fraction P in n rounds is
# 1 - (1 - P)^(1/n); and Lenet 300-100-10, pruned at 20% per round with the output layer at 10%, has P_m = 21.1%, 13.5%,
# 3.6% and 2.9% after 7, 9, 15 and 16 rounds, with whole weights pruned in each round.
import math


def alive_after(n, rate, rounds):
    """Weights left in a layer of n weights after `rounds` rounds that prune round(rate * alive), half up."""
    for _ in range(rounds):
        n -= math.floor(rate * n + 0.5)
    return n


# (1 - p)^n by repeated pruning of a real fraction.
for p in [0.1, 0.2, 0.3, 0.5]:
    left = 1.0
    for n in range(1, 31):
        left -= p * left
        assert abs(left - (1 - p) ** n) < 1e-12

# The least n that reaches a target s, against a direct search.
for p in [0.05, 0.1, 0.2, 0.3, 0.5, 0.9]:
    for s in [0.5, 0.2, 0.1, 0.05, 0.036, 0.01, 0.001]:
        n = 0
        while (1 - p) ** n > s:
            n += 1
        assert n == math.ceil(math.log(s) / math.log(1 - p)), (p, s, n)
assert math.ceil(math.log(0.1) / math.log(0.8)) == 11 and abs(0.8**11 - 0.0859) < 1e-4
assert math.ceil(math.log(0.036) / math.log(0.8)) == 15

# The per-round rate that removes a total fraction P in n rounds.
for big_p in [0.5, 0.9, 0.99]:
    for n in [1, 2, 5, 10]:
        r = 1 - (1 - big_p) ** (1 / n)
        assert abs((1 - r) ** n - (1 - big_p)) < 1e-12
assert abs(1 - 0.1 ** (1 / 10) - 0.2057) < 1e-4  # 90% in 10 rounds needs 20.6% per round

# Lenet 300-100-10: 266,200 weights; hidden layers pruned at 20% per round and the output layer at 10%.
layers = [(784 * 300, 0.2), (300 * 100, 0.2), (100 * 10, 0.1)]
total = sum(n for n, _ in layers)
assert total == 266200
for rounds, reported in [(7, 21.1), (9, 13.5), (15, 3.6), (16, 2.9)]:
    exact = (265200 * 0.8**rounds + 1000 * 0.9**rounds) / total
    whole = sum(alive_after(n, r, rounds) for n, r in layers) / total
    assert abs(100 * exact - reported) < 0.05 and abs(100 * whole - reported) < 0.05, (rounds, exact, whole)

# The MLP of the experiment: 20-100-100-2 with no biases, 30% per round and 15% for the output layer, 4 rounds.
layers = [(20 * 100, 0.3), (100 * 100, 0.3), (100 * 2, 0.15)]
whole = sum(alive_after(n, r, 4) for n, r in layers) / sum(n for n, _ in layers)
assert abs(whole - 0.2447) < 5e-5, whole
assert abs(0.7**4 - 0.2401) < 1e-12

print("pruning-schedule: ok")
