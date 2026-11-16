---
id: pruning-schedule
derives: [c-schedule, c-rounds, c-per-round, c-lenet-schedule]
kind: closed-form
assumes: ["each round removes the same fraction p of the weights still alive", "0 < p < 1", "a removed weight stays removed"]
uses: []
status: full
check: pruning-schedule.check.py
lean: null
sources:
  - {work: frankle-2018-lottery-ticket-hypothesis, locator: "p. 2"}
  - {work: frankle-2018-lottery-ticket-hypothesis, locator: "p. 3"}
  - {work: frankle-2018-lottery-ticket-hypothesis, locator: "p. 4"}
---

### Statement

Let each round of iterative pruning remove the fraction $p$ of the weights that are still alive, with $0 < p < 1$.
Then after $n$ rounds the fraction of weights left is

$$ P_m(n) = (1 - p)^n. $$

The least number of rounds that brings $P_m$ to a target $s \in (0, 1]$ or below is

$$ n^*(p, s) = \left\lceil \frac{\ln s}{\ln(1 - p)} \right\rceil. $$

To remove a total fraction $P$ in $n$ equal rounds, each round prunes $1 - (1 - P)^{1/n}$ of the survivors.
With a different rate $p_\ell$ in each layer $\ell$ of $N_\ell$ weights, $P_m(n) = \sum_\ell N_\ell (1 - p_\ell)^n / \sum_\ell N_\ell$.

### Assumptions

1. Each round prunes the fraction $p$ of the survivors, not of the original weights.
2. A pruned weight is never restored.
3. Weights are counted as real numbers; the check also prunes whole weights, rounding $p$ times the alive count half up.

### Steps

1. Let $a_n$ be the fraction alive after $n$ rounds, with $a_0 = 1$. Round $n + 1$ removes $p\,a_n$, so $a_{n+1} = (1 - p)\,a_n$, and by induction $a_n = (1 - p)^n$.
2. As $0 < 1 - p < 1$, $(1 - p)^n$ decreases in $n$. So $(1 - p)^n \le s$ holds exactly when $n \ln(1 - p) \le \ln s$, that is, when $n \ge \ln s / \ln(1 - p)$, since $\ln(1 - p) < 0$ flips the inequality. The least such integer is the ceiling.
3. For a total fraction $P$ in $n$ rounds, solve $(1 - r)^n = 1 - P$ for $r$.
4. With a rate per layer, step 1 holds in each layer, and the fraction of the whole network is the weighted mean of the layers.

### Result and check

At $p = 0.2$, reaching 10% takes $\lceil \ln 0.1 / \ln 0.8 \rceil = 11$ rounds, where $0.8^{11} = 0.0859$.
Reaching 3.6% takes 15 rounds, and each round is one full training, after the training of the dense network.
Frankle and Carbin prune the hidden layers of Lenet 300-100-10 at 20% per round and the output layer at 10% (p. 3).
The layers have $784 \cdot 300 = 235{,}200$, $300 \cdot 100 = 30{,}000$ and $100 \cdot 10 = 1{,}000$ weights, 266,200 in all.
The formula of step 4 gives $P_m$ = 21.1%, 13.5%, 3.6% and 2.9% after 7, 9, 15 and 16 rounds, the levels the paper reports for Lenet (p. 4).
To prune 90% in 10 equal rounds, each round removes $1 - 0.1^{1/10} = 20.6\%$.
`pruning-schedule.check.py` checks the recursion against the closed form, the round count against a direct search, the per-round rate, the Lenet levels with real and with whole weights, and the 24.47% left in the 20-100-100-2 MLP of this page's experiment after 4 rounds.
