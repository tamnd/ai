---
id: expected-tokens
derives: [c-et-pmf, c-et-mean, c-et-bounds, c-et-walltime, c-et-gamma-one, c-et-ops, c-et-tables]
kind: closed-form
assumes: ["the acceptance events of the gamma draft positions are independent with one probability alpha", "a run of the target on gamma + 1 prefixes in parallel takes the walltime of one run, and a draft run takes c times it"]
uses: []
status: full
check: expected-tokens.check.py
lean: null
sources:
  - {work: leviathan-2023-fast-inference-speculative-decoding, locator: "p. 3"}
  - {work: leviathan-2023-fast-inference-speculative-decoding, locator: "p. 4"}
  - {work: leviathan-2023-fast-inference-speculative-decoding, locator: "p. 5"}
---

### Statement

In one run of Algorithm 1 with $\gamma \ge 0$ draft tokens, let $N$ be the number of tokens produced, and suppose each draft position is accepted with probability $\alpha \in [0, 1]$, independently, as long as all earlier positions were accepted.
Then (i) $P(N = n) = \alpha^{n-1}(1 - \alpha)$ for $1 \le n \le \gamma$ and $P(N = \gamma + 1) = \alpha^\gamma$, a geometric variable capped at $\gamma + 1$;
(ii) $E[N] = \sum_{k=0}^{\gamma} \alpha^k = (1 - \alpha^{\gamma+1})/(1 - \alpha)$ for $\alpha < 1$, and $\gamma + 1$ at $\alpha = 1$;
(iii) $1 \le E[N] \le \min(\gamma + 1, 1/(1 - \alpha))$, and $E[N]$ increases with $\alpha$ and with $\gamma$;
(iv) if one run costs $\gamma c + 1$ target-call times, the expected walltime improvement over one token per target call is $E[N]/(\gamma c + 1)$;
(v) at $\gamma = 1$ that improvement is $(1 + \alpha)/(1 + c)$, above 1 exactly when $\alpha > c$;
(vi) if a draft run costs $\hat c$ times the arithmetic of a target run and the target runs on $\gamma + 1$ prefixes, the arithmetic per token grows by the factor $(\gamma \hat c + \gamma + 1)/E[N]$.

### Assumptions

1. The acceptance of position $i$ given that positions $1, \dots, i-1$ were accepted is a Bernoulli($\alpha$) event independent of the others; Leviathan et al. call this the simplifying assumption that the $\beta$s are i.i.d.
2. Every run produces the accepted draft tokens plus one token: the resample at the first rejection or the bonus token.
3. For (iv), the $\gamma + 1$ target evaluations run concurrently in the walltime of one, and each of the $\gamma$ draft runs takes $c$ times a target run. For (vi), arithmetic is counted in units of one target run on one prefix.

### Steps

1. $N = n \le \gamma$ means positions $1, \dots, n-1$ were accepted and position $n$ rejected: probability $\alpha^{n-1}(1-\alpha)$. $N = \gamma + 1$ means all $\gamma$ were accepted: $\alpha^\gamma$. These sum to $(1 - \alpha^\gamma) + \alpha^\gamma = 1$ by the geometric sum.
2. $N = 1 + \sum_{k=1}^{\gamma} \mathbb 1[\text{positions } 1..k \text{ accepted}]$, so by linearity $E[N] = \sum_{k=0}^{\gamma} \alpha^k$, which is $(1 - \alpha^{\gamma+1})/(1 - \alpha)$ for $\alpha < 1$ and $\gamma + 1$ at $\alpha = 1$.
3. Each term $\alpha^k$ lies in $[0, 1]$ and the first is 1, so $1 \le E[N] \le \gamma + 1$; the partial sums are below $\sum_{k \ge 0} \alpha^k = 1/(1-\alpha)$; each term grows with $\alpha$, and adding a term grows the sum.
4. One run takes $\gamma c + 1$ units of walltime and yields $E[N]$ tokens on average, against 1 token per unit for standard decoding, so the rate improves by $E[N]/(\gamma c + 1)$.
5. At $\gamma = 1$: $(1 + \alpha)/(1 + c)$, which exceeds 1 iff $\alpha > c$.
6. A run spends $\gamma \hat c$ units on drafts and $\gamma + 1$ units on the target and yields $E[N]$ tokens; standard decoding spends 1 unit per token.

### Result and check

The check confirms (i) and (ii) with exact fractions for $\alpha \in \{0, 1/4, 1/2, 3/4, 1\}$ and $\gamma \le 8$, samples the capped count 200,000 times for several $(\alpha, \gamma)$, checks the bounds and the monotonicity on a grid, recomputes the six rows of Table 1 of Leviathan et al. (speed and operations at $c = \hat c = 0$) and two rows of their Table 4 to the printed precision, and the worked example $E[N] = 175/64$ for $\alpha = 3/4$, $\gamma = 3$.
