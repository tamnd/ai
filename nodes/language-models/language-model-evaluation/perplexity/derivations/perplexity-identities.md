---
id: perplexity-identities
derives: [c-geometric-mean, c-base-invariant, c-uniform, c-bounds, c-corpus-weighting, c-expected-loss, eq-perplexity]
kind: proof
assumes: ["every scored token has a probability p_i in (0, 1] under the model", "the unit counts n_j are positive integers"]
uses: [shannon-entropy@m2, cross-entropy@m2, kullback-leibler-divergence@m2, gibbs-inequality@m2]
status: full
check: perplexity-identities.check.py
lean: null
sources:
  - {work: brown-1992-class-based-n-gram-models, locator: "p. 3"}
  - {work: brown-1992-entropy-upper-bound-english, locator: "p. 2"}
---

### Statement

A corpus has sequences $j = 1, \dots, m$.
Sequence $j$ has scored tokens with model probabilities $p_{j1}, \dots, p_{jT_j} \in (0, 1]$ and is normalised by $n_j \ge 1$ units.
Write $S_j = -\sum_i \ln p_{ji}$, $\mathrm{PPL}_j = e^{S_j / n_j}$, $N = \sum_j n_j$, and
$$ \mathrm{PPL} = \exp\Big(\frac{1}{N}\sum_{j} S_j\Big). $$

1. With one sequence and $n = T$ units, $\mathrm{PPL} = \big(\prod_{i=1}^{T} 1/p_i\big)^{1/T}$, the geometric mean of the inverse probabilities.
2. For any base $b > 1$, $\mathrm{PPL} = b^{H_b}$ with $H_b = \frac{1}{N}\sum_j S_j / \ln b$, so the perplexity is the same whether the cross-entropy is in nats or bits.
3. A model that gives every token probability $1/V$ has perplexity $V$ per token.
4. With $n = T$, $1 \le 1/\max_i p_i \le \mathrm{PPL} \le 1/\min_i p_i$ and $1/\bar{p} \le \mathrm{PPL} \le \overline{1/p}$, where the bars are arithmetic means over the tokens.
5. $\ln \mathrm{PPL} = \sum_j (n_j / N) \ln \mathrm{PPL}_j$, so $\min_j \mathrm{PPL}_j \le \mathrm{PPL} \le \max_j \mathrm{PPL}_j$; when all $n_j$ are equal, $\mathrm{PPL}$ is the geometric mean of the $\mathrm{PPL}_j$ and is at most their arithmetic mean $\frac{1}{m}\sum_j \mathrm{PPL}_j$.
6. If the next symbol has distribution $P$ on a finite set and the model predicts $M$, the expected log-loss is $\mathbb{E}_P[-\ln M] = H(P) + D_{\mathrm{KL}}(P \,\|\, M) \ge H(P)$, with equality exactly when $M = P$.

### Assumptions

- Every scored token has positive probability; a token with $p = 0$ makes $S_j$ and the perplexity infinite, and none of the inequalities is needed then.
- In item 6, $P$ and $M$ are distributions on the same finite set, and $M(x) > 0$ wherever $P(x) > 0$, so that the expectation is finite.

### Steps

1. With $n = T$, $\mathrm{PPL} = \exp(-\frac{1}{T}\sum_i \ln p_i) = \prod_i \exp(\ln(1/p_i))^{1/T} = (\prod_i 1/p_i)^{1/T}$.
2. $b^{H_b} = \exp(H_b \ln b) = \exp(\frac{1}{N}\sum_j S_j)$.
3. Each $-\ln p_i = \ln V$, so $\frac{1}{T}\sum_i(-\ln p_i) = \ln V$ and $\mathrm{PPL} = V$.
4. The geometric mean of the numbers $1/p_i \ge 1$ lies between their minimum and maximum. By concavity of $\ln$ (the inequality of arithmetic and geometric means), $\frac{1}{T}\sum_i \ln(1/p_i) \le \ln(\overline{1/p})$, which is the upper bound. Applied to the $p_i$ themselves, $\frac{1}{T}\sum_i \ln p_i \le \ln \bar{p}$, so $\ln \mathrm{PPL} = -\frac{1}{T}\sum_i \ln p_i \ge -\ln \bar{p}$, which is the lower bound.
5. $\frac{1}{N}\sum_j S_j = \sum_j \frac{n_j}{N}\cdot\frac{S_j}{n_j} = \sum_j \frac{n_j}{N}\ln \mathrm{PPL}_j$, a convex combination, so it lies between the smallest and largest $\ln \mathrm{PPL}_j$, and $\exp$ is increasing. With $n_j = n$ the weights are $1/m$, which gives the geometric mean, and the inequality of arithmetic and geometric means bounds it by the arithmetic mean.
6. $\mathbb{E}_P[-\ln M] = -\sum_x P(x)\ln M(x) = -\sum_x P(x)\ln P(x) + \sum_x P(x)\ln\frac{P(x)}{M(x)} = H(P) + D_{\mathrm{KL}}(P\,\|\,M)$, and Gibbs' inequality gives $D_{\mathrm{KL}} \ge 0$ with equality exactly when $M = P$.

### Result and check

Perplexity is the exponential of the per-unit cross-entropy, it does not depend on the base of the logarithm, it equals $V$ for the uniform model, it is bounded by the harmonic and arithmetic means of the inverse probabilities, the corpus value is the unit-weighted geometric mean of the per-sequence values, and in expectation no model beats the true distribution.
`perplexity-identities.check.py` draws random probabilities, unit counts and distributions and checks items 1 to 6 in float64; it ends with `perplexity-identities: ok`.
