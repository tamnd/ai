---
id: unbiased-norm-estimators
derives: [c-two-batch-unbiased, c-per-example-bessel, c-ratio-biased]
kind: closed-form
assumes: ["the examples of each batch are independent draws from the data distribution", "the two batch sizes differ"]
uses: []
status: full
check: unbiased-norm-estimators.check.py
lean: null
sources:
  - {work: mccandlish-2018-empirical-model-large-batch-training, locator: "p. 5"}
  - {work: mccandlish-2018-empirical-model-large-batch-training, locator: "p. 17"}
---

### Statement

Let $G_{\text{small}}$ and $G_{\text{big}}$ be mean gradients over batches of sizes $B_{\text{small}} < B_{\text{big}}$.
Define

$$\widehat{\lVert G\rVert^2} = \frac{B_{\text{big}} \lVert G_{\text{big}}\rVert^2 - B_{\text{small}} \lVert G_{\text{small}}\rVert^2}{B_{\text{big}} - B_{\text{small}}}, \qquad \hat S = \frac{\lVert G_{\text{small}}\rVert^2 - \lVert G_{\text{big}}\rVert^2}{1/B_{\text{small}} - 1/B_{\text{big}}}.$$

(i) $\mathbb{E}\,\widehat{\lVert G\rVert^2} = \lVert G\rVert^2$ and $\mathbb{E}\,\hat S = \operatorname{tr}\Sigma$, whether or not the small batch is part of the big one.

(ii) With $B_{\text{small}} = 1$, averaging $\lVert G_{\text{small}}\rVert^2$ over the $n$ examples of the big batch turns $\hat S$ into the sample variance with divisor $n - 1$, and $\widehat{\lVert G\rVert^2}$ into $\lVert \bar g\rVert^2 - \hat S/n$.

(iii) The ratio $\hat S / \widehat{\lVert G\rVert^2}$ is not an unbiased estimate of $B_{\text{simple}}$; the ratio of the two averages over many steps converges to it.

### Assumptions

1. Each batch is a set of independent draws, so $\mathbb{E}\lVert G_B\rVert^2 = \lVert G\rVert^2 + \operatorname{tr}(\Sigma)/B$ (McCandlish et al., eq. A.1, p. 17). This holds for both batch sizes whatever the dependence between the two batches, which is why nesting does not matter.
2. $B_{\text{small}} \neq B_{\text{big}}$, so the denominators are nonzero.

### Steps

1. $\mathbb{E}\lVert G_B\rVert^2 = \lVert \mathbb{E} G_B\rVert^2 + \operatorname{tr}\operatorname{cov} G_B = \lVert G\rVert^2 + \operatorname{tr}(\Sigma)/B$.
2. Both estimators are linear in $\lVert G_{\text{small}}\rVert^2$ and $\lVert G_{\text{big}}\rVert^2$, so their means follow from step 1 alone. For $\hat S$, the $\lVert G\rVert^2$ terms cancel and $\operatorname{tr}(\Sigma)(1/B_{\text{small}} - 1/B_{\text{big}})$ remains. For $\widehat{\lVert G\rVert^2}$, the $\operatorname{tr}\Sigma$ terms cancel and $(B_{\text{big}} - B_{\text{small}})\lVert G\rVert^2$ remains. This is (i).
3. With $B_{\text{small}} = 1$, $B_{\text{big}} = n$ and the small-batch norm averaged over the examples, $\hat S = \frac{n}{n-1}\big(\frac1n\sum_i \lVert g_i\rVert^2 - \lVert\bar g\rVert^2\big) = \frac{1}{n-1}\sum_i \lVert g_i - \bar g\rVert^2$, the Bessel-corrected variance named in footnote 11 (p. 17). Substituting into the first estimator gives $\lVert\bar g\rVert^2 - \hat S/n$. This is (ii).
4. The mean of a ratio is not the ratio of means when the denominator varies, and here $\widehat{\lVert G\rVert^2}$ can come close to zero or go negative. Averaging numerator and denominator separately, as the paper does with moving averages, gives a ratio that converges to $\operatorname{tr}\Sigma / \lVert G\rVert^2$ by the law of large numbers. This is (iii).

### Result and check

The check draws per-example gradients from a skewed population of 40 examples in 5 dimensions, with $\lVert G\rVert^2 = 0.840$, $\operatorname{tr}\Sigma = 8.13$ and $B_{\text{simple}} = 9.67$ (our run).
For the pairs (1, 8), (4, 32) and (16, 32), with the small batch both nested in the big one and drawn apart from it, 400,000 trials put both means within 2.4 standard errors of the truth.
The per-example identity (ii) holds to rounding.
At $B_{\text{small}} = 8$, $B_{\text{big}} = 64$ the median of the per-step ratio is 6.60, while the ratio of the means is 9.60, within 1% of the true 9.67.
The paper's footnote 12 adds that $\mathbb{E}[x/y] \ge \mathbb{E}[x]/\mathbb{E}[y]$ for positive variables; that inequality follows from Jensen's inequality when $x$ and $y$ are independent, which these two estimates are not, so this page relies only on the weaker statement that the ratio is biased.
