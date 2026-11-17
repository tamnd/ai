---
id: attention-scaling
derives: [c-attn-random, c-attn-aligned, c-attn-base]
kind: closed-form
assumes: ["query and key coordinates with zero mean and variance s^2, independent across coordinates", "for the aligned case, k = rho q + sqrt(1 - rho^2) z with z independent of q", "the head dimension d grows while alpha and the base dimension d0 are fixed"]
uses: []
status: full
check: attention-scaling.check.py
lean: null
sources:
  - {work: yang-2022-tensor-programs-v-hyperparameter-transfer, locator: "p. 6"}
  - {work: yang-2022-tensor-programs-v-hyperparameter-transfer, locator: "p. 24"}
  - {work: yang-2022-tensor-programs-v-hyperparameter-transfer, locator: "p. 47"}
---

### Statement

Let $q, k \in \mathbb{R}^d$ have coordinates of variance $s^2$.

1. If $q$ and $k$ are independent, $q^\top k$ has mean 0 and standard deviation $\sqrt{d}\, s^2$, so $q^\top k / \sqrt{d}$ stays $\Theta(1)$ and $q^\top k / d \to 0$.
2. If $k = \rho q + \sqrt{1 - \rho^2}\, z$ with $\rho \ne 0$, then $q^\top k / d \to \rho s^2$ and $q^\top k / \sqrt{d} = \Theta(\sqrt{d})$ grows without bound.
3. The $\mu$P logit scale $\alpha \sqrt{d_0} / d$ (eq-attn) equals the SP scale $\alpha / \sqrt{d}$ at $d = d_0$, and is $\sqrt{d_0/d}$ times it elsewhere.

### Assumptions

1. Coordinates are independent across $j$ with zero mean, variance $s^2$ and finite fourth moment.
2. In the aligned case, $z$ has the same law as $q$ and is independent of it, so $k$ again has coordinates of variance $s^2$.
3. $\alpha$ and $d_0$ are constants; only $d$ grows.

### Steps

1. Independent case. $q^\top k = \sum_{j=1}^d q_j k_j$ is a sum of $d$ independent terms with mean $\mathbb{E} q_j\, \mathbb{E} k_j = 0$ and variance $\mathbb{E} q_j^2\, \mathbb{E} k_j^2 = s^4$. Its variance is $d s^4$, so by the central limit theorem $q^\top k / \sqrt{d} \to \mathcal{N}(0, s^4)$ and $q^\top k / d$ has standard deviation $s^2 / \sqrt{d} \to 0$. This is the initialization picture that motivates $1/\sqrt{d}$.
2. Aligned case. $q_j k_j = \rho q_j^2 + \sqrt{1 - \rho^2}\, q_j z_j$ has mean $\rho s^2 \ne 0$. By the law of large numbers $q^\top k / d \to \rho s^2$, so $q^\top k = \Theta(d)$ and $q^\top k / \sqrt{d} = \Theta(\sqrt{d})$.
3. Training correlates $q$ and $k$: after an update, $q$ and $k$ are generally correlated (Tensor Programs V, Definition 4.1 and footnote 8, p. 6). So the contraction is $\Theta(d)$ by the law of large numbers, and the scale that keeps logits $\Theta(1)$ is $1/d$ (p. 47).
4. With the base-width constant of Appendix B.1 (p. 24), the scale is $\alpha \sqrt{d_0}/d$. At $d = d_0$ it is $\alpha / \sqrt{d_0}$, the SP value, and the ratio of the two scales is $(\alpha \sqrt{d_0}/d)/(\alpha/\sqrt{d}) = \sqrt{d_0/d}$.

### Result and check

The check draws 400 independent pairs per dimension from 16 to 4096 with $s = 1.5$. It confirms that the standard deviation of $q^\top k/\sqrt{d}$ stays near $s^2$ and that $q^\top k/d$ falls with slope below $-0.4$. With $\rho = 0.6$ it confirms that $q^\top k/d$ approaches $\rho s^2$ to within 5% and that $q^\top k/\sqrt{d}$ grows with log-log slope $0.5 \pm 0.05$. It also checks the base-dimension identity and the factor $1/2$ at $d = 4 d_0$.
