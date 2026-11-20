---
id: iwae-bound
derives: [c-iwae, c-iwae-monotone, c-iwae-grad]
kind: identity
assumes: ["w_i = p(x, z_i) / q(z_i | x) for z_1, ..., z_k drawn independently from q(z|x)", "q(z|x) > 0 wherever p(x, z) > 0"]
uses: [linearity-of-expectation@m2, law-of-large-numbers@m2, evidence-lower-bound@al]
status: full
check: iwae-bound.check.py
lean: null
sources:
  - {work: burda-2015-importance-weighted-autoencoders, locator: "pp. 3 to 4", note: "eq. 8, Theorem 1 and its proof in appendix A, and the gradient with normalized importance weights"}
---

### Statement

1. $\mathcal{L}_k(x) = \mathbb{E}\big[\log \frac1k \sum_{i=1}^{k} w_i\big]$ is a lower bound on $\log p(x)$ for every $k$, and $\mathcal{L}_1$ is the ELBO.
2. $\mathcal{L}_k \le \mathcal{L}_{k+1} \le \log p(x)$, and if $p(x, z)/q(z \mid x)$ is bounded, $\mathcal{L}_k \to \log p(x)$ as $k \to \infty$.
3. With $z_i$ reparameterized as functions of independent noise, $\nabla \log \frac1k \sum_i w_i = \sum_i \tilde{w}_i \nabla \log w_i$ with the normalized weights $\tilde{w}_i = w_i / \sum_j w_j$: the gradient of the ELBO of each draw, weighted by how well that draw explains $x$. With $k = 1$ it is the gradient of the ELBO.

### Assumptions

1. The draws are independent and identically distributed under $q$. Used in steps 1 and 2.
2. $q$ covers the support of the posterior, so $\mathbb{E}_q[w] = p(x)$. Used in steps 1 and 2.

### Steps

1. $\mathbb{E}_q[w_i] = \int q\, p(x, z)/q\, dz = p(x)$, so $\mathbb{E}\big[\frac1k \sum_i w_i\big] = p(x)$; the logarithm is concave, so by Jensen's inequality $\mathcal{L}_k \le \log p(x)$. With $k = 1$, $\mathbb{E}_q[\log w] = \mathbb{E}_q[\log p(x, z) - \log q(z \mid x)]$, the ELBO. This is statement 1.
2. Let $I$ be a subset of $\{1, \dots, k+1\}$ of size $k$ drawn uniformly. Each index is in $I$ with probability $k/(k+1)$, so $\frac{1}{k+1}\sum_{i=1}^{k+1} w_i = \mathbb{E}_I\big[\frac1k \sum_{i \in I} w_i\big]$, and by Jensen's inequality its logarithm is at least $\mathbb{E}_I\big[\log \frac1k \sum_{i \in I} w_i\big]$. Taking the expectation over the draws, every subset has the distribution of $k$ fresh draws, so $\mathcal{L}_{k+1} \ge \mathcal{L}_k$. If the weights are bounded, $\frac1k \sum_i w_i \to p(x)$ almost surely by the strong law of large numbers and the logarithms are bounded below on that event, so the expectations converge. This is statement 2.
3. $\nabla \log \sum_i w_i = \sum_i \nabla w_i / \sum_j w_j = \sum_i (w_i / \sum_j w_j) \nabla \log w_i$. This is statement 3, eq. 14 of Burda et al.

### Result and check

$$ \mathcal{L}_1 = \mathrm{ELBO} \le \mathcal{L}_k \le \mathcal{L}_{k+1} \le \log p(x), \qquad \mathcal{L}_k = \mathbb{E}\Big[\log \frac1k \sum_{i=1}^{k} \frac{p(x, z_i)}{q(z_i \mid x)}\Big] . $$

The check `iwae-bound.check.py` computes $\mathcal{L}_k$ exactly, by enumerating every $k$-tuple of a latent variable with five values, for random models and encoders and $k = 1, \dots, 5$; it checks the order of statement 2, that $\mathcal{L}_k = \log p(x)$ for every $k$ when $q$ is the posterior, that a Monte Carlo $\mathcal{L}_{2000}$ is within 0.01 of $\log p(x)$, and that the gradient of the reference's `iwae` objective is the normalized-weight average of the gradients of its draws.
