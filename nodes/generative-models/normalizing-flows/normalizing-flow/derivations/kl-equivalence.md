---
id: kl-equivalence
derives: [c-mle-kl, c-kl-base]
kind: closed-form
assumes: ["f is a diffeomorphism of R^D", "the data distribution pi_x has a density, and the expectations below are finite"]
uses: [kullback-leibler-divergence@m2, differential-entropy@m2, maximum-likelihood-estimation@al]
status: full
check: kl-equivalence.check.py
lean: null
sources:
  - {work: papamakarios-2021-normalizing-flows-probabilistic-modeling, locator: "p. 6", note: "eqs. (13) and (14): the forward KL divergence and maximum likelihood"}
  - {work: papamakarios-2021-normalizing-flows-probabilistic-modeling, locator: "p. 8", note: "eq. (21): the forward KL divergence in data space equals a KL divergence in the base space"}
  - {work: papamakarios-2017-masked-autoregressive-flow, locator: "p. 5", note: "eq. (9) and the reading of maximum-likelihood training of MAF as variational inference with an implicit IAF"}
---

### Statement

Let the flow $f$ map data to the base space and define the model $p_x(\mathbf{x}) = p_z(f(\mathbf{x}))\, |\det J_f(\mathbf{x})|$.
Let $\pi_x$ be the data distribution and $\pi_z$ the distribution of $f(\mathbf{x})$ for $\mathbf{x} \sim \pi_x$.

1. The expected negative log-likelihood is $-\mathbb{E}_{\pi_x} \log p_x(\mathbf{x}) = H(\pi_x) + D_{\mathrm{KL}}(\pi_x \,\|\, p_x)$, so maximum likelihood minimizes the forward KL divergence, and the entropy $H(\pi_x)$ is the floor of the loss.
2. $D_{\mathrm{KL}}(\pi_x \,\|\, p_x) = D_{\mathrm{KL}}(\pi_z \,\|\, p_z)$: fitting the model to the data is fitting the pushed-forward data to the base density.

### Assumptions

1. $f$ is a diffeomorphism, so the change of variables of [[#eq-change-of-variables]] holds for $p_x$ and for $\pi_z$. Used in step 2.
2. $\pi_x$ has a density with finite entropy. Used in step 1.

### Steps

1. $D_{\mathrm{KL}}(\pi_x \,\|\, p_x) = \mathbb{E}_{\pi_x}[\log \pi_x(\mathbf{x}) - \log p_x(\mathbf{x})] = -H(\pi_x) - \mathbb{E}_{\pi_x} \log p_x(\mathbf{x})$. Rearranged, this is statement 1; $H(\pi_x)$ does not depend on the parameters, and the KL divergence is non-negative and zero only when $p_x = \pi_x$.
2. By the change of variables, $\pi_z(f(\mathbf{x})) = \pi_x(\mathbf{x})\, |\det J_f(\mathbf{x})|^{-1}$ and $p_z(f(\mathbf{x})) = p_x(\mathbf{x})\, |\det J_f(\mathbf{x})|^{-1}$. In the ratio the determinants cancel: $\log \pi_z(f(\mathbf{x})) - \log p_z(f(\mathbf{x})) = \log \pi_x(\mathbf{x}) - \log p_x(\mathbf{x})$ at every $\mathbf{x}$.
3. Taking the expectation over $\mathbf{x} \sim \pi_x$, which is the expectation over $\mathbf{z} = f(\mathbf{x}) \sim \pi_z$, gives statement 2. MAF writes the same identity with the names swapped, as $D_{\mathrm{KL}}(\pi_x \,\|\, p_x) = D_{\mathrm{KL}}(p_u \,\|\, \pi_u)$, where its $p_u$ is the density of the pushed-forward data and $\pi_u$ the base.

### Result and check

$$ -\mathbb{E}_{\pi_x} \log p_x = H(\pi_x) + D_{\mathrm{KL}}(\pi_x \,\|\, p_x), \qquad D_{\mathrm{KL}}(\pi_x \,\|\, p_x) = D_{\mathrm{KL}}(\pi_z \,\|\, p_z) . $$

The check `kl-equivalence.check.py` verifies statement 2 in closed form for an affine flow in one dimension, where both sides are KL divergences between Gaussians, and verifies the pointwise identity of step 2 for a random coupling flow on samples of a Gaussian mixture whose density is known; for statement 1 it compares the expected negative log-likelihood of a Gaussian model of Gaussian data with the entropy plus the closed-form KL divergence.
