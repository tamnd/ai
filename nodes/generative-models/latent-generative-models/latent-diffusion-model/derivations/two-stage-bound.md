---
id: two-stage-bound
derives: [c-two-stage-bound]
kind: bound
assumes: ["q(z_0 | x) has a density with finite entropy", "the prior family contains q(z_0)"]
uses: [kullback-leibler-divergence@m2, cross-entropy@m2, differential-entropy@m2, evidence-lower-bound@al]
status: full
check: two-stage-bound.check.py
lean: null
sources:
  - {work: vahdat-2021-score-based-generative-modeling-latent, locator: "pp. 3 to 4, Eqs. 5 to 6", note: "the KL term split into the negative encoder entropy and the cross entropy to the score-based prior"}
  - {work: kingma-2013-auto-encoding-variational-bayes, locator: "p. 3", note: "the variational lower bound with a reconstruction and a KL term"}
---

### Statement

For an encoder $q(z_0 \mid x)$, a decoder $p(x \mid z_0)$ and a prior $p_\theta(z_0)$,

$$ -\log p(x) \le \mathbb{E}_{q}[-\log p(x \mid z_0)] - H\big(q(z_0 \mid x)\big) + \mathrm{CE}\big(q(z_0 \mid x) \,\|\, p_\theta\big) . $$

Averaged over the data $x \sim p_{\text{data}}$, the cross-entropy term equals $\mathrm{CE}(q(z_0) \,\|\, p_\theta)$ with the aggregate posterior $q(z_0) = \mathbb{E}_x\, q(z_0 \mid x)$, and

$$ \mathrm{CE}(q(z_0) \,\|\, p_\theta) = H(q(z_0)) + \mathrm{KL}(q(z_0) \,\|\, p_\theta) \ge H(q(z_0)) , $$

with equality exactly when $p_\theta = q(z_0)$.

### Assumptions

1. $q(z_0 \mid x)$ has a density with finite entropy, so the terms are finite. Used in step 2.
2. The prior family contains $q(z_0)$. Used in step 4.

### Steps

1. The variational bound is $-\log p(x) \le \mathbb{E}_q[-\log p(x \mid z_0)] + \mathrm{KL}(q(z_0 \mid x) \,\|\, p_\theta(z_0))$, by Jensen's inequality on $\log \mathbb{E}_q[p(x \mid z_0)p_\theta(z_0)/q(z_0 \mid x)]$.
2. $\mathrm{KL}(q \,\|\, p) = \mathbb{E}_q[\log q] - \mathbb{E}_q[\log p] = -H(q) + \mathrm{CE}(q \,\|\, p)$, which gives the first display.
3. $\mathbb{E}_x\, \mathrm{CE}(q(z_0 \mid x) \,\|\, p_\theta) = -\mathbb{E}_x \int q(z_0 \mid x) \log p_\theta(z_0)\,dz_0 = -\int q(z_0) \log p_\theta(z_0)\,dz_0$, by exchanging the expectation and the integral; this is $\mathrm{CE}(q(z_0) \,\|\, p_\theta)$.
4. Step 2 applied to $q(z_0)$ gives $H(q(z_0)) + \mathrm{KL}(q(z_0) \,\|\, p_\theta)$; the KL divergence is non-negative and zero only at $p_\theta = q(z_0)$, which the family contains.

### Result and check

Only the cross entropy involves the prior, and for a fixed encoder the best prior is the distribution of encoded data. So LDM's two stages minimize this bound coordinate-wise: the first stage trains the encoder and decoder with a nearly free prior term, and the second fits a diffusion model to $q(z_0)$ with the encoder frozen, the minimizer of the cross entropy. LSGM minimizes the same bound jointly.
The check `two-stage-bound.check.py` takes $x \sim \mathcal{N}(0, 1)$ and the encoder $q(z_0 \mid x) = \mathcal{N}(a x, s^2)$, compares the Gaussian closed forms of the entropy, the cross entropy and the KL divergence with Monte Carlo estimates and the identity of step 2, checks that the average cross entropy equals that of the aggregate $\mathcal{N}(0, a^2 + s^2)$, and that over a grid of Gaussian priors it is smallest at the aggregate, where it equals the aggregate's entropy. It also checks the decomposition for a two-component Gaussian mixture prior by quadrature.
