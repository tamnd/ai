---
id: ml-gradient
derives: [c-ml-grad]
kind: gradient
assumes: ["the input space is finite, or the integrals defining Z and its derivative converge and can be differentiated under the integral sign", "E is differentiable in the parameters"]
uses: [expected-value@m2, exponential-family@m2, kullback-leibler-divergence@m2, maximum-likelihood-estimation@m2, partition-function@al]
status: full
check: ml-gradient.check.py
lean: null
sources:
  - {work: ackley-1985-learning-algorithm-boltzmann-machines, locator: "p. 7", note: "the derivative of the log probability of a global state in a weight, through the equilibrium co-occurrence p_ij"}
  - {work: hinton-2002-training-products-experts-contrastive-divergence, locator: "p. 4", note: "the two expectations of the log-likelihood gradient, over the data and over the equilibrium distribution"}
---

### Statement

For $p_\theta(\mathbf{x}) = e^{-E_\theta(\mathbf{x})} / Z(\theta)$,

$$ \nabla_\theta \log p_\theta(\mathbf{x}) = -\nabla_\theta E_\theta(\mathbf{x}) + \mathbb{E}_{\mathbf{x}' \sim p_\theta}\big[\nabla_\theta E_\theta(\mathbf{x}')\big] . $$

Averaged over a data distribution $q$, the gradient is $\mathbb{E}_{p_\theta}[\nabla_\theta E] - \mathbb{E}_{q}[\nabla_\theta E]$, which is minus the gradient of $\mathrm{KL}(q \,\|\, p_\theta)$. For an exponential family, $E_\theta(\mathbf{x}) = -\theta \cdot T(\mathbf{x})$, it vanishes exactly when $\mathbb{E}_{p_\theta}[T] = \mathbb{E}_q[T]$.

### Assumptions

1. $Z(\theta)$ is finite and can be differentiated term by term, or under the integral sign. Used in step 2.
2. $E_\theta$ is differentiable in $\theta$. Used in steps 1 and 2.

### Steps

1. $\log p_\theta(\mathbf{x}) = -E_\theta(\mathbf{x}) - \log Z(\theta)$, so $\nabla_\theta \log p_\theta(\mathbf{x}) = -\nabla_\theta E_\theta(\mathbf{x}) - \nabla_\theta \log Z(\theta)$.
2. $\nabla_\theta \log Z = \frac{1}{Z} \sum_{\mathbf{x}'} \nabla_\theta e^{-E_\theta(\mathbf{x}')} = -\sum_{\mathbf{x}'} \frac{e^{-E_\theta(\mathbf{x}')}}{Z} \nabla_\theta E_\theta(\mathbf{x}') = -\mathbb{E}_{p_\theta}[\nabla_\theta E_\theta]$.
3. Substituting step 2 into step 1 gives the statement. Taking the expectation over $\mathbf{x} \sim q$ gives the averaged form.
4. $\mathrm{KL}(q \,\|\, p_\theta) = \mathbb{E}_q[\log q] - \mathbb{E}_q[\log p_\theta]$, and the first term does not depend on $\theta$, so its gradient is minus the averaged gradient of step 3.
5. For $E_\theta = -\theta \cdot T$, $\nabla_\theta E_\theta = -T$, and the averaged gradient is $\mathbb{E}_q[T] - \mathbb{E}_{p_\theta}[T]$, zero when the moments match.

For an RBM the parameters enter through the free energy, $p(\mathbf{v}) = e^{-F(\mathbf{v})}/Z$, and $-\nabla_W F(\mathbf{v}) = \mathbf{v}^\top \sigma(\mathbf{c} + \mathbf{v} W)$, which is the statistic of the RBM update with the hidden units replaced by their conditional means.

### Result and check

$$ \mathbb{E}_q\big[\nabla_\theta \log p_\theta\big] = \mathbb{E}_{p_\theta}[\nabla_\theta E_\theta] - \mathbb{E}_q[\nabla_\theta E_\theta] = -\nabla_\theta \mathrm{KL}(q \,\|\, p_\theta) . $$

The check `ml-gradient.check.py` compares the exact gradient of `exact_grad` with central differences of the mean log-likelihood for random RBMs with up to 5 visible units, checks the KL form, and checks moment matching for a fully visible model fitted to a data distribution.
