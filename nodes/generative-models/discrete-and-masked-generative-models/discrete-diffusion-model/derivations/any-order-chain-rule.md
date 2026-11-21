---
id: any-order-chain-rule
derives: [c-any-order]
kind: proof
assumes: ["the assumptions of masked-bound-coefficients", "the denoiser at a masked position is the true conditional of that token given the visible tokens, the other masked tokens summed out", "p(x) > 0"]
uses: [joint-probability-distribution@m2, conditional-probability@m2, expected-value@m2]
status: full
check: any-order-chain-rule.check.py
lean: null
sources:
  - {work: sahoo-2024-simple-effective-masked-diffusion-language, locator: "p. 5", note: "the bound as a weighted average of masked language modelling losses"}
  - {work: austin-2021-structured-denoising-diffusion-discrete, locator: "pp. 5 to 6", note: "autoregressive models as diffusion models that reveal the tokens in one fixed order"}
---

### Statement

Let $\mathbf{x}_\theta^\ell(\mathbf{x}_M)$ be $p(x^\ell \mid \mathbf{x}^{M^c})$ for $\ell \in M$, where $M^c$ is the set of visible positions. Then for every $\mathbf{x}$ with $p(\mathbf{x}) > 0$,

$$ \sum_{M \ne \emptyset} c_{|M|} \sum_{\ell \in M} -\log p(x^\ell \mid \mathbf{x}^{M^c}) = -\log p(\mathbf{x}) , $$

with $c_m$ the coefficients of masked-bound-coefficients. The left side is the average over a uniformly random order $\sigma$ of the chain rule of probability along $\sigma$.

### Assumptions

1. The coefficients are those of masked-bound-coefficients. Used in step 3.
2. The denoiser is the true conditional. Used in step 2.
3. $p(\mathbf{x}) > 0$, so every conditional probability in the chain is positive. Used in step 1.

### Steps

1. For any order $\sigma$ of the $L$ positions, the chain rule of probability gives $-\log p(\mathbf{x}) = \sum_{k=1}^{L} -\log p(x^{\sigma_k} \mid \mathbf{x}^{\sigma_{<k}})$, where $\sigma_{<k}$ is the set of the first $k - 1$ positions. Averaging over a uniformly random $\sigma$ leaves the left side unchanged.
2. Each term is indexed by a position $\ell = \sigma_k$ and a set $V = \sigma_{<k}$ of positions revealed before it, with $\ell \notin V$. Set $M = V^c$, which contains $\ell$; the term is $-\log p(x^\ell \mid \mathbf{x}^{M^c})$, the cross-entropy of the true denoiser at $\ell$ when exactly the positions of $M$ are masked.
3. The probability that a uniformly random $\sigma$ puts the positions of $V$ first, in any order, and then $\ell$, is $|V|!\,(L - |V| - 1)!/L!$: $|V|!$ orders for $V$, $\ell$ next, and $(L - |V| - 1)!$ orders for the rest. With $m = |M| = L - |V|$ this is $(L - m)!(m - 1)!/L! = c_m$.
4. Collecting the terms of step 1 by $(M, \ell)$ with the weights of step 3 gives $\mathbb{E}_\sigma \sum_k -\log p(x^{\sigma_k} \mid \mathbf{x}^{\sigma_{<k}}) = \sum_{M} c_{|M|} \sum_{\ell \in M} -\log p(x^\ell \mid \mathbf{x}^{M^c})$, the left side of the statement.

### Result and check

The bound of eq-mdlm is tight for the exact denoiser: masked diffusion with the true conditionals is the uniform mixture over orders of autoregressive models, each of which is exact.
Any other denoiser gives a larger expected bound, since the bound is the expectation of cross-entropies that the true conditionals minimize.
The check `any-order-chain-rule.check.py` enumerates the $L!$ orders for $L \le 5$ and confirms that the probability of each pair $(M, \ell)$ is `mask_coefficient`; it evaluates `exact_nelbo` with `conditional_denoiser` for every sequence of the Markov data of the tests and of random distributions with $L \le 4$, finding $-\log p(\mathbf{x})$ to $10^{-12}$; and it finds the bound of a perturbed denoiser larger in expectation.
