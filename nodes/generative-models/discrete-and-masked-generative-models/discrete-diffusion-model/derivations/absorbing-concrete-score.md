---
id: absorbing-concrete-score
derives: [c-absorbing-ratio]
kind: identity
assumes: ["each token is masked independently with probability 1 - alpha, alpha in (0, 1)", "the partly masked sequence has positive probability"]
uses: [independence-of-random-variables@m2, conditional-probability@m2]
status: full
check: absorbing-concrete-score.check.py
lean: null
sources:
  - {work: lou-2023-discrete-diffusion-ratios-data-distribution, locator: "p. 3", note: "the concrete score as the ratios p(y)/p(x) over neighbouring states"}
  - {work: sahoo-2024-simple-effective-masked-diffusion-language, locator: "p. 6", note: "the time-independent denoiser of MDLM against the time-conditioned network of SEDD"}
---

### Statement

Let $p_\alpha(\mathbf{z})$ be the marginal probability of a partly masked sequence $\mathbf{z}$ when $\mathbf{x} \sim p$ and each token is masked independently with probability $1 - \alpha$. Let $\mathbf{z}$ have its masked positions in $M$ and visible positions $V$, let $i \in M$, and let $\mathbf{z}^{i \to y}$ be $\mathbf{z}$ with position $i$ set to $y$. Then

$$ \frac{p_\alpha(\mathbf{z}^{i \to y})}{p_\alpha(\mathbf{z})} = \frac{\alpha}{1 - \alpha}\, p(x^i = y \mid \mathbf{x}^V = \mathbf{z}^V) , $$

and the ratios out of $\mathbf{z}$ at position $i$ sum to $\alpha/(1 - \alpha)$.

### Assumptions

1. Independent masking with $\alpha \in (0, 1)$. Used in step 1.
2. $p_\alpha(\mathbf{z}) > 0$. Used in step 3.

### Steps

1. A sequence $\mathbf{x}$ produces $\mathbf{z}$ exactly when it agrees with $\mathbf{z}$ on $V$ and the masking hides exactly the positions of $M$, which has probability $(1 - \alpha)^{|M|}\alpha^{L - |M|}$ whatever $\mathbf{x}$ is. So $p_\alpha(\mathbf{z}) = p(\mathbf{x}^V = \mathbf{z}^V)\,(1 - \alpha)^{|M|}\alpha^{L - |M|}$.
2. $\mathbf{z}^{i \to y}$ has visible set $V \cup \{i\}$ and masked set $M \setminus \{i\}$, so $p_\alpha(\mathbf{z}^{i \to y}) = p(\mathbf{x}^V = \mathbf{z}^V, x^i = y)\,(1 - \alpha)^{|M| - 1}\alpha^{L - |M| + 1}$.
3. Dividing, the masking probabilities leave $\alpha/(1 - \alpha)$ and the data probabilities leave $p(\mathbf{x}^V = \mathbf{z}^V, x^i = y)/p(\mathbf{x}^V = \mathbf{z}^V) = p(x^i = y \mid \mathbf{x}^V = \mathbf{z}^V)$.
4. Summing over $y$ gives $\alpha/(1 - \alpha)$, since the conditional probabilities sum to 1.

### Result and check

The concrete score of the masked process at a masked position is the exact denoiser times $\alpha_t/(1 - \alpha_t)$, a factor that depends only on time. A score network must represent that factor and so needs $t$ as an input; a network that predicts the denoiser does not, which is why MDLM and MD4 drop the time input and can cache, by c-mdlm-caching.
The check `absorbing-concrete-score.check.py` compares `masked_marginal` with the probability of each partly masked sequence computed by enumerating every sequence and every mask, then checks the ratio against `conditional_denoiser` for every partly masked sequence of a random distribution with $L = 3$ and $K = 3$ and of the Markov data, at several values of $\alpha$.
