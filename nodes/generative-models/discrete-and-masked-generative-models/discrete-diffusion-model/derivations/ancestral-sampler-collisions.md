---
id: ancestral-sampler-collisions
derives: [c-sampler-collisions]
kind: bound
assumes: ["the denoiser at a masked position is the true conditional of that token given the visible tokens", "the decision to reveal a token uses randomness independent of the token draws", "the sampler takes T steps at times t_i = i / T from t = 1, where every token is masked, to t = 0, where none is"]
uses: [joint-probability-distribution@m2, independence-of-random-variables@m2, conditional-probability@m2, total-variation-distance-between-probability-measures@m2, ancestral-sampling@al]
status: full
check: ancestral-sampler-collisions.check.py
lean: null
sources:
  - {work: sahoo-2024-simple-effective-masked-diffusion-language, locator: "pp. 4, 6", note: "the reverse step of eq-subs and ancestral sampling with caching"}
  - {work: shi-2024-simplified-generalized-masked-diffusion, locator: "p. 3", note: "the observation that a schedule that unmasks slowly at the start lowers the chance of conflicting tokens being unmasked simultaneously"}
---

### Statement

Let $\tau_\ell \in \{1, \dots, T\}$ be the step at which the sampler reveals position $\ell$, and let $C$ be the event that two positions share a step. With the exact denoiser the output distribution is

$$ p_T = (1 - P(C))\, p + P(C)\, q' $$

for some distribution $q'$, so $\mathrm{TV}(p_T, p) \le P(C)$. The $\tau_\ell$ are independent and identically distributed with $P(\tau_\ell = i) = \pi_i = \alpha_{t_{i-1}} - \alpha_{t_i}$, so $1 - P(C) = L!\, e_L(\pi_1, \dots, \pi_T)$, with $e_L$ the elementary symmetric polynomial of degree $L$. For the linear schedule $\pi_i = 1/T$ and

$$ P(C) = 1 - \frac{T!}{(T - L)!\, T^L} \quad (T \ge L) , \qquad P(C) = 1 \quad (T < L) . $$

With $T = 1$ every token is revealed at the one step and drawn from its marginal, so $p_1$ is the product of the marginals of $p$.

### Assumptions

1. The exact denoiser. Used in step 4.
2. Independent reveal decisions. Used in steps 2 and 3.
3. The sampler's time grid. Used in step 2.

### Steps

1. At the step from $t_i$ to $t_{i-1}$, a masked token is revealed with probability $r_i = (\alpha_{t_{i-1}} - \alpha_{t_i})/(1 - \alpha_{t_i})$, by eq-mask-posterior, using its own uniform draw.
2. A token is still masked after the steps from $t_T$ down to $t_{i-1}$ with probability $\prod_{j \ge i} (1 - r_j) = \prod_{j \ge i} \frac{1 - \alpha_{t_{j-1}}}{1 - \alpha_{t_j}} = \frac{1 - \alpha_{t_{i-1}}}{1 - \alpha_{t_T}} = 1 - \alpha_{t_{i-1}}$, since the product telescopes and $\alpha_{t_T} = \alpha_1 = 0$. The probability that it is revealed exactly at step $i$ is the difference $(1 - \alpha_{t_{i-1}}) - (1 - \alpha_{t_i}) = \pi_i$.
3. The reveal decisions of different positions use independent uniforms and do not depend on the token values, so $\tau = (\tau_1, \dots, \tau_L)$ is independent across positions and independent of the token draws.
4. Fix $\tau$ with distinct entries. Then the sampler reveals one position at a time, in the order given by $\tau$, and draws each from the exact conditional given the tokens already revealed; by the chain rule of probability the result has distribution $p$. Positions revealed in the same step are drawn independently of each other, so for $\tau$ with a repeated entry the conditional law is some other distribution.
5. Averaging over $\tau$ gives $p_T = P(C^c)\, p + P(C)\, q'$ with $q'$ the law given $C$. Then $p_T - p = P(C)(q' - p)$ and $\mathrm{TV}(p_T, p) = P(C)\,\mathrm{TV}(q', p) \le P(C)$.
6. $P(C^c)$ is the probability that $L$ independent draws from $\pi$ are distinct: summing $\prod_\ell \pi_{\tau_\ell}$ over injective assignments gives $L!\, e_L(\pi)$. For $\pi_i = 1/T$ there are $T!/(T - L)!$ injective assignments, each of probability $T^{-L}$.
7. With $T = 1$, $r_1 = 1$: all positions are revealed together, from the denoiser on the all-mask sequence, which gives each token its marginal.

### Result and check

$$ \mathrm{TV}(p_T, p) \le 1 - \frac{T!}{(T - L)!\, T^L} \le \frac{L(L - 1)}{2T} . $$

The last inequality is the union bound over the $\binom{L}{2}$ pairs, each colliding with probability $1/T$, so about $L^2/2$ steps are needed for a small bound; with a trained network the error of the denoiser adds to this.
The check `ancestral-sampler-collisions.check.py` computes the exact output distribution of the sampler with `sampler_distribution` for Markov data and random distributions with $L \le 4$, $T$ from 1 to 32 and the linear and cosine schedules; it verifies that $p_T - (1 - P(C))p$ is non-negative and sums to $P(C)$, that the total variation is at most $P(C)$, that $P(C)$ matches the closed form for the linear schedule and $L!\,e_L(\pi)$ for the cosine one, and that $p_1$ is the product of the marginals.
