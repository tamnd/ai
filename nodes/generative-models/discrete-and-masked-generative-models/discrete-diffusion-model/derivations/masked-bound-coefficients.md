---
id: masked-bound-coefficients
derives: [c-mask-coefficient]
kind: closed-form
assumes: ["alpha_t is differentiable and decreasing on [0, 1] with alpha_0 = 1 and alpha_1 = 0", "the denoiser does not depend on t", "the tokens are masked independently, each with probability 1 - alpha_t"]
uses: [independence-of-random-variables@m2, expected-value@m2, beta-distribution@m2, cross-entropy@m2]
status: full
check: masked-bound-coefficients.check.py
lean: null
sources:
  - {work: sahoo-2024-simple-effective-masked-diffusion-language, locator: "p. 5", note: "the continuous-time bound and its invariance to the schedule by the change of variables gamma = log(1 - alpha_t)"}
  - {work: shi-2024-simplified-generalized-masked-diffusion, locator: "p. 4", note: "the bound as a weighted integral of cross-entropies and its invariance through the log-SNR"}
---

### Statement

For a sequence $\mathbf{x}$ of length $L$ and a denoiser that does not depend on $t$, write $\mathbf{x}_M$ for $\mathbf{x}$ with the positions of $M$ masked and $\mathrm{CE}_M = \sum_{\ell \in M} -\log \langle \mathbf{x}_\theta^\ell(\mathbf{x}_M), \mathbf{x}^\ell \rangle$. Then eq-mdlm is

$$ \mathcal{L}_\infty = \sum_{M \ne \emptyset} c_{|M|}\, \mathrm{CE}_M , \qquad c_m = \frac{(L - m)!\,(m - 1)!}{L!} = B(L - m + 1, m) , $$

for every schedule that satisfies the assumptions. Summed over the $\binom{L}{m}$ sets of size $m$, the coefficients give $\binom{L}{m} c_m = 1/m$, so each mask size contributes its average cross-entropy per masked token with the same weight.

### Assumptions

1. $\alpha$ is differentiable and decreasing, from 1 to 0. Used in step 3.
2. The denoiser ignores $t$. Used in step 2.
3. Independent masking. Used in step 1.

### Steps

1. At time $t$ each token is masked independently with probability $1 - \alpha_t$, so the set of masked positions is exactly $M$ with probability $(1 - \alpha_t)^{|M|} \alpha_t^{L - |M|}$. Only masked positions contribute to the sum inside eq-mdlm, and with mask set $M$ that sum is $\mathrm{CE}_M$.
2. Because $\mathrm{CE}_M$ does not depend on $t$, the integral factors: $\mathcal{L}_\infty = \sum_M \mathrm{CE}_M \int_0^1 \frac{-\alpha_t'}{1 - \alpha_t} (1 - \alpha_t)^{m} \alpha_t^{L - m}\, dt$ with $m = |M|$, and the empty set contributes nothing.
3. The integrand is $-\alpha_t' (1 - \alpha_t)^{m - 1} \alpha_t^{L - m}$. Substituting $a = \alpha_t$, $da = \alpha_t'\, dt$, which maps $t = 0$ to $a = 1$ and $t = 1$ to $a = 0$, gives $\int_0^1 a^{L - m}(1 - a)^{m - 1}\, da$, the Beta function $B(L - m + 1, m)$. The schedule has disappeared.
4. $B(L - m + 1, m) = \Gamma(L - m + 1)\Gamma(m)/\Gamma(L + 1) = (L - m)!(m - 1)!/L!$.
5. $\binom{L}{m} c_m = \frac{L!}{m!(L - m)!} \cdot \frac{(L - m)!(m - 1)!}{L!} = \frac{(m - 1)!}{m!} = \frac{1}{m}$.

### Result and check

$$ \mathcal{L}_\infty = \sum_{m=1}^{L} \frac{1}{m} \cdot \frac{1}{\binom{L}{m}} \sum_{|M| = m} \mathrm{CE}_M . $$

A denoiser that sees $t$ breaks step 2, and the bound then depends on the schedule through the network, though the optimal bound still does not.
The check `masked-bound-coefficients.check.py` integrates $a^{L-m}(1-a)^{m-1}$ numerically for $L$ up to 8 and compares it with `mask_coefficient`, checks $\binom{L}{m} c_m = 1/m$, and evaluates eq-mdlm for a random network denoiser by Gauss-Legendre quadrature under the linear and cosine schedules with `nelbo_by_quadrature`, finding both equal to `exact_nelbo` to $10^{-10}$.
