---
id: tilted-gaussian
derives: [c-tilted-gaussian]
kind: closed-form
assumes: ["p_1 = N(m_1, v_1) and p_0 = N(m_0, v_0) in one coordinate, with v_1, v_0 > 0", "s is real"]
uses: [normal-distribution@m2, exponential-family@m2]
status: full
check: tilted-gaussian.check.py
lean: null
sources:
  - {work: ho-2022-classifier-free-diffusion-guidance, locator: "p. 4", note: "the guided density p(z | c) p(c | z)^w, equivalently p(z | c)^(w + 1) p(z)^(-w)"}
  - {work: dhariwal-2021-diffusion-models-beat-gans, locator: "p. 8", note: "s grad log p(y | x) = grad log of p(y | x)^s / Z, sharper than p(y | x) for s > 1"}
---

### Statement

For $p_1 = \mathcal{N}(m_1, v_1)$ and $p_0 = \mathcal{N}(m_0, v_0)$, the tilted density $p_1^s p_0^{1 - s}$ is integrable exactly when $\lambda_s = s/v_1 + (1 - s)/v_0 > 0$, and then

$$ \frac{p_1^s\, p_0^{1-s}}{\int p_1^s\, p_0^{1-s}} = \mathcal{N}(m_s, v_s) , \qquad \frac{1}{v_s} = \frac{s}{v_1} + \frac{1 - s}{v_0} , \qquad m_s = m_1 + (1 - s)\,\frac{v_s}{v_0}\,(m_0 - m_1) . $$

If $v_1 \le v_0$, as for a class inside a wider unconditional distribution, it is integrable for every $s \ge 0$, and for $s > 1$ its variance is below $v_1$ and its mean lies beyond $m_1$, on the side away from $m_0$. If $v_1 > v_0$ it fails to be integrable for every $s \ge v_1/(v_1 - v_0)$.

### Assumptions

1. Two one-dimensional Gaussians; in several dimensions with isotropic or diagonal covariances the statement holds per coordinate. Used in step 1.
2. A real $s$; it may exceed 1, where $1 - s < 0$ and $p_0$ is divided out. Used in steps 1 to 3.

### Steps

1. $\log(p_1^s p_0^{1-s}) = -\frac{s(x - m_1)^2}{2v_1} - \frac{(1 - s)(x - m_0)^2}{2v_0} + \text{const} = -\frac{\lambda_s}{2}x^2 + \Big(\frac{s m_1}{v_1} + \frac{(1-s)m_0}{v_0}\Big)x + \text{const}$.
2. The exponential of a quadratic $-\frac{\lambda}{2}x^2 + bx$ is integrable exactly when $\lambda > 0$, and is then proportional to $\mathcal{N}(b/\lambda, 1/\lambda)$, which gives $v_s$ and $m_s = v_s b$. Since $m_1 = v_s(s/v_1 + (1 - s)/v_0)\, m_1$, subtracting gives $m_s - m_1 = v_s (1 - s)(m_0 - m_1)/v_0$.
3. $\lambda_s = 1/v_0 + s(1/v_1 - 1/v_0)$. If $v_1 \le v_0$ the bracket is nonnegative and $\lambda_s \ge 1/v_0 > 0$ for $s \ge 0$; for $s > 1$, $\lambda_s - 1/v_1 = (s - 1)(1/v_1 - 1/v_0) \ge 0$, so $v_s \le v_1$, and $(1 - s) < 0$ puts $m_s$ beyond $m_1$. If $v_1 > v_0$ the bracket is negative and $\lambda_s \le 0$ exactly when $s \ge (1/v_0)/(1/v_0 - 1/v_1) = v_1/(v_1 - v_0)$.

### Result and check

The density that guidance targets at each noise level is a narrower Gaussian pushed away from the unconditional mean: this is the trade of diversity for fidelity in its simplest form.
For $m_1 = 1$, $v_1 = 0.25$, $m_0 = 0$, $v_0 = 1.25$ and $s = 2$ it is $\mathcal{N}(1.111, 0.1389)$; at $s = 3$ it is $\mathcal{N}(1.154, 0.0962)$.
If the unconditional model is the narrower one, as for a weak model guided by a stronger one, a large enough scale leaves no distribution at all.
The check `tilted-gaussian.check.py` compares `tilted_gaussian` with the normalized product on a fine grid, checks the two regimes and the threshold $v_1/(v_1 - v_0)$, and recomputes the numbers above.
