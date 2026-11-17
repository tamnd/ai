---
id: power-iteration
derives: [c-power-converges, c-power-underestimates, eq-power, eq-power-bound]
kind: bound
assumes: ["the start u_0 has a non-zero component along the top left singular vector", "the bound is useful only when sigma_1 > sigma_2"]
uses: [singular-value-decomposition@m2, rayleigh-quotient@m2, powers-of-a-diagonalisable-matrix@m2]
status: full
check: power-iteration.check.py
lean: null
sources:
  - {work: miyato-2018-spectral-normalization, locator: "p. 15"}
---

### Statement

Let $W \in \mathbb{R}^{d_o \times d_i}$ have singular values $\sigma_1 \geq \sigma_2 \geq \dots \geq \sigma_r > 0$, left singular vectors $p_i$ and right singular vectors $q_i$. Start from $u_0$ and repeat, for $k = 1, 2, \dots$:

$$ v_k = \frac{W^\top u_{k-1}}{\lVert W^\top u_{k-1}\rVert}, \qquad u_k = \frac{W v_k}{\lVert W v_k\rVert}, \qquad \hat\sigma_k = u_k^\top W v_k . $$

Let $\theta_0$ be the angle between $u_0$ and $p_1$. If $\cos\theta_0 \neq 0$, then

$$ 0 \;\leq\; \sigma_1^2 - \hat\sigma_k^2 \;\leq\; \sigma_1^2 \left(\frac{\sigma_2}{\sigma_1}\right)^{4k-2} \tan^2\theta_0 . $$

So $\hat\sigma_k \to \sigma_1$ when $\sigma_1 > \sigma_2$, and $\hat\sigma_k$ never exceeds $\sigma_1$.

### Assumptions

1. $a_1 = p_1\cdot u_0 \neq 0$. Then $W^\top u_{k-1}$ and $W v_k$ are never zero, so every step is defined. Used in step 1 and step 4.
2. $\sigma_1 > \sigma_2$ makes the bound go to 0. The bound holds without it but is then no help.

### Steps

1. Write $u_0 = \sum_i a_i p_i + z$, where $z$ is orthogonal to every $p_i$, so $W^\top z = 0$. Since $W^\top p_i = \sigma_i q_i$ and $W q_i = \sigma_i p_i$, each round multiplies the coefficient on $q_i$ or $p_i$ by $\sigma_i$. After the normalizations, $v_k \propto \sum_i a_i\sigma_i^{2k-1}q_i$.
2. $u_k = W v_k/\lVert W v_k\rVert$, so $\hat\sigma_k = u_k^\top W v_k = \lVert W v_k\rVert$ and $\hat\sigma_k^2 = v_k^\top W^\top W v_k$. This is the Rayleigh quotient of $W^\top W$ at $v_k$. Its largest value is the top eigenvalue $\sigma_1^2$, which gives the left inequality.
3. With $c_i = a_i\sigma_i^{2k-1}$, the Rayleigh quotient is $\hat\sigma_k^2 = \sum_i c_i^2\sigma_i^2 / \sum_i c_i^2$. So
   $$ \sigma_1^2 - \hat\sigma_k^2 = \frac{\sum_{i\geq 2} c_i^2(\sigma_1^2 - \sigma_i^2)}{\sum_i c_i^2} \leq \sigma_1^2\,\frac{\sum_{i\geq 2} a_i^2\sigma_i^{4k-2}}{a_1^2\sigma_1^{4k-2}} \leq \sigma_1^2\left(\frac{\sigma_2}{\sigma_1}\right)^{4k-2}\frac{\sum_{i\geq 2}a_i^2}{a_1^2} . $$
   The first inequality drops $\sigma_i^2 \geq 0$ from the numerator and every term but $i = 1$ from the denominator. The second uses $\sigma_i \leq \sigma_2$ for $i \geq 2$.
4. $\tan^2\theta_0 = (\lVert u_0\rVert^2 - a_1^2)/a_1^2 = (\sum_{i\geq 2} a_i^2 + \lVert z\rVert^2)/a_1^2 \geq \sum_{i\geq 2}a_i^2/a_1^2$. This gives the right inequality.

### Result and check

$$ \sigma_1^2 - \hat\sigma_k^2 \leq \sigma_1^2 \left(\frac{\sigma_2}{\sigma_1}\right)^{4k-2}\tan^2\theta_0 . $$

The error falls geometrically, and the start enters only through $\tan^2\theta_0$. This is why reusing $u$ from the previous update helps: when $W$ changes little in one step, the old $u$ is close to the new $p_1$ and $\theta_0$ is small. Since $\hat\sigma_k \leq \sigma_1$, $W/\hat\sigma_k$ has spectral norm $\sigma_1/\hat\sigma_k \geq 1$. It is slightly above 1 until the iteration has converged.
For $W = \mathrm{diag}(3, 1)$ and $u_0 = (1,1)/\sqrt{2}$, step 3 is exact: $9 - \hat\sigma_k^2 = 8/(9^{2k-1}+1)$, so $\hat\sigma_1^2 = 8.2$ and $\hat\sigma_2^2 = 6562/730$.
The check `power-iteration.check.py` builds 80 matrices of sizes 3×3 to 16×16 with known singular values. It checks both inequalities for $k = 1, 2, 3, 5, 10, 20$ and that 400 rounds match `numpy.linalg.svd` to a relative $10^{-9}$. It also checks the worked example for $k = 1$ to 4.
