---
id: high-temperature-limit
derives: [c-softmax-expansion, c-t2-factor, c-high-t-limit, c-logit-matching, c-kl-limit, c-limit-worked, eq-logit-matching]
kind: limit
assumes: ["the teacher logits v and the student logits z are fixed finite vectors of length N ≥ 2", "T > 0 grows while v and z stay fixed", "the zero-mean condition Σ_i z_i = Σ_i v_i = 0 only where stated"]
uses: [kullback-leibler-divergence@m2, mean-squared-error@m2]
status: full
check: high-temperature-limit.check.py
lean: null
sources:
  - {work: hinton-2015-distilling-knowledge, locator: "p. 3"}
  - {work: stanton-2021-does-knowledge-distillation-really-work, locator: "p. 3"}
---

### Statement

Keep the notation of `soft-target-gradient`: $p = \operatorname{softmax}(v/T)$, $q = \operatorname{softmax}(z/T)$, and $\partial C/\partial z_i = (q_i - p_i)/T$.
Write $\bar z = \frac1N\sum_i z_i$, $\bar v = \frac1N\sum_i v_i$, and $\Delta_i = (z_i - \bar z) - (v_i - \bar v)$ for the difference of the centred logits. Then, as $T \to \infty$ with $v$ and $z$ fixed,

1. $\operatorname{softmax}(x/T)_i = \frac1N + \frac{x_i - \bar x}{NT} + O(T^{-2})$ for any fixed $x$;
2. the unscaled gradient $\partial C/\partial z_i$ is $O(T^{-2})$, so it shrinks like $1/T^2$, and $T^2\,\partial C/\partial z_i$ stays of order 1;
3. $T^2\,\partial C/\partial z_i = \Delta_i/N + O(T^{-1})$;
4. if both sets of logits have zero mean, $\Delta_i = z_i - v_i$, and the limit $(z_i - v_i)/N$ is the gradient of $\frac{1}{2N}\sum_i (z_i - v_i)^2$; without that condition the limit is the gradient of $\frac{1}{2N}\sum_i \Delta_i^2$, which is zero exactly when $z - v$ is a constant vector;
5. $T^2 D_{\mathrm{KL}}(p \,\Vert\, q) = \frac{1}{2N}\sum_i \Delta_i^2 + O(T^{-1})$.

### Assumptions

1. $v$ and $z$ are fixed and finite while $T$ grows; the error terms depend on their size, through $\max_i |z_i|$ and $\max_i |v_i|$, and are small only once $T$ is large compared with the spread of the logits.
2. $N \ge 2$; with one class every softmax is 1 and every gradient 0.
3. Item 4 uses the zero-mean condition of Hinton et al. only for its first sentence; the second sentence needs no condition.

### Steps

1. Write $\varepsilon = 1/T$ and expand $e^{\varepsilon x_i} = 1 + \varepsilon x_i + O(\varepsilon^2)$. The sum is $N(1 + \varepsilon\bar x + O(\varepsilon^2))$, so $\operatorname{softmax}(\varepsilon x)_i = \frac1N(1 + \varepsilon x_i)(1 - \varepsilon\bar x) + O(\varepsilon^2) = \frac1N + \frac{\varepsilon(x_i - \bar x)}{N} + O(\varepsilon^2)$. This is item 1, and it is the approximation $e^x \approx 1 + x$ that Hinton et al. use.
2. Subtracting the expansions for $z$ and $v$, $q_i - p_i = \Delta_i/(NT) + O(T^{-2})$, and dividing by $T$, $\partial C/\partial z_i = \Delta_i/(NT^2) + O(T^{-3})$. This is $O(T^{-2})$, item 2, and with zero-mean logits it is equation 4 of Hinton et al., $\partial C/\partial z_i \approx (z_i - v_i)/(NT^2)$.
3. Multiplying by $T^2$ gives $T^2\,\partial C/\partial z_i = \Delta_i/N + O(T^{-1})$, item 3. The $O(T^{-1})$ comes from the second-order term of the expansion, of size $1/T^2$ in $q - p$, multiplied by $T$. The gradient of the hard-label cross-entropy does not depend on $T$, so the factor $T^2$ keeps the share of the two terms roughly the same when $T$ changes, as Hinton et al. recommend.
4. With $\bar z = \bar v = 0$, $\Delta_i = z_i - v_i$, and $\partial\big(\frac{1}{2N}\sum_k (z_k - v_k)^2\big)/\partial z_i = (z_i - v_i)/N$, the limit of step 3. In general, $\partial\Delta_k/\partial z_i = [k = i] - 1/N$ and $\sum_k \Delta_k = 0$, so $\partial\big(\frac{1}{2N}\sum_k \Delta_k^2\big)/\partial z_i = \frac1N\big(\Delta_i - \frac1N\sum_k\Delta_k\big) = \Delta_i/N$. This is zero exactly when every $\Delta_i = 0$, that is when $z$ and $v$ differ by a constant, which leaves the softmax unchanged.
5. For distributions close to each other, $D_{\mathrm{KL}}(p \,\Vert\, q) = \frac12\sum_i (p_i - q_i)^2/p_i + O(\lVert p - q\rVert^3)$. Here $p_i = 1/N + O(T^{-1})$ and $p_i - q_i = -\Delta_i/(NT) + O(T^{-2})$, so $D_{\mathrm{KL}} = \frac{N}{2}\sum_i \frac{\Delta_i^2}{N^2T^2} + O(T^{-3}) = \frac{1}{2NT^2}\sum_i\Delta_i^2 + O(T^{-3})$, and multiplying by $T^2$ gives item 5.

### Result and check

$$ \lim_{T \to \infty} T^2\,\frac{\partial C}{\partial z_i} = \frac{(z_i - \bar z) - (v_i - \bar v)}{N} \;\overset{\bar z = \bar v = 0}{=}\; \frac{z_i - v_i}{N} = \frac{\partial}{\partial z_i}\,\frac{1}{2N}\sum_k (z_k - v_k)^2 , \qquad \lim_{T\to\infty} T^2 D_{\mathrm{KL}}(p\,\Vert\,q) = \frac{1}{2N}\sum_i \Delta_i^2 . $$

The check `high-temperature-limit.check.py` draws 500 random pairs of logit vectors with $N$ from 2 to 30.
For each it evaluates $T^2(q - p)/T$ at $T = 10^3, 10^4, 10^5$ and $T^2 D_{\mathrm{KL}}$ at $T = 10^3, 10^4$, since at higher temperatures rounding swamps it, and checks that while the distance to the limits of items 3 and 5 is above $10^{-6}$ it falls by a factor of at least five for each tenfold $T$, as the $O(T^{-1})$ bound requires; for $N = 2$ the second-order term vanishes and the error falls faster.
It also checks the expansion of item 1 to second order at $T = 10^2$ and $10^3$.
It checks item 4 by finite differences of $\frac{1}{2N}\sum_k\Delta_k^2$ and of the zero-mean form, and checks that the limit vanishes for $z = v + c$.
On the worked example, $v = (3, 1, -4)$ and $z = (1, 0.5, -1.5)$, which already have the same mean $0$, the scaled gradient is $(-0.592381, 0.097268, 0.495113)$ at $T = 4$ and $(-0.667294, -0.155354, 0.822648)$ at $T = 100$, against the limit $(z - v)/3 = (-0.666667, -0.166667, 0.833333)$; $T^2 D_{\mathrm{KL}}$ is $1.095802$ at $T = 4$ and $1.732200$ at $T = 100$, against the limit $1.75$.
