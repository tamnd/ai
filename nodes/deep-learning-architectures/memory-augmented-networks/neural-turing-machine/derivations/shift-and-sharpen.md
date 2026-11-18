---
id: shift-and-sharpen
derives: [c-shift-is-roll, c-gate-extremes, c-blur-spreads, c-sharpen-limits]
kind: identity
assumes: ["the shift weighting is a weighting over the shifts -S to S with 2S + 1 <= N", "gamma >= 1"]
uses: [addressing-keeps-a-weighting]
status: full
check: shift-and-sharpen.check.py
lean: null
sources:
  - {work: graves-2014-neural-turing-machines, locator: "Section 3.3.2"}
---

### Statement

1. A shift weighting with all its weight on the shift $d$ rotates the weighting by $d$ locations: $\tilde w(i) = w^g(i - d)$, indices modulo $N$.
2. With $g_t = 1$ the head ignores its weighting of the previous step and uses the content weighting; with $g_t = 0$ it ignores the content and moves its previous weighting by the shift alone.
3. Without sharpening, a shift weighting that is not one-hot spreads the weighting at every step. If $s$ has mean $\mu$ and variance $\sigma^2$ as a distribution on $-S, \dots, S$, then after $k$ steps of $g = 0$ a weighting that starts on one location has mean $k\mu$ and variance $k\sigma^2$ about it, as long as it does not wrap around the memory. For the shifts $-1, 0, 1$ with weights $0.1, 0.8, 0.1$, $\sigma^2 = 0.2$.
4. Sharpening with $\gamma = 1$ leaves the weighting as it is. For $\gamma > 1$ it keeps the order of the weights and moves weight towards the largest ones; as $\gamma \to \infty$ the weighting tends to the uniform weighting on the locations of the largest weight. A one-hot weighting stays one-hot.

### Assumptions

1. $s_t$ is a weighting over the shifts $-S, \dots, S$ and $2S + 1 \le N$, as in addressing-keeps-a-weighting. Used in steps 1 and 3.
2. $\gamma_t \ge 1$. Used in step 4.

### Steps

1. By eq-shift, $\tilde w(i) = \sum_j w^g(j)\, s(i - j)$. With $s(d) = 1$ and $s = 0$ elsewhere, only $j = i - d$ remains, so $\tilde w(i) = w^g(i - d)$.
2. By eq-interpolate, $w^g_t = g_t w^c_t + (1 - g_t) w_{t-1}$, which is $w^c_t$ at $g_t = 1$ and $w_{t-1}$ at $g_t = 0$; the shift and the sharpening then act on it.
3. Read a weighting with no wrap as the distribution of a location $L$ on the integers. By eq-shift with $g = 0$ the next weighting is the distribution of $L + D$, with $D \sim s$ independent of $L$: $\Pr(L + D = i) = \sum_j \Pr(L = j) s(i - j)$. After $k$ steps the location is $L_0 + D_1 + \dots + D_k$ with independent $D$'s, so the means and the variances add: mean $L_0 + k\mu$, variance $k \sigma^2$. For $s = (0.1, 0.8, 0.1)$, $\mu = 0$ and $\sigma^2 = 0.1 \cdot 1 + 0.1 \cdot 1 = 0.2$. Graves et al. name this dispersion of the weightings over time as the reason for the sharpening.
4. By eq-sharpen, $w(i) = \tilde w(i)^\gamma / \sum_j \tilde w(j)^\gamma$. At $\gamma = 1$ the denominator is 1. For $\gamma > 1$, $x \mapsto x^\gamma$ is increasing on $[0, \infty)$, so the order is kept, and the ratio $w(i)/w(j) = (\tilde w(i)/\tilde w(j))^\gamma$ grows with $\gamma$ when $\tilde w(i) > \tilde w(j)$. Divide by $\tilde w_{\max}^\gamma$: the terms with $\tilde w(j) < \tilde w_{\max}$ go to 0, the others stay 1, which gives the limit. A one-hot $\tilde w$ has one nonzero power, so $w = \tilde w$.

### Result and check

$$ \tilde w(i) = w^g(i - d) \text{ for } s = \delta_d, \qquad \operatorname{Var}_k = k \sigma^2, \qquad \lim_{\gamma \to \infty} w = \text{uniform on } \arg\max_i \tilde w(i). $$

The check `shift-and-sharpen.check.py` rotates weightings with every one-hot shift, runs the interpolation gate at 0 and 1, repeats the shift $(0.1, 0.8, 0.1)$ for up to 20 steps on 101 locations and compares the mean and the variance with $k\mu$ and $k\sigma^2$, then sharpens with $\gamma$ from 1 to 500.
