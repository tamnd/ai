---
id: score-entropy-minimum
derives: [c-score-entropy-min]
kind: proof
assumes: ["every probability p(x), p0(x0) and kernel entry p(x | x0) is positive", "the weights w_xy are positive", "p(x) = sum over x0 of p0(x0) p(x | x0)"]
uses: [expected-value@m2, kullback-leibler-divergence@m2]
status: full
check: score-entropy-minimum.check.py
lean: null
sources:
  - {work: lou-2023-discrete-diffusion-ratios-data-distribution, locator: "p. 3, Definition 3.1, Theorem 3.4", note: "score entropy, its normalizing constant K(a) = a(log a - 1), and denoising score entropy"}
---

### Statement

For a positive $s$ and the true ratios $a_{xy} = p(y)/p(x)$, each term of eq-score-entropy satisfies

$$ s - a\log s + a(\log a - 1) = a\,(r - \log r - 1) , \qquad r = s/a , $$

which is non-negative and zero exactly at $s = a$. Hence $\mathcal{L}_{\text{SE}} \ge 0$, with equality exactly when $s_\theta(x)_y = p(y)/p(x)$ for every $x$ and $y \ne x$. The denoising score entropy

$$ \mathcal{L}_{\text{DSE}} = \sum_{x_0} p_0(x_0) \sum_x p(x \mid x_0) \sum_{y \ne x} w_{xy}\Big(s_{xy} - \frac{p(y \mid x_0)}{p(x \mid x_0)}\log s_{xy}\Big) $$

satisfies $\mathcal{L}_{\text{SE}} - \mathcal{L}_{\text{DSE}} = \sum_x p(x)\sum_{y \ne x} w_{xy} K(a_{xy})$, which does not depend on $s$, so the two have the same minimizer and the same gradient.

### Assumptions

1. Positive probabilities. Used in steps 1 and 3.
2. Positive weights. Used in step 2.
3. $p$ is the perturbation of $p_0$ by the kernel. Used in step 3.

### Steps

1. With $s = ar$, $s - a\log s + a\log a - a = ar - a\log a - a\log r + a\log a - a = a(r - \log r - 1)$.
2. $f(r) = r - \log r - 1$ has $f'(r) = 1 - 1/r$ and $f''(r) = 1/r^2 > 0$, so it is strictly convex with its minimum $f(1) = 0$. Each term is $w_{xy}\,a_{xy} f(r)$ with $w a > 0$, so it is non-negative and zero exactly at $r = 1$, and the sum weighted by $p(x) > 0$ is zero exactly when every term is.
3. In $\mathcal{L}_{\text{SE}}$, the part linear in $s$ is $\sum_x p(x) \sum_{y \ne x} w_{xy} s_{xy}$, and $p(x) = \sum_{x_0} p_0(x_0) p(x \mid x_0)$ makes it the same as the linear part of $\mathcal{L}_{\text{DSE}}$.
4. The part with $\log s$ is $-\sum_x \sum_{y \ne x} w_{xy}\, p(x)\frac{p(y)}{p(x)} \log s_{xy} = -\sum_x \sum_{y \ne x} w_{xy}\, p(y) \log s_{xy}$, and $p(y) = \sum_{x_0} p_0(x_0) p(x \mid x_0) \frac{p(y \mid x_0)}{p(x \mid x_0)}$, which is the $\log s$ part of $\mathcal{L}_{\text{DSE}}$.
5. What remains of $\mathcal{L}_{\text{SE}}$ is the constant $\sum_x p(x)\sum_{y \ne x} w_{xy} K(a_{xy})$.

### Result and check

$$ \arg\min_s \mathcal{L}_{\text{DSE}} = \arg\min_s \mathcal{L}_{\text{SE}} = \Big(\frac{p(y)}{p(x)}\Big)_{x, y} . $$

The denoising form needs only the ratios of the known kernel, which is what makes score entropy trainable.
The check `score-entropy-minimum.check.py` evaluates $f$ on a grid, checks the per-term identity, finds `score_entropy` zero at `concrete_score` and positive at random scores, finds $\mathcal{L}_{\text{SE}} - \mathcal{L}_{\text{DSE}}$ the same for many random $s$ and equal to the constant of step 5, and finds the finite-difference gradient of `denoising_score_entropy` zero at the true ratios.
