---
id: addressing-keeps-a-weighting
derives: [c-weighting-stays-normalized, c-content-beta-limits]
kind: identity
assumes: ["the weighting of the previous step is nonnegative and sums to 1", "the shift weighting s is nonnegative and sums to 1 over the shifts -S to S, with 2S + 1 <= N", "beta >= 0, 0 <= g <= 1, gamma >= 1"]
uses: []
status: full
check: addressing-keeps-a-weighting.check.py
lean: null
sources:
  - {work: graves-2014-neural-turing-machines, locator: "Section 3.3"}
---

### Statement

1. Each of the four stages of the addressing maps weightings to weightings, so the output $w_t$ of a head is nonnegative and sums to 1 over the $N$ locations, whatever the memory and the outputs of the controller.
2. With $\beta = 0$ the content weighting is uniform, $w^c_t(i) = 1/N$. As $\beta \to \infty$ it puts all its weight, in equal parts, on the rows with the largest cosine similarity to the key.

### Assumptions

1. The weighting $w_{t-1}$ of the previous step is a weighting: $w_{t-1}(i) \ge 0$ and $\sum_i w_{t-1}(i) = 1$. It holds at $t = 0$ for the one-hot start, and the statement passes it on. Used in step 2.
2. $s_t$ is a weighting over the shifts $-S, \dots, S$, and $2S + 1 \le N$, so distinct shifts land on distinct locations modulo $N$. Used in step 3.
3. $\beta_t \ge 0$, $g_t \in [0, 1]$ and $\gamma_t \ge 1$, as the softplus, the sigmoid and $1 + $ softplus give them. Used in steps 1, 2 and 4.

### Steps

1. Content (eq-content): $w^c_t(i) = e^{\beta_t K(k_t, M_t(i))} / \sum_j e^{\beta_t K(k_t, M_t(j))}$. Each term is positive and the terms sum to 1. With $\beta_t = 0$ every exponential is 1, so $w^c_t(i) = 1/N$. For $\beta_t \to \infty$, divide numerator and denominator by $e^{\beta_t K^*}$ with $K^* = \max_j K(k_t, M_t(j))$: the terms with $K < K^*$ go to 0, those with $K = K^*$ stay 1, so the limit is uniform on the maximizers.
2. Interpolation (eq-interpolate): $w^g_t = g_t w^c_t + (1 - g_t) w_{t-1}$ is a convex combination of two weightings, so it is nonnegative and sums to $g_t + (1 - g_t) = 1$.
3. Shift (eq-shift): $\tilde w_t(i) = \sum_{j} w^g_t(j)\, s_t(i - j)$, indices modulo $N$, with $s_t(d) = 0$ for the shifts outside $-S, \dots, S$. It is a sum of products of nonnegative numbers. Summing over $i$ and swapping the sums, $\sum_i \tilde w_t(i) = \sum_j w^g_t(j) \sum_i s_t(i - j) = \sum_j w^g_t(j) \cdot 1 = 1$, since for fixed $j$ the map $i \mapsto i - j$ is a bijection of $\mathbb{Z}_N$ and, by assumption 2, each shift is hit once.
4. Sharpening (eq-sharpen): $w_t(i) = \tilde w_t(i)^{\gamma_t} / \sum_j \tilde w_t(j)^{\gamma_t}$. The powers are nonnegative and not all 0, since $\tilde w_t$ sums to 1, so the quotient is defined, nonnegative and sums to 1.

### Result and check

$$ w_t(i) \ge 0, \qquad \sum_{i=1}^{N} w_t(i) = 1, \qquad \lim_{\beta \to \infty} w^c_t = \text{uniform on } \arg\max_i K(k_t, M_t(i)). $$

The check `addressing-keeps-a-weighting.check.py` runs each stage of the NumPy reference and the whole head on random memories, keys and controller outputs, then checks $\beta = 0$, a large $\beta$ with a tie between two rows, and $2S + 1 = N$.
