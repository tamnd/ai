---
id: scale
derives: [c-dot-variance, c-softmax-saturation, eq-scores]
kind: bound
assumes: ["the 2 d_k entries of a query and a key are independent, with mean 0 and variance 1", "a row has n >= 2 scores and one largest score that exceeds every other score of the row by at least Delta > 0"]
uses: [dot-product@m2, variance@m2, expectation-of-a-product-of-independent-random-variables@m2, variance-of-a-sum-of-independent-random-variables@m2, variance-under-scaling@m2]
status: full
check: scale.check.py
lean: null
sources:
  - {work: vaswani-2017-attention, locator: "Section 3.2.1"}
  - {work: vaswani-2017-attention, locator: "p. 4"}
---

### Statement

Let $\mathbf{q}, \mathbf{k} \in \mathbb{R}^{d_k}$ have $2 d_k$ independent entries, each with mean 0 and variance 1. Then

1. $\mathbf{q}^\top \mathbf{k}$ has mean 0 and variance $d_k$, and $\mathbf{q}^\top \mathbf{k} / \sqrt{d_k}$ has mean 0 and variance 1.

Let $\mathbf{z} \in \mathbb{R}^n$, $n \ge 2$, be one row of scores, $\mathbf{a} = \mathrm{softmax}(\mathbf{z})$, and $J = \partial \mathbf{a} / \partial \mathbf{z} = \mathrm{diag}(\mathbf{a}) - \mathbf{a}\mathbf{a}^\top$.
Suppose $z_m \ge z_j + \Delta$ for one index $m$, every $j \neq m$, and some $\Delta > 0$. Then

2. $a_j \le e^{-\Delta}$ for every $j \neq m$, $1 - a_m \le (n - 1) e^{-\Delta}$, and every entry of $J$ has absolute value at most $(n - 1) e^{-\Delta}$.

### Assumptions

1. The entries of $\mathbf{q}$ and $\mathbf{k}$ are independent, with mean 0 and variance 1. Used in step 1. Learned queries and keys need not satisfy it; Vaswani et al. give it as an illustration of why the dot products grow.
2. One score of the row is the largest by a margin $\Delta > 0$. Used in steps 3 and 4.

### Steps

1. Each term $q_c k_c$ is a product of two independent variables, so $\mathbb{E}[q_c k_c] = \mathbb{E}[q_c]\,\mathbb{E}[k_c] = 0$ and $\mathrm{Var}(q_c k_c) = \mathbb{E}[q_c^2]\,\mathbb{E}[k_c^2] - 0 = 1$. The $d_k$ terms are independent, because they use disjoint entries, so the variance of their sum is the sum of their variances, $d_k$. Dividing by $\sqrt{d_k}$ divides the variance by $d_k$. This is item 1.
2. The softmax is $a_i = e^{z_i} / \sum_j e^{z_j}$. Its derivative is $\partial a_i / \partial z_j = a_i(\delta_{ij} - a_j)$, which is $J = \mathrm{diag}(\mathbf{a}) - \mathbf{a}\mathbf{a}^\top$.
3. For $j \neq m$, $a_j = a_m e^{z_j - z_m} \le a_m e^{-\Delta} \le e^{-\Delta}$. Summing over the $n - 1$ indices $j \neq m$ gives $1 - a_m = \sum_{j \neq m} a_j \le (n - 1) e^{-\Delta}$.
4. The entries of $J$ are of three kinds. The diagonal entry at $m$ is $a_m (1 - a_m) \le 1 - a_m$. A diagonal entry at $j \neq m$ is $a_j(1 - a_j) \le a_j \le e^{-\Delta}$. An entry off the diagonal is $-a_i a_j$, and at least one of $i, j$ differs from $m$, so $|a_i a_j| \le \min(a_i, a_j) \le e^{-\Delta}$. With $n \ge 2$, $e^{-\Delta} \le (n - 1) e^{-\Delta}$, which gives item 2.
5. Together: without the scale, under assumption 1, the scores of a row have standard deviation $\sqrt{d_k}$, so the gaps between them grow with $d_k$, and by item 2 the gradient through the softmax shrinks exponentially in the gap. With the scale, the standard deviation is 1 for every $d_k$.

### Result and check

$$ \mathrm{Var}\Big(\frac{\mathbf{q}^\top\mathbf{k}}{\sqrt{d_k}}\Big) = 1, \qquad \max_{i,j} \Big| \frac{\partial a_i}{\partial z_j} \Big| \le (n - 1)\, e^{-\Delta} . $$

The check `scale.check.py` draws $2 \cdot 10^5$ pairs of standard normal and of Rademacher vectors for $d_k \in \{1, 4, 64, 512\}$ with a fixed seed, and checks the sample mean and variance of the scaled dot product against 0 and 1 within 3%.
It compares $J$ with central finite differences, and checks item 2 on random rows with $n$ from 2 to 50 and $\Delta$ from 0.1 to 20, including the case where one score is far above the others.
