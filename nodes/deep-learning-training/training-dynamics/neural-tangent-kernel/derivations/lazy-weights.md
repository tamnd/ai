---
id: lazy-weights
derives: [c-lazy-bound, c-lazy-relative]
kind: bound
assumes: ["the two-layer ReLU network of relu-ntk.md in the NTK parametrization", "the squared loss (1/(2n)) sum_i (f(x_i) - y_i)^2 on n fixed points with a positive definite limit kernel", "gradient flow, or gradient descent with eta <= n / lambda_max", "the linearized model; for the network itself, the linearization bound of Lee et al."]
uses: []
status: full
check: lazy-weights.check.py
lean: null
sources:
  - {work: lee-2019-wide-networks-linear-models, locator: "Section 2.2"}
  - {work: lee-2019-wide-networks-linear-models, locator: "Section 2.4"}
  - {work: chizat-2019-lazy-training, locator: "p. 3"}
---

### Statement

Take the setting of [relu-ntk.md](relu-ntk.md) and [kernel-flow.md](kernel-flow.md): width $m$, input dimension $d$, $n$ training points, $J$ the Jacobian at $\theta_0$, $\hat\Theta = J J^\top$ with smallest eigenvalue $\lambda_{\min} > 0$, and $r_0 = f(X; \theta_0) - y$.
Let $\omega_k = \theta_k - \theta_0$ along gradient descent on the linearized model with $0 < \eta \le n / \lambda_{\max}$.

1. For every $k$, $\lVert \omega_k \rVert^2 \le r_0^\top \hat\Theta^{-1} r_0 \le \lVert r_0 \rVert^2 / \lambda_{\min}$, and $\lVert \omega_k \rVert^2 \to r_0^\top \hat\Theta^{-1} r_0$. The same bound holds along gradient flow.
2. $\mathbb{E} \lVert \theta_0 \rVert^2 = m(d + 1)$, and $\mathbb{E}[f(x; \theta_0)^2] = \lVert x \rVert^2 / 2$ for every $m$.
3. Hence the relative change obeys $\lVert \omega_k \rVert / \lVert \theta_0 \rVert \le \lVert r_0 \rVert / \sqrt{\lambda_{\min} \lVert \theta_0 \rVert^2}$, which is of order $1 / \sqrt{m}$: the numerator and $\lambda_{\min}$ have laws that settle as $m$ grows, and the denominator grows like $\sqrt{m (d+1)}$.

### Assumptions

1. $\hat\Theta$ is invertible. As $m \to \infty$ it tends to $\Theta_\infty$ (relu-ntk.md), and we assume $\Theta_\infty$ is positive definite on the training inputs; Lee et al. make the same assumption, $\lambda_{\min}(\Theta) > 0$, in their Theorem 2.1. Used in steps 2 and 4.
2. $0 < \eta \le n / \lambda_{\max}$, so every factor $q_i = 1 - \eta \lambda_i / n$ lies in $[0, 1)$. Used in step 3.
3. The targets $y$ do not depend on $m$. Used in step 5.

### Steps

1. From kernel-flow.md, step 5, $\omega_k = -\frac{\eta}{n} J^\top \sum_{l=0}^{k-1} r_l = -\frac{\eta}{n} J^\top \sum_{l=0}^{k-1} (I - \eta \hat\Theta / n)^l r_0$.
2. The geometric sum is $\sum_{l<k} (I - \eta \hat\Theta / n)^l = \frac{n}{\eta} \hat\Theta^{-1} \big(I - (I - \eta \hat\Theta / n)^k\big)$, since $\hat\Theta$ is invertible (assumption 1). So $\omega_k = -J^\top \hat\Theta^{-1} s_k$ with $s_k = (I - (I - \eta\hat\Theta/n)^k) r_0$. This is the discrete form of eq. (8) of Lee et al.
3. Then $\lVert \omega_k \rVert^2 = s_k^\top \hat\Theta^{-1} J J^\top \hat\Theta^{-1} s_k = s_k^\top \hat\Theta^{-1} s_k = \sum_i (1 - q_i^k)^2 c_i^2 / \lambda_i$, with $c_i = v_i^\top r_0$. By assumption 2, $0 \le 1 - q_i^k \le 1$, so the sum is at most $\sum_i c_i^2 / \lambda_i = r_0^\top \hat\Theta^{-1} r_0 \le \lVert r_0 \rVert^2 / \lambda_{\min}$, and it tends to that value as $k \to \infty$. Along gradient flow, $1 - e^{-\eta\lambda_i t/n}$ lies in $[0, 1)$ too. This is part 1. {#step-norm-bound}
4. $\lVert \theta_0 \rVert^2 = \sum_j (\lVert w_j \rVert^2 + a_j^2)$ is a sum of $m(d+1)$ squared standard Gaussians, with mean $m(d + 1)$; by the law of large numbers $\lVert \theta_0 \rVert^2 / (m(d+1)) \to 1$.
5. $\mathbb{E}[f(x; \theta_0)^2] = \frac{1}{m} \sum_j \mathbb{E}[a_j^2]\, \mathbb{E}[\sigma(w_j^\top x)^2] = \mathbb{E}[\sigma(w^\top x)^2] = \lVert x \rVert^2 / 2$, the second-layer term of relu-ntk.md at $\varphi = 0$, for every $m$. So $\lVert r_0 \rVert$ has a law with second moment bounded in $m$, and by relu-ntk.md, $\lambda_{\min}(\hat\Theta) \to \lambda_{\min}(\Theta_\infty) > 0$. With step 4, the bound of part 1 divided by $\lVert \theta_0 \rVert$ is of order $1/\sqrt{m}$. This is parts 2 and 3. {#step-relative}

### Result and check

The total displacement $\lVert \theta - \theta_0 \rVert$ stays of order 1 while the norm of $\theta_0$ grows like $\sqrt{m}$, so the relative change of the weights falls like $1/\sqrt{m}$.
The bound is for the linearized model.
For the network itself, Lee et al. (Theorem 2.1, Section 2.4) prove that, below a critical learning rate and with $\lambda_{\min}(\Theta) > 0$, $\sup_t \lVert \theta_t - \theta_0 \rVert_2 / \sqrt{n}$ is $O(n^{-1/2})$ in their width $n$; their Figure 1 shows the $1/\sqrt{n}$ fall for the input and output layers.
The width does not cause this by itself: in the scaled model $\alpha h$ of Chizat et al. (p. 3), a large $\alpha$ makes any model with $h(w_0) = 0$ lazy at a fixed size.

[lazy-weights.check.py](lazy-weights.check.py) trains the network itself by gradient descent ($\eta = 1$, 300 steps) on six points of the unit sphere of $\mathbb{R}^3$, five draws per width.
In our run, $\lVert \theta_T - \theta_0 \rVert$ stays between 2.0 and 2.6 for $m = 64$ to $8192$, while the relative change falls from 0.155 to 0.0116; the fitted slope of log relative change against log $m$ is $-0.512$.
For the linearized model at $m = 512$ with $\eta = n / \lambda_{\max}$ and 2000 steps, the largest $\lVert \omega_k \rVert$ is 2.632, equal to $\sqrt{r_0^\top \hat\Theta^{-1} r_0}$, below the looser $\lVert r_0 \rVert / \sqrt{\lambda_{\min}} = 8.81$.
At the fixed width $m = 256$, the centred model $\alpha (f - f_0)$ trained on the loss divided by $\alpha^2$ moves its weights by 0.444, 0.111 and 0.0276 for $\alpha = 4$, 16 and 64; the fitted slope against $\log\alpha$ is $-1.001$, the $O(1/\alpha)$ of Theorem 2.2 of Chizat et al.
