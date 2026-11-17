---
id: spectral-filters
derives: [c-gd-filter, c-ridge-filter, c-matched-slope, c-sandwich, c-not-identical, eq-gd-filter, eq-ridge-filter, eq-matched-lambda, eq-sandwich]
kind: identity
assumes: ["least squares loss (1/2n) |y - X w|^2", "gradient descent from w = 0 with a constant step eta", "eta s_max <= 1 for the sandwich"]
uses: [ridge-regression@al, stochastic-gradient-descent@m2]
status: full
check: spectral-filters.check.py
lean: null
sources:
  - {work: ali-2019-continuous-time-early-stopping, locator: "Section 3.1"}
  - {work: yao-2007-early-stopping-gradient-descent, locator: "Section 3.1"}
---

### Statement

Let $\hat\Sigma = X^\top X/n = V S V^\top$ with eigenvalues $s_i \ge 0$ and eigenvectors $v_i$, and let $b = X^\top y/n$.
Write $c_i = v_i^\top b$.
Then

1. $t$ steps of gradient descent with step $\eta$ from $w_0 = 0$ give $w_t = \sum_{s_i > 0} \varphi_t(s_i)\, (c_i/s_i)\, v_i$ with $\varphi_t(s) = 1 - (1 - \eta s)^t$;
2. ridge with penalty $\lambda$ gives the same sum with $r_\lambda(s) = s/(s + \lambda)$;
3. both filters vanish at $s = 0$, with slopes $\eta t$ and $1/\lambda$, so they agree to first order in $s$ exactly when $\lambda = 1/(\eta t)$;
4. their second-order terms differ, $-\eta^2 t(t-1)/2$ against $-\eta^2 t^2$, so no $\lambda$ makes them equal;
5. for $0 \le \eta s \le 1$ and $\lambda = 1/(\eta t)$, $r_\lambda(s) \le \varphi_t(s) \le \min(1, \eta t s) \le 2 r_\lambda(s)$, and hence $\lVert w^{\text{ridge}}\rVert \le \lVert w_t\rVert \le 2 \lVert w^{\text{ridge}}\rVert$.

### Assumptions

1. The loss is $\frac{1}{2n}\lVert y - Xw\rVert^2$, whose gradient is $\hat\Sigma w - b$. Used in steps 1 and 2.
2. Gradient descent starts at $w_0 = 0$ with a constant step $\eta$. Used in step 1. Another start adds $(1 - \eta s_i)^t v_i^\top w_0$ in each direction.
3. $0 \le \eta s_i \le 1$ for all $i$. Used in step 5 only; steps 1 to 4 hold for any $\eta$.

### Steps

1. The step is $w_k = w_{k-1} + \eta(b - \hat\Sigma w_{k-1})$. In the eigenbasis, with $a_{k,i} = v_i^\top w_k$, it reads $a_{k,i} = (1 - \eta s_i)\, a_{k-1,i} + \eta c_i$. From $a_{0,i} = 0$ the geometric sum gives $a_{t,i} = \eta c_i \sum_{j<t} (1 - \eta s_i)^j = \varphi_t(s_i)\, c_i/s_i$ for $s_i > 0$. When $s_i = 0$, $c_i = v_i^\top X^\top y/n = (X v_i)^\top y/n = 0$ because $\lVert X v_i\rVert^2 = n s_i = 0$, so $a_{t,i} = 0$.
2. Ridge solves $(\hat\Sigma + \lambda I) w = b$, so $v_i^\top w = c_i/(s_i + \lambda) = r_\lambda(s_i)\, c_i/s_i$. This is the shrinkage map of Ali, Kolter and Tibshirani, Section 3.1, and the filter form of Yao, Rosasco and Caponnetto, Section 3.1.
3. Expanding, $\varphi_t(s) = \eta t s - \binom{t}{2}\eta^2 s^2 + O(s^3)$ and $r_\lambda(s) = s/\lambda - s^2/\lambda^2 + O(s^3)$. The linear terms agree if and only if $\lambda = 1/(\eta t)$.
4. At $\lambda = 1/(\eta t)$ the quadratic terms are $-\eta^2 t(t-1)/2$ and $-\eta^2 t^2$, never equal for $t \ge 1$. Ali et al. state the same for gradient flow: no parametrization $\lambda = \phi(t)$ equates the two maps, and $\phi(t) = 1/t$ makes them similar.
5. Write $x = \eta s \in [0, 1]$ and $u = t x$, so $r_\lambda(s) = u/(1 + u)$. Lower bound: $(1 - x)^t \le e^{-tx} \le 1/(1 + tx)$, the second being $e^{u} \ge 1 + u$, so $\varphi_t \ge 1 - 1/(1 + u) = r_\lambda$. Middle: $(1 - x)^t \ge 0$ gives $\varphi_t \le 1$, and Bernoulli's inequality $(1 - x)^t \ge 1 - t x$ for $x \le 1$ gives $\varphi_t \le u$. Upper: $\min(1, u) \le 2u/(1 + u)$, checked separately for $u \le 1$ and $u > 1$. The coefficients of $w_t$ and $w^{\text{ridge}}$ share signs in each direction and their ratio lies in $[1, 2]$, which gives the bound on norms.

### Result and check

$$ \varphi_t(s) = 1 - (1 - \eta s)^t, \qquad r_{1/(\eta t)}(s) = \frac{\eta t s}{1 + \eta t s} \le \varphi_t(s) \le 2\, r_{1/(\eta t)}(s) \quad (0 \le \eta s \le 1). $$

The check `spectral-filters.check.py` draws two random designs, one with $n = 60, p = 8$ and one with $n = 10, p = 16$, and runs gradient descent with $\eta = 1/s_{\max}$.
At $t = 1, 5, 25, 200$ the iterate equals the filtered sum to a relative error of $10^{-10}$, ridge equals its filtered sum, and the norm ratio $\lVert w_t\rVert/\lVert w^{\text{ridge}}\rVert$ lies in $[1, 2]$; it falls from 1.92 at $t = 1$ to 1.03 at $t = 200$ on the first design.
It confirms both slopes $\eta t$ and the two quadratic coefficients by finite differences, and checks the sandwich on a grid of $x \in [0, 1]$ for $t$ from 1 to 1000.
On that grid the largest ratio $\varphi_t/r$ is 2 at $t = 1$ and 1.2974 at $t = 1000$, approaching the constant 1.2985 that Ali et al. prove for gradient flow.
