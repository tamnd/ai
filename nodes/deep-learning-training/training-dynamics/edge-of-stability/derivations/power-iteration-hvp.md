---
id: power-iteration-hvp
derives: [c-hvp-central, c-power-iteration]
kind: closed-form
assumes: ["the loss is three times continuously differentiable near theta", "the Hessian at theta is symmetric with a unique eigenvalue of largest modulus", "the start vector has a nonzero component along its eigenvector"]
uses: []
status: full
check: power-iteration-hvp.check.py
lean: null
sources:
  - {work: cohen-2021-gradient-descent-edge-stability, locator: "p. 2"}
  - {work: damian-2023-self-stabilization-edge-stability, locator: "p. 2"}
---

### Statement

Let $L$ be a loss with gradient $g(\theta)$ and Hessian $H(\theta)$ near a point $\theta$, and let $v$ be a vector.
(i) The central difference
$$ \frac{g(\theta + \epsilon v) - g(\theta - \epsilon v)}{2\epsilon} = H(\theta) v + O(\epsilon^2) $$
and the error is exactly zero when $L$ is quadratic.
(ii) Let $H$ be symmetric with eigenvalues $\mu_1, \dots, \mu_d$ ordered so that $|\mu_1| > |\mu_2| \ge \dots \ge |\mu_d|$, with eigenvectors $q_i$, and let $q^{(0)}$ have $\langle q_1, q^{(0)}\rangle \ne 0$.
The iteration $q^{(k+1)} = H q^{(k)} / \lVert H q^{(k)}\rVert$ converges in direction to $\pm q_1$, and the Rayleigh quotient $q^{(k)\top} H q^{(k)}$ converges to $\mu_1$ with error $O(|\mu_2/\mu_1|^{2k})$.
(iii) When $H$ is positive semi-definite, $\mu_1$ is the largest eigenvalue, the sharpness. When $H$ has a negative eigenvalue larger in modulus than the top positive one, power iteration returns that negative eigenvalue instead.

### Assumptions

1. $L$ is three times continuously differentiable near $\theta$, so its third derivative is bounded on the segment from $\theta - \epsilon v$ to $\theta + \epsilon v$.
2. $H$ is symmetric and its eigenvalue of largest modulus is unique.
3. The start has a nonzero component along $q_1$, which holds with probability one for a Gaussian start.

### Steps

1. Taylor's theorem along the line gives $g(\theta \pm \epsilon v) = g(\theta) \pm \epsilon H v + \tfrac{\epsilon^2}{2} T[v, v] \pm O(\epsilon^3)$, with $T$ the third derivative. The even terms cancel in the difference, so dividing by $2\epsilon$ leaves $Hv + O(\epsilon^2)$. For a quadratic $L$, $g$ is affine, $g(\theta \pm \epsilon v) = g(\theta) \pm \epsilon Hv$, and the difference is exact. This is (i).
2. Write $q^{(0)} = \sum_i c_i q_i$. After $k$ applications, $H^k q^{(0)} = \sum_i c_i \mu_i^k q_i = c_1\mu_1^k\big(q_1 + \sum_{i>1} (c_i/c_1)(\mu_i/\mu_1)^k q_i\big)$; the normalization removes the scalar, and the remaining terms shrink as $|\mu_2/\mu_1|^k$.
3. With $q^{(k)} = q_1 + e$, $\lVert e\rVert = O(|\mu_2/\mu_1|^k)$, renormalized, the Rayleigh quotient of a symmetric matrix is stationary at an eigenvector, so its error is quadratic in $e$: $O(|\mu_2/\mu_1|^{2k})$. This is (ii).
4. If $H \succeq 0$, every $|\mu_i| = \mu_i$ and the eigenvalue of largest modulus is the largest. Otherwise the ordering is by modulus, so a negative eigenvalue of larger modulus wins, which is (iii). Away from a minimum, a shift $H + sI$ with $s$ larger than the most negative eigenvalue's modulus restores the order.

### Result and check

The sharpness of a network is computed without forming the Hessian: two gradient evaluations give one Hessian-vector product, and a few dozen products of power iteration give the top eigenvalue when the Hessian's largest eigenvalue dominates its negative ones, as it does near the trajectories studied here.
The sharpness that Cohen et al. (2021) and Damian et al. (2023) track is defined as the largest eigenvalue of the Hessian of the training loss (Cohen et al., p. 2; Damian et al., p. 2).
`power-iteration-hvp.check.py` checks that the central difference is exact for random quadratics, that its error on a quartic is $\epsilon^2$ times the predicted third-order term, and that power iteration with these products matches `numpy.linalg.eigvalsh` on random positive semi-definite matrices and on the full Hessian of a small tanh network; it also checks the geometric rate on a diagonal matrix and the negative-eigenvalue case.
