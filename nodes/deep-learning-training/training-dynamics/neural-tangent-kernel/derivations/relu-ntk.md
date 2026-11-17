---
id: relu-ntk
derives: [c-relu-ntk-limit, c-relu-ntk-rate, eq-relu-ntk, eq-jacobian]
kind: closed-form
assumes: ["f(x) = (1/sqrt(m)) sum_j a_j relu(w_j . x) with w_j ~ N(0, I_d) and a_j ~ N(0, 1), all independent", "no biases", "the width m grows with the inputs fixed"]
uses: []
status: full
check: relu-ntk.check.py
lean: null
sources:
  - {work: jacot-2018-neural-tangent-kernel, locator: "p. 5"}
  - {work: jacot-2018-neural-tangent-kernel, locator: "p. 6"}
  - {work: lee-2019-wide-networks-linear-models, locator: "p. 4"}
---

### Statement

Let $f(x) = \frac{1}{\sqrt m} \sum_{j=1}^m a_j\, \sigma(w_j^\top x)$ with $\sigma(u) = \max(u, 0)$, $x \in \mathbb{R}^d$, and independent $w_j \sim \mathcal{N}(0, I_d)$, $a_j \sim \mathcal{N}(0, 1)$.
For two nonzero inputs $x, x'$, let $\varphi \in [0, \pi]$ be the angle between them.

1. The empirical NTK is a mean of $m$ independent terms:
$\hat\Theta_m(x, x') = \frac{1}{m} \sum_{j=1}^m \big[ a_j^2\, \sigma'(w_j^\top x)\, \sigma'(w_j^\top x')\, x^\top x' + \sigma(w_j^\top x)\, \sigma(w_j^\top x') \big]$.
2. As $m \to \infty$ it tends almost surely to
$\Theta_\infty(x, x') = x^\top x' \, \frac{\pi - \varphi}{2\pi} + \lVert x \rVert \lVert x' \rVert \, \frac{\sin\varphi + (\pi - \varphi)\cos\varphi}{2\pi}$.
The first term comes from the first-layer weights $w_j$ and the second from the second-layer weights $a_j$.
3. $\mathbb{E}[\hat\Theta_m(x, x')] = \Theta_\infty(x, x')$ for every $m$, and $\mathbb{E}\big[(\hat\Theta_m(x, x') - \Theta_\infty(x, x'))^2\big] = s^2(x, x') / m$ with $s^2$ the variance of one term. So the root-mean-square error is exactly proportional to $1/\sqrt m$, and $\sqrt m\, (\hat\Theta_m - \Theta_\infty)$ tends in law to a Gaussian.

### Assumptions

1. The parameters are independent across $j$, and $a_j$ is independent of $w_j$ with $\mathbb{E}[a_j^2] = 1$. Used in steps 2 and 3.
2. The law of $w_j$ is $\mathcal{N}(0, I_d)$, so it is invariant under rotations. Used in steps 4 and 5.
3. $\sigma'(u) = \mathbf{1}[u > 0]$, with any value at $u = 0$; this set has probability 0 for $x \neq 0$. Used in step 1.
4. One term has finite variance: $\sigma(u)^2 \le u^2$ and Gaussian moments are finite. Used in step 6.

### Steps

1. The derivatives are $\partial f / \partial a_j = \sigma(w_j^\top x) / \sqrt m$ and $\partial f / \partial w_j = a_j\, \sigma'(w_j^\top x)\, x / \sqrt m$. {#step-jacobian} Summing the products of the derivatives at $x$ and at $x'$ over all parameters gives part 1; the $w_j$ block contributes $a_j^2 \sigma'\sigma'\, x^\top x'$ and the $a_j$ block $\sigma \sigma$.
2. The $m$ terms are independent and identically distributed. By the strong law of large numbers, $\hat\Theta_m$ tends almost surely to the mean of one term, $\mathbb{E}[a^2]\, x^\top x'\, \mathbb{E}[\sigma'(w^\top x)\sigma'(w^\top x')] + \mathbb{E}[\sigma(w^\top x)\sigma(w^\top x')]$, and $\mathbb{E}[a^2] = 1$.
3. By rotation invariance (assumption 2), choose coordinates with $x = \lVert x \rVert e_1$ and $x' = \lVert x' \rVert (\cos\varphi\, e_1 + \sin\varphi\, e_2)$. Only $(w_1, w_2)$ enters, a standard Gaussian in the plane. Write it in polar form $(\rho \cos\psi, \rho \sin\psi)$, with density $\frac{1}{2\pi} \rho\, e^{-\rho^2/2}$ in $(\rho, \psi)$.
4. Then $w^\top x = \lVert x \rVert \rho \cos\psi$ and $w^\top x' = \lVert x' \rVert \rho \cos(\psi - \varphi)$. Both are positive exactly when $\psi \in (\varphi - \pi/2, \pi/2)$, an arc of length $\pi - \varphi$. So $\mathbb{E}[\sigma'\sigma'] = \frac{1}{2\pi} \int_0^\infty \rho e^{-\rho^2/2} d\rho \cdot (\pi - \varphi) = \frac{\pi - \varphi}{2\pi}$.
5. On the same arc, $\mathbb{E}[\sigma\sigma] = \frac{\lVert x \rVert \lVert x' \rVert}{2\pi} \int_0^\infty \rho^3 e^{-\rho^2/2} d\rho \int_{\varphi - \pi/2}^{\pi/2} \cos\psi \cos(\psi - \varphi)\, d\psi$. The radial integral is 2. With $\cos\psi\cos(\psi - \varphi) = \frac12[\cos(2\psi - \varphi) + \cos\varphi]$, the angular integral is $\frac14 [\sin(\pi - \varphi) - \sin(\varphi - \pi)] + \frac12 (\pi - \varphi)\cos\varphi = \frac12 [\sin\varphi + (\pi - \varphi)\cos\varphi]$. So $\mathbb{E}[\sigma\sigma] = \lVert x \rVert \lVert x' \rVert\, \frac{\sin\varphi + (\pi - \varphi)\cos\varphi}{2\pi}$. With steps 2 and 4 this is part 2. {#step-arc-cosine}
6. Step 2 also shows that the mean of one term is $\Theta_\infty$, so $\hat\Theta_m$ is unbiased. The variance of a mean of $m$ independent terms with variance $s^2$ (finite by assumption 4) is $s^2 / m$, which is part 3; the central limit theorem gives the Gaussian law.

### Result and check

The two terms are the two summands of the recursion of Jacot et al. (Theorem 1) for $L = 2$ with no bias, $\Theta^{(2)}_\infty = \Theta^{(1)}_\infty \dot\Sigma^{(2)} + \Sigma^{(2)}$, written without their factor $1/\sqrt{n_0}$ on the input so that $\Sigma^{(1)}(x, x') = x^\top x'$.
In their Remark 4, the term $\Sigma^{(2)}$ is the learning due to the last layer and the other term the learning of the lower layers.
The two Gaussian integrals are the arc-cosine forms that Lee et al. (supplementary p. 4, eqs. S13 to S15) take from Cho and Saul.
At $x = x'$ the kernel is $\lVert x \rVert^2$, half from each term; at $\varphi = \pi/2$ with unit vectors it is $1/(2\pi)$; at $\varphi = \pi$ it is 0.

[relu-ntk.check.py](relu-ntk.check.py) uses six points of $\mathbb{R}^4$ with norms between 0.5 and 2.
In our run, one million Monte Carlo draws of $w$ match the two integrals of steps 4 and 5 within 2.4 standard errors at every pair.
Over 30 draws per width, the root-mean-square error of the empirical NTK is 0.567 at $m = 64$ and 0.0187 at $m = 65536$; $\sqrt m$ times the error stays between 3.95 and 5.26.
The fitted slope of log error against log width is $-0.505$, against $-1/2$ from part 3.
