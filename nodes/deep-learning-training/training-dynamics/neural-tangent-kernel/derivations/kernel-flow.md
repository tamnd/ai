---
id: kernel-flow
derives: [c-residual-flow, c-eigen-rates, c-gd-discrete, eq-residual-flow, eq-gd-step]
kind: closed-form
assumes: ["the linearized model f_lin, or any model whose tangent kernel on the training inputs stays constant", "the squared loss (1/(2n)) sum_i (f(x_i) - y_i)^2 over n training points", "gradient flow at rate eta, or gradient descent with step eta"]
uses: []
status: full
check: kernel-flow.check.py
lean: null
sources:
  - {work: lee-2019-wide-networks-linear-models, locator: "Section 2.2"}
  - {work: jacot-2018-neural-tangent-kernel, locator: "p. 6"}
  - {work: jacot-2018-neural-tangent-kernel, locator: "p. 7"}
---

### Statement

Let $X = (x_1, \dots, x_n)$ be the training inputs and $y \in \mathbb{R}^n$ the targets.
Let $J \in \mathbb{R}^{n \times P}$ be the Jacobian of the network outputs at $X$ with respect to the $P$ parameters, taken at the initial parameters $\theta_0$, and let $\hat\Theta = J J^\top$, an $n \times n$ symmetric positive semidefinite matrix with eigenpairs $(\lambda_i, v_i)$.
Train the linearized model $f_{\text{lin}}(x; \theta) = f(x; \theta_0) + \nabla_\theta f(x; \theta_0)^\top (\theta - \theta_0)$ on $L(\theta) = \frac{1}{2n} \lVert f_{\text{lin}}(X; \theta) - y \rVert^2$, and write $r = f_{\text{lin}}(X; \theta) - y$ for the residual.

1. Under gradient flow $\dot\theta = -\eta \nabla L(\theta)$, $r(t) = \exp(-\eta \hat\Theta t / n)\, r(0)$.
2. So $v_i^\top r(t) = e^{-\eta \lambda_i t / n}\, v_i^\top r(0)$: each eigen-component decays at the rate $\eta \lambda_i / n$, and the component in the null space of $\hat\Theta$ does not move.
3. Under gradient descent $\theta_{k+1} = \theta_k - \eta \nabla L(\theta_k)$, $r_k = (I - \eta \hat\Theta / n)^k r_0$ exactly. Each component is multiplied by $1 - \eta \lambda_i / n$ per step; the iteration converges on every component with $\lambda_i > 0$ when $\eta < 2n / \lambda_{\max}$, and at a fixed time $t = \eta k$ it tends to part 1 as $\eta \to 0$.

### Assumptions

1. The model is linear in $\theta$ with the fixed Jacobian $J$. Used in step 2. The same steps hold for any model whose tangent kernel on $X$ is constant in time; Jacot et al. (Theorem 2) show that the NTK of a network tends to such a constant kernel as the widths grow.
2. The loss is the squared loss with the factor $1/n$. Used in step 1. Lee et al. use the total loss without $1/n$, and their eq. (9) has $e^{-\eta \hat\Theta_0 t}$; Jacot et al. average over the data, so their map $\Pi$ carries the $1/N$.
3. $\eta > 0$ is constant. Used in steps 3 and 5.

### Steps

1. The gradient of the loss is $\nabla L(\theta) = \frac{1}{n} J^\top r$, because $\partial f_{\text{lin}}(x_i; \theta) / \partial \theta = \nabla_\theta f(x_i; \theta_0)$, the $i$-th row of $J$, and $\partial \frac{1}{2n} r_i^2 / \partial r_i = r_i / n$.
2. By the chain rule and assumption 1, $\dot r = J \dot\theta = -\frac{\eta}{n} J J^\top r = -\frac{\eta}{n} \hat\Theta\, r$. This is a linear ODE with a constant matrix.
3. Its solution is $r(t) = \exp(-\eta \hat\Theta t / n)\, r(0)$. That is part 1. Differentiating the series of the matrix exponential gives back the ODE, and the solution of a linear ODE with given initial value is unique.
4. Write $\hat\Theta = \sum_i \lambda_i v_i v_i^\top$ with orthonormal $v_i$. Then $\exp(-\eta \hat\Theta t / n) = \sum_i e^{-\eta \lambda_i t / n} v_i v_i^\top$, and multiplying part 1 by $v_i^\top$ gives part 2. For $\lambda_i = 0$ the factor is 1.
5. For gradient descent, step 1 gives $\theta_{k+1} - \theta_k = -\frac{\eta}{n} J^\top r_k$. Multiply by $J$: $r_{k+1} - r_k = -\frac{\eta}{n} \hat\Theta\, r_k$, so $r_{k+1} = (I - \eta \hat\Theta / n)\, r_k$ and $r_k = (I - \eta \hat\Theta / n)^k r_0$ with no error term, because the model is linear. {#step-gd-map}
6. In the eigenbasis, the factor of component $i$ is $(1 - \eta \lambda_i / n)^k$. It tends to 0 exactly when $\lvert 1 - \eta \lambda_i / n \rvert < 1$, that is $0 < \eta \lambda_i / n < 2$; for all positive $\lambda_i$ at once this is $\eta < 2n / \lambda_{\max}$.
7. Fix $t$ and set $k = t / \eta$. Then $(1 - \eta \lambda_i / n)^{t / \eta} = \exp\big( (t/\eta) \log(1 - \eta \lambda_i / n) \big) = \exp\big( -\lambda_i t / n - \eta \lambda_i^2 t / (2 n^2) + O(\eta^2) \big)$, which tends to $e^{-\lambda_i t / n}$, the flow of part 1 at rate 1 and time $t$, with an error of first order in $\eta$. This is part 3.

### Result and check

Under a constant kernel, training is a linear filter on the residual: the large eigen-directions of $\hat\Theta$ are fitted first, and the small ones last.
This is the statement of Jacot et al. (Section 5) that convergence is fastest along the largest kernel principal components, written for a finite data set and a learning rate $\eta$.
Lee et al. (Section 2.2) note the discrete form: replace $e^{-\eta \hat\Theta_0 t}$ by $1 - (1 - \eta \hat\Theta_0)^i$ in their formulas.

[kernel-flow.check.py](kernel-flow.check.py) builds a two-layer ReLU network of width 512 on 8 points of the unit sphere of $\mathbb{R}^3$ and runs gradient descent on its linearized model.
In our run, the residual after 200 steps at $\eta = 1$ matches $(I - \eta \hat\Theta / n)^k r_0$ to $8.9 \times 10^{-17}$.
At the fixed time $T = 100$, the gap to $\exp(-\hat\Theta T / n) r_0$ is $7.7 \times 10^{-4}$ at $\eta = 1$ and halves with each halving of $\eta$ (ratios 2.005, 2.002, 2.001), as step 7 says.
The decay rate fitted to each eigen-component over 400 steps at $\eta = 0.05$ equals $-\log(1 - \eta \lambda_i / n)$ to $10^{-9}$, and is within 1% of $\eta \lambda_i / n$ for all eight eigenvalues, from 0.118 to 3.01.
