---
id: catapult-update
derives: [c-catapult-update, c-catapult-decrease]
kind: closed-form
assumes: ["the model f = n^(-1/2) v^T u with u, v in R^n", "one training example x = 1 with label y = 0 and loss L = f^2 / 2", "full-batch gradient descent with step size eta"]
uses: []
status: full
check: catapult-update.check.py
lean: null
sources:
  - {work: lewkowycz-2020-large-learning-rate-phase, locator: "p. 4"}
  - {work: lewkowycz-2020-large-learning-rate-phase, locator: "p. 5"}
---

### Statement

Let $u, v \in \mathbb{R}^n$, $f = n^{-1/2} v^\top u$, $L = f^2/2$ and $\lambda = n^{-1}(\lVert u\rVert^2 + \lVert v\rVert^2)$, and take one gradient step
$$ u' = u - \eta n^{-1/2} f v, \qquad v' = v - \eta n^{-1/2} f u. $$
Then, exactly,
$$ f' = \Big(1 - \eta\lambda + \frac{\eta^2 f^2}{n}\Big) f, \qquad \lambda' = \lambda + \frac{\eta f^2}{n}(\eta\lambda - 4). $$
So whenever $f \ne 0$ and $\eta\lambda < 4$, the step lowers $\lambda$, and whenever $\eta\lambda > 4$ it raises it.

### Assumptions

1. The model, the single example and the loss are as above; $\lambda$ is then the one entry of the neural tangent kernel on the training set and equals $\lVert\nabla_\theta f\rVert^2$.
2. The step is plain gradient descent with a fixed $\eta$.

### Steps

1. $\partial L/\partial u = f\,\partial f/\partial u = f n^{-1/2} v$ and $\partial L/\partial v = f n^{-1/2} u$, which gives the update.
2. $v'^\top u' = (v - a u)^\top(u - a v)$ with $a = \eta n^{-1/2} f$; expanding, $v^\top u - a\lVert v\rVert^2 - a\lVert u\rVert^2 + a^2 u^\top v$. Dividing by $\sqrt n$: $f' = f - \eta f\lambda + \eta^2 f^2 n^{-1} f$, the first equation.
3. $\lVert u'\rVert^2 + \lVert v'\rVert^2 = \lVert u\rVert^2 + \lVert v\rVert^2 - 4a\,u^\top v + a^2(\lVert u\rVert^2 + \lVert v\rVert^2)$. With $u^\top v = \sqrt n f$ and $a^2 = \eta^2 f^2/n$, dividing by $n$ gives $\lambda' = \lambda - 4\eta f^2/n + \eta^2 f^2\lambda/n$, the second equation.
4. The change $\lambda' - \lambda = (\eta f^2/n)(\eta\lambda - 4)$ has the sign of $\eta\lambda - 4$ when $f \ne 0$.

### Result and check

These are equations (6) and (7) of Lewkowycz et al. (2020) (p. 4), and no term was dropped.
They give the three phases: for $\eta < 2/\lambda_0$, $|1 - \eta\lambda| < 1$ and $f$ shrinks while $\lambda$ moves by $O(1/n)$; for $2/\lambda_0 < \eta < 4/\lambda_0$, $|f|$ first grows until $f^2/n$ is of order one, and then the second equation pushes $\lambda$ down until $\eta\lambda < 2$ and $f$ converges; above $4/\lambda_0$, $\lambda$ grows with $f$ and the loss diverges.
`catapult-update.check.py` compares one exact gradient step with both equations for random $u$, $v$ at several $n$ and $\eta$, checks the sign of the change of $\lambda$, and runs the three phases at $n = 1000$.
