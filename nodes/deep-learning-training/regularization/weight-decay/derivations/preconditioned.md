---
id: preconditioned
derives: [c-precond-not-equal, c-precond-penalty, c-precond-fixed-point]
kind: identity
assumes: ["a fixed symmetric positive definite preconditioner P", "the L2 penalty is (lambda/2) |theta|^2 added to the loss", "a constant learning rate eta"]
uses: [stochastic-gradient-descent@m2, adamw@al]
status: full
check: preconditioned.check.py
lean: null
sources:
  - {work: loshchilov-2017-decoupled-weight-decay, locator: "Section 2"}
  - {work: zhang-2018-three-mechanisms-weight-decay, locator: "Section 3"}
---

### Statement

Let an optimizer step by $-\eta P g$ for a fixed symmetric positive definite preconditioner $P$, with L2 taking $\theta \leftarrow \theta - \eta P(g + \lambda\theta)$ and decoupled decay taking $\theta \leftarrow (1 - \delta)\theta - \eta P g$. Then

1. the two steps agree for every $\theta$ if and only if $\eta\lambda P = \delta I$, that is, only if $P$ is a multiple of the identity;
2. the decoupled step is the preconditioned step on the loss plus the penalty $\frac{\delta}{2\eta}\,\theta^\top P^{-1}\theta$; for a diagonal $P = \operatorname{diag}(1/s_i)$ the penalty on coordinate $i$ is $\frac{\delta}{2\eta} s_i\theta_i^2$, so coordinates with large $s_i$ are penalized more, while under L2 their decay $\eta\lambda\theta_i/s_i$ is smaller;
3. on a fixed loss, L2 stops where $\nabla f(\theta) + \lambda\theta = 0$ whatever $P$ is, while decoupled decay stops where $\nabla f(\theta) + \frac{\delta}{\eta}P^{-1}\theta = 0$, which depends on $P$.

### Assumptions

1. $P$ is fixed and symmetric positive definite. Used in all steps. Adam's $P$ is $\operatorname{diag}(1/(\sqrt{\hat v_t} + \epsilon))$, which changes at every step, so for Adam the statements hold step by step and the penalty of item 2 is only approximate, as Zhang et al. note.
2. The L2 penalty is $\frac{\lambda}{2}\lVert\theta\rVert^2$. Used in steps 1 and 3.
3. $\eta$ is constant. Used in steps 2 and 3.

### Steps

1. Subtract the two steps: L2 minus decoupled is $\delta\theta - \eta\lambda P\theta = (\delta I - \eta\lambda P)\theta$. This is zero for every $\theta$ if and only if $\eta\lambda P = \delta I$. If $P = kI$ the L2 coefficient $\lambda = \delta/(\eta k)$ reproduces the decay; otherwise no $\lambda$ does. This is Proposition 2 of Loshchilov and Hutter, and equations (6) and (7) of Zhang et al. For $P = \operatorname{diag}(1/s_i)$ the L2 decay of coordinate $i$ is $\eta\lambda\theta_i/s_i$.
2. The preconditioned step on $f(\theta) + \frac{\delta}{2\eta}\theta^\top P^{-1}\theta$ is $\theta - \eta P(g + \frac{\delta}{\eta}P^{-1}\theta) = \theta - \delta\theta - \eta P g$, which is the decoupled step. This is Proposition 3 of Loshchilov and Hutter, whose penalty is proportional to $\lVert\theta \odot \sqrt{s}\rVert^2$, and the squared $C$-norm of Zhang et al. with $C = P^{-1}$.
3. At a fixed point of the L2 step, $P(\nabla f + \lambda\theta) = 0$, and $P$ is invertible, so $\nabla f + \lambda\theta = 0$. At a fixed point of the decoupled step, $\delta\theta + \eta P\nabla f = 0$; multiplying by $P^{-1}/\eta$ gives $\nabla f + \frac{\delta}{\eta}P^{-1}\theta = 0$.

### Result and check

$$ (1 - \delta)\,\theta - \eta P g = \theta - \eta P\Big(g + \frac{\delta}{\eta}\,P^{-1}\theta\Big), \qquad \text{L2} = \text{decoupled} \iff \eta\lambda P = \delta I . $$

The check `preconditioned.check.py` takes $s = (100, 10, 1, 0.1, 0.01)$ and finds the difference of the two steps equal to $\delta\theta_i(1 - 1/s_i)$, zero only on the coordinate with $s_i = 1$.
It checks that $P = 3I$ with $\lambda = \delta/(3\eta)$ reproduces the decay, and that the decoupled step equals the preconditioned step on the penalized loss of item 2.
It runs both forms for 20000 steps on a quadratic with $s = (4, 2, 1, 0.5, 0.25)$, finds them at the two solutions of item 3, and finds the solutions different.
