---
id: sgd-equivalence
derives: [c-sgd-equivalence, c-exponential-decay, c-momentum-differs, c-momentum-fixed-point, eq-l2-step, eq-decay-step]
kind: identity
assumes: ["the L2 penalty is (lambda/2) |theta|^2 added to the loss", "a constant learning rate eta", "heavy-ball momentum in the form of torch.optim.SGD with dampening 0"]
uses: [stochastic-gradient-descent@m2, regularization@al]
status: full
check: sgd-equivalence.check.py
lean: null
sources:
  - {work: loshchilov-2017-decoupled-weight-decay, locator: "Section 2"}
  - {work: krogh-1991-simple-weight-decay, locator: "Section 1"}
---

### Statement

Let $g_t$ be the gradient of the loss at $\theta_{t-1}$, let L2 regularization add $\frac{\lambda}{2}\lVert\theta\rVert^2$ to the loss, and let decoupled weight decay multiply the parameters by $1 - \delta$ beside the gradient step. Then

1. for plain SGD with learning rate $\eta$, L2 with coefficient $\lambda$ takes exactly the steps of decoupled decay with $\delta = \eta\lambda$, for every sequence of gradients;
2. with no loss gradient, both shrink the weights geometrically, $\theta_k = (1 - \eta\lambda)^k \theta_0$, the discrete form of the exponential decay $\dot\theta = -\lambda\theta$;
3. with heavy-ball momentum $\mu > 0$, where L2 adds $\lambda\theta$ to the gradient that enters the buffer, the two forms with $\delta = \eta\lambda$ agree at the first step and differ at the second by exactly $\eta\mu\lambda\theta_0$, so no constant $\delta$ reproduces L2 along the whole trajectory;
4. on a fixed loss the L2 form stops where $\nabla f(\theta) + \lambda\theta = 0$, and the decoupled form where $\nabla f(\theta) + \frac{\delta(1 - \mu)}{\eta}\theta = 0$; the fixed points coincide when $\delta = \eta\lambda/(1 - \mu)$.

### Assumptions

1. The penalty is $\frac{\lambda}{2}\lVert\theta\rVert^2$, whose gradient is $\lambda\theta$. Used in all steps.
2. The learning rate $\eta$ is constant. Used in steps 1, 3 and 4; with a schedule, item 1 holds step by step with $\delta_t = \eta_t\lambda$.
3. Momentum is the form of `torch.optim.SGD` with dampening 0: $b_1 = g_1$, $b_t = \mu b_{t-1} + g_t$, $\theta_t = \theta_{t-1} - \eta b_t$, where $g_t$ includes $\lambda\theta_{t-1}$ under L2. Used in steps 3 and 4. Decoupled decay keeps the loss gradient alone in the buffer and takes $\theta_t = (1 - \delta)\theta_{t-1} - \eta b_t$.

### Steps

1. Plain SGD with L2 takes $\theta_t = \theta_{t-1} - \eta(g_t + \lambda\theta_{t-1}) = (1 - \eta\lambda)\theta_{t-1} - \eta g_t$, which is the decoupled step with $\delta = \eta\lambda$. Nothing in the step depends on $g_t$ beyond the shared term $-\eta g_t$, so the two trajectories agree for any gradients. This is Proposition 1 of Loshchilov and Hutter, written with $\delta$ for their $\lambda$ and $\lambda$ for their $\lambda'$.
2. With $g_t = 0$ step 1 gives $\theta_t = (1 - \eta\lambda)\theta_{t-1}$, so $\theta_k = (1 - \eta\lambda)^k\theta_0$. For small $\eta$ this is $e^{-\lambda k\eta}\theta_0$, the solution of $\dot\theta = -\lambda\theta$ at time $k\eta$, the exponential decay that Krogh and Hertz describe for the continuous learning equation.
3. With momentum, the first step under L2 is $\theta_1 = \theta_0 - \eta(g_1 + \lambda\theta_0)$, the decoupled first step with $\delta = \eta\lambda$. At the second step the L2 buffer is $\mu(g_1 + \lambda\theta_0) + g_2 + \lambda\theta_1$ and the decoupled buffer is $\mu g_1 + g_2$. Since $\theta_1$ is the same in both, the L2 step is larger by $\eta\mu\lambda\theta_0$, the decay of step 1 remembered by the buffer. A different $\delta$ would break the first step, so no constant $\delta$ matches both.
4. At a fixed point of the L2 form, $b = \mu b + \nabla f(\theta) + \lambda\theta$ and $\eta b = 0$, so $\nabla f(\theta) + \lambda\theta = 0$. At a fixed point of the decoupled form, $b = \nabla f(\theta)/(1 - \mu)$ and $\delta\theta + \eta b = 0$, so $\nabla f(\theta) + \frac{\delta(1 - \mu)}{\eta}\theta = 0$. The two equations are the same when $\delta = \eta\lambda/(1 - \mu)$.

### Result and check

$$ \theta_t = \theta_{t-1} - \eta\,(g_t + \lambda\theta_{t-1}) = (1 - \eta\lambda)\,\theta_{t-1} - \eta\, g_t . $$

The check `sgd-equivalence.check.py` runs both forms of plain SGD for 40 steps of a noisy quadratic and finds them equal to a relative error of $10^{-12}$.
It checks the geometric decay over 25 steps with no gradient.
With $\mu = 0.9$ it checks that the first steps agree, that the second steps differ by exactly $\eta\mu\lambda\theta_0$, and that the trajectories have separated after 40 steps.
On a deterministic quadratic it runs 3000 steps and finds the L2 form and the decoupled form with $\delta = \eta\lambda/(1 - \mu)$ at the solution of $(H + \lambda I)\theta = c$, and the decoupled form with $\delta = \eta\lambda$ at the solution of $(H + \lambda(1 - \mu)I)\theta = c$.
