---
id: scale-invariance
derives: [c-scale-gradient, c-norm-grows, c-effective-lr, c-li-arora-identity, c-equilibrium, eq-effective-lr, eq-equilibrium]
kind: identity
assumes: ["L(c w) = L(w) for every c > 0", "plain gradient descent with a constant learning rate eta", "for item 5, a gradient of norm G/|w| at every step"]
uses: [batch-normalization@al, layer-normalization@al, weight-normalization@al]
status: full
check: scale-invariance.check.py
lean: null
sources:
  - {work: van-laarhoven-2017-l2-regularization-normalization, locator: "Section 3.1"}
  - {work: li-2019-exponential-learning-rate, locator: "page 2"}
---

### Statement

Let $L(cw) = L(w)$ for all $c > 0$, as for the weights feeding a batch or layer normalization, and write $g = \nabla L(w)$.

1. $\nabla L(cw) = \nabla L(w)/c$ and $w^\top\nabla L(w) = 0$.
2. Without decay, each step of gradient descent increases the squared norm: $\lVert w_{t+1}\rVert^2 = \lVert w_t\rVert^2 + \eta^2\lVert g_t\rVert^2$.
3. The direction $u = w/\lVert w\rVert$ after a step is the direction after a step of size $\eta/\lVert w\rVert^2$ taken from $u$ with the gradient $\nabla L(u)$, so the effective learning rate of the direction is $\eta/\lVert w\rVert^2$.
4. Gradient descent with decay, $w_{t+1} = (1 - \eta\lambda)w_t - \eta\nabla L(w_t)$, and gradient descent without decay with the learning rate $\tilde\eta_t = \eta(1 - \eta\lambda)^{-2t-1}$, started at the same point, produce $\tilde w_t = (1 - \eta\lambda)^{-t}w_t$, so the same directions and the same losses. This is Theorem 1.1 of Li and Arora with momentum 0.
5. If $\lVert g_t\rVert = G/\lVert w_t\rVert$ at every step, the squared norm $r = \lVert w\rVert^2$ of item 4 converges to $r^\star = \sqrt{\eta G^2/(\lambda(2 - \eta\lambda))}$. There the effective learning rate is $\eta/r^\star = \sqrt{\eta\lambda(2 - \eta\lambda)}/G$, and the angle $\phi$ that the direction turns per step has $\tan\phi = \sqrt{\eta\lambda(2 - \eta\lambda)}/(1 - \eta\lambda) \approx \sqrt{2\eta\lambda}$.

### Assumptions

1. $L$ is differentiable away from 0 and $L(cw) = L(w)$ for $c > 0$. Used in all steps.
2. Plain gradient descent with a constant $\eta$ and no momentum. Used in steps 2 to 5; Li and Arora treat momentum as well.
3. In step 5, $\lVert g_t\rVert\,\lVert w_t\rVert = G$ is constant. This is a model, not a property of training: it says that the gradient at the unit sphere keeps the same size. It is used only for item 5.

### Steps

1. Differentiate $L(cw) = L(w)$ in $w$: $c\,\nabla L(cw) = \nabla L(w)$. Differentiate it in $c$ at $c = 1$: $w^\top\nabla L(w) = 0$. These are the scaling of the gradient in Section 3 of van Laarhoven.
2. $\lVert w - \eta g\rVert^2 = \lVert w\rVert^2 - 2\eta\,w^\top g + \eta^2\lVert g\rVert^2$, and the middle term is zero by step 1.
3. Write $w = \lVert w\rVert u$. By step 1, $\nabla L(w) = \nabla L(u)/\lVert w\rVert$, so $w - \eta\nabla L(w) = \lVert w\rVert\big(u - \frac{\eta}{\lVert w\rVert^2}\nabla L(u)\big)$, and the two vectors have the same direction. This is the effective learning rate $\eta/\lVert w\rVert_2^2$ of Section 3.1 of van Laarhoven.
4. Write $q = 1 - \eta\lambda$ and suppose $\tilde w_t = q^{-t}w_t$. By step 1, $\nabla L(\tilde w_t) = q^t\nabla L(w_t)$. The undecayed step gives $\tilde w_{t+1} = q^{-t}w_t - \eta q^{-2t-1}q^t\nabla L(w_t) = q^{-t-1}(q w_t - \eta\nabla L(w_t)) = q^{-t-1}w_{t+1}$. Induction from $\tilde w_0 = w_0$ proves item 4.
5. With decay, $w_{t+1} = q w_t - \eta g_t$ and $w_t^\top g_t = 0$, so $r_{t+1} = q^2 r_t + \eta^2 G^2/r_t$. A fixed point satisfies $(1 - q^2)r^2 = \eta^2G^2$, and $1 - q^2 = \eta\lambda(2 - \eta\lambda)$, which gives $r^\star$. The derivative of the map at $r^\star$ is $q^2 - \eta^2G^2/r^{\star 2} = 2q^2 - 1$, whose magnitude is below 1 for $0 < q < 1$, so the fixed point is stable. The direction turns by $\tan\phi = \eta\lVert g\rVert/(q\lVert w\rVert) = \eta G/(q r^\star)$, which simplifies to the stated value.

### Result and check

$$ \eta_{\text{eff}} = \frac{\eta}{\lVert w\rVert_2^2}, \qquad \lVert w\rVert^2 \to \sqrt{\frac{\eta G^2}{\lambda(2 - \eta\lambda)}}, \qquad \tan\phi = \frac{\sqrt{\eta\lambda(2 - \eta\lambda)}}{1 - \eta\lambda} \approx \sqrt{2\eta\lambda} . $$

The check `scale-invariance.check.py` builds a scale-invariant loss, a quadratic in $w/\lVert w\rVert$, and checks item 1 for three scales and by finite differences.
It checks item 2 at each of 50 steps, item 3 at one step, and item 4 over 60 steps to a relative error of $10^{-9}$.
It iterates the norm recursion of item 5 for three pairs $(\eta, \lambda)$ and finds $r^\star$ to $10^{-8}$, the angle within 2% of $\sqrt{2\eta\lambda}$, and the effective learning rate as stated.
