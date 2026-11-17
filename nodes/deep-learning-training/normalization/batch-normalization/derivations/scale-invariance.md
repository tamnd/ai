---
id: scale-invariance
derives: [c-scale-invariance, c-gradient-scaling, c-norm-grows]
kind: identity
assumes: ["epsilon is 0", "a is a positive scalar", "the batch variance of each feature of Wu is positive", "the loss depends on W only through BN(Wu)"]
uses: [sample-mean@m2, sample-variance@m2, stochastic-gradient-descent@m2]
status: full
check: scale-invariance.check.py
lean: null
sources:
  - {work: ioffe-2015-batch-normalization, locator: "p. 5"}
  - {work: van-laarhoven-2017-l2-regularization-normalization, locator: "Section 3"}
---

### Statement

Let $u_1, \dots, u_m$ be the inputs of a batch and $W$ a weight matrix, and let BN normalize each feature of $Wu$ over the batch with $\epsilon = 0$.
Let $f(W) = \ell(\mathrm{BN}(Wu_1), \dots, \mathrm{BN}(Wu_m))$ for a differentiable loss $\ell$, and let $a > 0$. Then

1. $\mathrm{BN}((aW)u_i) = \mathrm{BN}(Wu_i)$ for every $i$;
2. $\nabla f(aW) = \frac{1}{a} \nabla f(W)$;
3. each row $w_k$ of $W$ satisfies $\langle w_k, \nabla_{w_k} f(W) \rangle = 0$;
4. a gradient step $w_k' = w_k - \eta \nabla_{w_k} f(W)$ gives $\lVert w_k' \rVert^2 = \lVert w_k \rVert^2 + \eta^2 \lVert \nabla_{w_k} f(W) \rVert^2 \ge \lVert w_k \rVert^2$.

### Assumptions

1. $\epsilon = 0$ in the normalization. Used in step 1.
2. $a > 0$, and each feature of $Wu$ has a positive variance over the batch. Used in step 1.
3. The loss depends on $W$ only through the normalized outputs, and $\ell$ is differentiable. Used in steps 2 and 3.
4. The step is plain gradient descent, with no weight decay and no momentum. Used in step 4.

### Steps

1. Feature $k$ of $(aW)u_i$ is $a\, w_k^\top u_i$. Its batch mean is $a$ times the batch mean of $w_k^\top u_i$, and its batch variance is $a^2$ times the batch variance. With $\epsilon = 0$ and $a > 0$, the standard deviation is $a$ times the old one. So the factor $a$ cancels in $(a z - a\mu)/(a\sigma)$, and item 1 holds.
2. By step 1, $f(aW) = f(W)$ for every $W$ and every $a > 0$. Differentiate both sides with respect to $W$. The chain rule on the left gives $a \nabla f(aW)$, and the right gives $\nabla f(W)$. Dividing by $a$ gives item 2.
3. Feature $k$ is normalized on its own, so step 1 holds when only row $w_k$ is scaled: $f$ is unchanged when $w_k$ becomes $(1 + t) w_k$ for $t > -1$. The derivative of $f$ along this path at $t = 0$ is $\langle \nabla_{w_k} f(W), w_k \rangle$, and it is zero because $f$ is constant on the path. This is item 3.
4. Expand the square: $\lVert w_k - \eta \nabla_{w_k} f \rVert^2 = \lVert w_k \rVert^2 - 2\eta \langle w_k, \nabla_{w_k} f \rangle + \eta^2 \lVert \nabla_{w_k} f \rVert^2$. The middle term is zero by step 3, which gives item 4.
5. Items 2 and 4 together explain the effective learning rate of van Laarhoven (2017, Section 3). The function only sees the direction $w_k / \lVert w_k \rVert$. By item 2 the gradient scales as $1/\lVert w_k \rVert$, so a step of size $\eta$ turns the direction by about $\eta \lVert \nabla f(\hat{w}_k) \rVert / \lVert w_k \rVert^2$, with $\hat{w}_k$ the unit vector. The effective learning rate is $\eta / \lVert w_k \rVert^2$. Weight decay shrinks $\lVert w_k \rVert$ and so raises it; without weight decay, item 4 makes the norm grow and the effective learning rate fall.

### Result and check

$$ \mathrm{BN}((aW)u) = \mathrm{BN}(Wu), \qquad \nabla f(aW) = \tfrac{1}{a}\nabla f(W), \qquad \langle w_k, \nabla_{w_k} f \rangle = 0 . $$

The check `scale-invariance.check.py` draws a batch, a weight matrix and several values of $a$ with a fixed seed, and checks item 1 to $10^{-12}$.
It computes $\nabla f$ by central finite differences for a nonlinear loss, and checks items 2 to 4 to a relative error under $10^{-6}$.
