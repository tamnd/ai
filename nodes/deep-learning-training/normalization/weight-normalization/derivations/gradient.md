---
id: gradient
derives: [c-grad, c-proj, c-grad-orth, eq-grad, eq-proj]
kind: gradient
assumes: ["v is not the zero vector", "the loss is differentiable in w", "g is a scalar per weight vector"]
uses: [backpropagation@al, orthogonal-projection@m2]
status: full
check: gradient.check.py
lean: null
sources:
  - {work: salimans-2016-weight-normalization, locator: "p. 3"}
---

### Statement

Let $v \in \mathbb{R}^k$ with $v \neq 0$, let $g \in \mathbb{R}$, and let $w = g\,v/\lVert v\rVert$. Let $L$ be a loss that is differentiable in $w$, with gradient $\nabla_w L$. Then

1. $\nabla_g L = \dfrac{\nabla_w L \cdot v}{\lVert v\rVert}$;
2. $\nabla_v L = \dfrac{g}{\lVert v\rVert}\nabla_w L - \dfrac{g\,\nabla_g L}{\lVert v\rVert^2}\,v$;
3. $\nabla_v L = \dfrac{g}{\lVert v\rVert} M_w \nabla_w L$ with $M_w = I - \dfrac{w w^\top}{\lVert w\rVert^2}$, when $g \neq 0$;
4. $v \cdot \nabla_v L = 0$.

### Assumptions

1. $v \neq 0$, so $\lVert v\rVert$ is differentiable at $v$. Used in steps 1 and 2.
2. $L$ depends on $g$ and $v$ only through $w$. Used in every step.
3. $g \neq 0$ in item 3, so that $w \neq 0$ and $M_w$ is defined. Item 2 and item 4 hold for $g = 0$ too.

### Steps

1. Write $n = \lVert v\rVert$ and $\hat{v} = v/n$. Then $w = g\hat{v}$, so $\partial w/\partial g = \hat{v}$, and the chain rule gives $\nabla_g L = \nabla_w L \cdot \hat{v} = \nabla_w L \cdot v/n$. This is item 1.
2. $\partial n/\partial v = v/n$, so the Jacobian of $\hat{v}$ is $\partial\hat{v}/\partial v = \frac{1}{n}\big(I - \hat{v}\hat{v}^\top\big)$. It is symmetric. The Jacobian of $w$ is $g$ times it, so the chain rule gives $\nabla_v L = \frac{g}{n}\big(I - \hat{v}\hat{v}^\top\big)\nabla_w L = \frac{g}{n}\nabla_w L - \frac{g}{n}\,(\hat{v}\cdot\nabla_w L)\,\hat{v}$.
3. By step 1, $\hat{v}\cdot\nabla_w L = \nabla_g L$. So the last term is $\frac{g\nabla_g L}{n}\hat{v} = \frac{g\nabla_g L}{n^2}v$. This is item 2.
4. When $g \neq 0$, $w/\lVert w\rVert = \pm\hat{v}$, so $w w^\top/\lVert w\rVert^2 = \hat{v}\hat{v}^\top$ and $M_w = I - \hat{v}\hat{v}^\top$. Step 2 is then item 3.
5. $I - \hat{v}\hat{v}^\top$ maps every vector to one orthogonal to $\hat{v}$: $\hat{v}^\top(I - \hat{v}\hat{v}^\top)z = \hat{v}^\top z - \hat{v}^\top z = 0$. So $v\cdot\nabla_v L = 0$. This is item 4.

### Result and check

$$ \nabla_g L = \frac{\nabla_w L\cdot v}{\lVert v\rVert}, \qquad \nabla_v L = \frac{g}{\lVert v\rVert}M_w\nabla_w L, \qquad v\cdot\nabla_v L = 0 . $$

The output does not change when $v$ is scaled, so the loss has no slope along $v$. That is item 4 seen another way.
The check `gradient.check.py` compares items 1 and 2 with central finite differences for 100 random cases of sizes 1 to 8, with a loss that is not quadratic. It checks item 3 and item 4 in each case, and the worked example $\nabla_v L = (0.256, -0.192)$.
