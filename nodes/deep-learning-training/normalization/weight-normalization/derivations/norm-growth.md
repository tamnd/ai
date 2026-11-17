---
id: norm-growth
derives: [c-norm-growth, c-scale-step, c-momentum-breaks, eq-norm-growth]
kind: identity
assumes: ["the update is a plain gradient step v' = v - eta grad_v, without momentum or per-parameter step sizes", "v is not zero"]
uses: [orthogonal-projection@m2, stochastic-gradient-descent@m2]
status: full
check: norm-growth.check.py
lean: null
sources:
  - {work: salimans-2016-weight-normalization, locator: "p. 3"}
---

### Statement

Let $v \neq 0$ and let $v' = v - \eta\nabla_v L$ for a step size $\eta \in \mathbb{R}$. Then

1. $\lVert v'\rVert^2 = \lVert v\rVert^2 + \eta^2\lVert\nabla_v L\rVert^2$, so $\lVert v'\rVert \geq \lVert v\rVert$;
2. with $c = \eta\lVert\nabla_v L\rVert/\lVert v\rVert$, $\lVert v'\rVert = \sqrt{1 + c^2}\,\lVert v\rVert$;
3. at fixed $w$ and $\nabla_w L$, replacing $v$ by $s v$ for $s > 0$ replaces $\nabla_v L$ by $\nabla_v L/s$.

### Assumptions

1. The step is plain gradient descent. With momentum or Adam the step is not $-\eta\nabla_v L$, and item 1 fails. Used in step 1.
2. $v \cdot \nabla_v L = 0$, item 4 of the derivation `gradient`. Used in step 1.
3. $v \neq 0$. Used in step 2.

### Steps

1. Expand: $\lVert v - \eta\nabla_v L\rVert^2 = \lVert v\rVert^2 - 2\eta\,v\cdot\nabla_v L + \eta^2\lVert\nabla_v L\rVert^2$. The middle term is 0 by assumption 2. This is item 1, the theorem of Pythagoras for the right angle between $v$ and the step.
2. Factor $\lVert v\rVert^2$ out of item 1: $\lVert v'\rVert^2 = \lVert v\rVert^2(1 + c^2)$. This is item 2, the form of Salimans and Kingma.
3. Item 3 of the derivation `gradient` gives $\nabla_v L = \frac{g}{\lVert v\rVert}M_w\nabla_w L$. $M_w$ depends on $w$ only, and $w$ does not change when $v$ is scaled with $g$ fixed. So only the factor $1/\lVert v\rVert$ changes, and it becomes $1/(s\lVert v\rVert)$. This is item 3.

### Result and check

$$ \lVert v'\rVert^2 = \lVert v\rVert^2 + \eta^2\lVert\nabla_v L\rVert^2 . $$

So under plain gradient descent $\lVert v\rVert$ never decreases. A larger $\lVert v\rVert$ gives a smaller $\nabla_v L$, and the turn of the direction $v/\lVert v\rVert$ in one step is about $\eta\lVert\nabla_v L\rVert/\lVert v\rVert$, which falls like $1/\lVert v\rVert^2$. A large step size makes $\lVert v\rVert$ grow faster, which damps the later steps.
The check `norm-growth.check.py` runs 400 steps of gradient descent on a least-squares loss for three step sizes, checks item 1 at each step to a relative tolerance of $10^{-12}$ and that the norm never falls, checks item 2 on the worked example, where $\lVert\nabla_v L\rVert = 0.32$, and checks item 3 for three scales. It also runs heavy-ball momentum and finds steps that are not orthogonal to $v$, so the identity does not hold there.
