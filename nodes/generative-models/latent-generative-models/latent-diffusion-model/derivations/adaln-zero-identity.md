---
id: adaln-zero-identity
derives: [c-adaln-identity]
kind: gradient
assumes: ["alpha, gamma and beta are produced by a linear layer of the condition whose alpha rows are zero", "F is differentiable"]
uses: [backpropagation@al]
status: full
check: adaln-zero-identity.check.py
lean: null
sources:
  - {work: peebles-2022-scalable-diffusion-models-transformers, locator: "p. 5", note: "adaLN-Zero: the MLP is initialized to output zero for every alpha, which makes the DiT block the identity"}
---

### Statement

Let a block map $x \in \mathbb{R}^D$ and a condition $c \in \mathbb{R}^C$ to

$$ y = x + \alpha \odot F\big((1 + \gamma) \odot \mathrm{LN}(x) + \beta;\ \phi\big) , \qquad (\gamma, \beta, \alpha) = c\,(W_\gamma, W_\beta, W_\alpha) + (b_\gamma, b_\beta, b_\alpha) , $$

with $F$ differentiable in its input and its parameters $\phi$. If $W_\alpha = 0$ and $b_\alpha = 0$, then $y = x$ for every $x$ and $c$, and for any loss $L(y)$ with $g = \partial L/\partial y$,

$$ \frac{\partial L}{\partial W_\alpha} = c^\top (g \odot F) , \qquad \frac{\partial L}{\partial \phi} = 0 , \qquad \frac{\partial L}{\partial W_\gamma} = \frac{\partial L}{\partial W_\beta} = 0 , \qquad \frac{\partial L}{\partial x} = g . $$

### Assumptions

1. The gate $\alpha$ is a linear function of $c$ with zero weights and bias at initialization. Used in steps 1 to 3.
2. $F$ is differentiable. Used in steps 2 and 3.

### Steps

1. With $\alpha = 0$, $y = x + 0 \odot F(\cdot) = x$: the block is the identity whatever $\phi$, $\gamma$ and $\beta$ are.
2. $\partial y/\partial \alpha = \mathrm{diag}(F)$, so $\partial L/\partial \alpha = g \odot F$ and $\partial L/\partial W_\alpha = c^\top(g \odot F)$, non-zero in general.
3. Every other parameter acts on $y$ only through $F$, whose contribution is multiplied by $\alpha = 0$: $\partial y/\partial \phi = \alpha \odot \partial F/\partial \phi = 0$, and likewise for $\gamma$, $\beta$ and their weights. The input reaches $y$ directly, so $\partial L/\partial x = g$, plus a term through $F$ that is again multiplied by $\alpha$.

### Result and check

A deep stack of adaLN-Zero blocks starts as the identity, so the network's output at initialization is its final layer on the embedded input and the gradient reaches every layer unattenuated through the residual path. The first update moves only the gates; once they are non-zero the branches start to learn. This is a stronger form of zero-initializing the last layer of each residual branch.
The check `adaln-zero-identity.check.py` builds a block with $F$ a two-layer MLP, sets the $\alpha$ rows to zero with all other parameters random, and compares the output with $x$ and the analytic gradients with central finite differences of a random quadratic loss in every parameter.
