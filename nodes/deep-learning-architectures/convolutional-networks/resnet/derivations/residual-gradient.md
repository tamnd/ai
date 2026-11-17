---
id: residual-gradient
derives: [c-propagation, c-scaled-shortcut, c-paths, eq-residual-unroll, eq-residual-grad]
kind: gradient
assumes: ["x_{i+1} = λ_i x_i + F_i(x_i) for every unit i from l to L - 1, with λ_i = 1 unless stated", "each F_i is differentiable, with Jacobian J_i at x_i", "E is a scalar loss of x_L"]
uses: [backpropagation@al, matrix-multiplication@m2, binomial-distribution@m2]
status: full
check: residual-gradient.check.py
lean: null
sources:
  - {work: he-2016-identity-mappings, locator: "Section 2"}
  - {work: he-2016-identity-mappings, locator: "Section 3"}
  - {work: veit-2016-residual-networks-ensembles, locator: "Section 3"}
---

### Statement

1. If $x_{i+1} = x_i + F_i(x_i)$ for $i = l, \dots, L-1$, then $x_L = x_l + \sum_{i=l}^{L-1} F_i(x_i)$ and
$$ \frac{\partial E}{\partial x_l} = \frac{\partial E}{\partial x_L} \prod_{i=l}^{L-1} (I + J_i) = \frac{\partial E}{\partial x_L} \Big( I + \frac{\partial}{\partial x_l} \sum_{i=l}^{L-1} F_i(x_i) \Big), $$
where $J_i = \partial F_i / \partial x_i$ and the gradient is a row vector. The first term carries $\partial E / \partial x_L$ to $x_l$ unchanged.
2. If $x_{i+1} = \lambda_i x_i + F_i(x_i)$ with scalars $\lambda_i$, the term that does not go through any $F_i$ is $\big(\prod_{i=l}^{L-1} \lambda_i\big)\, \partial E / \partial x_L$.
3. The product $\prod_{i=1}^{n} (I + J_i)$ expands into $2^n$ terms, one for each subset $S$ of the units, the ordered product of the $J_i$ with $i \in S$; $\binom{n}{k}$ of them have $|S| = k$.

### Assumptions

1. Each unit computes $x_{i+1} = \lambda_i x_i + F_i(x_i)$, with $\lambda_i = 1$ in statements 1 and 3: an identity shortcut and no function after the addition, as in the pre-activation unit of He et al. (2016, Identity Mappings, Section 2). Used in all steps.
2. Each $F_i$ is differentiable at $x_i$, with Jacobian $J_i$. Used in steps 3 to 6.
3. $E$ is a scalar function of $x_L$; its gradient is a row vector. Used in steps 3 and 5.

### Steps

1. By assumption 1 with $\lambda_i = 1$, $x_{l+1} = x_l + F_l(x_l)$ and $x_{l+2} = x_{l+1} + F_{l+1}(x_{l+1}) = x_l + F_l(x_l) + F_{l+1}(x_{l+1})$.
2. By induction on $L$: if $x_L = x_l + \sum_{i=l}^{L-1} F_i(x_i)$, then $x_{L+1} = x_L + F_L(x_L) = x_l + \sum_{i=l}^{L} F_i(x_i)$. This gives
$$ x_L = x_l + \sum_{i=l}^{L-1} F_i(x_i) . $$
3. Differentiating step 2 with respect to $x_l$ and applying the chain rule to the scalar $E$ (assumptions 2 and 3):
$$ \frac{\partial E}{\partial x_l} = \frac{\partial E}{\partial x_L}\,\frac{\partial x_L}{\partial x_l} = \frac{\partial E}{\partial x_L} \Big( I + \frac{\partial}{\partial x_l} \sum_{i=l}^{L-1} F_i(x_i) \Big) . $$
The gradient at $x_l$ is $\partial E / \partial x_L$ itself plus a term through the residual functions. It is zero only if that term cancels $\partial E / \partial x_L$ exactly, which in general does not happen for all the examples of a mini-batch; and when the weights are small the second term is small, so the gradient is close to $\partial E / \partial x_L$ rather than close to zero. This is the argument of He et al. (2016, Identity Mappings, Section 2).
4. The same Jacobian also follows unit by unit: $\partial x_{i+1} / \partial x_i = I + J_i$, so $\partial x_L / \partial x_l = (I + J_{L-1}) \cdots (I + J_l)$, written in the order of the backward pass.
5. With $\lambda_i \ne 1$, step 4 gives $\partial x_{i+1} / \partial x_i = \lambda_i I + J_i$. Expanding the product, the only term without a $J$ is $\prod_i \lambda_i\, I$. If every $\lambda_i > 1$ it grows exponentially in $L - l$; if every $\lambda_i < 1$ it shrinks exponentially: the shortcut then no longer carries the gradient unchanged (Section 3 of the same paper).
6. Expand $\prod_{i=1}^{n} (I + J_i)$ by distributing each factor: from factor $i$ pick either $I$ or $J_i$. The $2^n$ ways of picking are the subsets $S$ of $\{1, \dots, n\}$, and the term of $S$ is the product of the $J_i$, $i \in S$, in order. There are $\binom{n}{k}$ subsets with $k$ elements, and $\sum_k \binom{n}{k} = 2^n$. When the $F_i$ are linear, the same expansion holds for the forward map: $x_{n+1} = \prod_i (I + W_i)\, x_1$ is a sum of $2^n$ paths, each of which applies $|S|$ residual functions. Veit et al. (2016, Section 3) call this the unraveled view; the path lengths follow the binomial distribution with mean $n/2$.

### Result and check

$$ \frac{\partial E}{\partial x_l} = \frac{\partial E}{\partial x_L} \Big( I + \frac{\partial}{\partial x_l} \sum_{i=l}^{L-1} F_i(x_i) \Big), \qquad \#\{\text{paths of length } k\} = \binom{n}{k} . $$

The check `residual-gradient.check.py` builds stacks of random nonlinear residual units, compares the unrolled sum with the forward pass and the product of the $I + J_i$ with a finite-difference Jacobian, expands the product over all subsets for linear units, and measures the scaled direct term for $\lambda = 0.9$ and $1.1$.
