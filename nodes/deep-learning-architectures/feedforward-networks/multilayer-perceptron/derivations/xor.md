---
id: xor
derives: [c-xor-linear, c-xor-lstsq, c-xor-network]
kind: proof
assumes: ["the inputs are the four points of {0, 1}^2 and the target is their XOR", "a linear classifier predicts class 1 where an affine function is positive"]
uses: []
status: full
check: xor.check.py
lean: null
sources: []
---

### Statement

1. No affine function $f(x) = w_1 x_1 + w_2 x_2 + c$ is positive on $(0, 1)$ and $(1, 0)$ and non-positive on $(0, 0)$ and $(1, 1)$, so no linear classifier computes XOR.
2. The least-squares affine fit to the four points of XOR is the constant $1/2$: $w_1 = w_2 = 0$ and $c = 1/2$.
3. The network $h = \operatorname{relu}(W_0 x + b_0)$, $y = W_1 h + b_1$ with
   $W_0 = \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$, $b_0 = (0, -1)$, $W_1 = (1, -2)$ and $b_1 = 0$
   computes XOR exactly on $\{0, 1\}^2$.

### Assumptions

1. The four inputs and targets are $(0,0) \mapsto 0$, $(0,1) \mapsto 1$, $(1,0) \mapsto 1$ and $(1,1) \mapsto 0$. Used in all steps.
2. A linear classifier says 1 where $f > 0$ and 0 where $f \le 0$. Used in step 1.

### Steps

1. Suppose $f$ separates the points. Then $f(0,1) + f(1,0) > 0$, since both terms are positive, and $f(0,0) + f(1,1) \le 0$, since both are non-positive. But both sums equal $w_1 + w_2 + 2c$. The contradiction proves statement 1.
2. Write the fit as least squares with the design matrix $A$ whose rows are $(x_1, x_2, 1)$ and the targets $t = (0, 1, 1, 0)$. The normal equations $A^\top A\, \theta = A^\top t$ read
   $\begin{pmatrix} 2 & 1 & 2 \\ 1 & 2 & 2 \\ 2 & 2 & 4 \end{pmatrix} \theta = (1, 1, 2)$.
   $A$ has full column rank, so the solution is unique, and $\theta = (0, 0, 1/2)$ satisfies all three rows: $2c = 1$, $2c = 1$, $4c = 2$. This is statement 2.
3. The pre-activations are $z = (x_1 + x_2,\ x_1 + x_2 - 1)$. On the four points $x_1 + x_2$ is $0, 1, 1, 2$, so $z$ is $(0, -1)$, $(1, 0)$, $(1, 0)$ and $(2, 1)$, and $h = \operatorname{relu}(z)$ is $(0, 0)$, $(1, 0)$, $(1, 0)$ and $(2, 1)$. Then $y = h_1 - 2 h_2$ is $0, 1, 1, 0$. This is statement 3.

### Result and check

$$ y = \operatorname{relu}(x_1 + x_2) - 2 \operatorname{relu}(x_1 + x_2 - 1) = x_1 \oplus x_2 \quad \text{on } \{0, 1\}^2 $$

The check `xor.check.py` solves the least-squares problem, searches a grid of affine functions for a separator and finds none, and runs the network of statement 3 through the NumPy reference.
