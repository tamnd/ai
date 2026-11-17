---
id: parameter-and-flop-count
derives: [c-params, c-flops, c-backward-cost, c-memory]
kind: count
assumes: ["a multiply-add counts as 2 FLOPs, and only the matrix products are counted", "a fixed-step solver with S evaluations of f per step: S = 1 for Euler and S = 4 for the classical Runge-Kutta method"]
uses: [adjoint]
status: full
check: parameter-and-flop-count.check.py
lean: null
sources:
  - {work: chen-2018-neural-ordinary-differential-equations, locator: "Section 3"}
---

### Statement

For the field eq-field of width $D$ and hidden width $H$, a batch of $B$ and $N$ steps:

1. The model has $P = 2 D H + 2 H + D$ parameters (eq-params), the same for every $N$.
2. The forward solve makes $S N$ evaluations of $f$ and costs $4 S N B D H$ FLOPs (eq-flops).
3. The adjoint backward pass, which evaluates $f$ and its two vector-Jacobian products from one shared pre-activation, costs three times the forward solve.
4. The adjoint backward pass holds a state of $2 B D + P$ numbers; backpropagation through the stages holds $S N B D$.

### Assumptions

1. Each evaluation computes $s = \tanh(h W_1^\top + t u + b_1)$ and $f = s W_2^\top + b_2$. Used in steps 2 and 3.

### Steps

1. $W_1$ has $H D$ entries, $u$ and $b_1$ have $H$ each, $W_2$ has $D H$ and $b_2$ has $D$, which sum to $2 D H + 2 H + D$. Every step uses the same parameters, so $N$ does not enter.
2. An evaluation makes two products of $B D H$ multiply-adds, $h W_1^\top$ and $s W_2^\top$, which is $4 B D H$ FLOPs, and the solver makes $S N$ evaluations.
3. One evaluation of the right-hand side of the adjoint system needs $f$ (two products) and four more: $a W_2$ and $g W_1$ for $a\, \partial f / \partial h$, with $g = (a W_2) \odot (1 - s^2)$, and $g^\top h$ and $a^\top s$ for the gradients of $W_1$ and $W_2$; the gradients of $u$, $b_1$ and $b_2$ are sums. That is six products, three times the two of the forward pass, and the backward solve makes the same $S N$ evaluations.
4. The adjoint state is $h$ and $a$, $B D$ numbers each, and the running gradient, $P$ numbers (step 3 of the derivation adjoint). Backpropagation keeps the input of every stage, $S N$ arrays of $B D$ numbers.

### Result and check

$$ P = 2 D H + 2 H + D, \qquad C_\text{fwd} = 4 S N B D H, \qquad C_\text{adjoint} = 3 C_\text{fwd} $$

The check `parameter-and-flop-count.check.py` counts the parameters of the NumPy reference and the multiply-adds of its forward and adjoint passes with an array type that counts its matrix products, and compares the four counts with those of `shapes.yaml` for every config; the adjoint check records the size of the state of the backward solve.
