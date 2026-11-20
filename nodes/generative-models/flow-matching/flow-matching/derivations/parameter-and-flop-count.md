---
id: parameter-and-flop-count
derives: [c-params]
kind: closed-form
assumes: ["the velocity network of the page: one tanh hidden layer of H units on D data coordinates and E time features", "only the matrix products are counted, a multiply-add as 2 FLOPs"]
uses: [matrix-multiplication@m2]
status: full
check: parameter-and-flop-count.check.py
lean: null
sources:
  - {work: lipman-2022-flow-matching-generative-modeling, locator: "p. 7", note: "sampling solves the learned ODE with a numerical solver, whose cost is counted in function evaluations"}
---

### Statement

Take the velocity network $v_\theta(\mathbf{x}, t) = \tanh([\mathbf{x}, e(t)]W_1 + b_1)W_2 + b_2$, with $W_1 \in \mathbb{R}^{(D + E) \times H}$ and $W_2 \in \mathbb{R}^{H \times D}$. It has

$$ (D + E + 1)H + (H + 1)D $$

parameters. One evaluation on $B$ points costs $2BH(2D + E)$ FLOPs. An Euler sample with $n$ steps costs $n$ evaluations, and a midpoint sample with $n$ steps costs $2n$.

### Assumptions

1. Biases, the tanh and the time features are not counted. They cost $O(B(H + D + E))$, against $O(BH(D + E))$ for the products. Used in step 2.

### Steps

1. Count the parameters: $W_1$ has $(D + E)H$ entries and $b_1$ has $H$; $W_2$ has $HD$ and $b_2$ has $D$. Their sum is $(D + E + 1)H + (H + 1)D$.
2. Count the FLOPs: the first product costs $2B(D + E)H$ and the second $2BHD$. Their sum is $2BH(2D + E)$.
3. Count the evaluations: each Euler step evaluates $v$ once. Each midpoint step evaluates it twice, once at the start and once at the midpoint.

### Result and check

$$ |\theta| = (D + E + 1)H + (H + 1)D, \qquad \text{FLOPs} = 2BH(2D + E) \text{ per evaluation} . $$

For the example configuration $D = 3$, $H = 5$, $E = 4$, the network has 58 parameters and costs 100 FLOPs per point.

The check `parameter-and-flop-count.check.py` compares `param_count` and `flops_forward` with the formulas for three configurations. It counts the calls the samplers make to a velocity function, and it confirms that both samplers solve $\dot{\mathbf{x}} = -\mathbf{x}$.
