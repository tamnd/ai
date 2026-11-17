---
id: parameter-and-flop-count
derives: [c-params, c-flops, c-width-depth-params]
kind: count
assumes: ["every layer is fully connected with a bias", "a multiply-add counts as 2 FLOPs, and the biases and activations are left out of the FLOPs"]
uses: []
status: full
check: parameter-and-flop-count.check.py
lean: null
sources: []
---

### Statement

1. A multilayer perceptron with widths $n_0, n_1, \dots, n_{L+1}$, from the input to the output, has
   $P = \sum_{l=0}^{L} (n_l n_{l+1} + n_{l+1})$ parameters.
2. Its forward pass over a batch of $B$ examples costs $C_\text{fwd} = 2 B \sum_{l=0}^{L} n_l n_{l+1}$ FLOPs.
3. With $L$ hidden layers of width $d$, input width $n$ and output width $k$, $P = (n + 1) d + (L - 1)(d + 1) d + (d + 1) k$: the count grows linearly in the depth and quadratically in the width. Doubling $d$ roughly quadruples the hidden part, doubling $L$ roughly doubles it.

### Assumptions

1. Layer $l$ is $z = W_l h + b_l$ with $W_l$ of shape $n_{l+1} \times n_l$ and $b_l$ of length $n_{l+1}$ (eq-layer, eq-output). Used in steps 1 and 2.
2. A matrix product of an $m \times n$ and an $n \times p$ matrix makes $m n p$ multiply-adds. Used in step 2.

### Steps

1. Layer $l$ has $n_{l+1} n_l$ weights and $n_{l+1}$ biases. Summing over the $L + 1$ layers gives statement 1.
2. Over a batch the layer computes $H W_l^\top$ with $H$ of shape $B \times n_l$, which by assumption 2 makes $B n_l n_{l+1}$ multiply-adds, $2 B n_l n_{l+1}$ FLOPs. Summing gives statement 2.
3. Put $n_0 = n$, $n_1 = \dots = n_L = d$ and $n_{L+1} = k$ in statement 1: the first layer contributes $n d + d$, each of the $L - 1$ hidden-to-hidden layers $d^2 + d$, and the output layer $d k + k$. This is statement 3.

### Result and check

$$ P = \sum_{l=0}^{L} (n_l n_{l+1} + n_{l+1}), \qquad C_\text{fwd} = 2 B \sum_{l=0}^{L} n_l n_{l+1} $$

The check `parameter-and-flop-count.check.py` counts the entries of the arrays that the NumPy reference draws, counts the multiply-adds of its forward pass by instrumenting the matrix products, and compares both with the formulas and with the shape contract.
