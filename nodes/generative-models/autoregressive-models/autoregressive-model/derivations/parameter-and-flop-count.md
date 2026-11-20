---
id: parameter-and-flop-count
derives: [c-params]
kind: closed-form
assumes: ["the MADE of the page: D inputs, one masked hidden layer of H units and a masked direct connection", "only the matrix products are counted, a multiply-add as 2 FLOPs", "the masks are applied to the weights once, before the pass, and their zeros are counted"]
uses: [matrix-multiplication@m2, hadamard-product@m2]
status: full
check: parameter-and-flop-count.check.py
lean: null
sources:
  - {work: germain-2015-made-masked-autoencoder, locator: "p. 6", note: "Table 1: the cost of an exact test negative log-likelihood, O(DK + D^2) for a one-layer MADE against O(DK) for NADE, and one pass against D"}
---

### Statement

A MADE with $W \in \mathbb{R}^{D \times H}$, $\mathbf{b} \in \mathbb{R}^H$, $V \in \mathbb{R}^{H \times D}$, $\mathbf{c} \in \mathbb{R}^D$ and $A \in \mathbb{R}^{D \times D}$ has

$$ P = DH + H + HD + D + D^2 = 2DH + H + D + D^2 $$

parameters, masked entries included. One pass on $B$ vectors costs $C = 2B(2DH + D^2)$ FLOPs and gives all $D$ conditionals. Ancestral sampling needs $D$ passes, $DC$ FLOPs for $B$ samples. NADE with $H$ hidden units computes its $D$ conditionals in $O(BDH)$ through its running sum, and MADE in $O(B(DH + D^2))$.

### Assumptions

1. Masked weights still exist as parameters; they are zero in every pass. About half of $W$ and $V$, and a little under half of $A$, are masked, so the effective count is lower. Used in step 1.
2. Biases, the activation, the sigmoid and the loss cost $O(B(D + H))$ and are not counted. Used in step 2.

### Steps

1. Count the parameters: $DH$ and $H$ for the hidden layer, $HD$ for the output weights, $D$ for the output biases and $D^2$ for the direct weights.
2. Count the FLOPs: the products $\mathbf{x}(W \odot M^W)$, $\mathbf{h}(V \odot M^V)$ and $\mathbf{x}(A \odot M^A)$ cost $2BDH$, $2BHD$ and $2BD^2$.
3. A sample cannot be drawn in one pass, since $x_d$ must be known before the logits that depend on it are computed. Each of the $D$ variables in the ordering takes one pass, so sampling costs $D$ passes.
4. Without the direct connection, drop $D^2$ from both counts. With $L$ hidden layers of widths $H_1, \dots, H_L$, the products cost $2B(DH_1 + \sum_l H_l H_{l+1} + H_L D)$.

### Result and check

$$ P = 2DH + H + D + D^2, \qquad C = 2B(2DH + D^2) . $$

For the example $D = 4$, $H = 6$, the MADE has 74 parameters and costs 128 FLOPs per vector. A one-layer MADE on binarized MNIST, $D = 784$ and $H = 500$, has 1,399,940 parameters and costs about 2.80 MFLOPs per image.

The check `parameter-and-flop-count.check.py` compares `param_count` and `flops_forward` with the formulas for four configurations, deep and shallow, with and without the direct connection. It counts the masked fraction of the example. It also counts the passes of `sample` by wrapping `logits`.
