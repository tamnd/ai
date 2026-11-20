---
id: parameter-and-flop-count
derives: [c-params]
kind: count
assumes: ["only the products with W are counted, a multiply-add as 2 FLOPs", "the biases, the sigmoids, the softplus and the Bernoulli draws are left out"]
uses: [matrix-multiplication@m2]
status: full
check: parameter-and-flop-count.check.py
lean: null
sources:
  - {work: hinton-2002-training-products-experts-contrastive-divergence, locator: "p. 8", note: "the RBM with 256 visible and 500 hidden units on 16 x 16 digit images"}
  - {work: tieleman-2008-training-rbms-approximations-likelihood, locator: "p. 3", note: "RBMs with 500 hidden units on 784-pixel MNIST images"}
---

### Statement

An RBM with $W \in \mathbb{R}^{D \times H}$, $\mathbf{b} \in \mathbb{R}^D$ and $\mathbf{c} \in \mathbb{R}^H$ has

$$ P = DH + D + H $$

parameters. On $B$ vectors the up pass $\mathbf{v}W$ costs $2BDH$ FLOPs, the down pass $\mathbf{h}W^\top$ another $2BDH$, so one full block Gibbs step costs $C = 4BDH$. The free energy costs the up pass alone, $2BDH$. A CD-$k$ update costs the up pass of the data, $k$ Gibbs steps and the up pass of the last sample, $4BDH(k + 1)$ before the gradient products, and the exact partition function costs $2^D$ free energies, $2^{D+1}DH$.

### Assumptions

1. Only products with $W$ are counted, a multiply-add as 2. Used in steps 2 and 3.

### Steps

1. The parameters are the $DH$ entries of $W$, the $D$ visible biases and the $H$ hidden biases.
2. $\mathbf{v}W$ for $B$ rows of length $D$ and $H$ outputs is $BDH$ multiply-adds, $2BDH$ FLOPs, and $\mathbf{h}W^\top$ is the same.
3. A Gibbs step is one of each, $4BDH$. CD-$k$ also needs $p(\mathbf{h} \mid \mathbf{v})$ at the data and at the last sample, one up pass each, which gives $4BDHk + 4BDH = 4BDH(k + 1)$. The gradient products $\mathbf{v}^\top p(\mathbf{h} \mid \mathbf{v})$ add $2BDH$ each.
4. The exact partition function sums $e^{-F}$ over all $2^D$ visible vectors, $2^D$ up passes of one row each.

For $D = 784$, $H = 500$ the RBM has $392{,}000 + 784 + 500 = 393{,}284$ parameters and a Gibbs step costs $1{,}568{,}000$ FLOPs per vector; for Hinton's $D = 256$, $H = 500$ it has $128{,}756$.

### Result and check

$$ P = DH + D + H , \qquad C = 4BDH . $$

The check `parameter-and-flop-count.check.py` compares `param_count` and `flops_forward` with the formulas for several sizes, including the three configurations of the shape contract, and counts the multiply-adds of one Gibbs step by instrumenting the products.
