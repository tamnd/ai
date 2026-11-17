---
id: linear-collapse
derives: [c-collapse, c-collapse-rank]
kind: identity
assumes: ["every hidden layer uses the identity as its activation"]
uses: []
status: full
check: linear-collapse.check.py
lean: null
sources: []
---

### Statement

1. A multilayer perceptron whose activations are all the identity computes an affine map, $y = W x + b$, with
   $W = W_L W_{L-1} \cdots W_0$ and $b = \sum_{l=0}^{L} W_L \cdots W_{l+1}\, b_l$, where the empty product for $l = L$ is the identity.
2. The rank of $W$ is at most the smallest width $\min_l n_l$ of the network, so a linear network with a narrow layer is a low-rank linear map.

### Assumptions

1. The network is eq-layer with $\sigma(z) = z$ in each hidden layer, followed by eq-output. Used in step 1.
2. $W_l$ has the shape $n_{l+1} \times n_l$. Used in step 2.

### Steps

1. By induction on the number of layers. One layer is $W_0 x + b_0$, the claim for $L = 0$. If the first $l$ layers compute $h_l = M_l x + c_l$ with $M_l = W_{l-1} \cdots W_0$ and $c_l = \sum_{j<l} W_{l-1} \cdots W_{j+1} b_j$, then the next one gives $W_l h_l + b_l = (W_l M_l)\, x + (W_l c_l + b_l)$, which has the same form with $l + 1$. This is statement 1.
2. The rank of a product is at most the rank of each factor, and the rank of $W_l$ is at most $\min(n_l, n_{l+1})$. Hence $\operatorname{rank} W \le \min_l n_l$. This is statement 2.

### Result and check

$$ W_L \big( \cdots W_1 (W_0 x + b_0) + b_1 \cdots \big) + b_L = (W_L \cdots W_0)\, x + \sum_{l=0}^{L} W_L \cdots W_{l+1}\, b_l $$

The check `linear-collapse.check.py` runs random identity-activation networks of several depths through the NumPy reference and compares them with the collapsed map, and checks the rank bound with a bottleneck of width 1 and 2.
