---
id: parameter-and-flop-count
derives: [c-params, c-flops, c-sqrt-rule, c-weight-sharing]
kind: identity
assumes: ["the layers keep the grid at n by n with 'same' padding", "a multiply-add counts as 2 FLOPs; biases, ReLU and pooling are left out"]
uses: [split-filter-bank]
status: full
check: parameter-and-flop-count.check.py
lean: null
sources:
  - {work: cohen-2016-group-equivariant-convolutional-networks, locator: "Sections 7 and 8.1"}
---

### Statement

1. The model of the page has $P = C_1 C_0 k^2 + C_1 + 4 C_2 C_1 k^2 + C_2 + M C_2 + M$ parameters (eq-params) and costs $C_{\text{fwd}} = 2 B n^2 k^2 (4 C_1 C_0 + 16 C_2 C_1) + 2 B C_2 M$ FLOPs per forward pass (eq-flops).
2. A planar CNN with the same number of planar maps, $4 C_1$ and $4 C_2$, has 4 times as many convolution weights, $4 C_1 C_0 k^2 + 16 C_2 C_1 k^2$, and the same FLOPs: the G-CNN shares each weight across the 4 rotations.
3. With a stabilizer of size $|S|$, a group layer with $C$ channels in and out has $|S| C^2 k^2$ weights and costs $2 B n^2 k^2 |S|^2 C^2$ FLOPs. Matching the $C'^2 k^2$ weights of a planar layer with $C'$ channels takes $C = C' / \sqrt{|S|}$. The G-CNN then has $\sqrt{|S|}$ times as many planar maps and costs $|S|$ times the FLOPs: for p4, half the channels, twice the planar maps and 4 times the FLOPs.

### Assumptions

1. Every correlation pads to keep the grid $n \times n$, so it has $n^2$ outputs per planar map. Used in steps 1 and 3.
2. A multiply-add counts 2. Biases, ReLU, the max over rotations and the mean are left out, being linear in the size of the maps and not in $k^2$ or the channel counts. Used in steps 1 and 3.

### Steps

1. The lifting filters are $W_1 \in \mathbb{R}^{C_1 \times C_0 \times k \times k}$ with one bias per G-feature map, $C_1 C_0 k^2 + C_1$. The group filters are functions on p4 restricted to a $k \times k$ window, $W_2 \in \mathbb{R}^{C_2 \times C_1 \times 4 \times k \times k}$, plus $C_2$ biases. The head has $M C_2 + M$. Summing gives eq-params.
2. By split-filter-bank, the lifting layer is a planar correlation with $4 C_1$ outputs and $C_0$ inputs: $2 B n^2 k^2 \cdot 4 C_1 C_0$ FLOPs. The group layer is one with $4 C_2$ outputs and $4 C_1$ inputs: $2 B n^2 k^2 \cdot 16 C_2 C_1$. The head costs $2 B C_2 M$. Summing gives eq-flops.
3. A planar CNN with the same input and output plane counts runs the same correlations, so it costs the same FLOPs, but stores each of its $4 C_1 C_0 k^2 + 16 C_2 C_1 k^2$ weights once, where the G-CNN's bank holds 4 copies of each of $C_1 C_0 k^2 + 4 C_2 C_1 k^2$.
4. In general a group layer from $C$ to $C$ channels stores $C \cdot C \cdot |S| \cdot k^2$ weights and runs a planar correlation from $|S| C$ to $|S| C$ planes, $2 B n^2 k^2 |S|^2 C^2$ FLOPs. Setting $|S| C^2 = C'^2$ gives $C = C' / \sqrt{|S|}$, so the planar maps number $|S| C = \sqrt{|S|}\, C'$ and the FLOPs are $2 B n^2 k^2 |S| C'^2$, $|S|$ times the planar layer's $2 B n^2 k^2 C'^2$. Cohen and Welling divided the filters by $\sqrt 4 = 2$ in their rotated MNIST model to keep the parameter count about fixed.

### Result and check

$$ P = C_1 C_0 k^2 + C_1 + 4 C_2 C_1 k^2 + C_2 + M C_2 + M, \qquad C_{\text{fwd}} = 2 B n^2 k^2 (4 C_1 C_0 + 16 C_2 C_1) + 2 B C_2 M, \qquad C = C' / \sqrt{|S|}. $$

The check `parameter-and-flop-count.check.py` counts the arrays of the NumPy reference for several configs, counts the multiply-adds of its correlations with the bank, compares a planar CNN of the same plane counts, and checks the $\sqrt{|S|}$ rule for $|S| = 4$ and $|S| = 8$.
