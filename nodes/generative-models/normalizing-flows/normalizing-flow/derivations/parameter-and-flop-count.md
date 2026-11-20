---
id: parameter-and-flop-count
derives: [c-params]
kind: closed-form
assumes: ["the coupling flow of the shape contract: L layers, each with one tanh hidden layer of H units on the D masked inputs and two output layers for s and t", "only the products are counted, a multiply-add as 2 FLOPs"]
uses: [matrix-multiplication@m2]
status: full
check: parameter-and-flop-count.check.py
lean: null
sources:
  - {work: dinh-2016-density-estimation-real-nvp, locator: "p. 5", note: "eq. (8): sampling runs the coupling layers in reverse at the cost of inference"}
---

### Statement

1. A flow of $L$ coupling layers on $D$ dimensions, each computing $s$ and $t$ from one hidden layer of $H$ units, has $P = L(3DH + H + 2D)$ parameters.
2. A density pass on $B$ points costs $C = 6LBDH$ FLOPs in its products, and a sampling pass costs the same.
3. The example config ($D = 3$, $H = 5$, $L = 3$, $B = 4$) has $P = 168$ and $C = 1{,}080$; the config `banana` ($D = 2$, $H = 32$, $L = 4$, $B = 256$) has $P = 912$ and $C = 393{,}216$.

### Assumptions

1. Each layer is $\mathbf{h} W + \mathbf{b}$ with $W$ of shape $(n_{\text{in}}, n_{\text{out}})$; the hidden layer reads all $D$ inputs, masked, and the output layers write all $D$ outputs, masked. Used in steps 1 and 2.
2. A product of a $(B, n)$ and an $(n, m)$ matrix costs $2Bnm$ FLOPs. Used in step 2.

### Steps

1. Per layer: $W_1$ and $\mathbf{b}_1$ have $DH + H$ entries, $W_s, \mathbf{b}_s$ and $W_t, \mathbf{b}_t$ have $HD + D$ each, in all $3DH + H + 2D$; times $L$ is statement 1.
2. Per layer the products are $(B, D)(D, H)$ and twice $(B, H)(H, D)$: $2BDH + 4BHD = 6BDH$; times $L$ is statement 2. The inverse computes $s$ and $t$ from the kept coordinates with the same three products.
3. $3(45 + 5 + 6) = 168$ and $6 \cdot 3 \cdot 4 \cdot 3 \cdot 5 = 1{,}080$; $4(192 + 32 + 4) = 912$ and $6 \cdot 4 \cdot 256 \cdot 2 \cdot 32 = 393{,}216$. This is statement 3.

### Result and check

$$ P = L(3DH + H + 2D), \qquad C = 6LBDH . $$

The check `parameter-and-flop-count.check.py` compares $P$ with the sizes of the arrays of the NumPy reference for random sizes, counts the products of a density pass and of a sampling pass with an instrumented array, and recomputes statement 3. A masked autoregressive layer costs as much as a coupling layer to evaluate a density but $D$ times as much to sample, one pass per coordinate.
