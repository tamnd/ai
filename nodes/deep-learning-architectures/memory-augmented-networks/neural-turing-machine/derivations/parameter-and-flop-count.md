---
id: parameter-and-flop-count
derives: [c-params, c-flops, c-params-independent-of-n, c-learned-init-grows]
kind: identity
assumes: ["a feedforward controller with one tanh layer and a linear output layer", "the memory, the weightings and the read vectors start at fixed values", "a multiply-add counts as 2 FLOPs; elementwise work is left out"]
uses: []
status: full
check: parameter-and-flop-count.check.py
lean: null
sources:
  - {work: graves-2014-neural-turing-machines, locator: "Section 4.6"}
  - {work: collier-2018-implementing-neural-turing-machines, locator: "Section 3"}
---

### Statement

1. With $K = 2S + 1$ shifts, a read head takes $M + 3 + K$ outputs of the controller and a write head $3M + 3 + K$, so the controller emits $P_h = R(M + 3 + K) + W(3M + 3 + K)$ head outputs, and the model has
   $P = H(X + RM) + H + P_h (H + 1) + Y(H + RM) + Y$ parameters (eq-params).
2. A forward pass over $T$ steps of a batch of $B$ costs
   $C_{\text{fwd}} = 2BT\,[\,H(X + RM) + P_h H + Y(H + RM) + (R + W) N (M + K) + R N M\,]$ FLOPs (eq-flops).
3. $P$ does not depend on $N$, while $C_{\text{fwd}}$ grows linearly in $N$: a larger memory costs compute and storage, $BNM$ numbers, but no parameters.
4. Learning the initial memory, read vectors and weightings, as some implementations do, adds $NM + RM + (R + W) N$ parameters, which grow with $N$.

### Assumptions

1. The controller is a tanh layer on $[x_t, r^1_{t-1}, \dots, r^R_{t-1}]$, a linear map from it to the head outputs, and a linear output layer on $[h_t, r^1_t, \dots, r^R_t]$, each with a bias. Used in steps 1 and 2.
2. $M_0$, $w_0$ and $r_0$ are constants, not parameters. Used in steps 1 and 4.
3. A multiply-add counts 2 FLOPs. The norms, the softmaxes, the powers, the erase and the add are elementwise or linear in the size of their arrays without a product of two sizes, and are left out. Used in step 2.

### Steps

1. A head emits a key of $M$ numbers, then $\beta$, $g$ and $\gamma$, one each, and a shift weighting of $K$: $M + 3 + K$. A write head also emits an erase and an add vector, $2M$ more. The tanh layer has $H(X + RM)$ weights and $H$ biases; the head map $P_h H$ weights and $P_h$ biases; the output layer $Y(H + RM)$ weights and $Y$ biases. Summing gives eq-params.
2. Per step and per sequence the three linear maps cost $H(X + RM) + P_h H + Y(H + RM)$ multiply-adds. Each head computes $N$ inner products of length $M$ for the cosine (eq-cosine) and the circular convolution, $K$ products for each of the $N$ locations (eq-shift): $N(M + K)$. Each read head forms $\sum_i w(i) M(i)$, $NM$ (eq-read). Multiplying by 2, $B$ and $T$ gives eq-flops.
3. No term of eq-params contains $N$. The terms $(R + W) N (M + K) + R N M$ of eq-flops are linear in $N$, and the memory of the batch holds $BNM$ numbers. Graves et al. make the same point for their models.
4. A learned $M_0$ has $NM$ entries, a learned $r_0$ one vector of $M$ per read head, and a learned $w_0$ one weighting of $N$ per head, $RM + (R + W)N$.

### Result and check

$$ P = H(X + RM) + H + P_h(H + 1) + Y(H + RM) + Y, \qquad C_{\text{fwd}} = 2BT\,[\,H(X + RM) + P_h H + Y(H + RM) + (R + W) N (M + K) + R N M\,]. $$

The check `parameter-and-flop-count.check.py` counts the arrays of the NumPy reference for several configs and several $N$, counts the multiply-adds of every product of one forward pass by wrapping the einsums and the matrix products, and compares both with the formulas.
