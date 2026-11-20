---
id: parameter-and-flop-count
derives: [c-params]
kind: count
assumes: ["only the two matrix products are counted, a multiply-add as 2 FLOPs", "the biases, the tanh, the one-hot encoding and the softmax are left out"]
uses: [matrix-multiplication@m2]
status: full
check: parameter-and-flop-count.check.py
lean: null
sources:
  - {work: austin-2021-structured-denoising-diffusion-discrete, locator: "p. 6", note: "text8 in chunks of 256 characters over 27 symbols, the shape of the text8_chunk configuration"}
---

### Statement

The denoiser with $W_1 \in \mathbb{R}^{L(K+1) \times H}$, $\mathbf{b}_1 \in \mathbb{R}^H$, $W_2 \in \mathbb{R}^{H \times LK}$ and $\mathbf{b}_2 \in \mathbb{R}^{LK}$ has

$$ P = L(K + 1)H + H + HLK + LK $$

parameters. One call on $B$ sequences costs $C = 2BL(K + 1)H + 2BHLK$ FLOPs, and the exact bound of one sequence by masked-bound-coefficients calls it on $2^L - 1$ partly masked sequences, one per nonempty mask.

### Assumptions

1. Only the two products are counted, a multiply-add as 2. Used in step 2.

### Steps

1. The input is the one-hot of $L$ tokens over $K + 1$ values, the $K$ symbols and the mask, a vector of length $L(K + 1)$; the hidden layer has $H$ units; the output has $K$ logits for each of $L$ positions, $LK$ in all. The parameters are the entries of the two matrices and two bias vectors.
2. The product of a $(B, L(K + 1))$ input with $W_1$ is $BL(K + 1)H$ multiply-adds, and the product of the $(B, H)$ hidden layer with $W_2$ is $BHLK$; doubling gives $C$.
3. The exact bound sums over every nonempty subset of the $L$ positions, of which there are $2^L - 1$.

For the configurations of the shape contract: the example with $L = K = 3$, $H = 5$ has $60 + 5 + 45 + 9 = 119$ parameters; `markov` with $L = 4$, $K = 3$, $H = 32$ has $512 + 32 + 384 + 12 = 940$; and `text8_chunk` with $L = 256$, $K = 27$, $H = 512$ has $3{,}670{,}016 + 512 + 3{,}538{,}944 + 6{,}912 = 7{,}216{,}384$.

### Result and check

$$ P = L(K + 1)H + H + HLK + LK , \qquad C = 2BL(K + 1)H + 2BHLK . $$

The check `parameter-and-flop-count.check.py` compares `param_count`, `param_shapes` and `flops_forward` with the formulas for many sizes and the configurations of `fixtures/shapes.json`, counts the multiply-adds of `forward` by wrapping its matrix products, and counts the denoiser calls of `exact_nelbo`.
