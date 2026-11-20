---
id: parameter-and-flop-count
derives: [c-params, c-params-tied, c-flops, c-hinton-sizes]
kind: count
assumes: ["an encoder of fully connected layers of widths n_0 = D, n_1, ..., n_K = C and a decoder of the same widths in reverse, every layer with a bias", "with tied weights, decoder layer j uses the transpose of encoder layer K - 1 - j and has its own bias", "a multiply-add counts as 2 FLOPs and only the matrix products are counted"]
uses: [matrix-multiplication@m2]
status: full
check: parameter-and-flop-count.check.py
lean: null
sources:
  - {work: hinton-2006-reducing-dimensionality-data-neural, locator: "p. 507", note: "the widths of the curves, MNIST, faces and documents autoencoders"}
---

### Statement

1. The autoencoder has $P = 2 \sum_{k=0}^{K-1} n_k n_{k+1} + \sum_{k=1}^{K} n_k + \sum_{k=0}^{K-1} n_k$ parameters with untied weights, and $P_\text{tied} = \sum_{k=0}^{K-1} n_k n_{k+1} + \sum_{k=1}^{K} n_k + \sum_{k=0}^{K-1} n_k$ with tied weights: tying halves the weights and keeps all the biases.
2. With the uniform widths of the shape contract, $n_1 = \dots = n_L = H$ and $n_{L+1} = C$, this is $P = 2 (D H + (L - 1) H^2 + H C) + L H + C + L H + D$.
3. A forward pass over a batch of $B$ costs $C_\text{fwd} = 4 B \sum_{k} n_k n_{k+1}$ FLOPs, the same with tied weights, and the backward pass of the reference costs $2 C_\text{fwd}$.
4. The deep autoencoders of Hinton and Salakhutdinov have, untied, 2,837,314 parameters (784-1000-500-250-30, MNIST), 842,340 (784-400-200-100-50-25-6, curves), 7,537,655 (625-2000-1000-500-30, faces) and 2,318,760 (2000-500-250-125-10, documents); after pretraining, the unrolled encoder and decoder initially use the same weights.

### Assumptions

1. The architecture of the assumptions above. Used in steps 1 and 2.
2. FLOPs of products only, a multiply-add as 2. Used in step 3.

### Steps

1. Encoder layer $k$ has an $n_k \times n_{k+1}$ matrix and $n_{k+1}$ biases; decoder layer $j$ maps $n_{K-j}$ to $n_{K-j-1}$ units with a matrix of the same size as encoder layer $K - 1 - j$ and $n_{K-j-1}$ biases. Summing gives statement 1: the encoder's biases are $n_1, \dots, n_K$ and the decoder's $n_{K-1}, \dots, n_0$. With tied weights only the matrices of one side are parameters.
2. Substituting the widths: the products are $D H$, $L - 1$ times $H^2$, and $H C$; the encoder's biases are $L H + C$, the decoder's $L H + D$. This is statement 2, and it is the formula `model.params` of shapes.yaml block by block.
3. Each layer multiplies a $B \times n_k$ matrix by an $n_k \times n_{k+1}$ one, $2 B n_k n_{k+1}$ FLOPs, and the decoder repeats every product once; a transposed matrix costs the same. The backward pass computes, for every layer, the gradient of the weights ($h^\top \delta$) and of the layer input ($\delta W^\top$), each a product of the same size, so twice the forward FLOPs. This is statement 3.
4. Statement 1 with the four width lists of the paper (p. 507) gives the numbers of statement 4. The tied counts, the weights of pretraining, are 1,420,814, 422,340, 3,772,655 and 1,161,260.

### Result and check

$$ P = 2 \sum_{k=0}^{K-1} n_k n_{k+1} + \sum_{k=1}^{K} n_k + \sum_{k=0}^{K-1} n_k , \qquad C_\text{fwd} = 4 B \sum_{k=0}^{K-1} n_k n_{k+1} . $$

The check `parameter-and-flop-count.check.py` counts the arrays that the NumPy reference allocates for random width lists, tied and untied, compares them with the closed forms and with the uniform-width formula of statement 2, counts the multiply-adds of the forward and backward passes by instrumenting the matrix products, and recomputes the four counts of statement 4.
