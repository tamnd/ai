---
id: compute
derives: [c-flops, c-attn-share, c-table6-check]
kind: count
assumes: ["the network of parameter-count with n = N + 1 tokens", "a multiply-add counts as 2 FLOPs, and only the matrix products are counted", "a training step costs three forward passes, one for the forward pass and two for the backward pass", "pre-training on the 303M images of JFT at 224 pixels for the number of epochs listed in Table 6"]
uses: [matrix-multiplication@m2]
status: full
check: compute.check.py
lean: null
sources:
  - {work: dosovitskiy-2021-image-worth-16x16-words, locator: "pp. 3 to 5", note: "eq. 1 to 4, JFT and Table 1"}
  - {work: dosovitskiy-2021-image-worth-16x16-words, locator: "p. 15", note: "Table 6"}
---

### Statement

1. A forward pass over $B$ images costs $C = B\big[2 N P^2 C D + L(8 n D^2 + 4 n^2 D + 4 n D D_{\text{MLP}}) + 2 D K\big]$ FLOPs in its products.
2. The share of a layer's FLOPs that grows with $n^2$ is $n / (n + 2D + D_{\text{MLP}})$: 4.1% for ViT-B/16 at 224 pixels and 11.1% at 384.
3. $3 \times C \times \text{epochs} \times 303\text{M}$ reproduces the exaFLOPs of the six ViT rows of Table 6 to within 2.1%, and all rows but ViT-B/32 to within 0.3%.

### Assumptions

1. The products of eq-patches, eq-msa, eq-mlp and eq-output, and nothing else: the softmax, the normalizations, the GELU and the additions are left out. Used in steps 1 to 3.
2. The rule of three forward passes for a step: the backward pass of a product $XW$ costs two products of the same size, one for the gradient of $X$ and one for that of $W$. Used in step 5.
3. The pre-training head of the paper, with $D$ hidden units and the 18k classes of JFT, is replaced by the linear head to 1000 classes; the difference is at most 0.31% of a forward pass, for ViT-B/32, and under 0.13% for the others. Used in step 5.

### Steps

1. An $m \times k$ by $k \times n$ product costs $2mkn$ FLOPs. The patch embedding multiplies $N \times P^2 C$ by $P^2 C \times D$: $2 N P^2 C D$.
2. In a layer, the qkv projection costs $2 n D \cdot 3D$ and the output projection $2 n D^2$, together $8 n D^2$. Each of the $k$ heads computes $\mathbf{q}\mathbf{k}^\top$, $n \times D_h$ by $D_h \times n$, and $A\mathbf{v}$, $n \times n$ by $n \times D_h$, $4 n^2 D_h$ for the two; over $k$ heads with $k D_h = D$ that is $4 n^2 D$. The MLP costs $2 n D D_{\text{MLP}}$ twice.
3. The head costs $2DK$. Summing over the $L$ layers and the batch gives statement 1.
4. Dividing the $n^2$ term of step 2 by the layer total, $4n^2 D / (8nD^2 + 4n^2 D + 4nDD_{\text{MLP}}) = n / (2D + n + D_{\text{MLP}})$. For ViT-B/16, $197 / 4805 = 0.0410$ at 224 pixels and $577 / 5185 = 0.1113$ at 384, which is statement 2.
5. For ViT-B/16, $C = 35{,}127{,}656{,}448$ per image; times 3, 7 epochs and $303 \times 10^6$ images, $2.235 \times 10^{20}$ FLOPs or 223.5 exaFLOPs, against 224 in Table 6. The other rows are in the check.

### Result and check

| model | epochs | this count (EF) | Table 6 (EF) | difference |
|---|---|---|---|---|
| ViT-B/32 | 7 | 56.1 | 55 | +2.0% |
| ViT-B/16 | 7 | 223.5 | 224 | −0.2% |
| ViT-L/32 | 7 | 195.7 | 196 | −0.2% |
| ViT-L/16 | 7 | 783.3 | 783 | +0.0% |
| ViT-L/16 | 14 | 1566.7 | 1567 | −0.0% |
| ViT-H/14 | 14 | 4258.3 | 4262 | −0.1% |

The paper does not say how it estimated the compute, so this agreement is evidence that it used a count of this kind, not a statement of the paper. The check `compute.check.py` counts the FLOPs of the matrix products of the NumPy reference on random configurations with a counting array, compares them with the closed form, and recomputes the table.
