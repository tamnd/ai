---
id: compute
derives: [c-flops, c-flops-token, c-attn-share, c-d1-check]
kind: count
assumes: ["the network of parameter-count, run on B sequences of n tokens", "a multiply-add counts as 2 FLOPs, and only the matrix products are counted", "a training token costs three forward passes, one for the forward pass and two for the backward pass", "the GPT-3 models of Table 2.1 with d_ff = 4 d_model, n = 2048 and 300 billion training tokens"]
uses: [matrix-multiplication@m2]
status: full
check: compute.check.py
lean: null
sources:
  - {work: brown-2020-language-models-few-shot, locator: "p. 8", note: "Table 2.1 and the 300 billion tokens"}
  - {work: brown-2020-language-models-few-shot, locator: "p. 46", note: "Table D.1 and its 6 FLOPs per parameter per token"}
  - {work: kaplan-2020-scaling-laws, locator: "pp. 6 to 7", note: "eq. 2.2, Table 1 and C = 6N"}
---

### Statement

1. A forward pass over $B$ sequences of $n$ tokens costs $C = B\big[L(8nD^2 + 4n^2D + 4nDF) + 2nDV\big]$ FLOPs in its products when the attention products are computed in full, as dense code does, and $C_{\text{causal}} = B\big[L(8nD^2 + 2n(n+1)D + 4nDF) + 2nDV\big]$ when only the places that the mask lets through are computed.
2. Per token, $C_{\text{causal}} / (Bn) = L(8D^2 + 4DF) + 2DV + 2L(n+1)D$, which is $2N + 2DV + 2L(n+1)D$ less the $2L(9D + F)$ that the biases and gains of the layers would add, with $N$ the parameters of the layers. For the GPT-3 models at $n = 2048$ this is between $1.014$ times (175B) and $1.137$ times (Small) twice the total number of parameters.
3. With $F = 4D$, the share of a layer's causal FLOPs spent in the products $\mathbf{q}\mathbf{k}^\top$ and $A\mathbf{v}$ is $(n+1)/(n+1+12D)$: at $n = 2048$ it is 18.2% for GPT-3 Small, 14.3% for Medium, 10.0% for Large, 7.7% for XL and 1.4% for 175B.
4. $6 \times N_{\text{total}} \times 300 \times 10^9$, with the counts of parameter-count, reproduces the training compute of each GPT-3 row of Table D.1 to within 0.5%, and with the parameters printed in the table to within 0.2%; for 175B it is $3.14 \times 10^{23}$ FLOPs or 3,637.6 petaFLOP/s-days.

### Assumptions

1. The network of parameter-count and its products: the qkv and output projections, $\mathbf{q}\mathbf{k}^\top$ and $A\mathbf{v}$, the two layers of the MLP and the unembedding. Used in steps 1 to 3.
2. A multiply-add counts as 2, so a product of an $m \times k$ and a $k \times p$ matrix costs $2mkp$; the softmax, the normalizations, the GELU, the additions and the lookups of the embeddings are not counted. Used in steps 1 to 3.
3. A training token costs a forward pass and a backward pass of twice its FLOPs, and the attention term is left out, as Table D.1 and Kaplan et al. do. Used in step 6.

### Steps

1. For $n$ tokens of width $D$, the qkv projection costs $2n \cdot D \cdot 3D = 6nD^2$ and the output projection $2nD^2$. Each of the $H$ heads forms $\mathbf{q}\mathbf{k}^\top$, an $n \times d_{\text{head}}$ by $d_{\text{head}} \times n$ product, $2n^2 d_{\text{head}}$, and $A\mathbf{v}$, the same again; over the heads that is $4n^2D$. The MLP costs $2nDF + 2nFD = 4nDF$ and the unembedding $2nDV$. Summing over $L$ layers and $B$ sequences gives the first count of statement 1.
2. Under the mask, query $i$ sees $i$ keys, so the two attention products of a head cost $4 i\, d_{\text{head}}$ for query $i$ and $\sum_{i=1}^n 4 i\, d_{\text{head}} = 2n(n+1) d_{\text{head}}$ for all of them. Over the heads this replaces $4n^2D$ by $2n(n+1)D$, the second count of statement 1.
3. Divide by $Bn$: $L(8D^2 + 4DF) + 2DV + 2L(n+1)D$. The parameters of a layer are $4D^2 + 2DF + 9D + F$, so $L(8D^2 + 4DF) = 2N - 2L(9D + F)$: the biases and gains add no products, and the per-token cost is $2N + 2DV + 2L(n+1)D$ up to the $2L(9D+F)$ of the biases. This is Kaplan et al.'s $C_{\text{forward}} \approx 2N + 2 n_{\text{layer}} n_{\text{ctx}} d_{\text{model}}$ with the unembedding added and $n + 1$ in place of $n_{\text{ctx}}$. For GPT-3 Small at $n = 2048$ it is 284,831,232 FLOPs per token against $2 \times 125{,}226{,}240$, a ratio of 1.137, and for 175B 353,961,664,512 against $2 \times 174{,}604{,}259{,}328$, a ratio of 1.014; the other six lie between.
4. With $F = 4D$ a layer's causal products per token are $8D^2 + 16D^2 + 2(n+1)D = 24D^2 + 2(n+1)D$, so the share of the attention products is $2(n+1)D / (24D^2 + 2(n+1)D) = (n+1)/(n+1+12D)$.
5. At $n = 2048$: Small, $D = 768$, gives $2049 / 11{,}265 = 0.182$; Medium, $D = 1024$, $2049/14{,}337 = 0.143$; Large, $D = 1536$, $2049/20{,}481 = 0.100$; XL, $D = 2048$, $2049/26{,}625 = 0.077$; 175B, $D = 12{,}288$, $2049/149{,}505 = 0.0137$. The share falls under 10% once $12D > 9(n+1)$, which at $n = 2048$ is from $D = 1537$ on: all models from XL up, and Large is at the boundary.
6. A training token costs $2N_{\text{total}}$ forward and twice that backward, $6N_{\text{total}}$. Times 300 billion tokens, GPT-3 175B costs $6 \times 174{,}604{,}259{,}328 \times 3 \times 10^{11} = 3.1429 \times 10^{23}$ FLOPs, and divided by the $10^{15} \times 86{,}400$ FLOPs of a petaFLOP/s-day, 3,637.6. Table D.1 lists $3.14 \times 10^{23}$ and $3.64 \times 10^{3}$. The other rows follow in the same way: with the exact counts the largest gap is that of XL, whose 1,315.7M parameters the table rounds to 1,320M, 0.49%, and with the printed parameters every row is within 0.17%.

### Result and check

$$ \frac{C_{\text{causal}}}{Bn} = L(8D^2 + 4DF) + 2DV + 2L(n+1)D, \qquad \frac{\text{attention}}{\text{layer}} = \frac{n+1}{n+1+12D} . $$

The check `compute.check.py` counts the products of the NumPy forward pass with an ndarray subclass that adds $2mkp$ at every matrix product, and compares the count with the first formula of statement 1 for random configurations; it also counts the products of decoding one token at a time with the cache, whose sum is the causal count. It then recomputes statements 2 to 4 for the eight GPT-3 models.
