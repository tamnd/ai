---
id: parameter-and-flop-count
derives: [c-params, c-params-models, c-params-ablations, c-flops, c-flops-models]
kind: calculation
assumes: ["the block of the reference code: in_proj and out_proj without bias, a depthwise convolution with bias, x_proj without bias, dt_proj with bias, A_log, D and a pre-RMSNorm with a weight", "an output layer tied to the embedding and a final RMSNorm", "a multiply-add counts as 2 FLOPs; the norms, the activations, the exponentials and the softmax are not counted"]
uses: [matrix-multiplication@m2]
status: full
check: parameter-and-flop-count.check.py
lean: null
sources:
  - {work: gu-2023-mamba, locator: "p. 7", note: "Section 3.4, the 3 E D^2 parameters of the projections"}
  - {work: gu-2023-mamba, locator: "Table 9", note: "the parameter counts of the ablation of the projection of Delta"}
  - {work: gu-2023-mamba, locator: "Table 10", note: "the parameter counts of the ablation of the state size"}
  - {work: state-spaces-mamba-simple-v2-3-2, locator: "p. 1", note: "the arrays of the block and dt_rank = ceil(d_model / 16)"}
---

### Statement

A Mamba language model with $L$ blocks of width $D$, inner width $I = ED$, state $N$, convolution $K$, $\Delta$ rank $R$ and vocabulary $V$ has

$$ P = VD + D + L\big(3ID + I(K + 1) + I(R + 2N) + (R + 1)I + IN + I + D\big) $$

parameters, and a forward pass over $B$ sequences of $T$ tokens costs

$$ C = 2BT\Big(L\big(2DI + I(R + 2N) + RI + IN + ID\big) + DV\Big) $$

FLOPs in its matrix products, of which the readout of the scan is $2BTLIN$; the convolution and the update of the state add $E = 2BTLI(K + N)$ elementwise. With $V = 50{,}280$, $N = 16$, $K = 4$, $E = 2$ and $R = \lceil D/16 \rceil$, the released models have 129,135,360, 371,516,416, 793,204,224, 1,372,178,432 and 2,768,345,600 parameters; with $D = 1024$ and $L = 48$, $P$ rounds to 371.5M for $N = 16$, $R = 64$, to 367.1M for $N = 1$, to 359.1M for $R = 1$ and to 358.9M for $R = 0$. For Mamba-2.8B with $B = 1$ and $T = 2048$, $C = 11{,}329{,}066{,}762{,}240$ and $E = 26{,}843{,}545{,}600$.

### Assumptions

1. The arrays of a block are those of the reference code, listed in step 1. Used in step 1.
2. The output layer is the transposed embedding. Used in step 2.
3. Only multiply-adds are counted, and $C$ counts those of the matrix products, as every page does. Used in steps 4 and 5.

### Steps

1. A block has: the RMSNorm weight, $D$; `in_proj`, $D \times 2I$; the convolution, $I \times K$ weights and $I$ biases; `x_proj`, $I \times (R + 2N)$, giving the low-rank $\Delta$, $B$ and $C$; `dt_proj`, $R \times I$ and $I$ biases; `A_log`, $I \times N$; `D`, $I$; `out_proj`, $I \times D$. Their sum is $3ID + I(K+1) + I(R + 2N) + (R+1)I + IN + I + D$.
2. The embedding has $VD$ and the final norm $D$; the tied head adds nothing.
3. The projections are $3ID = 3ED^2$ per block, $6D^2$ with $E = 2$, so two blocks have $12D^2$, the count of the paper. For Mamba-2.8B, $D = 2560$, $I = 5120$, $R = 160$: a block has $39{,}321{,}600 + 25{,}600 + 983{,}040 + 824{,}320 + 81{,}920 + 5{,}120 + 2{,}560 = 41{,}244{,}160$; 64 of them and $50{,}280 \cdot 2560 + 2560 = 128{,}719{,}360$ give $2{,}768{,}345{,}600$. The other models follow in the same way.
4. Per token and block the projections cost $2(D \cdot 2I + I(R + 2N) + RI + ID)$ and the readout $Ch$ of the scan, a product over the $N$ entries of the state, $2IN$. The head costs $2DV$. Multiplying by $BT$ gives $C$. The convolution, $2IK$, and the update $\bar{A}h + \bar{B}x$, $2IN$, are elementwise; times $BTL$ they give $E$.
5. For Mamba-2.8B per token and block: $2(26{,}214{,}400 + 983{,}040 + 819{,}200 + 13{,}107{,}200 + 81{,}920) = 82{,}411{,}520$; times 64, plus $2 \cdot 2560 \cdot 50{,}280 = 257{,}433{,}600$, is $5{,}531{,}770{,}880$ per token and $11{,}329{,}066{,}762{,}240$ for 2048 tokens. $E = 2 \cdot 2048 \cdot 64 \cdot 5120 \cdot 20 = 26{,}843{,}545{,}600$, $0.237\%$ of $C$. Twice the parameters is $5{,}536{,}691{,}200$, $0.089\%$ more than the cost per token.

### Result and check

$P$ is dominated by the projections and $C$ is linear in $T$ with about $2P$ per token. The check `parameter-and-flop-count.check.py` compares $P$ with the sizes of the arrays of the NumPy reference for random configs, counts the matrix products and the einsums of a forward pass with an instrumented array, and recomputes the numbers of steps 3 and 5 and of the ablations.
