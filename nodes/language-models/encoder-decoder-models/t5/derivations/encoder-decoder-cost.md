---
id: encoder-decoder-cost
derives: [c-pm-params, c-pm-flops, c-pm-attention]
kind: cost
assumes: ["an L + L encoder-decoder with the layers of this page, run on S source and T target tokens", "a decoder-only language model of the same width with L layers, each a self-attention of 4 M I parameters and a feed-forward network of 2 M F, run on the S + T tokens of the pair", "FLOPs of the matrix products only, a multiply-add counted as 2, attention computed in full, the output layer left out"]
uses: [matrix-multiplication@m2]
status: full
check: encoder-decoder-cost.check.py
lean: null
sources:
  - {work: raffel-2020-t5, locator: "pp. 18 and 19", note: "Section 3.2.2, the equivalences of parameters and computation between an L + L encoder-decoder and a language model"}
---

### Statement

1. The matrices of the $L + L$ encoder-decoder hold $L\,(12MI + 4MF)$ parameters, and those of a $2L$-layer language model $L\,(8MI + 4MF)$.
2. Leaving out the products of queries with keys and of weights with values, the encoder-decoder costs $2L\,(S + T)(6MI + 2MF)$ FLOPs on a pair and an $L$-layer language model on the same tokens $2L\,(S + T)(4MI + 2MF)$.
3. The two ratios in statements 1 and 2 are the same, $(12MI + 4MF) / (8MI + 4MF)$, which is $7/6$ when $I = M$ and $F = 4M$. The attention to the encoder causes both: the encoder-decoder has 7/6 of the parameters of the $2L$-layer model and 7/6 of the computation of the $L$-layer one.
4. The products of queries with keys and of weights with values cost $4LI\,(S^2 + ST + T^2)$ in the encoder-decoder and $4LI\,(S + T)^2$ in the language model, less by $4LIST$; for $S = T$ the encoder-decoder pays three quarters.
5. For Base with $S = T = 512$ the encoder-decoder costs 231.9 GFLOPs without the output layer and the 12-layer language model 212.6, a ratio of exactly 12/11; the $6 + 6$ model costs half the encoder-decoder.

### Assumptions

1. The encoder reads only the source and the decoder only the target, while the language model reads both, as the paper describes. Used in step 2.
2. $I = M$ and $F = 4M$ for the ratio $7/6$, as in Base. Used in step 3.

### Steps

1. A layer of the language model holds $4MI$ in its attention and $2MF$ in its feed-forward network, so $2L$ of them hold $L\,(8MI + 4MF)$. The $L$ encoder layers hold $L\,(4MI + 2MF)$ and the $L$ decoder layers $L\,(8MI + 2MF)$; the sum is $L\,(12MI + 4MF)$. This is statement 1.
2. A matrix of $M \times I$ applied to $n$ tokens costs $2nMI$. Each encoder layer applies four such matrices and two of the feed-forward network to the $S$ source tokens: $S\,(8MI + 4MF)$. Each decoder layer applies eight attention matrices and the feed-forward network to the $T$ target tokens, except the keys and values of the attention to the encoder, which are computed from the $S$ encoder outputs: $T\,(12MI + 4MF) + S \cdot 4MI$. The sum over the $L$ pairs is $L\,(S + T)(12MI + 4MF)$. A layer of the language model costs $(S + T)(8MI + 4MF)$, and $L$ of them $L\,(S + T)(8MI + 4MF)$. Dividing by 2 gives statement 2.
3. Both statements contain the factor $(12MI + 4MF)/(8MI + 4MF)$. With $I = M$ and $F = 4M$ it is $(12 + 16)/(8 + 16) = 28/24 = 7/6$. The extra $4MI$ of each pair of layers, in parameters and in FLOPs per token, is the attention to the encoder, which is $4/28 = 1/7$ of the layers' parameters. This is statement 3.
4. Self-attention over $n$ tokens computes $n^2$ scores of $I$ features and averages $n^2$ values of $I$ features, $4n^2 I$ FLOPs. The encoder pays $4S^2 I$, the decoder's self-attention $4T^2 I$ and its attention to the encoder $4STI$, per layer. The language model pays $4(S + T)^2 I = 4(S^2 + 2ST + T^2)I$: the source attending to the target and the target to the source, of which the causal mask throws half away. With $S = T$ the ratio is $3/4$. This is statement 4.
5. With $M = I = 768$, $F = 3072$, $L = 12$ and $S = T = n = 512$, the encoder-decoder costs $L\,(2n \cdot 28M^2 + 12n^2 M) = 231{,}928{,}233{,}984$ and the language model $L\,(2n \cdot 24M^2 + 16n^2 M) = 212{,}600{,}881{,}152$. Divided by $LnM$ they are $56M + 12n = 49{,}152$ and $48M + 16n = 45{,}056$, whose ratio is $12/11$. The $6 + 6$ model has half the layers of the $12 + 12$ one and costs half as much. This is statement 5.

### Result and check

$$ \frac{\text{params}(L + L)}{\text{params}(2L\ \text{LM})} = \frac{\text{FLOPs}(L + L)}{\text{FLOPs}(L\ \text{LM})}\Big|_{\text{matrices}} = \frac{12MI + 4MF}{8MI + 4MF} = \frac{7}{6} . $$

The check `encoder-decoder-cost.check.py` counts the FLOPs of the NumPy reference with flops() against the closed forms for random sizes, builds the language model's count from its layers, and recomputes the figures for Base.
