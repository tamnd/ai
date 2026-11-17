---
id: sparse-cost
derives: [c-cost-params-flops, c-cost-ratio, c-cost-worked, c-cost-intensity]
kind: cost
assumes: ["each expert is a two-layer feed-forward network d → h → d without biases, as in GShard", "the router is a d × N matrix", "a multiply-add counts as 2 FLOPs, and the cost of the nonlinearity, softmax, top-k, dispatch and combine is left out", "a static-shape implementation computes every slot of every expert, padding included"]
uses: [matrix-multiplication@m2]
status: full
check: sparse-cost.check.py
lean: null
sources:
  - {work: lepikhin-2020-gshard, locator: "p. 4"}
  - {work: fedus-2022-switch-transformers, locator: "p. 8"}
  - {work: shazeer-2017-sparsely-gated-mixture-of-experts, locator: "p. 5"}
---

### Statement

For a layer of $N$ experts of width $h$ on $d$-dimensional tokens, routed top-$k$ with capacity factor $c$:

1. the layer has $2Ndh + dN$ weights, and a dense feed-forward layer of the same width has $2dh$;
2. a token costs $2(2dh)k + 2dN = 4kdh + 2dN$ FLOPs if only the chosen experts run, and $4kc\,dh + 2dN$ FLOPs on average if every padded slot is computed, against $4dh$ for the dense layer;
3. the ratio of weights to per-token FLOPs is therefore about $N/(2k)$ for the sparse layer and $1/2$ for the dense one, so with $k$ fixed the parameter count grows with $N$ while the compute per token does not, apart from the $2dN$ of the router;
4. on a layer with $d = 768$, $h = 3072$ and $N = 128$: one expert has $4{,}718{,}592$ weights and the layer $604{,}078{,}080$, 128 times the dense layer plus the router; a token costs $9{,}633{,}792$ FLOPs at $k = 1$ without padding, $11{,}993{,}088$ at $k = 1$ and $c = 1.25$, and $19{,}070{,}976$ at $k = 2$, against $9{,}437{,}184$ for the dense layer;
5. an expert does $2dh$ multiply-adds per token and receives and returns $2d$ numbers, so its ratio of computation to communication is $h$, the hidden width, as Shazeer et al. observe.

### Assumptions

1. Biases, layer norms and the residual add are left out; they add $O(d + h)$ per expert and do not change the leading terms.
2. The router costs one $d \times N$ product per token whether or not the experts run.
3. Item 2's padded cost charges each token $kc$ expert evaluations: $N$ experts with $C = kTc/N$ slots each, ignoring the rounding, compute $kTc$ token-expert products for $T$ tokens.

### Steps

1. Each expert holds $W_{\text{in}} \in \mathbb R^{d\times h}$ and $W_{\text{out}} \in \mathbb R^{h\times d}$, $2dh$ weights; $N$ experts and the router give $2Ndh + dN$.
2. One expert applied to one token costs $dh$ multiply-adds for each matrix, $2dh$ in all, which is $4dh$ FLOPs. A token runs $k$ experts, $4kdh$, plus the router's $dN$ multiply-adds, $2dN$ FLOPs. With padding, the $NC = kTc$ slots each cost $4dh$, which is $4kc\,dh$ per token on average.
3. Dividing item 1 by item 2 without the router terms, $2Ndh/(4kdh) = N/(2k)$, and $2dh/(4dh) = 1/2$ for the dense layer.
4. $2 \cdot 768 \cdot 3072 = 4{,}718{,}592$; times 128 is $603{,}979{,}776$, and the router's $768 \cdot 128 = 98{,}304$ makes $604{,}078{,}080$. One expert costs $4 \cdot 768 \cdot 3072 = 9{,}437{,}184$ FLOPs per token and the router $2 \cdot 768 \cdot 128 = 196{,}608$. So $k = 1$ costs $9{,}437{,}184 + 196{,}608 = 9{,}633{,}792$; with $c = 1.25$ the experts cost $1.25 \cdot 9{,}437{,}184 = 11{,}796{,}480$ and the total is $11{,}993{,}088$; $k = 2$ costs $2 \cdot 9{,}437{,}184 + 196{,}608 = 19{,}070{,}976$. The router adds about 2% at $k = 1$.
5. The expert reads $d$ inputs and writes $d$ outputs, $2d$ numbers, and does $2dh$ multiply-adds, so the ratio is $2dh/2d = h$.

### Result and check

$$ \text{params} = 2Ndh + dN,\qquad \text{FLOPs per token} = 4k\,dh + 2dN\ \text{(no padding)},\qquad \frac{\text{params}}{\text{FLOPs}} \approx \frac{N}{2k}. $$

The check `sparse-cost.check.py` builds the layer with random $d$, $h$, $N$, $k$ and $c$, counts its weights, runs a forward pass that counts the multiply-adds of every matrix product it does, once running only the chosen experts and once computing every padded slot, and compares the counts with items 1 to 3.
It then recomputes the numbers of item 4 and the ratio of item 5.
