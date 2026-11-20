---
id: kv-cache
derives: [c-kv, c-kv-mqa, c-cache-flops]
kind: count
assumes: ["the network of parameter-count, decoding B sequences one token at a time", "every layer keeps the key and the value of every token it has seen, each of width D", "a multiply-add counts as 2 FLOPs, and only the matrix products are counted", "GPT-3 175B with n_layers = 96, d_model = 12288, 96 heads of 128 and a context of 2048 tokens"]
uses: [matrix-multiplication@m2]
status: full
check: kv-cache.check.py
lean: null
sources:
  - {work: brown-2020-language-models-few-shot, locator: "p. 8", note: "Table 2.1"}
  - {work: shazeer-2019-fast-transformer-decoding, locator: "pp. 1 to 5", note: "incremental decoding and multi-query attention"}
  - {work: ainslie-2023-gqa, locator: "p. 1", note: "grouped-query attention"}
---

### Statement

1. After $n$ tokens the cache of keys and values holds $2LDn$ numbers for each sequence: for GPT-3 175B, $2{,}359{,}296$ numbers per token, 4.5 MiB at two bytes a number, and 9 GiB for a full context of 2048 tokens.
2. If the $H$ heads of a layer share one key and one value of width $d_{\text{head}} = D/H$, as in multi-query attention, the cache holds $2L\,d_{\text{head}}\,n$ numbers, $H$ times fewer; with $G$ groups of heads it holds $2LG\,d_{\text{head}}\,n$. For GPT-3 175B with one shared key and value per layer that is 24,576 numbers, 48 KiB, per token.
3. Feeding $n$ tokens one at a time with the cache costs $n\,P + 2LDn(n+1)$ FLOPs per sequence, where $P = L(8D^2 + 4DF) + 2DV$, the causal count of one forward pass over the $n$ tokens. Running the whole prefix again for every new token costs $P\,n(n+1)/2 + \tfrac{2}{3}LDn(n+1)(2n+1)$; for GPT-3 175B and $n = 2048$ this is $7.46 \times 10^{17}$ FLOPs against $7.25 \times 10^{14}$, 1,029 times more.

### Assumptions

1. A layer computes the key and the value of a token once, from the token's normalized state, and they do not change when later tokens arrive, because the mask keeps the state of a token from depending on later ones. Used in steps 1 and 4.
2. Multi-query attention keeps the queries of all $H$ heads and projects the keys and values to one head of width $d_{\text{head}}$; grouped-query attention to $G$ heads. Used in step 3.
3. The cost conventions of compute. Used in steps 4 to 6.

### Steps

1. In each of the $L$ layers a token has a key and a value, each of width $D$ over its heads: $2D$ numbers per token per layer and $2LD$ per token.
2. GPT-3 175B: $2 \times 96 \times 12{,}288 = 2{,}359{,}296$ numbers per token. At two bytes, $4{,}718{,}592$ bytes $= 4.5 \times 2^{20}$, and for 2048 tokens $9{,}663{,}676{,}416$ bytes $= 9 \times 2^{30}$.
3. With one key and one value of width $d_{\text{head}}$ per layer the count is $2L\,d_{\text{head}} = 2 \times 96 \times 128 = 24{,}576$ per token, $1/96$ of step 2, or $49{,}152$ bytes, 48 KiB. $G$ groups multiply this by $G$.
4. With the cache, the step that feeds the token at position $m$ projects one token, $L(8D^2 + 4DF) + 2DV = P$, and its query meets the $m$ keys and values in the cache and its own, $4mD$ per layer. Summing over $m = 1, \dots, n$ gives $nP + 4LD\,n(n+1)/2 = nP + 2LDn(n+1)$, the causal count of statement 1 of compute.
5. Without it, the step at position $m$ runs a forward pass over $m$ tokens, $mP + 4LDm^2$ with dense attention. Summing, $P \sum m + 4LD \sum m^2 = P\,n(n+1)/2 + \tfrac{2}{3}LD\,n(n+1)(2n+1)$.
6. For GPT-3 175B, $P = 96(8 \cdot 12{,}288^2 + 4 \cdot 12{,}288 \cdot 49{,}152) + 2 \cdot 12{,}288 \cdot 50{,}257 = 349{,}127{,}467{,}008$. With $n = 2048$, step 4 gives $7.2491 \times 10^{14}$ and step 5 $7.4605 \times 10^{17}$, a ratio of 1,029; the term in $P$ alone has the ratio $(n+1)/2 = 1024.5$.

### Result and check

$$ \text{cache} = 2LDn, \qquad \frac{P\,n(n+1)/2 + \tfrac{2}{3}LDn(n+1)(2n+1)}{nP + 2LDn(n+1)} \approx \frac{n+1}{2} . $$

The check `kv-cache.check.py` reads the size of the cache that the NumPy reference keeps, counts the products of greedy decoding with and without the cache through an ndarray subclass, checks that both give the same tokens, and recomputes the GPT-3 figures.
