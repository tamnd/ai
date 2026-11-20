---
id: kv-cache
derives: [c-table1-cache, c-kv-8x, c-state-small]
kind: calculation
assumes: ["an attention layer keeps one key and one value of G A numbers per token, each number in 2 bytes", "a Mamba layer keeps a window of W - 1 inputs and N states for each of its I channels, whatever the length", "GiB means 2^30 bytes and 256K tokens means 2^18"]
uses: []
status: full
check: kv-cache.check.py
lean: null
sources:
  - {work: lieber-2024-jamba, locator: "p. 4", note: "Table 1, the KV cache at 256K tokens in 16-bit; an 8x smaller KV cache than a vanilla Transformer"}
  - {work: ai21labs-2024-jamba-v0-1-config, locator: "p. 1", note: "32 layers, attention period 8, 8 key-value heads, hidden size 4096 with 32 heads, the Mamba sizes"}
---

### Statement

After $T$ tokens, $n_a$ attention layers with $G$ key-value heads of size $A$ keep $S_{\text{KV}} = 2 n_a G A T$ numbers per sequence. For Jamba-v0.1, $n_a = 4$, $G = 8$, $A = 128$, and at $T = 2^{18}$ in 2 bytes this is $2^{32}$ bytes, 4 GiB. The same formula gives 32 GiB for 32 attention layers with 8 key-value heads (Mistral, Mixtral) and 128 GiB for 32 layers with 32 (Llama-2 7B), the three numbers of the paper's Table 1. The 28 Mamba layers keep $28 \cdot I(W - 1 + N) = 4{,}358{,}144$ numbers at any length, 0.2% of the 4 GiB at 256K tokens. Per sequence, one attention layer holds more than one Mamba layer once $T > I(W - 1 + N)/(2GA) = 76$.

### Assumptions

1. The cache of an attention layer is its keys and values, $2GA$ numbers per token, as in grouped-query attention with $G$ groups. Used in step 1.
2. The Mamba layer is that of the page: a causal convolution of width $W$ over $I$ channels and a diagonal SSM of $N$ states per channel. Used in step 3.
3. One number takes 2 bytes, GiB is $2^{30}$ bytes and 256K tokens is $2^{18}$, the reading under which Table 1's numbers come out exactly. Used in step 2.

### Steps

1. Each new token appends a key and a value of $G$ heads of $A$ numbers to each attention layer, so after $T$ tokens $n_a$ layers hold $2 n_a G A T$ numbers.
2. With $n_a = 4 = 2^2$, $G = 8 = 2^3$, $A = 128 = 2^7$, $T = 2^{18}$ and 2 bytes, $2 \cdot 2^2 \cdot 2^3 \cdot 2^7 \cdot 2^{18} \cdot 2 = 2^{32}$ bytes $= 4$ GiB. Replacing $n_a = 4$ by 32 multiplies this by 8, 32 GiB; also replacing $G = 8$ by 32 multiplies it by 4 more, 128 GiB.
3. A Mamba layer needs the last $W - 1$ inputs of its convolution and the state $h \in \mathbb{R}^{I \times N}$: $I(W - 1 + N) = 8192 \cdot 19 = 155{,}648$ numbers, so 28 layers hold $4{,}358{,}144$, which is $4{,}358{,}144 / 2^{31} \approx 0.2\%$ of the $2^{31}$ numbers of step 2.
4. An attention layer holds $2GAT = 2048\,T$ numbers, which exceeds $155{,}648$ when $T > 76$.

### Result and check

With one attention layer in eight, Jamba's cache grows 8 times more slowly than that of a Transformer of the same depth with the same grouped-query heads; the Mamba layers add a constant of a few megabytes. The check `kv-cache.check.py` computes the three numbers of Table 1 with `kv_cache_bytes` of the NumPy reference, the cache of the released configuration with `cache_numbers`, and counts the arrays that `step` keeps while it generates with a small model.
