---
id: recurrent-state
derives: [c-state, c-state-compare]
kind: calculation
assumes: ["the block of the reference code, with a causal convolution of width K and a diagonal SSM of N states per channel", "a Transformer layer caches one key and one value of width D per token"]
uses: []
status: full
check: recurrent-state.check.py
lean: null
sources:
  - {work: gu-2023-mamba, locator: "p. 2", note: "constant time per step, no cache of previous elements"}
  - {work: gu-2023-mamba, locator: "p. 7", note: "two Mamba blocks match the 12 D^2 of a Transformer layer"}
  - {work: state-spaces-2023-mamba-2-8b-config, locator: "p. 1", note: "d_model 2560, n_layer 64"}
---

### Statement

Generating one token at a time, a Mamba block needs from the past only $I(K - 1 + N)$ numbers per sequence; $L$ blocks need $S_{\text{Mamba}} = LI(K - 1 + N)$, 6,225,920 for Mamba-2.8B. A Transformer of $L/2$ layers of width $D$ caches $S_{\text{Transformer}} = LDT$ numbers after $T$ tokens. With $I = 2D$, $S_{\text{Transformer}} > S_{\text{Mamba}}$ exactly when $T > 2(K - 1 + N)$, which is $T > 38$ for $K = 4$ and $N = 16$.

### Assumptions

1. The block's only dependences on the past are the convolution and the SSM. Used in step 1.
2. Each attention layer keeps a key and a value of width $D$ for every earlier token. Used in step 3.

### Steps

1. The output at $t$ of a causal convolution of width $K$ depends on the inputs $t - K + 1, \dots, t$, so a step needs the last $K - 1$ inputs of each of the $I$ channels: $I(K - 1)$ numbers. The SSM's next state depends on the past only through $h_{t-1}$, $I \times N$ numbers. Everything else in the block is computed from the current token.
2. For Mamba-2.8B, $L = 64$, $I = 5120$, $K = 4$, $N = 16$: $64 \cdot 5120 \cdot 19 = 6{,}225{,}920$, the same at every length.
3. A Transformer layer stores $2D$ numbers per token; $L/2$ layers store $LDT$ after $T$ tokens.
4. With $I = 2D$, $LDT > 2LD(K - 1 + N)$ if and only if $T > 2(K - 1 + N) = 38$.

### Result and check

Past a few dozen tokens a Transformer of the same parameter count carries more state than Mamba, and the gap grows linearly with the context. The check `recurrent-state.check.py` runs the recurrent `step` of the NumPy reference and counts its state, compares the step outputs with the parallel forward pass, which shows the state is sufficient, and recomputes the numbers of steps 2 and 4.
