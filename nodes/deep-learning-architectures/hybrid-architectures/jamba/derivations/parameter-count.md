---
id: parameter-count
derives: [c-params, c-active, c-experts-share]
kind: count
assumes: ["the layers of the page, with the sizes of the released configuration", "the output layer is not tied to the embedding", "a token uses the router and K of the n experts of each mixture"]
uses: []
status: full
check: parameter-count.check.py
lean: null
sources:
  - {work: lieber-2024-jamba, locator: "p. 4", note: "Table 1, 52B available and 12B active parameters"}
  - {work: lieber-2024-jamba, locator: "p. 5", note: "4 Jamba blocks, l = 8, a : m = 1 : 7, e = 2, n = 16, K = 2"}
  - {work: ai21labs-2024-jamba-v0-1-config, locator: "p. 1", note: "the sizes of the released model"}
---

### Statement

With $D = 4096$, $F = 14336$, $V = 65536$, $I = 8192$, $N = 16$, $W = 4$, $R = 256$, 8 key-value heads of 128 and 4 blocks of 8 layers, Jamba-v0.1 has $P = 51{,}570{,}323{,}328$ parameters, of which a token uses $P_a = 12{,}110{,}311{,}296$: the 52B and 12B of Table 1. The difference is $16 \cdot 14 = 224$ unused experts of $3DF = 176{,}160{,}768$ parameters each, and the experts hold $45.1$B, 87% of the model; the 28 Mamba layers hold 2.95B and the 4 attention layers 0.17B.

### Assumptions

1. Attention has $W_Q, W_O \in \mathbb{R}^{D \times D}$ and $W_K, W_V \in \mathbb{R}^{D \times GA}$ with $GA = 1024$, and no biases. Used in step 1.
2. A Mamba layer has the parameters of eq-params: in_proj $D \times 2I$, the convolution $I \times W$ and its bias, x_proj $I \times (R + 2N)$, the three inner RMSNorm weights $R + 2N$, dt_proj and its bias $(R + 1)I$, $A$ of $I \times N$, $D$ of $I$ and out_proj $I \times D$. Used in step 1.
3. Every SwiGLU MLP and expert has $3DF$ parameters; a router has $Dn$. Used in steps 1 and 2.
4. Each layer has two RMSNorm weights of $D$, and there is a final one. Used in step 1.

### Steps

1. Per layer: attention $2D^2 + 2D \cdot 1024 = 41{,}943{,}040$; Mamba $3ID + I(W + 1) + I(R + 2N) + R + 2N + (R + 1)I + IN + I = 105{,}308{,}448$; an MLP $3DF = 176{,}160{,}768$; a mixture $16 \cdot 3DF + 16D = 2{,}818{,}637{,}824$.
2. The model has 4 attention layers, 28 Mamba layers, 16 layers with an MLP (the even ones) and 16 with a mixture (the odd ones), so the experts hold $256 \cdot 3DF = 45{,}097{,}156{,}608$, the Mamba layers $2{,}948{,}636{,}544$, the MLPs $2{,}818{,}572{,}288$, the embedding and output layer $2VD = 536{,}870{,}912$, attention $167{,}772{,}160$, the routers $1{,}048{,}576$ and the norms $65D = 266{,}240$; the sum is $51{,}570{,}323{,}328$.
3. A token goes through 2 of the 16 experts of each of the 16 mixtures, so $16 \cdot 14 = 224$ experts are unused: $P_a = P - 224 \cdot 3DF = 51{,}570{,}323{,}328 - 39{,}460{,}012{,}032 = 12{,}110{,}311{,}296$.

### Result and check

Rounded, $P = 52$B and $P_a = 12$B, as in Table 1. Most of the parameters sit in experts that a token does not use, which is why the paper can call a 52B model one with 12B active and fit its weights in 8 bits on one 80GB GPU. The check `parameter-count.check.py` builds the parameter shapes of the released configuration with `param_shapes` of the NumPy reference, sums them by kind, and compares with `param_count`.
