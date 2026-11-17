---
id: transformer-memory
derives: [c-layer-memory, c-selective-share, eq-transformer-memory]
kind: count
assumes: ["one transformer layer with a attention heads, hidden size h, sequence length s and micro-batch b", "no model parallelism", "16-bit activations (2 bytes) and 1-byte dropout masks", "an MLP of width 4h with GeLU, dropout after the attention scores, after the attention projection and after the MLP"]
uses: []
status: full
check: transformer-memory.check.py
lean: null
sources:
  - {work: korthikanti-2022-reducing-activation-recomputation, locator: "Section 4"}
  - {work: korthikanti-2022-reducing-activation-recomputation, locator: "Section 4.1"}
  - {work: korthikanti-2022-reducing-activation-recomputation, locator: "Section 5"}
---

### Statement

The activations that one transformer layer keeps for its backward pass take

$$ \text{bytes per layer} = sbh\left(34 + \frac{5as}{h}\right), $$

and the attention-score part, $5as^2b$ bytes, is the share $\frac{5as/h}{34 + 5as/h}$ of the total.

### Assumptions

1. Each kept value takes 2 bytes, and each dropout mask takes 1 byte per element (Korthikanti et al. 2022, Section 4).
2. A layer's input and output have shape $s \times b \times h$, and the MLP widens to $4h$.
3. There is no tensor, sequence or pipeline parallelism. The sharded forms are in the same source, Table 2.

### Steps

1. **Attention block.** The input shared by the Q, K and V products takes $2sbh$. The product $QK^\top$ needs Q and K, which take $4sbh$. The softmax output, of shape $a \times s \times s$ per sample, takes $2as^2b$. The softmax-dropout mask takes $as^2b$. Attention over V needs the dropout output and V, which take $2as^2b + 2sbh$. The output projection keeps its input, $2sbh$, and the attention dropout keeps a mask of $sbh$. The sum is $11sbh + 5as^2b$.
2. **MLP block.** The two linear layers keep their inputs: $2sbh$ for the one of width $h$, and $8sbh$ for the one of width $4h$. GeLU keeps its input, $8sbh$, and dropout keeps a mask, $sbh$. The sum is $19sbh$.
3. **Layer norms.** Two layer norms keep their inputs, $2sbh$ each, which is $4sbh$.
4. The total is $(11 + 19 + 4)\,sbh + 5as^2b = sbh(34 + 5as/h)$.
5. The softmax output, its mask and its dropout output are the $5as^2b$ term. They are cheap to recompute from Q and K. Storing everything except them leaves $34\,sbh$, which is the selective recomputation of Section 5.

### Result and check

For GPT-3 ($a = 96$, $s = 2048$, $h = 12288$), $5as/h = 80$, and the share is $80/114 \approx 0.70$.
For MT-NLG ($a = 128$, $h = 20480$), $5as/h = 64$, and the share is $64/98 \approx 0.65$.
These match the 70% and 65% of Section 5.
The check adds the terms in exact integer and rational arithmetic for the 22B, GPT-3, MT-NLG and 1T configurations of Table 3 and for a small case.
It asserts the three block sums and the total, and prints the bytes per layer.
At $b = 1$, GPT-3 keeps $2{,}868{,}903{,}936$ bytes per layer. Selective recomputation leaves $855{,}638{,}016$, and full recomputation of the layer, which keeps only the $2sbh$ input, leaves $50{,}331{,}648$. These byte counts are our own arithmetic.
