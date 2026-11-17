---
id: parameter-count
derives: [c-count, c-gpt3-counts, c-checkpoint-size, eq-lora-count]
kind: count
assumes: ["one adapter of rank r on each adapted matrix", "each adapted matrix of GPT-3 175B is 12,288 by 12,288", "GPT-3 175B has 96 layers", "an entry stored in FP16 takes 2 bytes"]
uses: [matrix-multiplication@m2]
status: full
check: parameter-count.check.py
lean: null
sources:
  - {work: hu-2021-lora-low-rank-adaptation, locator: "Section 4.1"}
  - {work: hu-2021-lora-low-rank-adaptation, locator: "Section 7.1"}
  - {work: hu-2021-lora-low-rank-adaptation, locator: "Section 1"}
---

### Statement

1. An adapter of rank $r$ on a weight $W_0 \in \mathbb{R}^{d \times k}$ has $r(d + k)$ trainable entries, against $dk$ for the weight itself. The ratio is $r(d + k)/(dk) = r\,(1/d + 1/k)$.
2. For GPT-3 175B with adapters on $W_q$ and $W_v$ of all 96 layers, each of shape $12{,}288 \times 12{,}288$, the count is $4{,}718{,}592\, r$: 4.7 million for $r = 1$, 18.9 million for $r = 4$ and 37.7 million for $r = 8$.
3. For $r = 4$ in FP16, the adapter takes $37{,}748{,}736$ bytes, about 36 MiB.

### Assumptions

1. Each adapted matrix gets one pair $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, and nothing else is trained. Used in steps 1 and 3.
2. In GPT-3 175B, $d_\text{model} = 12{,}288$ (Hu et al. 2021, Section 1, give the full rank $d$ as 12,288), and $W_q$ and $W_v$ are treated as $d_\text{model} \times d_\text{model}$ matrices (Section 4.2). Used in step 3.
3. There are 96 layers (Section 7.1). Used in step 3.
4. An FP16 number takes 16 bits, that is 2 bytes. Used in step 5.

### Steps

1. $B$ has $d$ rows and $r$ columns, so $dr$ entries; $A$ has $r$ rows and $k$ columns, so $rk$ entries. By assumption 1, one adapter has $dr + rk = r(d + k)$ trainable entries.
2. The ratio to the $dk$ entries of $W_0$ is $r(d + k)/(dk) = r/k + r/d$, by splitting the fraction. For $d = k$ it is $2r/d$.
3. By assumptions 2 and 3, each adapter on GPT-3 has $r(12{,}288 + 12{,}288) = 24{,}576\, r$ entries, and there are $2 \times 96 = 192$ adapters. The total is $192 \times 24{,}576\, r = 4{,}718{,}592\, r$.
4. Substituting: $r = 1$ gives $4{,}718{,}592 \approx 4.7$ million; $r = 4$ gives $18{,}874{,}368 \approx 18.9$ million; $r = 8$ gives $37{,}748{,}736 \approx 37.7$ million. These are the 4.7M and 37.7M rows of Table 4 of Hu et al. (2021) and the budget of about 18M of their Section 7.1, which is also $r = 8$ on one type of matrix: $96 \times 24{,}576 \times 8 = 18{,}874{,}368$.
5. By assumption 4, $r = 4$ in FP16 takes $2 \times 18{,}874{,}368 = 37{,}748{,}736$ bytes. Dividing by $2^{20} = 1{,}048{,}576$ gives 36.0 MiB, the "roughly 35MB" of Hu et al. (2021, Section 7.1).
6. With the 175 billion parameters of the model, the share is $18{,}874{,}368 / (175 \times 10^9) \approx 1.08 \times 10^{-4}$, about one in 9,300: the order of the reduction "by 10,000 times" of their abstract.

### Result and check

$$ |\Theta| = r\,(d + k) \text{ per adapted matrix}, \qquad |\Theta_\text{GPT-3},\, W_q, W_v| = 4{,}718{,}592\, r . $$

The check `parameter-count.check.py` builds the adapter arrays of a small model and counts their entries, and evaluates the GPT-3 counts and sizes of steps 3 to 6.
