---
id: expert-capacity
derives: [c-capacity-count, c-capacity-balanced, c-capacity-forced-drops, c-capacity-worked, c-capacity-gshard-group, eq-capacity]
kind: count
assumes: ["a batch or group of T tokens routed over N experts, each token making k distinct choices", "the capacity C = floor(k T c / N) slots per expert for a capacity factor c > 0", "an expert keeps the first C choices it receives and drops the rest"]
uses: []
status: full
check: expert-capacity.check.py
lean: null
sources:
  - {work: fedus-2022-switch-transformers, locator: "p. 7"}
  - {work: lepikhin-2020-gshard, locator: "p. 6"}
  - {work: shazeer-2017-sparsely-gated-mixture-of-experts, locator: "p. 4"}
---

### Statement

Let expert $i$ receive $n_i$ of the $kT$ choices, so $\sum_i n_i = kT$, and let $C = \lfloor kTc/N \rfloor$.
Then

1. expert $i$ processes $\min(n_i, C)$ choices, the layer drops $D = \sum_i \max(0, n_i - C)$ choices, and of the $NC$ slots that a static-shape implementation computes, $NC - (kT - D)$ are padding;
2. under perfectly even routing, $n_i = kT/N$ with $kT/N$ an integer, and $c \ge 1$, no choice is dropped, and the fraction of computed slots that hold padding is $1 - kT/(NC)$, close to $1 - 1/c$;
3. for any routing, $D \ge kT - NC \ge kT(1 - c)$, so a capacity factor below 1 drops at least a fraction $1 - c$ of the choices; and if every choice goes to one expert, $D = kT - \min(kT, C)$;
4. on the worked example, $T = 8$ tokens, $N = 4$ experts, $k = 1$, loads $(4, 2, 1, 1)$: $c = 1$ gives $C = 2$ and drops 2 tokens with 6 of 8 slots used; $c = 1.5$ gives $C = 3$ and drops 1 token with 7 of 12 slots used; $c = 0.5$ gives $C = 1$ and drops 4 tokens with all 4 slots used;
5. GShard's per-group capacity $2N_{\text{batch}}/(G E)$ for top-2 routing over $G$ groups of $S = N_{\text{batch}}/G$ tokens is the formula with $k = 2$, $T = S$, $N = E$ and $c = 1$.

### Assumptions

1. Each token's $k$ choices are distinct experts, so no expert receives the same token twice.
2. Capacity is rounded down; Switch Transformers writes the capacity as tokens per batch over experts times the capacity factor without saying how it is rounded, and an implementation that rounds up keeps a slot more per expert.
3. Dropping is the only overflow rule; Switch Transformers and GShard both pass a dropped token on through the residual connection.

### Steps

1. Expert $i$ has $C$ slots and receives $n_i$ choices, so it fills $\min(n_i, C)$ and drops $n_i - \min(n_i, C) = \max(0, n_i - C)$. Summing over experts gives $D$, and the kept choices number $kT - D$. The layer computes all $NC$ slots, so the padding is $NC - (kT - D)$.
2. With $n_i = kT/N$ and $c \ge 1$, $C = \lfloor kTc/N\rfloor \ge kT/N = n_i$ because $kT/N$ is an integer no larger than $kTc/N$; so every $\max(0, n_i - C) = 0$. The padding fraction is $(NC - kT)/(NC) = 1 - kT/(NC)$, and since $kTc/N - 1 < C \le kTc/N$ it tends to $1 - 1/c$ as $kT/N$ grows.
3. At most $C$ choices fit in each expert, so at most $NC$ are kept and $D \ge kT - NC$. Since $C \le kTc/N$, $NC \le kTc$ and $kT - NC \ge kT(1-c)$. If all choices go to expert $j$, $n_j = kT$ and the others are zero; only possible with $k = 1$ under assumption 1, and then $D = T - \min(T, C)$.
4. $C = \lfloor 8 \cdot 1/4 \rfloor = 2$: drops $\max(0, 4-2) = 2$, kept $6$ of $NC = 8$ slots. $C = \lfloor 8 \cdot 1.5/4\rfloor = 3$: drops $1$, kept $7$ of $12$. $C = \lfloor 8 \cdot 0.5/4 \rfloor = 1$: drops $3 + 1 + 0 + 0 = 4$, kept $4$ of $4$; the bound of item 3 gives $D \ge 8 - 4 = 4$, met with equality.
5. Substituting $k = 2$, $T = N_{\text{batch}}/G$, $N = E$, $c = 1$ gives $C = 2N_{\text{batch}}/(GE)$, rounded down. Shazeer et al. make the same count for the batch an expert sees: with $b$ examples and $k$ of $n$ experts, about $kb/n$ examples each, and $kbd/n$ when $d$ data-parallel batches are combined.

### Result and check

$$ C = \left\lfloor \frac{kTc}{N} \right\rfloor,\qquad D = \sum_{i=1}^N \max(0, n_i - C) \ge kT - NC \ge kT(1-c),\qquad \text{padding} = NC - kT + D. $$

The check `expert-capacity.check.py` routes 3,000 random batches with $N$ from 2 to 16, $k$ from 1 to 3 and $c$ from 0.25 to 3, simulates the slot-by-slot dispatch, and compares the kept, dropped and padded counts with item 1 and the bounds of item 3.
It confirms item 2 on even routings, recomputes the worked example of item 4, and checks the GShard form of item 5 for several group counts.
