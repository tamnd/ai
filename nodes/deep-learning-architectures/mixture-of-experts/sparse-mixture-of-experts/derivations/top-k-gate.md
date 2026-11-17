---
id: top-k-gate
derives: [c-topk-equivalence, eq-noisy-top-k]
kind: identity
assumes: ["h is a finite real vector of length N", "1 <= k <= N", "the same k indices are kept in both forms, ties broken the same way"]
uses: []
status: full
check: top-k-gate.check.py
lean: null
sources:
  - {work: shazeer-2017-sparsely-gated-mixture-of-experts, locator: "p. 4"}
  - {work: lepikhin-2020-gshard, locator: "p. 7"}
---

### Statement

Let $S$ be the set of the indices of the $k$ largest entries of $h \in \mathbb{R}^N$, and let $\mathrm{KeepTopK}(h, k)_i = h_i$ for $i \in S$ and $-\infty$ otherwise.
Then for every $i$,

$$ \mathrm{softmax}(\mathrm{KeepTopK}(h, k))_i = \begin{cases} \dfrac{\mathrm{softmax}(h)_i}{\sum_{j \in S} \mathrm{softmax}(h)_j} & i \in S, \\ 0 & i \notin S. \end{cases} $$

So the noisy top-k gate of Shazeer et al. and the renormalized top-2 gates of GShard's Algorithm 1 are the same function of the logits, and for $k = 1$ the gate is 1.

### Assumptions

1. The entries of $h$ are finite, so every $e^{h_j}$ is positive and finite.
2. $S$ is the same set in both forms; with ties, both must break them the same way.
3. $e^{-\infty} = 0$, the usual convention for the masked softmax.

### Steps

1. For $i \notin S$, the numerator of the left side is $e^{-\infty} = 0$; the denominator is $\sum_{j \in S} e^{h_j} > 0$, so the left side is 0.
2. For $i \in S$, the left side is $e^{h_i} / \sum_{j \in S} e^{h_j}$.
3. Write $Z = \sum_{j=1}^{N} e^{h_j}$. Then $\mathrm{softmax}(h)_i / \sum_{j \in S} \mathrm{softmax}(h)_j = (e^{h_i}/Z) / (\sum_{j \in S} e^{h_j} / Z) = e^{h_i} / \sum_{j \in S} e^{h_j}$, the same as step 2.
4. With $k = 1$ and $S = \{i\}$, the value is $e^{h_i}/e^{h_i} = 1$.

### Result and check

The two forms agree entry by entry.
The check compares them on random logit vectors for every $k$ from 1 to $N$, and confirms that the $k = 1$ gate is 1 and that the gate sums to 1.
