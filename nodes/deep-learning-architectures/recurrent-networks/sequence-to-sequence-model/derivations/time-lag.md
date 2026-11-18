---
id: time-lag
derives: [c-lag-forward, c-lag-reversed]
kind: count
assumes: ["a source of n words read at steps 1 to n, and a target of n words of which word i is predicted at step n + i", "source word i corresponds to target word i"]
uses: []
status: full
check: time-lag.check.py
lean: null
sources:
  - {work: sutskever-2014-sequence-to-sequence-learning, locator: "Section 3.3"}
---

### Statement

1. When the source is read in its own order, every source word is $n$ steps from the target word it corresponds to.
2. When the source is reversed, source word $i$ is $2i - 1$ steps from target word $i$. The average distance is still $n$, but the smallest is 1 and the first $k$ pairs are all within $2k - 1$ steps.

### Assumptions

1. The encoder reads one word per step and the decoder predicts one word per step right after it, with no other steps in between. Used in steps 1 and 2.
2. The correspondence is monotone, word $i$ to word $i$, the case that Sutskever et al. describe for a, b, c and α, β, γ. Used in steps 1 and 2.

### Steps

1. In order, source word $i$ is read at step $i$ and target word $i$ is predicted at step $n + i$; the distance is $n + i - i = n$ for every $i$. This is statement 1.
2. Reversed, source word $i$ is read at step $n + 1 - i$; the distance is $n + i - (n + 1 - i) = 2i - 1$. The sum $\sum_{i=1}^{n} (2i - 1) = n^2$ equals $\sum_{i=1}^{n} n$, so the averages agree; the minimum, at $i = 1$, is 1, and the distance grows with $i$. This is statement 2.

### Result and check

$$ \text{lag}_{\text{in order}}(i) = n, \qquad \text{lag}_{\text{reversed}}(i) = 2i - 1, \qquad \frac{1}{n} \sum_{i=1}^{n} (2i - 1) = n . $$

The check `time-lag.check.py` compares `time_lags` of the NumPy reference with these formulas for $n$ up to 200, and checks that the reversed model is the same function as the model in order applied to the reversed source.
