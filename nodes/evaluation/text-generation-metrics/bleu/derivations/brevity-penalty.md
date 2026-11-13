---
id: brevity-penalty
derives: [c-bp-range, c-score-range, eq-brevity-penalty, eq-bleu]
kind: bound
assumes: ["c > 0 hypothesis tokens in the test set", "r > 0, the effective reference length", "each modified precision p_n is in [0, 1]", "uniform weights 1/K"]
uses: [precision-and-recall@al]
status: full
lean: null
sources:
  - {work: papineni-2002-bleu-acl, locator: "Sections 2.2.2 and 2.3"}
---

### Statement

Let $c > 0$ be the number of hypothesis tokens of the test set and $r > 0$ its effective reference length.
With $\mathrm{BP} = 1$ when $c \ge r$ and $\mathrm{BP} = e^{1 - r/c}$ when $c < r$, as in eq-brevity-penalty:

1. $0 < \mathrm{BP} \le 1$, and $\mathrm{BP} = 1$ exactly when $c \ge r$.
2. $\log \mathrm{BP} = \min(1 - r/c,\ 0)$, so the two forms of eq-bleu are the same.
3. $\mathrm{BP}$ is continuous in $c$ at $c = r$, and $\mathrm{BP} \to 0$ as $c \to 0$ with $r$ fixed.
4. $0 \le \mathrm{BLEU} \le 1$, and $\mathrm{BLEU} = 1$ exactly when $p_1 = \dots = p_K = 1$ and $c \ge r$.

### Assumptions

1. $c > 0$: the test set has at least one hypothesis token. It is used in every step that divides by $c$. The case $c = 0$ is set to $\mathrm{BP} = 0$ by definition and is not derived.
2. $r > 0$: every segment has a reference of at least one token, so the sum of the closest lengths is positive. It is used in step 3 and step 7.
3. $0 \le p_n \le 1$ for $n = 1, \dots, K$: a clipped count is at most the count it clips, so the numerator of eq-modified-precision is at most its denominator. It is used in step 8.
4. The weights are $1/K$ for each order. It is used in step 8.

### Steps

1. If $c \ge r$, then $\mathrm{BP} = 1$, by the definition of eq-brevity-penalty.
2. If $c < r$, then $r/c > 1$ by assumption 1, because dividing both sides of $c < r$ by $c > 0$ keeps the order. So $1 - r/c < 0$.
3. The exponential function is positive and increasing, and $e^0 = 1$. From step 2, $0 < e^{1 - r/c} < 1$ when $c < r$.
4. Steps 1 and 3 together give $0 < \mathrm{BP} \le 1$, with $\mathrm{BP} = 1$ in the case $c \ge r$ and $\mathrm{BP} < 1$ in the case $c < r$. This is part 1.
5. In the case $c \ge r$, $1 - r/c \ge 0$, so $\min(1 - r/c, 0) = 0 = \log 1 = \log \mathrm{BP}$. In the case $c < r$, $1 - r/c < 0$ by step 2, so $\min(1 - r/c, 0) = 1 - r/c = \log e^{1 - r/c} = \log \mathrm{BP}$. Both cases give $\log \mathrm{BP} = \min(1 - r/c, 0)$, which is part 2. Taking the logarithm of the product $\mathrm{BP} \cdot \exp(\frac{1}{K}\sum_n \log p_n)$ gives the second form of eq-bleu, when every $p_n > 0$.
6. The map $c \mapsto 1 - r/c$ is continuous on $c > 0$, and so is $\min(\cdot, 0)$, so $\log \mathrm{BP}$ is continuous on $c > 0$ by step 5, as a composition of continuous maps. At $c = r$ it is $\min(0, 0) = 0$ from both sides. The exponential of a continuous function is continuous, so $\mathrm{BP}$ is continuous at $c = r$ with the value 1.
7. As $c \to 0$ from above with $r > 0$ fixed (assumption 2), $r/c \to +\infty$, so $1 - r/c \to -\infty$ and $e^{1 - r/c} \to 0$, because $e^x \to 0$ as $x \to -\infty$. For $c < r$ this is $\mathrm{BP}$, so $\mathrm{BP} \to 0$. This and step 6 are part 3.
8. By assumptions 3 and 4, if some $p_n = 0$ then BLEU is 0, by the rule of eq-bleu. Otherwise each $\log p_n \le 0$, so $\frac{1}{K}\sum_n \log p_n \le 0$ and $0 < \exp(\frac{1}{K}\sum_n \log p_n) \le 1$, with equality exactly when every $\log p_n = 0$, that is every $p_n = 1$. Multiplying by $\mathrm{BP} \in (0, 1]$ from step 4 keeps the product in $[0, 1]$, and the product is 1 exactly when both factors are 1, because each factor is at most 1. So $\mathrm{BLEU} = 1$ exactly when every $p_n = 1$ and $c \ge r$. This is part 4.

### Result and check

For $c > 0$, $\mathrm{BP} = \min(1, e^{1 - r/c})$ is in $(0, 1]$, equals 1 exactly when $c \ge r$, is continuous at $c = r$ and falls to 0 as the output gets shorter; BLEU is in $[0, 1]$ and is 1 only for full clipped precision at every order with output at least as long as the effective reference length.
The penalty is computed once from the sums $c$ and $r$ of the whole test set, not for each segment, so a short segment can be balanced by a long one (Papineni et al., 2002, Section 2.2.2).

The check is numeric, in the tests of `impl/`: `test_range_and_bp` draws 200 random test sets for each smoothing and asserts $0 \le \mathrm{BLEU} \le 100$, $\mathrm{BP} = 1$ when $c \ge r$ and $0 < \mathrm{BP} < 1$ otherwise; `test_worked_example` checks $\mathrm{BP} = e^{-1/6}$ for $c = 6$ and $r = 7$; `test_brevity_penalty_is_corpus_level` checks that a short and a long segment give a penalty of 1; and the row `empty-hypothesis` of `vectors/edge.jsonl` checks $\mathrm{BP} = 0$ for $c = 0$.
