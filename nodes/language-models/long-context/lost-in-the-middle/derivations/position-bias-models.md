---
id: position-bias-models
derives: [c-pb-calibration, c-pb-calibration-noise, c-pb-sort-monotone, c-pb-sort-one-step, c-pb-sort-stall, c-pb-rescale]
kind: proof
assumes: ["the attention a document receives is a function of its relevance and its position only, as in the models of Hsieh et al. and Peysakhovich and Lerer", "RoPE rotates each pair of query and key dimensions by the position times a fixed frequency"]
uses: []
status: full
check: position-bias-models.check.py
lean: null
sources:
  - {work: hsieh-2024-found-in-the-middle-calibrating-attention, locator: "pp. 5 and 6"}
  - {work: peysakhovich-2023-attention-sorting-recency-bias, locator: "p. 5"}
  - {work: zhang-2024-found-in-the-middle-ms-poe, locator: "p. 4"}
---

### Statement

Let $K$ documents sit at positions $k = 1, \dots, K$, and let $\mathrm{Attn}(x, k)$ be the attention that document $x$ receives at position $k$.

1. (Calibration.) Suppose $\mathrm{Attn}(x, k) = \mathrm{rel}(x) + \mathrm{bias}(k) + \epsilon(x, k)$ with $|\epsilon| \le \delta$, and fix a dummy document $x^{\mathrm{dum}}$. The calibrated attention $c(x, k) = \mathrm{Attn}(x, k) - \mathrm{Attn}(x^{\mathrm{dum}}, k)$ satisfies $|c(x, k) - (\mathrm{rel}(x) - \mathrm{rel}(x^{\mathrm{dum}}))| \le 2\delta$. With $\delta = 0$ it does not depend on $k$, and for any $\delta$, if $\mathrm{rel}(x_1) - \mathrm{rel}(x_2) > 4\delta$ then $c(x_1, k) > c(x_2, l)$ for all positions $k$ and $l$.
2. (Sorting.) Suppose $\mathrm{Attn}(x, k) = f(\mathrm{rel}(x), k)$ with $f$ strictly increasing in its first argument and non-decreasing in $k$, and let the gold document $g$ be strictly the most relevant. One step of attention sorting reorders the documents by increasing attention, the highest last. Then
   1. every document before $g$ is still before $g$ after the step, so the position of $g$ never decreases;
   2. if $f(\mathrm{rel}(g), p) > f(\mathrm{rel}(x), K)$ for every other document $x$, where $p$ is the position of $g$, then $g$ is last after one step; for $f(r, k) = r + \beta k$ this holds when the relevance margin exceeds $\beta(K - p)$;
   3. the condition of item 2 cannot be dropped: with $f(r, k) = r + 0.2k$, relevances $1.0, 0.9, 0.9$ and $g$ first, $g$ stays first under every number of steps.
3. (Rescaling.) Under RoPE the score of a query at position $m$ and a key at position $n$ depends on $m - n$ only. If both positions are divided by $r > 0$, the score at distance $D$ equals the unscaled score at distance $D/r$.

### Assumptions

1. Item 1 uses the additive model of Hsieh et al. (their Eq. 2), with the noise bounded rather than random.
2. Item 2 idealizes the recency bias that Peysakhovich and Lerer find: attention never falls as a document moves later. Their models also show a weaker bias toward the first tokens, which this item leaves out.
3. Item 3 assumes the plain RoPE of Su et al., with no other position term.

### Steps

1. Write $\epsilon_1 = \epsilon(x, k)$ and $\epsilon_2 = \epsilon(x^{\mathrm{dum}}, k)$. Then $c(x, k) = \mathrm{rel}(x) - \mathrm{rel}(x^{\mathrm{dum}}) + \epsilon_1 - \epsilon_2$, the terms $\mathrm{bias}(k)$ cancelling, and $|\epsilon_1 - \epsilon_2| \le 2\delta$. With $\delta = 0$ nothing left depends on $k$. For two documents, $c(x_1, k) - c(x_2, l) \ge \mathrm{rel}(x_1) - \mathrm{rel}(x_2) - 4\delta > 0$.
2. Let $g$ sit at position $p$ and let $x$ sit at a position $k < p$. Then $f(\mathrm{rel}(x), k) \le f(\mathrm{rel}(x), p) < f(\mathrm{rel}(g), p)$, the first step because $f$ does not fall with position and the second because $f$ rises with relevance. So $x$ has less attention than $g$ and is placed before it: item 2.1. Under the condition of item 2.2, every other document $x$ at any position $k \le K$ has $f(\mathrm{rel}(x), k) \le f(\mathrm{rel}(x), K) < f(\mathrm{rel}(g), p)$, so $g$ has the most attention and is placed last. For $f(r, k) = r + \beta k$ the condition reads $\mathrm{rel}(g) - \mathrm{rel}(x) > \beta(K - p)$. For item 2.3, the attentions are $1.0 + 0.2 = 1.2$, $0.9 + 0.4 = 1.3$ and $0.9 + 0.6 = 1.5$ with positions counted from 1; they are already increasing, so the step leaves the order unchanged, and every later step repeats it.
3. RoPE multiplies the $j$-th pair of a query at position $m$ by the rotation through $m\theta_j$, and of a key at position $n$ by the rotation through $n\theta_j$. The rotations are orthogonal, so the product of the two rotated pairs is the query pair times the rotation through $(n - m)\theta_j$ times the key pair, which depends on $m - n$ only. Dividing both positions by $r$ replaces $m - n$ by $(m - n)/r$ in every pair, which is the unscaled score at distance $(m - n)/r$.

### Result and check

$$ c(x, k) = \mathrm{rel}(x) - \mathrm{rel}(x^{\mathrm{dum}}) + O(2\delta), \qquad \text{one sort suffices if } \mathrm{rel}(g) - \max_{x \ne g} \mathrm{rel}(x) > \beta(K - p) . $$

The check `position-bias-models.check.py` draws 2000 random additive models with U-shaped and other biases and bounded noise, and checks item 1 on every pair of documents. It runs attention sorting to its end on 3000 random cases with additive, multiplicative and logarithmic recency biases, checks items 2.1 and 2.2 at every step, and replays the stall of item 2.3. It compares RoPE scores with scaled positions against unscaled scores at the scaled distance for 500 random queries, keys, positions and ratios.
