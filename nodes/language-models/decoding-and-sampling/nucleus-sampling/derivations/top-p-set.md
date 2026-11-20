---
id: top-p-set
derives: [c-tp-prefix, c-tp-mass, c-tp-tv, c-tp-limits, c-tp-monotone, eq-top-p-set, eq-renormalize]
kind: proof
assumes: ["P is a distribution on a finite vocabulary V", "0 < p ≤ 1", "ties among equal probabilities are broken by a fixed order of the tokens"]
uses: [conditional-probability@m2]
status: full
check: top-p-set.check.py
lean: null
sources:
  - {work: holtzman-2020-curious-case-neural-text-degeneration, locator: "p. 4"}
  - {work: holtzman-2020-curious-case-neural-text-degeneration, locator: "p. 5"}
---

### Statement

Let $P$ be a distribution on a finite vocabulary $V$ of size $|V|$, sort its probabilities as $q_1 \ge q_2 \ge \dots \ge q_{|V|}$, let $0 < p \le 1$, and let $m$ be the least index with $q_1 + \dots + q_m \ge p$. Then

1. the first $m$ tokens form a smallest set with total probability at least $p$, as eq-top-p-set asks, and every smallest such set has exactly $m$ tokens;
2. the kept mass $p' = q_1 + \dots + q_m$ satisfies $p \le p' < p + q_m$;
3. the total variation distance between $P$ and the renormalized $P'$ of eq-renormalize is $1 - p'$, so it is at most $1 - p$ for the nucleus; for top-$k$ it is $1 - p'_k$ with $p'_k = q_1 + \dots + q_k$, which depends on the context;
4. if $q_1 \ge p$ the nucleus is the single top token, and if $p > 1 - q_s$, where $q_s$ is the smallest positive probability, the nucleus is every token with positive probability;
5. with ties broken by a fixed order, the nucleus is monotone in $p$: if $p_1 \le p_2$ then $V^{(p_1)} \subseteq V^{(p_2)}$.

### Assumptions

1. $P$ is a distribution on a finite set, so the sorted list exists and its partial sums reach $1 \ge p$; $m$ is well defined.
2. $0 < p \le 1$. At $p = 0$ the empty set would qualify, and no decoder uses it.
3. Equal probabilities are ordered by a fixed rule, such as the token index. Without it the smallest set can fail to be unique, though its size $m$ and mass $p'$ do not change.

### Steps

1. Any set $S$ of $j$ tokens has mass at most $q_1 + \dots + q_j$, the sum of the $j$ largest probabilities. If $j < m$ this is below $p$ by the choice of $m$, so no set with fewer than $m$ tokens reaches $p$. The first $m$ tokens reach it, so $m$ is the smallest size, and a smallest set has $m$ tokens by definition.
2. $p' \ge p$ by the choice of $m$. Since $m$ is least, $q_1 + \dots + q_{m-1} < p$ (the empty sum $0$ when $m = 1$), and adding $q_m$ gives $p' < p + q_m$.
3. Write $C$ for the kept set. The distance is $\tfrac12 \sum_x |P(x) - P'(x)|$. On $C$, $P'(x) - P(x) = P(x)(1/p' - 1) \ge 0$, and these differences sum to $p'(1/p' - 1) = 1 - p'$. Off $C$, $P(x) - P'(x) = P(x)$, summing to $1 - p'$. Half the total is $1 - p'$, and step 2 bounds it by $1 - p$. The same computation with $C$ the top $k$ tokens gives $1 - p'_k$, which has no bound of its own: it is near $0$ for a peaked $P$ and near $1 - k/|V|$ for a flat one.
4. If $q_1 \ge p$ then $m = 1$. If $p > 1 - q_s$, a set that misses a token $x$ with $P(x) > 0$ has mass at most $1 - P(x) \le 1 - q_s < p$, so the nucleus contains every such token. The positive tokens come first in the order and already sum to $1 \ge p$, so the nucleus is exactly them.
5. With a fixed order the sorted list does not depend on $p$, and $m(p)$, the least index whose partial sum reaches $p$, is nondecreasing in $p$. The nucleus is the first $m(p)$ entries of one list, so the sets are nested.

### Result and check

$$ V^{(p)} = \{x_{(1)}, \dots, x_{(m)}\}, \qquad p \le p' < p + q_m, \qquad d_{\mathrm{TV}}(P, P') = 1 - p' \le 1 - p . $$

The check `top-p-set.check.py` builds random distributions on up to 9 tokens, finds by brute force every smallest set with mass at least $p$, and compares its size with $m$ and the sorted prefix with one of them. It tests the bounds of item 2, computes the distance of item 3 directly, checks the two limits of item 4, and checks over a grid of $p$ that the nuclei are nested. It also evaluates the worked example of the node: $P = (0.5, 0.2, 0.15, 0.1, 0.05)$ with $p = 0.8$ keeps 3 tokens, $P' = (10/17, 4/17, 3/17, 0, 0)$, and the distance is $0.15$.
