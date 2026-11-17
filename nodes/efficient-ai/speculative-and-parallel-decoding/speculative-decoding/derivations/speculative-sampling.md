---
id: speculative-sampling
derives: [c-ss-exact, c-ss-accept-mass, c-ss-tv, c-ss-extremes, c-ss-greedy, c-ss-zero]
kind: closed-form
assumes: ["p and q are probability distributions on one finite vocabulary", "the draft token is drawn from q and the uniform variable is independent of it"]
uses: []
status: full
check: speculative-sampling.check.py
lean: null
sources:
  - {work: leviathan-2023-fast-inference-speculative-decoding, locator: "p. 11"}
  - {work: leviathan-2023-fast-inference-speculative-decoding, locator: "p. 3"}
  - {work: chen-2023-accelerating-llm-decoding-speculative-sampling, locator: "p. 10"}
---

### Statement

Let $p$ (target) and $q$ (draft) be distributions on a finite vocabulary $V$.
Draw $x \sim q$ and an independent $u \sim U[0, 1)$; output $x$ if $u < \min(1, p(x)/q(x))$, and otherwise output a fresh draw $y \sim p'$ with $p'(y) = \max(0, p(y) - q(y)) / \sum_z \max(0, p(z) - q(z))$.
Then (i) the output $Y$ has law exactly $p$;
(ii) the probability of acceptance is $\beta = \sum_x \min(p(x), q(x))$;
(iii) $\beta = 1 - \mathrm{TV}(p, q)$ with $\mathrm{TV}(p, q) = \tfrac12 \sum_x |p(x) - q(x)|$;
(iv) $\beta = 1$ exactly when $p = q$ and $\beta = 0$ exactly when $p$ and $q$ have disjoint supports;
(v) if $p$ puts all its mass on one token $t$ (greedy decoding), the output is always $t$, and the draft is accepted exactly when $x = t$;
(vi) a token with $p(y) = 0$ is never output.

### Assumptions

1. $p$ and $q$ are probability distributions on the same finite set.
2. $x \sim q$, and $u$ and the resampling draw are independent of $x$ and of each other.
3. The residual $p'$ is used only after a rejection, which has probability 0 when $p = q$, so its normalizer is positive whenever it is used.

### Steps

1. Acceptance and output $x$: $P(\text{accept}, Y = x) = q(x) \min(1, p(x)/q(x)) = \min(q(x), p(x))$, also when $q(x) = 0$, where both sides are 0.
2. Summing over $x$ gives (ii): $\beta = \sum_x \min(p(x), q(x))$.
3. Since $\sum_x p(x) = 1$, $1 - \beta = \sum_x (p(x) - \min(p(x), q(x))) = \sum_x \max(0, p(x) - q(x))$, which is the normalizer of $p'$.
4. Rejection then output $y$: $P(\text{reject}, Y = y) = (1 - \beta)\, p'(y) = \max(0, p(y) - q(y))$.
5. Adding steps 1 and 4: $P(Y = y) = \min(p(y), q(y)) + \max(0, p(y) - q(y)) = p(y)$, which is (i).
6. For (iii), $|a - b| = a + b - 2\min(a, b)$, so $\tfrac12 \sum_x |p(x) - q(x)| = \tfrac12 (1 + 1 - 2\beta) = 1 - \beta$.
7. For (iv), $\min(p(x), q(x)) \le p(x)$ with equality for every $x$ iff $p \le q$ pointwise, which for two distributions means $p = q$; and $\sum_x \min(p(x), q(x)) = 0$ iff every term is 0, iff no $x$ has both $p(x) > 0$ and $q(x) > 0$.
8. For (v), $p(x) / q(x)$ is 0 for $x \ne t$, so those drafts are rejected surely, and $p' \propto \max(0, p - q)$ is positive only at $t$; the draft $x = t$ has $p(t)/q(t) = 1/q(t) \ge 1$ and is accepted surely.
9. For (vi), $p(y) = 0$ gives $\min(p(y), q(y)) = 0$ and $\max(0, p(y) - q(y)) = 0$.

### Result and check

The check computes the law of the output with exact fractions for every pair of distributions on 2 and 3 tokens whose probabilities are multiples of 1/4 and 1/6 and finds it equal to $p$, with $\beta = \sum \min(p, q) = 1 - \mathrm{TV}$; it then samples the procedure 200,000 times on random pairs, including a pair with disjoint supports and a one-hot target, and compares the frequencies with $p$.
