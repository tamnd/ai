---
id: error-barrier
derives: [c-barrier-bounds, c-barrier-grid, c-barrier-permutation]
kind: closed-form
assumes: ["E maps a network to a real error", "the alphas of the grid include 0 and 1"]
uses: []
status: full
check: error-barrier.check.py
lean: null
sources:
  - {work: frankle-2019-linear-mode-connectivity-lottery, locator: "p. 2"}
  - {work: frankle-2019-linear-mode-connectivity-lottery, locator: "p. 3"}
---

### Statement

Let $E(W)$ be the error of a network with weights $W$, and let $W_1$ and $W_2$ be two networks of the same architecture.
Frankle et al. (2020) define the error on the line between them, its highest value, and the mean of the ends (p. 2 and p. 3):

$$ E_\alpha(W_1, W_2) = E(\alpha W_1 + (1 - \alpha) W_2), \qquad E_{\sup} = \sup_{\alpha \in [0, 1]} E_\alpha, \qquad \bar E = \tfrac12 \big(E(W_1) + E(W_2)\big). $$

The error barrier height is $B = E_{\sup} - \bar E$.
Then (i) $B \ge |E(W_1) - E(W_2)|/2 \ge 0$; (ii) if $E$ is convex in $W$, $B = |E(W_1) - E(W_2)|/2$; (iii) the estimate $\hat B$ from $m$ evenly spaced alphas that include 0 and 1 satisfies $\hat B \le B \le \hat B + L / (2(m - 1))$ when $\alpha \mapsto E_\alpha$ is $L$-Lipschitz; (iv) $B$ can be large even when $E(W_1) = E(W_2) = 0$ and both compute the same function.

### Assumptions

1. $E$ is a real function of the weights; the paper uses the test or train error.
2. The grid of alphas includes both ends; the paper uses 30 evenly spaced values (p. 3).
3. For (iii), $E_\alpha$ is $L$-Lipschitz in $\alpha$. The 0-1 error is not continuous; (iii) then bounds only the smooth part.

### Steps

1. At $\alpha = 1$ and $\alpha = 0$ the path gives $E(W_1)$ and $E(W_2)$. So $E_{\sup} \ge \max(E(W_1), E(W_2))$, and $\max(u, v) - (u + v)/2 = |u - v|/2 \ge 0$. This is (i).
2. If $E$ is convex, $E_\alpha \le \alpha E(W_1) + (1 - \alpha) E(W_2) \le \max(E(W_1), E(W_2))$. With step 1, $E_{\sup} = \max(E(W_1), E(W_2))$, which gives (ii). So a nonzero barrier between two networks of equal error needs a nonconvex error.
3. The grid is a subset of $[0, 1]$, so its maximum is at most the sup. The sup is attained at some $\alpha^*$ when $E_\alpha$ is continuous, and a grid point lies within $1/(2(m - 1))$ of it; the Lipschitz bound gives (iii). With $m = 30$ the gap is at most $L/58$.
4. Permute the hidden units of a one-hidden-layer ReLU network: permute the columns of the first layer and the rows of the second. The function is unchanged, so both networks have the same error. The midpoint averages different units, and its function differs. This is (iv).

### Result and check

Frankle et al. call a network stable to SGD noise when the barrier between two copies trained with different noise is below 2%, and say the two copies are then linearly mode connected (p. 3).
Step 1 shows why they subtract the mean of the ends: the barrier measures the rise above the ends, and it is zero for two equal minima joined by a flat line.
`error-barrier.check.py` checks (ii) on the squared error of a linear model, (i) and (iii) on a nonconvex error with 30 alphas against 2,901, and (iv) on a 10-8-3 ReLU network.
There, labels come from the first network, both networks have zero 0-1 error, and the barrier over 30 alphas is 0.237.
