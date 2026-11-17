---
id: patience
derives: [c-patience-overshoot, c-patience-best, c-patience-unimodal, c-patience-plateau, c-patience-bias, eq-patience]
kind: proof
assumes: ["evaluations e_1, e_2, ... of a loss to minimize, possibly NaN", "patience p >= 1 and min_delta >= 0"]
uses: []
status: full
check: patience.check.py
lean: null
sources: []
---

### Statement

The patience rule with patience $p$ and threshold $\delta \ge 0$ reads evaluations $e_1, e_2, \dots$ in order.
Evaluation $t$ improves when it is not NaN and $e_t < B_{t-1} - \delta$, where $B_0 = +\infty$ and $B_t$ is the value of the last improvement up to $t$.
The rule stops at the first $T$ at which the last $p$ evaluations did not improve, and returns the epoch $b$ of the last improvement with its state.
For every sequence:

1. $T = b + p$: the rule always trains exactly $p$ evaluations past the one it returns;
2. every evaluation before $b$ is strictly worse than $e_b$, and every one after is at least $e_b - \delta$, so $e_b \le \min_{t \le T} e_t + \delta$, with equality to the minimum when $\delta = 0$;
3. if the sequence decreases strictly to epoch $m$ and never decreases after it, and $\delta = 0$, the rule returns $m$ and stops at $m + p$.

It does not guarantee:

4. that $b$ is the minimum of the whole curve: for every $p$, a plateau of $p$ evaluations followed by a drop stops before the drop;
5. that $e_b$ estimates the error at $b$ without bias: it is a minimum of noisy values, and is low in expectation;
6. anything about a test set, which the rule never reads.

### Assumptions

1. Lower is better. For a metric to maximize, apply the rule to $-e_t$. Used in all steps.
2. The first evaluation that is not NaN improves, because $B_0 = +\infty$. Used in steps 1 and 2.
3. For item 5, the evaluations are a fixed value plus independent noise of mean zero. Used in step 5.

### Steps

1. After the improvement at $b$ none follows, so the count of evaluations without improvement is $t - b$ for $t > b$. It first reaches $p$ at $t = b + p$, and it was below $p$ at every earlier $t$, since before $b$ any count was reset at each improvement and the rule did not stop.
2. Improvements form a chain $e_{b_1} > e_{b_2} + \delta > \dots$ that ends at $b$, and $B_t$ does not increase. Take $t < b$. If $t$ improved, a later improvement is below $e_t - \delta$, so $e_b < e_t$. If $t$ did not improve, $e_t \ge B_{t-1} - \delta > e_b$, because the improvement at or after $t$ that leads to $b$ lies below $B_{t-1} - \delta$. Take $b < t \le T$: $t$ did not improve, so $e_t \ge B_{t-1} - \delta = e_b - \delta$.
3. Under the shape assumption every $t \le m$ improves when $\delta = 0$, and no $t > m$ does, since $e_t \ge e_m$. Step 1 gives $T = m + p$.
4. Take $e_1 = \dots = e_{p+1} = 1$ and $e_{p+2} = 1/2$. Evaluations $2, \dots, p+1$ tie and do not improve, so $T = p + 1$ and the drop at $p + 2$ is never read.
5. $e_b \le e_t + \delta$ for every $t$ read, so with $\delta = 0$, $\mathbb{E}[e_b] \le \mathbb{E}[\min_{t \le T} e_t] < $ the fixed value when the noise is not degenerate. The size of the bias depends on $p$ and the noise and is not given by the rule.
6. The rule is a function of $e_1, \dots, e_T$ only.

### Result and check

$$ T = b + p, \qquad e_b \le \min_{t \le T} e_t + \delta . $$

The check `patience.check.py` runs the rule on 5000 random walks with $p$ from 1 to 7 and $\delta \in \{0, 0.01, 0.1\}$, and asserts items 1 and 2 on each.
It asserts item 3 on curves with minima at epochs 1, 5 and 30 for $p = 1, 3, 10$, and item 4 for $p = 1, \dots, 20$.
On a flat curve of value 1 with Gaussian noise of standard deviation 0.05 and $p = 10$, the mean kept value over 20000 runs is 0.090 below the true value; this number is from our own run.
