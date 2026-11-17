---
id: dynamic-loss-scale
derives: [c-dyn-formula, c-dyn-start, c-dyn-cycle, c-dyn-growth, c-dyn-defaults, eq-dyn-scale]
kind: count
assumes: ["the scale halves on each overflow, which skips the step and resets the count", "the scale doubles after N steps in a row without an overflow, which resets the count", "in items 2 to 5 a step overflows exactly when its scale is at least a fixed threshold T > 0"]
uses: []
status: full
check: dynamic-loss-scale.check.py
lean: null
sources:
  - {work: micikevicius-2017-mixed-precision-training, locator: "p. 5"}
  - {work: micikevicius-2017-mixed-precision-training, locator: "p. 9"}
---

### Statement

Run the dynamic loss scale from $S_0$ with growth interval $N$: on an overflow the step is skipped, the scale halves and the count of good steps returns to 0; on a good step the count goes up by one, and when it reaches $N$ the scale doubles and the count returns to 0. Then

1. for any pattern of overflows the scale at step $t$ is
   $$ S_t = S_0\, 2^{G_t - B_t} , $$
   where $B_t$ is the number of skipped steps before $t$ and $G_t = \sum_r \lfloor \ell_r / N \rfloor$ sums over the maximal runs of good steps before $t$, of lengths $\ell_r$;

and if a step overflows exactly when its scale is at least a threshold $T$:

2. when $S_0 \ge T$ the first $\lfloor \log_2(S_0/T) \rfloor + 1$ steps are skipped, which brings the scale to $s^* = S_0 2^{j}$, the largest such value below $T$, with $T/2 \le s^* < T$;
3. from there the run repeats a cycle of $N + 1$ steps: $N$ good steps at $s^*$, then one skipped step at $2s^*$; one step in $N + 1$ is skipped;
4. when $S_0 < T$ no step is skipped until step $N \lceil \log_2(T/S_0) \rceil$, the scale doubling every $N$ steps, and the cycle of item 3 follows;
5. with the defaults of `torch.amp.GradScaler`, $S_0 = 2^{16}$ and $N = 2000$, and $T = 1000$, the first 7 steps are skipped, the scale settles at 512, and one step in 2001 is skipped.

### Assumptions

1. The growth and backoff factors are 2 and 1/2, the values of Micikevicius et al. and the defaults of PyTorch, and the scale never reaches the largest float, so it can always grow.
2. In items 2 to 5 the overflow depends only on the scale, through a threshold $T$ that does not change; a real run has a threshold that drifts with the gradients, so the result describes the regime around a slowly moving $T$.

### Steps

1. Each skip divides the scale by 2. Inside a run of good steps the count starts at 0, because the run begins at the start or after a skip, and the scale doubles at the $N$th, $2N$th, … step of the run, $\lfloor \ell_r/N \rfloor$ times in a run of length $\ell_r$. The scale is $S_0$ times 2 to the number of doublings minus the number of halvings.
2. While $S \ge T$ every step overflows and halves the scale. Starting from $S_0 \ge T$, the scales $S_0, S_0/2, \ldots, S_0 2^{-j}$ are skipped as long as $S_0 2^{-j} \ge T$, that is for $j = 0, \ldots, \lfloor \log_2(S_0/T) \rfloor$. The first scale below $T$ is $s^*$, and $2s^* \ge T$ because the step before it overflowed.
3. At $s^* < T$ the next $N$ steps are good, and the $N$th doubles the scale to $2s^* \ge T$. That step overflows and halves the scale back to $s^*$ with the count at 0, and the cycle starts again.
4. From $S_0 < T$ the steps are good until the scale reaches $T$; it is $S_0 2^{\lfloor t/N \rfloor}$ at step $t$, which first reaches $T$ after $\lceil \log_2(T/S_0) \rceil$ doublings, at step $N \lceil \log_2(T/S_0) \rceil$. That step is skipped and the scale falls to $s^*$.
5. $2^{16}/1000 = 65.536$, and $\lfloor \log_2 65.536 \rfloor + 1 = 7$ skips bring the scale to $2^{16}/2^7 = 512$; the cycle then has 2001 steps.

### Result and check

$$ S_t = S_0\, 2^{G_t - B_t} , \qquad \text{skipped fraction at steady state} = \frac{1}{N + 1} . $$ {#eq-dyn-scale}

The check `dynamic-loss-scale.check.py` runs the rule against 300 random thresholds, starting scales and intervals $N \in \{1, 2, 3, 7, 50, 2000\}$, and finds the start of item 2, the cycle of item 3 and the growth of item 4 step by step.
It runs the defaults of item 5, and checks item 1 at every step of a run with random overflows.
The tests of the NumPy and PyTorch implementations check that the rule agrees with `torch.amp.GradScaler` and its update kernel on CPU.
