---
id: loss-barrier
derives: [c-barrier-convex, c-barrier-relation]
kind: closed-form
assumes: ["L is a real function on the weights", "the barrier is taken over the segment from w1 to w2, t in [0, 1]"]
uses: []
status: full
check: loss-barrier.check.py
lean: null
sources:
  - {work: frankle-2019-linear-mode-connectivity-lottery, locator: "p. 3"}
  - {work: entezari-2022-permutation-invariance-mode-connectivity, locator: "p. 3"}
  - {work: ainsworth-2023-git-re-basin, locator: "p. 3"}
---

### Statement

Write $L(t) = L((1-t)w_1 + t w_2)$ and $L_0 = L(0)$, $L_1 = L(1)$. The two barriers in use are
$$ B_{\text{mean}} = \max_{t} L(t) - \tfrac12(L_0 + L_1), \qquad B_{\text{lin}} = \max_{t}\,[\,L(t) - (1-t)L_0 - tL_1\,]. $$
1. Both are non-negative: $B_{\text{lin}} \ge 0$ and $B_{\text{mean}} \ge \tfrac12\lvert L_0 - L_1\rvert$.
2. If $L$ is convex, $B_{\text{lin}} = 0$ and $B_{\text{mean}} = \tfrac12\lvert L_0 - L_1\rvert$. Both are zero when, in addition, $L_0 = L_1$.
3. $\lvert B_{\text{mean}} - B_{\text{lin}}\rvert \le \tfrac12\lvert L_0 - L_1\rvert$. The two gaps are equal at $t = \tfrac12$, and the barriers are equal when $L_0 = L_1$.

### Assumptions

1. The maximum is over $t \in [0, 1]$, endpoints included. A grid that contains 0 and 1 gives the same statements with max over the grid.
2. Convexity, where used, is convexity of $L$ along the segment.

### Steps

1. At $t = 0$ the gap of $B_{\text{lin}}$ is $L_0 - L_0 = 0$, so the maximum is at least 0. At the end with the larger loss, the gap of $B_{\text{mean}}$ is $\max(L_0, L_1) - \tfrac12(L_0 + L_1) = \tfrac12\lvert L_0 - L_1\rvert$.
2. Convexity gives $L(t) \le (1-t)L_0 + tL_1$ (Jensen's inequality), so every gap of $B_{\text{lin}}$ is at most 0, and with step 1, $B_{\text{lin}} = 0$. The same bound gives $L(t) \le \max(L_0, L_1)$, so $B_{\text{mean}}$ is at most $\tfrac12\lvert L_0 - L_1\rvert$, and step 1 gives equality.
3. The gaps differ by $\tfrac12(L_0 + L_1) - (1-t)L_0 - tL_1 = (t - \tfrac12)(L_0 - L_1)$, whose size is at most $\tfrac12\lvert L_0 - L_1\rvert$ on $[0, 1]$ and is 0 at $t = \tfrac12$. Two functions that differ everywhere by at most $\delta$ have maxima that differ by at most $\delta$.

### Result and check

$B_{\text{mean}}$ is the error barrier height of Frankle et al. (2020) and Definition 2.2 of Ainsworth et al. (2023). $B_{\text{lin}}$ is Eq. (1) of Entezari et al. (2022), who note that it gives no barrier to a loss that changes linearly along the path. Part 1 is the non-negativity that Ainsworth et al. point out as a shortcoming of the metric. Part 3 shows why the choice matters little when the two networks reach similar losses, as trained networks usually do.
`loss-barrier.check.py` evaluates both barriers for random convex quadratics, for a non-convex loss with a bump, and for random endpoint losses, and checks the three parts.
