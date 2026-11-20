---
id: quadratic-form
derives: [c-quadratic]
kind: identity
assumes: ["a diagonal A", "Ā_t = exp(Δ_t A) at every position t", "the initial state h_0 = 0"]
uses: [diagonal-matrix@m2, matrix-multiplication@m2]
status: full
check: quadratic-form.check.py
lean: null
sources:
  - {work: gu-2023-mamba, locator: "p. 6", note: "Algorithm 2, the time-varying parameters of shape (B, L, D, N)"}
  - {work: gu-2023-mamba, locator: "p. 7", note: "the recurrent computation in O(BLDN)"}
---

### Statement

For one channel with $h_t = \operatorname{diag}(e^{\Delta_t A}) h_{t-1} + \bar{B}_t x_t$, $h_0 = 0$ and $y_t = C_t^\top h_t$,

$$ y_t = \sum_{s=1}^{t} M_{t,s} x_s, \qquad M_{t,s} = \sum_{n=1}^{N} C_{t,n}\, e^{A_n (S_t - S_s)}\, \bar{B}_{s,n}, \qquad S_t = \sum_{r=1}^{t} \Delta_r , $$

so $y = Mx$ with $M$ lower triangular. Computing $M$ costs $O(T^2 N)$ per channel and the scan $O(TN)$. For $A_n < 0$ every entry $e^{A_n(S_t - S_s)}$ with $s \le t$ is in $(0, 1]$.

### Assumptions

1. $A$ is diagonal, so the state entries evolve separately. Used in step 1.
2. $\bar{A}_t = e^{\Delta_t A}$. Used in step 2.
3. $h_0 = 0$. Used in step 1.

### Steps

1. Unrolling the recurrence from $h_0 = 0$, entry $n$ of the state is $h_{t,n} = \sum_{s \le t} \big(\prod_{r = s+1}^{t} \bar{A}_{r,n}\big) \bar{B}_{s,n} x_s$, by induction on $t$: $h_{t,n} = \bar{A}_{t,n} h_{t-1,n} + \bar{B}_{t,n} x_t$ multiplies every earlier term by $\bar{A}_{t,n}$ and adds the term $s = t$ with the empty product 1.
2. $\prod_{r=s+1}^{t} e^{\Delta_r A_n} = e^{A_n \sum_{r=s+1}^{t} \Delta_r} = e^{A_n(S_t - S_s)}$.
3. $y_t = \sum_n C_{t,n} h_{t,n}$; exchanging the sums over $n$ and $s$ gives $M_{t,s}$.
4. $\Delta_r > 0$ makes $S$ increasing, so $S_t - S_s \geq 0$ for $s \leq t$, and with $A_n < 0$ the exponent is $\le 0$. Above the diagonal the exponent is positive, which is why an implementation must mask it before taking the exponential.
5. There are $T(T+1)/2$ entries of $M$, each a sum of $N$ products: $O(T^2N)$. The scan does $O(N)$ work per position.

### Result and check

The selective SSM is a data-dependent lower-triangular matrix applied to its input, a masked mixing of positions like causal attention, with weights that decay with the sum of the steps between two positions. The check `quadratic-form.check.py` builds $M$ for random inputs and compares $Mx$ with the scan of the NumPy reference, for both discretizations, and checks the bound of step 4.
