---
id: mean-rate
derives: [c-mean-rate, c-mean-discrete]
kind: closed-form
assumes: ["one cycle of eq-cosine with fixed eta_max, eta_min and length T", "no warmup inside the cycle", "for the discrete form, T is a positive integer and the rate is read at integer steps"]
uses: []
status: full
check: mean-rate.check.py
lean: null
sources:
  - {work: loshchilov-2017-sgdr, locator: "p. 4"}
---

### Statement

Let $\eta(t) = \eta_{\min} + \tfrac12(\eta_{\max} - \eta_{\min})(1 + \cos(\pi t/T))$ on $0 \le t \le T$.

(i) The mean of $\eta$ over the cycle is $\frac1T \int_0^T \eta(t)\,dt = \frac{\eta_{\max} + \eta_{\min}}{2}$.

(ii) Read at the $T + 1$ points $t = 0, 1, \dots, T$, the mean is exactly $\frac{\eta_{\max} + \eta_{\min}}{2}$.
Read at the $T$ points $t = 0, \dots, T - 1$, as a loop over $T$ training steps does, the mean is $\frac{\eta_{\max} + \eta_{\min}}{2} + \frac{\eta_{\max} - \eta_{\min}}{2T}$.

### Assumptions

1. The rate follows eq-cosine for one whole cycle, with no restart inside it.
2. For (ii), the rate is evaluated at integer steps.

### Steps

1. The constant part of $\eta$ is $\eta_{\min} + \tfrac12(\eta_{\max} - \eta_{\min}) = \tfrac12(\eta_{\max} + \eta_{\min})$. The rest is $\tfrac12(\eta_{\max} - \eta_{\min})\cos(\pi t/T)$.
2. $\int_0^T \cos(\pi t/T)\,dt = \frac{T}{\pi}[\sin(\pi t/T)]_0^T = 0$. So the mean is the constant part, which gives (i).
3. The cosine is odd about the middle of the cycle: $\cos(\pi(T - t)/T) = -\cos(\pi t/T)$. So in $\sum_{t=0}^{T} \cos(\pi t/T)$ the terms for $t$ and $T - t$ cancel, and the middle term, if there is one, is $\cos(\pi/2) = 0$. The sum is 0, which gives the first part of (ii).
4. Dropping the last point removes $\cos(\pi) = -1$, so $\sum_{t=0}^{T-1} \cos(\pi t/T) = 1$. The mean of the $T$ rates is the constant part plus $\tfrac12(\eta_{\max} - \eta_{\min}) \cdot \frac1T$, which gives the second part of (ii).

### Result and check

The total learning rate of one cosine cycle, $\sum_t \eta_t$, is the same as that of a constant rate at the midpoint $(\eta_{\max} + \eta_{\min})/2$, and the same as that of a linear decay between the same ends.
The check integrates eq-cosine with the trapezoidal rule and with Simpson's rule for several $(\eta_{\max}, \eta_{\min}, T)$, compares the discrete sums at $T + 1$ and $T$ points with the formulas to $10^{-12}$, and confirms the extra term $(\eta_{\max} - \eta_{\min})/(2T)$ for $T = 1$ to $1000$.
