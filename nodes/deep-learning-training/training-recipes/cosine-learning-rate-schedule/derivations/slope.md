---
id: slope
derives: [c-slope-ends, c-slope-max]
kind: closed-form
assumes: ["one cycle of eq-cosine with eta_max > eta_min and length T", "the schedule is read as a function of continuous time t in [0, T]"]
uses: []
status: full
check: slope.check.py
lean: null
sources:
  - {work: loshchilov-2017-sgdr, locator: "p. 4"}
---

### Statement

For $\eta(t) = \eta_{\min} + \tfrac12(\eta_{\max} - \eta_{\min})(1 + \cos(\pi t/T))$ on $0 \le t \le T$:

(i) $\eta'(t) = -\frac{\pi}{2T}(\eta_{\max} - \eta_{\min}) \sin(\pi t/T)$, which is 0 at $t = 0$ and at $t = T$;

(ii) $\lvert\eta'(t)\rvert$ is largest at $t = T/2$, where it is $\frac{\pi}{2} \cdot \frac{\eta_{\max} - \eta_{\min}}{T}$, which is $\pi/2 \approx 1.571$ times the constant slope of a linear decay between the same ends;

(iii) the rate stays above $\eta_{\max} - \frac{\pi^2}{4}(\eta_{\max} - \eta_{\min})(t/T)^2$ near the start, so it leaves $\eta_{\max}$ only to second order in $t/T$, and by symmetry it reaches $\eta_{\min}$ to second order in $(T - t)/T$.

### Assumptions

1. $\eta_{\max} > \eta_{\min}$, so the cycle decreases.
2. The derivative is taken in continuous time; at integer steps the same shape holds for the differences $\eta_{t+1} - \eta_t$ up to $O(1/T^2)$.

### Steps

1. Differentiate: $\frac{d}{dt}\cos(\pi t/T) = -\frac{\pi}{T}\sin(\pi t/T)$, which gives the formula in (i). Since $\sin 0 = \sin\pi = 0$, the slope is 0 at both ends.
2. $\sin(\pi t/T)$ is non-negative on $[0, T]$ and takes its largest value 1 only at $t = T/2$. So $\lvert\eta'\rvert$ is largest there and equals $\frac{\pi}{2T}(\eta_{\max} - \eta_{\min})$, which gives (ii). A linear decay over the same $T$ has slope $(\eta_{\max} - \eta_{\min})/T$ at every $t$.
3. With $1 - \cos x \le x^2/2$, $\eta_{\max} - \eta(t) = \tfrac12(\eta_{\max} - \eta_{\min})(1 - \cos(\pi t/T)) \le \frac{\pi^2}{4}(\eta_{\max} - \eta_{\min})(t/T)^2$, which gives (iii) at the start. The substitution $t \to T - t$ maps the cycle onto itself, upside down, which gives the end.

### Result and check

The cosine keeps the rate close to $\eta_{\max}$ for longer than a linear decay, drops it fastest in the middle, and lands gently on $\eta_{\min}$.
The check compares centred finite differences of eq-cosine with the formula of (i) on a grid of $10^4$ points, finds the largest $\lvert\eta'\rvert$ at $T/2$ with the value of (ii), confirms that the slopes at the two ends are below $10^{-6}$ of that maximum, and checks the bound of (iii).
