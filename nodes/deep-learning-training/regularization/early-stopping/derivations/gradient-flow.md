---
id: gradient-flow
derives: [c-flow-filter, c-flow-limit, c-flow-ridge-ratio, eq-flow-filter, eq-flow-gap]
kind: limit
assumes: ["least squares loss (1/2n) |y - X w|^2", "gradient descent and gradient flow both start at w = 0", "eta s_max <= 1"]
uses: [ridge-regression@al]
status: full
check: gradient-flow.check.py
lean: null
sources:
  - {work: ali-2019-continuous-time-early-stopping, locator: "Lemma 1"}
  - {work: ali-2019-continuous-time-early-stopping, locator: "Lemma 2"}
  - {work: ali-2019-continuous-time-early-stopping, locator: "Lemma 7"}
---

### Statement

With the notation of [the spectral filters derivation](spectral-filters.md):

1. gradient flow $\dot w = b - \hat\Sigma w$ from $w(0) = 0$ is $w(\tau) = \sum_{s_i > 0} (1 - e^{-s_i \tau})\,(c_i/s_i)\, v_i$;
2. if $\eta s_{\max} \le 1$, then for every $k \ge 1$, $\lVert w_k - w(k\eta)\rVert \le \frac{\eta}{2}\lVert b\rVert$, so gradient descent converges to the flow at rate $O(\eta)$ uniformly in time;
3. for every $s \ge 0$ and $\tau > 0$, $r_{1/\tau}(s) \le 1 - e^{-s\tau} \le 1.2985\, r_{1/\tau}(s)$.

### Assumptions

1. The loss and its gradient are as in the spectral filters derivation. Used in steps 1 and 2.
2. Both paths start at zero. Used in steps 1 and 2.
3. $0 \le \eta s_i \le 1$ for all $i$. Used in step 2.

### Steps

1. In the eigenbasis the flow is $\dot a_i = c_i - s_i a_i$ with $a_i(0) = 0$, solved by $a_i(\tau) = (1 - e^{-s_i\tau})\, c_i/s_i$ for $s_i > 0$ and $a_i = 0$ when $s_i = 0$, since then $c_i = 0$. This is Lemma 1 of Ali et al. written per direction.
2. Fix a direction and put $x = \eta s \in [0, 1]$. The two filters differ by $e^{-kx} - (1 - x)^k$. With $\alpha = 1 - x$ and $\beta = e^{-x}$ we have $0 \le \alpha \le \beta \le 1$ and $\beta - \alpha \le x^2/2$, from $e^{-x} \le 1 - x + x^2/2$ for $x \ge 0$. Then $\beta^k - \alpha^k = (\beta - \alpha)\sum_{j<k} \beta^{k-1-j}\alpha^j \le k\,\frac{x^2}{2}\, e^{-(k-1)x}$. Writing $u = kx$, this is $\frac{x}{2} e^{x}\, u e^{-u} \le \frac{x}{2} e^{x-1} \le \frac{x}{2}$, using $u e^{-u} \le 1/e$ and $x \le 1$. The coefficient gap is the filter gap times $|c_i|/s_i$, at most $\frac{\eta}{2}|c_i|$, and summing squares over $i$ gives $\frac{\eta}{2}\lVert c\rVert = \frac{\eta}{2}\lVert b\rVert$. Lemma 2 of Ali et al. gives a bound for step sizes below $1/s_{\max}$ that grows like $e^{K\eta s_{\max}}$ with the horizon; this one does not grow.
3. Put $u = s\tau$. The lower bound is $e^{-u} \le 1/(1 + u)$, and the upper bound is $1 - e^{-u} \le 1.2985\, u/(1 + u)$; these are parts (a) and (b) of Lemma 7 of Ali et al.

### Result and check

$$ \bigl|(1 - \eta s)^k - e^{-k\eta s}\bigr| \le \frac{\eta s}{2}\, e^{\eta s - 1}, \qquad \lVert w_k - w(k\eta)\rVert \le \frac{\eta}{2}\,\lVert X^\top y/n\rVert . $$

The check `gradient-flow.check.py` builds a random design with $n = 50, p = 6$.
It confirms by finite differences that the closed form solves the flow.
It runs gradient descent to time 40 with $\eta s_{\max} = 1, 1/2, 1/4, 1/8, 1/16$; the largest gap over the run stays under the bound, and halving $\eta$ divides it by 2.26, 2.10 and 2.05 over the last three halvings.
It checks the scalar bound for $k$ up to 1000.
It finds the largest ratio of the flow filter to the ridge filter on a grid to be 1.29843, at $s\tau \approx 1.79$.
