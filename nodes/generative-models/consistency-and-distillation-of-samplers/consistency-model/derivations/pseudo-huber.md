---
id: pseudo-huber
derives: [c-huber-limits]
kind: inequality
assumes: ["c > 0", "r = ||x - y||_2 is the Euclidean distance between the two outputs"]
uses: [huber-loss@m2, robust-statistics@m2, loss-function@m2]
status: full
check: pseudo-huber.check.py
lean: null
sources:
  - {work: song-2023-improved-techniques-training-consistency, locator: "pp. 6 to 7, Eq. 8 and Section 3.3", note: "the Pseudo-Huber metric, its bridge between l1 and squared l2, its two continuous derivatives, c = 0.03 on CIFAR-10 and the heuristic c = 0.00054 sqrt(d)"}
---

### Statement

For $c > 0$ and $r = \lVert x - y \rVert_2$, the Pseudo-Huber metric

$$ d(x, y) = \sqrt{r^2 + c^2} - c $$ {#eq-huber}

satisfies

$$ \frac{r^2}{2\sqrt{r^2 + c^2}} \le \frac{r^2}{\sqrt{r^2 + c^2} + c} = d(x, y) \le \min\Bigl(\frac{r^2}{2c},\; r\Bigr) , $$

so $d \approx r^2/(2c)$ when $r \ll c$, a squared $\ell_2$ distance scaled by $1/(2c)$, and $d \approx r - c$ when $r \gg c$, an $\ell_2$ distance; its gradient in $x$ is $(x - y)/\sqrt{r^2 + c^2}$, of norm below 1, against $2(x - y)$ for the squared distance; and it has continuous derivatives of every order in $x$, including at $x = y$, where the plain distance $r$ has none. With Song and Dhariwal's heuristic $c = 0.00054\sqrt d$ a CIFAR-10 image, $d = 3 \times 32 \times 32 = 3072$, gets $c \approx 0.0299$, their best value 0.03.

### Assumptions

1. $c > 0$. Used in steps 1 to 4.

### Steps

1. Multiplying and dividing by the conjugate, $\sqrt{r^2 + c^2} - c = r^2/(\sqrt{r^2 + c^2} + c)$.
2. Since $c \le \sqrt{r^2 + c^2}$, the denominator lies between $2c$ and $2\sqrt{r^2 + c^2}$, which gives the lower bound and $d \le r^2/(2c)$; and $\sqrt{r^2 + c^2} \le r + c$ gives $d \le r$.
3. The gradient of $\sqrt{\lVert x - y \rVert^2 + c^2}$ in $x$ is $(x - y)/\sqrt{r^2 + c^2}$, whose norm $r/\sqrt{r^2 + c^2}$ is below 1.
4. $\sqrt{u + c^2}$ is smooth for $u \ge 0$ when $c > 0$, and $u = \lVert x - y \rVert^2$ is a polynomial in $x$, so $d$ is smooth everywhere.
5. $0.00054 \times \sqrt{3072} = 0.00054 \times 55.43 = 0.0299$.

### Result and check

Near the target the metric behaves like the squared error, so small errors are fitted as in the squared $\ell_2$ loss; far from it each sample's gradient has norm at most 1, so a few outliers among the training pairs, which are common when the levels are far apart, cannot dominate a step, which is the variance reduction Song and Dhariwal measure on the Adam updates.
Theorem 2 of Song and colleagues needs a metric with bounded second derivatives, which the $\ell_1$ and $\ell_2$ distances lack at zero and the Pseudo-Huber metric has.
The scale $c$ should grow like the typical distance, which for errors spread over $d$ coordinates grows like $\sqrt d$, hence the heuristic.
The check `pseudo-huber.check.py` compares the metric with the bounds over distances from $10^{-6}$ to $10^3$ and several $c$, the gradient of `metric` with `metric_grad` and with finite differences, the second derivative at $x = y$ with $1/c$ times the identity, and the value of the heuristic for CIFAR-10 and for ImageNet $64 \times 64$.
