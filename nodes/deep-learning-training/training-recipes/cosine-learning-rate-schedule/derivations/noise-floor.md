---
id: noise-floor
derives: [c-noise-recursion, c-noise-constant, c-noise-decay]
kind: closed-form
assumes: ["a one-dimensional quadratic loss f(x) = h x^2 / 2 with h > 0", "stochastic gradients h x + xi with xi of mean zero and variance sigma^2, independent across steps and of x", "plain SGD without momentum"]
uses: []
status: full
check: noise-floor.check.py
lean: null
sources: []
---

### Statement

Run SGD $x_{t+1} = x_t - \eta_t (h x_t + \xi_t)$ on $f(x) = \tfrac12 h x^2$, and let $v_t = \mathbb{E}[x_t^2]$, so that the expected loss is $\tfrac12 h v_t$.

(i) $v_{t+1} = (1 - \eta_t h)^2 v_t + \eta_t^2 \sigma^2$, exactly.

(ii) With a constant rate $\eta$ and $0 < \eta h < 2$, $v_t$ tends to $v^\ast = \frac{\eta \sigma^2}{h(2 - \eta h)}$ from any start. For small $\eta h$ the expected loss at this floor is about $\eta\sigma^2/4$, in proportion to $\eta$.

(iii) In the page's own run with $h = 1$, $\sigma^2 = 1$, $v_0 = v^\ast(0.1)$ and 1,000 steps that start at $\eta = 0.1$, a constant rate stays at $v = 0.0526$, while a cosine decay to 0 ends at $0.0036$, a linear decay to 0 at $0.0044$, a step decay by 10 at steps 500 and 750 at $0.0034$, and a WSD schedule with a 20% linear cooldown at $0.0100$.

### Assumptions

1. The loss is a quadratic in one variable. In many dimensions with a diagonal Hessian the same holds for each coordinate.
2. The noise is additive, zero-mean, of fixed variance and independent of $x_t$ and of the past. Real minibatch noise depends on the weights and on the batch size.
3. This is a toy model of the page, not a result of the cited papers. It shows why a decay lowers the final loss; it does not rank the shapes of decay.

### Steps

1. $x_{t+1} = (1 - \eta_t h) x_t - \eta_t \xi_t$. Square and take the expectation. The cross term $\mathbb{E}[x_t \xi_t]$ is 0 because $\xi_t$ is independent of $x_t$ and has mean 0. This gives (i).
2. With $\eta_t = \eta$, (i) is $v_{t+1} = a v_t + b$ with $a = (1 - \eta h)^2 < 1$ when $0 < \eta h < 2$. Its fixed point is $b/(1 - a) = \eta^2\sigma^2/(\eta h(2 - \eta h))$, which simplifies to $v^\ast$, and $v_t - v^\ast = a^t (v_0 - v^\ast) \to 0$. Then $\tfrac12 h v^\ast = \eta\sigma^2/(2(2 - \eta h)) \approx \eta\sigma^2/4$. This gives (ii).
3. A smaller $\eta_t$ gives a smaller noise term $\eta_t^2\sigma^2$, but also a weaker pull $(1 - \eta_t h)^2$ toward 0. Over a decay, $v_t$ tracks the floor $v^\ast(\eta_t)$ while $\eta_t h$ is large enough for the pull to act, and it freezes once $\eta_t h$ is small compared with the steps left. The final variance is set by the rates near the end of the run.
4. Running the recursion (i) for the five schedules gives the numbers of (iii).

### Result and check

On this toy problem a constant rate leaves the iterate bouncing at a noise floor in proportion to $\eta$, and every decay tried ends at least five times lower.
The check runs the exact recursion for the five schedules, confirms the fixed point of (ii) for several $\eta$, and compares the recursion with a Monte Carlo average over 20,000 independent SGD runs to within 4%.
