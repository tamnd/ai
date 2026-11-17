---
id: weight-decay-shrinkage
derives: [c-decay-geometric, c-decay-time, c-decay-numbers]
kind: closed-form
assumes: ["the AdamW update with decoupled weight decay", "a zero gradient of the data loss and zero Adam moments", "0 < lr * wd < 1"]
uses: []
status: full
check: weight-decay-shrinkage.check.py
lean: null
sources:
  - {work: liu-2022-omnigrok-grokking-beyond-algorithmic, locator: "p. 3"}
  - {work: nanda-2023-progress-measures-grokking, locator: "p. 3"}
  - {work: power-2022-grokking-generalization-beyond-overfitting, locator: "p. 6"}
---

### Statement

Let AdamW update a parameter vector $\theta$ with learning rate $\eta$ and decoupled weight decay $\lambda$, $0 < \eta\lambda < 1$:

$$ \theta_{t+1} = (1 - \eta\lambda)\,\theta_t - \eta\,\frac{\hat m_{t+1}}{\sqrt{\hat v_{t+1}} + \epsilon}. $$

If the gradient of the data loss is zero at every step and the moments start at zero, then (i) $\theta_t = (1 - \eta\lambda)^t \theta_0$ and $\lVert\theta_t\rVert = (1 - \eta\lambda)^t \lVert\theta_0\rVert$;
(ii) $(1 - \eta\lambda)^t \le e^{-\eta\lambda t}$, and the number of steps to bring the norm from $w_0$ down to $w_c < w_0$ is the least integer $t$ with $t \ge \ln(w_0/w_c) / (-\ln(1 - \eta\lambda))$, which is at most $\lceil \ln(w_0/w_c)/(\eta\lambda) \rceil$ and approaches it as $\eta\lambda \to 0$, so it grows as $1/\lambda$;
(iii) with $\eta = 10^{-3}$ and $\lambda = 1$, the settings of the original grokking runs and of the mainline modular-addition model, one step multiplies the weights by $0.999$, 1,000 steps by $0.3677$, and halving the norm takes 693 steps.

### Assumptions

1. The update is the decoupled form above, with $\hat m$ and $\hat v$ the bias-corrected first and second moments of the gradient.
2. The gradient of the data loss is zero at every step considered, and the moments are zero at the start, so they stay zero.
3. $0 < \eta\lambda < 1$.

### Steps

1. With zero gradients from zero moments, $m_{t+1} = \beta_1 m_t + (1 - \beta_1)\cdot 0 = 0$ and likewise $v_{t+1} = 0$, so the Adam term is $0/(0 + \epsilon) = 0$ and $\theta_{t+1} = (1 - \eta\lambda)\theta_t$.
2. By induction, $\theta_t = (1 - \eta\lambda)^t\theta_0$; the norm is homogeneous, $\lVert c\,\theta\rVert = |c|\,\lVert\theta\rVert$, and $1 - \eta\lambda > 0$, which gives (i).
3. $\ln(1 - x) \le -x$ for $0 \le x < 1$, so $(1 - \eta\lambda)^t = e^{t\ln(1 - \eta\lambda)} \le e^{-\eta\lambda t}$.
4. $(1 - \eta\lambda)^t w_0 \le w_c$ is $t\,(-\ln(1 - \eta\lambda)) \ge \ln(w_0/w_c)$; since $-\ln(1 - \eta\lambda) \ge \eta\lambda$, the bound $\ln(w_0/w_c)/(\eta\lambda)$ is at least the exact threshold, and since $-\ln(1 - x) = x + O(x^2)$ the two agree to first order. This is (ii), the discrete form of the estimate $t \approx \ln(w_0/w_c)/\gamma$ of Liu, Michaud and Tegmark (2022) (p. 3), with their decay rate $\gamma$ equal to $\eta\lambda$ per step.
5. With $\eta\lambda = 10^{-3}$: $0.999^{1000} = e^{1000 \ln 0.999} = 0.36770$, and $\ln 2/(-\ln 0.999) = 692.8$, so 693 steps halve the norm.

### Result and check

Weight decay alone shrinks the weights geometrically, by $(1 - \eta\lambda)$ per step, and the time it needs to carry the weights from a large norm to a smaller one grows like $1/(\eta\lambda)$.
This is the clock behind the explanation of grokking by weight norm: once the training loss is near zero, the decay term is what still moves the weights (Liu, Michaud and Tegmark (2022), p. 3).
It is a bound on the decay alone; in a real run the gradient of the data loss is small but not zero, and the Adam moments carry the history of earlier gradients.
`weight-decay-shrinkage.check.py` runs the AdamW update with zero gradients from zero moments and compares it with $(1 - \eta\lambda)^t\theta_0$, checks the exponential bound and the step counts over a grid of $\eta\lambda$, ratios and step numbers, and recomputes the numbers of (iii).
