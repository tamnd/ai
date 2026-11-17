---
id: confidence-loss
derives: [c-conf-mixture, c-conf-alpha-zero, c-conf-alpha-one, c-conf-minimizer, c-conf-gradient, c-conf-side, c-conf-half]
kind: closed-form
assumes: ["binary classification with predictive probabilities in (0, 1)", "the hardened predictions are treated as constants, with no gradient through the threshold", "0 <= alpha <= 1"]
uses: []
status: full
check: confidence-loss.check.py
lean: null
sources:
  - {work: burns-2023-weak-to-strong-generalization, locator: "p. 33"}
---

### Statement

For a binary task write $p = f(x) \in (0, 1)$ for the strong student's probability of the positive class, $q_w = f_w(x) \in [0, 1]$ for the weak label, $h = \hat f_t(x) = \mathbb{1}[p > t] \in \{0, 1\}$ for the hardened prediction, and $\mathrm{CE}(p, q) = -q \log p - (1 - q) \log (1 - p)$.
For $\alpha \in [0, 1]$,

$$ L_{\text{conf}} = (1 - \alpha)\,\mathrm{CE}(p, q_w) + \alpha\,\mathrm{CE}(p, h) = \mathrm{CE}\big(p, (1 - \alpha) q_w + \alpha h\big). $$

At $\alpha = 0$ it is the cross-entropy to the weak labels, naive finetuning; at $\alpha = 1$ it is the cross-entropy to the student's own hardened predictions, self-training with pseudo-labels.
With the target held fixed, the loss of one example is minimized at $p = (1 - \alpha) q_w + \alpha h$, and its gradient with respect to the logit $z$, $p = \sigma(z)$, is $p - \big((1 - \alpha) q_w + \alpha h\big)$.
For $\alpha > 1/2$ the target lies on the side of $1/2$ of the hardened label whatever the weak label; for $\alpha \le 1/2$ the weak label wins when it is confident enough, and a weak label $q_w < 1/2$ is overridden by $h = 1$ exactly when $\alpha > (1/2 - q_w)/(1 - q_w)$.
If $t$ is the largest value of the lower half of a batch of $B$ distinct probabilities, $p > t$ holds for exactly $\lfloor B/2 \rfloor$ of them.

### Assumptions

1. Binary classification with $p \in (0, 1)$, so both logarithms are finite.
2. $h$ is computed from the current predictions but treated as a constant when differentiating, as an indicator has zero derivative almost everywhere and is not differentiable at the threshold.
3. $0 \le \alpha \le 1$, so the mixed target is in $[0, 1]$.
4. For the last statement, the $B$ probabilities in the batch are distinct.

### Steps

1. $\mathrm{CE}(p, q) = -\log(1 - p) - q\,\big(\log p - \log(1 - p)\big)$ is affine in $q$. An affine function commutes with convex combinations, so $(1 - \alpha)\mathrm{CE}(p, q_w) + \alpha\mathrm{CE}(p, h) = \mathrm{CE}(p, (1 - \alpha) q_w + \alpha h)$, which is Eq. 2 of Burns et al.; the target is in $[0, 1]$ because it is a convex combination of two numbers there.
2. At $\alpha = 0$ the target is $q_w$; at $\alpha = 1$ it is $h$, a label the student assigns to itself.
3. For fixed $q$, $\partial \mathrm{CE}/\partial p = -q/p + (1 - q)/(1 - p) = (p - q)/\big(p(1 - p)\big)$, negative for $p < q$ and positive for $p > q$, so the unique minimizer on $(0, 1)$ is $p = q$ when $0 < q < 1$; when $q \in \{0, 1\}$ the infimum is approached as $p \to q$. With $q = (1 - \alpha) q_w + \alpha h$ this is the statement.
4. With $p = \sigma(z)$, $dp/dz = p(1 - p)$, so by the chain rule $\partial \mathrm{CE}/\partial z = p - q$.
5. If $h = 1$, the target is $(1 - \alpha) q_w + \alpha \ge \alpha$; if $h = 0$ it is $(1 - \alpha) q_w \le 1 - \alpha$. For $\alpha > 1/2$ these are $> 1/2$ and $< 1/2$, so the target sides with $h$. For $h = 1$ and $q_w < 1/2$, $(1 - \alpha) q_w + \alpha > 1/2$ rearranges to $\alpha (1 - q_w) > 1/2 - q_w$.
6. Sort the batch, $s_1 < \dots < s_B$, and set $k = \lfloor B/2 \rfloor$ and $t = s_{B - k}$. Then $s_j > t$ exactly for $j = B - k + 1, \dots, B$, which is $k$ values.
7. Example: $q_w = 0.3$ and $h = 1$ need $\alpha > 0.2/0.7 \approx 0.286$; at $\alpha = 0.5$ the target is $0.65$.

### Result and check

The check confirms the two forms agree on random batches and the values at $\alpha = 0$ and $\alpha = 1$, finds the minimizer of each fixed-target loss on a grid, compares the logit gradient with central differences, scans $\alpha$ and $q_w$ for the side of the target, confirms the threshold hardens exactly half of random batches, and reproduces the example.
