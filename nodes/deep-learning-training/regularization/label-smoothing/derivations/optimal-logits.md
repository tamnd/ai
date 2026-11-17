---
id: optimal-logits
derives: [c-optimum, c-logit-gap, c-gap-imagenet, c-convention, eq-logit-gap]
kind: identity
assumes: ["the logits of one example are free parameters", "u is uniform over K ≥ 2 classes", "0 ≤ ε ≤ 1"]
uses: [cross-entropy@m2, kullback-leibler-divergence@m2, gibbs-inequality@m2, stochastic-gradient-descent@m2]
status: full
check: optimal-logits.check.py
lean: null
sources:
  - {work: szegedy-2016-rethinking-inception, locator: "p. 7"}
  - {work: muller-2019-when-does-label-smoothing-help, locator: "Section 2"}
---

### Statement

Let $q' = (1 - \varepsilon)\delta_y + \varepsilon/K$ be the uniformly smoothed target of an example with label $y$ over $K \ge 2$ classes, and let $p = \operatorname{softmax}(z)$ with the logits $z$ free. Then

1. for $\varepsilon > 0$, $H(q', p)$ is smallest exactly when $p = q'$, and its minimum is the entropy $H(q') > 0$;
2. at the minimum every incorrect logit lies below the correct one by the same gap
   $$ z_y - z_k = \log\frac{1 - \varepsilon + \varepsilon/K}{\varepsilon/K} = \log\Big(1 + \frac{K(1 - \varepsilon)}{\varepsilon}\Big), \qquad k \ne y ; $$
3. for $\varepsilon = 0$ no finite logits reach the infimum $0$ of the loss, and under gradient descent the gap $z_y - \max_{k \ne y} z_k$ grows at every step and without bound;
4. for $K = 1000$ and $\varepsilon = 0.1$ the gap is $\log 9001 \approx 9.105$ and the optimal largest probability is $0.9001$;
5. the target that puts $1 - \varepsilon$ on the label and $\varepsilon/(K - 1)$ on each other class is uniform smoothing with $\varepsilon' = \varepsilon K/(K - 1)$.

### Assumptions

1. The logits of the example are free parameters, as in a model that can fit each training example; a real network couples the examples through shared weights, so the result describes what the loss rewards, not where training ends.
2. $u$ is uniform over $K \ge 2$ classes, and $0 \le \varepsilon \le 1$. Item 1 needs $\varepsilon > 0$ so that every entry of $q'$ is positive. Item 5 needs $\varepsilon \le (K - 1)/K$ so that $\varepsilon' \le 1$.
3. In item 3 gradient descent uses a constant step $\eta > 0$ on the logits directly.

### Steps

1. Write $H(q', p) = H(q') + D_{\mathrm{KL}}(q' \,\Vert\, p)$. By Gibbs' inequality $D_{\mathrm{KL}}(q' \,\Vert\, p) \ge 0$, with equality only at $p = q'$. When every $q'_k > 0$ the softmax reaches $p = q'$ with the finite logits $z_k = \log q'_k + c$ for any constant $c$. The minimum is $H(q')$, which is positive because $q'$ is not a point mass.
2. At those logits $z_y - z_k = \log q'_y - \log q'_k$. With $q'_y = 1 - \varepsilon + \varepsilon/K$ and $q'_k = \varepsilon/K$ for $k \ne y$ the gap is the same for every incorrect class, and $q'_y/q'_k = 1 + K(1 - \varepsilon)/\varepsilon$. The gap falls from $+\infty$ at $\varepsilon \to 0$ to $0$ at $\varepsilon = 1$, where the target is uniform. This is the bounded gap of Szegedy et al. and the constant gap of Müller et al.
3. With $\varepsilon = 0$ the loss is $-\log p_y > 0$ for finite logits, and it tends to $0$ only as $z_y - z_k \to \infty$ for every $k \ne y$. By the gradient $p - \delta_y$ of the other derivation, a step adds $\eta(1 - p_y) > 0$ to $z_y$ and subtracts $\eta p_k > 0$ from each $z_k$, so every gap $z_y - z_k$ grows by $\eta(1 - p_y + p_k) > 0$. If the smallest gap $g$ stayed below some $M \ge 0$, then $1 - p_y \ge \min(1/2, e^{-M}/K)$ at every step: if $g < 0$ some class beats $y$ and $p_y < 1/2$, and if $0 \le g < M$ then $p_y \ge 1/K$ and the closest class has $p_y e^{-g} \ge e^{-M}/K$. Each increase would then be bounded from below, and the gap would pass $M$; so it grows without bound.
4. With $K = 1000$ and $\varepsilon = 0.1$, $q'_y = 0.9 + 0.0001 = 0.9001$ and $q'_k = 0.0001$, so the gap is $\log 9001 \approx 9.1051$.
5. With $\varepsilon' = \varepsilon K/(K - 1)$, uniform smoothing gives each wrong class $\varepsilon'/K = \varepsilon/(K - 1)$ and the label $1 - \varepsilon' + \varepsilon'/K = 1 - \varepsilon'(K - 1)/K = 1 - \varepsilon$. The two conventions differ only in how $\varepsilon$ is counted.

### Result and check

$$ z_y - z_k = \log\Big(1 + \frac{K(1 - \varepsilon)}{\varepsilon}\Big) \quad (k \ne y), \qquad \min_z H(q', p) = H(q') . $$

The check `optimal-logits.check.py` runs gradient descent on the logits of one example for several $K$ and $\varepsilon$ and finds $p = q'$ to $10^{-9}$, the gap of item 2 for every incorrect class, and the loss $H(q')$. It checks with random logits that no point has a lower loss.
With $\varepsilon = 0$ it checks that the gap grows at every one of 20000 steps and passes 9.
It evaluates item 4, and checks item 5 for $K$ from 2 to 20.
