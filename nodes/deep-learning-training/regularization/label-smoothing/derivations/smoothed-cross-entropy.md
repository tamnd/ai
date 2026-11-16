---
id: smoothed-cross-entropy
derives: [c-decomposition, c-kl-form, c-confidence-relation, c-marginalized, c-gradient, eq-smoothed-target, eq-decomposition]
kind: identity
assumes: ["p is the softmax of finite logits, so p_k > 0 for every k", "u is a distribution over the K classes, uniform where stated", "0 ≤ ε ≤ 1", "the target does not depend on the logits"]
uses: [cross-entropy@m2, kullback-leibler-divergence@m2, shannon-entropy@m2]
status: full
check: smoothed-cross-entropy.check.py
lean: null
sources:
  - {work: szegedy-2016-rethinking-inception, locator: "p. 6"}
  - {work: szegedy-2016-rethinking-inception, locator: "p. 7"}
  - {work: pereyra-2017-penalizing-confident-outputs, locator: "Section 3.2"}
---

### Statement

Let $p = \operatorname{softmax}(z)$ over $K$ classes, let $y$ be the label, let $u$ be a distribution over the classes, and let the smoothed target be $q' = (1 - \varepsilon)\,\delta_y + \varepsilon u$ with $0 \le \varepsilon \le 1$. Write $H(a, p) = -\sum_k a_k \log p_k$ for the cross-entropy. Then

1. $H(q', p) = (1 - \varepsilon)\,(-\log p_y) + \varepsilon\, H(u, p)$, for every $u$;
2. for uniform $u_k = 1/K$, $H(q', p) = (1 - \varepsilon)(-\log p_y) + \varepsilon\, D_{\mathrm{KL}}(u \,\Vert\, p) + \varepsilon \log K$;
3. for uniform $u$, $-H(p) = D_{\mathrm{KL}}(p \,\Vert\, u) - \log K$, so the confidence penalty $-\beta H(p)$ is $\beta D_{\mathrm{KL}}(p \,\Vert\, u)$ up to a constant;
4. if the label is kept with probability $1 - \varepsilon$ and otherwise replaced by a draw $\tilde y \sim u$, then $\mathbb E[-\log p_{\tilde y}] = H(q', p)$;
5. $\partial H(q', p)/\partial z = p - q'$, whose entries lie in $[-1, 1]$ and sum to $0$.

### Assumptions

1. The logits are finite, so every $p_k > 0$ and every logarithm is finite. Used in all steps.
2. $u$ is a distribution: $u_k \ge 0$ and $\sum_k u_k = 1$. Items 2 and 3 take $u_k = 1/K$.
3. In item 4 the replacement label is drawn independently of the input given the label, and the prediction $p$ does not depend on it.
4. In item 5 the target $q'$ is a constant: it does not depend on $z$.

### Steps

1. The cross-entropy is linear in its first argument: $H(a + b, p) = H(a, p) + H(b, p)$ and $H(ca, p) = cH(a, p)$. So $H(q', p) = (1 - \varepsilon)H(\delta_y, p) + \varepsilon H(u, p)$, and $H(\delta_y, p) = -\log p_y$. This is the pair of losses of Szegedy et al.
2. For any distributions, $D_{\mathrm{KL}}(u \,\Vert\, p) = \sum_k u_k \log(u_k/p_k) = H(u, p) - H(u)$. For uniform $u$, $H(u) = \log K$, so $H(u, p) = D_{\mathrm{KL}}(u \,\Vert\, p) + \log K$. Substituting into step 1 gives item 2. The constant $\varepsilon\log K$ has no gradient, so up to it label smoothing adds $\varepsilon D_{\mathrm{KL}}(u \,\Vert\, p)$ to $(1 - \varepsilon)$ times the negative log-likelihood.
3. $D_{\mathrm{KL}}(p \,\Vert\, u) = \sum_k p_k \log(p_k K) = -H(p) + \log K$, which rearranges to item 3. So label smoothing penalizes $D_{\mathrm{KL}}(u \,\Vert\, p)$ and the confidence penalty $D_{\mathrm{KL}}(p \,\Vert\, u)$, the two directions of the same divergence, as Pereyra et al. state. The two directions differ: $D_{\mathrm{KL}}(u \,\Vert\, p)$ grows without bound as any $p_k \to 0$, while $D_{\mathrm{KL}}(p \,\Vert\, u)$ is at most $\log K$.
4. The noisy label has distribution $P(\tilde y = k) = (1 - \varepsilon)[k = y] + \varepsilon u_k = q'_k$. Then $\mathbb E[-\log p_{\tilde y}] = \sum_k q'_k (-\log p_k) = H(q', p)$. Training on the smoothed target is training on label dropout with the noise averaged out, the marginalized label-dropout of Szegedy et al.
5. $\log p_k = z_k - \log\sum_j e^{z_j}$, so $\partial \log p_k/\partial z_i = [k = i] - p_i$. Then $\partial H(q', p)/\partial z_i = -\sum_k q'_k([k = i] - p_i) = -q'_i + p_i\sum_k q'_k = p_i - q'_i$. Both $p_i$ and $q'_i$ lie in $[0, 1]$, so the difference lies in $[-1, 1]$, and both sum to 1, so the entries sum to 0. With $\varepsilon = 0$ this is the gradient $p(k) - q(k)$ of Szegedy et al.

### Result and check

$$ H(q', p) = (1 - \varepsilon)(-\log p_y) + \varepsilon H(u, p) = (1 - \varepsilon)(-\log p_y) + \varepsilon D_{\mathrm{KL}}(u \,\Vert\, p) + \varepsilon\log K , \qquad \nabla_z H(q', p) = p - q' . $$

The check `smoothed-cross-entropy.check.py` draws 2000 random logit vectors with $K$ from 2 to 50, labels and values of $\varepsilon$ in $[0, 1]$, and checks items 1 and 2 to $10^{-12}$, item 1 also for random non-uniform $u$. It checks item 3 on the same draws, and item 4 exactly by summing over the $K$ outcomes and by a Monte Carlo run of 200000 draws to $10^{-2}$.
It compares the gradient of item 5 with central finite differences to $10^{-7}$, and checks its bounds and its sum.
It also evaluates the worked example of the page: logits $(2, 0, 0)$, label 0, $\varepsilon = 0.3$, which gives a loss of $0.639545$ and a gradient of $(-0.013014, 0.006507, 0.006507)$.
