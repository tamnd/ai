---
id: soft-target-gradient
derives: [c-soft-gradient, c-kl-same-gradient, c-gradient-sums-to-zero, c-hard-special-case, c-full-gradient, c-we-gradient, eq-soft-gradient]
kind: gradient
assumes: ["the teacher logits v and the student logits z are finite vectors of the same length N ≥ 2", "T > 0", "the teacher distribution p = softmax(v/T) does not depend on z", "0 ≤ α ≤ 1"]
uses: [cross-entropy@m2, kullback-leibler-divergence@m2, shannon-entropy@m2]
status: full
check: soft-target-gradient.check.py
lean: null
sources:
  - {work: hinton-2015-distilling-knowledge, locator: "p. 2"}
  - {work: hinton-2015-distilling-knowledge, locator: "p. 3"}
---

### Statement

Let $v, z \in \mathbb R^N$ be the teacher's and the student's logits, $T > 0$ the temperature, $p = \operatorname{softmax}(v/T)$ and $q = \operatorname{softmax}(z/T)$, so $p_i = e^{v_i/T}/\sum_j e^{v_j/T}$ and likewise for $q$.
Write $C = -\sum_i p_i \log q_i$ for the soft cross-entropy and $D = \sum_i p_i \log(p_i/q_i)$ for the KL divergence $D_{\mathrm{KL}}(p \,\Vert\, q)$. Then

1. $\partial C/\partial z_i = (q_i - p_i)/T$;
2. $C = D + H(p)$ with $H(p) = -\sum_i p_i\log p_i$, so $\partial D/\partial z_i = \partial C/\partial z_i$;
3. $\sum_i \partial C/\partial z_i = 0$, and $C$ and $D$ do not change when the same constant is added to every $z_i$;
4. with $T = 1$ and a one-hot target $y$ in place of $p$, the gradient is $q - y$, the gradient of the ordinary cross-entropy;
5. the loss $L = \alpha T^2 D + (1 - \alpha)\big(-\log \operatorname{softmax}(z)_y\big)$ has gradient $\alpha T (q - p) + (1 - \alpha)(\operatorname{softmax}(z) - \delta_y)$.

### Assumptions

1. The logits are finite, so every $p_i$ and $q_i$ is positive and every logarithm is finite. Used in all steps.
2. $T > 0$. Steps 1 and 5 divide by it.
3. The teacher's distribution $p$ is a constant for the student: it does not depend on $z$. In code this is the teacher's output computed without gradient, or detached.
4. $0 \le \alpha \le 1$ in item 5; the derivation holds for any real $\alpha$, but only this range is a weighted average.

### Steps

1. $\log q_k = z_k/T - \log\sum_j e^{z_j/T}$, so $\partial \log q_k/\partial z_i = ([k = i] - q_i)/T$.
2. Then $\partial C/\partial z_i = -\sum_k p_k([k = i] - q_i)/T = (-p_i + q_i\sum_k p_k)/T = (q_i - p_i)/T$, since $\sum_k p_k = 1$. This is equation 2 of Hinton et al.
3. $D = \sum_i p_i\log p_i - \sum_i p_i \log q_i = -H(p) + C$. The entropy $H(p)$ depends only on $v$ and $T$, so its gradient in $z$ is 0 and $D$ has the gradient of $C$. The values differ by $H(p)$, which matters when the losses are compared or logged but not for training.
4. Both $q$ and $p$ sum to 1, so the entries of $q - p$ sum to 0. Adding $c$ to every $z_i$ multiplies every $e^{z_i/T}$ by $e^{c/T}$, which cancels in the softmax, so $q$, $C$ and $D$ are unchanged; the zero sum of the gradient is the derivative of this invariance along the direction $(1, \dots, 1)$.
5. With $T = 1$ and $p = \delta_y$, step 2 gives $q - \delta_y$, and $C = -\log q_y$ is the ordinary cross-entropy.
6. By step 2, $\partial(T^2 D)/\partial z = T^2(q - p)/T = T(q - p)$, and by step 5 the hard term contributes $(1 - \alpha)(\operatorname{softmax}(z) - \delta_y)$. Summing gives item 5.

### Result and check

$$ \frac{\partial C}{\partial z_i} = \frac{\partial D_{\mathrm{KL}}(p \,\Vert\, q)}{\partial z_i} = \frac{1}{T}\,(q_i - p_i) , \qquad \nabla_z L = \alpha T (q - p) + (1 - \alpha)(\operatorname{softmax}(z) - \delta_y) . $$

The check `soft-target-gradient.check.py` draws 2000 random pairs of logit vectors with $N$ from 2 to 40, logit scales from 0.1 to 10 and temperatures from 0.2 to 50.
It checks items 2 and 3 to $10^{-10}$ on all of them, and compares items 1 and 5 with central finite differences to $10^{-6}$ relative error on the draws with $N \le 8$.
It evaluates the worked example of the page, $v = (3, 1, -4)$, $z = (1, 0.5, -1.5)$, which at $T = 1$ gives the gradient $(-0.287889, 0.240081, 0.047808)$ and at $T = 4$ gives $(-0.037024, 0.006079, 0.030945)$, and the loss $1.038612$ with label 0 and $\alpha = 0.9$.
