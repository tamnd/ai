---
id: mixed-cross-entropy
derives: [c-mixed-loss, c-mixed-batch, eq-mix, eq-mixed-loss]
kind: identity
assumes: ["p is the softmax of finite logits, so p_k > 0 for every k", "the target is λ e_{y_i} + (1 − λ) e_{y_j} with 0 ≤ λ ≤ 1", "λ and the pairing do not depend on the logits"]
uses: [cross-entropy@m2, linearity-of-expectation@m2]
status: full
check: mixed-cross-entropy.check.py
lean: null
sources:
  - {work: zhang-2018-mixup, locator: "Section 1"}
  - {work: zhang-2018-mixup, locator: "Section 2"}
  - {work: zhang-2021-how-does-mixup-help, locator: "p. 3"}
---

### Statement

Let $p = \operatorname{softmax}(z)$ over $K$ classes, let $e_k$ be the one-hot vector of class $k$, and let the mixup target of a pair with labels $y_i, y_j$ be $\tilde y = \lambda e_{y_i} + (1 - \lambda) e_{y_j}$ with $0 \le \lambda \le 1$. Write $H(a, p) = -\sum_k a_k \log p_k$. Then

1. $H(\tilde y, p) = \lambda\,(-\log p_{y_i}) + (1 - \lambda)\,(-\log p_{y_j})$;
2. $\partial H(\tilde y, p)/\partial z = p - \tilde y$, whose entries lie in $[-1, 1]$ and sum to 0;
3. for a minibatch of $B$ examples and a permutation $\pi$ of $\{1, \dots, B\}$, with $\tilde x_b = \lambda x_b + (1 - \lambda) x_{\pi(b)}$ and logits $z_b = f(\tilde x_b)$, the mean of $H(\lambda e_{y_b} + (1 - \lambda) e_{y_{\pi(b)}}, p_b)$ over $b$ equals $\lambda\,\overline{\mathrm{CE}}(z, y) + (1 - \lambda)\,\overline{\mathrm{CE}}(z, y_\pi)$, where $\overline{\mathrm{CE}}$ is the mean cross-entropy with integer labels, and so do its gradients in the parameters of $f$;
4. if $y_i = y_j$ the target is $e_{y_i}$ and the loss is the ordinary cross-entropy.

### Assumptions

1. The logits are finite, so every $p_k > 0$ and every logarithm is finite.
2. $\lambda \in [0, 1]$ and the pairing $\pi$ are drawn before the forward pass and do not depend on the logits; the gradient treats them as constants.
3. The loss is the cross-entropy, or any loss of the form $h(f) - y^\top f$ that is affine in the target, which is the class of Zhang et al. (2021).

### Steps

1. $H(a, p)$ is linear in its first argument: $H(\lambda a + (1 - \lambda) b, p) = \lambda H(a, p) + (1 - \lambda) H(b, p)$, since it is a sum of the entries of $a$ times fixed numbers $-\log p_k$. With $a = e_{y_i}$ and $b = e_{y_j}$, $H(e_y, p) = -\log p_y$, which gives item 1.
2. From $\log p_k = z_k - \log\sum_m e^{z_m}$, $\partial \log p_k / \partial z_m = [k = m] - p_m$. Then $\partial H(\tilde y, p)/\partial z_m = -\sum_k \tilde y_k([k = m] - p_m) = p_m \sum_k \tilde y_k - \tilde y_m = p_m - \tilde y_m$, because $\sum_k \tilde y_k = \lambda + (1 - \lambda) = 1$. Both $p_m$ and $\tilde y_m$ are in $[0, 1]$ and both sum to 1, so the entries are in $[-1, 1]$ and sum to 0.
3. Apply item 1 to each $b$ and average over the minibatch; the average of a sum is the sum of the averages. The two sides are the same function of the parameters, because $\tilde x$, $\lambda$ and $\pi$ are the same on both sides, so their gradients agree. The one-hot targets $\tilde y$ never need to be built.
4. With $y_i = y_j$, $\tilde y = \lambda e_{y_i} + (1 - \lambda) e_{y_i} = e_{y_i}$.

### Result and check

$$ H(\lambda e_{y_i} + (1 - \lambda) e_{y_j}, p) = \lambda(-\log p_{y_i}) + (1 - \lambda)(-\log p_{y_j}) , \qquad \nabla_z H = p - \tilde y . $$

The check `mixed-cross-entropy.check.py` draws 2000 random logit vectors with $K$ from 2 to 30, label pairs and $\lambda \in [0, 1]$, and checks item 1 to $10^{-12}$ and item 2 against central finite differences to $10^{-7}$.
For item 3 it mixes random minibatches of a linear model through a random permutation and compares the soft-target loss and its gradient in the weights with the two-label form.
It evaluates the worked example of the page: logits $(1, 0.5, -0.5)$, labels 0 and 2, $\lambda = 0.7$, which give the loss $1.054131$ and the gradient $(-0.153451, 0.331499, -0.178048)$.
