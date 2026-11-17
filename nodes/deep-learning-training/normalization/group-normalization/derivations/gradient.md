---
id: gradient
derives: [c-grad, c-grad-mean-zero, c-grad-eps-leak, c-affine-grad, eq-gn-grad]
kind: gradient
assumes: ["the mean and the variance of a group depend on the input, and the backward pass differentiates through them", "epsilon is 0 for the exact orthogonality of the gradient to the normalized values"]
uses: [backpropagation@al, sample-mean@m2, variance@m2, orthogonal-projection@m2]
status: full
check: gradient.check.py
lean: null
sources:
  - {work: wu-2018-group-normalization, locator: "p. 3"}
  - {work: ioffe-2015-batch-normalization, locator: "p. 4"}
---

### Statement

Let one group hold $m$ entries $x_1, \dots, x_m$, with mean $\mu$, biased variance $\sigma^2$, $s = \sqrt{\sigma^2 + \epsilon} > 0$ and $\hat{x}_k = (x_k - \mu)/s$. Let $y_k = \gamma_{c(k)}\hat{x}_k + \beta_{c(k)}$, where $c(k)$ is the channel of entry $k$, let $\delta_k = \partial\ell/\partial y_k$, and let $u_k = \gamma_{c(k)}\delta_k$. Write $\bar{u} = \frac{1}{m}\sum_k u_k$ and $\overline{u\hat{x}} = \frac{1}{m}\sum_k u_k\hat{x}_k$. Then

1. $\dfrac{\partial\ell}{\partial x_k} = \dfrac{1}{s}\big(u_k - \bar{u} - \hat{x}_k\,\overline{u\hat{x}}\big)$;
2. $\sum_k \partial\ell/\partial x_k = 0$;
3. $\frac{1}{m}\sum_k \hat{x}_k\,\partial\ell/\partial x_k = \dfrac{\overline{u\hat{x}}}{s}\cdot\dfrac{\epsilon}{\sigma^2 + \epsilon}$, which is $0$ when $\epsilon = 0$;
4. $\partial\ell/\partial\gamma_c = \sum \delta_k\hat{x}_k$ and $\partial\ell/\partial\beta_c = \sum \delta_k$, both summed over the examples and the positions of channel $c$.

### Assumptions

1. $\mu$ and $\sigma^2$ depend on the $x_k$ of the group, and the backward pass differentiates through them. A backward pass that holds them fixed gives $u_k/s$ instead. Used in step 1.
2. $s > 0$. Used in every step.
3. Different groups share no entries, so the gradient of one group does not involve another. Used in step 1.

### Steps

1. $\partial\mu/\partial x_j = 1/m$ and $\partial\sigma^2/\partial x_j = \frac{2}{m}(x_j - \mu)$, so $\partial s/\partial x_j = \hat{x}_j/m$. The quotient rule gives $\partial\hat{x}_k/\partial x_j = \frac{1}{s}\big(\mathbb{1}[k=j] - \frac{1}{m} - \frac{\hat{x}_k\hat{x}_j}{m}\big)$. The chain rule with $\partial\ell/\partial\hat{x}_k = u_k$ gives $\partial\ell/\partial x_j = \frac{1}{s}\big(u_j - \bar{u} - \hat{x}_j\,\overline{u\hat{x}}\big)$. This is item 1. It is the batch normalization backward pass of Ioffe and Szegedy, with the sum over a mini-batch replaced by a sum over a group.
2. Sum item 1 over $k$: $\sum_k u_k - m\bar{u} = 0$ and $\sum_k \hat{x}_k = 0$ by the group moments. So the sum is $0$. This is item 2.
3. Multiply item 1 by $\hat{x}_k$ and average. $\frac{1}{m}\sum_k \hat{x}_k u_k = \overline{u\hat{x}}$, $\frac{1}{m}\sum_k\hat{x}_k\bar{u} = 0$ and $\frac{1}{m}\sum_k\hat{x}_k^2 = \sigma^2/(\sigma^2+\epsilon)$. So the average is $\frac{1}{s}\overline{u\hat{x}}\big(1 - \frac{\sigma^2}{\sigma^2+\epsilon}\big) = \frac{\overline{u\hat{x}}}{s}\cdot\frac{\epsilon}{\sigma^2+\epsilon}$. This is item 3.
4. $y_k$ depends on $\gamma_c$ and $\beta_c$ only through the entries of channel $c$, with $\partial y_k/\partial\gamma_c = \hat{x}_k$ and $\partial y_k/\partial\beta_c = 1$. Summing over those entries gives item 4.

### Result and check

$$ \frac{\partial\ell}{\partial x_k} = \frac{1}{s}\Big(u_k - \bar{u} - \hat{x}_k\,\overline{u\hat{x}}\Big), \qquad \sum_k \frac{\partial\ell}{\partial x_k} = 0 . $$

With $\epsilon = 0$ the input gradient of a group is orthogonal to $\mathbf{1}$ and to $\hat{x}$: adding a constant to a group, or scaling it, does not change the output. A positive $\epsilon$ breaks only the second.
The check `gradient.check.py` compares item 1 with central finite differences of the full layer, with per-channel $\gamma$ and $\beta$, for several $G$, shapes and $\epsilon$, to a relative tolerance of $10^{-6}$. It checks items 2, 3 and 4 in each case, and the gradient of the worked example, $(1, -2, -2, 3)/54$.
