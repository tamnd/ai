---
id: bn-backward
derives: [c-backward, c-dx-orthogonal, eq-bn-backward]
kind: gradient
assumes: ["the batch statistics come from the same batch as the output", "the gradient flows through the batch mean and the batch variance", "m > 1 and the batch variance plus epsilon is positive"]
uses: [backpropagation@al, sample-mean@m2, sample-variance@m2, orthogonal-projection@m2]
status: full
check: bn-backward.check.py
lean: null
sources:
  - {work: ioffe-2015-batch-normalization, locator: "Algorithm 1"}
  - {work: ioffe-2015-batch-normalization, locator: "p. 4"}
---

### Statement

Let $x_1, \dots, x_m$ be the values of one feature over a batch, with $m > 1$.
Let $\mu = \frac{1}{m}\sum_i x_i$, $\sigma^2 = \frac{1}{m}\sum_i (x_i - \mu)^2$, $s = \sqrt{\sigma^2 + \epsilon}$, $\hat{x}_i = (x_i - \mu)/s$ and $y_i = \gamma \hat{x}_i + \beta$.
Let $g_i = \partial \ell / \partial y_i$. Then

1. $\partial \ell / \partial \beta = \sum_i g_i$ and $\partial \ell / \partial \gamma = \sum_i g_i \hat{x}_i$;
2. $\partial \ell / \partial x_i = \frac{\gamma}{m s} \big( m g_i - \sum_j g_j - \hat{x}_i \sum_j g_j \hat{x}_j \big)$;
3. $\sum_i \partial \ell / \partial x_i = 0$;
4. if $\epsilon = 0$, then $\sum_i \hat{x}_i \, \partial \ell / \partial x_i = 0$, and $\partial \ell / \partial \mathbf{x} = \frac{\gamma}{s} P \mathbf{g}$ with $P = I - \frac{1}{m}\mathbf{1}\mathbf{1}^\top - \frac{1}{m}\hat{\mathbf{x}}\hat{\mathbf{x}}^\top$ an orthogonal projection.

### Assumptions

1. The mean and the variance are computed from the same batch that is normalized, and they are not treated as constants. Used in steps 3 to 5.
2. $m > 1$ and $\sigma^2 + \epsilon > 0$, so $s > 0$ and the division is defined. Used in steps 3 and 4.
3. $\epsilon = 0$ for item 4, so that $\sum_i \hat{x}_i^2 = m$. Used in step 7.

### Steps

1. Each $y_i$ depends on $\gamma$ and $\beta$ only through $\gamma \hat{x}_i + \beta$. So $\partial \ell / \partial \beta = \sum_i g_i$ and $\partial \ell / \partial \gamma = \sum_i g_i \hat{x}_i$. This is item 1.
2. The output $y_j$ depends on $x$ only through $\hat{x}_j$, so $\partial \ell / \partial \hat{x}_j = \gamma g_j$ and $\partial \ell / \partial x_i = \gamma \sum_j g_j \, \partial \hat{x}_j / \partial x_i$.
3. By assumption 1, $\mu$ and $\sigma^2$ depend on every $x_i$. The mean gives $\partial \mu / \partial x_i = 1/m$. The variance gives $\partial \sigma^2 / \partial x_i = \frac{2}{m}(x_i - \mu) - \frac{2}{m^2}\sum_j (x_j - \mu) = \frac{2}{m}(x_i - \mu)$, because $\sum_j (x_j - \mu) = 0$.
4. Then $\partial s / \partial x_i = \frac{1}{2s} \cdot \frac{2}{m}(x_i - \mu) = \hat{x}_i / m$. The quotient rule on $\hat{x}_j = (x_j - \mu)/s$ gives $\partial \hat{x}_j / \partial x_i = \frac{1}{s}\big(\delta_{ij} - \frac{1}{m}\big) - \frac{x_j - \mu}{s^2} \cdot \frac{\hat{x}_i}{m} = \frac{1}{s}\big(\delta_{ij} - \frac{1}{m} - \frac{1}{m}\hat{x}_i \hat{x}_j\big)$.
5. Put step 4 into step 2: $\partial \ell / \partial x_i = \frac{\gamma}{s}\big(g_i - \frac{1}{m}\sum_j g_j - \frac{1}{m}\hat{x}_i \sum_j g_j \hat{x}_j\big)$. Multiplying inside by $m$ and dividing outside by $m$ gives item 2. The same result follows from the separate chain-rule equations for $\partial \ell / \partial \sigma^2$ and $\partial \ell / \partial \mu$ of Ioffe and Szegedy (2015, p. 4).
6. The mean of the normalized values is zero: $\sum_i \hat{x}_i = \frac{1}{s}\sum_i (x_i - \mu) = 0$. Summing item 2 over $i$ gives $\frac{\gamma}{ms}\big(m\sum_i g_i - m\sum_j g_j - 0\big) = 0$. This is item 3.
7. In general $\sum_i \hat{x}_i^2 = m\sigma^2/(\sigma^2 + \epsilon)$, which is $m$ when $\epsilon = 0$ by assumption 3. Then $\sum_i \hat{x}_i \, \partial \ell / \partial x_i = \frac{\gamma}{ms}\big(m \sum_i g_i \hat{x}_i - 0 - m \sum_j g_j \hat{x}_j\big) = 0$.
8. In vector form, step 5 reads $\partial \ell / \partial \mathbf{x} = \frac{\gamma}{s} P \mathbf{g}$. The vectors $\mathbf{1}$ and $\hat{\mathbf{x}}$ are orthogonal by step 6, and each has squared norm $m$ by step 7. So $P$ is the identity minus the orthogonal projections onto $\mathbf{1}$ and onto $\hat{\mathbf{x}}$, which is the orthogonal projection onto the complement of their span. This is item 4.

### Result and check

$$ \frac{\partial \ell}{\partial x_i} = \frac{\gamma}{m s}\Big(m g_i - \sum_j g_j - \hat{x}_i \sum_j g_j \hat{x}_j\Big), \qquad \sum_i \frac{\partial \ell}{\partial x_i} = 0 . $$

With $\epsilon > 0$, the orthogonality to $\hat{\mathbf{x}}$ is only approximate: step 7 gives $\sum_i \hat{x}_i \, \partial \ell / \partial x_i = \frac{\gamma}{s} \cdot \frac{\epsilon}{\sigma^2 + \epsilon} \sum_j g_j \hat{x}_j$.
The check `bn-backward.check.py` draws batches of 2 to 64 values with a fixed seed, for $\epsilon$ of 0, $10^{-5}$ and 0.3.
It compares items 1 and 2 with central finite differences of $\ell = \sum_i c_i \tanh(y_i)$, to a relative error under $10^{-7}$.
It compares item 2 with the separate equations of Ioffe and Szegedy, and checks items 3 and 4 and the formula for $\epsilon > 0$ to $10^{-12}$.
