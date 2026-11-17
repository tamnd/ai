---
id: init-moments
derives: [c-init-moments, c-init-batch-only, c-init-unbiased, eq-init]
kind: identity
assumes: ["the pre-activations t of a unit are not constant on the initialization batch", "sigma[t] is the biased standard deviation over the batch, without an epsilon"]
uses: [sample-mean@m2, variance@m2]
status: full
check: init-moments.check.py
lean: null
sources:
  - {work: salimans-2016-weight-normalization, locator: "p. 4"}
---

### Statement

Take a unit with direction $v \neq 0$ and a batch $x_1, \dots, x_m$ with $m \geq 2$. Let $t_n = v\cdot x_n/\lVert v\rVert$, let $\mu = \frac{1}{m}\sum_n t_n$ and $\sigma^2 = \frac{1}{m}\sum_n (t_n - \mu)^2 > 0$. Set $g = 1/\sigma$ and $b = -\mu/\sigma$. Then the pre-activations $y_n = w\cdot x_n + b$, with $w = g v/\lVert v\rVert$, satisfy

1. $y_n = (t_n - \mu)/\sigma$;
2. $\frac{1}{m}\sum_n y_n = 0$ and $\frac{1}{m}\sum_n y_n^2 = 1$;
3. if $\sigma$ is the unbiased standard deviation, with $m - 1$ in place of $m$, the biased variance of $y$ is $(m-1)/m$ instead of 1.

### Assumptions

1. $\sigma > 0$: the unit's projections are not all equal on the batch. Used in the definition of $g$.
2. $\sigma$ is the biased standard deviation. Used in step 3; step 4 drops it.
3. The statement is about the batch used for the initialization. On another batch, $\mu$ and $\sigma$ differ and neither moment is fixed.

### Steps

1. $w\cdot x_n = g\,v\cdot x_n/\lVert v\rVert = g\,t_n$. So $y_n = g t_n + b = t_n/\sigma - \mu/\sigma = (t_n - \mu)/\sigma$. This is item 1.
2. The mean of $t_n - \mu$ over the batch is $\mu - \mu = 0$, so the mean of $y$ is 0.
3. The mean of $(t_n - \mu)^2$ is $\sigma^2$ by definition, so the mean of $y_n^2$ is $\sigma^2/\sigma^2 = 1$. Since the mean of $y$ is 0, this is the variance. This is item 2.
4. With the unbiased $\sigma_u^2 = \frac{1}{m-1}\sum_n (t_n-\mu)^2 = \frac{m}{m-1}\sigma^2$, the mean of $y_n^2$ is $\sigma^2/\sigma_u^2 = (m-1)/m$. This is item 3.

### Result and check

$$ g \leftarrow \frac{1}{\sigma[t]}, \quad b \leftarrow -\frac{\mu[t]}{\sigma[t]} \quad\Longrightarrow\quad \mu[y] = 0, \ \ \sigma^2[y] = 1 \text{ on the initialization batch.} $$

This is batch normalization applied once, at the start, with the result stored in $g$ and $b$. After that, $g$ and $b$ are ordinary parameters.
The check `init-moments.check.py` applies the rule to 100 random layers with 7 units and batches of 2, 3, 10 and 100 rows, and checks item 2 to $10^{-10}$ and item 3. It checks that a second batch is not normalized: the variance is off by more than 0.1 somewhere. It also checks the worked example $g = 0.4$, $b = -1.4$.
