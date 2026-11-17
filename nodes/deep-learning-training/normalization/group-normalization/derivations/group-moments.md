---
id: group-moments
derives: [c-group-moments]
kind: identity
assumes: ["the mean and the biased variance are computed over the same group that is normalized", "the moments are read before the per-channel scale and shift"]
uses: [sample-mean@m2, variance@m2]
status: full
check: group-moments.check.py
lean: null
sources:
  - {work: wu-2018-group-normalization, locator: "p. 3"}
---

### Statement

Let $\mathcal{S}$ be one group of $m$ entries of one example, with mean $\mu = \frac{1}{m}\sum_{k \in \mathcal{S}} x_k$ and biased variance $\sigma^2 = \frac{1}{m}\sum_{k \in \mathcal{S}} (x_k - \mu)^2$. Let $\hat{x}_k = (x_k - \mu)/\sqrt{\sigma^2 + \epsilon}$ with $\epsilon \ge 0$ and $\sigma^2 + \epsilon > 0$. Then, over the group,

1. the mean of $\hat{x}$ is $0$;
2. the biased variance of $\hat{x}$ is $\sigma^2/(\sigma^2 + \epsilon)$, which is $1$ when $\epsilon = 0$ and below $1$ when $\epsilon > 0$.

After the affine map $y_k = \gamma_c \hat{x}_k + \beta_c$, a channel $c$ of the group has its own moments, and the group as a whole need not have mean $0$ or variance $1$.

### Assumptions

1. $\mu$ and $\sigma^2$ come from the same entries that are normalized. Used in steps 1 and 2.
2. $\sigma^2 + \epsilon > 0$, so the division is defined. Used in every step.

### Steps

1. $\sum_{k \in \mathcal{S}} \hat{x}_k = \frac{1}{\sqrt{\sigma^2+\epsilon}}\sum_{k \in \mathcal{S}}(x_k - \mu) = \frac{1}{\sqrt{\sigma^2+\epsilon}}(m\mu - m\mu) = 0$. This is item 1.
2. Since the mean of $\hat{x}$ is $0$, its variance is the mean of $\hat{x}_k^2$, which is $\frac{1}{\sigma^2+\epsilon}\cdot\frac{1}{m}\sum_{k \in \mathcal{S}}(x_k-\mu)^2 = \frac{\sigma^2}{\sigma^2+\epsilon}$. This is item 2.
3. The statement holds for the whole group, not for each channel in it. A channel of the group can have any mean. In the worked example of the page the group of channels $0$ and $1$ normalizes to $(-1, -1)$ and $(1, 1)$: each channel has mean $\mp 1$ and variance $0$, and only the pair has mean $0$ and variance $1$.

### Result and check

$$ \frac{1}{m}\sum_{k \in \mathcal{S}} \hat{x}_k = 0, \qquad \frac{1}{m}\sum_{k \in \mathcal{S}} \hat{x}_k^2 = \frac{\sigma^2}{\sigma^2 + \epsilon} . $$

The check `group-moments.check.py` normalizes random tensors of shape $(N, C, H, W)$ for several $G$ and $\epsilon$, and checks the mean and the variance of every group to a relative tolerance of $10^{-10}$. It checks the worked example, where the channels of one group keep different means.
