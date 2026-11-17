---
id: dpo-gradient
derives: [c-gradient-form, c-we-pair, eq-dpo-gradient]
kind: gradient
assumes: ["the policy log-probabilities log pi_theta(y|x) are differentiable in theta", "the reference log-probabilities do not depend on theta", "beta > 0"]
uses: [logistic-regression@m2]
status: full
check: dpo-gradient.check.py
lean: null
sources:
  - {work: rafailov-2023-direct-preference-optimization, locator: "p. 5"}
  - {work: rafailov-2023-direct-preference-optimization, locator: "p. 17"}
---

### Statement

For one pair, write $a_w = \log\pi_\theta(y_w\mid x)$, $a_l = \log\pi_\theta(y_l\mid x)$, $\hat r_w = \beta(a_w - \log\pi_{\mathrm{ref}}(y_w\mid x))$, $\hat r_l = \beta(a_l - \log\pi_{\mathrm{ref}}(y_l\mid x))$ and $\ell = -\log\sigma(\hat r_w - \hat r_l)$. Then

1. $\partial\ell/\partial a_w = -\beta\,\sigma(\hat r_l - \hat r_w)$ and $\partial\ell/\partial a_l = +\beta\,\sigma(\hat r_l - \hat r_w)$;
2. $\nabla_\theta\ell = -\beta\,\sigma(\hat r_l - \hat r_w)\big(\nabla_\theta\log\pi_\theta(y_w\mid x) - \nabla_\theta\log\pi_\theta(y_l\mid x)\big)$, and the gradient of the loss is its expectation over the data;
3. the weight $\sigma(\hat r_l - \hat r_w)$ lies in $(0, 1)$, is $\tfrac12$ when $\hat r_w = \hat r_l$, and decreases to 0 as the margin $\hat r_w - \hat r_l$ grows.

### Assumptions

1. $\log\pi_\theta(y\mid x)$ is differentiable in $\theta$, as it is for a neural network with a softmax output. Used in step 3.
2. The reference log-probabilities are constants for $\theta$; in code they are computed without gradient or detached. Used in step 2.
3. $\beta > 0$.

### Steps

1. $\frac{d}{dm}\big(-\log\sigma(m)\big) = -\frac{\sigma(m)(1 - \sigma(m))}{\sigma(m)} = -(1 - \sigma(m)) = -\sigma(-m)$.
2. With $m = \hat r_w - \hat r_l$, $\partial m/\partial a_w = \beta$ and $\partial m/\partial a_l = -\beta$, since the reference terms are constants. The chain rule gives item 1, with $\sigma(-m) = \sigma(\hat r_l - \hat r_w)$.
3. The chain rule through $a_w(\theta)$ and $a_l(\theta)$ gives item 2. Taking the expectation over the data gives the gradient of the loss, equation 7's gradient in Rafailov et al.
4. $\sigma$ maps the real line onto $(0, 1)$, is increasing, and $\sigma(0) = \tfrac12$; so $\sigma(-m)$ is decreasing in $m$, equal to $\tfrac12$ at $m = 0$, and tends to 0 as $m\to\infty$. This is item 3. The weight is never exactly 0, so a pair is never fully learned.

### Result and check

$$ \nabla_\theta\mathcal L_{\mathrm{DPO}} = -\beta\,\mathbb E\Big[\sigma\big(\hat r_\theta(x, y_l) - \hat r_\theta(x, y_w)\big)\big(\nabla_\theta\log\pi_\theta(y_w\mid x) - \nabla_\theta\log\pi_\theta(y_l\mid x)\big)\Big] . $$

The check `dpo-gradient.check.py` compares item 1 with central finite differences on 2000 random pairs with log-probabilities of scale 8 around $-20$ and $\beta$ from 0.01 to 2, and item 2 on 500 softmax policies over 2 to 7 responses.
For the worked example of the page, log-probabilities $(-12, -15)$ under the policy and $(-13, -14)$ under the reference with $\beta = 0.1$, it gives the loss 0.598139, the weight 0.450166 and $\partial\ell/\partial a_w = -0.045017$.
