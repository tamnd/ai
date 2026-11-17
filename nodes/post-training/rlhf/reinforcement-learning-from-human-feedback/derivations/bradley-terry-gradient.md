---
id: bradley-terry-gradient
derives: [c-bt-logistic, c-bt-gradient, c-bt-curvature, c-bt-shift, c-bt-softmax-two, eq-bt-grad]
kind: identity
assumes: ["the human prefers y_w to y_l with probability exp(r_w) / (exp(r_w) + exp(r_l))", "the reward model r_theta(x, y) is differentiable in theta"]
uses: [logistic-regression@m2, maximum-likelihood-estimation@m2, cross-entropy@m2]
status: full
check: bradley-terry-gradient.check.py
lean: null
sources:
  - {work: christiano-2017-deep-rl-human-preferences, locator: "Section 2.2.3"}
  - {work: rafailov-2023-direct-preference-optimization, locator: "Section 3"}
  - {work: ouyang-2022-training-language-models-follow-instructions, locator: "Section 3.5"}
---

### Statement

Write $r_w = r_\theta(x, y_w)$ and $r_l = r_\theta(x, y_l)$ for the rewards of the preferred and the other response, $\Delta = r_w - r_l$, and $L = -\log \sigma(\Delta)$ for the loss of one pair (eq-rm-loss). Then

1. $\dfrac{e^{r_w}}{e^{r_w} + e^{r_l}} = \sigma(\Delta)$, so $L = \log(1 + e^{-\Delta})$, the logistic loss of $\Delta$ with the label 1, and $L = \log 2$ at $\Delta = 0$;
2. $\partial L / \partial r_w = -\sigma(-\Delta) = -(1 - \sigma(\Delta))$ and $\partial L / \partial r_l = +\sigma(-\Delta)$, so that
   $$ \nabla_\theta L = -\sigma(-\Delta)\,\big(\nabla_\theta r_w - \nabla_\theta r_l\big) ; $$ {#eq-bt-grad}
   the weight $\sigma(-\Delta)$ is below 1, above $1/2$ when the model orders the pair wrong and below $1/2$ when it orders it right;
3. $\partial^2 L / \partial \Delta^2 = \sigma(\Delta)\,\sigma(-\Delta) \in (0, 1/4]$, so $L$ is convex in $\Delta$;
4. for any function $c(x)$ of the prompt alone, $r_\theta + c$ gives the same $L$ for every pair, so the data fix the reward only up to such a shift;
5. the cross-entropy over the two responses with the preferred one as the label, $-\log\big(e^{r_w} / (e^{r_w} + e^{r_l})\big)$, is $L$, so the softmax loss over several options of Ziegler et al. with two options is the Bradley–Terry loss.

### Assumptions

1. The Bradley–Terry model of eq-bt, with the exponentiated reward as the strength of a response. Used in item 1. Christiano et al. sum the reward over the segment before exponentiating; the identities hold for any such score.
2. $r_\theta$ is differentiable in $\theta$. Used in item 2 for the chain rule.

### Steps

1. Divide the numerator and the denominator of $e^{r_w}/(e^{r_w} + e^{r_l})$ by $e^{r_w}$: it becomes $1/(1 + e^{-(r_w - r_l)}) = \sigma(\Delta)$. Then $-\log \sigma(\Delta) = \log(1 + e^{-\Delta})$, which is $\log 2$ at $\Delta = 0$. This is item 1, and it is why the loss is the negative log-likelihood of a binary classification in eq. 2 of Rafailov et al.
2. $\frac{d}{d\Delta}\log(1 + e^{-\Delta}) = -e^{-\Delta}/(1 + e^{-\Delta}) = -\sigma(-\Delta)$, and $\sigma(-\Delta) = 1 - \sigma(\Delta)$. Since $\partial \Delta/\partial r_w = 1$ and $\partial \Delta / \partial r_l = -1$, the two partial derivatives follow, and the chain rule through $r_\theta$ gives eq-bt-grad. $\sigma(-\Delta) > 1/2$ exactly when $\Delta < 0$. This is item 2.
3. Differentiate $-\sigma(-\Delta)$ once more: $\sigma'(u) = \sigma(u)\sigma(-u)$ gives $\sigma(\Delta)\sigma(-\Delta)$, which is positive, and is at most $1/4$ because $s(1 - s) \le 1/4$ for $s \in (0, 1)$, with equality at $s = 1/2$. This is item 3.
4. $(r_w + c(x)) - (r_l + c(x)) = \Delta$, and $L$ depends on the rewards only through $\Delta$. This is item 4, the invariance that Ouyang et al. note for their loss, which they remove by a bias that gives the demonstrations a mean score of 0.
5. The cross-entropy with the label $y_w$ is $-\log$ of the softmax probability of $y_w$, which item 1 shows is $\sigma(\Delta)$. With four options, the loss (1) of Ziegler et al. is the same cross-entropy over four scores. This is item 5.

### Result and check

$$ L = \log\!\big(1 + e^{-(r_w - r_l)}\big), \qquad \frac{\partial L}{\partial r_w} = -\frac{\partial L}{\partial r_l} = -\sigma(r_l - r_w), \qquad \frac{\partial^2 L}{\partial \Delta^2} = \sigma(\Delta)\,\sigma(-\Delta) \le \tfrac14 . $$

The check `bradley-terry-gradient.check.py` compares these with central differences on 1000 random pairs with a fixed seed, checks the shift invariance for random shifts, the two-way cross-entropy, and the gradient through a linear reward model.
