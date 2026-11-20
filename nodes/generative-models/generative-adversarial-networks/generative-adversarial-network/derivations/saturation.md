---
id: saturation
derives: [c-saturation]
kind: closed-form
assumes: ["D = sigmoid(a), with a the logit of the discriminator"]
uses: [cross-entropy@m2, logistic-regression@m2]
status: full
check: saturation.check.py
lean: null
sources:
  - {work: goodfellow-2014-generative-adversarial-nets, locator: "p. 3", note: "the saturation of log(1 - D(G(z))) and the alternative of maximizing log D(G(z))"}
---

### Statement

Let $a$ be the discriminator's logit at a fake, $D = \sigma(a)$.

1. The minimax generator loss $\log(1 - D) = -\operatorname{softplus}(a)$ has $\partial / \partial a = -\sigma(a) = -D$, which goes to $0$ as $D \to 0$.
2. The non-saturating loss $-\log D = \operatorname{softplus}(-a)$ has $\partial / \partial a = \sigma(a) - 1 = D - 1$, which goes to $-1$ as $D \to 0$.
3. Both derivatives are negative, so both losses push $a$ up, and the two losses are stationary in the generator at the same points: where $\mathbb{E}_z\, w(a)\, \nabla_\theta a(G_\theta(\mathbf{z})) = 0$ with the positive weights $w = D$ or $w = 1 - D$; at the equilibrium of `jsd-identity`, $D = \tfrac12$ and the weights agree.
4. Their ratio is $(1 - D) / D$: when the discriminator rejects the fakes with $D = 10^{-3}$ the non-saturating gradient is 999 times larger.

### Assumptions

1. $D = \sigma(a) = 1 / (1 + e^{-a})$. Used in steps 1 and 2.

### Steps

1. $\log(1 - \sigma(a)) = \log \frac{e^{-a}}{1 + e^{-a}} = -\log(1 + e^{a}) = -\operatorname{softplus}(a)$, and $\operatorname{softplus}'(a) = \sigma(a)$. This is statement 1.
2. $-\log \sigma(a) = \log(1 + e^{-a}) = \operatorname{softplus}(-a)$, with derivative $-\sigma(-a) = \sigma(a) - 1$. This is statement 2.
3. By the chain rule the gradient in $\theta$ of either loss averaged over $\mathbf{z}$ is $-\mathbb{E}_z\, w(a)\, \nabla_\theta a$ with $w = \sigma(a)$ or $1 - \sigma(a)$; at $D = \tfrac12$ the two are equal. This is statement 3.
4. $(1 - D) / D = 999$ at $D = 10^{-3}$. This is statement 4.

### Result and check

$$ \frac{\partial}{\partial a} \log\big(1 - \sigma(a)\big) = -\sigma(a), \qquad \frac{\partial}{\partial a} \big(-\log \sigma(a)\big) = \sigma(a) - 1 . $$

The check `saturation.check.py` compares both derivatives with finite differences over logits from $-30$ to $30$, compares the generator gradients of the NumPy reference for both losses with the weights $D$ and $1 - D$, and checks the ratio of statement 4.
