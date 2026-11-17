---
id: z-loss-gradient
derives: [c-zloss-grad, c-zloss-shift, c-zloss-pull, eq-z-loss-grad]
kind: gradient
assumes: ["the logits z are a vector of n >= 2 real numbers", "the coefficient alpha is positive", "the loss is the softmax cross-entropy plus alpha (log Z)^2"]
uses: [cross-entropy-loss@al]
status: full
check: z-loss-gradient.check.py
lean: null
sources:
  - {work: chowdhery-2022-palm, locator: "p. 10"}
  - {work: zoph-2022-st-moe, locator: "p. 7"}
---

### Statement

Let $\mathbf{z} \in \mathbb{R}^n$ be the logits of one position, $Z = \sum_j e^{z_j}$, $\mathbf{p} = \mathrm{softmax}(\mathbf{z})$ and $\ell_z = \alpha (\log Z)^2$, as in eq-z-loss.

1. $\frac{\partial \ell_z}{\partial z_i} = 2\alpha \log Z \cdot p_i$ (eq-z-loss-grad). The gradient is 0 exactly when $\log Z = 0$.
2. Adding $s$ to every logit leaves the cross-entropy unchanged and adds $s$ to $\log Z$. On that line, $\ell_z$ is smallest at $s = -\log Z$, where $\log Z = 0$ and $z_i = \log p_i$.
3. A gradient step $\mathbf{z} \leftarrow \mathbf{z} - \eta \nabla \ell_z$ changes $\log Z$ by $-2\eta\alpha \log Z \sum_i p_i^2$ to first order in $\eta$, a change of the opposite sign to $\log Z$.

### Assumptions

1. $n \ge 2$ and $\alpha > 0$. Used in steps 2 and 5.
2. The task loss is the softmax cross-entropy $-\log p_y = \log Z - z_y$ for a target $y$. Used in step 4.

### Steps

1. $\frac{\partial \log Z}{\partial z_i} = \frac{1}{Z} \frac{\partial Z}{\partial z_i} = \frac{e^{z_i}}{Z} = p_i$, by the chain rule and the definition of the softmax.
2. By the chain rule on $\alpha u^2$ with $u = \log Z$, $\frac{\partial \ell_z}{\partial z_i} = 2\alpha \log Z \cdot p_i$. Each $p_i > 0$ and $\alpha > 0$, so the gradient is 0 exactly when $\log Z = 0$. This is part 1. Over $T$ positions with the mean, as in eq-z-loss, each position's gradient is divided by $T$.
3. The entries of the gradient sum to $2\alpha \log Z$, because $\sum_i p_i = 1$.
4. For any $s$, $\log \sum_j e^{z_j + s} = s + \log Z$ and $(z_y + s) - (s + \log Z) = z_y - \log Z$, so the cross-entropy of assumption 2 does not change and $\log Z$ moves by $s$. On this line, $\ell_z = \alpha(\log Z + s)^2$, a parabola with its minimum at $s = -\log Z$. There the new $\log Z$ is 0, so $Z = 1$ and $e^{z_i + s} = p_i$: the logits are the log-probabilities. This is part 2.
5. To first order, a change $\delta\mathbf{z}$ changes $\log Z$ by $\sum_i p_i\, \delta z_i$, by step 1. With $\delta z_i = -\eta \cdot 2\alpha \log Z \cdot p_i$ this is $-2\eta\alpha \log Z \sum_i p_i^2$. Since $\frac{1}{n} \le \sum_i p_i^2 \le 1$ (the lower bound by the Cauchy-Schwarz inequality, the upper because each $p_i \le 1$), the factor is positive and the change has the opposite sign to $\log Z$. This is part 3.

### Result and check

$$ \nabla_{\mathbf{z}}\, \alpha (\log Z)^2 = 2\alpha \log Z\; \mathrm{softmax}(\mathbf{z}). $$

The cross-entropy leaves the common shift of the logits free; the z-loss fixes it at $\log Z = 0$ and pulls $\log Z$ toward 0 at each step, with a force proportional to $\log Z$.

`z-loss-gradient.check.py` draws 50 logit vectors with $n$ from 2 to 29 and $\alpha$ from $10^{-5}$ to 1.
It compares the formula with central differences (in our run the largest relative gap was about $3 \times 10^{-9}$), checks that the entries sum to $2\alpha \log Z$, that the shift $s = -\log Z$ keeps the cross-entropy and gives $\log Z = 0$, and that one small step moves $\log Z$ by the first-order amount, toward 0.
The tests `test_z_loss_gradient_matches_finite_differences` and `test_z_loss_gradient_step_pulls_log_z_toward_zero` of `impl/numpy`, and `test_z_loss_gradient_matches_autograd` of `impl/pytorch`, check the code.
