---
id: one-step-improvement
derives: [c-one-step-optimum, c-overshoot, c-simple-relative-error, c-simple-isotropic]
kind: closed-form
assumes: ["the loss is quadratic near theta, or the expansion to second order in the step is accurate", "the batch gradient is unbiased with covariance Sigma / B", "plain SGD steps along the batch gradient"]
uses: []
status: full
check: one-step-improvement.check.py
lean: null
sources:
  - {work: mccandlish-2018-empirical-model-large-batch-training, locator: "p. 5"}
  - {work: mccandlish-2018-empirical-model-large-batch-training, locator: "p. 6"}
  - {work: mccandlish-2018-empirical-model-large-batch-training, locator: "p. 7"}
---

### Statement

Let $G$ be the true gradient and $H$ the Hessian at $\theta$, and let $G_B$ be the mean gradient over a batch of $B$ examples drawn with replacement, so that $\mathbb{E}[G_B] = G$ and $\operatorname{cov}(G_B) = \Sigma / B$.
After one step $\theta - \varepsilon G_B$ the expected change in the loss is

$$\mathbb{E}[\Delta L(\varepsilon)] = -\varepsilon \lVert G \rVert^2 + \frac{\varepsilon^2}{2}\Big(G^\top H G + \frac{\operatorname{tr}(H\Sigma)}{B}\Big).$$

(i) It is smallest at $\varepsilon_{\text{opt}}(B) = \varepsilon_{\max} / (1 + B_{\text{noise}}/B)$, where $\varepsilon_{\max} = \lVert G\rVert^2 / G^\top H G$ and $B_{\text{noise}} = \operatorname{tr}(H\Sigma) / G^\top H G$, and the best expected gain is $\Delta L_{\max} / (1 + B_{\text{noise}}/B)$ with $\Delta L_{\max} = \lVert G\rVert^4 / (2 G^\top H G)$.

(ii) Any step $\varepsilon > 2\varepsilon_{\text{opt}}(B)$ raises the expected loss.

(iii) $\mathbb{E}\lVert G_B - G\rVert^2 / \lVert G\rVert^2 = B_{\text{simple}} / B$ with $B_{\text{simple}} = \operatorname{tr}(\Sigma) / \lVert G\rVert^2$.

(iv) When $H = hI$ for a scalar $h > 0$, $B_{\text{noise}} = B_{\text{simple}}$.

### Assumptions

1. The loss is quadratic in the step, or the second-order expansion is accurate over the step taken. McCandlish et al. call this one of several "unfounded assumptions" (p. 6) and test it empirically instead.
2. The examples in a batch are independent draws from the data distribution. Without replacement from a finite set of $D$ examples, $1/B$ becomes $1/B - 1/D$ (p. 5).
3. The step is plain SGD; momentum and Adam fall outside the derivation.

### Steps

1. Expand the loss to second order: $L(\theta - \varepsilon G_B) = L(\theta) - \varepsilon G^\top G_B + \tfrac{\varepsilon^2}{2} G_B^\top H G_B$.
2. Take the expectation over the batch. $\mathbb{E}[G^\top G_B] = \lVert G\rVert^2$, and $\mathbb{E}[G_B^\top H G_B] = G^\top H G + \operatorname{tr}(H \operatorname{cov} G_B) = G^\top H G + \operatorname{tr}(H\Sigma)/B$. This gives the expression in the statement.
3. It is a parabola in $\varepsilon$ with positive curvature $a = G^\top H G + \operatorname{tr}(H\Sigma)/B$. Its minimum is at $\varepsilon = \lVert G\rVert^2 / a$. Dividing the numerator and denominator by $G^\top H G$ gives (i), and the value at the minimum is $-\lVert G\rVert^4 / (2a)$, which is the stated gain.
4. The parabola is zero at $\varepsilon = 0$ and at $\varepsilon = 2\lVert G\rVert^2 / a = 2\varepsilon_{\text{opt}}$, and positive beyond, giving (ii).
5. $\mathbb{E}\lVert G_B - G\rVert^2 = \operatorname{tr}\operatorname{cov}(G_B) = \operatorname{tr}(\Sigma)/B$. Dividing by $\lVert G\rVert^2$ gives (iii).
6. With $H = hI$, $\operatorname{tr}(H\Sigma) = h \operatorname{tr}\Sigma$ and $G^\top H G = h\lVert G\rVert^2$, and $h$ cancels, giving (iv).

### Result and check

The check builds 64 examples in 8 dimensions with per-example loss $\tfrac12 (\theta - c_i)^\top H (\theta - c_i)$, where $H$ has eigenvalues from 0.3 to 3.
At the chosen $\theta$ this gives $B_{\text{noise}} = 17.6$, $B_{\text{simple}} = 11.6$ and $\varepsilon_{\max} = 0.675$ (our run).
For $B$ from 1 to 1,024 it draws 200,000 batches with replacement, measures the mean loss after a step at 31 step sizes between a quarter and 1.75 times $\varepsilon_{\text{opt}}$, and fits a parabola.
The fitted minimum lies within 2% of $\varepsilon_{\text{opt}}(B)$ and the measured gain within 2% of $\Delta L_{\max}/(1 + B_{\text{noise}}/B)$.
For example, at $B = 16$ the formula gives $\varepsilon_{\text{opt}} = 0.3213$ and the simulation 0.3215.
The check also confirms, at $B = 16$, that a step of $2.2\,\varepsilon_{\text{opt}}$ raises the mean loss while $1.8\,\varepsilon_{\text{opt}}$ lowers it, that the relative gradient error matches $B_{\text{simple}}/B$, and that the two noise scales agree when $H$ is a multiple of the identity.
