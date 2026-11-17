---
id: optimal-step-noise-scale
derives: [c-expected-loss, c-eps-opt, c-dl-opt, c-eps-regimes, c-b-simple-reduction, c-twice-eps]
kind: closed-form
assumes: ["the loss is quadratic near theta, or its second-order Taylor expansion is accurate over one step", "the batch holds B examples drawn independently, so the batch gradient has mean G and covariance Sigma / B", "G^T H G > 0"]
uses: []
status: full
check: optimal-step-noise-scale.check.py
lean: null
sources:
  - {work: mccandlish-2018-empirical-model-large-batch-training, locator: "p. 5"}
  - {work: mccandlish-2018-empirical-model-large-batch-training, locator: "p. 6"}
  - {work: mccandlish-2018-empirical-model-large-batch-training, locator: "p. 7"}
---

### Statement

Let $G$ be the true gradient and $H$ the Hessian at $\theta$, $\Sigma$ the covariance of the per-example gradients, and $G_{\mathrm{est}}$ the mean gradient of a batch of $B$ examples, so that $\mathbb{E}[G_{\mathrm{est}}] = G$ and $\operatorname{cov}(G_{\mathrm{est}}) = \Sigma / B$.
For the step $\theta - \epsilon\, G_{\mathrm{est}}$ on a quadratic loss:

(i) $\mathbb{E}[L(\theta - \epsilon G_{\mathrm{est}})] = L(\theta) - \epsilon \lvert G \rvert^2 + \tfrac12 \epsilon^2 \big(G^\top H G + \operatorname{tr}(H\Sigma)/B\big)$;

(ii) it is smallest at $\epsilon_{\mathrm{opt}}(B) = \epsilon_{\max} / (1 + B_{\mathrm{noise}}/B)$, with $\epsilon_{\max} = \lvert G \rvert^2 / G^\top H G$ and $B_{\mathrm{noise}} = \operatorname{tr}(H\Sigma) / G^\top H G$;

(iii) the best expected improvement is $\Delta L_{\mathrm{opt}}(B) = \Delta L_{\max} / (1 + B_{\mathrm{noise}}/B)$ with $\Delta L_{\max} = \tfrac12 \lvert G \rvert^4 / G^\top H G$;

(iv) for $B \ll B_{\mathrm{noise}}$, $\epsilon_{\mathrm{opt}} \approx \epsilon_{\max} B / B_{\mathrm{noise}}$ grows linearly in $B$; for $B \gg B_{\mathrm{noise}}$ it tends to $\epsilon_{\max}$; at $B = B_{\mathrm{noise}}$ it is $\epsilon_{\max}/2$;

(v) if $H = hI$, then $B_{\mathrm{noise}} = B_{\mathrm{simple}} = \operatorname{tr}(\Sigma) / \lvert G \rvert^2$, and $\mathbb{E}\lvert G_{\mathrm{est}} - G \rvert^2 / \lvert G \rvert^2 = B_{\mathrm{simple}} / B$;

(vi) the expected loss is back to $L(\theta)$ at $\epsilon = 2\epsilon_{\mathrm{opt}}$ and higher beyond it.

### Assumptions

1. $L$ is quadratic around $\theta$ (for a general loss, (i) holds to second order in $\epsilon$).
2. The $B$ examples are drawn independently from the data distribution, so the covariance of their mean is $\Sigma/B$; sampling without replacement from a finite set multiplies it by about $1 - B/N$.
3. $G^\top H G > 0$, so the parabola in $\epsilon$ opens upward.

### Steps

1. For a quadratic, $L(\theta - \epsilon V) = L(\theta) - \epsilon G^\top V + \tfrac12 \epsilon^2 V^\top H V$ exactly.
2. Take $V = G_{\mathrm{est}}$ and expectations: $\mathbb{E}[G^\top G_{\mathrm{est}}] = \lvert G \rvert^2$ and $\mathbb{E}[G_{\mathrm{est}}^\top H G_{\mathrm{est}}] = G^\top H G + \operatorname{tr}(H \operatorname{cov} G_{\mathrm{est}}) = G^\top H G + \operatorname{tr}(H\Sigma)/B$, which gives (i).
3. Setting the derivative in $\epsilon$ to zero gives $\epsilon_{\mathrm{opt}} = \lvert G \rvert^2 / (G^\top H G + \operatorname{tr}(H\Sigma)/B)$; dividing numerator and denominator by $G^\top H G$ gives (ii).
4. Substituting back, the improvement is $\tfrac12 \lvert G \rvert^4 / (G^\top H G + \operatorname{tr}(H\Sigma)/B)$, which is (iii).
5. (iv) reads off (ii) in the two limits.
6. With $H = hI$ the factors of $h$ cancel in $B_{\mathrm{noise}}$; and $\mathbb{E}\lvert G_{\mathrm{est}} - G \rvert^2 = \operatorname{tr}(\Sigma)/B$, which gives (v).
7. The expected change in loss is $-\epsilon\lvert G\rvert^2 + \tfrac12 \epsilon^2 \lvert G \rvert^2 / \epsilon_{\mathrm{opt}}$, a parabola with roots $0$ and $2\epsilon_{\mathrm{opt}}$, which gives (vi).

### Result and check

The check draws a random positive definite $H$ in six dimensions with eigenvalues from 0.2 to 5 and a random $\Sigma$, and for $B \in \{1, 4, 16, 64, 256\}$ samples 200,000 batch gradients.
The Monte Carlo expected loss matches (i) on a grid of step sizes; the minimum of a parabola fitted to it is within 3% of $\epsilon_{\mathrm{opt}}$; the gain at $\epsilon_{\mathrm{opt}}$ is within 3% of $\Delta L_{\mathrm{opt}}$; the gain at $2\epsilon_{\mathrm{opt}}$ is close to zero and at $2.3\,\epsilon_{\mathrm{opt}}$ the loss rises.
It also checks the linear and saturated regimes of (iv), the reduction (v) for $H = 2.5\, I$, and the normalized squared error of (v) by Monte Carlo.
