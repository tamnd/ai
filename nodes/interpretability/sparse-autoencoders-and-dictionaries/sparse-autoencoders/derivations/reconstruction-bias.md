---
id: reconstruction-bias
derives: [c-gamma-def, c-gamma-shrink-example, c-fvu-mean-one, c-worked-l1, c-worked-topk]
kind: identity
assumes: ["a finite batch of inputs x and reconstructions x_hat", "sum of x_hat . x over the batch is positive", "for the FVU part, the inputs do not all coincide"]
uses: [l1-shrinkage]
status: full
check: reconstruction-bias.check.py
lean: null
sources:
  - {work: rajamanoharan-2024-gated-sparse-autoencoders, locator: "p. 8"}
  - {work: gao-2024-scaling-sparse-autoencoders, locator: "p. 2"}
---

### Statement

For a batch of inputs $x$ and reconstructions $\hat{x}$, with expectations taken over the batch, define

$$ \gamma = \frac{\mathbb{E}\lVert \hat{x} \rVert_2^2}{\mathbb{E}[\hat{x} \cdot x]}. $$

1. $1/\gamma$ is the scalar $c$ that minimizes $\mathbb{E}\lVert c\,\hat{x} - x \rVert_2^2$, so $\gamma < 1$ means that scaling every reconstruction up would lower the error: the reconstructions are systematically too short.
2. Equivalently $\gamma = 2\,\mathbb{E}\lVert \hat{x} \rVert^2 / (\mathbb{E}\lVert \hat{x} \rVert^2 + \mathbb{E}\lVert x \rVert^2 - \mathbb{E}\lVert \hat{x} - x \rVert^2)$.
3. For the one-latent case of the derivation `l1-shrinkage` with $x = s\,d$, $s > \lambda/2$, the reconstruction is $(s - \lambda/2)\,d$ and $\gamma = 1 - \lambda/(2s)$. For $s = \lambda = 1$ this is $1/2$.
4. The fraction of variance unexplained, $\mathrm{FVU} = \sum \lVert x - \hat{x} \rVert^2 / \sum \lVert x - \bar{x} \rVert^2$, equals 1 for the predictor that always outputs the batch mean $\bar{x}$, and 0 for a perfect reconstruction.

5. Worked example. For the inputs $(3, 1)$ and $(1, 3)$, an identity decoder, a tied encoder and no decoder bias, the L1 SAE with $\lambda = 1$ (encoder bias $-1/2$) codes them as $(2.5, 0.5)$ and $(0.5, 2.5)$, with L0 2, squared error 1, FVU 0.25 and $\gamma = 13/16$; with $\lambda = 2$ the codes are $(2, 0)$ and $(0, 2)$, with L0 1, squared error 4, FVU 1 and $\gamma = 2/3$; TopK with $k = 1$ gives $(3, 0)$ and $(0, 3)$, with L0 1, squared error 2, FVU 0.5 and $\gamma = 1$.

### Assumptions

1. The batch is finite and $\mathbb{E}[\hat{x} \cdot x] > 0$, so $\gamma$ is defined. If every $\hat{x}$ is zero, $\gamma$ is undefined.
2. For FVU, not all $x$ are equal, so the denominator is positive.

### Steps

1. $h(c) = c^2\,\mathbb{E}\lVert \hat{x} \rVert^2 - 2c\,\mathbb{E}[\hat{x} \cdot x] + \mathbb{E}\lVert x \rVert^2$ is a convex quadratic in $c$ when $\mathbb{E}\lVert \hat x \rVert^2 > 0$. Setting $h'(c) = 0$ gives $c^\star = \mathbb{E}[\hat{x} \cdot x] / \mathbb{E}\lVert \hat{x} \rVert^2 = 1/\gamma$.
2. The polarization identity $2\,a \cdot b = \lVert a \rVert^2 + \lVert b \rVert^2 - \lVert a - b \rVert^2$, averaged over the batch, turns the denominator of $\gamma$ into the second form.
3. With $x = s\,d$, $\lVert d \rVert = 1$, the derivation `l1-shrinkage` gives $f^\star = s - \lambda/2$. Then $\hat{x} = f^\star d$, $\lVert \hat{x} \rVert^2 = (f^\star)^2$, $\hat{x} \cdot x = f^\star s$, and $\gamma = f^\star / s = 1 - \lambda/(2s)$.
4. If $\hat{x} = \bar{x}$ for every row, the numerator of FVU equals the denominator, so FVU is 1. If $\hat{x} = x$ the numerator is 0.

5. The batch mean is $(2, 2)$, so the FVU denominator is $1 + 1 + 1 + 1 = 4$. With $\lambda = 1$ each input loses $1/2$ on both coordinates, so the squared error is $4 \cdot 1/4 = 1$; $\sum \lVert \hat{x} \rVert^2 = 2 \cdot 6.5 = 13$ and $\sum \hat{x} \cdot x = 2 \cdot 8 = 16$. With $\lambda = 2$ the small coordinate goes to zero and the large one loses 1, so the error is $2 \cdot (1 + 1) = 4$, $\sum \lVert \hat{x} \rVert^2 = 8$ and $\sum \hat{x} \cdot x = 12$. TopK keeps $3$ and drops $1$, so the error is $2$ and $\gamma = 18/18 = 1$.

### Result and check

The check verifies both forms of $\gamma$ against the minimizing scale found by a dense grid on random batches, the one-latent formula $1 - \lambda/(2s)$ against a numerical minimization of the one-latent loss, the two FVU end points, and every number of the worked example.
