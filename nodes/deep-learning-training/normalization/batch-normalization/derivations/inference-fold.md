---
id: inference-fold
derives: [c-fold, c-momentum-conventions, eq-inference]
kind: identity
assumes: ["the layer is in inference mode with fixed running statistics", "the linear layer comes directly before batch normalization", "both momentum conventions update with the same batch statistic"]
uses: [matrix-multiplication@m2]
status: full
check: inference-fold.check.py
lean: null
sources:
  - {work: ioffe-2015-batch-normalization, locator: "p. 4"}
  - {work: ioffe-2015-batch-normalization, locator: "Algorithm 2"}
---

### Statement

Let $\hat\mu, \hat v \in \mathbb{R}^C$ be the running mean and variance, $\gamma, \beta \in \mathbb{R}^C$ the scale and shift, and $\epsilon > 0$.
Let $a = \gamma / \sqrt{\hat v + \epsilon}$ elementwise. Then

1. in inference mode, $\mathrm{BN}(x) = a \odot x + (\beta - a \odot \hat\mu)$, an affine map (eq-inference);
2. for a linear layer $x = Wz + b$ with $W \in \mathbb{R}^{C \times k}$, $\mathrm{BN}(Wz + b) = W'z + b'$ with $W' = \mathrm{diag}(a)\, W$ and $b' = a \odot (b - \hat\mu) + \beta$;
3. the update $r \leftarrow (1 - \alpha)\, r + \alpha\, t$ equals the update $r \leftarrow \rho\, r + (1 - \rho)\, t$ when $\rho = 1 - \alpha$.

### Assumptions

1. In inference mode the layer uses the fixed $\hat\mu$ and $\hat v$, not the statistics of the input. Used in step 1.
2. Nothing comes between the linear layer and batch normalization. Used in step 2.
3. Both updates use the same batch statistic $t$, for example the unbiased batch variance in PyTorch. Used in step 3.

### Steps

1. By assumption 1, $\mathrm{BN}(x) = \gamma \odot (x - \hat\mu) / \sqrt{\hat v + \epsilon} + \beta = a \odot x - a \odot \hat\mu + \beta$. The terms $a$ and $\beta - a \odot \hat\mu$ do not depend on $x$, so the map is affine. This is item 1, Algorithm 2 of Ioffe and Szegedy (2015).
2. By assumption 2, put $x = Wz + b$ in step 1: $a \odot (Wz + b) - a \odot \hat\mu + \beta$. The elementwise product with $a$ is the product with $\mathrm{diag}(a)$, and matrix products are associative, so $a \odot (Wz) = (\mathrm{diag}(a) W) z$. Collect the terms without $z$ to get $b'$. This is item 2.
3. For a convolution, each output channel $k$ has one filter, and step 2 applies channel by channel: the filter of channel $k$ is multiplied by $a_k$, and the bias becomes $a_k (b_k - \hat\mu_k) + \beta_k$.
4. Put $\rho = 1 - \alpha$ in the second update: $\rho\, r + (1 - \rho)\, t = (1 - \alpha)\, r + \alpha\, t$, which is the first update. This is item 3. With assumption 3, a PyTorch momentum of 0.1 is a Keras or Flax momentum of 0.9, and their default 0.99 is a PyTorch momentum of 0.01.

### Result and check

$$ \mathrm{BN}(Wz + b) = \mathrm{diag}\Big(\frac{\gamma}{\sqrt{\hat v + \epsilon}}\Big) W z + \frac{\gamma \odot (b - \hat\mu)}{\sqrt{\hat v + \epsilon}} + \beta . $$

The check `inference-fold.check.py` draws the parameters, the running statistics and a batch with a fixed seed.
It compares the layer in inference mode with the affine map, and the linear layer followed by the layer with the folded layer.
It also compares the two forms of the running update over five steps, each comparison to a relative error under $10^{-12}$.
