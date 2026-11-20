---
id: jsd-identity
derives: [c-jsd]
kind: closed-form
assumes: ["the optimal discriminator of c-dstar", "p_data and p_g have densities"]
uses: [kullback-leibler-divergence@m2, f-divergence@m2]
status: full
check: jsd-identity.check.py
lean: null
sources:
  - {work: goodfellow-2014-generative-adversarial-nets, locator: "p. 5", note: "Theorem 1 and its proof, eqs 4 to 6"}
---

### Statement

With $D^*$ of `optimal-discriminator`, the virtual training criterion $C(G) = V(G, D^*_G)$ is

$$ C(G) = -\log 4 + 2\, \mathrm{JSD}(p_{\text{data}} \,\|\, p_g), $$

where $\mathrm{JSD}(p \,\|\, q) = \tfrac12 \mathrm{KL}(p \,\|\, m) + \tfrac12 \mathrm{KL}(q \,\|\, m)$ with $m = \tfrac12(p + q)$.
Its minimum, $-\log 4$, is reached exactly when $p_g = p_{\text{data}}$, where $D^* = \tfrac12$ everywhere.

### Assumptions

1. $D^* = p_{\text{data}} / (p_{\text{data}} + p_g)$. Used in step 1.
2. The Jensen-Shannon divergence is nonnegative and zero only when its arguments agree, because each KL term is. Used in step 3.

### Steps

1. Substitute $D^*$: $C(G) = \mathbb{E}_{p_{\text{data}}} \log \frac{p_{\text{data}}}{p_{\text{data}} + p_g} + \mathbb{E}_{p_g} \log \frac{p_g}{p_{\text{data}} + p_g}$.
2. Write $p_{\text{data}} + p_g = 2m$: each term is $\mathrm{KL}(\cdot \,\|\, m) - \log 2$, so $C(G) = -\log 4 + \mathrm{KL}(p_{\text{data}} \,\|\, m) + \mathrm{KL}(p_g \,\|\, m) = -\log 4 + 2\,\mathrm{JSD}(p_{\text{data}} \,\|\, p_g)$.
3. The JSD is $\ge 0$ with equality only for $p_g = p_{\text{data}}$, so $C(G) \ge -\log 4$ with equality there; then $D^* = p / 2p = \tfrac12$.

### Result and check

$$ C(G) = -\log 4 + 2\,\mathrm{JSD}(p_{\text{data}} \,\|\, p_g) \;\ge\; -\log 4 . $$

The check `jsd-identity.check.py` compares $V(G, D^*)$ with $-\log 4 + 2\,\mathrm{JSD}$ for pairs of Gaussians and Gaussian mixtures on a grid, checks the value $-\log 4$ when the two agree and the bound $\mathrm{JSD} \le \log 2$, which is reached by distributions with disjoint supports.
