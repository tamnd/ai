---
id: made-masks
derives: [c-made-masks]
kind: closed-form
assumes: ["input degrees m_0 are a permutation of 1..D", "the degrees of hidden layer l lie in [min_k m_(l-1)(k), D - 1]", "weights are multiplied elementwise by the masks before every pass"]
uses: [matrix-multiplication@m2, hadamard-product@m2, permutation-matrix@m2, conditional-probability@m2]
status: full
check: made-masks.check.py
lean: null
sources:
  - {work: germain-2015-made-masked-autoencoder, locator: "pp. 3, 4", note: "the masks of eqs. 8, 9, 12 and 13, the direct mask of eq. 11 and the proof by mask products of eq. 10"}
---

### Statement

Give input $i$ the degree $m_0(i)$, its position in the ordering, and give unit $k$ of hidden layer $l$ a degree $m_l(k) \in [\min_{k'} m_{l-1}(k'), D - 1]$. Mask the weights from layer $l - 1$ to layer $l$ with $M^{W_l}_{k, k'} = 1[m_l(k) \ge m_{l-1}(k')]$, the weights into output $d$ with $M^V_{d, k} = 1[m_0(d) > m_L(k)]$ and the direct weights with $M^A_{d, i} = 1[m_0(d) > m_0(i)]$. Then, for every value of the weights, logit $a_d$ is a function of the inputs $x_i$ with $m_0(i) < m_0(d)$ alone. So $\sigma(a_d)$ is a valid conditional $p(x_d = 1 \mid \text{the inputs before } d)$, and the product of eq-chain is normalized.

### Assumptions

1. The masks are applied to the weights at every pass, so a masked weight contributes nothing whatever its value. Used in step 2.
2. Hidden degrees are at most $D - 1$. A unit of degree $D$ could only feed outputs with $m_0(d) > D$, of which there are none, so it would be wasted. Degrees are at least the smallest degree of the layer below, so that every unit has at least one input. Neither bound is needed for the autoregressive property. Both keep units from being dead or constant.

### Steps

1. Claim: unit $k$ of hidden layer $l$ depends only on inputs $i$ with $m_0(i) \le m_l(k)$. For $l = 1$ this is the mask $1[m_1(k) \ge m_0(i)]$.
2. Induction step: unit $k$ of layer $l$ reads unit $k'$ of layer $l - 1$ only if $m_{l-1}(k') \le m_l(k)$. By the induction hypothesis, unit $k'$ depends only on inputs with $m_0(i) \le m_{l-1}(k')$, so unit $k$ depends only on inputs with $m_0(i) \le m_l(k)$. An activation applied elementwise does not mix units.
3. Output $d$ reads last-layer unit $k$ only if $m_L(k) < m_0(d)$, so through the hidden path it depends only on inputs with $m_0(i) \le m_L(k) < m_0(d)$. The direct path adds only inputs with $m_0(i) < m_0(d)$.
4. In matrix form, the number of paths from input $i$ to output $d$ is the $(d, i)$ entry of $M^V M^{W_L} \cdots M^{W_1}$, and the argument shows that this entry is zero unless $m_0(i) < m_0(d)$. For the natural ordering $m_0(d) = d$ the product is strictly lower triangular, as in eq. 10 of Germain and colleagues.
5. The Jacobian $\partial a_d / \partial x_i$ is therefore zero unless $m_0(i) < m_0(d)$. Reordered by $m_0$, it is strictly lower triangular. The first variable in the ordering gets a constant logit $c_d$, its marginal.

### Result and check

$$ \frac{\partial a_d}{\partial x_i} = 0 \quad \text{whenever} \quad m_0(i) \ge m_0(d), $$

for every value of the weights, so MADE is an autoregressive model in the ordering $m_0$ with one pass of an ordinary network.

The check `made-masks.check.py` draws 40 random architectures, orderings and degree assignments. For each it computes the path counts as the product of the masks and confirms that they vanish outside the strict ordering. It confirms that the finite-difference Jacobian of the logits vanishes there too. It also confirms that with a direct connection every allowed pair is linked, and that the masks of a layer with one hidden unit of degree $D$ would cut that unit off from every output.
