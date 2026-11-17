---
id: logit-bound
derives: [c-ln-norm, c-logit-bound, c-bound-tight, c-scale-invariance, c-quadratic-growth, c-weight-floor, c-rms-norm, eq-bound]
kind: bound
assumes: ["the LayerNorm of queries and keys has eps = 0 and no bias", "each query and key projection is not constant over its d coordinates", "the gains g_q and g_k are fixed vectors in R^d"]
uses: [cauchy-schwarz-inequality@m2, layer-normalization@al]
status: full
check: logit-bound.check.py
lean: null
sources:
  - {work: dehghani-2023-scaling-vision-transformers-22b, locator: "p. 2"}
---

### Statement

Write $d$ for $d_k$, the width of one head. Let $\tilde{\mathbf{q}}, \tilde{\mathbf{k}} \in \mathbb{R}^d$ be the raw query and key of one head, $\tilde{\mathbf{q}} = \mathbf{W}_Q^\top \mathbf{x}_i$ and $\tilde{\mathbf{k}} = \mathbf{W}_K^\top \mathbf{x}_j$.
Let $\mathbf{q} = \mathbf{g}_q \odot \hat{\mathbf{q}}$ and $\mathbf{k} = \mathbf{g}_k \odot \hat{\mathbf{k}}$, where $\hat{\mathbf{q}}$ is $\tilde{\mathbf{q}}$ minus its mean, divided by its standard deviation, as in eq-qk-layernorm.
Let $z = \mathbf{q}^\top \mathbf{k} / \sqrt{d}$ be the attention logit and $M = \sqrt{d}\, \lVert \mathbf{g}_q \rVert_\infty \lVert \mathbf{g}_k \rVert_\infty$.

1. $\lVert \hat{\mathbf{q}} \rVert = \sqrt{d}$ for every $\tilde{\mathbf{q}}$ that is not constant.
2. $|z| \le M$, and with unit gains $|z| \le \sqrt{d}$.
3. The bound does not depend on $\mathbf{W}_Q$, $\mathbf{W}_K$ or $\mathbf{x}$: the normalized logit is the same for $c\mathbf{W}_Q$ and $c\mathbf{W}_K$ with $c > 0$, while the raw logit $\tilde{\mathbf{q}}^\top \tilde{\mathbf{k}} / \sqrt{d}$ is multiplied by $c^2$.
4. The bound is tight: with unit gains, $z = \sqrt{d}$ when $\hat{\mathbf{k}} = \hat{\mathbf{q}}$, and $z = -\sqrt{d}$ when $\hat{\mathbf{k}} = -\hat{\mathbf{q}}$.
5. For a row of $n$ logits, each attention weight is at least $1 / (1 + (n - 1) e^{2M})$ and at most $1 / (1 + (n - 1) e^{-2M})$.
6. Parts 1 to 4 also hold for the form without centering, $\hat{\mathbf{q}} = \tilde{\mathbf{q}} / \mathrm{rms}(\tilde{\mathbf{q}})$, for every $\tilde{\mathbf{q}} \ne \mathbf{0}$.

### Assumptions

1. The LayerNorm has $\epsilon = 0$ and no bias. Used in steps 1 and 2. Step 7 treats $\epsilon > 0$.
2. $\tilde{\mathbf{q}}$ and $\tilde{\mathbf{k}}$ are not constant vectors, so their standard deviations $s_q$ and $s_k$ are positive. Used in step 1, which divides by them.
3. The gains $\mathbf{g}_q$ and $\mathbf{g}_k$ are fixed vectors. Used in step 3. A gain that training changes moves $M$ with it.

### Steps

1. Write $\mu_q = \frac{1}{d}\sum_m \tilde{q}_m$ and $s_q^2 = \frac{1}{d}\sum_m (\tilde{q}_m - \mu_q)^2$, so $\hat{q}_m = (\tilde{q}_m - \mu_q)/s_q$ by assumption 1. Then $\sum_m \hat{q}_m^2 = \frac{1}{s_q^2}\sum_m (\tilde{q}_m - \mu_q)^2 = \frac{d s_q^2}{s_q^2} = d$, by the definition of $s_q^2$ and assumption 2. So $\lVert \hat{\mathbf{q}} \rVert = \sqrt{d}$. This is part 1.
2. For the gain, $\lVert \mathbf{g}_q \odot \hat{\mathbf{q}} \rVert^2 = \sum_m g_{q,m}^2 \hat{q}_m^2 \le \lVert \mathbf{g}_q \rVert_\infty^2 \sum_m \hat{q}_m^2 = \lVert \mathbf{g}_q \rVert_\infty^2 d$, because each $g_{q,m}^2$ is at most the largest one. The same holds for $\mathbf{k}$.
3. By the Cauchy-Schwarz inequality, $|\mathbf{q}^\top \mathbf{k}| \le \lVert \mathbf{q} \rVert \lVert \mathbf{k} \rVert \le \lVert \mathbf{g}_q \rVert_\infty \lVert \mathbf{g}_k \rVert_\infty d$, by step 2.
4. Dividing by $\sqrt{d} > 0$ keeps the order, so $|z| \le \sqrt{d}\, \lVert \mathbf{g}_q \rVert_\infty \lVert \mathbf{g}_k \rVert_\infty = M$. With $\mathbf{g}_q = \mathbf{g}_k = \mathbf{1}$, $M = \sqrt{d}$. This is part 2.
5. For $c > 0$, the raw query of $c\mathbf{W}_Q$ is $c\tilde{\mathbf{q}}$, with mean $c\mu_q$ and standard deviation $cs_q$, so its normalized form is $(c\tilde{\mathbf{q}} - c\mu_q\mathbf{1})/(cs_q) = \hat{\mathbf{q}}$. The same holds for the key, so $z$ does not change. The raw logit is bilinear: $(c\tilde{\mathbf{q}})^\top (c\tilde{\mathbf{k}}) / \sqrt{d} = c^2\, \tilde{\mathbf{q}}^\top \tilde{\mathbf{k}} / \sqrt{d}$. Since $M$ contains neither $\mathbf{W}_Q$, $\mathbf{W}_K$ nor $\mathbf{x}$, this is part 3.
6. If $\hat{\mathbf{k}} = \hat{\mathbf{q}}$, for example when $\tilde{\mathbf{k}} = c\tilde{\mathbf{q}} + s\mathbf{1}$ with $c > 0$, then with unit gains $z = \hat{\mathbf{q}}^\top \hat{\mathbf{q}} / \sqrt{d} = d / \sqrt{d} = \sqrt{d}$ by step 1. With $c < 0$, $\hat{\mathbf{k}} = -\hat{\mathbf{q}}$ and $z = -\sqrt{d}$. Constant gains $\mathbf{g}_q = \alpha\mathbf{1}$ and $\mathbf{g}_k = \beta\mathbf{1}$ give $z = \alpha\beta\sqrt{d}$, so the bound with gains is also reached. This is part 4.
7. With $\epsilon > 0$, step 1 becomes $\sum_m \hat{q}_m^2 = ds_q^2 / (s_q^2 + \epsilon) < d$, so the bound of part 2 still holds, with room to spare.
8. Fix a row and a key $j$. Each logit lies in $[-M, M]$ by part 2. The weight is $p_j = e^{z_j} / (e^{z_j} + \sum_{l \ne j} e^{z_l}) \ge e^{z_j} / (e^{z_j} + (n - 1)e^{M})$, because each other term is at most $e^{M}$. The map $t \mapsto t/(t + C)$ with $C > 0$ is increasing in $t > 0$, and $e^{z_j} \ge e^{-M}$, so $p_j \ge e^{-M} / (e^{-M} + (n - 1)e^{M}) = 1 / (1 + (n - 1)e^{2M})$. The same argument with the inequalities reversed gives $p_j \le 1 / (1 + (n - 1)e^{-2M})$. This is part 5.
9. Without centering, $\mathrm{rms}(\tilde{\mathbf{q}})^2 = \frac{1}{d}\sum_m \tilde{q}_m^2 = \lVert \tilde{\mathbf{q}} \rVert^2 / d$, which is positive when $\tilde{\mathbf{q}} \ne \mathbf{0}$. So $\lVert \hat{\mathbf{q}} \rVert = \lVert \tilde{\mathbf{q}} \rVert / \mathrm{rms}(\tilde{\mathbf{q}}) = \sqrt{d}$, and steps 2 to 6 go through unchanged, with $c\tilde{\mathbf{q}}$ normalized to $\hat{\mathbf{q}}$ for $c > 0$. This is part 6.

### Result and check

With unit gains and $\epsilon = 0$,

$$ \left| \frac{\mathrm{LN}(\tilde{\mathbf{q}})^\top \mathrm{LN}(\tilde{\mathbf{k}})}{\sqrt{d}} \right| \le \sqrt{d}, $$

for every choice of weights and inputs, and the bound is reached when the normalized key equals the normalized query.
Learned gains multiply the bound by $\lVert \mathbf{g}_q \rVert_\infty \lVert \mathbf{g}_k \rVert_\infty$.
For $d = 64$ and $n = 512$, part 5 gives weights between about $2.2 \times 10^{-10}$ and $0.99994$, so a row cannot become exactly one-hot.

`logit-bound.check.py` draws inputs and weights at scales from $10^{-3}$ to $10^{6}$ for $d$ from 2 to 128.
It asserts the norm $\sqrt{d}$, the bound with and without gains, the invariance to $c = 7$ and the factor 49 on the raw logits, the equality $z = \sqrt{d}$ when the key is the query, the smaller norm when $\epsilon = 10^{-2}$, the norm $\sqrt{d}$ without centering, and the weight floor and ceiling of part 5.
The tests `test_logit_bound_holds_for_any_weight_norm` and `test_logit_bound_is_tight` of `impl/numpy` check the same facts on the attention code.
