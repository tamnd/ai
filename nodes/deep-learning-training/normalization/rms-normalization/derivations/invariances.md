---
id: invariances
derives: [c-scale-invariance, c-scale-invariance-eps, c-row-not-invariant, c-not-recentering, c-ln-is-rms-of-centred, c-prms-invariance, eq-ln-as-rms]
kind: identity
assumes: ["the root mean square is computed from the same vector that is normalized", "epsilon is 0 for the scaling invariances", "delta > 0"]
uses: [sample-mean@m2, standard-deviation@m2]
status: full
check: invariances.check.py
lean: null
sources:
  - {work: zhang-2019-root-mean-square-layer-normalization, locator: "Section 4"}
  - {work: zhang-2019-root-mean-square-layer-normalization, locator: "Section 4.1"}
  - {work: zhang-2019-root-mean-square-layer-normalization, locator: "p. 5"}
---

### Statement

Let $\mathbf{a} = W\mathbf{x} \in \mathbb{R}^n$, let $1 \le k \le n$, let $r_k(\mathbf{a}) = \sqrt{\frac{1}{k}\sum_{i=1}^k a_i^2 + \epsilon}$ and let $\mathrm{RN}_k(\mathbf{a}) = \mathbf{g} \odot \mathbf{a} / r_k(\mathbf{a})$. With $k = n$ this is RMSNorm, eq-rms; with $k = \lceil np \rceil$ it is pRMSNorm, eq-prms. Let $\mathrm{LN}$ be layer normalization with the same $\epsilon$ and gain and no bias. Then

1. with $\epsilon = 0$ and $\delta > 0$, $\mathrm{RN}_k(\delta\mathbf{a}) = \mathrm{RN}_k(\mathbf{a})$, so $\mathrm{RN}_k((\delta W)\mathbf{x}) = \mathrm{RN}_k(W(\delta\mathbf{x})) = \mathrm{RN}_k(W\mathbf{x})$;
2. with $\epsilon > 0$ this fails: $r_k(\delta\mathbf{a})^2 = \delta^2 m_k + \epsilon$, where $m_k$ is the mean square of the first $k$ entries, so the output changes unless $\delta = 1$ or $\mathbf{a} = \mathbf{0}$;
3. scaling one row of $W$ alone can change the output;
4. adding the same constant to every entry of $\mathbf{a}$ can change the output;
5. $\mathrm{LN}(\mathbf{a}) = \mathrm{RN}_n(\mathbf{a} - \mu\mathbf{1})$ for every $\epsilon \ge 0$, where $\mu$ is the mean of $\mathbf{a}$.

### Assumptions

1. $r_k$ is computed from the same $\mathbf{a}$ that is normalized. Used in every step.
2. $\epsilon = 0$ for item 1. Used in step 1.
3. $\delta > 0$, so that $\sqrt{\delta^2} = \delta$. Used in step 1.
4. $r_k(\mathbf{a}) > 0$. Used in every step.

### Steps

1. The mean square of the first $k$ entries of $\delta\mathbf{a}$ is $\delta^2 m_k$. With $\epsilon = 0$ and $\delta > 0$, $r_k(\delta\mathbf{a}) = \delta\, r_k(\mathbf{a})$, which is Eq. 6 of Zhang and Sennrich. So $\delta\mathbf{a}/r_k(\delta\mathbf{a}) = \mathbf{a}/r_k(\mathbf{a})$. Both $(\delta W)\mathbf{x}$ and $W(\delta\mathbf{x})$ equal $\delta\mathbf{a}$, which gives item 1. The step uses only the first $k$ entries, so it holds for pRMSNorm too.
2. With $\epsilon > 0$, $\mathrm{RN}_k(\delta\mathbf{a}) = \mathbf{g}\odot\mathbf{a}\,\delta/\sqrt{\delta^2 m_k + \epsilon}$. The factor $\delta/\sqrt{\delta^2 m_k + \epsilon}$ grows strictly with $\delta$ when $m_k > 0$, so it equals $1/\sqrt{m_k + \epsilon}$ only at $\delta = 1$. This is item 2. The effect is small when $\epsilon \ll \delta^2 m_k$.
3. Item 3, by a counterexample. Take $W = I_2$, $\mathbf{x} = (1, 1)$ and $\mathbf{g} = \mathbf{1}$, so $\mathrm{RN}(\mathbf{a}) = (1, 1)$. Doubling the first row gives $\mathbf{a} = (2, 1)$ with $r = \sqrt{5/2}$, and the second output becomes $\sqrt{2/5} \ne 1$. The factor of one row changes the mean square by a different amount than the row itself, so it does not cancel.
4. Item 4, by a counterexample with $\mathbf{g} = \mathbf{1}$ and $\epsilon = 0$. The vector $(1, -1)$ has $r = 1$ and maps to $(1, -1)$. Adding 1 to both entries gives $(2, 0)$, with $r = \sqrt{2}$, which maps to $(\sqrt{2}, 0)$. Layer normalization maps both vectors to $(1, -1)$.
5. The vector $\mathbf{c} = \mathbf{a} - \mu\mathbf{1}$ has mean zero, so its mean square is $\frac{1}{n}\sum_i (a_i - \mu)^2 = \sigma^2$, the biased variance of $\mathbf{a}$. So $\mathrm{RN}_n(\mathbf{c}) = \mathbf{g}\odot(\mathbf{a} - \mu\mathbf{1})/\sqrt{\sigma^2 + \epsilon} = \mathrm{LN}(\mathbf{a})$. This is eq-ln-as-rms. Layer normalization first removes the mean and then does exactly what RMSNorm does.

### Result and check

$$ \mathrm{RN}(\delta\mathbf{a}) = \mathrm{RN}(\mathbf{a}) \ (\epsilon = 0,\ \delta > 0), \qquad \mathrm{LN}(\mathbf{a}) = \mathrm{RN}(\mathbf{a} - \mu\mathbf{1}) . $$

Item 5 says that RMSNorm is layer normalization without its first step. Item 4 is the price: the invariance to a shift belongs to the centring step, and RMSNorm gives it up.
The check `invariances.check.py` draws $W$, $\mathbf{x}$ and $\mathbf{g}$ with a fixed seed for $n \in \{2, 3, 7, 16\}$. It checks item 1 for four values of $\delta$ from $10^{-3}$ to $10^3$ and for $k = 1$, $k = \lfloor n/2 \rfloor$ and $k = n$, to a relative error of $10^{-12}$. It checks that item 1 fails with $\epsilon = 0.5$, checks item 5 for three values of $\epsilon$, and checks items 3 and 4 with random rows and with the numbers of step 4.
