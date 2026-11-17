---
id: zero-mean
derives: [c-zero-mean-equal, c-mean-mix, c-gap-formula, c-output-mean, eq-rms-mix]
kind: identity
assumes: ["the same epsilon and gain in both layers, and no bias", "epsilon is 0 for the formulas with a non-zero mean", "the input is not constant, so sigma > 0"]
uses: [sample-mean@m2, standard-deviation@m2]
status: full
check: zero-mean.check.py
lean: null
sources:
  - {work: zhang-2019-root-mean-square-layer-normalization, locator: "Section 4"}
---

### Statement

Let $\mathbf{a} \in \mathbb{R}^n$ have mean $\mu$ and biased standard deviation $\sigma > 0$. Let $\mathrm{RN}$ be RMSNorm and $\mathrm{LN}$ layer normalization, both with the gain $\mathbf{g}$, the same $\epsilon$ and no bias. Then

1. if $\mu = 0$, $\mathrm{RN}(\mathbf{a}) = \mathrm{LN}(\mathbf{a})$, for every $\epsilon \ge 0$;
2. with $\epsilon = 0$ and $\rho = \mu/\sigma$, $\mathrm{RN}(\mathbf{a}) = \dfrac{\mathrm{LN}(\mathbf{a}) + \rho\,\mathbf{g}}{\sqrt{1 + \rho^2}}$;
3. with $\epsilon = 0$ and $\mathbf{g} = \mathbf{1}$, the root mean square of $\mathrm{RN}(\mathbf{a}) - \mathrm{LN}(\mathbf{a})$ is $\sqrt{2 - 2/\sqrt{1 + \rho^2}}$, which is at most $\lvert\rho\rvert$, is about $\lvert\rho\rvert$ for small $\rho$, and tends to $\sqrt{2}$ as $\lvert\rho\rvert$ grows;
4. with $\epsilon = 0$ and $\mathbf{g} = \mathbf{1}$, the mean of $\mathrm{RN}(\mathbf{a})$ is $\rho/\sqrt{1 + \rho^2}$.

### Assumptions

1. Both layers use the same $\epsilon$ and $\mathbf{g}$ and no bias. Used in steps 1 and 2.
2. $\epsilon = 0$ in items 2 to 4. With $\epsilon > 0$, replace $\sigma^2$ by $\sigma^2 + \epsilon$ in $\rho$ for the denominator. Used in step 2.
3. $\sigma > 0$. Used in steps 2 to 4.

### Steps

1. If $\mu = 0$ then $\mathbf{a} - \mu\mathbf{1} = \mathbf{a}$, and the mean square of $\mathbf{a}$ is $\sigma^2$. So both layers divide $\mathbf{g}\odot\mathbf{a}$ by $\sqrt{\sigma^2 + \epsilon}$. This is item 1, and also follows from eq-ln-as-rms.
2. The mean square of $\mathbf{a}$ splits as $\frac{1}{n}\sum a_i^2 = \sigma^2 + \mu^2$, since the cross term $\frac{2\mu}{n}\sum(a_i - \mu)$ is zero. Write $\mathbf{z} = (\mathbf{a} - \mu\mathbf{1})/\sigma$, so $\mathrm{LN}(\mathbf{a}) = \mathbf{g}\odot\mathbf{z}$. Then $\mathbf{a} = \sigma(\mathbf{z} + \rho\mathbf{1})$ and $\mathrm{RMS}(\mathbf{a}) = \sigma\sqrt{1 + \rho^2}$, so $\mathrm{RN}(\mathbf{a}) = \mathbf{g}\odot(\mathbf{z} + \rho\mathbf{1})/\sqrt{1 + \rho^2}$. This is item 2.
3. With $\mathbf{g} = \mathbf{1}$, put $c = 1/\sqrt{1 + \rho^2}$. Then $\mathrm{RN} - \mathrm{LN} = (c - 1)\mathbf{z} + c\rho\mathbf{1}$. The vector $\mathbf{z}$ has mean 0 and mean square 1, so it is orthogonal to $\mathbf{1}$, and the mean square of the difference is $(c - 1)^2 + c^2\rho^2 = c^2(1 + \rho^2) - 2c + 1 = 2 - 2c$. With $\tan\theta = \lvert\rho\rvert$, $c = \cos\theta$ and $\sqrt{2 - 2\cos\theta} = 2\sin(\theta/2) \le \theta \le \tan\theta$. For small $\rho$, $2 - 2c = \rho^2 - \frac{3}{4}\rho^4 + \dots$, so the gap is about $\lvert\rho\rvert$. As $\lvert\rho\rvert$ grows, $c$ goes to 0.
4. The mean of $\mathbf{z}$ is 0, so the mean of $\mathrm{RN}(\mathbf{a})$ is $c\rho = \rho/\sqrt{1 + \rho^2}$.

### Result and check

$$ \mathrm{RN}(\mathbf{a}) = \cos\theta\;\mathrm{LN}(\mathbf{a}) + \sin\theta\;\mathbf{g}, \qquad \tan\theta = \mu/\sigma . $$

So RMSNorm is a mix of the layer normalization output and a constant vector, and the weight of the constant grows with the ratio of the mean to the spread. When the summed inputs are close to mean zero, as Zhang and Sennrich expect, the two layers give almost the same output.
The check `zero-mean.check.py` checks item 1 for $n \in \{2, 5, 64\}$ and three values of $\epsilon$. It checks items 2 to 4 for six values of $\mu$ from $-5$ to $10$, the bound by $\lvert\rho\rvert$, the small-$\rho$ limit and the limit $\sqrt{2}$. It also checks the worked example, where $\rho^2 = 1/3$, the gap is $\sqrt{2 - \sqrt{3}} \approx 0.518$ and the output mean is $0.5$.
