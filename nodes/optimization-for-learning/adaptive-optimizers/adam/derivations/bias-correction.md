---
id: bias-correction
derives: [c-bias-expectation, c-constant-gradient, c-uncorrected-first-step, c-eps-hat, eq-unrolled, eq-efficient]
kind: identity
assumes: ["m_0 = v_0 = 0", "the gradients g_1, ..., g_t have the same mean and the same second moment", "0 <= beta1, beta2 < 1"]
uses: [expected-value@m2, linearity-of-expectation@m2]
status: full
check: bias-correction.check.py
lean: null
sources:
  - {work: kingma-2014-adam, locator: "Section 3"}
  - {work: kingma-2014-adam, locator: "Section 2"}
---

### Statement

Let $g_1, \dots, g_t$ be the gradients, $m_0 = v_0 = 0$, and $m_t$, $v_t$ the moving averages of eq-moments. Then

1. $m_t = (1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i} g_i$ and $v_t = (1 - \beta_2) \sum_{i=1}^t \beta_2^{t-i} g_i^2$, and the weights of each sum add up to $1 - \beta^t$ (eq-unrolled);
2. if $\mathbb{E}[g_i] = \mu$ and $\mathbb{E}[g_i^2] = s$ for all $i$, then $\mathbb{E}[m_t] = (1 - \beta_1^t)\,\mu$ and $\mathbb{E}[v_t] = (1 - \beta_2^t)\,s$, so $\mathbb{E}[\hat m_t] = \mu$ and $\mathbb{E}[\hat v_t] = s$;
3. for a constant gradient $g \ne 0$, $\hat m_t = g$ and $\hat v_t = g^2$ at every step, so with $\epsilon = 0$ the step is exactly $\eta \operatorname{sign}(g)$;
4. without the correction, the first step is $\eta (1 - \beta_1)/\sqrt{1 - \beta_2}$ in size when $\epsilon = 0$, about $3.16\,\eta$ for the default $\beta_1 = 0.9$, $\beta_2 = 0.999$;
5. the update of eq-update equals $\eta_t m_t / (\sqrt{v_t} + \hat\epsilon)$ with $\eta_t = \eta \sqrt{1 - \beta_2^t}/(1 - \beta_1^t)$ and $\hat\epsilon = \epsilon \sqrt{1 - \beta_2^t}$ (eq-efficient). A fixed $\hat\epsilon$ is therefore a step-dependent $\epsilon = \hat\epsilon / \sqrt{1 - \beta_2^t}$, which is $1/\sqrt{1 - \beta_2} \approx 31.6$ times $\hat\epsilon$ at $t = 1$ for $\beta_2 = 0.999$.

### Assumptions

1. $m_0 = v_0 = 0$, as in Algorithm 1 of Kingma and Ba. Used in steps 1 and 4.
2. The first and second moments of the gradient do not change with $i$ (stationarity). Used in step 2. Kingma and Ba note that a non-stationary second moment adds an error term $\zeta$ that stays small when $\beta_2$ gives little weight to the distant past.
3. $0 \le \beta_1, \beta_2 < 1$, so $1 - \beta^t > 0$. Used in steps 2, 3 and 5.

### Steps

1. Unroll $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$ down to $m_0 = 0$: each $g_i$ enters once, multiplied by $1 - \beta_1$ and then by $\beta_1$ at each of the $t - i$ later steps. The same holds for $v_t$ with $g_i^2$. The weights sum to $(1 - \beta)\sum_{j=0}^{t-1} \beta^j = 1 - \beta^t$, a geometric series. This is item 1.
2. Take expectations in item 1. Expectation is linear, and by assumption 2 every $\mathbb{E}[g_i]$ is $\mu$ and every $\mathbb{E}[g_i^2]$ is $s$, so they come out of the sums, leaving the sum of the weights: $\mathbb{E}[m_t] = (1 - \beta_1^t)\mu$ and $\mathbb{E}[v_t] = (1 - \beta_2^t) s$. Dividing by $1 - \beta^t$ (assumption 3) gives item 2. This is eqs. 1 to 4 of Section 3 of Kingma and Ba.
3. For $g_i = g$, item 1 gives $m_t = (1 - \beta_1^t) g$ and $v_t = (1 - \beta_2^t) g^2$ with no expectation needed, so $\hat m_t = g$, $\hat v_t = g^2$ and $\hat m_t / \sqrt{\hat v_t} = g/|g|$. This is item 3.
4. At $t = 1$, $m_1 = (1 - \beta_1) g_1$ and $v_1 = (1 - \beta_2) g_1^2$, so $m_1/\sqrt{v_1} = \operatorname{sign}(g_1)(1 - \beta_1)/\sqrt{1 - \beta_2}$. For the defaults this is $0.1/\sqrt{0.001} \approx 3.16$. With the correction the ratio is $\pm 1$, by step 3 with $t = 1$. This is item 4.
5. Write $\hat m_t/(\sqrt{\hat v_t} + \epsilon) = \frac{m_t}{1 - \beta_1^t} \cdot \frac{\sqrt{1 - \beta_2^t}}{\sqrt{v_t} + \epsilon\sqrt{1 - \beta_2^t}}$, by multiplying the numerator and the denominator of the second factor by $\sqrt{1 - \beta_2^t}$. Collecting the factors gives $\eta_t$ and $\hat\epsilon$. This is the efficient form of the last paragraph of Section 2 of Kingma and Ba. Solving $\hat\epsilon = \epsilon\sqrt{1 - \beta_2^t}$ for $\epsilon$ gives item 5.

### Result and check

$$ \mathbb{E}[\hat m_t] = \mathbb{E}[g], \qquad \mathbb{E}[\hat v_t] = \mathbb{E}[g^2], \qquad \eta\,\frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon} = \eta\,\frac{\sqrt{1 - \beta_2^t}}{1 - \beta_1^t}\,\frac{m_t}{\sqrt{v_t} + \epsilon\sqrt{1 - \beta_2^t}} . $$

The check `bias-correction.check.py` compares the recursions with the unrolled sums, and it estimates $\mathbb{E}[m_t]$ and $\mathbb{E}[v_t]$ over 200,000 runs of Gaussian gradients with a fixed seed.
It checks that a constant gradient gives a ratio of exactly $\pm 1$, that the uncorrected first ratio is $3.16$ for the defaults, and that the two forms of the update agree to a relative error under $10^{-12}$.
