---
id: step-bound
derives: [c-step-bound-tight, c-step-bound-limits, c-first-step, c-single-spike, eq-ratio-bound]
kind: bound
assumes: ["epsilon = 0", "m_0 = v_0 = 0", "v_t > 0", "0 <= beta1 < 1 and 0 < beta2 < 1"]
uses: [cauchy-schwarz-inequality@m2, cauchy-schwarz-inequality-for-expectations@m2]
status: full
check: step-bound.check.py
lean: null
sources:
  - {work: kingma-2014-adam, locator: "Section 2.1"}
---

### Statement

Let $\epsilon = 0$, write $\gamma = \beta_1^2/\beta_2$, and let $r_t = \hat m_t / \sqrt{\hat v_t}$, so the step of eq-update is $\eta\,|r_t|$. For every sequence of gradients with $v_t > 0$,

$$ |r_t| \le B_t = \frac{1 - \beta_1}{\sqrt{1 - \beta_2}} \cdot \frac{\sqrt{1 - \beta_2^t}}{1 - \beta_1^t} \cdot \sqrt{\sum_{k=0}^{t-1} \gamma^k} , \qquad \text{(eq-ratio-bound)} $$

with equality when $g_{t-k} = c\,(\beta_1/\beta_2)^k$ for some $c \ne 0$. In particular:

1. $B_1 = 1$: the first step is exactly $\eta$ in size;
2. if $\gamma < 1$, $B_t$ tends to $(1 - \beta_1)/\sqrt{(1 - \beta_2)(1 - \gamma)}$, which is about 7.27 for $\beta_1 = 0.9$, $\beta_2 = 0.999$, about 2.35 for $\beta_2 = 0.99$, about 1.70 for $\beta_2 = 0.98$ and about 1.16 for $\beta_2 = 0.95$;
3. if $\gamma \ge 1$, $B_t$ grows without bound;
4. a gradient that is zero at every step but the last gives $|r_t| = \frac{1 - \beta_1}{\sqrt{1 - \beta_2}} \cdot \frac{\sqrt{1 - \beta_2^t}}{1 - \beta_1^t}$, which tends to $(1 - \beta_1)/\sqrt{1 - \beta_2}$;
5. for any distribution of $g$, $|\mathbb{E}[g]| \le \sqrt{\mathbb{E}[g^2]}$.

Section 2.1 of Kingma and Ba states the bounds $(1 - \beta_1)/\sqrt{1 - \beta_2}$ when $1 - \beta_1 > \sqrt{1 - \beta_2}$ and 1 otherwise. Items 2 and 4 show that the first is the value of the single spike, not the largest value: the defaults allow ratios up to about 7.27, and $\beta_2 = 0.99$, for which the paper states $|r_t| < 1$, allows about 2.35.

### Assumptions

1. $\epsilon = 0$. Used in every step; $\epsilon > 0$ only makes the ratio smaller.
2. $m_0 = v_0 = 0$, so the unrolled sums of derivations/bias-correction.md hold. Used in step 1.
3. $v_t > 0$: some gradient is non-zero. Used in step 2.
4. $0 \le \beta_1 < 1$ and $0 < \beta_2 < 1$. Used in steps 2 and 3.

### Steps

1. By the unrolled sums, $m_t = (1 - \beta_1)\sum_{k=0}^{t-1} \beta_1^k g_{t-k}$ and $v_t = (1 - \beta_2)\sum_{k=0}^{t-1} \beta_2^k g_{t-k}^2$, where $k$ is the age of a gradient.
2. Write $\beta_1^k g_{t-k} = a_k b_k$ with $a_k = (\beta_1/\sqrt{\beta_2})^k$ and $b_k = \beta_2^{k/2} g_{t-k}$. By the Cauchy–Schwarz inequality, $|\sum_k a_k b_k| \le \sqrt{\sum_k a_k^2}\sqrt{\sum_k b_k^2} = \sqrt{\sum_k \gamma^k}\,\sqrt{v_t/(1 - \beta_2)}$. So $|m_t|/\sqrt{v_t} \le \frac{1 - \beta_1}{\sqrt{1 - \beta_2}}\sqrt{\sum_k \gamma^k}$. Equality holds when $b_k$ is proportional to $a_k$, that is when $\beta_2^{k/2} g_{t-k} = c\,(\beta_1/\sqrt{\beta_2})^k$, or $g_{t-k} = c\,(\beta_1/\beta_2)^k$.
3. The bias correction multiplies $|m_t|/\sqrt{v_t}$ by $\sqrt{1 - \beta_2^t}/(1 - \beta_1^t)$, which gives eq-ratio-bound.
4. At $t = 1$ the sum is 1 and the correction is $\sqrt{1 - \beta_2}/(1 - \beta_1)$, so $B_1 = 1$. This is item 1.
5. If $\gamma < 1$, the geometric series tends to $1/(1 - \gamma)$ and the correction tends to 1, which gives the limit of item 2. The numbers follow from $\gamma = 0.81/\beta_2$. If $\gamma \ge 1$, the series diverges, which is item 3.
6. For the single spike only $k = 0$ is non-zero, so $m_t = (1 - \beta_1) g_t$ and $v_t = (1 - \beta_2) g_t^2$, and the correction gives item 4.
7. Item 5 is the Cauchy–Schwarz inequality for the expectation, $\mathbb{E}[g \cdot 1]^2 \le \mathbb{E}[g^2]\,\mathbb{E}[1]$.

The equality case needs gradients that grow by the factor $\beta_2/\beta_1 \approx 1.11$ per step for the defaults, for many steps. Such gradients are rare in practice, which is why Section 2.1 calls $|r_t| \approx 1$ the common case. The bound matters for the worst case, for example at a sudden rise of the loss.

### Result and check

$$ \sup |r_t| = \frac{1 - \beta_1}{\sqrt{(1 - \beta_2)(1 - \beta_1^2/\beta_2)}} \approx 7.27 \quad \text{for } \beta_1 = 0.9, \ \beta_2 = 0.999 . $$

The check `step-bound.check.py` draws 200 random sequences with magnitudes over several orders for each of seven pairs $(\beta_1, \beta_2)$ and five lengths, with a fixed seed.
None exceeds $B_t$, and the sequence $g_{t-k} = (\beta_1/\beta_2)^k$ reaches it to a relative error under $10^{-9}$.
It also checks the limits of item 2, the growth when $\gamma \ge 1$, the single spike, and item 5 on random discrete distributions.
