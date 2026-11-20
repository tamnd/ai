---
id: reward-model-arithmetic
derives: [c-rm-bt, c-rm-grad, c-rm-shift, c-rm-ranked, c-rm-noise, c-rm-margin, c-rm-calib, c-rm-ens, c-rm-bonkl, c-rm-robot, c-rm-we]
kind: proof
assumes: ["comparisons follow the Bradley-Terry model with a real-valued reward", "with probability epsilon a label is a uniformly random answer", "the reward of a sample from the base policy has a continuous distribution", "the return of a cleaning policy is the sum of the rewards of the rooms it cleans"]
uses: []
status: full
check: reward-model-arithmetic.check.py
lean: null
sources:
  - {work: christiano-2017-deep-rl-human-preferences, locator: "p. 5"}
  - {work: ouyang-2022-training-language-models-follow-instructions, locator: "pp. 8 and 42"}
  - {work: touvron-2023-llama-2, locator: "p. 11"}
  - {work: gao-2023-scaling-laws-reward-model-overoptimization, locator: "p. 4"}
  - {work: coste-2024-reward-model-ensembles, locator: "p. 4"}
  - {work: skalse-2022-defining-characterizing-reward-hacking, locator: "pp. 2 to 3"}
---

### Statement

Write $\sigma(t) = 1/(1 + e^{-t})$, and for a comparison of a preferred response $y_w$ and a rejected one $y_l$ to the same prompt $x$ write $r_w = r(x, y_w)$, $r_l = r(x, y_l)$ and $\Delta = r_w - r_l$.

1. (Bradley-Terry.) If $P(y_w \succ y_l \mid x) = e^{r_w} / (e^{r_w} + e^{r_l})$, then $P = \sigma(\Delta)$ and the negative log-likelihood of the comparison is $\ell(\Delta) = \log(1 + e^{-\Delta}) = -\log \sigma(\Delta)$.
2. (Its gradient.) $\partial \ell / \partial r_w = -\sigma(-\Delta)$ and $\partial \ell / \partial r_l = \sigma(-\Delta)$. They sum to zero, and $\sigma(-\Delta)$ is the probability the model gives to the opposite order.
3. (Shift invariance.) Replacing $r(x, y)$ by $r(x, y) + c(x)$ leaves every $\Delta$, every probability and the loss unchanged. Replacing it by $a\,r(x, y)$ with $a \ne 1$ changes $\sigma(\Delta)$ whenever $\Delta \ne 0$.
4. (Ranked responses.) For $K$ responses ranked $y_1 \succ \dots \succ y_K$, the loss $\binom{K}{2}^{-1} \sum_{i < j} \ell(r_i - r_j)$ needs the $K$ scores $r_1, \dots, r_K$ once, so $K$ forward passes; treating the pairs as separate examples needs $2\binom{K}{2} = K(K-1)$. For $K = 9$ that is 9 against 72, and 64 prompts give $64 \times 36 = 2{,}304$ comparisons.
5. (Label noise.) If with probability $\epsilon$ the label is a uniformly random answer, $P = \epsilon/2 + (1 - \epsilon)\sigma(\Delta) \in (\epsilon/2, 1 - \epsilon/2)$. The loss $-\log P$ is below $-\log(\epsilon/2)$, which is $\log 20 \approx 3.00$ for $\epsilon = 0.1$, and $\partial(-\log P)/\partial \Delta \to 0$ as $\Delta \to -\infty$. At $\Delta = -3$ the plain loss is $3.0486$ with gradient magnitude $0.9526$, the noisy loss $2.3786$ with gradient magnitude $0.4387$.
6. (Margin.) $\ell_m(\Delta) = -\log \sigma(\Delta - m)$ equals $\log 2$ at $\Delta = m$ and is decreasing in $\Delta$. At $\Delta = 1$, $\ell_1 = \log 2 = 0.6931$ and $\ell_0 = 0.3133$.
7. (Accuracy ceiling.) If labels follow the Bradley-Terry model with reward $r^*$, a pair with $|\Delta^*| = d$ is labelled in the order of $r^*$ with probability $\sigma(d)$, which is $0.7311$, $0.8808$ and $0.9526$ for $d = 1, 2, 3$, and no rule for ordering the pair is right with higher probability.
8. (Ensembles.) For rewards $R_1, \dots, R_k$ with mean $\bar R$ and population variance $V = k^{-1}\sum_i (R_i - \bar R)^2$, $\min_i R_i \le \bar R$ and $\bar R - \lambda V \le \bar R$ for $\lambda \ge 0$. For $(1.0, 1.2, 3.0)$, $\bar R = 1.7333$, $\min = 1.0$, $V = 0.8089$ and $\bar R - 0.5 V = 1.3289$.
9. (Best-of-$n$.) If the reward of a base-policy sample has a continuous distribution, the policy that draws $n$ samples and returns the one of highest reward has $\mathrm{KL}(\pi_{\text{bon}} \,\|\, \pi) = \log n - (n-1)/n$: $0.6363$, $1.8351$ and $3.1745$ nats for $n = 4, 16, 64$. For a base policy on finitely many outputs with distinct rewards the KL is at most this value.
10. (Cleaning robot.) Let a policy be a subset $S$ of three rooms and its return $J(S) = \sum_{i \in S} r_i$. Call a true reward $r$ and a proxy $\tilde r$ hackable when some $S, S'$ have $\tilde J(S) > \tilde J(S')$ and $J(S) < J(S')$. For $r = (1,1,1)$, $\tilde r = (1,0,0)$ is hackable and $\tilde r = (1,1,0)$ is not; for $\tilde r = (1,1,1)$, $r = (1, 1.5, 2)$ is not hackable and $r = (1, 1.5, 3)$ is, with witness $S = \{1, 2\}$, $S' = \{3\}$: proxy returns 2 against 1, true returns 2.5 against 3.
11. (Worked example.) For rewards $(2.0, 0.5, 0.0)$ of three responses ranked in that order, the pair losses are $\ell(1.5) = 0.201413$, $\ell(2.0) = 0.126928$ and $\ell(0.5) = 0.474077$, the ranked loss is $0.267473$, and its gradient with respect to the three rewards is $(-0.100543, -0.065038, 0.165581)$.

### Assumptions

1. Items 1 to 4, 6, 7 and 11 assume the Bradley-Terry model. Item 5 mixes it with a uniformly random answer, as Christiano et al. do with $\epsilon = 0.1$.
2. Item 9 needs ties between samples to have probability zero. Language-model outputs are discrete and can repeat, and then the formula is an upper bound rather than an equality.
3. Item 10 restricts the policies to the eight deterministic subsets; Skalse et al. show that over all stochastic policies a non-trivial unhackable pair does not exist.

### Steps

1. $e^{r_w}/(e^{r_w} + e^{r_l}) = 1/(1 + e^{-(r_w - r_l)}) = \sigma(\Delta)$, so $-\log P = \log(1 + e^{-\Delta})$.
2. $\ell'(\Delta) = -e^{-\Delta}/(1 + e^{-\Delta}) = -\sigma(-\Delta)$, and $\partial \Delta / \partial r_w = 1$, $\partial \Delta / \partial r_l = -1$.
3. $(r_w + c(x)) - (r_l + c(x)) = \Delta$, since both responses share the prompt. Scaling gives $a\Delta$, and $\sigma(a\Delta) \ne \sigma(\Delta)$ when $a \ne 1$ and $\Delta \ne 0$ because $\sigma$ is strictly increasing.
4. Each pair $(i, j)$ uses $r_i$ and $r_j$, which are among the $K$ scores. $\binom{9}{2} = 36$, $9 \cdot 8 = 72$ and $64 \cdot 36 = 2304$.
5. $\sigma(\Delta) \in (0, 1)$ gives the bounds on $P$, and $-\log P < -\log(\epsilon/2)$. The derivative is $-(1-\epsilon)\sigma(\Delta)\sigma(-\Delta)/P$, whose numerator tends to 0 and denominator to $\epsilon/2$ as $\Delta \to -\infty$. The values at $\Delta = -3$ follow from $\sigma(3) = 0.952574$.
6. $\sigma(0) = 1/2$, and $\sigma$ is increasing. $-\log \sigma(1) = \log(1 + e^{-1}) = 0.313262$.
7. The label agrees with the order of $r^*$ with probability $\sigma(|\Delta^*|) \ge 1/2$, and any rule that picks one order is right with probability $\sigma(|\Delta^*|)$ or $1 - \sigma(|\Delta^*|)$. $\sigma(1) = 0.731059$, $\sigma(2) = 0.880797$, $\sigma(3) = 0.952574$.
8. The minimum of numbers is at most their mean, and $\lambda V \ge 0$. For $(1.0, 1.2, 3.0)$ the deviations from $5.2/3$ square to $0.537778$, $0.284444$ and $1.604444$, whose mean is $0.808889$.
9. Let $F$ be the distribution function of the reward. The best of $n$ has density $n F^{n-1}$ times the base density in reward, so $\mathrm{KL} = \mathbb E[\log(n F^{n-1})]$ under the best-of-$n$ law. With $U = F(R)$ uniform, $\mathbb E[\log U^{n-1}]$ under the density $n u^{n-1}$ is $(n-1)\int_0^1 n u^{n-1}\log u\,du = -(n-1)/n$. In the discrete case the best-of-$n$ probability of the $i$-th smallest reward is $F_i^n - F_{i-1}^n$, and the check confirms it is below the bound.
10. Enumerate the $8 \times 8$ pairs of subsets for each case; the witness is computed directly.
11. $\sigma(-1.5) = 0.182426$, $\sigma(-2) = 0.119203$ and $\sigma(-0.5) = 0.377541$. Response 1 takes $-(0.182426 + 0.119203)/3$, response 2 takes $(0.182426 - 0.377541)/3$ and response 3 takes $(0.119203 + 0.377541)/3$.

### Result and check

$$ P(y_w \succ y_l \mid x) = \sigma(r_w - r_l), \qquad \mathrm{KL}(\pi_{\text{bon}} \,\|\, \pi) = \log n - \frac{n-1}{n}, \qquad P_\epsilon = \frac{\epsilon}{2} + (1-\epsilon)\,\sigma(\Delta) . $$

The check `reward-model-arithmetic.check.py` verifies the Bradley-Terry identity and the gradients against finite differences, and shift invariance on random rewards. It recounts the forward passes and comparisons, and evaluates the noisy and margin losses and their limits. It simulates labels from a Bradley-Terry model to confirm the accuracy ceiling, and recomputes the ensemble values. It checks the best-of-$n$ KL by Monte Carlo on a continuous reward and exactly on discrete ones, enumerates the cleaning-robot policies, and recomputes the worked example.
