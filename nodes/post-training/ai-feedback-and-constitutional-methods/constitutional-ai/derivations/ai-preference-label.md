---
id: ai-preference-label
derives: [c-label-logistic, c-soft-gradient, c-soft-minimizer, c-hard-no-minimizer, c-clamp-bound, c-ensemble-average, c-swap-symmetric, c-swap-shrinks]
kind: identity
assumes: ["the label is the softmax of the feedback model's log-probabilities of the two options", "the preference model is trained by cross-entropy on the soft label p of the pair, through the logistic function of its score difference", "an additive bias b on the log-odds of the first position"]
uses: [logistic-regression@m2, cross-entropy@m2, binary-entropy-function@m2]
status: full
check: ai-preference-label.check.py
lean: null
sources:
  - {work: bai-2022-constitutional-ai, locator: "Sections 4.1 and 4.3"}
  - {work: lee-2023-rlaif-vs-rlhf, locator: "Sections 2.1, 2.1.1 and 2.2.1"}
---

### Statement

Let $\ell_A$ and $\ell_B$ be the feedback model's log-probabilities of the answers (A) and (B), $p = e^{\ell_A} / (e^{\ell_A} + e^{\ell_B})$ the label (eq-label), $\Delta = r_\phi(x, y_A) - r_\phi(x, y_B)$ the preference model's score difference, and

$$ L_p(\Delta) = -p \log \sigma(\Delta) - (1 - p) \log \sigma(-\Delta) $$

the loss of one pair in eq-soft-loss. Then

1. $p = \sigma(\ell_A - \ell_B)$, and $p$ is unchanged when the same constant is added to $\ell_A$ and $\ell_B$;
2. $L_p'(\Delta) = \sigma(\Delta) - p$ and $L_p''(\Delta) = \sigma(\Delta)\sigma(-\Delta) > 0$;
3. for $0 < p < 1$, $L_p$ has the unique minimizer $\Delta^* = \log\big(p / (1 - p)\big)$, and $L_p(\Delta^*) = H(p) = -p \log p - (1 - p) \log(1 - p)$; with $p$ from item 1, $\Delta^* = \ell_A - \ell_B$;
4. for $p = 1$, $L_1(\Delta) = -\log \sigma(\Delta)$ is strictly decreasing and has no minimizer;
5. if $p \in [a, 1 - a]$ with $0 < a < 1/2$, then $|\Delta^*| \le \log\big((1 - a)/a\big)$, which is $\log 1.5$ for $a = 0.4$ and $\log 4$ for $a = 0.2$;
6. if the label is $p_k$ with probability $w_k$, independently of $\Delta$, then $\sum_k w_k L_{p_k}(\Delta) = L_{\bar p}(\Delta)$ with $\bar p = \sum_k w_k p_k$, whose minimizer is $\log\big(\bar p / (1 - \bar p)\big)$;
7. with $f(b) = \tfrac12\big(\sigma(d + b) + \sigma(d - b)\big)$ (eq-swap), $f(-b) = f(b)$, and $f(b) = 1/2$ for every $b$ when $d = 0$;
8. for $d > 0$, $f$ is strictly decreasing in $|b|$, with $f(0) = \sigma(d)$ and $f(b) > 1/2$ for every $b$, and $f(b) \to 1/2$ as $|b| \to \infty$.

### Assumptions

1. The label is the two-way softmax of the feedback model's log-probabilities of the two options, as in Bai et al. (Section 4.1) and Lee et al. (Section 2.1). Used in items 1 and 3.
2. The preference model is fitted by the cross-entropy of the soft label against $\sigma(\Delta)$, the softmax of the two scores, as in Lee et al. (Section 2.2.1). Used in items 2 to 6.
3. For items 7 and 8, the feedback model's log-odds in favor of the response shown first are $d + b$, with $d$ its preference between the contents and $b$ a constant bias towards the first position; this model of position bias is an assumption of the derivation, not a finding of either paper. Lee et al. average the two orders (Section 2.1.1); averaging probabilities, as they describe, is what $f$ computes.

### Steps

1. Divide the numerator and denominator by $e^{\ell_A}$: $p = 1/(1 + e^{-(\ell_A - \ell_B)}) = \sigma(\ell_A - \ell_B)$. Adding $c$ to both log-probabilities leaves the difference, so $p$, unchanged. This is item 1.
2. With $\frac{d}{d\Delta}\log\sigma(\Delta) = \sigma(-\Delta)$ and $\frac{d}{d\Delta}\log\sigma(-\Delta) = -\sigma(\Delta)$, $L_p' = -p\,\sigma(-\Delta) + (1 - p)\sigma(\Delta) = \sigma(\Delta) - p$, using $\sigma(\Delta) + \sigma(-\Delta) = 1$. Differentiating again, $\sigma' = \sigma(\Delta)\sigma(-\Delta) > 0$. This is item 2.
3. By item 2, $L_p$ is strictly convex, and $L_p' = 0$ exactly when $\sigma(\Delta) = p$, that is $\Delta = \log(p/(1 - p))$, which exists for $0 < p < 1$. There $\sigma(\Delta^*) = p$ and $\sigma(-\Delta^*) = 1 - p$, so $L_p(\Delta^*) = H(p)$. With item 1, $\log(p/(1-p)) = \ell_A - \ell_B$. This is item 3.
4. For $p = 1$, $L_1' = \sigma(\Delta) - 1 < 0$ for every $\Delta$, so $L_1$ falls strictly and tends to 0 only as $\Delta \to \infty$. This is item 4.
5. $\log(p/(1 - p))$ is increasing in $p$, so on $[a, 1 - a]$ it lies between $\log(a/(1 - a))$ and $\log((1 - a)/a)$, which are opposite numbers. With $a = 0.4$ the bound is $\log(0.6/0.4) = \log 1.5$, and with $a = 0.2$ it is $\log(0.8/0.2) = \log 4$. This is item 5.
6. $L_p(\Delta) = -\log\sigma(-\Delta) - p\big(\log\sigma(\Delta) - \log\sigma(-\Delta)\big)$ is affine in $p$, so its average over $p_k$ with weights $w_k$ is $L_{\bar p}$. Item 3 gives the minimizer. This is item 6.
7. $f(-b) = \tfrac12(\sigma(d - b) + \sigma(d + b)) = f(b)$. At $d = 0$, $\sigma(b) + \sigma(-b) = 1$, so $f(b) = 1/2$. This is item 7.
8. $f'(b) = \tfrac12\big(\sigma'(d + b) - \sigma'(d - b)\big)$, where $\sigma'(u) = \sigma(u)\sigma(-u)$ is even and strictly decreasing in $|u|$. For $d > 0$ and $b > 0$, $|d + b| > |d - b|$, so $f'(b) < 0$; with item 7, $f$ falls in $|b|$ from $f(0) = \sigma(d)$. As $b \to \infty$, $\sigma(d + b) \to 1$ and $\sigma(d - b) \to 0$, so $f \to 1/2$; since $f$ is strictly decreasing towards this limit, $f(b) > 1/2$ for every $b$. This is item 8.

### Result and check

$$ \Delta^* = \log\frac{p}{1 - p} = \ell_A - \ell_B , \qquad |\Delta^*| \le \log\frac{1 - a}{a} , \qquad \tfrac12 < \tfrac12\big(\sigma(d + b) + \sigma(d - b)\big) \le \sigma(d) \quad (d > 0) . $$

The check `ai-preference-label.check.py` verifies the items on random inputs with a fixed seed, the derivatives by central differences, the minimizers by a grid search, and the numbers of the worked example.
