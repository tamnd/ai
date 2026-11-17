---
id: unbiased-estimator
derives: [c-unbiased, c-product-form, c-plug-in-biased, c-range-monotone, c-special-cases, eq-pass-at-k, eq-estimator, eq-product-form]
kind: identity
assumes: ["the n samples of a problem are drawn independently from the same model and prompt", "each sample passes the unit tests with the same probability p", "n >= k >= 1"]
uses: [binomial-distribution@m2, hypergeometric-distribution@m2, bias-of-an-estimator@m2, linearity-of-expectation@m2]
status: full
check: unbiased-estimator.check.py
lean: null
sources:
  - {work: chen-2021-evaluating-large-language-models-trained-on-code, locator: "p. 19"}
  - {work: chen-2021-evaluating-large-language-models-trained-on-code, locator: "p. 3"}
---

### Statement

Fix one problem. Draw $n$ samples, let $c$ of them pass the unit tests, and let $1 \le k \le n$.
Write $\binom{a}{b} = 0$ when $0 \le a < b$, and let

$$ \hat{\pi}_k = 1 - \binom{n-c}{k} \Big/ \binom{n}{k}, \qquad \tilde{\pi}_k = 1 - \Big(1 - \frac{c}{n}\Big)^k. $$

1. If $c \sim \mathrm{Binomial}(n, p)$, then $\mathbb{E}[\hat{\pi}_k] = 1 - (1 - p)^k$: $\hat{\pi}_k$ is an unbiased estimator of pass@k for the problem.
2. If $n - c \ge k$, then $\binom{n-c}{k} / \binom{n}{k} = \prod_{i=n-c+1}^{n} (1 - k/i)$; if $n - c < k$, then $\hat{\pi}_k = 1$.
3. If $k \ge 2$ and $0 < p < 1$, then $\mathbb{E}[\tilde{\pi}_k] < 1 - (1 - p)^k$; for $k = 1$, $\tilde{\pi}_1 = \hat{\pi}_1 = c/n$.
4. $0 \le \hat{\pi}_k \le 1$; $\hat{\pi}_k = 0$ exactly when $c = 0$; $\hat{\pi}_k$ does not decrease when $c$ or $k$ grows; $\hat{\pi}_n = 1$ when $c \ge 1$; and $\hat{\pi}_k = k/n$ when $c = 1$.

The mean over problems of eq-estimator is then an unbiased estimator of the mean of eq-pass-at-k, by linearity of expectation.

### Assumptions

1. The $n$ samples of the problem are independent draws from the same model with the same prompt and the same sampling settings. Used in step 1.
2. Each sample passes the unit tests with the same probability $p$, the pass@1 of the problem, and passing is a fixed function of the sample (the tests are deterministic). Used in step 1.
3. $1 \le k \le n$. Used in steps 2, 3 and 6, where $\binom{n}{k} > 0$.
4. $0 < p < 1$ for the strict inequality of part 3 only. Used in step 9.

### Steps

1. By assumptions 1 and 2, the number $c$ of passing samples is a sum of $n$ independent Bernoulli($p$) variables, so $\Pr[c = j] = \binom{n}{j} p^j (1 - p)^{n-j}$ for $j = 0, \dots, n$, the binomial distribution.
2. Given $c$, take a subset $S$ of $k$ of the $n$ samples uniformly at random, without replacement. There are $\binom{n}{k}$ subsets, all equally likely, and $\binom{n-c}{k}$ of them hold only failing samples, because they choose all $k$ members among the $n - c$ failing ones. So $\Pr[S \text{ has no passing sample} \mid c] = \binom{n-c}{k} / \binom{n}{k}$, the probability that a hypergeometric draw gets no success. When $n - c < k$ there is no such subset and the count is 0.
3. Unconditionally, the $k$ samples in $S$ are $k$ of the $n$ independent draws, chosen by an index set that does not look at the samples. So by assumptions 1 and 2 they are $k$ independent samples that each fail with probability $1 - p$, and $\Pr[S \text{ has no passing sample}] = (1 - p)^k$ by independence.
4. By the law of total probability over the values of $c$, $\Pr[S \text{ has no passing sample}] = \mathbb{E}_c\big[\Pr[S \text{ has no passing sample} \mid c]\big] = \mathbb{E}_c\big[\binom{n-c}{k} / \binom{n}{k}\big]$, using step 2 for the inner probability.
5. Steps 3 and 4 give $\mathbb{E}_c\big[\binom{n-c}{k} / \binom{n}{k}\big] = (1 - p)^k$. Subtracting from 1, by linearity of expectation, $\mathbb{E}[\hat{\pi}_k] = 1 - (1 - p)^k$. This is part 1. The same result follows by algebra, as Chen et al. (2021, Appendix A) show: $\binom{n}{j}\binom{n-j}{k} / \binom{n}{k} = \binom{n-k}{j}$, because both sides count the ways to pick $j$ items and then $k$ others from $n$, divided by the ways to pick the $k$ first; then $\sum_{j=0}^{n-k} \binom{n-k}{j} p^j (1-p)^{n-j} = (1-p)^k \sum_{j=0}^{n-k} \binom{n-k}{j} p^j (1-p)^{n-k-j} = (1-p)^k$ by the binomial theorem.
6. For part 2, write each coefficient as a falling product: $\binom{a}{k} = a(a-1)\cdots(a-k+1)/k!$. So, when $n - c \ge k$, $\binom{n-c}{k} / \binom{n}{k} = \prod_{t=0}^{k-1} \frac{n - c - t}{n - t}$. The same ratio is $\frac{(n-c)!\,(n-k)!}{n!\,(n-c-k)!} = \prod_{i=n-c+1}^{n} \frac{i - k}{i}$, because $n!/(n-c)! = \prod_{i=n-c+1}^{n} i$ and $(n-k)!/(n-c-k)! = \prod_{i=n-c+1}^{n} (i - k)$, both products of $c$ consecutive integers. Each factor $(i - k)/i$ is $1 - k/i$. This is eq-product-form. It has $c$ factors, each in $[0, 1)$, so no factor overflows; when $n - c < k$, $\binom{n-c}{k} = 0$ and $\hat{\pi}_k = 1$.
7. For part 4, the ratio of step 2 is a probability, so $0 \le \hat{\pi}_k \le 1$. With $c = 0$ the ratio is $\binom{n}{k}/\binom{n}{k} = 1$ and $\hat{\pi}_k = 0$; with $c \ge 1$ the product of step 6 has a factor $1 - k/n < 1$, and every factor is less than 1 and at least 0, so the ratio is below 1 and $\hat{\pi}_k > 0$. Going from $c$ to $c + 1$ multiplies the product by one more factor $1 - k/(n - c) \le 1$, or makes $\hat{\pi}_k = 1$, so $\hat{\pi}_k$ does not decrease in $c$. Going from $k$ to $k + 1 \le n$ multiplies the falling product of step 6 by $\frac{n - c - k}{n - k} \le 1$, so $\hat{\pi}_k$ does not decrease in $k$. With $k = n$ and $c \ge 1$, $n - c < k$, so $\hat{\pi}_n = 1$. With $c = 1$ the product has the one factor $1 - k/n$, so $\hat{\pi}_k = k/n$.
8. For $k = 1$, $\hat{\pi}_1 = 1 - (n - c)/n = c/n$ by step 6, and $\tilde{\pi}_1 = 1 - (1 - c/n) = c/n$. This is the second half of part 3.
9. For $k \ge 2$, the function $g(x) = 1 - (1 - x)^k$ has $g''(x) = -k(k-1)(1 - x)^{k-2} < 0$ for $0 \le x < 1$, so it is strictly concave on $[0, 1]$. By assumption 4, $\hat{p} = c/n$ takes at least two values with positive probability, so Jensen's inequality is strict: $\mathbb{E}[g(\hat{p})] < g(\mathbb{E}[\hat{p}]) = g(p) = 1 - (1 - p)^k$, because $\mathbb{E}[c/n] = p$ for the binomial distribution of step 1. This is the first half of part 3: $\tilde{\pi}_k$ is biased low.

### Result and check

For a problem whose $n$ samples are independent with pass probability $p$, the estimate $1 - \binom{n-c}{k}/\binom{n}{k}$ has mean $1 - (1 - p)^k$ for every $n \ge k$, it equals the product $1 - \prod_{i=n-c+1}^{n} (1 - k/i)$, and the plug-in $1 - (1 - c/n)^k$ has a smaller mean for $k \ge 2$.
The mean over problems inherits the unbiasedness by linearity; it does not need the problems to share $p$ or $n$.

The numeric check `unbiased-estimator.check.py` sums the exact binomial expectation with fractions for 1,000 random $(n, k, p)$ with $n \le 40$ and asserts equality with $(1 - p)^k$; it compares the product with the ratio of binomials for every $c$ and $k$ with $n \le 60$; it asserts the strict bias of the plug-in; and it runs a Monte Carlo average over 400,000 draws at $n = 200$, $k = 10$, $p = 0.03$.
The tests `test_unbiased`, `test_monte_carlo`, `test_product_form`, `test_plug_in_is_biased_low`, `test_special_cases` and `test_range_and_monotone` of `impl/` repeat the parts on the implementations.
