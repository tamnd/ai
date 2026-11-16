---
id: balance-loss-minimum
derives: [c-balance-uniform, c-balance-coupled-minimum, c-balance-maximum, c-balance-below-alpha, c-balance-logit-gradient, c-balance-worked, eq-switch-balance]
kind: bound
assumes: ["N ≥ 2 experts and a batch of T ≥ 1 tokens", "each token's router probabilities p(x) form a distribution over the N experts", "f_i is the fraction of tokens whose argmax is i, with ties broken so that each token counts once", "f is treated as a constant when differentiating, as in Switch Transformers"]
uses: [softmax@m2, cauchy-schwarz-inequality@m2]
status: full
check: balance-loss-minimum.check.py
lean: null
sources:
  - {work: fedus-2022-switch-transformers, locator: "p. 7"}
  - {work: fedus-2022-switch-transformers, locator: "p. 8"}
---

### Statement

Let $f_i = \frac1T\sum_x \mathbb 1\{\arg\max p(x) = i\}$ and $P_i = \frac1T\sum_x p_i(x)$, and let $\mathcal L = \alpha N \sum_{i=1}^N f_i P_i$ be the Switch balance loss.
Then

1. under uniform routing, $f_i = P_i = 1/N$ for every $i$, the loss is exactly $\alpha$, whatever $N$ is;
2. if the probabilities match the dispatch fractions, $P = f$, as they do when every $p(x)$ is one-hot, then $\mathcal L = \alpha N\sum_i f_i^2 \ge \alpha$, with equality exactly when $f$ is uniform;
3. always $\mathcal L \le \alpha N$, with equality when every token sends probability 1 to the same expert;
4. without the coupling of item 2, $\mathcal L$ can fall below $\alpha$, but never below $\alpha \sum_i f_i^2 \ge \alpha/N$; with $N = 2$, three quarters of the tokens at $p = (\tfrac12 + \varepsilon, \tfrac12 - \varepsilon)$ and a quarter at $p = (0, 1)$ give $\mathcal L \to 0.875\,\alpha$ as $\varepsilon \to 0$;
5. with $f$ held constant and $p(x) = \operatorname{softmax}(h(x))$, the gradient in the logits of one token is $\partial\mathcal L/\partial h_j(x) = \frac{\alpha N}{T}\, p_j(x)\big(f_j - \sum_i f_i p_i(x)\big)$, so a step of gradient descent lowers the logit of every expert whose share $f_j$ is above the $p(x)$-weighted average share, and raises the others.

### Assumptions

1. $N \ge 2$ and $T \ge 1$, and each $p(x)$ is a probability vector.
2. $f$ counts the top-1 choice of each token once, before any token is dropped for capacity; the loss uses this $f$, not the tokens an expert actually processed.
3. In item 5 the indicator in $f$ has zero derivative almost everywhere, so treating $f$ as a constant is exact away from ties; Fedus et al. note that $P$ is differentiable and $f$ is not.

### Steps

1. Uniform routing: $\sum_i f_i P_i = N \cdot \frac1N \cdot \frac1N = \frac1N$, so $\mathcal L = \alpha N \cdot \frac1N = \alpha$. This is the reason for the factor $N$: Fedus et al. multiply by the number of experts so that the loss stays constant when $N$ changes.
2. With $P = f$, $\sum_i f_i P_i = \sum_i f_i^2$. By the Cauchy-Schwarz inequality, $1 = (\sum_i 1 \cdot f_i)^2 \le N\sum_i f_i^2$, so $\sum_i f_i^2 \ge 1/N$ and $\mathcal L \ge \alpha$; equality in Cauchy-Schwarz holds exactly when $f$ is proportional to the all-ones vector, that is $f_i = 1/N$.
3. Since $f$ is a probability vector and $P_i \le 1$, $\sum_i f_i P_i \le \max_i P_i \le 1$, so $\mathcal L \le \alpha N$. If every token puts probability 1 on expert $j$, then $f = P = e_j$ and $\sum_i f_iP_i = 1$.
4. A token whose argmax is $i$ has $p_i(x) \ge 1/N$, because the largest of $N$ numbers summing to 1 is at least $1/N$. Summing over those $Tf_i$ tokens and dropping the others, $P_i \ge f_i/N$. Hence $\sum_i f_iP_i \ge \frac1N\sum_i f_i^2$ and $\mathcal L \ge \alpha\sum_i f_i^2 \ge \alpha/N$ by step 2. For the example, $f = (\tfrac34, \tfrac14)$ and $P = (\tfrac34(\tfrac12+\varepsilon),\ \tfrac34(\tfrac12-\varepsilon) + \tfrac14)$, so $\sum_i f_iP_i = \tfrac34 \cdot \tfrac38 + \tfrac14 \cdot \tfrac58 + O(\varepsilon) = \tfrac{9}{32} + \tfrac{5}{32} + O(\varepsilon) = \tfrac{7}{16} + O(\varepsilon)$, and $\mathcal L = 2\alpha \cdot \tfrac{7}{16} = 0.875\,\alpha$ in the limit. Uniform routing is the minimum only under a coupling such as item 2, which is what training with a confident router approaches; the loss is a proxy, not an exact measure of imbalance.
5. $\partial\mathcal L/\partial p_i(x) = \alpha N f_i/T$. The softmax Jacobian is $\partial p_i/\partial h_j = p_i(\mathbb 1\{i=j\} - p_j)$, so $\partial\mathcal L/\partial h_j(x) = \frac{\alpha N}{T}\sum_i f_i\,p_i(x)(\mathbb 1\{i = j\} - p_j(x)) = \frac{\alpha N}{T}\,p_j(x)\big(f_j - \sum_i f_ip_i(x)\big)$.

### Result and check

$$ \mathcal L = \alpha N \sum_{i=1}^N f_i P_i,\qquad \mathcal L_{\text{uniform}} = \alpha,\qquad P = f \Rightarrow \alpha \le \mathcal L \le \alpha N,\qquad \alpha\sum_i f_i^2 \le \mathcal L \le \alpha N \text{ in general}. $$

The check `balance-loss-minimum.check.py` confirms item 1 for $N$ from 2 to 64, and items 2 and 3 on 2,000 random one-hot batches, where it also confirms that the minimum over one-hot batches with $T = N$ is $\alpha$, reached only at a perfectly even assignment.
It confirms the bounds of item 4 on 2,000 random softmax batches and the $0.875\,\alpha$ example, and it compares item 5 with central finite differences of $\mathcal L$ in the logits, holding $f$ fixed.
On the worked example of the page, one-hot routing of eight tokens over four experts with loads $(4, 2, 1, 1)$, $f = (0.5, 0.25, 0.125, 0.125)$ and $\mathcal L = 4\alpha(0.25 + 0.0625 + 0.015625 + 0.015625) = 1.375\,\alpha$.
