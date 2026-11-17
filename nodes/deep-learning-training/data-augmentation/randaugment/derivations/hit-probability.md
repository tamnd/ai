---
id: hit-probability
derives: [c-hit-probability, c-distinct-ops, eq-hit]
kind: identity
assumes: ["the N operations are drawn independently and uniformly from K, with replacement", "K >= 1 and N >= 0"]
uses: [discrete-uniform-distribution@m2, probability-of-the-complement@m2, independence-of-random-variables@m2, linearity-of-expectation@m2]
status: full
check: hit-probability.check.py
lean: null
sources:
  - {work: cubuk-2019-randaugment, locator: "Section 3"}
  - {work: cubuk-2019-randaugment, locator: "Figure 2"}
  - {work: muller-2021-trivialaugment, locator: "Algorithm 1"}
---

### Statement

Let RandAugment draw $N$ operations $o_1, \dots, o_N$ from a list of $K$ operations, as in eq-randaugment. For a fixed operation $k$:

1. the probability that $k$ is drawn at least once is $p_{\mathrm{hit}}(K, N) = 1 - (1 - 1/K)^N$, eq-hit;
2. the mean number of times $k$ is drawn is $N/K$, and the mean number of distinct operations in the list is $K\,p_{\mathrm{hit}}(K, N)$;
3. the probability that the $N$ draws are all different is $K!/((K - N)!\,K^N)$ for $N \le K$.

For $K = 14$: $p_{\mathrm{hit}} = 1/14 \approx 0.0714$ at $N = 1$, $27/196 \approx 0.1378$ at $N = 2$ and $547/2744 \approx 0.1993$ at $N = 3$; at $N = 2$ the list holds $27/14 \approx 1.93$ distinct operations on average and repeats an operation with probability $1/14$. TrivialAugment draws one operation per image, so its $p_{\mathrm{hit}}$ is $1/K$.

### Assumptions

1. Each $o_i$ is uniform on the $K$ operations, $\Pr(o_i = k) = 1/K$, as in Section 3 of Cubuk et al. (2019). Used in steps 1 and 2.
2. The draws are independent and with replacement, as `np.random.choice(transforms, N)` does in their Figure 2, whose default is `replace=True`. Used in steps 1 and 3.

### Steps

1. The event "$k$ is never drawn" is the intersection of the $N$ events $o_i \ne k$, each of probability $1 - 1/K$ (assumption 1). By independence (assumption 2) it has probability $(1 - 1/K)^N$, and its complement is item 1.
2. Write the count of $k$ as $\sum_i \mathbf 1[o_i = k]$; by linearity of expectation its mean is $N/K$. The number of distinct operations is $\sum_k \mathbf 1[k \text{ drawn}]$, and linearity with step 1 gives $K\,p_{\mathrm{hit}}$.
3. The ordered lists with no repeat number $K(K - 1)\cdots(K - N + 1) = K!/(K - N)!$ out of $K^N$ equally likely lists.
4. For $K = 14$, $N = 2$: $1 - (13/14)^2 = 1 - 169/196 = 27/196$, and $14 \cdot 27/196 = 27/14$. For $N = 3$: $1 - 2197/2744 = 547/2744$.

### Result and check

$$ p_{\mathrm{hit}}(K, N) = 1 - \Big(1 - \frac{1}{K}\Big)^N , \qquad \mathbb E[\#\text{distinct}] = K\,p_{\mathrm{hit}}(K, N) . $$

The check `hit-probability.check.py` verifies items 1 to 3 in exact rational arithmetic by enumerating all $K^N$ lists for $K \le 6$ and $N \le 4$, and for $K = 14$, $N \le 3$.
It then draws $2 \times 10^5$ lists of $N = 1, 2, 3$ operations from $K = 14$ with `numpy.random.Generator.choice` with replacement and checks that the share containing operation 0 is within four standard errors of $1/14$, $27/196$ and $547/2744$.
All the numbers in this derivation are this page's arithmetic.
