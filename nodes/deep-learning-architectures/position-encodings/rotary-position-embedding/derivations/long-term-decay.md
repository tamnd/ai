---
id: long-term-decay
derives: [c-decay-bound, c-decay-not-monotone, eq-decay]
kind: bound
assumes: ["d is even", "theta_j = b^(-2j/d) with b > 1", "h_j is the product of pair j of q and the conjugate of pair j of k, with h_{d/2} = 0"]
uses: [dot-product@m2]
status: full
check: long-term-decay.check.py
lean: null
sources:
  - {work: su-2024-roformer, locator: "Section 3.4.3"}
  - {work: su-2024-roformer, locator: "Section 3.3"}
  - {work: chen-2023-position-interpolation, locator: "Section 2.2"}
---

### Statement

Let $r = m - n$ be the relative position, let $\hat{q}_j = q_{2j} + i q_{2j+1}$ and $\hat{k}_j = k_{2j} + i k_{2j+1}$, and let $h_j = \hat{q}_j \overline{\hat{k}_j}$ for $j = 0, \dots, d/2 - 1$, with $h_{d/2} = 0$.
Let $S_0 = 0$ and $S_j = \sum_{i=0}^{j-1} e^{i r \theta_i}$ for $j = 1, \dots, d/2$.
Then the score of eq-score satisfies

$$ \big| (\mathbf{R}_m \mathbf{q})^\top (\mathbf{R}_n \mathbf{k}) \big| \le \Big| \sum_{j=0}^{d/2-1} h_j e^{i r \theta_j} \Big| \le \Big( \max_{0 \le j < d/2} |h_{j+1} - h_j| \Big) \sum_{j=1}^{d/2} |S_j|. $$

The first factor depends on the content only and the second on $r$ only.
For $d = 128$ and $b = 10000$ the average $\frac{1}{d/2}\sum_j |S_j|$ falls from $32.5$ at $r = 0$ to about $4.5$ at $r = 1000$, but it does not fall at every step.

### Assumptions

1. $d$ is even and the pairs are adjacent, as in eq-rope-matrix. A different pairing permutes the terms and changes nothing below.
2. $h_{d/2} = 0$ is a convention that closes the sum; Su et al. state it in Section 3.4.3.
3. The angles are those of eq-theta. Steps 1 to 3 hold for any real angles; only step 4 uses the values.

### Steps

1. By part 5 of the derivation relative-position, the score equals $\mathrm{Re} \sum_j h_j e^{i r \theta_j}$. For a complex number $z$, $|\mathrm{Re}\, z| \le |z|$, which gives the first inequality.
2. Since $S_{j+1} - S_j = e^{i r \theta_j}$, the sum is $\sum_{j=0}^{d/2-1} h_j (S_{j+1} - S_j)$. Summation by parts (the Abel transformation) with $S_0 = 0$ gives $h_{d/2-1} S_{d/2} - \sum_{j=0}^{d/2-2} S_{j+1}(h_{j+1} - h_j)$. With $h_{d/2} = 0$ the first term is $-S_{d/2}(h_{d/2} - h_{d/2-1})$, so the sum is $-\sum_{j=0}^{d/2-1} S_{j+1}(h_{j+1} - h_j)$. This is Su et al.'s Equation 36 with indices from 0.
3. The triangle inequality bounds the modulus of that sum by $\sum_{j} |S_{j+1}|\,|h_{j+1} - h_j|$, and each $|h_{j+1} - h_j|$ is at most their maximum. This gives the second inequality, Su et al.'s Equation 37.
4. At $r = 0$ every term of $S_j$ is $1$, so $|S_j| = j$ and the average over $j = 1, \dots, 64$ is $65/2 = 32.5$ for $d = 128$. For $r \ne 0$ the terms $e^{i r \theta_j}$ turn at different speeds and partly cancel. A direct sum in double precision gives the averages in the table below. The value at $r = 2000$ is larger than at $r = 1000$, and so the average is not monotone in $r$.

| $r$ | 0 | 1 | 10 | 100 | 1000 | 2000 |
|---|---|---|---|---|---|---|
| average of $\lvert S_j \rvert$ | 32.5 | 31.538 | 17.954 | 10.227 | 4.471 | 5.047 |

### Result and check

The score is at most a content factor times a factor that depends on the relative position only.
Su et al. (2021, Section 3.4.3) plot that second factor in their Figure 2 and read it as decaying when the relative distance grows.
Step 4 shows that the decay holds on average over long ranges, with bumps, and the bound is a bound on the size of the score, not the score itself.
Chen et al. (2023, Section 2.2) call the bound possibly too loose: at large $r$ it does not stop the score of a trained model from growing.

The check `long-term-decay.check.py` verifies the Abel identity of step 2 to $10^{-10}$, the two inequalities on random $\mathbf{q}$ and $\mathbf{k}$ for many $r$, and the numbers of the table to three decimals.
The test `test_decay_bound` of the NumPy reference checks the second inequality as well.
