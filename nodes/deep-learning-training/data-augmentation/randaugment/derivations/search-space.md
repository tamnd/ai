---
id: search-space
derives: [c-aa-space-count, c-ra-grid-count, eq-aa-space, eq-ra-grid]
kind: count
assumes: ["each AutoAugment operation slot picks one of 16 operations, 10 magnitudes and 11 probabilities, independently of the other slots", "an AutoAugment policy is 5 sub-policies of 2 slots each", "a RandAugment policy is fixed by the pair (N, M)"]
uses: [discrete-uniform-distribution@m2]
status: full
check: search-space.check.py
lean: null
sources:
  - {work: cubuk-2018-autoaugment, locator: "Section 3"}
  - {work: cubuk-2019-randaugment, locator: "Section 3"}
  - {work: cubuk-2019-randaugment, locator: "Appendix A.2.3"}
---

### Statement

1. An AutoAugment operation slot has $16 \cdot 10 \cdot 11 = 1760$ settings. A sub-policy of 2 slots has $1760^2 = 3{,}097{,}600$, and a policy of 5 sub-policies has $1760^{10} \approx 2.85 \times 10^{32}$, eq-aa-space, which Cubuk et al. round to $2.9 \times 10^{32}$.
2. A RandAugment grid search over a set $\mathcal N$ of values of $N$ and a set $\mathcal M$ of values of $M$ tries $|\mathcal N| \cdot |\mathcal M|$ settings, eq-ra-grid. The ImageNet ResNet-50 grid of Cubuk et al. (2019), $N \in \{1, 2, 3\}$ and $M \in \{5, 7, 9, 11, 13, 15\}$, has 18. All of $N \in \{1, 2, 3\}$ with $M \in \{0, \dots, 10\}$ is 33, of order $10^2$.
3. These are counts of different objects. The AutoAugment count is the number of policies the controller can name; the RandAugment count is the number of settings a person tries. Inside one RandAugment setting the sampler still draws one of $K^N$ ordered operation lists per image, $14^2 = 196$ for $K = 14$ and $N = 2$.

### Assumptions

1. Each slot of an AutoAugment sub-policy picks an operation from the 16 of Section 3 of Cubuk et al. (2018), a magnitude from 10 discrete values and a probability from 11 discrete values, and no choice restricts another. Used in step 1.
2. A policy is an ordered list of 5 sub-policies, each an ordered pair of slots, and two lists that differ in any slot count as different policies, even when they act the same. Used in steps 1 and 2.
3. A RandAugment policy is the pair $(N, M)$; the operations are not searched, since each is drawn with probability $1/K$. Used in step 3.

### Steps

1. By the rule of product over the three independent choices of assumption 1, one slot has $16 \cdot 10 \cdot 11 = 1760$ settings.
2. A policy has $2 \cdot 5 = 10$ slots (assumption 2), chosen independently, so the rule of product gives $1760^{10}$. Since $\log_{10} 1760 = 3.2455\ldots$, this is $10^{32.455} \approx 2.85 \times 10^{32}$. One sub-policy alone has $1760^2 = 3{,}097{,}600$.
3. A grid search tries each pair $(N, M) \in \mathcal N \times \mathcal M$ once, which is $|\mathcal N| \cdot |\mathcal M|$ pairs (assumption 3): $3 \cdot 6 = 18$ for the ResNet-50 grid and $3 \cdot 11 = 33$ for the full range of the 0 to 10 scale.
4. Given $(N, M)$, the sampler of eq-randaugment draws an ordered list of $N$ operations from $K$ with replacement, and there are $K^N$ such lists.

### Result and check

$$ |\mathcal S_{\mathrm{AA}}| = (16 \cdot 10 \cdot 11)^{2 \cdot 5} = 1760^{10} \approx 2.85 \times 10^{32}, \qquad |\mathcal S_{\mathrm{RA}}| = |\mathcal N| \cdot |\mathcal M| \le 33 . $$

The check `search-space.check.py` computes $1760^{10}$ in exact integer arithmetic, checks that it rounds to $2.9 \times 10^{32}$, enumerates the slot settings with `itertools.product` for a small case to confirm the rule of product, counts the RandAugment grids, and enumerates the $14^2$ ordered lists.
The numbers 2.85, 3,097,600, 18 and 33 are this page's arithmetic; $2.9 \times 10^{32}$ is printed by Cubuk et al. (2018).
