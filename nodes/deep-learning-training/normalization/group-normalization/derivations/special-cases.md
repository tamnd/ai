---
id: special-cases
derives: [c-g1-is-ln, c-gc-is-in, c-bn-batch1-is-in]
kind: identity
assumes: ["the input has the layout (N, C, *S) and the channels are split into G contiguous groups of C/G channels", "all four methods use the biased variance and the same epsilon inside the square root"]
uses: [sample-mean@m2, variance@m2]
status: full
check: special-cases.check.py
lean: null
sources:
  - {work: wu-2018-group-normalization, locator: "p. 3"}
  - {work: wu-2018-group-normalization, locator: "p. 4"}
---

### Statement

Let $x \in \mathbb{R}^{N \times C \times S_1 \times \dots \times S_d}$ and write an index as $i = (i_N, i_C, i_S)$. For a set $\mathcal{S}_i$ of indices, let $\hat{x}_i = (x_i - \mu_i)/\sqrt{\sigma_i^2 + \epsilon}$, where $\mu_i$ and $\sigma_i^2$ are the mean and the biased variance of $x$ over $\mathcal{S}_i$. Let $G$ divide $C$. Then

1. group normalization with $G = 1$ equals layer normalization over all of $(C, S_1, \dots, S_d)$ for each example;
2. group normalization with $G = C$ equals instance normalization, which normalizes each channel of each example over its positions;
3. batch normalization in training mode on a batch of one example equals instance normalization.

### Assumptions

1. The channels of group $g$ are $gC/G, \dots, (g+1)C/G - 1$, so the group of channel $c$ is $\lfloor c/(C/G) \rfloor$. Used in steps 1 and 2.
2. All methods use the same $\epsilon$ and the biased variance, so equal sets give equal outputs. Used in every step.
3. The affine map is left out. It is per channel in all four methods, so it does not change the comparison. Used in every step.

### Steps

1. The group normalization set is $\mathcal{S}_i = \{k : k_N = i_N,\ \lfloor k_C/(C/G) \rfloor = \lfloor i_C/(C/G) \rfloor\}$. With $G = 1$, $C/G = C$ and $\lfloor k_C/C \rfloor = 0$ for every channel. So the second condition always holds and $\mathcal{S}_i = \{k : k_N = i_N\}$, the layer normalization set. Equal sets give equal means and variances, so the outputs are equal. This is item 1.
2. With $G = C$, $C/G = 1$ and $\lfloor k_C/1 \rfloor = k_C$. So $\mathcal{S}_i = \{k : k_N = i_N,\ k_C = i_C\}$, the instance normalization set. This is item 2.
3. The batch normalization set is $\mathcal{S}_i = \{k : k_C = i_C\}$. With $N = 1$ every index has $k_N = 0 = i_N$, so the condition $k_N = i_N$ adds nothing and the set is $\{k : k_N = i_N,\ k_C = i_C\}$, the instance normalization set. This is item 3. At test time batch normalization uses stored population statistics instead, so the equality is for training mode only.

### Result and check

$$ \mathrm{GN}_{G=1} = \mathrm{LN}_{(C, S)}, \qquad \mathrm{GN}_{G=C} = \mathrm{IN}, \qquad \mathrm{BN}^{\text{train}}_{N=1} = \mathrm{IN} . $$

The check `special-cases.check.py` draws random tensors of shape $(N, C, H, W)$ for several $N$, $C$, $H$ and $W$. It compares group normalization with $G = 1$ to a direct layer normalization over $(C, H, W)$, group normalization with $G = C$ to a direct instance normalization over $(H, W)$, and batch normalization of one example to instance normalization, all to $10^{-12}$. It also builds every set $\mathcal{S}_i$ by a loop over indices and checks that the sets themselves are equal.
