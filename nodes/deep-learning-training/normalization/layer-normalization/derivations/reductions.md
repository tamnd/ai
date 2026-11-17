---
id: reductions
derives: [c-gn-g1, c-batch-independent]
kind: identity
assumes: ["the statistics are over all the features of one example", "for c-gn-g1, the gain and bias of layer normalization are constant over the spatial positions of each channel"]
uses: [sample-mean@m2, variance@m2]
status: full
check: reductions.check.py
lean: null
sources:
  - {work: wu-2018-group-normalization, locator: "Section 3"}
  - {work: wu-2018-group-normalization, locator: "Section 4.1"}
  - {work: ba-2016-layer-normalization, locator: "Section 3"}
---

### Statement

Let a batch be $X \in \mathbb{R}^{N \times C \times H \times W}$ (or $N \times F$ with $F$ features), and let layer normalization take the mean and the biased variance of each example over all of its $C H W$ entries.

1. Group normalization with $G = 1$ group is layer normalization over $(C, H, W)$, when the per-element gain and bias of layer normalization are set to the per-channel gain and bias of group normalization, repeated over the positions. With $G = 1$ the two methods differ only in the shape of their affine parameters: group normalization has one gain and one bias per channel, and layer normalization over $(C, H, W)$ has one per entry.
2. The output for example $i$ depends only on example $i$: $\partial\,\mathrm{LN}(X)_i / \partial X_j = 0$ for $j \ne i$. So the output does not change when the other examples, or the batch size, change; and since the function has no state, it is the same at training and at test time.

### Assumptions

1. The statistics are over all the features of one example and over nothing else. Used in steps 1 and 2.
2. For item 1, the gain and bias of layer normalization are $\mathbf{g}_{c,h,w} = \gamma_c$ and $\mathbf{b}_{c,h,w} = \beta_c$, and both methods use the same $\epsilon$. Used in step 1.

### Steps

1. Group normalization splits the $C$ channels into $G$ groups of $C/G$ channels. Its mean and variance for example $i$ and group $k$ are over the $(C/G) H W$ entries of that group, and it applies $\gamma_c, \beta_c$ per channel. With $G = 1$ the one group is all of $(C, H, W)$. So its mean and variance are those of layer normalization by assumption 1, and its normalized entries are the same. Applying $\gamma_c$ and $\beta_c$ at every position of channel $c$ is the per-element affine of assumption 2. This is item 1.
2. $\mu_i$ and $\sigma_i$ are functions of $X_i$ alone by assumption 1, and so is the output $\mathbf{g}\odot(X_i - \mu_i)/\sigma_i + \mathbf{b}$. A function of $X_i$ alone has a zero derivative in $X_j$ for $j \ne i$. Nothing is stored between calls, so the function computed at test time is the one computed in training. This is item 2. Batch normalization takes its statistics over the batch dimension, so neither part holds for it in training.

### Result and check

$$ \mathrm{GN}_{G=1}(X;\boldsymbol{\gamma},\boldsymbol{\beta}) = \mathrm{LN}_{(C,H,W)}(X;\ \boldsymbol{\gamma}\otimes\mathbf{1}_{HW},\ \boldsymbol{\beta}\otimes\mathbf{1}_{HW}), \qquad \frac{\partial\,\mathrm{LN}(X)_i}{\partial X_j} = 0 \ (j \ne i) . $$

The check `reductions.check.py` writes group normalization by hand from its definition. It compares it with $G = 1$ with layer normalization on random inputs with a fixed seed, and shows that for $G = 2$ they differ.
It computes the full Jacobian of layer normalization over a batch of 4 by finite differences and checks that its blocks off the diagonal are zero. It also checks that each example's output is unchanged when the batch is cut to that one example or filled with other examples.
