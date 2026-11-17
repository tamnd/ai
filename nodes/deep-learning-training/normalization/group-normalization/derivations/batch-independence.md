---
id: batch-independence
derives: [c-batch-independence, c-bn-depends]
kind: identity
assumes: ["the statistics of group normalization are computed from one example only", "batch normalization is in training mode and uses the statistics of the current batch"]
uses: [sample-mean@m2, variance@m2]
status: full
check: batch-independence.check.py
lean: null
sources:
  - {work: wu-2018-group-normalization, locator: "the abstract"}
  - {work: wu-2018-group-normalization, locator: "p. 4"}
---

### Statement

Let $x$ be a batch of shape $(N, C, *S)$, $x_n$ its example $n$, and $\mathrm{GN}$ group normalization with any $G$ that divides $C$. Then

1. $\mathrm{GN}(x)_n = \mathrm{GN}(x_n)$ for every $n$: the output for an example does not depend on the batch size or on the other examples;
2. batch normalization in training mode has no such property: there are batches $x$ and $x'$ with $x_1 = x'_1$ but $\mathrm{BN}(x)_1 \ne \mathrm{BN}(x')_1$.

### Assumptions

1. Group normalization computes its mean and variance over a set that has $k_N = i_N$ for every member. Used in step 1.
2. Batch normalization in training mode computes its mean and variance over $\{k : k_C = i_C\}$, which holds every example. Used in step 2.

### Steps

1. Every member $k$ of the group normalization set of $i$ has $k_N = i_N$. So $\mu_i$ and $\sigma_i$ are functions of $x_{i_N}$ alone. The output $\hat{x}_i = (x_i - \mu_i)/\sqrt{\sigma_i^2+\epsilon}$, and the per-channel scale and shift, are then functions of $x_{i_N}$ alone. Running the layer on $x_n$ by itself gives the same sets restricted to that example, so the same numbers. This is item 1. It holds at training time and at test time, so group normalization needs no running averages.
2. Take one channel, one position and $\epsilon = 0$. In the batch $x = (0, 2)$ the mean is $1$ and the variance is $1$, so batch normalization maps the first example to $(0 - 1)/1 = -1$. In the batch $x' = (0, 2, 4)$ the first example is the same, but the mean is $2$ and the variance is $8/3$, so the first output is $-2/\sqrt{8/3} = -\sqrt{3/2} \ne -1$. This is item 2.

### Result and check

$$ \mathrm{GN}(x)_n = \mathrm{GN}(x_n), \qquad \mathrm{BN}(x)_n \ \text{depends on}\ x_m,\ m \ne n . $$

The check `batch-independence.check.py` normalizes random batches with group normalization for several $G$, then normalizes each example alone and after the other examples are replaced by random data with another mean and scale, and checks that the outputs agree to $10^{-12}$. It checks the counterexample of step 2 and that the batch normalization output of one example moves when the other examples change.
