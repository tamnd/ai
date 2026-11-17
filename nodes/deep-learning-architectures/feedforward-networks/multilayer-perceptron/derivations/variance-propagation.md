---
id: variance-propagation
derives: [c-variance-recursion, c-he-condition, c-half-variance, c-glorot-variance]
kind: closed-form
assumes: ["at initialization the weights of a layer are independent, zero-mean and identically distributed, and independent of the layer input", "the biases are zero", "the entries of the layer input are identically distributed"]
uses: []
status: full
check: variance-propagation.check.py
lean: null
sources:
  - {work: he-2015-delving-deep-rectifiers, locator: "Section 2.2"}
  - {work: glorot-2010-understanding-difficulty-training, locator: "Section 4.2.1"}
---

### Statement

Let $z_l = W_l h_l$ be the pre-activations of layer $l$, with $n_l$ inputs per unit and $h_l = \operatorname{relu}(z_{l-1})$.

1. $\operatorname{Var}[z_l] = n_l \operatorname{Var}[w_l]\, \mathbb{E}[h_l^2]$.
2. If $z_{l-1}$ is symmetric about zero, $\mathbb{E}[h_l^2] = \tfrac{1}{2} \operatorname{Var}[z_{l-1}]$, so $\operatorname{Var}[z_l] = \tfrac{1}{2} n_l \operatorname{Var}[w_l] \operatorname{Var}[z_{l-1}]$, and the variance stays constant from layer to layer exactly when $\operatorname{Var}[w_l] = 2 / n_l$.
3. With $\operatorname{Var}[w_l] = 1 / n_l$ instead, each ReLU layer halves the variance, and after $m$ layers it is $2^{-m}$ of where it started.
4. The uniform distribution on $[-a, a]$ with $a = \sqrt{6 / (n_l + n_{l+1})}$ has variance $2 / (n_l + n_{l+1})$, the compromise of Glorot and Bengio between the forward condition $n_l \operatorname{Var}[w] = 1$ and the backward condition $n_{l+1} \operatorname{Var}[w] = 1$.

### Assumptions

1. The entries of $W_l$ are independent with mean zero, and independent of $h_l$. Used in step 1.
2. The biases are zero and the weights of the layer below are symmetric about zero, so $z_{l-1}$ is symmetric about zero. Used in step 2.

### Steps

1. A unit computes $z = \sum_{j=1}^{n_l} w_j h_j$. The terms have mean $\mathbb{E}[w_j] \mathbb{E}[h_j] = 0$ and are uncorrelated, since the $w_j$ are independent and zero-mean. So $\operatorname{Var}[z] = \sum_j \operatorname{Var}[w_j h_j] = n_l \mathbb{E}[w^2] \mathbb{E}[h^2] = n_l \operatorname{Var}[w] \mathbb{E}[h^2]$. This is statement 1. Note $\mathbb{E}[h^2]$, not $\operatorname{Var}[h]$: a ReLU output has a positive mean.
2. For $z$ symmetric about zero, $\mathbb{E}[\operatorname{relu}(z)^2] = \mathbb{E}[z^2 \mathbf{1}_{z > 0}] = \tfrac{1}{2} \mathbb{E}[z^2] = \tfrac{1}{2} \operatorname{Var}[z]$. Putting this into statement 1 gives the recursion, and its factor $\tfrac{1}{2} n_l \operatorname{Var}[w_l]$ is 1 exactly when $\operatorname{Var}[w_l] = 2 / n_l$. This is statement 2.
3. With $\operatorname{Var}[w_l] = 1 / n_l$ the factor is $\tfrac{1}{2}$, and $m$ layers multiply the variance by $2^{-m}$. This is statement 3.
4. The uniform distribution on $[-a, a]$ has variance $a^2 / 3$, and $a^2 / 3 = 2 / (n_l + n_{l+1})$ for the given $a$. That value is the harmonic mean of $1 / n_l$ and $1 / n_{l+1}$, the variances asked by the two conditions, and it equals both when $n_l = n_{l+1}$. This is statement 4.

### Result and check

$$ \operatorname{Var}[z_l] = \tfrac{1}{2}\, n_l \operatorname{Var}[w_l]\, \operatorname{Var}[z_{l-1}] , \qquad \operatorname{Var}[w_l] = \frac{2}{n_l} $$

The check `variance-propagation.check.py` draws wide ReLU networks with the NumPy reference and measures the variance of the pre-activations layer by layer, under eq-he and with half its variance, and samples the uniform distribution of eq-glorot.
