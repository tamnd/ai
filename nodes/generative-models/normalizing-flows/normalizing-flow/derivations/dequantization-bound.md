---
id: dequantization-bound
derives: [c-dequant]
kind: closed-form
assumes: ["the data are integer vectors k in {0, ..., L - 1}^D", "q is any density on [0, 1)^D"]
uses: [probability-density-function@m2, linearity-of-expectation@m2]
status: full
check: dequantization-bound.check.py
lean: null
sources:
  - {work: kingma-2018-glow, locator: "p. 3", note: "eq. (2): the objective on data with added uniform noise, its constant c = -M log a, and both objectives as an expected compression cost"}
  - {work: dinh-2014-nice, locator: "p. 6", note: "uniform noise of 1/256 added to the data, rescaled to [0, 1]"}
  - {work: papamakarios-2021-normalizing-flows-probabilistic-modeling, locator: "p. 43", note: "dequantizing discrete data by adding continuous noise, fixed or learned"}
---

### Statement

Dequantize an integer vector $\mathbf{k} \in \{0, \dots, L - 1\}^D$ as $\mathbf{y} = (\mathbf{k} + \mathbf{u}) / L$ with $\mathbf{u} \sim U[0, 1)^D$.
For any density $q$ on $[0, 1)^D$, define the discrete model $P(\mathbf{k}) = \int_{\text{bin}(\mathbf{k})} q(\mathbf{y})\, d\mathbf{y}$, which sums to 1 over the $L^D$ bins.
Then

$$ \mathbb{E}_{\mathbf{u}}\big[\log q\big((\mathbf{k} + \mathbf{u}) / L\big)\big] - D \log L \le \log P(\mathbf{k}), $$

so the continuous log-likelihood of dequantized data, shifted by $-D \log L$, is a lower bound on a discrete log-likelihood, and $(\text{NLL} + D \log L) / (D \log 2)$ is an upper bound on the bits per dimension of a discrete model.
A density that is uniform on $[0, 1)^D$ scores 8 bits per dimension on 8-bit data.

### Assumptions

1. $q$ is a density on $[0, 1)^D$, so the bins partition its support. Used in step 1.
2. $\log$ is concave. Used in step 2.

### Steps

1. The bin of $\mathbf{k}$ is $\prod_i [k_i / L, (k_i + 1) / L)$, of volume $L^{-D}$. With $\mathbf{y} = (\mathbf{k} + \mathbf{u}) / L$, $P(\mathbf{k}) = L^{-D} \int_{[0,1)^D} q\big((\mathbf{k} + \mathbf{u}) / L\big)\, d\mathbf{u} = L^{-D}\, \mathbb{E}_{\mathbf{u}}\big[q\big((\mathbf{k} + \mathbf{u}) / L\big)\big]$. The bins partition $[0, 1)^D$, so $\sum_{\mathbf{k}} P(\mathbf{k}) = 1$.
2. By Jensen's inequality, $\log \mathbb{E}_{\mathbf{u}}[q] \ge \mathbb{E}_{\mathbf{u}}[\log q]$, so $\log P(\mathbf{k}) \ge \mathbb{E}_{\mathbf{u}}[\log q((\mathbf{k} + \mathbf{u}) / L)] - D \log L$, with equality when $q$ is constant on the bin.
3. Dividing the negative of either side by $D \log 2$ converts nats per example into bits per dimension. For $q = 1$ on $[0, 1)^D$, the continuous term is 0 and the bound is $D \log L / (D \log 2) = \log_2 L$, which is 8 for $L = 256$.

### Result and check

$$ -\log P(\mathbf{k}) \le D \log L - \mathbb{E}_{\mathbf{u}} \log q\big((\mathbf{k} + \mathbf{u}) / L\big), \qquad \text{bits/dim} = \frac{\text{NLL} + D \log L}{D \log 2} . $$

The check `dequantization-bound.check.py` takes non-uniform densities on $[0, 1)$ and $[0, 1)^2$ with closed-form bin masses, estimates the left side by Monte Carlo on dequantized samples, and confirms the bound and its equality for a density that is constant on the bins; it also evaluates `bits_per_dim` of the NumPy reference for the uniform density.
