---
id: autoregressive-normalization
derives: [c-normalization]
kind: closed-form
assumes: ["x = (x_1, ..., x_D) takes values in a finite product of finite sets", "the ordering of the variables is fixed", "each conditional p(x_d | x_<d) is nonnegative and sums to 1 over x_d for every value of x_<d"]
uses: [conditional-probability@m2, joint-probability-distribution@m2, marginal-distribution@m2, ancestral-sampling@al]
status: full
check: autoregressive-normalization.check.py
lean: null
sources:
  - {work: germain-2015-made-masked-autoencoder, locator: "p. 2", note: "the product rule, eq. 4, and the negative log-likelihood of eq. 5"}
  - {work: van-den-oord-2016-wavenet-generative-model-raw-audio, locator: "p. 2", note: "the same factorization of a waveform, eq. 1"}
---

### Statement

Let $q_d(x_d \mid x_{<d})$, $d = 1, \dots, D$, be any functions with $q_d \ge 0$ and $\sum_{x_d} q_d(x_d \mid x_{<d}) = 1$ for every prefix $x_{<d}$. Then

$$ p(\mathbf{x}) = \prod_{d=1}^{D} q_d(x_d \mid x_{<d}) $$

is a probability distribution: it is nonnegative and $\sum_{\mathbf{x}} p(\mathbf{x}) = 1$. Its conditionals are the $q_d$, so $p(x_d \mid x_{<d}) = q_d(x_d \mid x_{<d})$ wherever $p(x_{<d}) > 0$. Ancestral sampling, which draws $x_1 \sim q_1$, then $x_2 \sim q_2(\cdot \mid x_1)$, and so on, returns $\mathbf{x}$ with probability exactly $p(\mathbf{x})$.

### Assumptions

1. The values are discrete, so the sums are finite. For densities, replace each sum by an integral; Tonelli's theorem allows the integrals to be taken one variable at a time. Used in step 1.
2. The ordering is fixed. A model whose ordering depends on $\mathbf{x}$ needs a separate argument. Used in step 1.

### Steps

1. Sum out the last variable first. For fixed $x_{<D}$, $\sum_{x_D} p(\mathbf{x}) = \prod_{d<D} q_d(x_d \mid x_{<d}) \sum_{x_D} q_D(x_D \mid x_{<D}) = \prod_{d<D} q_d(x_d \mid x_{<d})$.
2. By induction on $D$, summing out $x_D, x_{D-1}, \dots, x_1$ in turn leaves $1$. Each partial sum $p(x_{\le d}) = \prod_{e \le d} q_e(x_e \mid x_{<e})$ is the marginal of the first $d$ variables.
3. Divide two consecutive marginals: $p(x_d \mid x_{<d}) = p(x_{\le d}) / p(x_{<d}) = q_d(x_d \mid x_{<d})$.
4. Ancestral sampling draws $x_1$ with probability $q_1(x_1)$ and then, given the prefix drawn so far, $x_d$ with probability $q_d(x_d \mid x_{<d})$. The chance of the whole sequence is the product of these probabilities, which is $p(\mathbf{x})$.

### Result and check

$$ \sum_{\mathbf{x}} \prod_{d=1}^D q_d(x_d \mid x_{<d}) = 1, \qquad \Pr[\text{ancestral sample} = \mathbf{x}] = p(\mathbf{x}) . $$

There is no normalizing constant to compute, unlike an RBM or an energy-based model. In return every conditional must be normalized on its own, which is why MADE's outputs pass through a sigmoid and WaveNet's through a softmax.

The check `autoregressive-normalization.check.py` builds random conditional tables over four variables with three values each and sums the product over all 81 outcomes. It sums a random MADE over all binary vectors of length 6 and a random WaveNet stack over all 81 ternary sequences of length 4. It also compares the empirical frequencies of 60000 ancestral samples of the MADE with its probabilities.
