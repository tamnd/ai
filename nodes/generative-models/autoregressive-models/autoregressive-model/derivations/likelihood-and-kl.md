---
id: likelihood-and-kl
derives: [c-kl]
kind: closed-form
assumes: ["the data distribution q and the model p live on a finite set", "p(x) > 0 wherever q(x) > 0"]
uses: [kullback-leibler-divergence@m2, shannon-entropy@m2, cross-entropy@m2, gibbs-inequality@m2, maximum-likelihood-estimation@m2, expected-value@m2]
status: full
check: likelihood-and-kl.check.py
lean: null
sources:
  - {work: germain-2015-made-masked-autoencoder, locator: "pp. 2, 4", note: "the negative log-likelihood of eq. 5 as the training loss and the ensemble that averages probabilities over orderings"}
---

### Statement

For a model $p$ and a data distribution $q$,

$$ \mathbb{E}_{\mathbf{x} \sim q}[-\log p(\mathbf{x})] = H(q) + \mathrm{KL}(q \,\|\, p) , $$

so minimizing the expected negative log-likelihood over $p$ minimizes $\mathrm{KL}(q \,\|\, p) \ge 0$, which is zero only at $p = q$. For an ensemble of $R$ models $p_1, \dots, p_R$, such as one MADE under $R$ mask sets,

$$ \log \frac{1}{R} \sum_{r=1}^{R} p_r(\mathbf{x}) \ \ge\ \frac{1}{R} \sum_{r=1}^{R} \log p_r(\mathbf{x}) , $$

so the ensemble's negative log-likelihood is no larger than the average of its members'. The ensemble's probabilities still sum to 1.

### Assumptions

1. Finite support, so every expectation is a finite sum. Used in step 1.
2. $p > 0$ on the support of $q$; otherwise both sides are $+\infty$. Used in step 1.

### Steps

1. Write $-\log p = -\log q + \log(q / p)$ and take the expectation under $q$. The first term gives the entropy $H(q) = -\sum q \log q$. The second gives $\mathrm{KL}(q \,\|\, p) = \sum q \log(q / p)$.
2. $H(q)$ does not depend on $p$, so the two objectives differ by a constant. By Gibbs' inequality, $\mathrm{KL} \ge 0$ with equality only at $p = q$.
3. The average negative log-likelihood of a training set is the same expectation under the empirical distribution, so maximum likelihood fits the empirical distribution in KL. Divided by $D \log 2$, it is the number of bits per dimension that an arithmetic coder driven by $p$ would spend.
4. The logarithm is concave, so by Jensen's inequality the log of a mean is at least the mean of the logs, which gives the ensemble bound. Equality holds only when all the $p_r(\mathbf{x})$ agree.
5. The ensemble is a mixture with weights $1/R$, so it sums to 1. It is no longer autoregressive in a single ordering, but sampling is still exact: pick $r$ uniformly at random, then sample from $p_r$.

### Result and check

$$ \mathbb{E}_q[-\log p] = H(q) + \mathrm{KL}(q \,\|\, p) \ \ge\ H(q), \qquad \log \overline{p_r} \ \ge\ \overline{\log p_r} . $$

The check `likelihood-and-kl.check.py` verifies the identity and the bound for random pairs of distributions on 16 points. It verifies the ensemble bound and the normalization for a MADE under five mask sets on all binary vectors of length 5. It also confirms that a MADE trained to fit a known distribution has a cross-entropy equal to the entropy plus its KL.
