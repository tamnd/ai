---
id: cd-k-limit
derives: [c-cd-limit]
kind: limit
assumes: ["finite weights and biases, so every conditional probability lies strictly between 0 and 1", "the expectation is over the Gibbs draws, with the hidden statistics replaced by their conditional means"]
uses: [markov-chain@m2, stationary-distribution@m2, detailed-balance@m2, expected-value@m2]
status: full
check: cd-k-limit.check.py
lean: null
sources:
  - {work: hinton-2002-training-products-experts-contrastive-divergence, locator: "p. 4", note: "the one-step distribution is one step closer to equilibrium, and for chains whose transitions all have non-zero probability the contrastive divergence is zero only for a perfect model"}
  - {work: tieleman-2008-training-rbms-approximations-likelihood, locator: "p. 2", note: "CD-n for greater n is closer to the likelihood gradient, and CD-1 differs from it when mixing is slow"}
---

### Statement

Let $T$ be the transition matrix of one block Gibbs step $\mathbf{v} \to \mathbf{h} \to \mathbf{v}'$ of an RBM, $T_{\mathbf{v}\mathbf{v}'} = \sum_\mathbf{h} p(\mathbf{h} \mid \mathbf{v}) p(\mathbf{v}' \mid \mathbf{h})$, let $\hat q$ be the empirical distribution of the data, and let $s(\mathbf{v})$ be the statistics $\big(\mathbf{v}^\top \sigma(\mathbf{c} + \mathbf{v}W), \mathbf{v}, \sigma(\mathbf{c} + \mathbf{v}W)\big)$. The expected CD-$k$ update is

$$ g_k = \mathbb{E}_{\hat q}[s] - \mathbb{E}_{\hat q T^k}[s] , $$

and $g_k \to g_\infty = \mathbb{E}_{\hat q}[s] - \mathbb{E}_{p}[s]$, the exact gradient of the mean log-likelihood, as $k \to \infty$, geometrically fast. For finite $k$, $g_k \ne g_\infty$ in general.

### Assumptions

1. Finite parameters, so $0 < \sigma(\cdot) < 1$ and every entry of $T$ is positive. Used in step 3.
2. The update averages over the Gibbs draws; a single chain gives an unbiased estimate of $g_k$, not of $g_\infty$. Used in step 4.

### Steps

1. The chains of CD-$k$ start at the data, distribution $\hat q$, and after $k$ steps have distribution $\hat q T^k$. The negative statistics are averaged over it, which gives $g_k$.
2. $T$ leaves $p(\mathbf{v})$ invariant: $\sum_\mathbf{v} p(\mathbf{v}) T_{\mathbf{v}\mathbf{v}'} = \sum_\mathbf{h} p(\mathbf{h}) p(\mathbf{v}' \mid \mathbf{h}) = p(\mathbf{v}')$, since $\sum_\mathbf{v} p(\mathbf{v}) p(\mathbf{h} \mid \mathbf{v}) = p(\mathbf{h})$. The chain also satisfies detailed balance, $p(\mathbf{v}) T_{\mathbf{v}\mathbf{v}'} = \sum_\mathbf{h} p(\mathbf{v}, \mathbf{h}) p(\mathbf{v}', \mathbf{h}) / p(\mathbf{h})$, which is symmetric in $\mathbf{v}$ and $\mathbf{v}'$.
3. All entries of $T$ are positive, so the chain is irreducible and aperiodic on the finite set $\{0,1\}^D$, and by the Perron-Frobenius theorem $\hat q T^k \to p$ for every starting distribution, with error bounded by $C\rho^k$ for the second-largest eigenvalue modulus $\rho < 1$ of $T$.
4. The statistics are bounded functions on a finite set, so $\mathbb{E}_{\hat q T^k}[s] \to \mathbb{E}_p[s]$ at the same rate, and $g_k \to g_\infty$. By the derivation `ml-gradient`, $g_\infty$ is the exact gradient.
5. For finite $k$, $\hat q T^k \ne p$ unless $\hat q = p$, since $T$ has $p$ as its unique fixed point, so $g_k$ is a biased estimate of $g_\infty$; the bias shrinks as $\rho^k$ and is largest when $\rho$ is close to 1, when the chain mixes slowly.

### Result and check

$$ \|g_k - g_\infty\| \le C \rho^k \to 0 , \qquad \rho = |\lambda_2(T)| < 1 . $$

The check `cd-k-limit.check.py` builds $T$ for random RBMs, checks that its rows sum to 1, that $p$ is a fixed point and that detailed balance holds, computes $g_k$ exactly by matrix powers, and checks that its error decreases with $k$, falls below $10^{-12}$ by $k = 79$, and shrinks at the geometric rate $\rho$ to within 5%. It also checks that scaling up the weights raises $\rho$, so the chain mixes more slowly and the bias decays more slowly in $k$.
