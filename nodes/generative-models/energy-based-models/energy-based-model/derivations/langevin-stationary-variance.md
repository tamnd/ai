---
id: langevin-stationary-variance
derives: [c-langevin-variance]
kind: closed-form
assumes: ["the noise variance equals the step", "the chain is run long enough for the initial condition to be forgotten"]
uses: [langevin-equation@m2, markov-chain@m2, stationary-distribution@m2, normal-distribution@m2, multivariate-normal-distribution@m2]
status: full
check: langevin-stationary-variance.check.py
lean: null
sources:
  - {work: welling-2011-bayesian-learning-stochastic-gradient-langevin, locator: "p. 2", note: "Langevin dynamics as gradient steps plus Gaussian noise of variance equal to the step, and a discretization error that decreases with the step"}
  - {work: du-2019-implicit-generation-generalization-energy-based, locator: "p. 3", note: "the sampler of a network energy, which samples the model only as the step goes to 0 and the number of steps to infinity"}
---

### Statement

For $E(\mathbf{x}) = \|\mathbf{x}\|^2/(2s^2)$, $\mathbf{x} \in \mathbb{R}^D$, whose Gibbs distribution is $\mathcal{N}(0, s^2 I)$, the chain

$$ \mathbf{x}_k = \mathbf{x}_{k-1} - \frac{\lambda}{2} \nabla E(\mathbf{x}_{k-1}) + \boldsymbol{\omega}_k , \qquad \boldsymbol{\omega}_k \sim \mathcal{N}(0, \lambda I) , $$

has, for $0 < \lambda < 4s^2$, the stationary distribution $\mathcal{N}(0, v I)$ with

$$ v = \frac{s^2}{1 - \lambda/(4s^2)} > s^2 , $$

and $v \to s^2$ as $\lambda \to 0$. For $\lambda \ge 4s^2$ the chain has no stationary distribution.

### Assumptions

1. The noise variance equals $\lambda$. Used in step 2.
2. The chain has run for many steps. Used in step 3.

### Steps

1. $\nabla E = \mathbf{x}/s^2$, so $\mathbf{x}_k = a \mathbf{x}_{k-1} + \boldsymbol{\omega}_k$ with $a = 1 - \lambda/(2s^2)$, an independent first-order autoregression in each coordinate.
2. The variance obeys $v_k = a^2 v_{k-1} + \lambda$. For $|a| < 1$, which is $0 < \lambda < 4s^2$, it converges to the fixed point $v = \lambda / (1 - a^2)$ at the rate $a^{2k}$, and the stationary distribution is Gaussian because each $\mathbf{x}_k$ is a linear combination of Gaussians.
3. $1 - a^2 = \lambda/s^2 - \lambda^2/(4s^4) = (\lambda/s^2)(1 - \lambda/(4s^2))$, so $v = s^2/(1 - \lambda/(4s^2))$.
4. $1 - \lambda/(4s^2) < 1$, so $v > s^2$: the discretized chain overdisperses, by a relative error $\lambda/(4s^2) + O(\lambda^2)$. For $|a| \ge 1$ the variance grows without bound.
5. A Metropolis-Hastings correction with this proposal would restore $\mathcal{N}(0, s^2 I)$ exactly; without it, the bias vanishes only as $\lambda \to 0$.

### Result and check

$$ v(\lambda) = \frac{s^2}{1 - \lambda/(4s^2)} = s^2 + \frac{\lambda}{4} + O(\lambda^2) . $$

The check `langevin-stationary-variance.check.py` iterates the variance recursion for several steps and widths, runs `langevin` from the NumPy reference with the network part switched off on 40,000 chains, and compares the sample variance with $v$. It also checks that the chain diverges for $\lambda > 4s^2$.
