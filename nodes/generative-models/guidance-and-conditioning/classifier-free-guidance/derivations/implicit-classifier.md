---
id: implicit-classifier
derives: [c-tilted-score, c-cg-equivalence]
kind: identity
assumes: ["the two noise predictors are exact, eps(z_t, c) = -sqrt(1 - abar_t) grad log p_t(z_t | c) and eps(z_t) = -sqrt(1 - abar_t) grad log p_t(z_t)", "p_t(z_t | c) and p_t(z_t) are positive and differentiable"]
uses: [bayes-theorem@m2, score@m2, mixture-model@m2]
status: full
check: implicit-classifier.check.py
lean: null
sources:
  - {work: ho-2022-classifier-free-diffusion-guidance, locator: "pp. 3 to 5", note: "eps = -sigma grad log p, the implicit classifier p(c | z) proportional to p(z | c)/p(z), and the equivalence of the weights w + 1 and w"}
  - {work: dhariwal-2021-diffusion-models-beat-gans, locator: "p. 8, Eq. 14", note: "the classifier-guided noise eps - sqrt(1 - abar) grad log p(y | x_t)"}
  - {work: song-2020-score-based-generative-modeling-sdes, locator: "p. 9, Eq. 14", note: "the conditional reverse SDE with the score plus the gradient of a time-dependent classifier"}
---

### Statement

With exact noise predictors and $\sigma_t = \sqrt{1 - \bar\alpha_t}$, the guided noise $\tilde\epsilon_s = \epsilon(z_t) + s(\epsilon(z_t, c) - \epsilon(z_t))$ is

$$ \tilde\epsilon_s = -\sigma_t \nabla_{z_t} \log \tilde p_s(z_t \mid c) , \qquad \tilde p_s(z_t \mid c) \propto p_t(z_t \mid c)^s\, p_t(z_t)^{1 - s} \propto p_t(z_t)\, p_t(c \mid z_t)^s , $$

and it equals classifier guidance with the exact noisy classifier at scale $s$ on the unconditional model, and at scale $s - 1$ on the conditional model:

$$ \tilde\epsilon_s = \epsilon(z_t) - s\,\sigma_t \nabla \log p_t(c \mid z_t) = \epsilon(z_t, c) - (s - 1)\,\sigma_t \nabla \log p_t(c \mid z_t) . $$

### Assumptions

1. Exact predictors, which the noise-prediction loss has as its minimizer. Used in step 1.
2. Positive, differentiable densities, so the logarithms and gradients exist. Used in steps 1 and 2.

### Steps

1. Substituting the two predictors, $\tilde\epsilon_s = -\sigma_t\big[(1 - s)\nabla\log p_t(z_t) + s\nabla \log p_t(z_t \mid c)\big] = -\sigma_t \nabla\log\big(p_t(z_t \mid c)^s p_t(z_t)^{1 - s}\big)$; a normalizing constant does not change the gradient.
2. By Bayes' theorem $p_t(z_t \mid c) = p_t(c \mid z_t)\, p_t(z_t)/p(c)$, so $p_t(z_t \mid c)^s p_t(z_t)^{1 - s} = p_t(z_t)\, p_t(c \mid z_t)^s / p(c)^s$, and $\nabla \log p_t(z_t \mid c) = \nabla\log p_t(z_t) + \nabla\log p_t(c \mid z_t)$.
3. Hence $\tilde\epsilon_s = -\sigma_t\nabla\log p_t(z_t) - s\sigma_t \nabla\log p_t(c \mid z_t) = \epsilon(z_t) - s\sigma_t\nabla\log p_t(c\mid z_t)$, and writing $\nabla\log p_t(z_t) = \nabla\log p_t(z_t \mid c) - \nabla\log p_t(c \mid z_t)$ gives the form with $\epsilon(z_t, c)$ and $s - 1$.

### Result and check

Classifier-free guidance is classifier guidance with the classifier $p_t(c \mid z_t) \propto p_t(z_t \mid c)/p_t(z_t)$ that the two generative models imply, which is why Ho and Salimans call it the implicit classifier.
The identity holds for the exact scores; a trained network's two outputs are not the gradients of one function in general, so the guided field need not be a gradient at all, and the tilted density $\tilde p_s(z_t \mid c)$ at each $t$ is not the marginal of one diffusion process started from $\tilde p_s(x_0 \mid c)$ (derivation guided-ode-variance).
The check `implicit-classifier.check.py` uses the exact predictors of a mixture of three Gaussians in three dimensions, compares $\tilde\epsilon_s$ with central differences of $s\log p_t(z_t \mid c) + (1 - s)\log p_t(z_t)$, and compares both classifier-guided forms with `classifier_grad`, at three noise levels and scales 0, 2.5 and 7.
