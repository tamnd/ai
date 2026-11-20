---
id: classifier-free-guidance
derives: [c-guidance-implicit]
kind: closed-form
assumes: ["exact noise predictions eps*(x_t, c) of q_t(x_t | c) and eps*(x_t) of q_t(x_t), related to the scores by tweedie-score", "q_t(c | x_t) > 0"]
uses: [bayes-theorem@m2, conditional-expectation@m2]
status: full
check: classifier-free-guidance.check.py
lean: null
sources:
  - {work: ho-2022-classifier-free-diffusion-guidance, locator: "pp. 4 and 5", note: "classifier guidance with weight w + 1 on an unconditional model, eq. (6) for the guided prediction, and its relation to the gradient of the implicit classifier p(c | z) proportional to p(z | c) / p(z)"}
  - {work: dhariwal-2021-diffusion-models-beat-gans, locator: "abstract", note: "classifier guidance trades diversity for fidelity using gradients from a classifier"}
---

### Statement

With exact predictions, the guided prediction $\tilde{\boldsymbol\epsilon} = (1 + w)\, \boldsymbol\epsilon^*(\mathbf{x}_t, c) - w\, \boldsymbol\epsilon^*(\mathbf{x}_t)$ satisfies

$$ \tilde{\boldsymbol\epsilon} = -\sqrt{1 - \bar\alpha_t}\; \nabla_{\mathbf{x}_t} \big[ \log q_t(\mathbf{x}_t \mid c) + w \log q_t(c \mid \mathbf{x}_t) \big] , $$

so it is the noise prediction of the tilted density $q_t(\mathbf{x}_t \mid c)\, q_t(c \mid \mathbf{x}_t)^w$, unnormalized, which favors points that the implicit classifier $q_t(c \mid \mathbf{x}_t)$ assigns to $c$.
At $w = 0$ it is the conditional prediction; the classifier is never trained.

### Assumptions

1. $\boldsymbol\epsilon^*(\mathbf{x}_t, c) = -\sqrt{1 - \bar\alpha_t}\, \nabla \log q_t(\mathbf{x}_t \mid c)$ and $\boldsymbol\epsilon^*(\mathbf{x}_t) = -\sqrt{1 - \bar\alpha_t}\, \nabla \log q_t(\mathbf{x}_t)$, by tweedie-score applied with and without the condition. Used in step 2.
2. Bayes' rule: $\log q_t(c \mid \mathbf{x}_t) = \log q_t(\mathbf{x}_t \mid c) + \log q(c) - \log q_t(\mathbf{x}_t)$. Used in step 1.

### Steps

1. The prior $q(c)$ does not depend on $\mathbf{x}_t$, so $\nabla \log q_t(c \mid \mathbf{x}_t) = \nabla \log q_t(\mathbf{x}_t \mid c) - \nabla \log q_t(\mathbf{x}_t)$.
2. By assumption 1 this is $-\big(\boldsymbol\epsilon^*(\mathbf{x}_t, c) - \boldsymbol\epsilon^*(\mathbf{x}_t)\big) / \sqrt{1 - \bar\alpha_t}$.
3. Hence $-\sqrt{1 - \bar\alpha_t}\, \nabla[\log q_t(\mathbf{x}_t \mid c) + w \log q_t(c \mid \mathbf{x}_t)] = \boldsymbol\epsilon^*(\mathbf{x}_t, c) + w\big(\boldsymbol\epsilon^*(\mathbf{x}_t, c) - \boldsymbol\epsilon^*(\mathbf{x}_t)\big) = (1 + w)\boldsymbol\epsilon^*(\mathbf{x}_t, c) - w \boldsymbol\epsilon^*(\mathbf{x}_t)$.
4. With learned predictions the identity holds only approximately: as Ho and Salimans note, the difference of two network outputs need not be the gradient of any classifier.

### Result and check

$$ (1 + w)\, \boldsymbol\epsilon^*(\mathbf{x}_t, c) - w\, \boldsymbol\epsilon^*(\mathbf{x}_t) = \boldsymbol\epsilon^*(\mathbf{x}_t, c) - w \sqrt{1 - \bar\alpha_t}\; \nabla \log q_t(c \mid \mathbf{x}_t) . $$

The check `classifier-free-guidance.check.py` takes three Gaussian classes in the plane with unequal priors, computes the exact conditional and unconditional predictions with the NumPy reference's `mixture_eps`, and compares `guided_eps` with the right side, the classifier's gradient taken by finite differences of `mixture_log_density`, for $w \in \{0, 0.5, 2, 7.5\}$ and several noise levels.
