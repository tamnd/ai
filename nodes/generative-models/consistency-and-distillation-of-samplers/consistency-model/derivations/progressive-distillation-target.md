---
id: progressive-distillation-target
derives: [c-pd-target]
kind: identity
assumes: ["a variance-preserving schedule, alpha_t^2 + sigma_t^2 = 1, with sigma_t / alpha_t increasing in t", "0 <= t'' < t' < t <= 1", "the sampler is the deterministic DDIM step"]
uses: [knowledge-distillation@al, multivariate-normal-distribution@m2]
status: full
check: progressive-distillation-target.check.py
lean: null
sources:
  - {work: salimans-2022-progressive-distillation-fast-sampling, locator: "p. 4, Algorithm 2", note: "two teacher DDIM steps from t to t' = t - 0.5/N and t'' = t - 1/N, and the target x-tilde for one student step"}
  - {work: salimans-2022-progressive-distillation-fast-sampling, locator: "p. 5", note: "the epsilon-prediction link x-hat = (z - sigma eps-hat) / alpha fails as alpha goes to 0, and the x, (x, eps) and v parameterizations"}
---

### Statement

Write $z_t = \alpha_t x + \sigma_t \epsilon$ with $\alpha_t^2 + \sigma_t^2 = 1$, and the DDIM step from $t$ to $s < t$ with an $x$-prediction $\hat x$ as

$$ z_s = \alpha_s \hat x + \frac{\sigma_s}{\sigma_t}\,(z_t - \alpha_t \hat x) . $$

Let the teacher take two such steps, $z_t \to z_{t'} \to z_{t''}$. With $r = \sigma_{t''}/\sigma_t$, the student target of progressive distillation

$$ \tilde x = \frac{z_{t''} - r\, z_t}{\alpha_{t''} - r\, \alpha_t} $$ {#eq-pd}

is the one $x$-prediction whose single DDIM step from $t$ to $t''$ lands on $z_{t''}$; the denominator $\alpha_{t''} - r\alpha_t$ is positive whenever $t'' < t$, so the target exists even at $t = 1$ where $\alpha_1 = 0$; when the teacher predicts the same $\hat x$ at both of its steps, $\tilde x = \hat x$.
The $\epsilon$-prediction link $\hat x = (z_t - \sigma_t \hat\epsilon)/\alpha_t$ multiplies an error $\delta$ in $\hat\epsilon$ by $\sigma_t/\alpha_t$, which grows without bound as $\alpha_t \to 0$, while the $v$-prediction $v = \alpha_t \epsilon - \sigma_t x$ gives $\hat x = \alpha_t z_t - \sigma_t \hat v$, which multiplies the error by $\sigma_t \le 1$.

### Assumptions

1. $\alpha_t^2 + \sigma_t^2 = 1$ and $\sigma_t/\alpha_t$ increases in $t$. Used in steps 2 and 4.
2. $t'' < t$. Used in step 2.
3. The sampler is the deterministic DDIM step. Used in step 1.

### Steps

1. The DDIM step from $t$ to $t''$ with prediction $\tilde x$ is $\alpha_{t''}\tilde x + r(z_t - \alpha_t \tilde x)$, which is affine in $\tilde x$ with slope $\alpha_{t''} - r\alpha_t$. Setting it equal to $z_{t''}$ and solving gives eq-pd.
2. Write $\alpha_t = \cos\phi_t$ and $\sigma_t = \sin\phi_t$ with $\phi_t \in [0, \pi/2]$ increasing in $t$. Then $\sigma_t(\alpha_{t''} - r\alpha_t) = \cos\phi_{t''}\sin\phi_t - \sin\phi_{t''}\cos\phi_t = \sin(\phi_t - \phi_{t''}) > 0$.
3. If the teacher predicts $\hat x$ at $t$ and at $t'$, its two steps give $z_{t'} = \alpha_{t'}\hat x + (\sigma_{t'}/\sigma_t)(z_t - \alpha_t\hat x)$, so $z_{t'} - \alpha_{t'}\hat x = (\sigma_{t'}/\sigma_t)(z_t - \alpha_t \hat x)$ and $z_{t''} = \alpha_{t''}\hat x + r(z_t - \alpha_t\hat x)$, which is one step with $\hat x$; by step 1, $\tilde x = \hat x$.
4. $\hat x = (z_t - \sigma_t\hat\epsilon)/\alpha_t$ changes by $\sigma_t\delta/\alpha_t$ when $\hat\epsilon$ changes by $\delta$. For the $v$-prediction, $\alpha_t z_t - \sigma_t v = \alpha_t^2 x + \alpha_t\sigma_t\epsilon - \sigma_t\alpha_t\epsilon + \sigma_t^2 x = x$, and a change $\delta$ in $\hat v$ changes $\hat x$ by $\sigma_t\delta$.

### Result and check

A student trained on eq-pd reproduces two teacher steps with one, and repeating the halving takes Salimans and Ho's sampler from 8192 steps to 4.
The same construction is one step of the consistency function: both distil a solver of the probability flow, but progressive distillation matches the end of a fixed pair of steps, so each round fixes the number of steps of its student, while consistency distillation matches any point of the trajectory to the next and lets one model sample in one step or several.
The check `progressive-distillation-target.check.py` takes a cosine schedule $\alpha_t = \cos(\pi t/2)$, teachers that predict nonlinear functions of $z$, and $N$ from 1 to 64; it compares the student step with eq-pd against the two teacher steps, the denominator against $\sin(\phi_t - \phi_{t''})/\sigma_t$, including at $t = 1$, the target against $\hat x$ for a teacher with a constant prediction, and the error that a perturbation of size $10^{-3}$ in $\hat\epsilon$ and in $\hat v$ causes in $\hat x$ as $t \to 1$.
