---
id: parameterization
derives: [c-param-boundary]
kind: identity
assumes: ["F_theta(x, t) is finite for every x and t, whatever theta", "sigma_data > 0 and 0 < eps < T"]
uses: [variance@m2, loss-function@m2]
status: full
check: parameterization.check.py
lean: null
sources:
  - {work: song-2023-consistency-models, locator: "p. 3, Eq. 5; pp. 25 to 26, Appendix C", note: "the skip parameterization with c_skip(eps) = 1 and c_out(eps) = 0, and the scales of Karras and colleagues shifted by eps"}
  - {work: karras-2022-elucidating-design-space-diffusion, locator: "p. 2", note: "the preconditioning c_skip, c_out and c_in of the denoiser with sigma_data = 0.5"}
---

### Statement

Let $\sigma_d > 0$ and $0 < \epsilon < T$, and define for $t \in [\epsilon, T]$

$$ c_{\text{skip}}(t) = \frac{\sigma_d^2}{(t - \epsilon)^2 + \sigma_d^2} , \qquad c_{\text{out}}(t) = \frac{\sigma_d\,(t - \epsilon)}{\sqrt{\sigma_d^2 + t^2}} . $$

Then for every network $F_\theta$ with finite outputs the model $f_\theta(x, t) = c_{\text{skip}}(t)\,x + c_{\text{out}}(t)\,F_\theta(x, t)$ satisfies the boundary condition $f_\theta(x, \epsilon) = x$ exactly, for every $\theta$; $c_{\text{skip}}$ falls from 1 to $\sigma_d^2/((T - \epsilon)^2 + \sigma_d^2)$ and $c_{\text{out}}$ rises from 0 towards $\sigma_d$ as $t$ grows; and both are differentiable in $t$, so $f_\theta$ is differentiable at $t = \epsilon$ when $F_\theta$ is.

### Assumptions

1. $F_\theta(x, \epsilon)$ is finite. Used in step 1.
2. $\sigma_d > 0$ and $\epsilon < T$. Used in steps 2 and 3.

### Steps

1. At $t = \epsilon$, $c_{\text{skip}}(\epsilon) = \sigma_d^2/\sigma_d^2 = 1$ and $c_{\text{out}}(\epsilon) = 0$, so $f_\theta(x, \epsilon) = x + 0 \cdot F_\theta(x, \epsilon) = x$, with no condition on $\theta$.
2. For $t > \epsilon$ the denominator of $c_{\text{skip}}$ grows with $t$, so $c_{\text{skip}}$ falls; at $T = 80$, $\epsilon = 0.002$ and $\sigma_d = 0.5$ it is $0.25/(79.998^2 + 0.25) \approx 3.9 \times 10^{-5}$.
3. $c_{\text{out}}(t) = \sigma_d\,(t - \epsilon)/\sqrt{\sigma_d^2 + t^2}$ has derivative $\sigma_d(\sigma_d^2 + \epsilon t)/(\sigma_d^2 + t^2)^{3/2} > 0$, so it rises, and it tends to $\sigma_d$ as $t \to \infty$.
4. Both scales are quotients of polynomials and square roots with positive denominators, so they are smooth on $[\epsilon, T]$.

### Result and check

The boundary condition is built into the architecture, not learned: training never has to find it, and it rules out the trivial solution $f_\theta \equiv 0$ that would otherwise make every consistency loss zero.
At large $t$ the input carries almost no information about the data and the output is almost all $c_{\text{out}} F_\theta$, of scale $\sigma_d$; near $\epsilon$ the output is almost all the input, which is already nearly clean.
With $\epsilon = 0$ the scales are exactly Karras and colleagues' $c_{\text{skip}}(\sigma) = \sigma_d^2/(\sigma^2 + \sigma_d^2)$ and $c_{\text{out}}(\sigma) = \sigma\sigma_d/\sqrt{\sigma^2 + \sigma_d^2}$; the shift by $\epsilon$ is what moves the identity from $t = 0$ to the smallest level the model is trained on.
The check `parameterization.check.py` evaluates the model at $t = \epsilon$ with random and with huge parameters and compares the output with the input bit for bit, checks the monotonicity and the limit on a grid, compares the derivative of $c_{\text{out}}$ with a finite difference, and compares the scales at $\epsilon = 0$ with those of Karras and colleagues.
