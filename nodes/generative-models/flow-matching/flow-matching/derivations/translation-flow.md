---
id: translation-flow
derives: [c-translation]
kind: closed-form
assumes: ["noise N(0, 1) and data N(2, 1) in one dimension, independent", "sigma_min = 0", "the marginal field of marginal-field"]
uses: [multivariate-normal-distribution@m2, conditional-expectation@m2]
status: full
check: translation-flow.check.py
lean: null
sources:
  - {work: liu-2022-flow-straight-fast-rectified, locator: "p. 5", note: "perfectly straight paths can be simulated exactly with a single Euler step; the rectified flow of an independent coupling is in general not straight"}
---

### Statement

Take noise $\mathcal{N}(0, 1)$ and data $\mathcal{N}(2, 1)$, coupled independently, with $\sigma_{\min} = 0$. Let $g(t) = \sqrt{(1 - t)^2 + t^2}$.

The marginal field is $u_t(x) = 2 + \kappa_t(x - 2t)$, where $\kappa_t = (2t - 1)/g(t)^2$.

The flow from $x_0$ is

$$ x_t = 2t + x_0\, g(t) . $$

It ends at $x_1 = x_0 + 2$. This is the translation, which is the optimal transport map between the two Gaussians. Along the way, though, $x_t - 2t$ first shrinks by the factor $g(1/2) = 1/\sqrt 2$ and then grows back, so the paths are curved.

Euler's method with $n$ steps maps two noise points one unit apart to points $0$, $0.5$ and $0.72$ apart, for $n = 1, 2, 4$. One midpoint step is exact.

### Assumptions

1. The marginal field of a single Gaussian component, from step 4 of marginal-field, with $m = 2$, $s = 1$ and $a = 1 - t$. Then $\kappa_t = (t - (1 - t))/((1 - t)^2 + t^2)$. Used in step 1.

### Steps

1. Let $y = x - 2t$. Then $\dot y = u_t(x) - 2 = \kappa_t y$.
2. Note that $\kappa_t = \tfrac{1}{2}\frac{d}{dt}\ln g(t)^2 = \frac{d}{dt}\ln g(t)$. Since $g(0) = 1$, the solution is $y_t = y_0\, g(t) = x_0\, g(t)$. Because $g(1) = 1$, the flow ends at $x_1 = 2 + x_0$.
3. An Euler step of size $h$ multiplies $y$ by $1 + h\kappa_t$. The distance between two points is a difference of $y$ values, so it is multiplied by the product of these factors over the steps:
   - $n = 1$: $\kappa_0 = -1$, so the product is $1 - 1 = 0$.
   - $n = 2$: $(1 - \tfrac12)(1 + \tfrac12\kappa_{1/2}) = \tfrac12 \cdot 1 = 0.5$.
   - $n = 4$: with $\kappa_{1/4} = -0.8$, $\kappa_{1/2} = 0$ and $\kappa_{3/4} = 0.8$, the product is $0.75 \cdot 0.8 \cdot 1 \cdot 1.2 = 0.72$.
4. One midpoint step goes from $x$ to $x + u_{1/2}\big(x + \tfrac12 u_0(x)\big)$. Since $\kappa_{1/2} = 0$, the field at $t = \tfrac12$ is the constant 2, so the step lands exactly on $x + 2$.

### Result and check

$$ x_t = 2t + x_0\sqrt{(1 - t)^2 + t^2}, \qquad x_1 = x_0 + 2 . $$

The end-to-end map is a straight translation, but the flow follows curved paths, and the Euler error comes from that curvature. Reflow, run on the coupling $(x_0, x_0 + 2)$, would straighten the paths.

The check `translation-flow.check.py`:

- differentiates the claimed path by central differences and compares the result with `mixture_field`;
- confirms the closed form of $\kappa_t$;
- runs `midpoint_sample` with 400 steps and with 1 step, checking that both give $x_0 + 2$;
- runs `euler_sample` with 1, 2 and 4 steps and checks the gaps 0, 0.5 and 0.72.
