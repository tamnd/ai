---
id: conditional-ot-field
derives: [c-cond-ot]
kind: closed-form
assumes: ["0 <= sigma_min < 1", "x_0 ~ N(0, I)", "x_1 is a fixed data point"]
uses: [multivariate-normal-distribution@m2, expected-value@m2]
status: full
check: conditional-ot-field.check.py
lean: null
sources:
  - {work: lipman-2022-flow-matching-generative-modeling, locator: "pp. 5, 6", note: "the optimal transport conditional path with mean t x_1 and standard deviation 1 - (1 - sigma_min) t, its flow and its vector field, eqs. 20 to 23"}
---

### Statement

Let $\sigma_t = 1 - (1 - \sigma_{\min})t$ and

$$ \psi_t(\mathbf{x}_0) = \sigma_t \mathbf{x}_0 + t\mathbf{x}_1 . $$

If $\mathbf{x}_0 \sim \mathcal{N}(0, I)$, then $\psi_t(\mathbf{x}_0) \sim \mathcal{N}(t\mathbf{x}_1, \sigma_t^2 I)$. This path starts at the noise when $t = 0$ and ends at $\mathcal{N}(\mathbf{x}_1, \sigma_{\min}^2 I)$ when $t = 1$. Each point of it moves with the constant velocity

$$ \frac{d}{dt}\psi_t(\mathbf{x}_0) = \mathbf{x}_1 - (1 - \sigma_{\min})\mathbf{x}_0 . $$

Written as a field of the position $\mathbf{x}$, this velocity is

$$ u_t(\mathbf{x} \mid \mathbf{x}_1) = \frac{\mathbf{x}_1 - (1 - \sigma_{\min})\mathbf{x}}{1 - (1 - \sigma_{\min})t} , $$

and the field is defined for every $t \in [0, 1]$.

### Assumptions

1. An affine map of a standard normal vector is normal: $a\mathbf{x}_0 + \mathbf{b} \sim \mathcal{N}(\mathbf{b}, a^2 I)$. Used in step 1.
2. $\sigma_t > 0$ on $[0, 1]$. This holds because $\sigma_t$ is linear in $t$, equals 1 at $t = 0$ and equals $\sigma_{\min} \ge 0$ at $t = 1$; for $\sigma_{\min} = 0$ it reaches 0 only at $t = 1$. Used in step 3.

### Steps

1. $\psi_t(\mathbf{x}_0)$ is $\sigma_t\mathbf{x}_0$ shifted by $t\mathbf{x}_1$. Its mean is $t\mathbf{x}_1$ and its covariance is $\sigma_t^2 I$. At $t = 0$ it is $\mathbf{x}_0$; at $t = 1$ it is $\sigma_{\min}\mathbf{x}_0 + \mathbf{x}_1$.
2. Differentiate in $t$ to get $\dot\sigma_t \mathbf{x}_0 + \mathbf{x}_1$, where $\dot\sigma_t = -(1 - \sigma_{\min})$. The result does not depend on $t$, so each particle moves along a straight line at constant speed.
3. Since $\sigma_t > 0$, the map $\psi_t$ can be inverted: $\mathbf{x}_0 = (\mathbf{x} - t\mathbf{x}_1)/\sigma_t$. Substituting this into step 2 gives
   $$\mathbf{x}_1 - (1 - \sigma_{\min})\frac{\mathbf{x} - t\mathbf{x}_1}{\sigma_t} = \frac{\sigma_t\mathbf{x}_1 + (1 - \sigma_{\min})t\mathbf{x}_1 - (1 - \sigma_{\min})\mathbf{x}}{\sigma_t} .$$
   The numerator simplifies to $\mathbf{x}_1 - (1 - \sigma_{\min})\mathbf{x}$.

### Result and check

$$ u_t(\psi_t(\mathbf{x}_0) \mid \mathbf{x}_1) = \frac{d}{dt}\psi_t(\mathbf{x}_0) = \mathbf{x}_1 - (1 - \sigma_{\min})\mathbf{x}_0 . $$

The regression target of the conditional flow matching loss is therefore the right-hand side, and it costs no more to compute than the path itself.

The check `conditional-ot-field.check.py` runs the NumPy reference with $\sigma_{\min} = 0$ and $\sigma_{\min} = 0.1$. It:

- pushes 200 000 noise draws through `path` and compares the sample mean and standard deviation with $t\mathbf{x}_1$ and $\sigma_t$;
- differentiates `path` in $t$ by central differences and compares the result with `target`;
- evaluates `cond_field` at the moved points and checks that it equals `target`.
