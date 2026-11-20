---
id: gaussian-consistency-function
derives: [c-gauss-cf]
kind: closed-form
assumes: ["the data are N(mu, s2 I) with s2 > 0", "x_t = x_0 + t z with z standard normal, so p_t = N(mu, (s2 + t^2) I)", "the probability-flow ODE dx/dt = -t grad log p_t(x)"]
uses: [multivariate-normal-distribution@m2, sum-of-independent-normal-random-variables@m2, stochastic-differential-equation@m2, score@m2]
status: full
check: gaussian-consistency-function.check.py
lean: null
sources:
  - {work: song-2023-consistency-models, locator: "pp. 2 to 3, Eqs. 2 to 3 and the definition of the consistency function", note: "the probability-flow ODE dx/dt = -t grad log p_t(x) of the settings of Karras and colleagues, and the consistency function f(x_t, t) = x_eps with its self-consistency"}
---

### Statement

Let the data be $x_0 \sim \mathcal N(\mu, s^2 I)$ and $x_t = x_0 + t z$ with $z \sim \mathcal N(0, I)$. Along every solution of the probability-flow ODE

$$ \frac{dx}{dt} = -t\,\nabla_x \log p_t(x) $$

the quantity $(x - \mu)/\sqrt{s^2 + t^2}$ is constant, so the consistency function, which maps a point $x$ at level $t$ to the point of its path at level $\epsilon$, is

$$ f(x, t) = \mu + (x - \mu)\sqrt{\frac{s^2 + \epsilon^2}{s^2 + t^2}} . $$ {#eq-gauss-cf}

It satisfies the boundary condition $f(x, \epsilon) = x$, its total derivative along the ODE is zero, and a sample $x_T \sim \mathcal N(\mu, (s^2 + T^2) I)$ maps to $f(x_T, T) \sim \mathcal N(\mu, (s^2 + \epsilon^2) I)$, the noisy data at level $\epsilon$.

### Assumptions

1. The data are Gaussian with $s^2 > 0$. Used in step 1.
2. The noise is added as $x_t = x_0 + t z$, independent of $x_0$. Used in step 1.
3. The ODE is the probability-flow ODE of this process. Used in step 2.

### Steps

1. The sum of independent Gaussians $x_0$ and $t z$ is $\mathcal N(\mu, (s^2 + t^2) I)$, so $\nabla_x \log p_t(x) = -(x - \mu)/(s^2 + t^2)$.
2. The ODE is $dx/dt = t\,(x - \mu)/(s^2 + t^2)$. With $u = x - \mu$, $du/u = t\,dt/(s^2 + t^2) = \tfrac12\,d \log(s^2 + t^2)$ coordinate by coordinate, so $\log |u| - \tfrac12 \log(s^2 + t^2)$ is constant: $u(t)/\sqrt{s^2 + t^2} = u(t')/\sqrt{s^2 + t'^2}$ for any two levels on one path.
3. Taking $t' = \epsilon$ gives $x_\epsilon = \mu + (x_t - \mu)\sqrt{(s^2 + \epsilon^2)/(s^2 + t^2)}$, which is eq-gauss-cf; at $t = \epsilon$ the factor is 1.
4. The total derivative is $\partial_t f + \nabla_x f \cdot dx/dt = -(x - \mu)\,t\,\sqrt{s^2 + \epsilon^2}\,(s^2 + t^2)^{-3/2} + \sqrt{(s^2 + \epsilon^2)/(s^2 + t^2)}\;t\,(x - \mu)/(s^2 + t^2) = 0$.
5. $f(\cdot, T)$ is affine with slope $\sqrt{(s^2 + \epsilon^2)/(s^2 + T^2)}$, so it maps $\mathcal N(\mu, (s^2 + T^2) I)$ to $\mathcal N(\mu, (s^2 + \epsilon^2) I)$.

### Result and check

For Gaussian data the consistency function is a shrinkage towards the mean by the ratio of the standard deviations at the two levels; one evaluation turns noise into data with the right distribution, which is what a consistency model learns to do for any data.
The map is deterministic and invertible, a one-to-one map from noise to data, so nearby noise vectors give nearby samples and interpolation in the noise moves smoothly between samples.
The check `gaussian-consistency-function.check.py` solves the ODE with 400 Heun steps from several levels and compares the end with eq-gauss-cf, checks that eq-gauss-cf takes the same value at every point of one solved path, computes the total derivative by finite differences, and compares the mean and covariance of $f(x_T, T)$ over 200000 samples with $\mathcal N(\mu, (s^2 + \epsilon^2) I)$.
