---
id: rescale
derives: [c-rescale-std]
kind: identity
assumes: ["the standard deviations are taken over all the elements of one sample, the same way for both outputs", "sigma_cfg > 0 and 0 <= phi <= 1"]
uses: [standard-deviation@m2]
status: full
check: rescale.check.py
lean: null
sources:
  - {work: lin-2023-common-diffusion-noise-schedules, locator: "p. 4, Eqs. 13 to 16 and Algorithm 2", note: "the rescale of the guided output to the standard deviation of the conditional one, blended with the guided output by phi"}
---

### Statement

Let $x_{\text{pos}}$ be the conditional output and $x_{\text{cfg}}$ the guided output of one sample, with standard deviations $\sigma_{\text{pos}}$ and $\sigma_{\text{cfg}} > 0$ over its elements. Lin and colleagues' Eqs. 15 and 16,

$$ x_{\text{rescaled}} = x_{\text{cfg}}\,\frac{\sigma_{\text{pos}}}{\sigma_{\text{cfg}}} , \qquad x_{\text{final}} = \phi\, x_{\text{rescaled}} + (1 - \phi)\, x_{\text{cfg}} , $$

equal the single multiplication $x_{\text{final}} = x_{\text{cfg}}\,(\phi\,\sigma_{\text{pos}}/\sigma_{\text{cfg}} + 1 - \phi)$ of their Algorithm 2, and

$$ \mathrm{std}(x_{\text{final}}) = \phi\,\sigma_{\text{pos}} + (1 - \phi)\,\sigma_{\text{cfg}} . $$

The factor is the same whether the standard deviations divide by $n$ or by $n - 1$. With $\phi = 1$ the output has exactly the conditional output's standard deviation; with $\phi = 0$ it is the guided output.

### Assumptions

1. Both standard deviations are computed over the same elements with the same divisor. Used in step 2.
2. $\sigma_{\text{cfg}} > 0$ and $0 \le \phi \le 1$, so the factor is positive. Used in step 3.

### Steps

1. $\phi\, x_{\text{cfg}}\sigma_{\text{pos}}/\sigma_{\text{cfg}} + (1 - \phi)x_{\text{cfg}} = x_{\text{cfg}}(\phi\sigma_{\text{pos}}/\sigma_{\text{cfg}} + 1 - \phi)$.
2. A standard deviation with divisor $n - k$ is $\sqrt{n/(n - k)}$ times the one with divisor $n$, for both outputs alike, so the ratio $\sigma_{\text{pos}}/\sigma_{\text{cfg}}$ does not depend on $k$.
3. For a constant $c > 0$, $\mathrm{std}(c\,x) = c\,\mathrm{std}(x)$; with $c = \phi\sigma_{\text{pos}}/\sigma_{\text{cfg}} + 1 - \phi$, $\mathrm{std}(x_{\text{final}}) = c\,\sigma_{\text{cfg}} = \phi\sigma_{\text{pos}} + (1 - \phi)\sigma_{\text{cfg}}$.

### Result and check

The rescale undoes the growth of the overall scale of the output with $s$, which over-exposes images when guidance is large, while keeping the direction that guidance chose; $\phi$ interpolates the standard deviation linearly between the guided and the conditional one. It changes only one number per sample, so it cannot fix a wrong mean or a wrong distribution of values within the sample.
Because the factor is a ratio of standard deviations, the rescale depends on which quantity is rescaled: Lin and colleagues apply it to the outputs of a model that predicts $v$, and applied to the noise or to $\hat x_0$ it gives another result (derivation guidance-conventions).
The check `rescale.check.py` compares the two-step form with `rescale`, the standard deviations of the output with the blend for divisors $n$ and $n - 1$, and the two ends $\phi = 0$ and $\phi = 1$.
