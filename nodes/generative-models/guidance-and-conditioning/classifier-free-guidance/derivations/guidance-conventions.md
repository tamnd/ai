---
id: guidance-conventions
derives: [c-conventions]
kind: identity
assumes: ["the two predictions are made at the same noisy sample z_t and noise level", "the conversions between the noise, the x-prediction, v and the score are the affine maps of the Gaussian forward process"]
uses: [score@m2]
status: full
check: guidance-conventions.check.py
lean: null
sources:
  - {work: ho-2022-classifier-free-diffusion-guidance, locator: "p. 5, Eq. 6", note: "the guided noise (1 + w) eps(z, c) - w eps(z)"}
  - {work: nichol-2021-glide-towards-photorealistic-image, locator: "p. 5", note: "eps(x | empty) + s (eps(x | c) - eps(x | empty)) with s >= 1"}
  - {work: saharia-2022-photorealistic-text-image-diffusion, locator: "p. 4, Eq. 2", note: "w eps(z, c) + (1 - w) eps(z)"}
  - {work: karras-2024-guiding-diffusion-model-bad, locator: "p. 2, Eq. 3", note: "D_w = w D_1 + (1 - w) D_0 on denoiser outputs"}
---

### Statement

Write $\epsilon_c = \epsilon_\theta(z_t, c)$ and $\epsilon_\varnothing = \epsilon_\theta(z_t, \varnothing)$, and let $\tilde\epsilon_s = \epsilon_\varnothing + s(\epsilon_c - \epsilon_\varnothing)$. Then

$$ (1 + w)\,\epsilon_c - w\,\epsilon_\varnothing = \tilde\epsilon_{1 + w} , \qquad w\,\epsilon_c + (1 - w)\,\epsilon_\varnothing = \tilde\epsilon_{w} , $$

so Ho and Salimans' weight $w$ is the scale $s = 1 + w$, and the weight of GLIDE, Imagen and Karras and colleagues is $s$ itself; $w = 0$ in Ho and Salimans' form, or $s = 1$, is the conditional model. Moreover, for any of the parameterizations
$\hat x_0 = (z_t - \sqrt{1 - \bar\alpha_t}\,\epsilon)/\sqrt{\bar\alpha_t}$, $v = \sqrt{\bar\alpha_t}\,\epsilon - \sqrt{1 - \bar\alpha_t}\,\hat x_0$ and $\nabla \log p = -\epsilon/\sqrt{1 - \bar\alpha_t}$, applying the same combination to the two outputs gives the conversion of $\tilde\epsilon_s$:

$$ \hat x_0^{\,\varnothing} + s(\hat x_0^{\,c} - \hat x_0^{\,\varnothing}) = \hat x_0(\tilde\epsilon_s) , $$

and likewise for $v$ and for the score.

### Assumptions

1. The two outputs are made at the same $z_t$ and $\bar\alpha_t$. Used in step 2.
2. The conversions are the affine maps above, whose coefficients depend on $z_t$ and $\bar\alpha_t$ only, not on the condition. Used in step 2.

### Steps

1. $(1 + w)\epsilon_c - w\epsilon_\varnothing = \epsilon_\varnothing + (1 + w)(\epsilon_c - \epsilon_\varnothing)$ and $w\epsilon_c + (1 - w)\epsilon_\varnothing = \epsilon_\varnothing + w(\epsilon_c - \epsilon_\varnothing)$ by expanding.
2. Each conversion has the form $g(\epsilon) = a + b\epsilon$ with $a$ and $b$ fixed by $z_t$ and $\bar\alpha_t$. The weights $1 - s$ and $s$ sum to 1, so $(1 - s)g(\epsilon_\varnothing) + s\,g(\epsilon_c) = a + b\big((1 - s)\epsilon_\varnothing + s\epsilon_c\big) = g(\tilde\epsilon_s)$. For $v$, substituting $\hat x_0$ gives $v = (\epsilon - \sqrt{1 - \bar\alpha_t}\, z_t)/\sqrt{\bar\alpha_t}$, again affine in $\epsilon$.

### Result and check

A guidance scale means the same thing whichever quantity the network predicts, but the same number does not mean the same scale across papers: $w = 7.5$ in Imagen's or Stable Diffusion's form is $w = 6.5$ in Ho and Salimans' form.
The identity needs the weights to sum to 1; a combination such as $s\epsilon_c$ alone is not invariant, and the rescale of eq-rescale, which multiplies by a ratio of standard deviations, is not affine and does depend on the parameterization.
The check `guidance-conventions.check.py` compares the three forms with `guide` and `guide_ho`, and converts random predictions with `eps_to_x0` and `eps_to_v` before and after guiding.
