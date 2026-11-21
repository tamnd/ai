---
tag: DSB2H8
kind: method
slug: classifier-free-guidance
name: Classifier-free guidance
aliases: [CFG, classifier free guidance, guidance scale, guidance weight, classifier guidance, unconditional guidance, negative prompt guidance, guidance interval, dynamic thresholding, guidance rescale, autoguidance]
area: generative-models/guidance-and-conditioning
level: core
summary: >-
  Train one diffusion model with and without its condition, then sample by extrapolating from the unconditional prediction past the conditional one, trading diversity for fidelity and alignment.
purpose: conditional sampling
uses: [bayes-theorem@m2, conditional-probability@m2, score@m2, normal-distribution@m2, multivariate-normal-distribution@m2, mixture-model@m2, variance@m2, standard-deviation@m2, quantile-function@m2, orthogonal-projection@m2, stochastic-differential-equation@m2, matrix-multiplication@m2, gaussian-mixture-model@al, logistic-regression@al, backpropagation@al, ancestral-sampling@al]
related_to: [diffusion-model, latent-diffusion-model, flow-matching, generative-adversarial-network, energy-based-model, adam]
hyperparameters:
  - {name: s, meaning: "the guidance scale of eq-cfg; s = 1 is the conditional model and s = 0 the unconditional one; Ho and Salimans write w = s - 1", typical: "1.5 to 3 for the best FID of class-conditional models; 5 to 10 for text-to-image, such as 7.5"}
  - {name: p_uncond, meaning: "the probability of replacing the condition by the null token during training", typical: "0.1 or 0.2; 0.5 was worse in Ho and Salimans' ablation"}
  - {name: "(sigma_lo, sigma_hi]", meaning: "the interval of noise levels on which guidance is applied; outside it the conditional model runs alone", typical: "everywhere, (0, inf), in the original method; (0.28, 2.90] for EDM2-S on ImageNet-512"}
  - {name: phi, meaning: "the rescale factor of eq-rescale; 0 is plain guidance", typical: "0.7, or 0.5 to 0.75"}
  - {name: p, meaning: "the percentile of dynamic thresholding, eq-dyn", typical: "99.5 in Imagen"}
cost:
  train: "the same loss and network as a conditional diffusion model, plus one embedding row for the null condition; no classifier is trained"
  inference: "two network evaluations per guided sampling step, usually as one batch of two; N + k evaluations for N steps with k of them inside a guidance interval"
history:
  year: 2021
  people: [Jonathan Ho, Tim Salimans, Prafulla Dhariwal, Alex Nichol]
  note: "Dhariwal and Nichol (2021) introduced classifier guidance with a classifier trained on noisy images; Ho and Salimans replaced the classifier by the difference of a conditional and an unconditional model trained as one network, first at the NeurIPS 2021 Workshop on Deep Generative Models and then as arXiv 2207.12598 (2022). GLIDE (2021) used it for text, and it became standard in Imagen, Stable Diffusion and later text-to-image models."
prereqs: [diffusion-model, score@m2, bayes-theorem@m2, normal-distribution@m2]
same_as:
  arxiv: ["2207.12598"]
sources:
  - {paper: "Ho and Salimans (2022)", title: "Classifier-Free Diffusion Guidance", locator: "arXiv 2207.12598v1, pp. 3 to 9", note: "the method, the implicit classifier, joint training, the ImageNet results, the p_uncond ablation, the cost and the discussion"}
  - {paper: "Dhariwal and Nichol (2021)", title: "Diffusion Models Beat GANs on Image Synthesis", locator: "arXiv 2105.05233v4, pp. 1, 6 to 8", note: "classifier guidance, its derivation and scale, and the precision and recall trade-off"}
  - {paper: "Nichol, Dhariwal, Ramesh, Shyam, Mishkin, McGrew, Sutskever and Chen (2021)", title: "GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models", locator: "arXiv 2112.10741v3, pp. 5, 8 to 9", note: "guidance for text with empty captions, and the comparison with CLIP guidance"}
  - {paper: "Saharia, Chan, Saxena, Li, Whang, Denton, Ghasemipour, Ayan, Mahdavi, Lopes, Salimans, Ho, Fleet and Norouzi (2022)", title: "Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding", locator: "arXiv 2205.11487v1, pp. 4, 42 to 43", note: "Imagen's guidance weight, the saturation of large weights, and static and dynamic thresholding"}
  - {paper: "Lin, Liu, Li and Yang (2023)", title: "Common Diffusion Noise Schedules and Sample Steps are Flawed", locator: "arXiv 2305.08891v4, pp. 4, 6", note: "guidance rescale and its factor"}
  - {paper: "Kynkäänniemi, Aittala, Karras, Laine, Aila and Lehtinen (2024)", title: "Applying Guidance in a Limited Interval Improves Sample and Distribution Quality in Diffusion Models", locator: "arXiv 2404.07724v2, pp. 1 to 4", note: "the guidance interval and the mode drop of guidance at high noise"}
  - {paper: "Karras, Aittala, Kynkäänniemi, Lehtinen, Aila and Laine (2024)", title: "Guiding a Diffusion Model with a Bad Version of Itself", locator: "arXiv 2406.02507v3, pp. 1 to 6", note: "why the guided ODE does not sample the guided density, and autoguidance"}
  - {paper: "Song, Sohl-Dickstein, Kingma, Kumar, Ermon and Poole (2021)", title: "Score-Based Generative Modeling through Stochastic Differential Equations", locator: "arXiv 2011.13456v2, p. 9", note: "conditional generation with the score of p(y | x_t) added to the unconditional score"}
---

Classifier-free guidance is how almost every conditional [[diffusion-model|diffusion model]] is sampled.
One network is trained to predict the noise both with its condition, a class or a caption, and with the condition replaced by a null token.
At each sampling step it is run both ways, and the step uses the unconditional prediction plus $s$ times the difference between the two; with $s > 1$ the sampler is pushed further towards what makes the sample fit its condition than the conditional model alone would go.
Samples become sharper, more typical of their class and more faithful to their prompt, and less varied.
The method replaced classifier guidance, which needed a separate classifier trained on noisy images, and its scale is the "guidance scale" or "CFG" setting of every text-to-image system built on the [[latent-diffusion-model|latent diffusion model]].

## In plain words

Imagine an artist who can paint either "anything" or "a dog".
Ask for a dog and you get a dog, but often a vague one, half like any animal, because the artist hedges.
Classifier-free guidance asks the artist for both pictures at every brushstroke and looks at the difference: whatever the "dog" painting has that the "anything" painting lacks is what makes it a dog.
It then exaggerates that difference, say three times, and paints with the exaggerated version.
The result is unmistakably a dog, with crisp details, but the dogs come out more alike, as if the artist had been told to paint the most dog-like dog each time.
Push the exaggeration too far and the colours burn out and the shapes become caricatures.
The trick needs only one artist: during training the artist is sometimes told the subject and sometimes not, so the same skills serve both questions, and nobody has to train a separate critic that judges whether a half-finished painting looks like a dog.

## Setting

A diffusion model corrupts data $x$ into $z_t = \sqrt{\bar\alpha_t}\,x + \sqrt{1 - \bar\alpha_t}\,\epsilon$ with $\epsilon \sim \mathcal{N}(0, I)$, and learns a noise predictor $\epsilon_\theta(z_t, t, c)$ for a condition $c$.
The best such predictor is tied to the [[m2:score|score]] of the noisy distribution, $\epsilon(z_t, c) = -\sqrt{1 - \bar\alpha_t}\,\nabla_{z_t} \log p_t(z_t \mid c)$, and sampling follows the scores from noise to data.
Sampling from $p(x \mid c)$ exactly is often not what one wants: the conditional model spreads its mass over everything that is plausible for $c$, including blurry or ambiguous samples, and the useful trade is to give up some diversity for samples that are clearly of class $c$.
[[m2:bayes-theorem|Bayes' theorem]] splits the conditional score into the unconditional score plus the score of a classifier, $\nabla \log p_t(z_t \mid c) = \nabla \log p_t(z_t) + \nabla \log p_t(c \mid z_t)$, and turning up the classifier term is guidance.
Song and colleagues (2021) added the classifier's score to the unconditional score for conditional generation, and Dhariwal and Nichol (2021) introduced a scale on it.

## The method

### Classifier guidance

Dhariwal and Nichol train a classifier $p_\phi(y \mid x_t, t)$ on noisy images and use its gradients to guide sampling towards an arbitrary class $y$, a simple, compute-efficient method for trading off diversity for fidelity. {#c-cg-def}
For the noise predictor, their guided prediction with scale $s$ is

$$ \hat\epsilon(z_t, c) = \epsilon_\theta(z_t) - s\,\sqrt{1 - \bar\alpha_t}\;\nabla_{z_t} \log p_\phi(c \mid z_t) , $$ {#eq-cg}

and for an ancestral step with Gaussian $\mathcal{N}(\mu, \Sigma)$ it shifts the mean to $\mu + s\Sigma\nabla \log p_\phi(c \mid \mu)$.
That shift is exact when the log-classifier is linear around $\mu$: $\mathcal{N}(\mu, \Sigma)\,p_\phi(y \mid x_t)^s$ is then proportional to $\mathcal{N}(\mu + s\Sigma g, \Sigma)$, Dhariwal and Nichol's Algorithm 1. {#c-perturbed-mean}
The scale matters: at $s = 1$ the classifier gave the desired classes reasonable probabilities, around 50%, yet the samples did not look like the intended classes, so the gradients had to be scaled by a factor above 1, and since $s\nabla \log p(y \mid x) = \nabla \log \frac{1}{Z} p(y \mid x)^s$, a larger scale focuses on the modes of the classifier, for higher fidelity and less diversity. {#c-cg-scale}
The cost is a second model, a classifier that must work on noisy inputs at every noise level, so off-the-shelf classifiers do not serve.

### Classifier-free guidance

Ho and Salimans drop the classifier.
They train one network as both the conditional and the unconditional model by replacing $c$ with a null token $\varnothing$ with probability $p_{\text{uncond}}$, and sample with $(1 + w)\epsilon_\theta(z_t, c) - w\,\epsilon_\theta(z_t, \varnothing)$; joint training is simple to implement and adds no parameters. {#c-cfg-def}
Writing $s = 1 + w$, the guided prediction is

$$ \tilde\epsilon_s(z_t, c) = \epsilon_\theta(z_t, \varnothing) + s\,\big(\epsilon_\theta(z_t, c) - \epsilon_\theta(z_t, \varnothing)\big) . $$ {#eq-cfg}

The form is inspired by an implicit classifier $p^i(c \mid z) \propto p(z \mid c)/p(z)$; no classifier gradient is used, so the method cannot be read as an adversarial attack on a classifier, and because the network's score estimates are not in general conservative vector fields, the guided estimate is not in general the gradient of any classifier. {#c-cfg-implicit}
Training is Algorithm 1 of Ho and Salimans: draw $(x, c)$, replace $c$ by $\varnothing$ with probability $p_{\text{uncond}}$, draw $t$ and $\epsilon$, and take a gradient step on $\|\epsilon_\theta(z_t, t, c) - \epsilon\|^2$.
Sampling is any diffusion sampler with $\tilde\epsilon_s$ in place of $\epsilon_\theta$.

The literature uses three conventions for the same thing.
Ho and Salimans' weight $w$ is $s - 1$; GLIDE, Imagen and Karras and colleagues write $s$ or $w$ for the $s$ of eq-cfg; and because the conversions from $\epsilon$ to $\hat x_0$, to $v$ and to the score are affine with weights summing to 1, guiding any of these outputs gives the same step, while the rescale below does not commute with the conversion. {#c-conventions}
A guidance weight of 7.5 in Imagen's form is $w = 6.5$ in Ho and Salimans' form.
The derivation `guidance-conventions` proves this and `test_conventions` and `test_parameterization_invariance` check it.

GLIDE carried the method over to text: captions are sometimes replaced with an empty sequence during training, and guidance uses $\epsilon(x_t \mid \varnothing) + s(\epsilon(x_t \mid c) - \epsilon(x_t \mid \varnothing))$ with $s \ge 1$; the single model uses its own knowledge rather than a separate classifier's, and guidance becomes simple for conditions such as text that are hard to predict with a classifier. {#c-glide}
Imagen writes the same thing as $w\epsilon_\theta(z_t, c) + (1 - w)\epsilon_\theta(z_t)$, drops $c$ with probability for example 10%, notes that $w = 1$ disables guidance, and depends critically on it for text conditioning. {#c-imagen-weight}
A "negative prompt" in text-to-image software is the same formula with the null token replaced by the embedding of a prompt to steer away from.

## Derivation

### Guidance is the score of a tilted density

Suppose the network were exact, so that $\epsilon(z_t, c)$ and $\epsilon(z_t, \varnothing)$ are $-\sigma_t$ times the scores of $p_t(z_t \mid c)$ and $p_t(z_t)$, with $\sigma_t = \sqrt{1 - \bar\alpha_t}$.
Eq-cfg is a linear combination of the two scores with weights $s$ and $1 - s$, so

$$ \tilde\epsilon_s = -\sigma_t \nabla_{z_t} \log \tilde p_s(z_t \mid c) , \qquad \tilde p_s(z_t \mid c) \propto p_t(z_t \mid c)^s\, p_t(z_t)^{1 - s} \propto p_t(z_t)\, p_t(c \mid z_t)^s , $$ {#eq-tilt}

where the second form uses $p_t(z_t \mid c) = p_t(c \mid z_t)\,p_t(z_t)/p(c)$.
With exact noise predictors the guided noise is $-\sigma_t$ times the score of $p_t(z_t \mid c)^s p_t(z_t)^{1 - s}$, which is proportional to $p_t(z_t)\,p_t(c \mid z_t)^s$. {#c-tilted-score}
Taking the gradient of the logarithm of the second form gives classifier guidance with the exact classifier, and of $p_t(z_t \mid c)\,p_t(c \mid z_t)^{s - 1}$ the same with scale $s - 1$ on the conditional model: with exact models, classifier-free guidance at scale $s$ equals classifier guidance with the exact noisy classifier at scale $s$ on the unconditional model and $s - 1$ on the conditional one. {#c-cg-equivalence}
This is the "implicit classifier" of Ho and Salimans made exact.
The derivation `implicit-classifier` proves both statements, and `test_guided_noise_is_tilted_score` checks them with autograd and `jax.grad` on a [[al:gaussian-mixture-model|Gaussian mixture]], where every quantity is known: the class posterior $p_t(c \mid z_t)$ of a mixture is a softmax of quadratic scores, a [[al:logistic-regression|logistic regression]] in $z_t$ when the classes share a variance.

### What the tilted density looks like

For one-dimensional [[m2:normal-distribution|normal distributions]] the tilted density is again normal.
With $p_1 = \mathcal{N}(m_1, v_1)$, the class, and $p_0 = \mathcal{N}(m_0, v_0)$, the unconditional distribution,

$$ p_1^s\, p_0^{1-s} \propto \mathcal{N}(m_s, v_s) , \qquad \frac{1}{v_s} = \frac{s}{v_1} + \frac{1 - s}{v_0} , \qquad m_s = m_1 + (1 - s)\,\frac{v_s}{v_0}\,(m_0 - m_1) , $$

which is integrable exactly when $s/v_1 + (1 - s)/v_0 > 0$; when $v_1 \le v_0$, as for a class inside a wider unconditional distribution, it is integrable for every $s \ge 0$, and for $s > 1$ its variance is below $v_1$ and its mean lies beyond $m_1$, on the side away from $m_0$. {#c-tilted-gaussian}
Guidance therefore both narrows the class and pushes it away from the other classes, the two effects behind higher fidelity and lower diversity.
With $m_1 = 1$, $v_1 = 0.25$, $m_0 = 0$ and $v_0 = 1.25$, the tilted density is $\mathcal{N}(1.111, 0.139)$ at $s = 2$ and $\mathcal{N}(1.154, 0.096)$ at $s = 3$.
The derivation `tilted-gaussian` proves the formula.

### The sampler does not sample the tilted density

At every noise level the guided score is the score of a tilted density, but the tilted densities at different noise levels are not the noised versions of one distribution, so following them does not end at $\tilde p_s(x \mid c)$.
Karras and colleagues put it this way: $p_w(x \mid c; \sigma)$ is not a valid heat diffusion of $p_w(x \mid c; 0)$, so the ODE does not follow the density; samples are pushed towards higher values of the implied density at each noise level, which can distort trajectories, exaggerate truncation, drop modes and over-saturate colours. {#c-not-diffusion}
The two Gaussians make this exact.
In $x = z_t/\sqrt{\bar\alpha_t}$ and $\sigma = \sqrt{(1 - \bar\alpha_t)/\bar\alpha_t}$, deterministic DDIM is the Euler method for the probability-flow ODE, a limit of the reverse [[m2:stochastic-differential-equation|stochastic differential equation]], and solving it with the guided predictor gives

$$ \mathrm{Var}(x_0) = (1 + \sigma_T^2)\Big(\frac{v_1}{v_1 + \sigma_T^2}\Big)^{s}\Big(\frac{v_0}{v_0 + \sigma_T^2}\Big)^{1-s} \;\longrightarrow\; v_1^{\,s}\, v_0^{\,1 - s} , $$ {#eq-ode-var}

and for $s > 1$ this is smaller than the variance of the tilted density, so the guided ODE does not sample it. {#c-ode-variance}
In the example above the guided ODE ends with variance 0.05 and mean 1.32 at $s = 2$, against 0.139 and 1.11 for the tilted density, and with 0.01 and 1.40 at $s = 3$, against 0.096 and 1.15.
The [[m2:variance|variance]] $v_1^s v_0^{1 - s}$ falls geometrically in $s$, which is the quantitative form of "large scales collapse diversity".
The derivation `guided-ode-variance` proves eq-ode-var, and `test_guided_ode_variance` runs 1000 DDIM steps in all three implementations and finds it within 3%.

## Worked example

Take two classes on the line with means $+1$ and $-1$, variance $1/4$ each, and equal weights, and a noisy sample $z = 0.5$ at $\bar\alpha_t = 0.5$.
The noisy classes are $\mathcal{N}(\pm\sqrt{0.5}, v)$ with $v = 0.5 \cdot 0.25 + 0.5 = 0.625$.
The exact classifier gives $p(+1 \mid z) = 0.7561$.
The conditional noise for class $+1$ is $\epsilon_c = \sqrt{0.5}\,(0.5 - 0.7071)/0.625 = -0.2343$, and the unconditional noise, the posterior-weighted average of the two classes' noises, is $\epsilon_\varnothing = 0.1559$.
Guidance at $s = 3$ gives $\tilde\epsilon = 0.1559 + 3(-0.2343 - 0.1559) = -1.0148$.
The three predictions of the clean sample, $\hat x_0 = (z - \sqrt{0.5}\,\epsilon)/\sqrt{0.5}$, are 0.9414 for the conditional model, 0.5512 for the unconditional model and 1.7219 for the guided one.
The guided prediction overshoots the class mean of 1; this overshoot, repeated over many steps, is what thresholding and rescaling are designed to contain.
The gradient of the log-classifier at $z$ is 0.5519, and eq-cg with $s = 3$ on the unconditional noise gives $0.1559 - 3\sqrt{0.5} \cdot 0.5519 = -1.0148$, the same value, as c-cg-equivalence says.
`test_worked_example` checks these numbers.

## Hyperparameters

The scale $s$ trades diversity for fidelity and alignment.
On ImageNet, Ho and Salimans found FID lowest with a small amount of guidance, $w = 0.1$ or $0.3$ ($s = 1.1$ or $1.3$), and the Inception score highest with $w \ge 4$, with a trade-off in between; text-to-image systems use much larger values.
The probability $p_{\text{uncond}}$ sets how much training goes to the null model: of $\{0.1, 0.2, 0.5\}$, 0.5 was consistently worse than 0.1 and 0.2, so only a small part of the model's capacity needs to serve unconditional generation. {#c-cfg-puncond}
The guidance interval $(\sigma_{\text{lo}}, \sigma_{\text{hi}}]$ restricts where guidance acts; it is best treated as a hyperparameter of its own, tuned with the scale, and its ends should fall on the sampler's steps so that the scale is constant within each step.
The rescale factor $\phi$ and the dynamic-thresholding percentile $p$ only matter for large scales; Lin and colleagues found $\phi$ between 0.5 and 0.75 best, and Imagen used $p = 99.5$.
The number of sampling steps matters too: Ho and Salimans found $T = 256$ a good balance, and each step costs two network evaluations.

## Cost

Training costs nothing extra: the same loss and network as a conditional model, with one more row in the class or text embedding for $\varnothing$.
Sampling costs two network evaluations per guided step instead of one; implementations batch the conditional and the unconditional input together, which doubles the batch rather than the latency on hardware with spare parallelism.
Each sampling step evaluates the model twice, and Ho and Salimans name sampling speed as a potential disadvantage, since classifier guidance can use a smaller, faster classifier. {#c-cfg-speed}
For the denoiser of this page, with $D$ inputs, $H$ hidden units, a time embedding of width $E$ and $C$ classes plus the null class, the model has $P = DH + EH + (C + 1)H + H + HD + D$ parameters and costs $C_f = 2B(DH + EH + HD)$ FLOPs per evaluation of a batch of $B$, a multiply-add counted as 2; $N$ steps cost $NC_f$ unguided, $2NC_f$ guided at every step and $(N + k)C_f$ guided on an interval that contains $k$ steps; with 32 DDIM steps of the linear schedule and the interval $(0.28, 2.90]$, 12 steps fall inside, so 44 evaluations replace 64. {#c-sampling-cost}
The derivation `sampling-cost` proves the counts and `test_counts` and `test_interval_count` check them against the shape contract in `shapes.yaml`.

## Evidence

Dhariwal and Nichol's classifier guidance came first: for the class Pembroke Welsh corgi, scale 1.0 gave an FID of 33.0 and scale 10.0 much more class-consistent samples with FID 12.0. {#c-cg-corgi}
Classifier guidance improved precision at the cost of recall, and with it diffusion models reached an FID of 2.97 on ImageNet 128 × 128 and matched BigGAN-deep with as few as 25 forward passes per sample. {#c-cg-precision}
Ho and Salimans found on ImageNet that FID was lowest with a small amount of guidance ($w = 0.1$ or $0.3$) and the Inception score highest with strong guidance ($w \ge 4$); at $w = 0.3$ the 128 × 128 FID outperforms the classifier-guided ADM-G, and between the extremes there is a trade-off of decreasing sample variety and increasing individual fidelity. {#c-cfg-results}
GLIDE found that increasing the scale traded diversity for fidelity, that classifier-free guidance was nearly Pareto optimal against CLIP guidance on MS-COCO 64 × 64, and that human evaluators, disagreeing with the CLIP score, judged its samples higher in quality and closer to the caption; its MS-COCO 256 × 256 FID used scale 1.5, which gave the best FID. {#c-glide-pareto}
These are the comparisons of the original papers on their own models; the gains from guidance in later systems are much larger, at scales far above the ones that optimise FID.
The limited-interval and autoguidance results below are the strongest later evidence that the trade-off of plain guidance is not intrinsic.

## Limitations and failure modes

**Diversity.** Guidance makes samples less varied by construction, eq-ode-var. Ho and Salimans' intuition is that guidance decreases the unconditional likelihood of the sample while increasing the conditional likelihood, which classifier-free guidance does with a negative score term; it raises fidelity at the expense of diversity, and whether decreased diversity is acceptable is a real question for deployed models. {#c-cfg-intuition}

**Saturation.** Large weights improve image-text alignment but damage fidelity with highly saturated, unnatural images; Imagen traces this to a train-test mismatch, since x-predictions exceed the data range $[-1, 1]$ under large weights, and because the model is applied to its own output, sampling sometimes diverges. {#c-imagen-saturation}
The worked example's guided $\hat x_0 = 1.72$ for a class at 1 is this mismatch in miniature.

**Over-exposure.** With large $w$ the guided output is much larger in scale than any training output, which over-exposes images; Lin and colleagues rescale the guided output to the standard deviation of the conditional output and mix the result with the plain guided output with a factor $\phi$, with $w = 7.5$ and $\phi = 0.7$ working well. {#c-lin-rescale}

**Mode dropping at high noise.** Guidance is harmful at the beginning of the chain, at high noise, largely unnecessary at the end and only beneficial in the middle. {#c-interval}
In a two-dimensional toy example, guidance with weight 6 at every noise level dropped one mode almost entirely, and disabling it at high $\sigma$ restored both modes. {#c-interval-modedrop}

**Two evaluations.** Sampling is twice as expensive as unguided sampling, which matters most for few-step samplers.

**The null model must exist.** The unconditional output is only meaningful if training dropped the condition; a model trained without it extrapolates from an output it never learned.

## Variants

**Classifier guidance**, eq-cg, is the predecessor and still useful when the condition is a property that a separate differentiable model can score but the generator was not trained on.

**Static and dynamic thresholding.** Imagen clips the x-prediction elementwise to $[-1, 1]$, static thresholding, which is essential with large weights but still gives over-saturated, less detailed images; dynamic thresholding sets $s$ to a percentile of $|\hat x_0|$ and, if $s > 1$, clips to $[-s, s]$ and divides by $s$, which gave better photorealism and alignment at very large weights. {#c-imagen-threshold}
In formulas,

$$ s = \max\big(\mathrm{perc}_p(|\hat x_0|),\, 1\big) , \qquad \hat x_0 \leftarrow \mathrm{clip}(\hat x_0, -s, s)/s , $$ {#eq-dyn}

where $\mathrm{perc}_p$ is the empirical [[m2:quantile-function|quantile]] over the coordinates of one sample.
Static thresholding is the Euclidean [[m2:orthogonal-projection|projection]] of the x-prediction onto the cube $[-1, 1]^n$; dynamic thresholding equals $\mathrm{clip}(\hat x/s, -1, 1)$, maps into the same cube, equals static thresholding when the percentile is at most 1, and leaves at least $\lfloor (p/100)(n - 1)\rfloor + 1$ coordinates unclipped, all divided by the same $s$. {#c-threshold-bounds}
The derivation `thresholding` proves this; thresholding only applies to models whose data have known bounds, such as pixels, and not to latents.

**Guidance rescale.** Lin and colleagues' rescale is

$$ x_{\text{final}} = \phi\, x_{\text{cfg}}\,\frac{\sigma_{\text{pos}}}{\sigma_{\text{cfg}}} + (1 - \phi)\, x_{\text{cfg}} , $$ {#eq-rescale}

with $\sigma$ the [[m2:standard-deviation|standard deviation]] over the elements of each sample.
Their Eqs. 15 and 16 equal the single multiplication $x_{\text{cfg}}(\phi\sigma_{\text{pos}}/\sigma_{\text{cfg}} + 1 - \phi)$ of their Algorithm 2, the factor does not depend on whether the standard deviation divides by $n$ or $n - 1$, and the output has standard deviation $\phi\sigma_{\text{pos}} + (1 - \phi)\sigma_{\text{cfg}}$. {#c-rescale-std}
With $\phi = 0$, plain guidance, images tend to over-expose, and $\phi$ between 0.5 and 0.75 gave the most appealing results. {#c-lin-phi}
The derivation `rescale` proves the identities.

**Guidance interval.** Kynkäänniemi and colleagues apply the scale only for $\sigma \in (\sigma_{\text{lo}}, \sigma_{\text{hi}}]$ and use $s = 1$ elsewhere, which improved the record FID on ImageNet-512 from 1.81 to 1.40 and saves the null evaluations outside the interval.

**Autoguidance.** Karras and colleagues guide a model with a smaller, less-trained version of itself instead of an unconditional model; this controls image quality without reducing variation, works for unconditional models too, and improved the ImageNet-512 FID of EDM2-S from 2.56 to 1.34, and to 1.25 with EDM2-XXL. {#c-autoguidance}
The guiding model is trained on the same condition, so autoguidance improves quality rather than alignment.

**Other generators.** The same extrapolation applies to any model with a conditional and an unconditional velocity or score, such as [[flow-matching]] models, and, on logits instead of scores, to autoregressive models of image or text tokens.
Combining scores of several conditions, $\epsilon_\varnothing + \sum_i s_i(\epsilon_{c_i} - \epsilon_\varnothing)$, composes concepts as a product of experts, the view of an [[energy-based-model|energy-based model]] in which log densities add.
It plays the role that the truncation trick plays in a [[generative-adversarial-network|generative adversarial network]]: a sampling-time knob that trades variety for quality.

## When to use it

Use it for any conditional diffusion or flow model whose samples are judged by fidelity or prompt alignment: text-to-image, class-conditional generation, text-to-audio.
Train with $p_{\text{uncond}}$ around 0.1 from the start, since adding a null token later needs fine-tuning.
Choose the scale by the metric you care about: a small scale near the FID optimum for distribution matching, a large one for alignment and visual punch.
With large scales, add an interval, rescale or thresholding; with a compute budget for a second small model, consider autoguidance.
Do not use it when diversity or calibrated likelihoods are the goal, such as generating training data that must cover the conditional distribution, or density estimation.

## Pitfalls

> [!pitfall]
> Conventions differ by one. Ho and Salimans' $w$ is $s - 1$ of eq-cfg, and $w = 0$ is the plain conditional model, while in Imagen's or Stable Diffusion's form $w = 1$ is. Check the formula before copying a scale between papers or codebases.

> [!pitfall]
> Imagen's reference listing writes `s = jnp.max(s, 1.0)`. JAX and NumPy read the second argument as an axis and raise a `TypeError`; the operation the text describes is the elementwise `jnp.maximum(s, 1.0)`, and the percentile must keep its dimensions to broadcast over the sample.

> [!pitfall]
> The null token must be trained. A model trained without dropping the condition has never seen $\varnothing$, and eq-cfg then extrapolates from an untrained output.

> [!pitfall]
> The guided sampler does not sample $p(x \mid c)^s p(x)^{1 - s}$, eq-ode-var. Do not use guided samples as if they came from a sharpened but correct distribution, for example to estimate class frequencies or likelihoods.

> [!pitfall]
> The rescale depends on what is rescaled. Lin and colleagues apply it to the outputs of a model that predicts $v$; applying the same $\phi$ to the noise or to $\hat x_0$ gives a different sampler, unlike eq-cfg itself, which commutes with the conversion.

> [!pitfall]
> Guidance at high noise drops modes. With a large scale and a full-range interval, whole compositions or classes of images can vanish from the output without any visible artefact in single samples; compare sample sets, not single images.

## Implementations

The NumPy reference `impl/numpy/classifier_free_guidance.py` implements:

- the exact models of a [[m2:mixture-model|mixture]] of [[m2:multivariate-normal-distribution|Gaussians]], one per class: the conditional and unconditional noise predictors, the log density and the noisy classifier $\log p(c \mid z_t)$ by [[m2:conditional-probability|Bayes' rule]], and its gradient;
- guidance by eq-cfg and Ho and Salimans' form, classifier guidance by eq-cg and the perturbed mean of Algorithm 1, the conversions between noise, $\hat x_0$ and $v$, the rescale of eq-rescale, static and dynamic thresholding by eq-dyn, and the guidance interval;
- the tilted Gaussian of eq-tilt and the ODE variance of eq-ode-var;
- deterministic DDIM with a guided predictor that counts network evaluations;
- the denoiser of the shape contract, its loss gradient written by hand, [[adam|Adam]], and the joint training of Algorithm 1 with the condition dropped with probability $p_{\text{uncond}}$.

The PyTorch version `impl/pytorch/classifier_free_guidance.py` has the denoiser as an `nn.Module` with a class embedding of $C + 1$ rows, the classifier gradient and the training gradients from autograd by [[al:backpropagation|backpropagation]], `torch.autograd.gradcheck` through the denoiser, `torch.quantile` for dynamic thresholding and training with `torch.optim.Adam`.
The JAX version `impl/jax/classifier_free_guidance.py` is pure functions: the classifier gradient by `jax.vmap(jax.grad(...))`, DDIM as a `jax.lax.scan` whose guided predictor uses `jax.lax.cond` to skip the null model outside the interval and counts evaluations inside the scan, and training as a jitted scan of Adam steps with `jax.random` keys.
All three load the shared fixture of `fixtures/make.py` and check the predicted noise, the loss and its gradient in every parameter, the guided noise at $s = 3$, its rescale at $\phi = 0.7$, dynamic thresholding at the 90th percentile, and a five-step DDIM sample guided at $s = 2$ in the interval $(0.3, 3.0]$ with its number of evaluations; `test_counts` checks the parameter and FLOP counts of `fixtures/shapes.json` and the output shapes.

## Experiments

This page has no experiment yet.
Some tests are small experiments.

`test_training_and_guided_sampling` trains the denoiser with [[al:ancestral-sampling|ancestral]] noising and dropped classes on three Gaussian classes in the plane for 2500 Adam steps and samples 3000 points with 25 DDIM steps: unguided, the null model covers all three classes; at $s = 1$ the samples of class 1 centre on its mean, and at $s = 3$ the exact classifier's mean confidence in class 1 rises above 0.97.
`test_tradeoff_on_overlapping_classes` guides the exact models of two overlapping classes: the classifier's confidence in the samples grows from 0.5 at $s = 0$ through $s = 1$ to above 0.9 at $s = 3$, the spread along the axis that separates the classes falls by more than a fifth, and the spread across it, where the classes agree, stays the class's own.
`test_guided_ode_variance` and `test_tilted_gaussian_and_guided_ode` run the guided ODE on two Gaussians and compare its end variance with eq-ode-var and with the tilted density.
`test_interval_and_count` counts 44 evaluations for 32 steps with the interval $(0.28, 2.90]$, against 64 without.

The checks of the eight derivations compare:

- the three conventions and the conversions to $\hat x_0$, $v$ and the score;
- the guided noise with the score of the tilted density, and with classifier guidance at $s$ and $s - 1$;
- the perturbed mean with the normalized product of the Gaussian step and an exponential classifier;
- the tilted Gaussian with numerical integration, including where it stops being integrable;
- eq-ode-var with a fine Runge-Kutta solution and with 1000 DDIM steps;
- the rescale identities for both standard deviations;
- static thresholding with a brute-force nearest point and dynamic thresholding with its bounds, and that Imagen's listing raises a `TypeError`;
- the parameter, FLOP and evaluation counts with the code.

## History

Song and colleagues (2021) wrote conditional generation from an unconditional score model as the unconditional score plus the gradient of a classifier on noisy inputs.
Dhariwal and Nichol (2021) scaled that gradient, called it classifier guidance, and with it made diffusion models beat GANs on ImageNet, at the cost of training a noise-aware classifier.
Ho and Salimans presented classifier-free guidance at the NeurIPS 2021 Workshop on Deep Generative Models, posted to arXiv in 2022: the classifier became the difference of a conditional and an unconditional model trained as one network.
GLIDE (December 2021) used it with text and found it preferred to CLIP guidance, and it became standard in Imagen and Stable Diffusion, which built on the [[latent-diffusion-model|latent diffusion model]].
Imagen (2022) introduced dynamic thresholding to allow large scales, Lin and colleagues (2023) the rescale, and the "negative prompt" of open-source tools turned the null token into a second prompt.
In 2024 Kynkäänniemi and colleagues restricted guidance to an interval of noise levels, and Karras and colleagues replaced the unconditional model with a weaker copy of the model itself, autoguidance, separating the gain in quality from the loss of variety.

## References

- Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon and B. Poole. Score-Based Generative Modeling through Stochastic Differential Equations. ICLR 2021; arXiv 2011.13456.
- P. Dhariwal and A. Nichol. Diffusion Models Beat GANs on Image Synthesis. 2021; arXiv 2105.05233.
- J. Ho and T. Salimans. Classifier-Free Diffusion Guidance. NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications; arXiv 2207.12598.
- A. Nichol, P. Dhariwal, A. Ramesh, P. Shyam, P. Mishkin, B. McGrew, I. Sutskever and M. Chen. GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models. 2021; arXiv 2112.10741.
- C. Saharia, W. Chan, S. Saxena, L. Li, J. Whang, E. Denton, S. K. S. Ghasemipour, B. K. Ayan, S. S. Mahdavi, R. G. Lopes, T. Salimans, J. Ho, D. J. Fleet and M. Norouzi. Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding. 2022; arXiv 2205.11487.
- S. Lin, B. Liu, J. Li and X. Yang. Common Diffusion Noise Schedules and Sample Steps are Flawed. 2023; arXiv 2305.08891.
- T. Kynkäänniemi, M. Aittala, T. Karras, S. Laine, T. Aila and J. Lehtinen. Applying Guidance in a Limited Interval Improves Sample and Distribution Quality in Diffusion Models. NeurIPS 2024; arXiv 2404.07724.
- T. Karras, M. Aittala, T. Kynkäänniemi, J. Lehtinen, T. Aila and S. Laine. Guiding a Diffusion Model with a Bad Version of Itself. NeurIPS 2024; arXiv 2406.02507.
