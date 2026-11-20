---
tag: NEW-consistency-model
kind: architecture
slug: consistency-model
name: Consistency model
aliases: [consistency models, CM, consistency distillation, CD, consistency training, CT, improved consistency training, iCT, latent consistency model, LCM, continuous-time consistency model, sCM, one-step diffusion, few-step diffusion, diffusion distillation, progressive distillation, denoising student, consistency function]
area: generative-models/consistency-and-distillation-of-samplers
level: core
summary: >-
  A network that maps any noisy point on a diffusion model's probability flow path straight to the path's clean end, so it samples in one step, or a few, instead of dozens.
uses: [stochastic-differential-equation@m2, score@m2, normal-distribution@m2, multivariate-normal-distribution@m2, sum-of-independent-normal-random-variables@m2, mixture-model@m2, conditional-expectation@m2, bayes-theorem@m2, expected-value@m2, variance@m2, loss-function@m2, huber-loss@m2, robust-statistics@m2, kullback-leibler-divergence@m2, stochastic-gradient-descent@m2, automatic-differentiation@al, knowledge-distillation@al, gaussian-mixture-model@al]
related_to: [diffusion-model, classifier-free-guidance, latent-diffusion-model, flow-matching, generative-adversarial-network, normalizing-flow, variational-autoencoder, residual-connection, adam]
history:
  year: 2023
  people: [Yang Song, Prafulla Dhariwal, Mark Chen, Ilya Sutskever, Tim Salimans, Jonathan Ho, Eric Luhman, Troy Luhman, Cheng Lu]
  note: "Luhman and Luhman (2021) trained a one-step student on the outputs of a deterministic diffusion sampler, and Salimans and Ho (2022) halved the steps of a sampler again and again by progressive distillation. Song, Dhariwal, Chen and Sutskever (ICML 2023) defined consistency models, which map every point of a probability flow path to its origin, and trained them by distillation or from data alone; Song and Dhariwal (2023) fixed the theory and recipe of training from data, Luo and colleagues (2023) moved the method into the latent space of Stable Diffusion, and Lu and Song (2024) made the continuous-time version stable at the scale of 1.5 billion parameters."
prereqs: [diffusion-model, score@m2, stochastic-differential-equation@m2, normal-distribution@m2, knowledge-distillation@al]
sources:
  - {paper: "Song, Dhariwal, Chen and Sutskever (2023)", title: "Consistency Models", locator: "ICML 2023; arXiv 2303.01469v2, pp. 1 to 8, 25 to 26", note: "the consistency function, the boundary condition and its parameterization, multistep sampling, distillation and training with their theorems, the EMA target, the experiments and the schedules"}
  - {paper: "Song and Dhariwal (2023)", title: "Improved Techniques for Training Consistency Models", locator: "arXiv 2310.14189v1, pp. 1, 4 to 9", note: "the teacher without EMA and Proposition 1, the Pseudo-Huber metric, the doubling curriculum, the lognormal noise schedule, the weighting and the results"}
  - {paper: "Salimans and Ho (2022)", title: "Progressive Distillation for Fast Sampling of Diffusion Models", locator: "ICLR 2022; arXiv 2202.00512v2, pp. 1, 3 to 5", note: "the halving of sampling steps, the target of Algorithm 2, and the x and v parameterizations"}
  - {paper: "Luhman and Luhman (2021)", title: "Knowledge Distillation in Iterative Generative Models for Improved Sampling Speed", locator: "arXiv 2101.02388v1, pp. 1 to 2, 7", note: "the denoising student trained on the outputs of DDIM"}
  - {paper: "Karras, Aittala, Aila and Laine (2022)", title: "Elucidating the Design Space of Diffusion-Based Generative Models", locator: "NeurIPS 2022; arXiv 2206.00364v2, pp. 1 to 2, 9", note: "the noise levels, the preconditioning with sigma_data, Heun's sampler and its 35 evaluations"}
  - {paper: "Meng, Rombach, Gao, Kingma, Ermon, Ho and Salimans (2022)", title: "On Distillation of Guided Diffusion Models", locator: "arXiv 2210.03142v3, pp. 1 to 2", note: "two-stage distillation of classifier-free guided models"}
  - {paper: "Luo, Tan, Huang, Li and Zhao (2023)", title: "Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference", locator: "arXiv 2310.04378v1, p. 1", note: "consistency distillation in the latent space of Stable Diffusion with guidance folded into the ODE"}
  - {paper: "Lu and Song (2024)", title: "Simplifying, Stabilizing and Scaling Continuous-Time Consistency Models", locator: "ICLR 2025; arXiv 2410.11081v2, p. 1", note: "stable continuous-time training and its results at scale"}
  - {paper: "Sauer, Lorenz, Blattmann and Rombach (2023)", title: "Adversarial Diffusion Distillation", locator: "arXiv 2311.17042v1, p. 1", note: "score distillation with an adversarial loss"}
  - {paper: "Yin, Gharbi, Zhang, Shechtman, Durand, Freeman and Park (2023)", title: "One-step Diffusion with Distribution Matching Distillation", locator: "arXiv 2311.18828v4, p. 1", note: "a one-step generator trained by the difference of two scores"}
family: diffusion-models
inputs: "for distillation, a batch of data, a pre-trained diffusion model as teacher and its ODE solver; for training from data, a batch of data alone; for sampling, Gaussian noise of scale T and optionally a list of intermediate noise levels"
outputs: "for training, the consistency loss between the outputs at two adjacent noise levels of one path; for the model, an estimate of the clean sample with the shape of the input; for sampling, a sample after one network evaluation, or after k evaluations with k levels"
components: [diffusion-model, residual-connection]
inductive_biases: ["the map from noise to data is a function of the noisy point alone, the deterministic probability flow ODE, so one evaluation can replace the whole solver", "the output equals the input at the smallest noise level by construction, through a skip connection whose weight goes to 1, so the network only has to learn a correction that grows with the noise", "every point of one path must give the same answer, so a model trained to agree between adjacent levels propagates the exact boundary value up the whole path", "the same model can be applied again after adding noise back, so quality can be traded for evaluations without retraining"]
params: "those of the network F, with one more copy for the target network when the target is an EMA; for the network of this page with samples of dimension D, H hidden units and a time embedding of width E, DH + EH + H + HD + D"
flops_per_example: "one network evaluation per sampling level, one in total for one-step sampling; for the network of this page, 2(DH + EH + HD) FLOPs per sample, a multiply-add counted as 2"
---

A consistency model generates a sample with one network evaluation where a [[diffusion-model|diffusion model]] needs dozens.
A diffusion model defines a probability flow ODE whose paths carry pure noise to data, and sampling solves that ODE step by step.
A consistency model learns the solver's end result directly: given any point on a path and its noise level, it returns the point where the path ends.
Song, Dhariwal, Chen and Sutskever introduced it in 2023 and trained it in two ways, by distilling a pre-trained diffusion model, which supplies adjacent points of each path, and from data alone, where the two points share a data point and a noise vector.
It can also be sampled in a few steps that trade compute for quality, and it edits images zero-shot as diffusion models do; distilled into the latent space of Stable Diffusion, it is the basis of fast text-to-image models.

## In plain words

Picture a river delta in reverse: water from all over a wide plain flows along channels that all end at a few springs.
A diffusion model is a guide who walks you back along your channel one step at a time, checking the current at each step, until you reach the spring; that takes dozens of steps.
A consistency model is a guide who looks at where you stand and points straight to the spring of your channel.
It learns this by a simple rule: two spots on the same channel must be pointed to the same spring.
Where the spring is obvious, right next to it, the answer is "here", and the rule passes that certainty upstream, one spot at a time, until the guide can answer from anywhere on the plain.
You can also ask twice: go where the guide points, wander a little way back up a channel, and ask again, which corrects the first answer's errors.

## Overview

A diffusion model in the form of Karras and colleagues adds Gaussian noise of standard deviation $t$ to data, $x_t = x + t z$ with $z \sim \mathcal N(0, I)$, for $t$ from a small $\epsilon$ to a large $T$, and learns the [[m2:score|score]] $\nabla \log p_t$ of the noisy distribution.
The same marginals are produced by the deterministic probability flow ODE

$$ \frac{dx_t}{dt} = -t\, \nabla \log p_t(x_t) , $$ {#eq-pf-ode}

the deterministic counterpart of the [[m2:stochastic-differential-equation|stochastic differential equation]] that adds the noise.
Its solution through a point $x_T \sim \mathcal N(0, T^2 I)$ ends near a data point at $t = \epsilon$, and a diffusion sampler integrates it with many steps.

The consistency function of the ODE maps each point of a solution path to the path's end,

$$ f : (x_t, t) \mapsto x_\epsilon , $$ {#eq-cf}

so it is self-consistent: points on the same path map to the same end, $f(x_t, t) = f(x_{t'}, t')$, and it fixes the end itself, $f(x_\epsilon, \epsilon) = x_\epsilon$.
A consistency model $f_\theta$ approximates it, so a sample is one evaluation, $f_\theta(x_T, T)$ with $x_T \sim \mathcal N(0, T^2 I)$.
Consistency models generate by directly mapping noise to data; they support one-step generation by design, multistep sampling that trades compute for quality, and zero-shot editing such as inpainting, colorization and super-resolution, and they can be trained by distilling a pre-trained diffusion model or as standalone generative models. {#c-cm-def}

For Gaussian data the consistency function has a closed form, eq-gauss-cf of the derivation `gaussian-consistency-function`: with data $\mathcal N(\mu, s^2 I)$, $f(x, t) = \mu + (x - \mu)\sqrt{(s^2 + \epsilon^2)/(s^2 + t^2)}$, a rescaling that shrinks the noisy point towards the mean by the ratio of the standard deviations at the two levels. {#c-gauss-cf}
For a [[m2:mixture-model|mixture]] of Gaussians, the data of the tests, the paths bend and split between the components, and the consistency function must decide from the noisy point alone which component a path ends in.

## Components

### The boundary condition and the parameterization

The second property, $f(x, \epsilon) = x$, is the boundary condition.
All consistency models must meet the boundary condition, which plays a crucial role in their training and is the most confining architectural constraint on them; it costs almost nothing through a skip connection $f_\theta = c_{\text{skip}}(t)\,x + c_{\text{out}}(t)\,F_\theta(x, t)$, and it precludes the trivial solution $f \equiv 0$ from arising in training. {#c-cm-boundary}
Song and colleagues take the preconditioning of Karras and colleagues, with $\sigma_d = 0.5$ the standard deviation of the data, and shift it by $\epsilon$:

$$ c_{\text{skip}}(t) = \frac{\sigma_d^2}{(t - \epsilon)^2 + \sigma_d^2} , \qquad c_{\text{out}}(t) = \frac{\sigma_d (t - \epsilon)}{\sqrt{\sigma_d^2 + t^2}} , \qquad c_{\text{in}}(t) = \frac{1}{\sqrt{\sigma_d^2 + t^2}} . $$ {#eq-param}

With these scales $f_\theta(x, \epsilon) = x$ exactly for every network with finite outputs and every parameter, and $f_\theta$ is differentiable in $t$ when $F_\theta$ is. {#c-param-boundary}
The derivation `parameterization` proves it and `test_boundary_condition` checks it in all three implementations.
The skip weight $c_{\text{skip}}$ falls from 1 at $\epsilon$ to almost 0 at large $t$, so at low noise the model is close to the identity, the [[residual-connection|residual connection]] of the output, and at high noise it is almost all network; $c_{\text{in}}$ scales the input to unit variance, so the network sees inputs of the same size at every level.

### Multistep sampling

One evaluation gives $x = f_\theta(x_T, T)$.
Algorithm 1 of Song and colleagues refines it: for a decreasing list of levels $\tau_1 > \tau_2 > \dots$, add fresh noise back to the current sample and map it again,

$$ x \leftarrow f_\theta\big(x + \sqrt{\tau_k^2 - \epsilon^2}\, z_k,\ \tau_k\big) , \qquad z_k \sim \mathcal N(0, I) , $$ {#eq-sample}

one evaluation per level.
The added noise brings the sample to level $\tau_k$ as if it were data, so each evaluation corrects some of the errors of the last.
Song and colleagues pick the levels by a greedy ternary search on FID, one level at a time.
The same loop, with a projection onto the known pixels after each evaluation, does inpainting, colorization and super-resolution without training on those tasks.

### Consistency distillation

Distillation uses a pre-trained diffusion model $s_\phi$ as the teacher.
Divide $[\epsilon, T]$ into levels $\epsilon = t_1 < t_2 < \dots < t_N = T$, those of Karras and colleagues with $\rho = 7$.
For a data point $x$, a level $n$ and noise $z$, put $x_{t_{n+1}} = x + t_{n+1} z$ and take one step of the teacher's ODE solver down to $t_n$:

$$ \hat x^\phi_{t_n} = x_{t_{n+1}} + (t_n - t_{n+1})\,\Phi(x_{t_{n+1}}, t_{n+1}; \phi) , $$ {#eq-teacher}

with $\Phi = -t\,s_\phi(x, t)$ for Euler's method, or Heun's second-order correction.
The two points lie, up to the solver's error, on the same path, so the model should give them the same output.
The consistency distillation loss is

$$ \mathcal L_{\text{CD}}^N(\theta, \theta^-; \phi) = \mathbb E\Big[\lambda(t_n)\, d\big(f_\theta(x_{t_{n+1}}, t_{n+1}),\ f_{\theta^-}(\hat x^\phi_{t_n}, t_n)\big)\Big] , $$ {#eq-cd}

with $n$ uniform, a weight $\lambda$ and a metric $d$, the squared $\ell_2$, $\ell_1$ or LPIPS distance.
The second argument uses a target network $\theta^-$ and no gradient flows through it; after each step of [[adam|Adam]] or another optimizer it moves towards the online network,

$$ \theta^- \leftarrow \operatorname{stopgrad}\big(\mu\,\theta^- + (1 - \mu)\,\theta\big) . $$ {#eq-ema}

Compared with simply setting $\theta^- = \theta$, this EMA update and the stop-gradient greatly stabilize training and improve the final model. {#c-cm-ema}
The boundary condition is what makes the loss informative: at $t_1 = \epsilon$ the target is the identity, which is right, and minimizing the loss pushes that correct value up the levels.
If $f_\theta$ satisfies the Lipschitz condition, the solver has local error $O((t_{n+1} - t_n)^{p+1})$ with $p \ge 1$, and the distillation loss is 0, then $f_\theta$ differs from the consistency function of the teacher's empirical PF ODE by $O((\Delta t)^p)$, with $\Delta t$ the maximum of $\lvert t_{n+1} - t_n\rvert$. {#c-cm-theorem1}

### Consistency training

Training from data replaces the teacher with an estimate of the score from a single data point.
For $x_t = x + t z$, the score is the conditional expectation $\nabla \log p_t(x_t) = -\mathbb E[(x_t - x)/t^2 \mid x_t]$, eq-score-est of the derivation `score-estimator`, so $-(x_t - x)/t^2$ is an unbiased one-sample estimate of the score.
Consistency training avoids the pre-trained score model with this estimator, which suffices to replace the diffusion model in distillation with the Euler solver as the number of levels grows. {#c-cm-ct-estimator}
One Euler step with this estimate from $x + t_{n+1} z$ lands exactly on $x + t_n z$, so the training pair is the same data point with the same noise at two adjacent levels, and the consistency training loss is

$$ \mathcal L_{\text{CT}}^N(\theta, \theta^-) = \mathbb E\Big[\lambda(t_n)\, d\big(f_\theta(x + t_{n+1} z, t_{n+1}),\ f_{\theta^-}(x + t_n z, t_n)\big)\Big] . $$ {#eq-ct}

The derivation `score-estimator` proves both steps: the pair of consistency training is the pair of consistency distillation with an Euler teacher using the one-sample estimate. {#c-ct-pair}
No teacher is needed, and the model is a new kind of generative model trained from scratch.

### Improved consistency training

Song and Dhariwal changed four parts of consistency training and removed the EMA from the target.
The EMA decay of the teacher network should always be zero for consistency training, although it can be nonzero for distillation; with a nonzero rate the limit of the training objective as $N$ grows is not a valid objective for learning the consistency function. {#c-ict-ema}
Their toy makes this concrete: data at one point $\xi$ and a one-parameter model that is exact at $\theta = \xi$.
For this toy, with the squared $\ell_2$ metric, as the gaps between levels shrink the loss tends to $\mathbb E[(1 - \sigma_{\min}/\sigma_i)^2](\theta - \theta^-)^2$, which does not involve $\xi$, when $\theta^- \ne \theta$; with $\theta^- = \operatorname{stopgrad}(\theta)$ the rescaled gradient is proportional to $\theta - \xi$ and vanishes only at the true consistency function. {#c-teacher-ema}
The derivation `teacher-ema` computes both limits and the descent each one drives.

The other changes are the metric, the number of levels, the choice of level and the weight:

$$ d(x, y) = \sqrt{\lVert x - y\rVert_2^2 + c^2} - c , \qquad \lambda(t_n) = \frac{1}{t_{n+1} - t_n} , \qquad p(n) \propto \operatorname{erf}\Big(\frac{\log t_{n+1} - P_{\text{mean}}}{\sqrt 2 P_{\text{std}}}\Big) - \operatorname{erf}\Big(\frac{\log t_n - P_{\text{mean}}}{\sqrt 2 P_{\text{std}}}\Big) , $$ {#eq-ict}

with $c = 0.00054\sqrt d$ for data of dimension $d$, $P_{\text{mean}} = -1.1$ and $P_{\text{std}} = 2$.
Pseudo-Huber metrics smoothly bridge the $\ell_1$ and squared $\ell_2$ metrics, are continuously twice differentiable, and reduce the variance of the Adam updates relative to the squared $\ell_2$ metric; $c = 0.03$ is optimal for CIFAR-10, and $c = 0.00054\sqrt d$ is proposed for images of $d$ dimensions. {#c-ict-huber}
For $c > 0$ the metric lies between $r^2/(2\sqrt{r^2 + c^2})$ and $\min(r^2/(2c), r)$, so it behaves like $r^2/(2c)$ for distances $r$ much smaller than $c$ and like $r - c$ for much larger ones; its gradient has norm below 1, it is smooth at $r = 0$ with second derivative $1/c$, and $0.00054\sqrt{3072} = 0.0299$ for CIFAR-10. {#c-huber-limits}
It is the [[m2:huber-loss|Huber loss]] of [[m2:robust-statistics|robust statistics]] in a smooth form, and the derivation `pseudo-huber` proves the bounds.
FID improves with the number of levels $N$ following a power law until saturation; doubling $N$ after a set number of iterations gave the lowest FID of the curricula tried, with $N$ capped at 1281, and sampling levels from a discretized lognormal with mean $-1.1$ and standard deviation 2 improved sample quality over the original schedule, which favours high noise levels. {#c-ict-curriculum}
The schedule of Song and colleagues takes $N$ from $s_0 = 2$ to $s_1 + 1 = 151$ and $\mu$ from 0.9 to $0.9^{2/151} = 0.99861$; the doubling curriculum with $s_0 = 10$ and $s_1 = 1280$ takes $N$ through $11, 21, \dots, 1281$; a uniform index over the levels gives $\log t$ a density proportional to $t^{1/\rho}$, with median $t \approx 2.5$, while the lognormal schedule gives each interval its mass under $\mathcal N(-1.1, 2^2)$ truncated to the range, and the weight $1/(t_{n+1} - t_n)$ decreases in $n$. {#c-schedules}
The derivation `schedules` proves these values.

### Progressive distillation

Progressive distillation, the method consistency distillation was first compared with, distils a deterministic sampler into a model that takes half as many steps, then distils that model again, going from samplers of 8192 steps to 4 steps with an FID of 3.0 on CIFAR-10; the whole procedure takes no more time than training the original model. {#c-pd}
In a variance-preserving process $z_t = \alpha_t x + \sigma_t \epsilon$, the student at $t$ is trained to predict the $x$ that makes one DDIM step to $t''$ land where two teacher steps, to $t'$ and $t''$, land, eq-pd of the derivation `progressive-distillation-target`.
For a variance-preserving schedule the target $\tilde x = (z_{t''} - r z_t)/(\alpha_{t''} - r\alpha_t)$ with $r = \sigma_{t''}/\sigma_t$ is the one $x$-prediction whose single DDIM step from $t$ to $t''$ lands on the end of the teacher's two steps; its denominator is positive whenever $t'' < t$, including at $\alpha_t = 0$; the $\epsilon$ link multiplies a prediction error by $\sigma_t/\alpha_t$ and the $v$ link by $\sigma_t \le 1$. {#c-pd-target}
As the signal-to-noise ratio goes to zero, small changes in the predicted noise are increasingly amplified in the implied prediction $\hat x = (z - \sigma\hat\epsilon)/\alpha$, which divides by $\alpha_t$, and with a single sampling step the link between noise prediction and $x$-prediction breaks down completely; predicting $x$ directly, predicting both $x$ and the noise, or predicting $v = \alpha\epsilon - \sigma x$ all keep the implied prediction stable and work well with progressive distillation. {#c-pd-param}
Each round of progressive distillation fixes the number of steps of its student, while a consistency model can be sampled with one step or several from the same weights.

## Shapes and parameters

The shape contract is the network of the implementations: $F$ reads the scaled sample $c_{\text{in}}(t)x$ of dimension $D$, adds a Fourier embedding of $\log(t)/4$ of width $E$ through $W_t$, passes the sum through a tanh layer of width $H$, and reads out a vector of dimension $D$; the output is eq-param:

$$ h = \tanh\big(c_{\text{in}}(t)\,x W_1 + \operatorname{emb}(t) W_t + b_1\big) , \qquad f_\theta(x, t) = c_{\text{skip}}(t)\,x + c_{\text{out}}(t)\,(h W_2 + b_2) . $$

Its parameter count is

$$ P = DH + EH + H + HD + D , $$ {#eq-params}

the scales $c_{\text{skip}}$, $c_{\text{out}}$ and $c_{\text{in}}$ holding none.
In the contract this is
{shape:params}
and the example with $D = 3$, $H = 5$, $E = 4$ has {shape:params@example} parameters.
The model trained in the tests, `mixture`, has {shape:params@mixture}, and the same form on a CIFAR-10 image, `cifar10`, would have {shape:params@cifar10}.
Distillation with an EMA target keeps a second copy of the parameters, and the teacher is a third network, often of the same architecture since the student is initialized from it; improved consistency training with $\mu = 0$ keeps one copy, the target being the online network with its gradient stopped.
The networks of the papers are the U-Nets of EDM, and Lu and Song trained continuous-time models of 1.5 billion parameters.

## Computation and cost

One call of the network on $B$ samples costs

$$ C_f = 2B\big(DH + EH + HD\big) $$ {#eq-flops}

FLOPs, a multiply-add counted as two, leaving out the scales, biases, embedding and tanh.
In the contract,
{shape:flops_forward}
which is {shape:flops_forward@example} FLOPs for the example batch, {shape:flops_forward@mixture} for one training batch of the `mixture` test, and {shape:flops_forward@cifar10} for one image of `cifar10`.
Heun sampling over $N$ noise levels ending with an Euler step to 0 costs $2N - 1$ network evaluations, 35 for $N = 18$, while $k$-step consistency sampling costs $k$; a consistency distillation sample costs two teacher calls plus about four student-equivalent passes; the page's network has $P = DH + EH + H + HD + D$ parameters and $C_f = 2B(DH + EH + HD)$ FLOPs, 834 parameters for the test model and 12,713,984 FLOPs for one CIFAR-10 image. {#c-cost}
The derivation `sampling-cost` counts the calls inside the implementations.
With its design choices, including Heun's second-order sampler, the EDM diffusion model reached an FID of 1.97 for unconditional CIFAR-10 with 35 network evaluations per image. {#c-edm}
A one-step consistency model of the same size samples 35 times faster than that sampler; training costs about as much per iteration as diffusion training, a little more for distillation because of the teacher calls.
Continuous-time training replaces the pair of levels with the time derivative of $f_\theta$ along the ODE, which needs a Jacobian-vector product through the network, as forward-mode [[al:automatic-differentiation|automatic differentiation]] gives, at about the cost of one more forward pass.

## Training

Distillation follows Algorithm 2 of Song and colleagues: initialize the student and the target from the teacher's weights, then repeat: draw a data batch, a level $n$ and noise; form $x_{t_{n+1}}$ and the teacher's step eq-teacher; take a gradient step on eq-cd; and update the target by eq-ema.
On CIFAR-10 the configuration with the lowest FID for consistency distillation was the LPIPS metric, Heun's solver and $N = 18$; consistency distillation outperformed progressive distillation across all datasets, sampling steps and metrics tried, except single-step generation on LSUN Bedroom $256 \times 256$ with the squared $\ell_2$ metric. {#c-cm-cd-config}
Training from data follows Algorithm 3: the same loop with the pair of eq-ct and schedules for $N$ and $\mu$.
The convergence of consistency training is highly sensitive to $N$: smaller $N$ leads to faster convergence but worse samples, larger $N$ to slower convergence but better samples, which motivates growing $N$ and $\mu$ during training. {#c-cm-ct-n}
Improved consistency training sets $\mu = 0$, doubles $N$ from 11 to 1281 in eight stages, draws levels from the lognormal of eq-ict with the weight $1/(t_{n+1} - t_n)$ and the Pseudo-Huber metric, and keeps an EMA of the weights for evaluation only.

The implementations train the network of the shape contract with $D = 2$, $H = 64$ and $E = 8$ on data from a [[al:gaussian-mixture-model|mixture of three Gaussians]] in the plane, of standard deviation 0.1 at distance 0.8 from the origin, whose exact score lets the teacher be the true probability flow, with batches of 256 and Adam at a learning rate of $3 \times 10^{-3}$.
`test_distillation_and_training` distils for 3000 steps with a Heun teacher, $N = 18$ and $\mu = 0.95$, and checks that one-step samples from 2000 noise vectors land on average within 0.2 of the end of the exact ODE path from the same noise, against about 0.8 for the untrained model, that more than 65% land near a component mean, and that the three components each get a third of the samples to within 0.06.
It then trains from data alone for 3000 steps with the doubling curriculum from 10 to 160 levels, $\mu = 0$, the lognormal choice of level and its weight, with the Pseudo-Huber metric at $c = 0.00054\sqrt 2$, and checks that more than 25% of one-step samples land near a mean, that two-step sampling with $\tau = 0.8$ puts more than 60% there and more than one step does, and that the proportions are within 0.1 of a third.
The NumPy reference writes every gradient by hand, the PyTorch version uses autograd and `torch.func.jvp`, and the JAX version runs distillation as one `jax.lax.scan` and training as a jitted scan per stage of the curriculum.

## Inductive biases

A consistency model assumes that the map from noise to data is deterministic, which holds for the probability flow ODE and not for the stochastic sampler; one noise vector gives one sample.
The parameterization eq-param builds in the boundary condition, so the network never has to learn the identity at low noise and the learning signal enters at the bottom of each path.
Self-consistency is a constraint between levels of the same path, not a regression to a fixed target, and the only fixed value is the boundary; errors at low noise therefore propagate to every higher level through the target, which is why Song and Dhariwal weight low noise levels more.
The multistep sampler assumes that a sample with noise added back looks like data with noise added, which holds when the model is good and degrades as errors accumulate.

## Evidence

Consistency distillation reached FIDs of 3.55 and 2.93 with one and two steps on CIFAR-10 and 6.20 and 4.70 on ImageNet $64 \times 64$, against 2.04 at 35 evaluations and 2.44 at 79 for the EDM teachers; consistency training reached 8.70 and 5.83 on CIFAR-10 and 13.0 and 11.1 on ImageNet $64 \times 64$. {#c-cm-results}
Improved consistency training reached one-step FIDs of 2.51 on CIFAR-10 and 3.25 on ImageNet $64 \times 64$, a 3.5 and 4 times improvement over prior consistency training, and two-step FIDs of 2.24 and 2.77, surpassing consistency distillation in both one-step and two-step settings. {#c-ict-results}
Continuous-time consistency models, stabilized by changes to the diffusion parameterization, network architecture and objective, were trained at 1.5B parameters on ImageNet $512 \times 512$ and reached two-step FIDs of 2.06 on CIFAR-10, 1.48 on ImageNet $64 \times 64$ and 1.88 on ImageNet $512 \times 512$, within 10% of the FIDs of the diffusion models they were compared with. {#c-scm}
These are FIDs on unconditional CIFAR-10 and class-conditional ImageNet; the papers report few human evaluations, and FID favours some failure modes over others.

## Limitations

Distillation learns the teacher's ODE, so the student's error adds to the teacher's, and the one-step and two-step FIDs of Song and colleagues stay above those of their teachers.
Training from data has no teacher, but its target is a one-sample estimate, unbiased only as the gaps between levels go to zero, and it needs many levels and long training; its first results lagged distillation until the changes of Song and Dhariwal.
The loss compares the model with itself, so it has degenerate fixed points that only the boundary condition rules out, and training is sensitive to the EMA rate, the metric and the schedule.
The original results used LPIPS, a metric built on ImageNet features, which Song and Dhariwal argue can bias FID evaluation, itself computed with ImageNet features.
More sampling steps need not help: errors of the model accumulate through eq-sample, unlike a diffusion sampler, which converges to the ODE solution as steps grow.
The continuous-time losses need Jacobian-vector products through the network, which not every kernel supports, and were unstable before Lu and Song's changes.

## Variants and descendants

Luhman and Luhman's knowledge distillation trained a one-step denoising student to predict the output of a deterministic DDIM sampler from the same initial noise, reaching an FID of 9.36 on CIFAR-10 without adversarial training, where the iterative models it distils are 2 to 3 orders of magnitude slower than GANs and VAEs. {#c-luhman}
It is [[al:knowledge-distillation|knowledge distillation]] with the sampler as teacher, and needs the sampler run on every training example.
Guided diffusion models can be distilled in two stages, first one model matching the combined conditional and unconditional outputs of [[classifier-free-guidance|classifier-free guidance]], then progressive distillation; in pixel space this gives samples comparable to the original with 4 steps on ImageNet $64 \times 64$ and CIFAR-10, up to 256 times faster, and in latent space 1 to 4 steps. {#c-meng}
Latent consistency models view guided reverse diffusion as solving an augmented probability flow ODE in the latent space of a pre-trained [[latent-diffusion-model|latent diffusion model]]; a 2 to 4 step model at $768 \times 768$ distilled from a classifier-free guided Stable Diffusion takes 32 A100 GPU hours to train. {#c-lcm}
Adversarial diffusion distillation combines score distillation from a large diffusion model with an adversarial loss, that of a [[generative-adversarial-network|generative adversarial network]], to sample in 1 to 4 steps; it outperforms GANs and latent consistency models in a single step and reaches SDXL in four steps. {#c-add}
Distribution matching distillation trains a one-step generator by minimizing an approximate [[m2:kullback-leibler-divergence|KL divergence]] whose gradient is the difference of two scores, one of the target and one of the generator's distribution, plus a regression loss, reaching an FID of 2.62 on ImageNet $64 \times 64$ and 11.49 on zero-shot COCO-30k. {#c-dmd}
Rectified flow and [[flow-matching|flow matching]] reach few-step sampling another way, by learning straighter paths that a few Euler steps follow well, and a consistency model can be trained on a flow-matching ODE in place of a diffusion one.
One-step generators also exist without any ODE: a [[generative-adversarial-network|GAN]], a [[variational-autoencoder|variational autoencoder]] and a [[normalizing-flow|normalizing flow]] map noise to data in one pass, and consistency models were introduced as a non-adversarial member of that group with the editing abilities of diffusion.

## Notable models

- The CIFAR-10, ImageNet $64 \times 64$ and LSUN $256 \times 256$ models of Song and colleagues (2023), distilled from EDM or trained from data, with open code.
- The improved consistency training models of Song and Dhariwal (2023), the first trained from data alone to beat distillation.
- Latent consistency models (Luo and colleagues, 2023), distilled from Stable Diffusion for text-to-image generation in 2 to 4 steps.
- sCM (Lu and Song, 2024), continuous-time models up to 1.5 billion parameters on ImageNet $512 \times 512$.

## Implementations

The NumPy reference `impl/numpy/consistency_model.py` implements:

- the noise levels of Karras and colleagues, the scales of eq-param and the network of the shape contract with its gradient written by hand;
- exact references for a mixture of Gaussians: the score, the ODE drift, Euler and Heun steps of eq-pf-ode, the consistency function by many Heun steps, and eq-gauss-cf for one Gaussian;
- the losses eq-cd and eq-ct with the squared $\ell_2$ and Pseudo-Huber metrics of eq-ict and their gradients, the EMA of eq-ema, Adam, and both schedules of $N$ and $\mu$ with the lognormal choice of level and its weight;
- one-step sampling and the multistep sampler eq-sample;
- the toy of Song and Dhariwal's Proposition 1, and the DDIM step and target eq-pd of progressive distillation.

The PyTorch version `impl/pytorch/consistency_model.py` has the model as an `nn.Module`, the target computed under `torch.no_grad`, the EMA by `lerp_`, the total derivative of $f_\theta$ along the ODE by `torch.func.jvp`, and `torch.autograd.gradcheck` through the loss.
The JAX version `impl/jax/consistency_model.py` writes the target with `jax.lax.stop_gradient`, takes the total derivative with `jax.jvp`, and trains with `jax.lax.scan` under `jax.jit`.
All three load the shared fixture of `fixtures/make.py` and check the model's output, the distillation loss and its gradient in every parameter, the Pseudo-Huber training loss and its gradient, one EMA step, a two-step sample and the progressive distillation target.

### Worked example

Take one-dimensional data $\mathcal N(0, 0.5^2)$, so $s = \sigma_d = 0.5$, with $\epsilon = 0.002$ and $T = 80$.
A noise sample $x_T = 40$ is half a standard deviation of $\mathcal N(0, 80^2)$ above the mean, and eq-gauss-cf maps it to $40\sqrt{0.250004/6400.25} = 0.25$, half a standard deviation of the data: one evaluation of the exact consistency function replaces the whole ODE.
At $t = 80$ the scales are $c_{\text{skip}} = 3.906 \times 10^{-5}$ and $c_{\text{out}} = 0.49998$, so the network must output $F = (0.25 - 3.906 \times 10^{-5} \cdot 40)/0.49998 = 0.4969$; at $t = 1$ a point $x = 1$ maps to $0.4472$, with $c_{\text{skip}} = 0.2006$ and $c_{\text{out}} = 0.4463$, so $F = 0.5525$.
For consistency training, the data point $x = 0.3$ and the noise $z = 1$ at levels 1 and 2 give the pair $(2.3, 1.3)$; one Euler step from 2.3 with the one-sample score $-(2.3 - 0.3)/2^2 = -0.5$, drift $-t \cdot (-0.5) = 1$, lands on $2.3 - 1 = 1.3$.
The Pseudo-Huber metric with $c = 0.03$ gives 0.00162 at distance 0.01, close to $0.01^2/0.06 = 0.00167$, 0.0744 at 0.1 and 0.970 at 1, close to $1 - 0.03$: quadratic for small errors and linear for large ones.
`test_worked_example` checks these numbers in the NumPy implementation.

### Pitfalls

> [!pitfall]
> Without the boundary condition the loss has a trivial minimum: $f \equiv 0$, or any constant, is perfectly self-consistent. Build the condition into the model with eq-param, not as a penalty, and check $f_\theta(x, \epsilon) = x$ for random parameters, as `test_boundary_condition` does.

> [!pitfall]
> In consistency training an EMA target with $\mu > 0$ makes the loss pull the student towards the target rather than towards the data, by c-teacher-ema. Use $\mu = 0$ for the target, the online network with its gradient stopped, and keep a separate EMA of the weights for evaluation only.

> [!pitfall]
> The target must not receive gradients. With autograd, compute $f_{\theta^-}$ under `torch.no_grad` or `jax.lax.stop_gradient`; otherwise the loss also moves the target branch and the boundary signal is lost, which `test_toy_student_only_gradient` and `test_toy_by_autograd` guard against.

> [!pitfall]
> The Pseudo-Huber $c$ is in units of the distance between whole samples, so it must grow with the dimension: $c = 0.03$ suits $3 \times 32 \times 32$ images, and the same value on a 12,288-dimensional sample makes the metric almost an $\ell_2$ distance with the variance it was meant to cut. Use $c = 0.00054\sqrt d$.

> [!pitfall]
> The multistep sampler adds noise of variance $\tau_k^2 - \epsilon^2$, not $\tau_k^2$, because the model's output already sits at level $\epsilon$; and each step needs fresh noise, since reusing the first noise vector correlates the steps and undoes the correction.

> [!pitfall]
> For progressive distillation and other few-step students, predicting $\epsilon$ fails at the start of sampling, where $\alpha_t \to 0$ and $\hat x = (z - \sigma\hat\epsilon)/\alpha$ multiplies errors without bound; predict $x$ or $v$ instead, or use the skip parameterization of eq-param, which is an $x$-prediction.

## Experiments

This page has no experiment yet.
Some tests are small experiments.

`test_distillation_and_training`, described in the Training section, distils a consistency model from the exact ODE of a mixture of three Gaussians and trains another from data alone, and compares their one-step and two-step samples with the ends of the exact paths and with the component weights.
`test_gaussian_consistency_function` solves the ODE with 400 Heun steps and checks that every point of a path maps to its end under eq-gauss-cf; `test_gaussian_total_derivative_is_zero` checks with `torch.func.jvp` and `jax.jvp` that the exact consistency function does not change along the ODE.
`test_proposition_1_toy` computes the consistency training loss of Song and Dhariwal's toy for growing $N$ and compares its gradient with and without an EMA target against the limits.

The checks of the eight derivations compare:

- the boundary values and monotonicity of the scales of eq-param;
- eq-gauss-cf with a fine ODE solution and with the end distribution;
- the one-sample score estimate with the exact score of a mixture, and the Euler step with the training pair;
- the toy loss and gradient with their limits, and descent with and without an EMA target;
- the Pseudo-Huber metric with its bounds, gradient and curvature, and the value of $c$ for CIFAR-10 and ImageNet $64 \times 64$;
- the progressive distillation target with two teacher DDIM steps, and the error of the $\epsilon$ and $v$ links as $\alpha_t \to 0$;
- the schedules of $N$ and $\mu$, the density of the uniform level and the lognormal masses;
- the numbers of evaluations of the samplers and the parameter and FLOP counts with the code.

## History

Diffusion models sampled with hundreds or thousands of steps when Ho and colleagues introduced denoising diffusion in 2020, and work on sampling speed went two ways: better solvers, such as DDIM and the Heun sampler of Karras and colleagues, and distillation.
Luhman and Luhman distilled a DDIM sampler into a one-step student in 2021, and Salimans and Ho's progressive distillation of 2022 halved the number of steps round by round, from 8192 to 4.
Song, Dhariwal, Chen and Sutskever's consistency models, at ICML 2023, replaced the fixed teacher trajectory with the self-consistency of the probability flow ODE, which allowed one model to sample in any number of steps and to be trained without a teacher.
Song and Dhariwal's improved consistency training of late 2023 made training from data better than distillation; Luo and colleagues' latent consistency models brought the method to Stable Diffusion the same year.
Lu and Song's continuous-time models of 2024 removed the discretization and scaled the method to 1.5 billion parameters; alongside adversarial and distribution-matching distillation, consistency models are among the standard ways to make diffusion models, and [[flow-matching|flow-matching]] models, fast.

## References

- Y. Song, P. Dhariwal, M. Chen and I. Sutskever. Consistency Models. ICML 2023; arXiv 2303.01469.
- Y. Song and P. Dhariwal. Improved Techniques for Training Consistency Models. arXiv 2310.14189, 2023.
- T. Salimans and J. Ho. Progressive Distillation for Fast Sampling of Diffusion Models. ICLR 2022; arXiv 2202.00512.
- E. Luhman and T. Luhman. Knowledge Distillation in Iterative Generative Models for Improved Sampling Speed. arXiv 2101.02388, 2021.
- T. Karras, M. Aittala, T. Aila and S. Laine. Elucidating the Design Space of Diffusion-Based Generative Models. NeurIPS 2022; arXiv 2206.00364.
- C. Meng, R. Rombach, R. Gao, D. P. Kingma, S. Ermon, J. Ho and T. Salimans. On Distillation of Guided Diffusion Models. arXiv 2210.03142, 2022.
- S. Luo, Y. Tan, L. Huang, J. Li and H. Zhao. Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference. arXiv 2310.04378, 2023.
- C. Lu and Y. Song. Simplifying, Stabilizing and Scaling Continuous-Time Consistency Models. ICLR 2025; arXiv 2410.11081.
- A. Sauer, D. Lorenz, A. Blattmann and R. Rombach. Adversarial Diffusion Distillation. arXiv 2311.17042, 2023.
- T. Yin, M. Gharbi, R. Zhang, E. Shechtman, F. Durand, W. T. Freeman and T. Park. One-step Diffusion with Distribution Matching Distillation. arXiv 2311.18828, 2023.
