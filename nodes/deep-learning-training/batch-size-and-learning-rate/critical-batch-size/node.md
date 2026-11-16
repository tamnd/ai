---
tag: NEW-critical-batch-size
kind: concept
slug: critical-batch-size
name: Critical batch size
aliases: [gradient noise scale, noise scale, B_crit, B_noise, B_simple, largest useful batch size, maximum useful batch size]
area: deep-learning-training/batch-size-and-learning-rate
level: core
summary: >-
  The batch size beyond which a larger batch stops cutting the steps of training in proportion; the gradient noise scale, tr(Σ)/|G|², predicts it.
uses: [stochastic-gradient-descent@m2, variance@m2, covariance-matrix@m2, quadratic-form@m2]
related_to: [learning-rate-schedule@al, gradient-accumulation@al, layer-wise-adaptive-rate-scaling@al]
history:
  year: 2018
  people: [Sam McCandlish, Jared Kaplan, Dario Amodei]
  note: "McCandlish et al. (arXiv 1812.06162, December 2018) defined the gradient noise scale and the critical batch size; Shallue et al. (JMLR 2019) measured the three regimes across 35 workloads; Kaplan et al. (2020) fit the critical batch size of language models as a power law in the loss."
prereqs: [stochastic-gradient-descent@m2, variance@m2, covariance-matrix@m2, bias-of-an-estimator@m2]
same_as:
  wikidata: null
  wikipedia: null
  arxiv: ["1812.06162", "1811.03600", "2001.08361"]
  al: null
sources:
  - {paper: "McCandlish et al. (2018)", title: "An Empirical Model of Large-Batch Training", locator: "pp. 1, 3 to 13, 17, 25 and 27", note: "the noise scale, the optimal step, the trade-off, the critical batch size, the two-batch estimator, temperature and the experiments"}
  - {paper: "Shallue et al. (2019)", title: "Measuring the Effects of Data Parallelism on Neural Network Training", locator: "pp. 1, 4, 10 and 15", note: "the three regimes of steps against batch size across workloads"}
  - {paper: "Kaplan et al. (2020)", title: "Scaling Laws for Neural Language Models", locator: "pp. 3, 5, 12 and 13", note: "the critical batch size of Transformer language models as a power law in the loss"}
  - {paper: "Smith and Le (2018)", title: "A Bayesian Perspective on Generalization and Stochastic Gradient Descent", locator: "pp. 1 and 7", note: "the SGD noise scale g and the batch size that maximizes test accuracy"}
---

The critical batch size is the batch size at which adding examples to each step of stochastic gradient descent stops buying a proportional cut in the number of steps.
McCandlish and colleagues found that a cheap statistic, the gradient noise scale, forecasts the biggest batch worth using in tasks as different as image classification, Atari and Dota. {#c-noise-scale-predicts}
It compares how noisy the per-example gradients are with how large the true gradient is.
The noise scale grows as the loss falls, so the critical batch size is a property of a point in training, not of a model. {#c-grows-with-loss}

## In plain words

Each step of training averages the gradients of a batch of examples, and a bigger batch gives a less noisy average.
While noise is what holds a step back, doubling the batch lets each step go about twice as far, so training needs about half the steps for the same total number of examples.
Once the average is already close to the true gradient, more examples do not help much: the step is now limited by the curvature of the loss, and the extra examples are wasted compute.
The batch size where one regime gives way to the other is the critical batch size.
The gradient noise scale estimates it from two numbers that training already has: the spread of the per-example gradients and the length of their mean.

## The idea

Data-parallel training spreads one batch over many devices, so a larger batch shortens the time of a run only if it cuts the number of steps.
Shallue and colleagues measured the steps needed to reach a goal error across many models and data sets and found the same shape every time. {#c-shallue-universal}
As the batch grows there is first perfect scaling, where each doubling halves the steps, then a region of diminishing returns, and finally a regime of maximal data parallelism where more parallelism gives no benefit. {#c-shallue-regimes}
Where these regimes fall varied greatly between workloads. {#c-shallue-variation}

McCandlish and colleagues explain the shape with one step of SGD on a locally quadratic loss.
A batch gradient $G_B$ is an unbiased estimate of the true gradient $G$ with covariance $\Sigma / B$, where $\Sigma$ is the covariance of the per-example gradients. {#c-batch-gradient}
The best step size and the best progress per step both scale as $1/(1 + B_{\text{noise}}/B)$.
For $B \ll B_{\text{noise}}$ this is about $B / B_{\text{noise}}$, linear in the batch, and for $B \gg B_{\text{noise}}$ it is about 1, flat. {#c-regimes}
Summed over a run, the same factor gives a hyperbola between steps and examples, and its corner is the critical batch size $B_{\text{crit}}$.
The model predicts $B_{\text{crit}} \approx B_{\text{noise}}$, with the noise scale averaged over training. {#c-crit-approx-noise}

The point of the concept is that the corner can be measured before a batch-size sweep is run.
The simplified noise scale $\operatorname{tr}(\Sigma)/\lVert G\rVert^2$ needs only squared gradient norms, which data-parallel training computes on each device before and after averaging. {#c-data-parallel-free}

## Formal statement

Let $L(\theta)$ be the expected loss over the data distribution, $G = \nabla L(\theta)$, $H = \nabla^2 L(\theta)$, and $\Sigma$ the covariance of the per-example gradient $\nabla L_x(\theta)$.
Let $G_B$ be the mean gradient over $B$ examples drawn independently, so $\mathbb{E}[G_B] = G$ and $\operatorname{cov}(G_B) = \Sigma / B$.
Drawn without replacement from $D$ examples, the factor $1/B$ becomes $1/B - 1/D$. {#c-without-replacement}

**One step.**
To second order, the expected change of the loss after the step $\theta - \varepsilon G_B$ is

$$ \mathbb{E}[L(\theta - \varepsilon G_B)] - L(\theta) = -\varepsilon \lVert G\rVert^2 + \frac{\varepsilon^2}{2}\Big(G^\top H G + \frac{\operatorname{tr}(H\Sigma)}{B}\Big). $$ {#eq-expected-step}

Its minimum over $\varepsilon$ is at

$$ \varepsilon_{\text{opt}}(B) = \frac{\varepsilon_{\max}}{1 + B_{\text{noise}}/B}, \qquad \Delta L_{\text{opt}}(B) = \frac{\Delta L_{\max}}{1 + B_{\text{noise}}/B}, \qquad \varepsilon_{\max} = \frac{\lVert G\rVert^2}{G^\top H G}, \quad \Delta L_{\max} = \frac{\lVert G\rVert^4}{2\,G^\top H G}, $$ {#eq-eps-opt}

with the gradient noise scale

$$ B_{\text{noise}} = \frac{\operatorname{tr}(H\Sigma)}{G^\top H G}. $$ {#eq-b-noise}

The optimal step and the gain it brings are derived in [one-step-improvement](derivations/one-step-improvement.md). {#c-one-step-optimum}
A step larger than $2\varepsilon_{\text{opt}}$ raises the expected loss, and in training it can lead to divergence. {#c-overshoot}
The noise scale does not depend on the size of the training set. {#c-no-dataset-size}

**The simplified noise scale.**
If $H$ is a multiple of the identity, $B_{\text{noise}}$ reduces to

$$ B_{\text{simple}} = \frac{\operatorname{tr}(\Sigma)}{\lVert G\rVert^2}, $$ {#eq-b-simple}

which needs no Hessian. {#c-simple-isotropic}
It is also the batch size at which the relative error of the batch gradient is one: $\mathbb{E}\lVert G_B - G\rVert^2 / \lVert G\rVert^2 = B_{\text{simple}}/B$. {#c-simple-relative-error}
McCandlish and colleagues report that the two noise scales usually differ only by a small constant factor. {#c-simple-vs-noise}

**The trade-off and the critical batch size.**
Train at a fixed batch size $B$ to a fixed loss, in $S$ steps and $E = BS$ examples.
Let $S_{\min}$ be the fewest steps at any batch size and $E_{\min}$ the fewest examples.
If each step makes the progress of eq-eps-opt, then

$$ \Big(\frac{S}{S_{\min}} - 1\Big)\Big(\frac{E}{E_{\min}} - 1\Big) = 1, \qquad B_{\text{crit}} = \frac{E_{\min}}{S_{\min}}. $$ {#eq-tradeoff}

The hyperbola is derived in [step-example-tradeoff](derivations/step-example-tradeoff.md), where $B_{\text{crit}}$ comes out as $B_{\text{noise}}$ averaged over the steps of a full-batch run. {#c-tradeoff-hyperbola}
McCandlish and colleagues define $B_{\text{crit}}$ by fitting eq-tradeoff to measured runs. {#c-crit-definition}
At $B = B_{\text{crit}}$ a run takes twice the fewest steps and twice the fewest examples. {#c-crit-twice}

**Measuring the noise scale.**
From squared norms of a small-batch and a big-batch gradient, McCandlish and colleagues form

$$ \widehat{\lVert G\rVert^2} = \frac{B_{\text{big}}\lVert G_{B_{\text{big}}}\rVert^2 - B_{\text{small}}\lVert G_{B_{\text{small}}}\rVert^2}{B_{\text{big}} - B_{\text{small}}}, \qquad \hat S = \frac{\lVert G_{B_{\text{small}}}\rVert^2 - \lVert G_{B_{\text{big}}}\rVert^2}{1/B_{\text{small}} - 1/B_{\text{big}}}, $$ {#eq-two-batch}

which are unbiased for $\lVert G\rVert^2$ and $\operatorname{tr}\Sigma$, as shown in [unbiased-norm-estimators](derivations/unbiased-norm-estimators.md). {#c-two-batch-unbiased}
With $B_{\text{small}} = 1$ and the per-example norms averaged over a batch of $n$, $\hat S$ is the sample variance with the Bessel divisor $n - 1$. {#c-per-example-bessel}
Their ratio is not an unbiased estimate of $B_{\text{simple}}$. {#c-ratio-biased}
The paper therefore keeps an exponential moving average of each across steps and divides the averages. {#c-ema}

**The language-model fit.**
For Transformer language models, Kaplan and colleagues fit

$$ B_{\text{crit}}(L) \approx \frac{B_*}{L^{1/\alpha_B}}, \qquad B_* \approx 2 \times 10^8 \text{ tokens}, \quad \alpha_B \approx 0.21, $$ {#eq-kaplan}

with $L$ the test loss in nats per token. {#c-kaplan-power-law}

The reference code is in [impl/numpy/critical_batch_size.py](impl/numpy/critical_batch_size.py), with a PyTorch version in [impl/pytorch/critical_batch_size.py](impl/pytorch/critical_batch_size.py) that also computes per-example gradients.

## Examples

**One step on a quadratic (our run).**
The check of [one-step-improvement](derivations/one-step-improvement.md) builds 64 examples in 8 dimensions with a quadratic loss whose Hessian eigenvalues run from 0.3 to 3.
At the chosen point it has $B_{\text{noise}} = 17.6$, $B_{\text{simple}} = 11.6$ and $\varepsilon_{\max} = 0.675$.
The formula gives $\varepsilon_{\text{opt}}(16) = 0.675/(1 + 17.6/16) = 0.321$, and the best step found by averaging 200,000 sampled batches is 0.3215.
Going from $B = 1$ to $B = 4$ multiplies the gain per step by 3.5, close to linear, while going from 256 to 1,024 multiplies it by 1.05.
The two noise scales differ here by a factor of 1.5, because the Hessian is far from a multiple of the identity.

**A whole run (our run).**
The check of [step-example-tradeoff](derivations/step-example-tradeoff.md) trains a 30-dimensional noisy quadratic by SGD at the optimal step, at 16 batch sizes from 2 to 65,536.
The steps fit $S = 18.7 + 8977/B$ within 11%, which gives $B_{\text{crit}} = 480$.
A full-batch run takes 21 steps, and the noise scale averaged over them is 451.
At $B = 512$ the run takes 34 steps; at $B = 4096$ it still takes 20.5.
Eight times the compute saves 40% of the steps, which is the large-batch regime.

**Language models.**
Kaplan's fit, eq-kaplan with $B_* = 2.1 \times 10^8$ tokens, gives $2.1\times 10^8 / 3^{4.76} \approx 1.1$ million tokens at a loss of 3 nats and about 2.7 million at 2.5 nats (our arithmetic).
The paper puts the critical batch size at roughly 1 to 2 million tokens at convergence for its largest models. {#c-kaplan-million}
Because $1/\alpha_B \approx 4.76$, a fall in the loss by a factor $0.87$ raises the critical batch size by $0.87^{-4.76} \approx 1.94$; the paper states that it roughly doubles for every 13% decrease in the loss. {#c-kaplan-doubles}

**A borderline case: a measurement taken at the wrong learning rate.**
The noise scale depends on the learning rate through a "temperature", $T \approx \varepsilon/B$ for SGD at small batch, and at equilibrium on a quadratic both noise scales vary as $1/T$. {#c-temperature}
So the noise scale is inflated when the learning rate is too small, and a measurement is useful only at a learning rate suited to the current point. {#c-lr-too-small}
A noise scale read off a run with a badly tuned rate is a number about that run, not the critical batch size of the task.

**A non-example: the spread across tasks.**
There is no single critical batch size for deep learning.
McCandlish and colleagues found that it ranged from about 20 for an SVHN autoencoder to over 10 million for Dota, and the noise scale predicted it at the order-of-magnitude level throughout. {#c-range}
On ImageNet the noise scale varied from 2,000 to 100,000 in the main phase of training. {#c-imagenet}
Over a single run, the critical batch size typically grows by an order of magnitude or more. {#c-crit-grows}

## Related ideas

- **The linear scaling rule** ([[linear-scaling-rule]]) multiplies the learning rate by $k$ when the batch grows $k$ times, and eq-eps-opt shows why this works for $B \ll B_{\text{noise}}$, where $\varepsilon_{\text{opt}} \approx \varepsilon_{\max} B / B_{\text{noise}}$ is linear in $B$, and why it stops near the critical batch size.
- **Learning-rate warmup** ([[learning-rate-warmup]]) ramps the rate up at the start of training, when the noise scale is still small and a large batch is far above it.
- **Gradient accumulation** ([[al:gradient-accumulation]]) reaches a large batch on one device by summing gradients over several passes; it changes the wall-clock trade-off but not where the critical batch size lies.
- **Layer-wise adaptive rates** ([[al:layer-wise-adaptive-rate-scaling]]) aim to train at larger batches; McCandlish and colleagues note that their ImageNet critical batch sizes were lower than those reported with such techniques (15k against 64k). {#c-imagenet-lars}
- **Adaptive batch sizes**: if the batch can change during a run, the trade-off is best when the batch grows as the square root of the noise scale, and the predicted gain is modest unless the noise scale changes greatly. {#c-adaptive-sqrt}
- **Learning-rate schedules** ([[al:learning-rate-schedule]]) change the temperature of training as they decay the rate, so the noise scale moves with them; on ImageNet, with three decays, both $B_{\text{simple}}$ and $B_{\text{crit}}$ changed significantly during training and had to be measured separately at each learning rate. {#c-schedule}
- **Scaling laws** ([[neural-scaling-laws]]) use the critical batch size to convert a run at any batch size to its minimum steps and minimum compute, $S_{\min} = S/(1 + B_{\text{crit}}/B)$ and $C_{\min} = C/(1 + B/B_{\text{crit}})$. {#c-kaplan-conversion}

## Common confusions

**The critical batch size is not a property of the model.**
The parameter count cancels in the noise scale, so it should not depend strongly on model size at a fixed loss; larger models reach lower loss, and through that they have larger noise scales. {#c-model-size}
Kaplan and colleagues found the critical batch size of their language models independent of model size, a function of the loss only. {#c-kaplan-size}

**Smith and Le's noise scale is a different quantity.**
Smith and Le call $g = \varepsilon(N/B - 1) \approx \varepsilon N/B$ the noise scale of SGD, with $N$ the training-set size, and predict an optimal batch size proportional to $\varepsilon N$. {#c-smith-le}
Their optimum maximizes test accuracy at a fixed learning rate, while $B_{\text{crit}}$ is about training speed at a tuned rate.
McCandlish and colleagues note that Smith and Le predict a dependence on the size of the data set that they did not observe. {#c-smith-le-dataset}
$B_{\text{noise}}$ has no $N$ in it, and $g$ has no gradient statistics.

**$B_{\text{simple}}$ is an estimate, not the critical batch size.**
The simplified noise scale drops the Hessian, and the step-averaged noise scale is a model of $B_{\text{crit}}$, not its definition.
For the generative models of McCandlish and colleagues, $B_{\text{simple}}$ was significantly smaller than $B_{\text{crit}}$, and on MNIST $B_{\text{crit}}$ was significantly smaller than $B_{\text{simple}}$. {#c-simple-misses}

**A larger batch is not a worse model.**
Shallue and colleagues found no evidence that larger batch sizes degrade out-of-sample performance, once the learning rate and other parameters are tuned for each batch size. {#c-shallue-no-degrade}
Above the critical batch size, the cost of a large batch is compute, not accuracy.

**The ratio of the two estimates is not the estimate.**
Dividing $\hat S$ by $\widehat{\lVert G\rVert^2}$ at each step and averaging gives a biased answer, and $\widehat{\lVert G\rVert^2}$ can be negative at a single step.
In our run of the check of [unbiased-norm-estimators](derivations/unbiased-norm-estimators.md), the median of the per-step ratio is 6.6 where the true value is 9.67, while the ratio of the averages is 9.60.

## History

The idea that gradient noise should guide the choice of batch size is older than the name; McCandlish and colleagues cite earlier work that used the gradient noise scale to pick batch sizes. {#c-earlier-work}
McCandlish and colleagues introduced the gradient noise scale and the critical batch size in December 2018 and tested the prediction on supervised learning, reinforcement learning and generative models. {#c-history-mccandlish}
Shallue and colleagues, working in parallel, measured steps against batch size for many workloads and named the three regimes; momentum let a workload use much larger batches than plain SGD. {#c-shallue-momentum}
Earlier, Smith and Le had tied the batch size to a noise scale of SGD through generalization, not speed.
Kaplan and colleagues in 2020 used the critical batch size to separate the effect of batch size from their scaling laws and fit it as a power law in the loss.
