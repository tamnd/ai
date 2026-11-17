---
tag: 3FZB17
kind: method
slug: rms-normalization
name: RMSNorm
aliases: [RMSNorm, root mean square layer normalization, RMS normalization, pRMSNorm]
area: deep-learning-training/normalization
level: core
summary: >-
  Divide the summed inputs of a layer by their root mean square over its units, per example, then apply a learned gain; layer normalization without the mean, invariant to rescaling but not to shifts.
purpose: the scale control of layer normalization at a lower cost, in recurrent networks and Transformers
uses: [sample-mean@m2, backpropagation@al]
related_to: [weight-normalization@al, gradient-checking@al, exploding-gradient-problem@al, transformer]
hyperparameters:
  - {name: eps, meaning: "the constant added to the mean square before the square root", typical: "a small constant such as 1e-6; with eps = 0 the invariances are exact, with eps > 0 they hold only when the mean square is large against eps"}
  - {name: p, meaning: "the fraction of the units whose mean square estimates the RMS, in pRMSNorm", typical: "1, plain RMSNorm; Zhang and Sennrich tried 6.25%"}
  - {name: gain, meaning: "whether there is a learned gain for every normalized unit, and its initial value", typical: "on, initialized to 1; there is no bias"}
  - {name: placement, meaning: "where the layer sits relative to the residual connection of a Transformer block", typical: "before each sub-layer, as in LLaMA, with a final normalization before the output layer"}
cost:
  train: "one mean of squares per example over the normalized units and two elementwise operations per activation; in the backward pass one dot product per example; no mean and no centring, so one reduction fewer than layer normalization in each direction"
  inference: "the same as the forward pass in training; the statistic depends on the input, so the layer cannot be folded into the weights"
history:
  year: 2019
  people: [Biao Zhang, Rico Sennrich]
  note: "Zhang and Sennrich, arXiv 1910.07467 (16 Oct 2019), NeurIPS 2019, from the University of Edinburgh and the University of Zurich."
prereqs: [backpropagation@al, sample-mean@m2, standard-deviation@m2, orthogonal-projection@m2]
same_as:
  wikidata: null
  wikipedia: null
  arxiv: ["1910.07467"]
  al: null
sources:
  - {paper: "Zhang and Sennrich (2019)", title: "Root Mean Square Layer Normalization", locator: "the abstract, Sections 1, 4, 4.1, 4.2 and 5 to 7, Figure 1, pp. 1 to 9", note: "the method, pRMSNorm, the invariances, the gradients, the experiments and the conclusion"}
  - {paper: "Ba, Kiros and Hinton (2016)", title: "Layer Normalization", locator: "the abstract, p. 2", note: "layer normalization, which RMSNorm simplifies"}
  - {paper: "Touvron et al. (2023)", title: "LLaMA: Open and Efficient Foundation Language Models", locator: "p. 3", note: "pre-normalization with RMSNorm in a large language model"}
  - {paper: "Xiong et al. (2020)", title: "On Layer Normalization in the Transformer Architecture", locator: "the abstract, Section 2", note: "the Pre-LN placement and training without warm-up"}
---

RMSNorm is a normalization layer that divides the summed inputs of a layer by their root mean square over the units, for each example on its own, and then multiplies each unit by a learned gain.
It is [[layer-normalization]] without the mean: no centring, no bias, and one statistic instead of two.
It keeps the invariance to rescaling the inputs and the weights, and gives up the invariance to shifts.
Zhang and Sennrich (2019) proposed it to cut the cost of layer normalization, and LLaMA uses it in every Transformer block.

## In plain words

Inside a deep network the size of the signal drifts as the weights change.
Layer normalization fixes this at each layer by subtracting the mean of the units and dividing by their spread.
RMSNorm only divides.
It takes the vector of summed inputs of one example, computes the square root of the mean of the squares, and divides every entry by that number.
After this the vector has a root mean square of 1, whatever its size was before.
A learned gain per unit then sets the scale the next layer sees.
If the input has mean zero, this is exactly layer normalization; if not, the output keeps part of the mean.

## Setting

RMSNorm applies wherever layer normalization does: to a vector of summed inputs for each example, in a feed-forward, recurrent or Transformer layer.
Write $\mathbf{a} = W\mathbf{x} \in \mathbb{R}^n$ for the summed inputs of a layer with $n$ units.
Layer normalization computes one mean and one standard deviation per example, used by every unit of the layer; each example gets its own pair. {#c-ln-shared}
So it computes the same function in training and at test time, unlike [[batch-normalization]], which normalizes each unit over the batch. {#c-ln-train-test}

The trouble is the price.
Zhang and Sennrich observe that the overhead of layer normalization counter-balances its gain: training needs fewer steps, but each step costs more. {#c-ln-overhead}
In a GRU-based RNNSearch, when the training loss of the model without normalization reaches 7.0, the model with LayerNorm is at 5.4 after the same number of steps but only at 5.9 after the same training time. {#c-overhead-figure}
They hypothesize that the re-scaling invariance of layer normalization, not its re-centering invariance, is the reason for its success, so that the centring can be dropped. {#c-hypothesis}
They also argue that the mean normalization does not reduce the variance of hidden states or of gradients. {#c-mean-argument}

## The method

RMSNorm normalizes with the root mean square and applies a gain $\mathbf{g}$,

$$ \bar{a}_i = \frac{a_i}{\mathrm{RMS}(\mathbf{a})}\, g_i, \qquad \mathrm{RMS}(\mathbf{a}) = \sqrt{\frac{1}{n}\sum_{i=1}^n a_i^2 + \epsilon} . $$ {#eq-rms}

RMSNorm regularizes the summed inputs according to the RMS statistic alone. {#c-rms-def}
This is Eq. 4 of Zhang and Sennrich, plus the small constant $\epsilon$ that implementations add for safety.
The gain re-scales the normalized summed inputs and starts at 1. {#c-gain-init}
There is no bias in the layer; a bias, if wanted, belongs to the next linear map.
The RMS measures the quadratic mean of the inputs and forces them into a $\sqrt{n}$-scaled unit sphere, so that the output distribution does not depend on the scaling of the inputs and of the weights. {#c-sphere}
The factor $1/n$ inside the root matters.
The plain Euclidean norm, which drops it, did not work for layer normalization in their experiments, and they hypothesize that scaling the sphere with the size of the vector is important. {#c-l2-fails}

**pRMSNorm.**
Partial RMSNorm estimates the RMS from the first $k = \lceil np \rceil$ units only, for a fraction $p \in (0, 1]$, {#c-prms-def}

$$ \overline{\mathrm{RMS}}(\mathbf{a}) = \sqrt{\frac{1}{k}\sum_{i=1}^k a_i^2 + \epsilon}, \qquad \bar{a}_i = \frac{a_i}{\overline{\mathrm{RMS}}(\mathbf{a})}\, g_i \quad (i = 1, \dots, n) . $$ {#eq-prms}

Every unit is divided by the partial statistic, not only the first $k$.
The motivation is that the units of one layer are independent and identically distributed, so a few of them estimate the RMS of all. {#c-prms-iid}

**Gradient.**
Write $r$ for $\mathrm{RMS}(\mathbf{a})$, $\boldsymbol{\delta} = \partial\ell/\partial\bar{\mathbf{a}}$ for the output gradient and $\mathbf{u} = \mathbf{g}\odot\boldsymbol{\delta}$.
The backward pass is

$$ \frac{\partial \ell}{\partial \mathbf{a}} = \frac{1}{r}\Big(\mathbf{u} - \frac{\mathbf{a}^\top\mathbf{u}}{n\,r^2}\,\mathbf{a}\Big), \qquad \frac{\partial \ell}{\partial \mathbf{g}} = \boldsymbol{\delta}\odot\frac{\mathbf{a}}{r} . $$ {#eq-rms-grad}

The derivation `gradient` gives the steps and the pRMSNorm form, and the tests compare eq-rms-grad with finite differences, as in gradient checking, and with autograd.

**In a Transformer.**
LLaMA normalizes the input of each sub-layer instead of the output, to improve training stability, and uses the RMSNorm function. {#c-llama}
This is the Pre-LN placement of Xiong et al., which puts the normalization inside the residual branch and adds a final normalization right before the prediction. {#c-preln}

## Derivation

Three facts carry the method.
Each is proved in a derivation file, with its assumptions and a numeric check.

**Invariances** (derivation `invariances`).
With $\epsilon = 0$ and $\delta > 0$, RMSNorm of $\delta\mathbf{a}$ is RMSNorm of $\mathbf{a}$, because the RMS is homogeneous of degree one; so scaling the whole weight matrix or the input does not change the output. {#c-scale-invariance}
Zhang and Sennrich state this invariance to weight matrix and input re-scaling, and note that it does not hold when only individual weight vectors are scaled. {#c-zhang-invariance}
With $\epsilon > 0$ the squared denominator at $\delta\mathbf{a}$ is $\delta^2 m + \epsilon$, with $m$ the mean square, so the invariance is only approximate. {#c-scale-invariance-eps}
Scaling one row alone breaks it: with $W = I_2$ and $\mathbf{x} = (1, 1)$, doubling the first row moves the second output from 1 to $\sqrt{2/5}$. {#c-row-not-invariant}
Shifts break it too: $(1, -1)$ maps to $(1, -1)$ but $(2, 0)$ maps to $(\sqrt{2}, 0)$, while layer normalization maps both to $(1, -1)$. {#c-not-recentering}
Zhang and Sennrich say as much: RMSNorm is not invariant to all re-centering operations. {#c-zhang-not-recentering}
The two layers are linked by one identity, for every $\epsilon \ge 0$ and no bias: {#c-ln-is-rms-of-centred}

$$ \mathrm{LN}(\mathbf{a}) = \mathrm{RMSNorm}(\mathbf{a} - \mu\mathbf{1}), \qquad \mu = \frac{1}{n}\sum_{i=1}^n a_i . $$ {#eq-ln-as-rms}

Layer normalization is RMSNorm applied after centring, so it pays for one more reduction and buys shift invariance.
pRMSNorm keeps the scale invariance, because the partial RMS also scales by $\delta$. {#c-prms-invariance}
Zhang and Sennrich note the same: the linearity of the RMS still holds for pRMSNorm. {#c-zhang-prms-invariance}

**The gradient** (derivation `gradient`).
The Jacobian is $\frac{\mathrm{diag}(\mathbf{g})}{r}\big(I - \frac{\mathbf{a}\mathbf{a}^\top}{n r^2}\big)$, which gives eq-rms-grad. {#c-jacobian}
With $\epsilon = 0$, $n r^2 = \lVert\mathbf{a}\rVert^2$, so the bracket is the projection onto the complement of $\mathbf{a}$: the input gradient is orthogonal to $\mathbf{a}$, and a step along $\mathbf{a}$ does nothing to first order. {#c-grad-orthogonal}
With $\epsilon > 0$ the projection leaks: $\mathbf{a}^\top\partial\ell/\partial\mathbf{a} = (\mathbf{a}^\top\mathbf{u}/r)\cdot\epsilon/(m + \epsilon)$. {#c-grad-eps-leak}
The gain gradient is the output gradient times the normalized input. {#c-gain-grad}
With $\epsilon = 0$ the input gradient at $\delta\mathbf{a}$ is $1/\delta$ times the gradient at $\mathbf{a}$, so the weight gradient is unchanged when the input is scaled and shrinks by $1/\delta$ when the weights are scaled by $\delta$. {#c-grad-inverse-scale}
For pRMSNorm only the first $k$ entries enter the correction term, and the gradient is still orthogonal to $\mathbf{a}$ at $\epsilon = 0$. {#c-prms-grad}
Zhang and Sennrich prove that the gradients of the gain and the bias are invariant to the scaling of the inputs and the weights. {#c-zhang-grad-g}
They read the $1/\delta$ factor of the weight gradient as an implicit learning rate adaptor: large weights get small gradients. {#c-zhang-weight-grad}

**Mean zero** (derivation `zero-mean`).
If $\mu = 0$, RMSNorm and layer normalization with the same $\epsilon$ and gain agree exactly, for every $\epsilon$. {#c-zero-mean-equal}
Zhang and Sennrich note this case: with a zero mean, RMSNorm is exactly LayerNorm. {#c-zhang-equal}
If $\mu \ne 0$, set $\epsilon = 0$ and $\rho = \mu/\sigma$, with $\sigma$ the biased standard deviation. Then {#c-mean-mix}

$$ \mathrm{RMSNorm}(\mathbf{a}) = \frac{\mathrm{LN}(\mathbf{a}) + \rho\,\mathbf{g}}{\sqrt{1 + \rho^2}} . $$ {#eq-rms-mix}

With $\mathbf{g} = \mathbf{1}$, the RMS of the difference between the two outputs is $\sqrt{2 - 2/\sqrt{1 + \rho^2}}$. It is at most $\lvert\rho\rvert$, about $\lvert\rho\rvert$ for small $\rho$, and tends to $\sqrt{2}$ as $\lvert\rho\rvert$ grows. {#c-gap-formula}
The output mean is $\rho/\sqrt{1 + \rho^2}$, where layer normalization gives 0. {#c-output-mean}
So the two layers differ by how far the mean sits from zero in units of the spread.
A layer whose inputs are roughly centred loses little by dropping the centring.

## Worked example

Take $n = 4$, $\mathbf{a} = (4, 0, -2, 4)$, $\mathbf{g} = \mathbf{1}$ and $\epsilon = 0$.
The mean square is $(16 + 0 + 4 + 16)/4 = 9$, so $r = 3$ and RMSNorm gives $(4/3, 0, -2/3, 4/3)$.
Its RMS is 1.
Doubling the input to $(8, 0, -4, 8)$ gives $r = 6$ and the same output.

Layer normalization of the same vector has $\mu = 1.5$ and $\sigma^2 = 6.75$, so it gives $(2.5, -1.5, -3.5, 2.5)/\sqrt{6.75} \approx (0.962, -0.577, -1.347, 0.962)$.
Here $\rho = 1.5/\sqrt{6.75} = 1/\sqrt{3}$ and $\sqrt{1 + \rho^2} = 2/\sqrt{3}$, so eq-rms-mix says RMSNorm is $(\sqrt{3}/2)\,\mathrm{LN} + 1/2$.
For the first unit, $0.866 \times 0.962 + 0.5 = 1.333 = 4/3$.
The output mean is $\rho/\sqrt{1+\rho^2} = 1/2$, and indeed $(4/3 + 0 - 2/3 + 4/3)/4 = 1/2$.
The RMS of the gap is $\sqrt{2 - \sqrt{3}} \approx 0.518$.

For the gradient, let the output gradient be $\boldsymbol{\delta} = (1, 0, 0, 0)$.
Then $\mathbf{u} = \boldsymbol{\delta}$, $\mathbf{a}^\top\mathbf{u} = 4$ and $n r^2 = 36$, so eq-rms-grad gives
$\frac{1}{3}\big((1, 0, 0, 0) - \frac{4}{36}(4, 0, -2, 4)\big) = \frac{1}{27}(5, 0, 2, -4)$.
Its dot product with $\mathbf{a}$ is $(20 + 0 - 4 - 16)/27 = 0$.
At $2\mathbf{a}$ the gradient is half as large.

With pRMSNorm and $p = 0.5$, $k = 2$ and the statistic comes from $(4, 0)$: $r = \sqrt{8}$, and all four entries are divided by it, giving $(\sqrt{2}, 0, -1/\sqrt{2}, \sqrt{2})$.
The tests `test_worked_example` in both implementations check these numbers.

## Hyperparameters

**eps.**
It keeps the division safe when the input is near zero.
It also bends the theory: the invariances and the orthogonality are exact only at $\epsilon = 0$, and the leak is of relative size $\epsilon/(m + \epsilon)$.
For activations of ordinary size this is negligible; for tiny activations in low precision it is not.

**p.**
The fraction of units used by pRMSNorm.
$p = 1$ is RMSNorm.
Smaller $p$ saves arithmetic in theory, and the experiments below show little gain in practice.

**gain.**
One learned scale per unit, initialized to 1.
Without it the layer pins every output vector to RMS 1.

**placement.**
Before each sub-layer of a Transformer, inside the residual branch, with a final normalization before the output layer, as in LLaMA.

## Cost

Per example, the forward pass computes one sum of squares over $n$ units, one square root, and one multiply per unit for the division and one for the gain.
Layer normalization needs the mean first and then the variance, so RMSNorm saves one reduction and one subtraction per unit.
The backward pass needs one dot product $\mathbf{a}^\top\mathbf{u}$ per example, against two for layer normalization.
Memory is the input, or the normalized output, and $r$ per example.
These savings are small in floating point operations, so the measured speedups depend on the implementation.

## Evidence

Zhang and Sennrich report that across several tasks and architectures RMSNorm matches LayerNorm while cutting running time by 7% to 64%. {#c-speed-range}

**Translation with RNNSearch.**
Both LayerNorm and RMSNorm reduce the training steps to convergence by about 50% against the model without normalization, and RMSNorm is comparable to LayerNorm in test accuracy. {#c-rnnsearch-steps}
In TensorFlow, LayerNorm is about 67% slower per step than no normalization, and RMSNorm recovers about 25%. {#c-rnnsearch-speed}
In Theano and PyTorch the speedups of RMSNorm over LayerNorm are 11% to 34%, with comparable quality. {#c-theano-pytorch}
The mean of the RMSNorm activations, though never normalized, is in practice more stable than without normalization, and both layers stabilize the standard deviation. {#c-mean-stable}
With the centre of the weight initialization moved to 0.2, LayerNorm becomes very unstable while RMSNorm is more robust, though both do worse. {#c-init-robust}

**Translation with a Transformer.**
In a base Transformer trained for 300K steps, training fails without normalization, and RMSNorm matches LayerNorm in BLEU with a speedup of 7% to 9%. {#c-transformer}

**Other tasks.**
On an attentive reader for the CNN corpus, LayerNorm and RMSNorm converge faster than BatchNorm and reach a lower validation error, and pRMSNorm is slightly worse than RMSNorm. {#c-reader}
RMSNorm is about 15% faster than LayerNorm there. {#c-reader-speed}
On an order-embedding model for image-caption retrieval, RMSNorm speeds up training by 40% to 64% over LayerNorm. {#c-oe-speed}
On CIFAR-10 with a ConvPool-CNN-C model, BatchNorm performs best of the normalizations compared. {#c-cifar-bn}
LayerNorm increases the test error of the model without normalization by 1.53%, while RMSNorm reduces it by 0.013% and is about 20.5% faster than LayerNorm. {#c-cifar}

**Placement.**
Pre-LN Transformers trained without the learning rate warm-up reach results comparable to the baselines with less training time and less tuning. {#c-preln-warmup}
That result is for layer normalization; LLaMA uses the same placement with RMSNorm.

All these results come from one paper per claim, at the model sizes of 2019, and the timings depend on the framework of the time.

## Limitations and failure modes

**Shifts are not removed.**
If the inputs of a layer carry a large common offset, RMSNorm divides by a number dominated by the offset, and the output carries a mean of $\rho/\sqrt{1+\rho^2}$.
By eq-rms-mix, the normalized deviations then shrink by $1/\sqrt{1+\rho^2}$.
Whether this hurts depends on the network; the derivation `zero-mean` measures the gap.

**The speedup is not guaranteed.**
Zhang and Sennrich note that the actual speed improvements depend on the framework, the hardware, the architecture and the cost of the other components. {#c-speed-depends}

**pRMSNorm is fragile.**
With few units in the statistic the gradients tend to explode, an instance of the exploding gradient problem, though models with $p = 6.25\%$ still converge. {#c-prms-unstable}
The ratio $p$ has little influence on the final translation quality. {#c-p-little-influence}
But pRMSNorm with $p = 6.25\%$ is sometimes slower than RMSNorm, which the authors put down to slow tensor slicing in the frameworks. {#c-prms-slow}
They did not consistently observe speed improvements for pRMSNorm. {#c-prms-no-speed}
On CIFAR-10 it gains a further 2.6% in speed at the cost of test accuracy. {#c-prms-cifar}

**Zero input.**
With $\epsilon = 0$ an all-zero vector gives $0/0$.
Keep $\epsilon > 0$ in practice and set it to 0 only to test the identities.

## Variants

- **Layer normalization** is RMSNorm of the centred input, eq-ln-as-rms, with a bias.
- **pRMSNorm** estimates the RMS from a fraction $p$ of the units, eq-prms.
- **The L2 norm** without the $1/n$ factor normalizes to a fixed sphere; it did not work in the experiments of Zhang and Sennrich.
- **No gain.** With `elementwise_affine=False` in the implementations the output has RMS 1 exactly, up to $\epsilon$.
- **Weight normalization** rescales the weight vector of each unit instead of the summed inputs; it acts on the parameters, while RMSNorm acts on the activations of each example.

## When to use it

Use RMSNorm where layer normalization would go and the inputs are close to centred, or where the next layer can absorb an offset.
LLaMA uses it inside Pre-LN blocks.
Prefer layer normalization when shift invariance matters, for example when a large common offset is expected in the summed inputs.
For convolutional image models with large batches, batch normalization did best in the CIFAR-10 experiments above.
Treat pRMSNorm as an experiment, not a default.

## Pitfalls

> **Putting eps in the wrong place.** RMSNorm adds $\epsilon$ to the mean square, inside the root. Adding it to the RMS outside the root gives a different layer, and two implementations then disagree for small inputs. `test_torch_rms_norm_agrees` and `test_matches_torch_rms_norm` compare the implementations with the PyTorch function.

> **Expecting layer normalization.** RMSNorm and layer normalization agree only for zero-mean inputs. A port that swaps one for the other changes the outputs by eq-rms-mix. `test_layer_norm_is_rms_of_centred_input` and `test_mean_gap` check the identity and the gap.

> **Slicing pRMSNorm wrongly.** Only the first $k$ units enter the statistic, but all $n$ units are divided by it. `test_partial_uses_only_first_k` checks that the last $n - k$ entries do not change the statistic.

> **Trusting the invariance with eps.** Tests of scale invariance or of orthogonal gradients must set $\epsilon = 0$, or allow for the leak. `test_eps_breaks_scale_invariance` and `test_eps_leak_of_orthogonality` check the leak.

> **Normalizing the wrong axis.** The statistic is over the units of one example, the last axis, never over the batch.

## History

Ba, Kiros and Hinton introduced layer normalization in 2016.
Zhang and Sennrich proposed RMSNorm in October 2019, at NeurIPS 2019, as a simple drop-in replacement of LayerNorm. {#c-drop-in}
As far as they knew, the RMS had not been used for neural network normalization before. {#c-first-rms}
They proposed pRMSNorm in the same paper, and planned to explore other norms and to simplify other normalization techniques such as BatchNorm. {#c-future-norms}
Xiong et al. (2020) analysed the Pre-LN placement.
LLaMA (Touvron et al., 2023) combined the Pre-LN placement with RMSNorm.
