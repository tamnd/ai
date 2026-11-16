---
tag: NEW-group-normalization
kind: method
slug: group-normalization
name: Group normalization
aliases: [GroupNorm, GN, group norm]
area: deep-learning-training/normalization
level: core
summary: >-
  Normalize each group of channels of each example by its own mean and variance, then scale and shift each channel; independent of the batch, with LN at G = 1 and IN at G = C.
purpose: normalization for convolutional networks trained with small batches, as in detection, segmentation and video
uses: [sample-mean@m2, variance@m2, backpropagation@al]
related_to: [weight-normalization@al, gradient-checking@al, image-classification, imagenet]
hyperparameters:
  - {name: G, meaning: "the number of groups; C/G channels share one mean and one variance", typical: "32, the default of Wu and He; any G that divides C works, and Table 3 shows little difference from 2 to 64 groups"}
  - {name: channels per group, meaning: "an alternative to a fixed G, so that G changes from layer to layer", typical: "16, with results close to G = 32 in Table 3"}
  - {name: eps, meaning: "the constant added to the variance inside the square root", typical: "a small constant such as 1e-5"}
  - {name: gamma init, meaning: "the initial value of the per-channel scale", typical: "1, and 0 in the last normalization layer of each residual block, as in Wu and He"}
  - {name: weight decay on gamma and beta, meaning: "whether the per-channel scale and shift are decayed", typical: "1e-4 in ImageNet pre-training in Wu and He; 0 when fine-tuning detectors"}
cost:
  train: "one mean and one variance per group per example, over C/G channels and all positions; the same arithmetic as layer or instance normalization; no running statistics and no communication across workers"
  inference: "the same as the forward pass in training; the statistics depend on the input, so the layer cannot be folded into the preceding convolution as frozen batch normalization can"
history:
  year: 2018
  people: [Yuxin Wu, Kaiming He]
  note: "Wu and He, arXiv 1803.08494v3 (11 Jun 2018), from Facebook AI Research."
prereqs: [batch-normalization@al, layer-normalization@al, backpropagation@al, sample-mean@m2, variance@m2]
same_as:
  wikidata: null
  wikipedia: null
  arxiv: ["1803.08494"]
  al: null
sources:
  - {paper: "Wu and He (2018)", title: "Group Normalization", locator: "the abstract, pp. 1 to 9, Tables 1 to 8", note: "the general form, the method, the relation to LN and IN, and the ImageNet, COCO and Kinetics experiments"}
  - {paper: "Ioffe and Szegedy (2015)", title: "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift", locator: "pp. 4 and 5", note: "batch normalization over the batch and the positions of a feature map, and its population statistics at inference"}
  - {paper: "Ba, Kiros and Hinton (2016)", title: "Layer Normalization", locator: "the abstract, Sections 3 and 6.7", note: "layer normalization, the G = 1 case"}
  - {paper: "Ulyanov, Vedaldi and Lempitsky (2016)", title: "Instance Normalization: The Missing Ingredient for Fast Stylization", locator: "the abstract, p. 4", note: "instance normalization, the G = C case"}
  - {paper: "Qiao et al. (2019)", title: "Micro-Batch Training with Batch-Channel Normalization and Weight Standardization", locator: "pp. 1 and 2", note: "weight standardization combined with group normalization"}
---

Group normalization (GN) is a normalization layer for convolutional networks.
It splits the channels of each example into $G$ groups and normalizes each group by its own mean and variance.
A learned scale and shift per channel follow.
The batch never enters the statistics, so the layer computes the same function for any batch size, in training and at test time.
With $G = 1$ it is [[layer-normalization]] over all channels and positions, and with $G = C$ it is instance normalization.
Wu and He (2018) proposed it for the cases where [[batch-normalization]] fails: small batches in detection, segmentation and video.

## In plain words

Batch normalization takes one channel and averages it over every image in the batch and every pixel.
That works when the batch is large.
With two images per GPU, the average is noisy and the network does worse.

Group normalization averages over one image only.
It takes a block of neighbouring channels, say 8 of 256, and computes one mean and one spread over all their pixels.
It subtracts the mean and divides by the spread, then lets each channel rescale itself.
The other images in the batch play no part.
So a batch of 2 gives the same output for each image as a batch of 32, and the layer behaves the same way at test time as in training.

Two settings of the group size give older layers.
One group holding every channel is layer normalization.
One channel per group is instance normalization.
Group normalization sits between them.

## Setting

A convolutional feature map has the layout $(N, C, H, W)$: batch, channels, height and width.
Batch normalization treats a convolutional layer by normalizing all locations of a feature map in the same way, with one pair of $\gamma$ and $\beta$ per feature map, over an effective mini-batch of $m \cdot p \cdot q$ values. {#c-bn-conv}
At inference it switches to population statistics gathered during training. {#c-inference-population}
So its training statistics depend on the batch.

This is a problem when the batch is small.
The error of batch normalization rises quickly as the batch shrinks, which Wu and He put down to inaccurate batch statistics. {#c-small-batch}
Here batch size means images per worker, since many libraries compute the statistics on each worker and do not share them. {#c-per-worker}
Batch normalization wants about 32 images per worker, while detectors in the Fast/er and Mask R-CNN family train with 1 or 2 images and freeze batch normalization into a linear layer. {#c-detection-batch}

Two older layers avoid the batch axis.
Layer normalization computes one mean and one standard deviation over all hidden units of a layer, shared by those units, with a separate pair for each training case. {#c-ln-definition}
It has no constraint on the batch size and works with a batch of 1. {#c-batch-size-one}
It performs the same computation at training and test time. {#c-train-test}
Instance normalization, also called contrast normalization, normalizes each channel of each image over its spatial positions. {#c-in-definition}
Ulyanov et al. swapped it in for batch normalization in a stylization network and applied it at test time as well. {#c-instance-norm}
Both work well for sequential and generative models, but Wu and He note that they have had limited success in visual recognition. {#c-ln-in-limited}

## The method

Wu and He write the four layers in one form. {#c-general-form}
For an index $i = (i_N, i_C, i_H, i_W)$ and a set $\mathcal{S}_i$ of $m$ indices,

$$ \hat{x}_i = \frac{x_i - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}}, \qquad \mu_i = \frac{1}{m}\sum_{k \in \mathcal{S}_i} x_k, \qquad \sigma_i^2 = \frac{1}{m}\sum_{k \in \mathcal{S}_i} (x_k - \mu_i)^2 . $$ {#eq-norm}

The layers differ only in the set. {#c-sets}

$$ \mathcal{S}^{\mathrm{BN}}_i = \{k : k_C = i_C\}, \qquad \mathcal{S}^{\mathrm{LN}}_i = \{k : k_N = i_N\}, \qquad \mathcal{S}^{\mathrm{IN}}_i = \{k : k_N = i_N,\ k_C = i_C\} . $$ {#eq-sets}

Group normalization picks a number of groups $G$ that divides $C$ and uses {#c-gn-set}

$$ \mathcal{S}^{\mathrm{GN}}_i = \Big\{k : k_N = i_N,\ \Big\lfloor \frac{k_C}{C/G} \Big\rfloor = \Big\lfloor \frac{i_C}{C/G} \Big\rfloor \Big\} . $$ {#eq-gn-set}

So the set is one example, one block of $C/G$ consecutive channels, and all their positions.
The default is $G = 32$.
All pixels of a group share one mean and one standard deviation, and the layer learns a scale and a shift per channel: {#c-affine}

$$ y_i = \gamma_{i_C}\,\hat{x}_i + \beta_{i_C} . $$ {#eq-affine}

In code this is a reshape to $(N, G, C/G, H, W)$, a mean and a variance over the last three axes, and a reshape back.
The same code serves video with $(N, C, T, H, W)$.

**Why groups.**
Wu and He point to SIFT and HOG, classical features that are built and normalized in groups. {#c-motivation}
They argue that layer normalization assumes all channels of a layer make similar contributions, which is less valid with convolutions, and that GN asks this only of each group. {#c-ln-assumption}
They also argue that instance normalization sees only the spatial positions and so cannot exploit the dependence between channels. {#c-in-channels}
Their own summary: GN is independent of the batch size and its accuracy is stable across a wide range of batch sizes. {#c-batch-indep-claim}

**Gradient.**
Write $s = \sqrt{\sigma^2 + \epsilon}$ for one group of $m$ entries, $\delta_k = \partial\ell/\partial y_k$ and $u_k = \gamma_{c(k)}\delta_k$, where $c(k)$ is the channel of entry $k$.
With bars for means over the group, the backward pass is

$$ \frac{\partial\ell}{\partial x_k} = \frac{1}{s}\Big(u_k - \bar{u} - \hat{x}_k\,\overline{u\hat{x}}\Big), \qquad \frac{\partial\ell}{\partial\gamma_c} = \sum_{k \in c} \delta_k\hat{x}_k, \qquad \frac{\partial\ell}{\partial\beta_c} = \sum_{k \in c} \delta_k . $$ {#eq-gn-grad}

This is the batch normalization backward pass with the batch replaced by a group.

## Derivation

Four facts carry the method.
Each is proved in a derivation file, with its assumptions and a numeric check.

**The special cases** (derivation `special-cases`).
Wu and He state that GN becomes LN at $G = 1$ and IN at $G = C$. {#c-extremes}
The proof is a look at eq-gn-set.
With $G = 1$, every channel is in group 0, so the set is every index of the same example: GN equals LN over $(C, H, W)$. {#c-g1-is-ln}
With $G = C$, the group of a channel is the channel itself, so the set is the positions of one channel of one example: GN equals IN. {#c-gc-is-in}
With a batch of one example, the condition $k_N = i_N$ is always true, so batch normalization in training mode uses the IN set too. {#c-bn-batch1-is-in}
The check compares these on random $(N, C, H, W)$ tensors to $10^{-12}$.

**Batch independence** (derivation `batch-independence`).
Every member of the GN set has $k_N = i_N$, so the output for an example is a function of that example alone. {#c-batch-independence}
Batch normalization in training mode has no such property. {#c-bn-depends}
For one channel and one position, the batch $(0, 2)$ maps its first example to $-1$, and the batch $(0, 2, 4)$ maps the same example to $-\sqrt{3/2}$.

**Group moments** (derivation `group-moments`).
Within a group, $\hat{x}$ has mean 0 and biased variance $\sigma^2/(\sigma^2 + \epsilon)$, so variance 1 only when $\epsilon = 0$. {#c-group-moments}
The statement is about the group, not its channels.
A channel inside a group can keep any mean, and the per-channel $\gamma$ and $\beta$ then act on that.

**The gradient** (derivation `gradient`).
The quotient rule through $\mu$ and $s$ gives the input gradient of eq-gn-grad. {#c-grad}
It sums to zero over each group, so adding a constant to a group changes nothing to first order. {#c-grad-mean-zero}
Its mean product with $\hat{x}$ is $(\overline{u\hat{x}}/s) \cdot \epsilon/(\sigma^2 + \epsilon)$, so it is orthogonal to $\hat{x}$ only when $\epsilon = 0$. {#c-grad-eps-leak}
The gradients of $\gamma_c$ and $\beta_c$ sum over the examples and the positions of channel $c$. {#c-affine-grad}
The check compares eq-gn-grad with central finite differences, as in gradient checking, for every divisor $G$ of $C$.

## Worked example

Take one example with $C = 4$ channels of 2 positions each, $G = 2$, $\epsilon = 0$, $\gamma = 1$ and $\beta = 0$:

$$ x = \big((0, 0),\ (4, 4),\ (0, 6),\ (6, 8)\big) . $$

Group 1 is channels 0 and 1, the values $(0, 0, 4, 4)$.
Its mean is 2 and its variance is 4, so $s = 2$ and the output is $(-1, -1)$ and $(1, 1)$.
Group 2 is channels 2 and 3, the values $(0, 6, 6, 8)$.
Its mean is 5 and its variance is $(25 + 1 + 1 + 9)/4 = 9$, so $s = 3$ and the output is $(-5/3, 1/3)$ and $(1/3, 1)$.
Each group has mean 0 and variance 1.
Channel 0 alone has mean $-1$ and variance 0, as the derivation `group-moments` allows.

With $G = 1$ the set is all 8 values, with mean 3.5 and variance 8.75, so the output is $(x - 3.5)/\sqrt{8.75}$.
With $G = 4$ the channels are normalized alone.
Channels 0 and 1 are constant, so with a small $\epsilon > 0$ they map to 0, and channels 2 and 3 map to about $(-1, 1)$.
Instance normalization erases the fact that channel 1 was larger than channel 0; GN with 2 groups keeps it.

For the gradient, let the output gradient be 1 on the first entry of channel 2 and 0 elsewhere.
In group 2, $u = (1, 0, 0, 0)$, $\bar{u} = 1/4$ and $\overline{u\hat{x}} = -5/12$.
eq-gn-grad gives $\frac{1}{3}\big((1, 0, 0, 0) - \frac{1}{4} + \frac{5}{12}\hat{x}\big) = \frac{1}{54}(1, -2, -2, 3)$.
It sums to 0, and its dot product with $\hat{x} = (-5/3, 1/3, 1/3, 1)$ is $(-5 - 2 - 2 + 9)/162 = 0$.
Group 1 gets a zero gradient.
The tests `test_worked_example` in both implementations check these numbers.

## Hyperparameters

**G.**
The number of groups.
32 is the default, and Table 3 shows little difference between 2 and 64 groups at 32 images per GPU.
$G$ must divide $C$ in every layer.
A fixed number of channels per group, such as 16, is the alternative, and then $G$ changes with the width of the layer.

**eps.**
It keeps the division safe for a constant group.
The group moments and the orthogonality of the gradient are exact only at $\epsilon = 0$.

**gamma init.**
Wu and He start $\gamma$ at 1, and at 0 in the last normalization layer of each residual block, so that each block starts as the identity.

**Weight decay on gamma and beta.**
In their ImageNet runs Wu and He decay all weights including $\gamma$ and $\beta$ by $10^{-4}$. {#c-setup}
When fine-tuning detectors they turn it off for $\gamma$ and $\beta$, which they find important. {#c-no-wd-gamma}

## Cost

The forward pass computes one mean and one variance per group per example, over $(C/G) \cdot H \cdot W$ values.
That is the same arithmetic as layer or instance normalization, and the reshape costs nothing.
There are no running statistics and no communication between workers.
The backward pass needs two means per group, $\bar{u}$ and $\overline{u\hat{x}}$.
Unlike frozen batch normalization, GN cannot be folded into the preceding convolution at inference, because its statistics depend on the input.

The saving is in memory, not arithmetic.
Wu and He note that removing the batch size constraint of BN can free 16 times or more memory for larger models. {#c-memory}

## Evidence

All numbers below are from Wu and He, except the last paragraph.

**ImageNet, ResNet-50.**
The first test is image classification, where batch normalization works well at large batches, so GN has to match it there and beat it at small ones.
At 32 images per GPU, the validation error is 23.6% for BN, 24.1% for GN, 25.3% for LN and 28.4% for IN. {#c-table1}
GN has lower training error than BN but slightly higher validation error, which Wu and He read as GN losing the regularization that the noisy batch statistics give BN. {#c-train-error}
At 2 images per GPU, BN rises to 34.7% while GN stays at 24.1%. {#c-gn-vs-bn}
With 2 to 64 groups GN stays between 24.1% and 24.7%, against 25.3% for one group, and with 2 to 64 channels per group it stays between 24.2% and 25.6%, against 28.4% for one channel per group. {#c-groups}
With one image, BN is similar to IN at training time, and IN at 28.4% beats BN with two images at 34.7%. {#c-bn-one-image}
Batch renormalization at 4 images per GPU reaches 26.3%, better than BN's 27.3% but 2.1% behind GN's 24.2%. {#c-renorm}

**Other ImageNet models.**
For ResNet-101, BN has 22.0% and GN 22.4% at 32 images per GPU; at 2 images per GPU, BN has 31.9% and GN 23.0%. {#c-resnet101}
VGG-16 has 29.2% without normalization, 28.0% with BN and 27.6% with GN. {#c-vgg}

**COCO detection and segmentation.**
Wu and He fine-tune Mask R-CNN at 1 image per GPU from ImageNet models.
Fine-tuning BN without freezing it loses about 6 AP at 2 images per GPU. {#c-finetune-bn}
With a ResNet-50 C4 backbone, GN beats frozen BN by 1.1 box AP and 0.8 mask AP, while LN is 1.9 box AP behind GN and 0.8 behind frozen BN. {#c-coco-c4}
With an FPN backbone, GN in the box head alone adds 0.9 box AP, to 39.5; BN in the head is about 9 AP worse, because the regions of a batch come from one image and are not i.i.d. {#c-coco-head}
With 270k instead of 180k iterations, the ResNet-50 FPN GN model is 2.2 box AP and 1.6 mask AP above frozen BN, which does not gain from longer training. {#c-coco-long}
GN also trains Mask R-CNN from scratch, without pre-training: the GN models reach 41.0 box AP and 36.4 mask AP. {#c-scratch}

**Kinetics video.**
With ResNet-50 I3D and 32-frame clips, GN is 0.3% top-1 behind BN at 8 clips per GPU. {#c-kinetics}
At 4 clips per GPU, BN drops 1.2% to 72.1% top-1 while GN keeps 72.8%. {#c-kinetics-small}
With 64-frame clips at 4 clips per GPU, GN reaches 74.5% against 73.3% for BN. {#c-kinetics-long}

**Weight standardization.**
Qiao et al. standardize the weights of each convolution to zero mean and unit variance. {#c-ws}
They report that GN with weight standardization at one image per GPU matches or beats BN with large batches on ImageNet, and improves COCO results significantly. {#c-gn-ws}

These results are from two papers, with ResNet-era models and the training recipes of 2018 and 2019.

## Limitations and failure modes

**Lost regularization.**
At large batches, GN is slightly behind BN in validation error on ResNet-50, as above.
The noise of batch statistics helps generalization, and GN does not have it.

**Systems tuned for BN.**
Wu and He note that many systems and their hyperparameters were designed for BN and may not suit GN. {#c-designed-for-bn}
The COCO schedule is one case: the default was too short for GN.

**Channel-based statistics.**
Qiao et al. argue that channel-based layers such as LN and GN cannot keep the network far from elimination singularities, which may explain why they trail BN at large batches. {#c-singularities}

**Layer normalization in ConvNets.**
Ba et al. found that units near the image boundary have very different statistics, and that layer normalization needs more work to suit ConvNets. {#c-conv-boundary}
GN with $G = 1$ inherits this.

**Divisibility and constant groups.**
$G$ must divide $C$ in every layer.
A group whose values are all equal has variance 0, so with $\epsilon = 0$ the output is $0/0$.

## Variants

- **Layer normalization** is GN with $G = 1$.
- **Instance normalization** is GN with $G = C$.
- **Batch normalization** uses the set $\{k : k_C = i_C\}$, over the batch; with one example it is IN in training mode.
- **[[rms-normalization]]** drops the mean and divides by the root mean square, per example over the features; it is usually used in Transformers, not ConvNets.
- **GN with weight standardization** pairs GN with standardized convolution weights, for training with one image per GPU.
- **Weight normalization** rescales the weights instead of the activations.

## When to use it

Use GN in convolutional networks when the batch per worker is small: detection, segmentation, video and other high-resolution tasks.
Use it when the same network must behave the same way in training and at test time, or when pre-training and fine-tuning use different batch sizes.
With large batches and an image classifier tuned for BN, BN is still slightly better in the results above.
For sequence models, LN or RMSNorm is the usual choice.

## Pitfalls

> **Grouping the wrong channels.** The groups are blocks of consecutive channels, in the order of the C axis. A reshape of $(N, C, H, W)$ to $(N, C/G, G, H, W)$ instead of $(N, G, C/G, H, W)$ interleaves the groups and gives a different layer. `test_sets_match_loop` compares the reshape with a loop over the sets of eq-gn-set.

> **Using a per-group scale.** $\gamma$ and $\beta$ are per channel, not per group. `test_shared_fixture` uses four different values of $\gamma$ in two groups.

> **Expecting each channel to have mean 0.** Only the group has mean 0 and variance 1. `test_group_moments` checks the group, and the worked example has a channel with mean $-1$.

> **Normalizing over the batch.** The statistics come from one example. `test_batch_independence` in both implementations checks that the other examples do not matter.

> **Forgetting the gradient through the statistics.** A backward pass that treats $\mu$ and $s$ as constants gives $u/s$. `test_gradient_finite_differences` and `test_autograd_matches_closed_form` check eq-gn-grad.

## History

Ioffe and Szegedy posted batch normalization in 2015. {#c-hist-ioffe}
Ba, Kiros and Hinton posted layer normalization in July 2016. {#c-hist-layer-norm}
Ulyanov, Vedaldi and Lempitsky posted instance normalization the same month; the cited version is from November 2017. {#c-hist-in}
Wu and He of Facebook AI Research posted group normalization in 2018. {#c-hist-gn}
They proposed to study GN in recurrent and generative models, where LN and IN do well, and in reinforcement learning. {#c-future}
Qiao et al. combined GN with weight standardization in 2019.
