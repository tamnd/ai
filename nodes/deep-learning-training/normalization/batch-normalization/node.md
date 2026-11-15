---
tag: NEW-batch-normalization
kind: method
slug: batch-normalization
name: Batch normalization
aliases: [BatchNorm, BN, batch norm]
area: deep-learning-training/normalization
level: core
summary: >-
  In training, normalize each feature of a layer with the mean and variance of the current mini-batch, then apply a learned scale and shift; at inference, use running averages of the statistics.
purpose: faster and more stable optimization of deep networks
uses: [sample-mean@m2, sample-variance@m2, backpropagation@al]
related_to: [layer-normalization@al, weight-normalization@al, dropout, weight-decay@al, gradient-clipping@al, exploding-gradient-problem@al, stochastic-gradient-descent@m2]
hyperparameters:
  - {name: momentum, meaning: "the weight of the running averages of the mean and the variance; the frameworks use opposite conventions", typical: "a weight of 0.9 to 0.99 on the old average", default_in: {pytorch: 0.1, keras: 0.99, flax: 0.99}}
  - {name: eps, meaning: "the constant added to the variance before the square root", typical: "1e-5 to 1e-3", default_in: {pytorch: 1.0e-5, keras: 1.0e-3, flax: 1.0e-5}}
  - {name: batch size per device, meaning: "the number of examples over which the statistics are taken", typical: "16 to 256; error rises below about 16"}
  - {name: initial scale, meaning: "the initial value of gamma", typical: "1; 0 for the last normalization of each residual block", default_in: {pytorch: 1.0, keras: 1.0, flax: 1.0}}
cost:
  train: "a mean and a variance per feature over the batch, a few elementwise operations per activation, and two reductions per feature in the backward pass; the normalized activations kept until the backward pass"
  inference: "one multiply and one add per activation, or none once folded into the preceding linear or convolutional layer"
history:
  year: 2015
  people: [Sergey Ioffe, Christian Szegedy]
  note: "Ioffe and Szegedy, arXiv 1502.03167 (February 2015), from Google; published at ICML 2015."
prereqs: [sample-mean@m2, sample-variance@m2, backpropagation@al, stochastic-gradient-descent@m2]
same_as:
  wikidata: Q55080248
  wikipedia: Batch_normalization
  arxiv: ["1502.03167"]
  al: batch-normalization
sources:
  - {paper: "Ioffe and Szegedy (2015)", title: "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift", locator: "Sections 2 to 4, Algorithms 1 and 2, Figure 3", note: "the method, the backward pass, inference, the convolutional case and the Inception results"}
  - {paper: "Santurkar, Tsipras, Ilyas and Madry (2018)", title: "How Does Batch Normalization Help Optimization?", locator: "the abstract, Sections 2 to 4", note: "the case against internal covariate shift, and the smoothness view"}
  - {paper: "Yang, Pennington, Rao, Sohl-Dickstein and Schoenholz (2019)", title: "A Mean Field Theory of Batch Normalization", locator: "p. 1, Theorem 3.9", note: "gradient explosion at initialization"}
  - {paper: "Wu and He (2018)", title: "Group Normalization", locator: "Section 3, Tables 1 and 2", note: "the error of batch normalization at small batch sizes"}
  - {paper: "Ioffe (2017)", title: "Batch Renormalization: Towards Reducing Minibatch Dependence in Batch-Normalized Models", locator: "the abstract, Algorithm 1, Section 4"}
  - {paper: "Ba, Kiros and Hinton (2016)", title: "Layer Normalization", locator: "the abstract"}
  - {paper: "Ulyanov, Vedaldi and Lempitsky (2016)", title: "Instance Normalization: The Missing Ingredient for Fast Stylization", locator: "the abstract"}
  - {paper: "Salimans and Kingma (2016)", title: "Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks", locator: "the abstract"}
  - {paper: "Brock, De, Smith and Simonyan (2021)", title: "High-Performance Large-Scale Image Recognition Without Normalization", locator: "the abstract, Sections 1, 2 and 4", note: "the costs and the benefits of batch normalization, and NFNets"}
  - {paper: "Li, Chen, Hu and Yang (2019)", title: "Understanding the Disharmony between Dropout and Batch Normalization by Variance Shift", locator: "p. 1, p. 2"}
  - {paper: "van Laarhoven (2017)", title: "L2 Regularization versus Batch and Weight Normalization", locator: "the abstract, Section 3"}
  - {paper: "Hoffer, Hubara and Soudry (2017)", title: "Train longer, generalize better: closing the generalization gap in large batch training of neural networks", locator: "the abstract"}
  - {paper: "Goyal et al. (2017)", title: "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour", locator: "Sections 2.3 and 5.1", note: "per-worker statistics, zero-gamma initialization and no weight decay on gamma and beta"}
  - {paper: "Wu and Johnson (2021)", title: "Rethinking \"Batch\" in BatchNorm", locator: "Section 3, Table 1", note: "the error of the moving averages, and PreciseBN"}
  - {url: "https://docs.pytorch.org/docs/2.15/generated/torch.nn.BatchNorm2d.html", title: "torch.nn.BatchNorm2d", locator: "the parameters eps, momentum, affine and track_running_stats", note: "the PyTorch defaults, the momentum convention and the unbiased running variance"}
  - {url: "https://keras.io/api/layers/normalization_layers/batch_normalization/", title: "Keras BatchNormalization layer", locator: "the arguments momentum, epsilon and training", note: "the Keras defaults, the momentum convention and the inference mode of a frozen layer"}
  - {url: "https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/normalization.html", title: "flax.nnx.BatchNorm", locator: "the parameters use_running_average, momentum, epsilon and axis_name", note: "the Flax defaults"}
---

Batch normalization is a layer that normalizes each feature of its input with the mean and variance of the current mini-batch.
A learned scale and shift follow, so the layer can still represent any mean and variance.
At inference it uses running averages of the statistics and becomes a fixed affine map.
It lets convolutional networks train with much higher learning rates and fewer steps.
Its dependence on the batch is also the source of most of its failures.

## In plain words

Deep networks train badly when the scale of the signal inside them drifts as the weights change.
Batch normalization fixes the scale at each layer.
For every feature, it subtracts the mean over the examples of the batch and divides by their standard deviation.
Then it multiplies by a learned number and adds another one, so the network decides the final scale.
At test time there may be no batch, so the layer uses averages of the statistics collected during training.
Why this helps so much is still debated.
The original paper credits a more stable distribution of layer inputs.
Later work points to a smoother loss surface instead.

## Setting

Batch normalization applies to a network trained by stochastic gradient descent on mini-batches of $m > 1$ examples.
It is a layer, placed after a linear or convolutional layer and usually before the nonlinearity.
Ioffe and Szegedy (2015) define internal covariate shift as the change in the distribution of activations caused by the training updates of the parameters. {#c-ics-definition}
They propose batch normalization to reduce it, and they report that it allows much higher learning rates and less careful initialization. {#c-higher-lr}
The input of the layer for one feature is a mini-batch $\mathcal{B} = \{x_1, \dots, x_m\}$ of values.

## The method

For each feature, the layer computes the mean and the biased variance of the mini-batch,

$$ \mu_\mathcal{B} = \frac{1}{m} \sum_{i=1}^m x_i, \qquad \sigma_\mathcal{B}^2 = \frac{1}{m} \sum_{i=1}^m (x_i - \mu_\mathcal{B})^2 , $$ {#eq-batch-stats}

then normalizes, scales and shifts,

$$ \hat{x}_i = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}}, \qquad y_i = \gamma \hat{x}_i + \beta . $$ {#eq-bn-forward}

This is Algorithm 1 of Ioffe and Szegedy, with a constant $\epsilon$ added to the variance for numerical stability. {#c-algorithm}
The parameters $\gamma$ and $\beta$ are learned with the rest of the network.
Each scalar feature is normalized on its own, without whitening the layer inputs jointly.
Full whitening would be costly, and its covariance would be singular for a small mini-batch. {#c-per-feature}
Setting $\gamma$ to the standard deviation and $\beta$ to the mean recovers the original activations, so the layer can represent the identity. {#c-identity-recoverable}

**Inference.**
The output for an example should not depend on the other examples of a test batch.
So at inference the layer uses population statistics, with the unbiased variance estimate $\frac{m}{m-1} \mathbb{E}_\mathcal{B}[\sigma_\mathcal{B}^2]$ over training mini-batches. {#c-inference-population}
Ioffe and Szegedy suggest moving averages instead, so that the accuracy of the model can be tracked as it trains. {#c-moving-averages}
With the running mean $\hat\mu$ and the running variance $\hat v$, the layer is the fixed affine map

$$ y = \frac{\gamma}{\sqrt{\hat v + \epsilon}} \, x + \Big( \beta - \frac{\gamma \hat\mu}{\sqrt{\hat v + \epsilon}} \Big) . $$ {#eq-inference}

The running averages are updated at each training step,

$$ \hat\mu \leftarrow (1 - \alpha) \, \hat\mu + \alpha \, \mu_\mathcal{B}, \qquad \hat v \leftarrow (1 - \alpha) \, \hat v + \alpha \, \tfrac{m}{m-1} \sigma_\mathcal{B}^2 , $$ {#eq-running}

where $\alpha$ is the PyTorch `momentum`, 0.1 by default.
PyTorch normalizes with the biased variance in training and stores the unbiased one in the running variance.
Keras and Flax write the same update with their momentum on the old average, so their 0.99 is a PyTorch momentum of 0.01.

**Placement.**
Ioffe and Szegedy put the layer before the nonlinearity, as $z = g(\mathrm{BN}(Wu))$, and they drop the bias of $Wu$ because $\beta$ takes its role. {#c-before-nonlinearity}

**Convolutional layers.**
In a convolutional layer, all locations of a feature map share one pair $\gamma$, $\beta$.
The statistics are taken over the batch and over all spatial locations, an effective mini-batch of $m' = m \cdot p \cdot q$ values for a $p \times q$ feature map. {#c-conv}
For an input of shape $(N, C, H, W)$, each of the $C$ channels has its own mean and variance over $N \cdot H \cdot W$ values.

## Derivation

**The backward pass.**
Write $g_i = \partial \ell / \partial y_i$ and $s = \sqrt{\sigma_\mathcal{B}^2 + \epsilon}$.
The mean and the variance depend on every $x_i$, so the gradient flows through them too.
The chain rule gives

$$ \frac{\partial \ell}{\partial \gamma} = \sum_i g_i \hat{x}_i, \qquad \frac{\partial \ell}{\partial \beta} = \sum_i g_i, \qquad \frac{\partial \ell}{\partial x_i} = \frac{\gamma}{m s} \Big( m g_i - \sum_j g_j - \hat{x}_i \sum_j g_j \hat{x}_j \Big) . $$ {#eq-bn-backward}

These are the gradients of the batch-normalizing transform, from Ioffe and Szegedy's chain-rule equations written in a compact form. {#c-backward}
In vector form, $\partial \ell / \partial \mathbf{x} = \frac{\gamma}{s} \big( I - \frac{1}{m} \mathbf{1}\mathbf{1}^\top - \frac{1}{m} \hat{\mathbf{x}}\hat{\mathbf{x}}^\top \big) \mathbf{g}$ when $\epsilon = 0$.
The input gradient sums to zero over the batch, and with $\epsilon = 0$ it is orthogonal to $\hat{\mathbf{x}}$. {#c-dx-orthogonal}
So the layer removes the parts of the gradient that would only shift or rescale the batch.
The derivation `bn-backward` gives the steps and a finite-difference check.

**Scale invariance.**
For $a > 0$ and $\epsilon = 0$, scaling the weights of the preceding layer does not change the output: $\mathrm{BN}((aW)u) = \mathrm{BN}(Wu)$. {#c-scale-invariance}
The gradient with respect to $aW$ is the gradient with respect to $W$ divided by $a$, so larger weights get smaller gradients. {#c-gradient-scaling}
The gradient of a scale-invariant loss is orthogonal to the weights, so a step of plain gradient descent never shrinks their norm. {#c-norm-grows}
The derivation `scale-invariance` gives the steps.

**Folding at inference.**
If a linear layer $Wx + b$ comes before the layer, eq-inference composes with it into one linear layer. {#c-fold}
Its weights are $W' = \mathrm{diag}(\gamma / \sqrt{\hat v + \epsilon}) \, W$ and its bias is $b' = \gamma (b - \hat\mu) / \sqrt{\hat v + \epsilon} + \beta$.
A PyTorch momentum $\alpha$ is the same update as a Keras or Flax momentum $1 - \alpha$. {#c-momentum-conventions}
The derivation `inference-fold` gives the steps.

**Dropout before the layer.**
Take a unit with mean 0 and variance $v$, followed by inverted dropout with drop probability $q$.
Each mask entry squares to itself, so in training the unit has variance $v / (1 - q)$; at inference it has variance $v$. {#c-dropout-variance}
The running variance of a following batch normalization then overestimates the variance at test time by a factor $1/(1 - q)$.

## Worked example

Take one feature with the batch $\mathbf{x} = (2, 4, 5, 6, 8)$, so $m = 5$, and take $\epsilon = 0$.
The mean is 5 and the biased variance is $(9 + 1 + 0 + 1 + 9)/5 = 4$, so $s = 2$.
The normalized values are $\hat{\mathbf{x}} = (-1.5, -0.5, 0, 0.5, 1.5)$.
With $\gamma = 2$ and $\beta = 1$, the output is $\mathbf{y} = (-2, 0, 1, 2, 4)$.

Let the gradient that comes back be $\mathbf{g} = (1, 0, 0, 0, 0)$.
Then $\sum_j g_j = 1$ and $\sum_j g_j \hat{x}_j = -1.5$, so $\partial \ell / \partial \gamma = -1.5$ and $\partial \ell / \partial \beta = 1$.
The prefactor of eq-bn-backward is $\gamma / (m s) = 0.2$.
The input gradient is $0.2 \cdot \big( (5, 0, 0, 0, 0) - \mathbf{1} + 1.5 \, \hat{\mathbf{x}} \big) = (0.35, -0.35, -0.2, -0.05, 0.25)$.
It sums to zero, and its dot product with $\hat{\mathbf{x}}$ is zero.
A layer that treated the mean and the variance as constants would give $(1, 0, 0, 0, 0)$ instead.

Start the running statistics at mean 0 and variance 1, the PyTorch initial values, with momentum 0.1.
The unbiased variance of the batch is $20/4 = 5$.
After this step, the running mean is $0.1 \cdot 5 = 0.5$ and the running variance is $0.9 \cdot 1 + 0.1 \cdot 5 = 1.4$.

## Hyperparameters

**Momentum.**
It sets how fast the running averages follow the batch statistics.
PyTorch defaults to 0.1, the weight of the current batch.
Keras and Flax default to 0.99, the weight of the old average.
A slow average lags behind the weights in early training.
Wu and Johnson (2021) find that this lag can make the moving averages inaccurate.
They propose PreciseBN, which computes the statistics afresh on many batches with the final weights. {#c-ema-inaccurate}

**Epsilon.**
It keeps the division stable when a variance is close to zero.
PyTorch and Flax default to $10^{-5}$, and Keras to $10^{-3}$.
A model ported between frameworks must keep its value, or its outputs change.

**Batch size per device.**
The statistics are a sample estimate, so a small batch gives noisy statistics.
Goyal et al. (2017) treat the number of examples per worker as a hyperparameter of batch normalization.
They use 32 per worker and do not share the statistics across workers. {#c-per-worker}

**Initialization and weight decay.**
The scale $\gamma$ starts at 1 and the shift $\beta$ at 0.
Goyal et al. set $\gamma = 0$ in the last batch normalization of each residual block, so each block starts as the identity. {#c-zero-gamma}
They apply no weight decay to $\gamma$ and $\beta$. {#c-no-wd-gamma}

## Cost

Let $d$ be the number of activations of the layer in one batch, and $C$ the number of features.
In training, the forward pass does two reductions per feature and a few operations per activation.
The backward pass does two more reductions per feature, $\sum_j g_j$ and $\sum_j g_j \hat{x}_j$.
It keeps $\hat{\mathbf{x}}$ or the input, of size $d$, and $1/s$ per feature, until the backward pass.
The layer has $2C$ parameters and $2C$ running statistics.
At inference it costs one multiply and one add per activation, and nothing once folded into the preceding layer.
When the statistics are shared across devices, each step adds a communication of $2C$ numbers per layer.
Brock et al. (2021) call batch normalization a surprisingly expensive computational primitive that incurs memory overhead. {#c-expensive}

## Evidence

On ImageNet, BN-x5, an Inception network with batch normalization and five times the learning rate, reaches 72.2% accuracy in $2.1 \cdot 10^6$ steps. {#c-inception-steps}
Plain Inception needs $31.0 \cdot 10^6$ steps for the same 72.2%. {#c-inception-baseline}
BN-x30, with thirty times the learning rate, reaches 74.8%. {#c-bn-x30}
With sigmoid units, BN-x5-Sigmoid reaches 69.8%, while plain Inception with sigmoid never gets better than 1/1000. {#c-sigmoid}
An ensemble of batch-normalized networks reaches 4.9% top-5 validation error and 4.8% test error. {#c-ensemble}
On MNIST, three hidden layers of 100 sigmoid units reach a higher test accuracy and more stable distributions with batch normalization. {#c-mnist}

## Limitations and failure modes

**The mechanism is disputed.**
Santurkar et al. (2018) inject noise with a non-zero mean and a non-unit variance after each batch normalization of a VGG network.
The noisy network performs almost as well as the standard one. {#c-noisy-bn}
They conclude that the stability of the distributions of layer inputs has little to do with the success of the method. {#c-little-to-do}
With a gradient-based measure of internal covariate shift, networks with batch normalization have similar or even worse shift. {#c-ics-worse}
They argue that batch normalization makes the loss landscape significantly smoother and the gradients more predictive. {#c-smoother}
They prove that the squared gradient norm at the activations is at most $\gamma^2 / \sigma^2$ times that without the layer, less two terms. {#c-gradient-bound}
They find that normalizing by the mean $\ell_p$ norm, for $p = 1, 2, \infty$, smooths the landscape in a similar way. {#c-lp-norms}

**Gradients explode at initialization in deep plain networks.**
Yang et al. (2019) study fully connected networks with batch normalization at initialization with mean field theory. {#c-mean-field}
They find that the gradients explode with depth, and that batch normalization itself is the cause. {#c-gradient-explosion}
Without skip connections, such networks are not trainable at large depth for common initialization schemes. {#c-not-trainable}
Tuning the network close to the linear regime reduces the explosion, but it cannot remove it. {#c-linear-regime}
This is a form of the exploding gradient problem, and their result is for networks without skip connections.

**Small and non-independent batches.**
Wu and He (2018) report that the error of batch normalization increases rapidly as the batch becomes smaller. {#c-small-batch}
With ResNet-50 on ImageNet, its error goes from 23.6% at 32 images per worker to 34.7% at 2. {#c-small-batch-numbers}
Ioffe (2017) adds that the method is less effective with non-independent mini-batches. {#c-non-iid}
The output of an example depends on the other examples of its batch.
Brock et al. (2021) note that this breaks the independence between training examples in the minibatch. {#c-independence}
It can cause subtle bugs in distributed training, and it can leak information in contrastive learning.

**Training and inference differ.**
The running averages only approximate the statistics of the final network.
Brock et al. list this discrepancy, with its hidden hyperparameters, among the disadvantages of the method. {#c-train-eval}
For a batch of 8192 and 30 epochs, Wu and Johnson (2021) measure 52.8% validation error with the moving averages.
With recomputed statistics, the error is 39.42%. {#c-precise-bn-numbers}

**Dropout before batch normalization.**
Li et al. (2019) call the change of variance between training and inference the variance shift. {#c-variance-shift}
They apply dropout after all batch normalization layers, or change the dropout formula to make it less sensitive to variance. {#c-li-remedy}

**Weight decay does not regularize.**
Because of scale invariance, an L2 penalty on the weights before the layer has no regularizing effect. {#c-l2-no-reg}
Van Laarhoven (2017) argues that weight decay acts instead on the scale of the weights, and thereby on the effective learning rate. {#c-effective-lr}

## Variants

- **Layer normalization** normalizes over the units of a layer for each example, the same way in training and at test (Ba et al. 2016). {#c-layer-norm}
- **Instance normalization** normalizes each channel of each example, at training and test time, and was proposed for style transfer (Ulyanov et al. 2016). {#c-instance-norm}
- **Group normalization** normalizes each group of channels of each example. It uses 32 groups by default, and one group is layer normalization (Wu and He 2018). {#c-group-norm}
- **Batch renormalization** corrects the batch statistics with the factors $r = \sigma_\mathcal{B} / \sigma$ and $d = (\mu_\mathcal{B} - \mu)/\sigma$, with $\mu$ and $\sigma$ the moving averages, clipped and constant in the backward pass (Ioffe 2017). {#c-batch-renorm}
- **Ghost batch normalization** takes the statistics over small virtual batches. It reduces the generalization gap of large-batch training (Hoffer et al. 2017). {#c-ghost-bn}
- **Weight normalization** splits each weight vector into a length and a direction, and does not depend on the batch (Salimans and Kingma 2016). {#c-weight-norm}
- **Normalizer-free networks** drop normalization and use adaptive gradient clipping, a gradient clipping relative to the weight norm (Brock et al. 2021). {#c-agc}

## When to use it

Use batch normalization in convolutional networks trained with batches of about 16 or more examples per device.
With smaller batches, use group normalization, or synchronize the statistics across devices.
In recurrent networks, layer normalization is the usual choice, because it needs no batch and works at each time step.
When examples must stay independent, as in contrastive learning, prefer a normalization that works on each example.
At 32 images per worker, group normalization has 24.1% error on ImageNet against 23.6% for batch normalization.
At 2 images per worker, it has 24.1% against 34.7%. {#c-gn-vs-bn}
Normalizer-free networks are an alternative when the batch dependence is a problem.
Brock et al. report that their smaller models match EfficientNet-B7 on ImageNet while being up to 8.7 times faster to train. {#c-nfnet}

## Pitfalls

> [!pitfall]
> **Training mode at inference.** A model left in training mode normalizes each test batch with its own statistics, so a prediction depends on the other examples of the batch. In PyTorch, call `model.eval()`. The test `test_eval_uses_running_stats` checks that evaluation mode uses the running statistics and treats each example on its own.

> [!pitfall]
> **The wrong variance in the running average.** PyTorch normalizes with the biased variance and stores the unbiased one. A copy that stores the biased variance drifts from PyTorch by a factor $m/(m-1)$. The test `test_running_stats_match_torch` compares the running statistics with `torch.nn.BatchNorm1d` and `torch.nn.BatchNorm2d` over several steps.

> [!pitfall]
> **The momentum of another framework.** A Keras momentum of 0.99 given to PyTorch makes the running averages almost equal to the last batch. The test `test_momentum_convention` checks that a PyTorch momentum $\alpha$ equals the update with weight $1 - \alpha$ on the old average.

> [!pitfall]
> **The wrong axes in a convolutional layer.** Normalizing over the batch only, separately at each pixel, gives $H \cdot W$ statistics per channel and a different layer. The test `test_conv_per_channel` checks the statistics over the batch and the spatial axes against `torch.nn.BatchNorm2d`.

> [!pitfall]
> **The statistics treated as constants.** A backward pass that ignores the dependence of the mean and the variance on the input gives a wrong gradient. The test `test_gradient_finite_differences` compares the gradient with finite differences, and `test_worked_example` checks the numbers of the page.

> [!pitfall]
> **One value per feature in training.** With one value per feature, the variance is zero and the output is $\beta$. PyTorch `BatchNorm1d` raises an error for a training batch of one. The test `test_single_value_rejected` checks that the implementations reject it.

## History

Sergey Ioffe and Christian Szegedy of Google posted batch normalization in February 2015, with the third version on 2 Mar 2015. {#c-hist-ioffe}
Ba, Kiros and Hinton posted layer normalization, which needs no batch, as arXiv 1607.06450 on 21 Jul 2016. {#c-hist-layer-norm}
Ulyanov et al. posted instance normalization as arXiv 1607.08022 in the same month.
Ioffe posted batch renormalization on 30 Mar 2017, in its second version. {#c-hist-renorm}
Wu and He of Facebook AI Research posted group normalization in 2018. {#c-hist-gn}
Santurkar et al. (2018) then questioned the explanation by internal covariate shift.
Brock et al. (2021) trained normalizer-free networks that reach 86.5% top-1 accuracy on ImageNet without normalization layers. {#c-hist-nfnet}
