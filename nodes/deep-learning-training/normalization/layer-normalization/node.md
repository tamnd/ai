---
tag: RA5R92
kind: method
slug: layer-normalization
name: Layer normalization
aliases: [LayerNorm, LN, layer norm]
area: deep-learning-training/normalization
level: core
summary: >-
  Normalize the summed inputs of a layer with their mean and variance over its units, for each example, then apply a learned gain and bias; the same in training and test, with no batch.
purpose: stable and fast training of recurrent networks and Transformers, independent of the batch
uses: [sample-mean@m2, standard-deviation@m2, backpropagation@al]
related_to: [batch-normalization@al, weight-normalization@al, backpropagation-through-time@al, exploding-gradient-problem@al, vanishing-gradient-problem@al, learning-rate-schedule@al]
hyperparameters:
  - {name: eps, meaning: "the constant added to the variance before the square root", typical: "1e-5 to 1e-6; the machine epsilon of the dtype for RMSNorm in PyTorch", default_in: {pytorch: 1.0e-5}}
  - {name: normalized_shape, meaning: "the trailing dimensions over which the mean and the variance are taken", typical: "the model width d of a Transformer; the hidden size of a recurrent network"}
  - {name: elementwise_affine and bias, meaning: "whether there is a learned gain, and a learned bias, for every normalized element", typical: "both on; RMSNorm has a gain and no bias", default_in: {pytorch: true}}
  - {name: placement, meaning: "where the layer sits relative to the residual connection of a Transformer block", typical: "Pre-LN, with a final normalization before the output layer"}
  - {name: initial gain and bias, meaning: "the initial values of the gain and the bias", typical: "1 and 0", default_in: {pytorch: "1 and 0"}}
cost:
  train: "a mean and a variance per example over the normalized features, a few elementwise operations per activation, and two reductions per example in the backward pass; the normalized activations or the inputs kept until the backward pass"
  inference: "the same as the forward pass in training: two reductions per example and a few operations per activation; it cannot be folded into the preceding layer, because the statistics depend on the input"
history:
  year: 2016
  people: [Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton]
  note: "Ba, Kiros and Hinton, arXiv 1607.06450 (21 Jul 2016), from the University of Toronto and Google."
prereqs: [sample-mean@m2, standard-deviation@m2, backpropagation@al, orthogonal-projection@m2]
same_as:
  wikidata: null
  wikipedia: null
  arxiv: ["1607.06450"]
  al: layer-normalization
sources:
  - {paper: "Ba, Kiros and Hinton (2016)", title: "Layer Normalization", locator: "the abstract, Sections 1, 3, 3.1, 5.1, 5.2, 5.2.2, 6.1, 6.2, 6.6 and 6.7, Tables 1 and 2, p. 13", note: "the method, recurrent networks, the invariances, the geometry of learning, the experiments and the LSTM equations"}
  - {paper: "Xiong et al. (2020)", title: "On Layer Normalization in the Transformer Architecture", locator: "the abstract, Sections 3.1 to 3.4 and 4.2", note: "Post-LN and Pre-LN, the gradient at initialization and the learning rate warm-up"}
  - {paper: "Liu, Liu, Gao, Chen and Han (2020)", title: "Understanding the Difficulty of Training Transformers", locator: "pp. 1 to 3", note: "the dependency on the residual branch, and the convergence of Post-LN and Pre-LN"}
  - {paper: "Zhang and Sennrich (2019)", title: "Root Mean Square Layer Normalization", locator: "the abstract, Sections 4, 4.1, 4.2 and 6.1, Figure 1", note: "RMSNorm, its invariances, its gradient and its speed"}
  - {paper: "Vaswani et al. (2017)", title: "Attention Is All You Need", locator: "Section 3.1", note: "the Post-LN Transformer"}
  - {paper: "Touvron et al. (2023)", title: "LLaMA: Open and Efficient Foundation Language Models", locator: "Section 2.2", note: "pre-normalization with RMSNorm"}
  - {paper: "Wu and He (2018)", title: "Group Normalization", locator: "Sections 3 and 4.1, Table 1", note: "layer normalization as group normalization with one group, and its error on ImageNet"}
  - {url: "https://docs.pytorch.org/docs/2.15/generated/torch.nn.LayerNorm.html", title: "torch.nn.LayerNorm", locator: "the parameters normalized_shape, eps, elementwise_affine and bias", note: "the PyTorch defaults, the biased variance and the per-element affine"}
  - {url: "https://docs.pytorch.org/docs/2.15/generated/torch.nn.RMSNorm.html", title: "torch.nn.RMSNorm", locator: "the parameters normalized_shape, eps and elementwise_affine", note: "the default eps of RMSNorm, the machine epsilon of the input dtype"}
  - {url: "https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.layer_norm.html", title: "torch.nn.functional.layer_norm", locator: "the signature", note: "the functional form used by the tests"}
---

Layer normalization is a layer that normalizes the summed inputs of a layer with their mean and standard deviation over the units of that layer, for each example on its own.
A learned gain and bias for each unit follow.
Because the statistics come from one example, the layer does not depend on the batch, works with a batch of one, and computes the same function in training and at test time.
This makes it the normalization of choice for recurrent networks and for Transformers.
Its position relative to the residual connection, before or after, changes how a Transformer trains.
RMSNorm, which drops the mean, is the common variant in large language models.

## In plain words

The scale of the signal inside a deep network drifts as the weights change, and this makes training slow and fragile.
Layer normalization fixes the scale at each layer, one example at a time.
It takes the vector of summed inputs of a layer for one example, subtracts its mean over the units, and divides by its standard deviation over the units.
Then it multiplies each unit by a learned gain and adds a learned bias.
Nothing depends on the other examples, so there is nothing to average at test time and nothing to break when the batch is small.
In a recurrent network it does the same at every time step, with the statistics of that step.
Batch normalization normalizes each unit over the batch instead; layer normalization normalizes each example over the units.

## Setting

Layer normalization applies to any layer whose summed inputs form a vector, or a tensor, of features for each example.
Ba et al. (2016) note that batch normalization cannot be applied to online learning tasks or to extremely large distributed models where the minibatches have to be small, and that in recurrent networks it appears to require different statistics for different time-steps. {#c-bn-limits}
They compute the mean and the variance used for normalization from all of the summed inputs to the neurons in a layer on a single training case. {#c-ln-idea}
Write $\mathbf{a} \in \mathbb{R}^H$ for the summed inputs of a layer with $H$ hidden units, for one example, so $\mathbf{a} = W\mathbf{x}$ for the input $\mathbf{x}$ of the layer.

## The method

The layer computes the mean and the biased standard deviation of the summed inputs over the $H$ units,

$$ \mu = \frac{1}{H} \sum_{i=1}^H a_i, \qquad \sigma = \sqrt{\frac{1}{H} \sum_{i=1}^H (a_i - \mu)^2} , $$ {#eq-ln-stats}

then normalizes, applies the gain $\mathbf{g}$ and the bias $\mathbf{b}$, and the nonlinearity $f$,

$$ \mathbf{h} = f\Big( \frac{\mathbf{g}}{\sigma} \odot (\mathbf{a} - \mu \mathbf{1}) + \mathbf{b} \Big) . $$ {#eq-ln-forward}

This is Eq. 3 of Ba et al.: all the hidden units in a layer share the same normalization terms, but different training cases have different normalization terms. {#c-ln-definition}
Each neuron gets its own adaptive bias and gain, which are applied after the normalization but before the nonlinearity. {#c-gain-bias}
Ba et al. initialize the adaptive gains to 1 and the biases to 0. {#c-init}
In practice $\sigma$ is computed as $\sqrt{v + \epsilon}$, with $v$ the biased variance and a small $\epsilon$; PyTorch uses $\epsilon = 10^{-5}$.
The PyTorch layer has one gain and one bias for every element of `normalized_shape`, not one per channel.

**No batch.**
Layer normalization does not impose any constraint on the size of a mini-batch, and it can be used in the pure online regime with batch size 1. {#c-batch-size-one}
It performs exactly the same computation at training and test times. {#c-train-test}
Unlike batch normalization, it does not introduce any new dependencies between training cases. {#c-no-dependencies}
The output for one example does not depend on any other example of the batch, and the layer has no state, so the function at test time is the function in training. {#c-batch-independent}
The derivation `reductions` gives the steps.

**Recurrent networks.**
In a recurrent network the summed inputs at time $t$ are $\mathbf{a}^t = W_{hh}\mathbf{h}^{t-1} + W_{xh}\mathbf{x}^t$, and the layer normalizes them at each step,

$$ \mathbf{h}^t = f\Big( \frac{\mathbf{g}}{\sigma^t} \odot (\mathbf{a}^t - \mu^t \mathbf{1}) + \mathbf{b} \Big), \qquad \mu^t = \frac{1}{H}\sum_{i=1}^H a_i^t, \qquad \sigma^t = \sqrt{\frac{1}{H}\sum_{i=1}^H (a_i^t - \mu^t)^2} . $$ {#eq-ln-rnn}

Ba et al. apply it to recurrent networks by computing the normalization statistics separately at each time step. {#c-rnn-per-step}
The normalization terms depend only on the summed inputs to a layer at the current time-step, and there is only one set of gain and bias parameters shared over all time-steps. {#c-rnn-shared}
Batch normalization with separate statistics for each time step is problematic if a test sequence is longer than any of the training sequences. {#c-bn-rnn-long}
In a standard recurrent network, the average magnitude of the summed inputs tends to either grow or shrink at every time-step, which leads to exploding gradients or vanishing gradients. {#c-rnn-explode}
Layer normalization makes the network invariant to re-scaling all of the summed inputs to a layer, which results in much more stable hidden-to-hidden dynamics. {#c-rnn-stable}
In the LSTM of Ba et al., layer normalization is applied separately to $W_h\mathbf{h}_{t-1}$ and to $W_x\mathbf{x}_t$, and once more to the cell state $\mathbf{c}_t$ before the tanh of the output. {#c-lstm}
Training unrolls the network and uses backpropagation through time; the gradient at each step passes through the statistics of that step, by eq-ln-backward, so the normalization also acts on the backward pass.

**Transformers.**
The original Transformer places the layer after the residual connection, the Post-LN Transformer,

$$ \mathbf{x}_{l+1} = \mathrm{LN}\big(\mathbf{x}_l + F_l(\mathbf{x}_l)\big) , $$ {#eq-post-ln}

where $F_l$ is the self-attention or the feed-forward sub-layer.
Vaswani et al. (2017) employ a residual connection around each of the two sub-layers, followed by layer normalization, so that the output of each sub-layer is LayerNorm(x + Sublayer(x)). {#c-vaswani-post-ln}
Xiong et al. (2020) describe this order as self-attention (FFN) sub-layer, then residual connection, then layer normalization. {#c-xiong-post-ln}
The Pre-LN Transformer moves the layer inside the residual branch,

$$ \mathbf{x}_{l+1} = \mathbf{x}_l + F_l\big(\mathrm{LN}(\mathbf{x}_l)\big), \qquad \text{output} = \mathrm{LN}(\mathbf{x}_L) . $$ {#eq-pre-ln}

Xiong et al. describe it as putting the layer normalization inside the residual connection and before all other non-linear transformations, with a final layer normalization right before the prediction. {#c-pre-ln}
LLaMA normalizes the input of each transformer sub-layer instead of the output, to improve the training stability, and uses RMSNorm. {#c-llama}

## Derivation

**The backward pass.**
Write $s = \sqrt{v + \epsilon}$, $\hat{\mathbf{a}} = (\mathbf{a} - \mu\mathbf{1})/s$, $\boldsymbol{\delta}$ for the gradient of the loss with respect to the output before the nonlinearity, and $\mathbf{u} = \mathbf{g} \odot \boldsymbol{\delta}$.
The mean and the variance depend on every $a_i$, so the gradient flows through them too.
The chain rule gives

$$ \frac{\partial \ell}{\partial \mathbf{g}} = \boldsymbol{\delta} \odot \hat{\mathbf{a}}, \qquad \frac{\partial \ell}{\partial \mathbf{b}} = \boldsymbol{\delta}, \qquad \frac{\partial \ell}{\partial \mathbf{a}} = \frac{1}{H s}\Big( H\mathbf{u} - (\mathbf{1}^\top\mathbf{u})\mathbf{1} - (\hat{\mathbf{a}}^\top\mathbf{u})\hat{\mathbf{a}} \Big) , $$ {#eq-ln-backward}

with the gain and bias gradients summed over the examples of a batch. {#c-ln-backward}
The Jacobian of $\hat{\mathbf{a}}$ is the symmetric matrix $\frac{1}{s}\big(I - \frac{1}{H}\mathbf{1}\mathbf{1}^\top - \frac{1}{H}\hat{\mathbf{a}}\hat{\mathbf{a}}^\top\big)$.
The input gradient sums to zero over the units for every $\epsilon$, and with $\epsilon = 0$ it is also orthogonal to $\hat{\mathbf{a}}$. {#c-ln-grad-sum-orthogonal}
For $H \ge 3$, the spectral norm of the Jacobian is $1/s$; with $\epsilon = 0$ it is $1/\sigma = \sqrt{H}/\lVert\mathbf{a} - \mu\mathbf{1}\rVert_2$, and the Jacobian is $1/\sigma$ times the orthogonal projection onto the complement of the span of $\mathbf{1}$ and $\hat{\mathbf{a}}$. {#c-ln-jacobian-norm}
With $\epsilon = 0$, $\lVert\hat{\mathbf{a}}\rVert_2 = \sqrt{H}$, so before the gain and bias every output lies on a sphere of radius $\sqrt{H}$ in the hyperplane of mean zero. {#c-ln-sphere}
Xiong et al. state the bound $\lVert J_{LN}(\mathbf{x})\rVert_2 = O(\sqrt{d}/\lVert\mathbf{x}\rVert_2)$ for the Jacobian of layer normalization at a vector $\mathbf{x}$ of dimension $d$. {#c-xiong-lemma3}
For an input of mean zero this is the exact norm above.
So large inputs get small gradients, and the layer normalizes the gradients as well as the activations.
The derivation `ln-backward` gives the steps and a finite-difference check.

**Invariances.**
Ba et al. compare the invariances of batch, weight and layer normalization in their Section 5.1 and Table 1. With $\epsilon = 0$ and no bias before the normalization, the following hold.
Layer normalization is invariant to scaling of the entire weight matrix and to a shift of all of the incoming weights: replacing $W$ by $W' = \delta W + \mathbf{1}\boldsymbol{\gamma}^\top$ for $\delta > 0$ does not change the output. {#c-matrix-invariance}
It is invariant to re-scaling of an individual training case, $\mathbf{x} \mapsto \delta\mathbf{x}$ with its own $\delta > 0$ for each case. {#c-case-invariance}
It is not invariant to the scaling of a single weight vector: with $W = I_3$ and $\mathbf{x} = (1, 2, 3)$, doubling the first row changes the output from $(-1.2247, 0, 1.2247)$ to $(-0.7071, -0.7071, 1.4142)$. {#c-not-vector-invariant}
It is not invariant to re-centering of the dataset: the shift $\mathbf{c} = (1, 0, 0)$ moves the same $\mathbf{x}$ to $(2, 2, 3)$, which gives the same changed output. {#c-not-data-recentering}
Ba et al. list it as invariant to weight matrix re-scaling, weight matrix re-centering, dataset re-scaling and single training case re-scaling, but not to weight vector re-scaling or dataset re-centering. {#c-ba-table1}
If the normalization is only applied to the input before the weights, the model is not invariant to re-scaling and re-centering of the weights. {#c-input-only}
The derivation `invariances` gives the steps.

**The geometry of learning.**
The gradient with respect to $\delta W + \mathbf{1}\boldsymbol{\gamma}^\top$ is $1/\delta$ times the gradient with respect to $W$. The columns of the weight gradient sum to zero, and with $\epsilon = 0$ the gradient is orthogonal to $W$ in the Frobenius inner product. So a step of plain gradient descent keeps the column sums of $W$ and grows its squared norm by $\eta^2\lVert\nabla f\rVert_F^2$. {#c-weight-gradient}
Ba et al. note that the normalization scalar $\sigma$ can implicitly reduce the learning rate and makes learning more stable. {#c-implicit-lr}
They argue that the norm of the weight vector effectively controls the learning rate, so that it is harder to change the orientation of a weight vector with a large norm. {#c-weight-norm-lr}
They find that learning the gain depends only on the magnitude of the prediction error, so that it is more robust to the scaling of the input. {#c-gain-robust}

**RMSNorm.**
RMSNorm drops the mean and divides by the root mean square,

$$ \bar{a}_i = \frac{a_i}{\mathrm{RMS}(\mathbf{a})}\, g_i, \qquad \mathrm{RMS}(\mathbf{a}) = \sqrt{\frac{1}{n}\sum_{i=1}^n a_i^2} . $$ {#eq-rms-forward}

This is Eq. 4 of Zhang and Sennrich (2019), who hypothesize that the re-centering invariance in LayerNorm is dispensable. {#c-rms-definition}
They write that when the mean of the summed inputs is zero, RMSNorm is exactly equal to LayerNorm. {#c-zhang-equal}
More precisely, with the same $\epsilon$ and no bias, layer normalization of $\mathbf{a}$ is RMSNorm of $\mathbf{a} - \mu\mathbf{1}$. {#c-rms-equals-ln}
With $\epsilon = 0$, RMSNorm is invariant to scaling of the weight matrix and to scaling of the input, by any $\delta > 0$. {#c-rms-invariance}
It is not invariant to re-centering: $(1, -1)$ normalizes to $(1, -1)$, but $(2, 0)$ normalizes to $(\sqrt{2}, 0)$, while layer normalization gives $(1, -1)$ for both. {#c-rms-not-recentering}
Zhang and Sennrich note that RMSNorm is not invariant to all re-centering operations. {#c-zhang-not-recentering}
With $r = \sqrt{\frac{1}{n}\sum a_i^2 + \epsilon}$ and $\hat{\mathbf{a}} = \mathbf{a}/r$, its input gradient is

$$ \frac{\partial \ell}{\partial \mathbf{a}} = \frac{1}{r}\Big(\mathbf{u} - \frac{1}{n}\hat{\mathbf{a}}(\hat{\mathbf{a}}^\top\mathbf{u})\Big) , $$ {#eq-rms-backward}

and the spectral norm of its Jacobian is $1/r$ for $n \ge 2$. With $\epsilon = 0$ and $\mathbf{a} = W\mathbf{x}$, the Jacobian is the matrix $R$ of their Eq. 9, and the gradient for $\delta W$ is $1/\delta$ times that for $W$. {#c-rms-backward}
With $\epsilon = 0$, $\lVert\hat{\mathbf{a}}\rVert_2 = \sqrt{n}$: the output before the gain lies on a sphere of radius $\sqrt{n}$, which does not need to have mean zero. {#c-rms-sphere}
The gradient of RMSNorm need not sum to zero, because only the direction $\hat{\mathbf{a}}$ is removed.
The derivation `rmsnorm` gives the steps.

**Group normalization.**
Group normalization with one group is layer normalization over the channels and the spatial positions $(C, H, W)$, with the per-element gain and bias of layer normalization set to the per-channel gain and bias of group normalization. {#c-gn-g1}
Wu and He (2018) describe layer normalization as computing $\mu$ and $\sigma$ along the $(C, H, W)$ axes for each sample. {#c-wu-ln}
They note that GN becomes LN with $G = 1$, and that LN assumes all channels in a layer make similar contributions, which is less valid with the presence of convolutions. {#c-wu-g1}

## Worked example

Take $H = 5$ units with the summed inputs $\mathbf{a} = (3, -4, 7, -1, 5)$, and take $\epsilon = 0$.
The mean is 2, the deviations are $(1, -6, 5, -3, 3)$, the biased variance is $(1 + 36 + 25 + 9 + 9)/5 = 16$, and $\sigma = 4$.
The normalized vector is $\hat{\mathbf{a}} = (0.25, -1.5, 1.25, -0.75, 0.75)$, which sums to 0 and has the squared norm $5$.
With the gain $\mathbf{g} = (2, 1, 2, 1, 4)$ and the bias $\mathbf{b} = (0, 1, 0, 1, 0)$, the output is $\mathbf{y} = (0.5, -0.5, 2.5, 0.25, 3)$.

Let the gradient that comes back be $\boldsymbol{\delta} = (0, 1, 0, 0, 0)$.
Then $\mathbf{u} = \mathbf{g}\odot\boldsymbol{\delta} = (0, 1, 0, 0, 0)$, $\mathbf{1}^\top\mathbf{u} = 1$ and $\hat{\mathbf{a}}^\top\mathbf{u} = -1.5$.
So $\partial\ell/\partial\mathbf{g} = (0, -1.5, 0, 0, 0)$ and $\partial\ell/\partial\mathbf{b} = (0, 1, 0, 0, 0)$.
The prefactor of eq-ln-backward is $1/(H\sigma) = 1/20$.
The input gradient is $\frac{1}{20}\big(5\mathbf{u} - \mathbf{1} + 1.5\,\hat{\mathbf{a}}\big) = \frac{1}{20}(-0.625, 1.75, 0.875, -2.125, 0.125)$.
It sums to zero, and its dot product with $\hat{\mathbf{a}}$ is zero.
A layer that treated the mean and the variance as constants would give $\mathbf{u}/\sigma = (0, 0.25, 0, 0, 0)$ instead.
The Jacobian has the spectral norm $1/\sigma = 0.25$.

Scaling $\mathbf{a}$ by 3, or adding 10 to every entry, leaves $\hat{\mathbf{a}}$ unchanged.
RMSNorm of $\mathbf{a}$ divides by $\mathrm{RMS}(\mathbf{a}) = \sqrt{(9 + 16 + 49 + 1 + 25)/5} = \sqrt{20}$, a different result.
RMSNorm of the centered vector $(1, -6, 5, -3, 3)$ divides by $\sqrt{80/5} = 4$, and gives $\hat{\mathbf{a}}$ exactly.

## Hyperparameters

**Epsilon.**
It keeps the division stable when the variance of an example is close to zero.
PyTorch `LayerNorm` defaults to $10^{-5}$, added to the biased variance inside the square root.
PyTorch `RMSNorm` defaults to `eps=None`, which means the machine epsilon of the input dtype, about $1.19 \cdot 10^{-7}$ for float32.
A port between the two, or between frameworks, must keep the value, or small inputs give different outputs.

**The normalized shape.**
The statistics are taken over the trailing dimensions given by `normalized_shape`.
In a Transformer this is the model width $d$, so each token is normalized on its own.
For an image tensor of shape $(N, C, H, W)$, normalizing over $(C, H, W)$ is group normalization with one group, with a gain for every element.

**Gain and bias.**
Both are learned, one for every normalized element, and start at 1 and 0.
RMSNorm keeps the gain and drops the bias.

**Placement.**
Post-LN normalizes after the residual sum, and Pre-LN normalizes the input of each sub-layer, with a final normalization before the output layer.
The choice interacts with the learning rate schedule: a Post-LN Transformer needs a warm-up, which raises the learning rate from a small value over the first steps, and a Pre-LN Transformer can drop it, as the Evidence section reports.

## Cost

Let $d$ be the number of normalized features, and $N$ the number of examples or tokens.
The forward pass does two reductions over $d$ for each example, a mean and a variance, and a few operations per activation.
The backward pass does two more reductions per example, $\mathbf{1}^\top\mathbf{u}$ and $\hat{\mathbf{a}}^\top\mathbf{u}$.
It keeps $\hat{\mathbf{a}}$ or the input, and $1/s$ per example, until the backward pass.
The layer has $2d$ parameters and no running statistics.
At inference the cost is the same as in training, and the layer cannot be folded into the preceding linear layer, because its statistics depend on the input.
RMSNorm saves the mean and one of the reductions.
Zhang and Sennrich report that it gives comparable performance to LayerNorm but reduces the running time by 7% to 64% on different models. {#c-zhang-speed}
The overhead matters: in their GRU RNNSearch, when the baseline training loss reaches 7.0, the loss with LayerNorm is 5.4 after the same number of training steps, but only 5.9 after the same training time. {#c-zhang-overhead}

## Evidence

**Recurrent networks.**
On an order-embedding model of images and captions, with a GRU sentence encoder, layer normalization offers a per-iteration speedup across all metrics and converges to its best validation model in 60% of the time it takes the baseline model. {#c-oe-speedup}
On the test sets, caption retrieval R@1 goes from 46.6 to 48.5, and image retrieval R@1 from 37.8 to 38.9. {#c-oe-table}
On a unidirectional attentive reader on the CNN corpus, with layer normalization only within the LSTM, it not only trains faster but converges to a better validation result over both the baseline and the batch normalization variants. {#c-reader}
With the initial gain at 1.0 or 0.1, the model with 1.0 performs significantly better, so layer normalization is not sensitive to the initial scale in the way recurrent batch normalization is. {#c-reader-init}

**Feed-forward and convolutional networks.**
On permutation invariant MNIST, with layer normalization on the hidden layers but not the last softmax layer, it is robust to the batch size and converges faster than batch normalization applied to all layers. {#c-mnist}
In preliminary experiments with convolutional networks, layer normalization offers a speedup over the baseline model without normalization, but batch normalization outperforms the other methods. {#c-conv}
With ResNet-50 on ImageNet at 32 images per GPU, the validation error is 23.6% with batch normalization, 25.3% with layer normalization and 24.1% with group normalization. {#c-wu-table}

**Transformers, Post-LN and Pre-LN.**
For a Post-LN Transformer on IWSLT14 De-En trained with Adam, without the warm-up stage the BLEU score only reaches 8.45, against around 34 with warm-up. {#c-warmup-numbers}
With a warm-up of 500 steps, the BLEU scores are 31.16 and 2.77 for the maximal learning rates $5 \cdot 10^{-4}$ and $10^{-3}$. {#c-warmup-500}
Xiong et al. prove that at initialization the gradient of the parameters near the output of a Post-LN Transformer has a scale of $O(d\sqrt{\ln d})$, independent of the depth $L$, and that of a Pre-LN Transformer a scale of $O(d\sqrt{\ln d / L})$. {#c-theorem1}
They prove in Lemma 2 that the scale of the hidden states of a Post-LN Transformer keeps to be the same in expectation over the layers, while that of a Pre-LN Transformer grows linearly along with the depth. {#c-lemma2}
In their experiments, the gradient norm stays at around 1.6 over the layers of a Post-LN Transformer and decreases in a Pre-LN Transformer. {#c-grad-norm}
Without warm-up, Pre-LN Transformers reach comparable results with baselines while requiring significantly less training time and hyper-parameter tuning. {#c-preln-no-warmup}
On translation, the 9th checkpoint of the Pre-LN Transformer achieves nearly the same performance as the 15th checkpoint of the Post-LN Transformer with warm-up. {#c-preln-checkpoints}
In BERT pre-training, Pre-LN reaches a similar validation loss with a speed-up rate of about 40%. {#c-bert}
Liu et al. (2020) find that Pre-LN converges in all 15 of their settings, Post-LN diverges in 7 of 15, and when Post-LN converges, it outperforms Pre-LN in 7 of 8. {#c-liu-settings}

**RMSNorm.**
On a GRU RNNSearch model, RMSNorm and layer normalization reduce the number of training steps to convergence by about 50% against the baseline. {#c-zhang-steps}
In the same TensorFlow setting, the model with LayerNorm is slower per step than the baseline by about 67%, and RMSNorm improves on LayerNorm by about 25%. {#c-zhang-tf}
On a Transformer, whose training fails without normalization, RMSNorm is comparable to LayerNorm and yields a speedup of 7% to 9%. {#c-zhang-transformer}

## Limitations and failure modes

**Convolutional networks.**
Hidden units whose receptive fields lie near the boundary of the image are rarely turned on and have very different statistics from the rest of the units in the same layer, and Ba et al. think further research is needed to make layer normalization work well in convolutional networks. {#c-conv-boundary}
Group normalization, which uses several groups of channels, does better in that setting.

**Post-LN needs warm-up.**
From their theory, Xiong et al. conclude that in Post-LN Transformers the expected gradients of the parameters near the output layer are large at initialization, so a large learning rate on them makes the training unstable. {#c-postln-large-grad}
They call the learning rate warm-up crucial to the final performance, but note that it slows down the optimization and brings more hyper-parameter tuning. {#c-warmup-crucial}

**Pre-LN is not free.**
The residual stream of a Pre-LN Transformer is never normalized, so its scale grows with depth, as Lemma 2 of Xiong et al. states.
Liu et al. argue that unbalanced gradients are not the root cause of the instability of Post-LN, and that a heavy dependency on the residual branch makes training unstable because it amplifies small parameter perturbations. {#c-liu-root}
A Post-LN layer has a heavier dependency on its residual branch than a Pre-LN layer. {#c-liu-dependency}
A light dependency, as in Pre-LN, limits the model potential and leads to inferior trained models. {#c-liu-light}

**Not before the logits.**
Layer normalization makes the output invariant to the scale of its input.
This is unnecessary for the logit outputs, where the prediction confidence is determined by the scale of the logits, so Ba et al. do not apply it to the last softmax layer. {#c-no-softmax}

**Weight decay.**
By the matrix invariance, an L2 penalty on the weights before the layer cannot change the function of the network, only the norm of its weights and so the effective learning rate.
The derivation `invariances` shows that plain gradient descent only grows that norm.

## Variants

- **RMSNorm** divides by the root mean square and drops the mean and the bias (Zhang and Sennrich 2019). It is used in LLaMA. {#c-rmsnorm-variant}
- **pRMSNorm** estimates the root mean square from p% of the summed inputs, with p = 6.25% in the experiments of Zhang and Sennrich. {#c-prms}
- **Pre-LN and Post-LN** are the two placements of the layer in a residual block (Xiong et al. 2020).
- **Group normalization** normalizes groups of channels of each example, and one group is layer normalization over $(C, H, W)$ with per-channel affine parameters (Wu and He 2018).
- **Weight normalization** reparameterizes each weight vector by a length and a direction, and normalizes the weights instead of the summed inputs; it does not use statistics of the activations at all. See weight-normalization.
- **Batch normalization** normalizes each feature over the batch, and it is a different layer, not a variant of this one; see batch-normalization.

## When to use it

Use layer normalization in recurrent networks and in Transformers, and wherever the batch is small, varies in size, or must not mix examples.
In a Transformer, Pre-LN trains without a careful warm-up and is the safer default for deep models.
Post-LN can reach a better final model when it converges, but it needs a warm-up and a tuned learning rate.
RMSNorm is a cheaper choice that usually matches it; most recent large language models use RMSNorm in the Pre-LN position.
In convolutional networks with large batches, prefer batch normalization; with small batches, prefer group normalization.

## Pitfalls

> [!pitfall]
> **Normalizing over the wrong axis.** A layer that takes the statistics over the batch, or over the sequence, is not layer normalization: its output for one example depends on the others. The test `test_batch_independence` checks that each example's output is unchanged when the rest of the batch changes, for both layer normalization and RMSNorm.

> [!pitfall]
> **The unbiased variance, or epsilon outside the square root.** PyTorch normalizes with the biased variance and adds $\epsilon$ inside the square root. A copy that uses the unbiased variance, or divides by $\sigma + \epsilon$, drifts from it. The test `test_matches_torch_layernorm` compares outputs and gradients with `torch.nn.LayerNorm` and `torch.nn.functional.layer_norm`.

> [!pitfall]
> **Running statistics carried over from batch normalization.** Layer normalization keeps no running mean or variance, and `model.eval()` does not change it. The test `test_train_equals_eval` checks that training and evaluation modes give the same output.

> [!pitfall]
> **A gain per channel instead of per element.** Over a normalized shape of several dimensions, the PyTorch gain and bias have that whole shape. A port of group normalization with one group has one gain per channel instead. The test `test_per_element_affine` checks the shape and the effect of the gain.

> [!pitfall]
> **The epsilon of RMSNorm.** PyTorch `RMSNorm` defaults to the machine epsilon of the dtype, not to $10^{-5}$. The tests `test_matches_torch_rmsnorm` and `test_rmsnorm_default_eps_is_not_layernorm_eps` compare with `torch.nn.RMSNorm` for small inputs, where the two values give different outputs.

> [!pitfall]
> **The statistics treated as constants.** A backward pass that ignores the dependence of the mean and the variance on the input gives a wrong gradient. The test `test_gradient_finite_differences` compares the gradient with finite differences, `test_statistics_treated_as_constants_are_wrong` shows the error, and `test_worked_example` checks the numbers of the page.

> [!pitfall]
> **Mixing up the placements.** Post-LN and Pre-LN differ only in the order of two operations, and Pre-LN needs a final normalization before the output layer. The test `test_placements` checks both against eq-post-ln and eq-pre-ln.

## History

Ba, Kiros and Hinton posted Layer Normalization as arXiv 1607.06450v1 on 21 Jul 2016. {#c-hist-ba}
Vaswani et al. used it after each residual connection of the Transformer in 2017.
Zhang and Sennrich posted Root Mean Square Layer Normalization as arXiv 1910.07467v1 on 16 Oct 2019. {#c-hist-zhang}
Xiong et al. posted On Layer Normalization in the Transformer Architecture, on Post-LN and Pre-LN, with its second version as arXiv 2002.04745v2 on 29 Jun 2020. {#c-hist-xiong}
Touvron et al. (2023) used pre-normalization with RMSNorm in LLaMA.
