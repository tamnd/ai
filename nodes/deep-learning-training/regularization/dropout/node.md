---
tag: Y8VNMB
kind: method
slug: dropout
name: Dropout
aliases: [inverted dropout, dropout regularization]
area: deep-learning-training/regularization
level: core
summary: >-
  During training, set each unit of a layer to zero with probability p and scale the other units by 1/(1-p), so that the network cannot rely on any single unit.
purpose: regularization
uses: [bernoulli-distribution@m2]
related_to: [ensemble-learning@al, bagging@al, weight-decay@al, data-augmentation@al]
hyperparameters:
  - {name: p, meaning: "the probability that a unit is set to zero", typical: "0.1 to 0.5 for hidden units, 0 to 0.2 for inputs", default_in: {pytorch: 0.5}}
cost:
  train: "one Bernoulli sample, one multiply and one scale per activation; one mask of the activation shape in memory until the backward pass"
  inference: "none: the layer is the identity"
history:
  year: 2012
  note: "Hinton et al. (2012), arXiv 1207.0580; the full study is Srivastava et al. (2014), JMLR 15."
prereqs: [overfitting@al, regularization@al, backpropagation@al, bernoulli-distribution@m2]
same_as:
  wikidata: Q25339462
  wikipedia: Dilution_(neural_networks)
  arxiv: ["1207.0580"]
  al: dropout
sources:
  - {paper: "Srivastava, Hinton, Krizhevsky, Sutskever and Salakhutdinov (2014)", title: "Dropout: A Simple Way to Prevent Neural Networks from Overfitting", locator: "JMLR 15(56), 1929 to 1958", note: "the full study and the canonical reference"}
  - {paper: "Hinton, Srivastava, Krizhevsky, Sutskever and Salakhutdinov (2012)", title: "Improving neural networks by preventing co-adaptation of feature detectors", locator: "arXiv 1207.0580", note: "the first publication"}
  - {book: "Goodfellow, Bengio and Courville (2016), Deep Learning", locator: "Section 7.12", note: "the textbook treatment, with the ensemble view"}
  - {paper: "Wager, Wang and Liang (2013)", title: "Dropout Training as Adaptive Regularization", locator: "NeurIPS 26, the sections on generalized linear models"}
  - {paper: "Baldi and Sadowski (2013)", title: "Understanding Dropout", locator: "NeurIPS 26, the sections on the normalized weighted geometric mean"}
  - {paper: "Gal and Ghahramani (2016)", title: "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning", locator: "ICML 2016, PMLR 48, Sections 3 and 4"}
  - {paper: "Li, Chen, Hu and Yang (2019)", title: "Understanding the Disharmony between Dropout and Batch Normalization by Variance Shift", locator: "CVPR 2019, the sections on variance shift"}
  - {paper: "Zaremba, Sutskever and Vinyals (2014)", title: "Recurrent Neural Network Regularization", locator: "arXiv 1409.2329, Section 3"}
  - {paper: "Wan, Zeiler, Zhang, LeCun and Fergus (2013)", title: "Regularization of Neural Networks using DropConnect", locator: "ICML 2013, PMLR 28(3), the model description"}
  - {paper: "Tompson, Goroshin, Jain, LeCun and Bregler (2015)", title: "Efficient Object Localization Using Convolutional Networks", locator: "CVPR 2015, Section 3.2"}
  - {paper: "Ghiasi, Lin and Le (2018)", title: "DropBlock: A regularization method for convolutional networks", locator: "NeurIPS 31, Section 3"}
  - {paper: "Huang, Sun, Liu, Sedra and Weinberger (2016)", title: "Deep Networks with Stochastic Depth", locator: "ECCV 2016, Section 3"}
  - {paper: "Krizhevsky, Sutskever and Hinton (2012)", title: "ImageNet Classification with Deep Convolutional Neural Networks", locator: "NeurIPS 25, Section 4.2"}
  - {paper: "Chowdhery et al. (2022)", title: "PaLM: Scaling Language Modeling with Pathways", locator: "arXiv 2204.02311, Section 5"}
  - {paper: "Bishop (1995)", title: "Training with Noise is Equivalent to Tikhonov Regularization", locator: "Neural Computation 7(1), 108 to 116"}
  - {paper: "Vaswani et al. (2017)", title: "Attention Is All You Need", locator: "NeurIPS 30, Section 5.4"}
  - {url: "https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html", title: "torch.nn.Dropout", locator: "the parameters p and inplace", note: "the PyTorch default and its scaling"}
  - {url: "https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.dropout.html", title: "torch.nn.functional.dropout", locator: "the parameter training"}
  - {url: "https://keras.io/api/layers/regularization_layers/dropout/", title: "Keras Dropout layer", locator: "the argument rate"}
  - {url: "https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/stochastic.html", title: "flax.nnx.Dropout", locator: "the parameters rate, deterministic and rngs"}
---

Dropout is a regularization method for neural networks.
During training, it sets a random part of the units of a layer to zero at each step, and it scales the units that stay on.
At inference it does nothing.
It reduces overfitting in feedforward, convolutional and recurrent networks, and it adds almost no cost.
It is also the base of Monte Carlo dropout, a cheap estimate of the uncertainty of a prediction.

## In plain words

While the network trains, switch off a random part of its units at each step.
The network cannot count on any one unit, so it learns features that work even when some are missing.
At test time all units are on, scaled so that the total signal is the same as in training.
The effect is close to averaging many smaller networks that share their weights.

## Setting

Dropout addresses overfitting: a large network fits the noise of its training set and does worse on new data.
It applies to a layer with activations $\mathbf{h} \in \mathbb{R}^d$, the output of a fully connected, convolutional or recurrent layer, or the input $\mathbf{x}$.
It needs no change to the loss, to the optimizer or to the architecture.
It assumes training by stochastic gradient descent with a new random mask at each step.
The original study used it in feedforward networks for vision, speech and text, in convolutional networks and in restricted Boltzmann machines.

## The method

Pick the drop probability $p \in [0, 1)$.
At each training step, draw a mask $\mathbf{m} \in \{0, 1\}^d$ with independent entries, each 1 with probability $1 - p$.
The layer gives

$$ \tilde{\mathbf{h}} = \frac{1}{1 - p} \, \mathbf{m} \odot \mathbf{h}, \qquad m_j \sim \mathrm{Bernoulli}(1 - p) $$ {#eq-inverted}

where $\odot$ is the elementwise product.
The backward pass uses the same mask: $\partial \ell / \partial \mathbf{h} = \frac{1}{1 - p} \, \mathbf{m} \odot \partial \ell / \partial \tilde{\mathbf{h}}$.
At inference, the layer gives $\tilde{\mathbf{h}} = \mathbf{h}$.

This form is inverted dropout, the form of the main frameworks.
The original papers used the other form: no scale in training, and a scale of $1 - p$ on the outgoing weights at test time.
The two forms give the same network at test time.

> [!note]
> Srivastava et al. (2014) write $p$ for the probability of keeping a unit. This page writes $p$ for the probability of dropping it, as PyTorch, Keras and Flax do. Their $p = 0.5$ is our $p = 0.5$; their $p = 0.8$ for inputs is our $p = 0.2$.

## Derivation

The scale $1/(1-p)$ keeps the expected activation the same in training and at inference.
For one unit, $m_j$ is 1 with probability $1 - p$, so $\mathbb{E}[m_j] = 1 - p$.
Then $\mathbb{E}[\tilde{h}_j] = \frac{1}{1-p} \, \mathbb{E}[m_j] \, h_j = \frac{1-p}{1-p} \, h_j = h_j$. {#c-inverted-scaling}
The noise has a variance that grows with $p$.
Because $m_j^2 = m_j$, we get $\mathbb{E}[\tilde{h}_j^2] = \frac{1-p}{(1-p)^2} \, h_j^2 = \frac{h_j^2}{1-p}$.
So $\mathrm{Var}[\tilde{h}_j] = \frac{h_j^2}{1-p} - h_j^2 = \frac{p}{1-p} \, h_j^2$. {#c-noise-variance}

For linear regression with dropout on the inputs, the expected loss is the least squares loss plus a penalty on the weights.
Let $\mathbf{X}$ be the data matrix with rows $\mathbf{x}_i$, let $\mathbf{w}$ be the weights and let $\mathbf{M}$ be a mask of the shape of $\mathbf{X}$.
The prediction for example $i$ is $\hat{y}_i = \frac{1}{1-p} \sum_j M_{ij} X_{ij} w_j$.
Its mean is $\mathbf{x}_i^\top \mathbf{w}$ by the result above.
Its variance is $\frac{p}{1-p} \sum_j X_{ij}^2 w_j^2$, because the entries of the mask are independent.
The mean squared error of a prediction is its squared bias plus its variance, so

$$ \mathbb{E}_{\mathbf{M}} \sum_i (y_i - \hat{y}_i)^2 = \lVert \mathbf{y} - \mathbf{X}\mathbf{w} \rVert^2 + \frac{p}{1-p} \sum_j \Big( \sum_i X_{ij}^2 \Big) w_j^2 . $$ {#eq-ridge}

This is ridge regression with a different penalty for each weight, scaled by the energy of its input feature. {#c-linear-ridge}
Srivastava et al. (2014, Section 9.1) give the same result for the original form of dropout.
Wager et al. (2013) extend it to generalized linear models, where the penalty is close to an L2 penalty scaled by an estimate of the Fisher information. {#c-adaptive-l2}

## Worked example

Take $\mathbf{h} = (2, 4, -1, 3)$ and $p = 0.5$, so the scale is $1/(1-p) = 2$.
Say the mask is $\mathbf{m} = (1, 0, 1, 1)$.
The output is $\tilde{\mathbf{h}} = 2 \cdot (2, 0, -1, 3) = (4, 0, -2, 6)$.
If the gradient that comes back is $(1, 1, 1, 1)$, the gradient for $\mathbf{h}$ is $2 \cdot (1, 0, 1, 1) = (2, 0, 2, 2)$.
The second unit gets no gradient at this step.

Take the same $\mathbf{h}$ with $p = 0.2$, so the scale is $1.25$, and the mask $(1, 1, 0, 1)$.
The output is $(2.5, 5, 0, 3.75)$.
Over all 16 masks, weighted by their probabilities, the mean of the output is $\mathbf{h}$, as the derivation says.

## Hyperparameters

The only hyperparameter is the drop probability $p$.
A larger $p$ gives more noise and a stronger regularization, and the network needs more steps to converge.
Srivastava et al. (2014, Appendix A.4) keep hidden units with probability 0.5 to 0.8 and real-valued inputs with probability 0.8. {#c-typical-rates}
In the notation of this page, that is $p$ from 0.2 to 0.5 for hidden units and $p = 0.2$ for inputs.
They also suggest a wider layer when $p$ is large, so that the number of units that stay on is about the same (Appendix A.1).
The base Transformer of Vaswani et al. (2017, Section 5.4) uses $p = 0.1$.
In PyTorch, `torch.nn.Dropout` has the default $p = 0.5$.
Keras `Dropout` and Flax `nnx.Dropout` have no default rate; the caller gives it.

## Cost

Let $d$ be the number of activations of the layer in one batch.
In training, dropout draws $d$ random bits, does $d$ multiplies and keeps a mask of $d$ entries until the backward pass.
A fused kernel can store the mask as bits or draw it again from the seed in the backward pass.
At inference, inverted dropout costs nothing, because the layer is the identity.
Training with dropout usually takes more steps to converge.
Srivastava et al. (2014, Section 11) report two to three times the training time of the same network without dropout. {#c-slower-training}

## Evidence

Srivastava et al. (2014) report lower test error with dropout than without it on each of their benchmarks. {#c-reduces-error}
The benchmarks are MNIST, SVHN, CIFAR-10, CIFAR-100, ImageNet, TIMIT and Reuters, with networks of the size of that time.
The same study shows that the weight scaling rule at test time gives about the same error as the average over many sampled masks (Section 7.5). {#c-weight-scaling-close}
Baldi and Sadowski (2013) show that for one layer with a logistic output, the scaled network computes the normalized weighted geometric mean of the sub-networks. {#c-geometric-mean}

## Limitations and failure modes

Dropout before batch normalization can hurt test accuracy.
The variance of a unit changes from training to inference, so the statistics that batch normalization keeps do not match at test time (Li et al. 2019). {#c-variance-shift}
Li et al. (2019) suggest putting dropout after the last batch normalization layer, or using a form with less variance.
Dropout on the recurrent connections of an LSTM, with a new mask at each time step, interferes with the memory of the network.
Zaremba et al. (2014) apply dropout only to the connections between layers, not to the recurrent ones. {#c-rnn-nonrecurrent}
With a lot of data per parameter, dropout gives less benefit, because the network overfits less (Srivastava et al. 2014, Section 7.4).
Some large language models are trained with no dropout at all, such as PaLM (Chowdhery et al. 2022, Section 5). {#c-llm-no-dropout}

## Variants

- **Gaussian dropout** multiplies each unit by noise from $\mathcal{N}(1, p/(1-p))$, which has the same mean and variance as the inverted Bernoulli mask (Srivastava et al. 2014, Section 10).
- **DropConnect** sets random weights to zero, not random units (Wan et al. 2013).
- **Spatial dropout** drops whole feature maps of a convolutional layer, because neighbouring pixels are strongly correlated (Tompson et al. 2015).
- **DropBlock** drops square regions of a feature map (Ghiasi et al. 2018).
- **Stochastic depth** drops whole residual blocks and keeps the skip path (Huang et al. 2016).
- **Variational dropout for recurrent networks** uses the same mask at each time step of a sequence (Gal and Ghahramani 2016, NeurIPS).
- **Monte Carlo dropout** keeps dropout on at inference and averages several forward passes, to estimate the uncertainty of a prediction (Gal and Ghahramani 2016, ICML). {#c-mc-dropout}

## When to use it

Use dropout when a network overfits and you can afford more training steps.
It works with weight decay and with max-norm constraints, and Srivastava et al. (2014) combine it with both.
Data augmentation and early stopping are alternatives for the same purpose.
In convolutional networks with batch normalization, put dropout after the last normalization layer, or use spatial dropout or DropBlock.
In recurrent networks, apply it between layers, or use the same mask at each time step.

## Pitfalls

> [!pitfall]
> **Dropout stays on at inference.** In PyTorch, `torch.nn.functional.dropout` has `training=True` by default, so a module that calls it without `training=self.training` drops units in evaluation mode. The same happens when the code does not call `model.eval()`. The test `test_eval_is_identity` of each implementation checks that the layer is the identity in evaluation mode.

> [!pitfall]
> **The same mask at each step.** In JAX, a random key that is not split gives the same mask at every step, and the noise is no longer random. The test `test_masks_differ` checks that two steps give two masks.

> [!pitfall]
> **The wrong scale.** Code that drops units in training and also scales by $1 - p$ at inference applies the scale twice with inverted dropout. The test `test_expectation` checks that the mean of many outputs is $\mathbf{h}$.

> [!pitfall]
> **The mask is not the one of the forward pass.** A backward pass that draws a new mask gives a wrong gradient. The test `test_gradient` compares the gradient with finite differences for a fixed mask.

## History

Geoffrey Hinton, Nitish Srivastava, Alex Krizhevsky, Ilya Sutskever and Ruslan Salakhutdinov first described dropout in 2012, in the preprint arXiv 1207.0580. {#c-first-2012}
Krizhevsky, Sutskever and Hinton (2012) used it in the first two fully connected layers of AlexNet, the winning entry of the ImageNet challenge of 2012. {#c-alexnet}
The full study, with the experiments, the theory for linear models and the practical guide, is Srivastava et al. (2014) in the Journal of Machine Learning Research. {#c-canonical-2014}
The idea is related to older work on training with noise: Bishop (1995) showed that training with small input noise is close to Tikhonov regularization.
