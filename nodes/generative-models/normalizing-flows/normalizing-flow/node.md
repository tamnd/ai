---
tag: NEW-normalizing-flow
kind: architecture
slug: normalizing-flow
name: Normalizing flow
aliases: [flow-based model, flow-based generative model, normalizing flows, NICE, RealNVP, Real NVP, Glow, MAF, IAF, planar flow, coupling flow, affine coupling layer]
area: generative-models/normalizing-flows
level: core
summary: >-
  A generative model that maps data to a simple base density by an invertible network whose Jacobian determinant is cheap, which gives exact likelihoods and sampling.
uses: [probability-density-function@m2, determinant-as-signed-volume@m2, determinant-of-a-triangular-matrix@m2, matrix-determinant-lemma@m2, cumulative-distribution-function@m2, kullback-leibler-divergence@m2, differential-entropy@m2, multivariate-normal-distribution@m2, matrix-multiplication@m2, maximum-likelihood-estimation@al, variational-inference@al, backpropagation@al]
related_to: [variational-autoencoder, generative-adversarial-network, neural-ordinary-differential-equation, autoencoder, multilayer-perceptron, convolutional-neural-network, adam, batch-normalization]
history:
  year: 2014
  people: [Laurent Dinh, David Krueger, Yoshua Bengio]
  note: "Dinh, Krueger and Bengio posted NICE as arXiv 1410.8516 in October 2014, a stack of additive coupling layers trained by exact maximum likelihood. Rezende and Mohamed named normalizing flows as posteriors for variational inference (2015), Dinh, Sohl-Dickstein and Bengio added the affine coupling layer and a multi-scale architecture with Real NVP (2016), Papamakarios, Pavlakou and Murray stacked autoregressive layers into MAF (2017), and Kingma and Dhariwal added the invertible 1 x 1 convolution with Glow (2018)."
prereqs: [multilayer-perceptron, probability-density-function@m2, determinant-of-a-triangular-matrix@m2, kullback-leibler-divergence@m2, maximum-likelihood-estimation@al]
same_as:
  wikipedia: Flow-based_generative_model
sources:
  - {paper: "Dinh, Krueger and Bengio (2014)", title: "NICE: Non-linear Independent Components Estimation", locator: "arXiv 1410.8516v6, pp. 1 to 2, 4, 6 to 7", note: "the change of variables, the additive coupling layer, alternating partitions, the diagonal scaling, dequantization and the log-likelihoods"}
  - {paper: "Rezende and Mohamed (2015)", title: "Variational Inference with Normalizing Flows", locator: "ICML 2015; arXiv 1505.05770v6, pp. 5, 8, 10", note: "the planar flow, its determinant by the matrix determinant lemma, the invertibility condition and u-hat, and the MNIST bounds"}
  - {paper: "Dinh, Sohl-Dickstein and Bengio (2016)", title: "Density estimation using Real NVP", locator: "ICLR 2017; arXiv 1605.08803v3, pp. 4 to 9", note: "the affine coupling layer, its Jacobian and inverse, the masks and the multi-scale architecture, and the bits per dimension"}
  - {paper: "Papamakarios, Pavlakou and Murray (2017)", title: "Masked Autoregressive Flow for Density Estimation", locator: "arXiv 1705.07057v4, pp. 2 to 5, 7", note: "the banana density, MAF and IAF and their passes, coupling as a special case, and the UCI and BSDS300 results"}
  - {paper: "Kingma and Dhariwal (2018)", title: "Glow: Generative Flow with Invertible 1x1 Convolutions", locator: "arXiv 1807.03039v2, pp. 3 to 8", note: "the dequantized objective, actnorm, the invertible 1 x 1 convolution and its LU form, zero initialization, the results, sampling speed and temperature"}
  - {paper: "Papamakarios, Nalisnick, Rezende, Mohamed and Lakshminarayanan (2021)", title: "Normalizing Flows for Probabilistic Modeling and Inference", locator: "JMLR 22; arXiv 1912.02762v2, pp. 3 to 8, 12 to 14, 19 to 21, 30 to 31, 43", note: "the definitions, the KL objectives, universality, the cost of determinants and inverses, continuous-time flows and dequantization"}
family: normalizing-flow
inputs: "a batch of data points as a (B, D) array for density evaluation, or a batch of base samples as a (B, D) array for sampling"
outputs: "for density evaluation the codes z as a (B, D) array and one log-density per point, a (B,) array; for sampling B points as a (B, D) array"
components: []
inductive_biases: ["the data distribution is the image of a simple base density, usually a standard normal, under a smooth invertible map", "the map is a composition of layers each of which changes volume in a way that is cheap to measure, so the model is restricted to maps with triangular or low-rank Jacobians", "the code has the dimension of the data, so the model has no bottleneck and assigns positive density to every point of the space", "the order of the coordinates or the masks of the coupling layers decides which dependencies each layer can express"]
params: "L(3DH + H + 2D) for L affine coupling layers on D dimensions, each with one hidden layer of H units that computes the scale and the shift"
flops_per_example: "6LDH for a density pass or a sampling pass, counting a multiply-add as 2 FLOPs and only the matrix products"
---

A normalizing flow is a generative model built from an invertible map between the data space and the space of a simple distribution, the base density, usually a standard normal.
Because the map is invertible, the model's density at any data point is exact: it is the base density at the point's image times the factor by which the map stretches or squeezes volume there.
Sampling runs the map backwards from draws of the base density.
Dinh, Krueger and Bengio's NICE learns a non-linear deterministic transformation of the data that maps it to a latent space so that the transformed data conform to a factorized distribution; the transformation is parametrized so that the determinant of the Jacobian and the inverse are trivial to compute, the training criterion is the exact log-likelihood, and unbiased ancestral sampling is easy. {#c-nice-def}
The design problem of every flow is the one NICE names: a map flexible enough to model images, speech or physical measurements, whose inverse and Jacobian determinant are still cheap.

## In plain words

Think of the data as a lump of dough with a complicated shape and the base density as a round ball of the same dough.
A flow is a sequence of reversible kneading moves that turn the ball into the lump.
Each move may stretch some regions and squeeze others, and the model keeps track of how much: where the dough has been stretched, it is thinner, so the density there is lower.
To ask how likely a data point is, run the moves backwards to see where the point came from in the ball, read the density of the ball there, and correct for the stretching along the way.
To make a new data point, take a random point of the ball and run the moves forwards.
The art is in choosing moves that are both expressive and easy to undo and to measure; a coupling layer, the most common move, leaves half the coordinates alone and stretches and shifts the other half by amounts that depend only on the half left alone.

## Overview

Let $f$ map a data point $\mathbf{x} \in \mathbb{R}^D$ to a code $\mathbf{z} = f(\mathbf{x}) \in \mathbb{R}^D$ and let the base density be $p_z$, for example $\mathcal{N}(0, I_D)$.
If $f$ is a diffeomorphism, a differentiable bijection with a differentiable inverse, the data have the density

$$ \log p_x(\mathbf{x}) = \log p_z\big(f(\mathbf{x})\big) + \log \left| \det \frac{\partial f}{\partial \mathbf{x}} \right| , $$ {#eq-change-of-variables}

the base density at the image times the factor by which $f$ changes volume near $\mathbf{x}$, the [[m2:determinant-as-signed-volume|determinant as a volume]]. {#c-cov}
The derivation `change-of-variables` proves it.
Papamakarios et al. explain the name: the term 'normalizing' refers to the fact that the inverse flow takes a collection of samples from $p_x$ and transforms them, in a sense normalizes them, into a collection of samples from a prescribed density, often a [[m2:multivariate-normal-distribution|multivariate normal]]. {#c-review-normalizing}
In this page the direction $f$ goes from the data to the base space, the one that evaluates densities, and sampling runs $f^{-1}$; papers use both conventions.

A flow is a composition $f = f_L \circ \cdots \circ f_1$ of simple layers.
With $\mathbf{h}_0 = \mathbf{x}$ and $\mathbf{h}_k = f_k(\mathbf{h}_{k-1})$, the log-determinant is the sum of the layers' log-determinants,

$$ \log \left| \det \frac{\partial f}{\partial \mathbf{x}} \right| = \sum_{k=1}^{L} \log \left| \det \frac{\partial f_k}{\partial \mathbf{h}_{k-1}} \right| , $$ {#eq-logdet}

each evaluated at that layer's input, and the inverse is the composition of the layers' inverses in reverse order, so a flow is built from layers whose determinants and inverses are cheap. {#c-compose}
A general $D \times D$ determinant costs $O(D^3)$.
Papamakarios et al. put the constraint as a budget: explicitly calculating the determinant of a general Jacobian has a time cost that can be intractable for large $D$, while for most applications of flow-based models the Jacobian-determinant computation should be at most $O(D)$. {#c-review-det-cost}
The families of flows differ in how they meet it: coupling layers and autoregressive layers have triangular Jacobians, whose determinants are the products of their diagonals, and planar flows have Jacobians that are rank-one updates of the identity.

Training is [[al:maximum-likelihood-estimation|maximum likelihood]]: eq-change-of-variables is exact, so the loss is the mean negative log-likelihood of the data, with no bound, no sampling and no second network.
Unlike a [[variational-autoencoder]], which maximizes a lower bound and has a code of lower dimension, and a [[generative-adversarial-network]], which has no likelihood at all, a flow has an exact density and an exact inverse, at the price of a code as large as the data and layers restricted to cheap Jacobians.

## Components

The notation has vectors as rows; a layer maps $\mathbf{h}$ to $\mathbf{h} W + \mathbf{b}$ with $W$ of shape $(n_\text{in}, n_\text{out})$, and $\odot$ is the element-wise product.

**The affine coupling layer.** A binary mask $\mathbf{b} \in \{0, 1\}^D$ marks the coordinates the layer keeps.
The kept coordinates go through a network, here a [[multilayer-perceptron]] with one tanh hidden layer of $H$ units, which outputs a scale $\mathbf{s}$ and a shift $\mathbf{t}$ for the other coordinates:

$$ \mathbf{a} = \tanh\big((\mathbf{b} \odot \mathbf{x}) W_1 + \mathbf{b}_1\big), \quad \mathbf{s} = (1 - \mathbf{b}) \odot (\mathbf{a} W_s + \mathbf{b}_s), \quad \mathbf{t} = (1 - \mathbf{b}) \odot (\mathbf{a} W_t + \mathbf{b}_t), \quad \mathbf{y} = \mathbf{x} \odot \exp(\mathbf{s}) + \mathbf{t} . $$ {#eq-coupling}

The masks on $\mathbf{s}$ and $\mathbf{t}$ make them 0 on the kept coordinates, so those pass through unchanged.
This is Real NVP's layer: given a $D$ dimensional input and $d < D$, the output of an affine coupling layer is $\mathbf{y}_{1:d} = \mathbf{x}_{1:d}$ and $\mathbf{y}_{d+1:D} = \mathbf{x}_{d+1:D} \odot \exp\big(s(\mathbf{x}_{1:d})\big) + t(\mathbf{x}_{1:d})$, where $s$ and $t$ stand for scale and translation. {#c-realnvp-affine}
Ordering the coordinates as kept then transformed, the Jacobian is block lower triangular,

$$ \frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \begin{pmatrix} I & 0 \\ \partial \mathbf{y}_T / \partial \mathbf{x}_K & \operatorname{diag}\big(\exp \mathbf{s}\big) \end{pmatrix}, \qquad \log \left| \det \frac{\partial \mathbf{y}}{\partial \mathbf{x}} \right| = \sum_j s_j , $$ {#eq-coupling-jacobian}

so log |det| is the sum of $\mathbf{s}$ over the transformed coordinates, whatever the functions $s$ and $t$ are, by the [[m2:determinant-of-a-block-triangular-matrix|determinant of a block-triangular matrix]]. {#c-coupling-jac}
The layer is inverted by $\mathbf{x}_K = \mathbf{y}_K$ and $\mathbf{x}_T = (\mathbf{y}_T - \mathbf{t}(\mathbf{y}_K)) \odot \exp(-\mathbf{s}(\mathbf{y}_K))$, which evaluates $s$ and $t$ once, in the forward direction, so the inverse costs what the layer costs and $s$ and $t$ never have to be inverted. {#c-coupling-inverse}
The derivation `coupling-jacobian` proves both.
Dinh et al. draw the consequence: since computing the Jacobian determinant of the coupling layer does not involve computing the Jacobian of $s$ or $t$, those functions can be arbitrarily complex, and Real NVP makes them deep convolutional neural networks; the inverse of the coupling layer does not require computing the inverse of $s$ or $t$. {#c-realnvp-arbitrary}

**The additive coupling layer** of NICE is the case $\mathbf{s} = 0$: it keeps one block of the input, $\mathbf{y}_1 = \mathbf{x}_1$, and shifts the other, $\mathbf{y}_2 = \mathbf{x}_2 + m(\mathbf{x}_1)$, with $m$ an arbitrarily complex function such as a ReLU MLP; it has unit Jacobian determinant for any $m$, and its inverse $\mathbf{x}_2 = \mathbf{y}_2 - m(\mathbf{y}_1)$ is only as expensive as the transformation itself. {#c-nice-additive}
Each additive coupling layer is volume preserving, so their composition has unit Jacobian determinant too; NICE adds a diagonal scaling matrix $S$ as the top layer, which multiplies the $i$-th output value by $S_{ii}$. {#c-nice-scaling}

**Alternating masks.** A coupling layer leaves part of its input unchanged, so consecutive layers exchange the roles of the two subsets; examining the Jacobian, Dinh et al. find that at least three coupling layers are necessary to allow all dimensions to influence one another, and NICE generally uses four. {#c-nice-three}
The reference's masks keep the coordinates of even index in even layers and those of odd index in odd layers.
For images, Real NVP uses two partitionings that exploit the local correlation structure of images, a spatial checkerboard pattern and a channel-wise masking, and a multi-scale architecture with a squeezing operation that reduces a $4 \times 4 \times 1$ tensor to a $2 \times 2 \times 4$ tensor. {#c-realnvp-masks}

**The base density.** A standard normal $\mathcal{N}(0, I_D)$, with $\log p_z(\mathbf{z}) = -\tfrac12 \lVert \mathbf{z} \rVert^2 - \tfrac{D}{2} \log 2\pi$; NICE also used a standard logistic density.

**The planar flow.** Rezende and Mohamed's planar flow is $f(\mathbf{z}) = \mathbf{z} + \mathbf{u}\, h(\mathbf{w}^\top \mathbf{z} + b)$ with free parameters $\mathbf{w}, \mathbf{u} \in \mathbb{R}^D$ and $b \in \mathbb{R}$ and a smooth element-wise non-linearity $h$; its logdet-Jacobian term is computed in $O(D)$ time using the matrix determinant lemma, as $|1 + \mathbf{u}^\top \boldsymbol{\psi}(\mathbf{z})|$ with $\boldsymbol{\psi}(\mathbf{z}) = h'(\mathbf{w}^\top \mathbf{z} + b)\, \mathbf{w}$. {#c-planar-def}
The Jacobian is $I + \boldsymbol{\psi} \mathbf{u}^\top$, a rank-one update of the identity, and the [[m2:matrix-determinant-lemma|matrix determinant lemma]] gives $\det \partial f / \partial \mathbf{z} = 1 + \mathbf{u}^\top \boldsymbol{\psi}(\mathbf{z})$, so its log-determinant costs $O(D)$ instead of the cost of a general determinant. {#c-planar-det}
The map is not always invertible.
With $h = \tanh$ it is invertible when $\mathbf{w}^\top \mathbf{u} \ge -1$, and for any $\mathbf{u}$ the vector

$$ \hat{\mathbf{u}} = \mathbf{u} + \big(m(\mathbf{w}^\top \mathbf{u}) - \mathbf{w}^\top \mathbf{u}\big) \frac{\mathbf{w}}{\lVert \mathbf{w} \rVert^2}, \qquad m(a) = -1 + \log(1 + e^a) , $$ {#eq-planar}

has $\mathbf{w}^\top \hat{\mathbf{u}} = m(\mathbf{w}^\top \mathbf{u}) > -1$, which makes the map invertible for every value of the free parameters. {#c-planar-fix}
Rezende and Mohamed give the condition $\mathbf{w}^\top \mathbf{u} \ge -1$ as sufficient and enforce it by taking an arbitrary vector $\mathbf{u}$ and modifying its component parallel to $\mathbf{w}$, with $m(x) = -1 + \log(1 + e^x)$. {#c-planar-appendix}
The derivation `planar-determinant` proves both statements.
A planar layer is invertible but has no closed-form inverse, so planar flows are used where only one direction is needed, as posteriors in [[al:variational-inference|variational inference]].

**The autoregressive layer.** A masked autoregressive layer computes, for each coordinate $i$, a shift $\mu_i$ and a log-scale $\alpha_i$ from the coordinates before it:

$$ u_i = \big(x_i - \mu_i(\mathbf{x}_{<i})\big) \exp\big(-\alpha_i(\mathbf{x}_{<i})\big), \qquad \log \left| \det \frac{\partial \mathbf{u}}{\partial \mathbf{x}} \right| = -\sum_i \alpha_i(\mathbf{x}_{<i}) . $$ {#eq-autoregressive}

The Jacobian is lower triangular because $u_i$ depends on $x_1, \dots, x_i$ only.
MAF computes all the $\mu_i$ and $\alpha_i$ in one pass of MADE, a network whose weights are multiplied by binary masks so that output $i$ sees only inputs before $i$; the reference uses one masked linear layer.
MAF can calculate the density $p(\mathbf{x})$ of any datapoint in one pass through the model, but sampling from it requires $D$ sequential passes; IAF can generate samples and calculate their density with one pass, but calculating the density of an externally provided datapoint requires $D$ passes. {#c-maf-passes}
The coupling layer is a special case of both the autoregressive transformation used by MAF and the one used by IAF: the coupling layer is recovered from MAF by setting $\mu_i = \alpha_i = 0$ for $i \le d$ and making $\mu_i$ and $\alpha_i$ functions of only $\mathbf{x}_{1:d}$ for $i > d$. {#c-maf-coupling}
Papamakarios et al. state the trade-off: inverting a masked autoregressive flow requires calling the conditioner $D$ times, so it is about $D$ times more expensive than evaluating the forward transformation, which can be prohibitively expensive for images or video; the coupling layer is computationally symmetric, equally fast to evaluate or invert. {#c-review-inversion}

**Glow's step of flow.** Each step of flow in Glow consists of actnorm, followed by an invertible $1 \times 1$ convolution, followed by an affine coupling layer; actnorm is an affine transformation of the activations initialized so that the post-actnorm activations per channel have zero mean and unit variance on an initial minibatch. {#c-glow-step}
Actnorm replaces the [[batch-normalization]] of Real NVP, whose statistics depend on the minibatch.
A $1 \times 1$ convolution with equal numbers of input and output channels generalizes a permutation of the channels; parameterizing its weight matrix in its LU decomposition $W = PL(U + \operatorname{diag}(\mathbf{s}))$ reduces the cost of computing $\det(W)$ from $O(c^3)$ to $O(c)$. {#c-glow-lu}
The learned mixing replaces the fixed alternation of masks.

## Shapes and parameters

The shape contract has data of $D$ dimensions, $L$ coupling layers and a hidden layer of $H$ units in each; the example is $D = 3$, $H = 5$, $L = 3$.
Each coupling layer has $DH + H$ weights and biases in its hidden layer and $2(HD + D)$ in the two output layers for $\mathbf{s}$ and $\mathbf{t}$, so

$$ P = L(3DH + H + 2D) . $$ {#eq-params}

In the contract this is

{shape:params}

and the example has {shape:params@example} parameters.
The flow of the banana test, `banana`, with $D = 2$, $H = 32$ and $L = 4$, has {shape:params@banana}.
The masks and the base density have no parameters.
The weight rows of the masked coordinates of $W_1$ and the columns of the kept coordinates of $W_s$ and $W_t$ are multiplied by 0, so about half of those parameters are inert in each layer; implementations that split the coordinates instead of masking them save that half.
The derivation `parameter-and-flop-count` proves the count.

## Computation and cost

A density pass on $B$ points runs the three products of each coupling layer and costs

$$ C = 6LBDH $$ {#eq-flops}

FLOPs, a multiply-add counted as two; a flow of $L$ coupling layers on $D$ dimensions, each computing $s$ and $t$ from one hidden layer of $H$ units, has $P = L(3DH + H + 2D)$ parameters, and a density pass or a sampling pass on $B$ points costs $6LBDH$ FLOPs in its products. {#c-params}
In the contract,

{shape:flops_forward}

which is {shape:flops_forward@example} FLOPs for the example and {shape:flops_forward@banana} for `banana` at its batch of 256.
The log-determinant adds $O(BD)$ for the sums of $\mathbf{s}$, and the base density $O(BD)$.
The backward pass costs about twice the forward one.
Sampling with coupling layers costs the same as density evaluation, one pass in reverse; with masked autoregressive layers it costs $D$ passes, and with inverse autoregressive layers density evaluation does.
Memory is the main cost at scale: the code has the size of the data, every layer works at full resolution unless a multi-scale architecture factors some dimensions out, and backpropagation stores the activations of every layer.
Because each layer is invertible, its input can be recomputed from its output during the backward pass instead of stored, which makes the memory of training independent of depth at the cost of extra computation.

## Training

The loss is the mean negative log-likelihood of the training data under eq-change-of-variables, minimized with [[adam|Adam]] or another stochastic gradient method; no sampling and no inference network are needed.
The expected negative log-likelihood of a flow is the entropy of the data distribution plus the forward KL divergence from the data to the model, so maximum likelihood minimizes the forward KL divergence and the entropy is the floor of the loss. {#c-mle-kl}
That divergence also has a reading in the base space: the KL divergence from the data distribution to the flow's model equals the KL divergence from the pushed-forward data distribution to the base density, so fitting the model to the data is fitting the transformed data to the base density. {#c-kl-base}
The derivation `kl-equivalence` proves both, with [[m2:kullback-leibler-divergence|the KL divergence]] and [[m2:differential-entropy|differential entropy]].
Papamakarios et al. compare the two directions: fitting a flow-based model by minimizing the forward KL divergence is maximum likelihood estimation and is well-suited for situations in which we have samples from the target distribution; the reverse KL divergence is suitable when we can evaluate the target density and is often used for variational inference. {#c-review-kl}
Papamakarios, Pavlakou and Murray connect the two kinds of autoregressive flow through this: training a MAF with maximum likelihood corresponds to fitting an implicit IAF to the base density with stochastic variational inference. {#c-maf-iaf}

### Discrete data and bits per dimension

Images are integers, and a density on integers is not a density on $\mathbb{R}^D$: a continuous model can put unbounded density on the grid points and drive the loss to $-\infty$.
Flow-based models are defined over continuous random variables, so to use them with discrete data the data are often dequantized by adding continuous noise, fixed, such as uniform in $[0, 1]$, or learned, as in variational dequantization. {#c-review-dequant}
With $L$ levels per coordinate, the dequantized point is $\mathbf{y} = (\mathbf{k} + \mathbf{u}) / L$ with $\mathbf{u} \sim U[0, 1)^D$, and for any density $q$ on $[0, 1)^D$ and the discrete model $P(\mathbf{k})$ that integrates $q$ over the bin of $\mathbf{k}$,

$$ \mathbb{E}_{\mathbf{u}}\Big[\log q\big((\mathbf{k} + \mathbf{u}) / L\big)\Big] - D \log L \le \log P(\mathbf{k}) . $$ {#eq-dequantization}

For integer data in $\{0, \dots, L - 1\}^D$ dequantized as $(\mathbf{k} + \mathbf{u}) / L$ with $\mathbf{u}$ uniform on $[0, 1)^D$, the expected continuous log-likelihood minus $D \log L$ is a lower bound on the log-likelihood of a discrete model, so $(\text{NLL} + D \log L) / (D \log 2)$ bounds the bits per dimension of the discrete data from above. {#c-dequant}
The derivation `dequantization-bound` proves it.
The number reported for image flows is that bound,

$$ \text{bits/dim} = \frac{\text{NLL} + D \log L}{D \log 2} , $$ {#eq-bpd}

with the NLL in nats of the data scaled to $[0, 1)$; a uniform density scores $\log_2 L$, 8 bits per dimension on 8-bit images.
NICE added a uniform noise of $\tfrac{1}{256}$ to the data and rescaled it to $[0, 1]^D$.

### Initialization and normalization

Glow initializes the last convolution of each NN() with zeros, so that each affine coupling layer initially performs an identity function, which the authors found helps training very deep networks. {#c-glow-zero}
The reference does the same with $W_s$, $W_t$ and their biases, so an untrained flow is the identity and its density is the base density.
The scale $\exp(\mathbf{s})$ can blow up when $\mathbf{s}$ is unbounded; the reference bounds the hidden layer with tanh, and Real NVP computed $s$ with a hyperbolic tangent multiplied by a learned scale.
Real NVP used batch normalization and weight normalization in $s$ and $t$ and on the output of whole coupling layers, with the batch normalization's own log-determinant added to eq-logdet; Glow replaced it with actnorm.

## Inductive biases

A flow assumes that the data are the image of a simple base density under a smooth invertible map.
Because the map is a bijection of $\mathbb{R}^D$, the model gives positive density to every point and cannot put the data on a set of lower dimension, so images that lie near a thin manifold need a map with very large and very small stretches, and a flow spends capacity on directions that carry noise.
The layers decide which maps are cheap: a coupling layer conditions half the coordinates on the other half, and the masks decide which dependencies each layer can express, so a checkerboard mask builds in the locality of images and a channel mask the correlation of channels.
An autoregressive layer builds in an order of the coordinates.
Papamakarios, Pavlakou and Murray find that autoregressive models are sensitive to the order of the variables: a MADE with order $(x_1, x_2)$ and Gaussian conditionals fits the density $p(x_1, x_2) = \mathcal{N}(x_2 \mid 0, 4)\, \mathcal{N}(x_1 \mid x_2^2 / 4, 1)$ poorly, while a 5 layer MAF with the same order fits it. {#c-maf-banana}
That density, the banana, is the one the reference learns.

Flows with enough layers can represent any reasonable density.
If $p_x > 0$ on $\mathbb{R}^D$ with differentiable conditional distribution functions, the map $z_i = F_i(x_i \mid \mathbf{x}_{<i})$ of conditional [[m2:cumulative-distribution-function|cumulative distribution functions]] has a lower-triangular Jacobian with determinant $p_x(\mathbf{x})$ and sends $\mathbf{x}$ to the uniform distribution on $(0, 1)^D$, so an autoregressive flow with flexible enough transformers and conditioners can represent any such density. {#c-universal}
The derivation `universality` proves it.
The result needs flexible transformers; for the affine ones of most flows the question is open: expressive flows can be obtained by stacking multiple affine autoregressive layers, but it is unknown whether affine autoregressive flows with multiple layers are universal approximators; affine transformers have been used in NICE, Real NVP, IAF, MAF and Glow. {#c-review-affine-open}
For coupling layers, a composition of $D$ coupling layers is a universal approximator, and it is an open problem whether a universal approximator can be obtained by composing strictly fewer than $O(D)$ coupling layers. {#c-review-coupling-open}

## Evidence

NICE, a stack of four coupling layers with a diagonal positive scaling, obtained a test log-likelihood of 1980.50 on MNIST, 5514.71 on TFD, 11496.55 for SVHN and 5371.78 for CIFAR-10, on dequantized data with a standard logistic prior, or a standard normal one for TFD. {#c-nice-results}
These are log-densities of continuous data and depend on the rescaling, so they are not comparable with the bits per dimension of later papers.
Rezende and Mohamed used planar flows as approximate posteriors of a deep latent Gaussian model on binarized MNIST.
On binarized MNIST a DLGM with a diagonal covariance posterior reached a bound of 89.9 on the negative log-probability, and with planar normalizing flows of length $k = 10$, 20, 40 and 80 the bound fell to 87.5, 86.5, 85.7 and 85.1, against 88.6, 87.9, 87.3 and 87.2 with NICE; increasing the flow length systematically improves the bound. {#c-planar-results}

Real NVP reached 3.49 bits per dimension on CIFAR-10, 4.28 on Imagenet $32 \times 32$ and 3.98 on Imagenet $64 \times 64$, against 3.00, 3.86 and 3.63 for PixelRNN; the authors write that the number of bits per dimension, while not improving over the Pixel RNN baseline, is competitive with other generative methods. {#c-realnvp-results}
Glow reached 3.35 bits per dimension on CIFAR-10, 4.09 on ImageNet $32 \times 32$ and 3.81 on ImageNet $64 \times 64$, against 3.49, 4.28 and 3.98 for RealNVP. {#c-glow-results}
On tabular data, the average test log likelihood in nats on POWER, GAS, HEPMASS, MINIBOONE and BSDS300 was 0.14, 9.07, $-17.70$, $-11.75$ and 155.69 for MAF (5), against $-0.02$, 4.78, $-19.62$, $-13.55$ and 152.97 for Real NVP (5) and $-7.74$, $-3.58$, $-27.93$, $-37.24$ and 96.67 for a Gaussian. {#c-maf-results}
In the same table MADE MoG, an autoregressive model with mixture-of-Gaussian conditionals and no stack, scored 0.40 on POWER and $-15.15$ on HEPMASS, above every MAF, so stacking affine layers is not always better than one layer with flexible conditionals.

Sampling is fast where coupling layers are used.
Generating a $256 \times 256$ image at batch size 1 with Glow takes about 130ms on a single 1080 Ti and about 550ms on a K80, while synthesis from MAF and from autoregressive models is non-parallelizable. {#c-glow-speed}
Sampling from a reduced-temperature model often results in higher-quality samples; for additive coupling layers Glow achieves this by multiplying the standard deviation of $p(\mathbf{z})$ by a factor of $T$. {#c-glow-temperature}
For a flow whose Jacobian determinant is constant, such as one of additive coupling layers, scaling the standard deviation of a normal base by $T$ gives the density $p(\mathbf{x})^{1/T^2}$ up to normalization; for affine layers it is only an approximation of that.
The likelihoods of image flows trailed those of autoregressive models such as PixelRNN throughout, and their samples trailed those of GANs and later diffusion models in visual quality.

## Limitations

The code has the dimension of the data, so a flow cannot compress, and every layer works on a tensor as large as the input, which makes image flows large: Glow used $K = 32$ steps per level and $L = 6$ levels for its $256 \times 256$ faces.
The architecture is constrained to cheap Jacobians and inverses, and the affine layers are known to need many layers for complex densities; whether a fixed number of affine layers is enough is open.
Maximum likelihood rewards covering all of the data, and samples suffer for it: Dinh et al. note that maximum likelihood values diversity over sample quality in a limited capacity setting, so Real NVP sometimes outputs highly improbable samples, especially on CelebA; unlike autoregressive models, sampling from it is parallelized over input dimensions. {#c-realnvp-samples}
A diffeomorphism preserves topology, so a flow from a unimodal normal to a density with several separated modes needs regions of very large Jacobian, and a flow cannot model data with holes or on a lower-dimensional manifold exactly.
Likelihoods of flows have been found to be poor detectors of unusual inputs: models trained on one image dataset have assigned higher likelihood to another, simpler one.

## Variants and descendants

- **Additive and affine coupling flows**: NICE (Dinh et al., 2014), Real NVP (Dinh et al., 2016) and Glow (Kingma and Dhariwal, 2018), with eq-coupling as their layer.
- **Planar and radial flows** (Rezende and Mohamed, 2015) and **Sylvester flows** (van den Berg et al., 2018), whose Jacobians are low-rank updates of the identity, used as variational posteriors.
- **Autoregressive flows**: IAF (Kingma et al., 2016) for posteriors and fast sampling, MAF (Papamakarios et al., 2017) for density estimation, and neural autoregressive flows with monotone networks as transformers.
- **Spline flows** (Durkan et al., 2019) replace the affine transformer of a coupling or autoregressive layer by a monotone rational-quadratic spline, which is more expressive at the same cost.
- **Residual flows** (i-ResNet, Behrmann et al., 2019, and Chen et al., 2019) make a residual block invertible by bounding its Lipschitz constant and estimate the log-determinant by a power series of traces.
- **Continuous-time flows**: a continuous-time flow is constructed by defining an ordinary differential equation that parameterizes the flow's infinitesimal dynamics, $d\mathbf{z}_t / dt = g(t, \mathbf{z}_t)$, and integrating to find the corresponding transformation. {#c-review-continuous} The [[neural-ordinary-differential-equation]] of Chen et al. (2018) and FFJORD (Grathwohl et al., 2019) compute its log-density with the trace of the Jacobian instead of the determinant, and flow matching trains the vector field by regression instead of maximum likelihood.
- **Discrete flows** and flows on manifolds, such as spheres and tori, carry the idea to other spaces.
- **Hybrids**: flows as priors or posteriors of a [[variational-autoencoder]], dequantization flows, and flows as invertible encoders in place of an [[autoencoder]].

## Notable models

NICE was the first deep network trained by exact maximum likelihood through a change of variables with coupling layers, and Real NVP's affine layer and multi-scale architecture became the template of image flows.
Glow generated faces at $256 \times 256$ with a learned latent space in which attributes such as smiling could be added by moving along a direction, and its step of flow was reused in speech: WaveGlow synthesized audio with Glow's layers.
Parallel WaveNet distilled an autoregressive WaveNet into an inverse autoregressive flow for fast speech synthesis.
In science, flows are used where an exact density matters: Boltzmann generators sample molecular configurations with importance weights, lattice field theory uses flows to sample gauge fields, and simulation-based inference fits posteriors with neural spline flows.

## Implementations

The NumPy reference `impl/numpy/flow.py` implements eq-coupling with alternating masks, eq-logdet, eq-change-of-variables with a standard normal base, the inverse, and the mean negative log-likelihood with every backward pass written by hand, in the parameters and in the input.
It also has the planar flow with eq-planar, one masked autoregressive layer with eq-autoregressive in both directions, dequantization and eq-bpd, sampling at a temperature, Adam, a training loop, and the counts eq-params and eq-flops.
The PyTorch version `impl/pytorch/flow.py` is a `Flow` of `Coupling` modules with the same parameters and autograd for the backward pass; `test_gradcheck` runs `torch.autograd.gradcheck` through the log-likelihood.
The JAX version `impl/jax/flow.py` is a set of pure functions of a parameter dictionary; `test_logdet_against_jacfwd_and_inverse_under_jit` compares the log-determinant with `jax.jacfwd` under `jax.jit`, and the autoregressive sampler loops with `jax.lax.fori_loop`.
All three load the shared fixture of `fixtures/`: the example config's parameters drawn from a normal distribution, so that no layer is the identity, 4 data points, and the codes, log-determinants, log-densities, inverses, loss and gradients, with the planar and autoregressive layers on the same points.

### Worked example

Take $D = 2$ and one coupling layer that keeps $x_1$, with $\mathbf{s} = (0, \log 2)$ and $\mathbf{t} = (0, 1)$, constant in $x_1$.
The point $\mathbf{x} = (1, 2)$ maps to $\mathbf{z} = (1, 2 \cdot 2 + 1) = (1, 5)$, and by eq-coupling-jacobian the log-determinant is $\log 2 = 0.6931$: the layer doubles lengths along $x_2$.
Under the standard normal base, $\log p_z(1, 5) = -\tfrac12(1 + 25) - \log 2\pi = -13 - 1.8379 = -14.8379$.
By eq-change-of-variables, $\log p_x(1, 2) = -14.8379 + 0.6931 = -14.1447$.
The inverse gives back $x_2 = (5 - 1) / 2 = 2$.
The test `test_worked_example` checks these numbers.
For bits per dimension, 3.49 on CIFAR-10, with $D = 32 \cdot 32 \cdot 3 = 3072$, is by eq-bpd an NLL of $3.49 \cdot 3072 \log 2 - 3072 \log 256 = -9603.4$ nats on data scaled to $[0, 1)$, which `test_bits_per_dim` checks.

### Pitfalls

> [!pitfall]
> **The direction of the log-determinant.** eq-change-of-variables adds the log-determinant of the map from the data to the base space; the map from the base space to the data enters with a minus sign. Code that samples with $g = f^{-1}$ and evaluates densities by inverting it must subtract $\log |\det \partial g / \partial \mathbf{z}|$. `test_logdet_is_the_log_abs_det_of_the_jacobian` compares the reference's log-determinant with the finite-difference Jacobian, and `test_density_integrates_to_one` checks that the density of a random flow in the plane integrates to 1 on a grid to within $10^{-8}$, which fails with the wrong sign.

> [!pitfall]
> **Masks that do not alternate.** If every layer keeps the same coordinates, those coordinates are never transformed and the flow is a model of the rest conditioned on them, with the kept ones following the base density. `test_masks_alternate` checks that consecutive layers transform complementary sets of coordinates.

> [!pitfall]
> **An unbounded scale.** $\exp(\mathbf{s})$ overflows, and the log-likelihood jumps, when $\mathbf{s}$ grows; bound it, as the reference does with a tanh hidden layer and Real NVP with a tanh on the scale, and start from the identity. `test_zero_init_is_the_identity` checks that the initial flow maps every point to itself with log-determinant 0.

> [!pitfall]
> **Continuous likelihoods of discrete data.** Training on integer pixels without dequantization lets the density grow without bound on the grid points, and the loss runs to $-\infty$ while the model learns nothing useful. Add uniform noise, scale to $[0, 1)$, and report eq-bpd with the $D \log L$ term. `test_dequantized_likelihood_bounds_the_discrete_one` checks eq-dequantization on data with 4 levels.

> [!pitfall]
> **Planar flows that are not invertible.** Without the fix of eq-planar, a planar layer with $\mathbf{w}^\top \mathbf{u} < -1$ folds the space, the determinant changes sign, and the log-density is wrong. `test_planar_logdet_and_the_invertibility_fix` checks the determinant against the Jacobian and that the fixed map is monotone along $\mathbf{w}$, and `test_u_hat_sets_w_dot_u_to_m` that $\mathbf{w}^\top \hat{\mathbf{u}} = m(\mathbf{w}^\top \mathbf{u})$.

> [!pitfall]
> **Sampling a masked autoregressive flow in one pass.** The inverse of eq-autoregressive needs $D$ sequential passes, each fixing one more coordinate; fewer passes return a point that is not the inverse. `test_autoregressive_density_in_one_pass_sampling_in_d` checks that $D - 1$ passes are wrong, that $D$ passes invert the density pass exactly, and that the Jacobian is lower triangular.

## Experiments

This page has no experiment yet.
Some tests are small experiments.
`test_learns_a_banana` trains the NumPy flow with $L = 4$ coupling layers of $H = 32$ hidden units for 2000 Adam steps on the banana density of Papamakarios et al.; its held-out negative log-likelihood ends within 0.05 nats of the entropy of the density, $\log(2\pi e) + \tfrac12 \log 4 = 3.531$, the floor of the loss by c-mle-kl, while the Gaussian with the data's mean and covariance scores more than 0.4 nats worse.
Its samples have $\mathbb{E}[x_1]$ within 0.1 of the true value 1, and samples at temperature 0.5 have less than 0.7 times the spread in $x_2$.
In PyTorch the same test trains the module for 1000 steps and lands within 0.1 nats of the entropy.
The checks of the seven derivations compare eq-change-of-variables with grid integrals, eq-coupling-jacobian with numerical Jacobians, the planar determinant and the fix with numerical Jacobians, the KL identities with closed forms for Gaussians and Monte Carlo estimates for a mixture, the dequantization bound with exact bin integrals, the conditional-CDF map on the banana density, and the counts with the arrays and an instrumented forward pass.

## History

Dinh, Krueger and Bengio posted NICE as arXiv 1410.8516, and it was accepted as a workshop contribution at ICLR 2015. {#c-nice-history}
The change of variables was older: Rippel and Adams (2013) had trained deep invertible maps to a factorized density, Chen and Gopinath's Gaussianization (2000) had iterated rotations and marginal transforms towards a normal, and Tabak and colleagues described density estimation by flows of maps around 2010, the source of the name.
Rezende and Mohamed (2015) brought the term normalizing flow into deep learning as a family of variational posteriors, and Kingma et al. (2016) added inverse autoregressive flows.
Real NVP (2016) and Glow (2018) made flows competitive generative models of images, MAF (2017) of tabular data, and WaveGlow and Parallel WaveNet (2018) of speech.
Neural ODEs and FFJORD (2018) introduced continuous-time flows, and spline flows and residual flows (2019) made the layers more expressive.
The survey of Papamakarios et al. (2021) unified the field.
Since about 2021, diffusion models and flow matching, which trains a continuous-time flow without simulating it, have replaced likelihood-trained flows for most image generation, while flows remain the standard where exact densities are needed, in variational inference, simulation-based inference and physics.

## References

- L. Dinh, D. Krueger and Y. Bengio. NICE: Non-linear Independent Components Estimation. arXiv 1410.8516, 2014; ICLR 2015 workshop.
- D. J. Rezende and S. Mohamed. Variational Inference with Normalizing Flows. ICML 2015, JMLR W&CP 37.
- L. Dinh, J. Sohl-Dickstein and S. Bengio. Density estimation using Real NVP. arXiv 1605.08803, 2016; ICLR 2017.
- D. P. Kingma, T. Salimans, R. Jozefowicz, X. Chen, I. Sutskever and M. Welling. Improved Variational Inference with Inverse Autoregressive Flow. NIPS 2016.
- G. Papamakarios, T. Pavlakou and I. Murray. Masked Autoregressive Flow for Density Estimation. arXiv 1705.07057, 2017; NIPS 2017.
- D. P. Kingma and P. Dhariwal. Glow: Generative Flow with Invertible 1x1 Convolutions. arXiv 1807.03039, 2018; NeurIPS 2018.
- C. Durkan, A. Bekasov, I. Murray and G. Papamakarios. Neural Spline Flows. NeurIPS 2019.
- W. Grathwohl, R. T. Q. Chen, J. Bettencourt, I. Sutskever and D. Duvenaud. FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models. ICLR 2019.
- G. Papamakarios, E. Nalisnick, D. J. Rezende, S. Mohamed and B. Lakshminarayanan. Normalizing Flows for Probabilistic Modeling and Inference. Journal of Machine Learning Research 22, 2021.
