---
tag: 35WSHD
kind: architecture
slug: autoregressive-model
name: Autoregressive model
aliases: [neural autoregressive model, deep autoregressive model, autoregressive generative model, neural autoregressive density estimator, NADE, MADE, masked autoencoder for distribution estimation, PixelRNN, PixelCNN, Gated PixelCNN, WaveNet, fully visible belief network, FVSBN]
area: generative-models/autoregressive-models
level: core
summary: >-
  A generative model that writes the probability of a vector as a product of conditionals, each variable given those before it, with a neural network computing every conditional.
uses: [conditional-probability@m2, joint-probability-distribution@m2, marginal-distribution@m2, independence-of-random-variables@m2, bernoulli-distribution@m2, logistic-regression@m2, cross-entropy@m2, shannon-entropy@m2, kullback-leibler-divergence@m2, gibbs-inequality@m2, expected-value@m2, permutation-matrix@m2, quantization@m2, matrix-multiplication@m2, hadamard-product@m2, maximum-likelihood-estimation@al, cross-entropy-loss@al, backpropagation@al, ancestral-sampling@al, teacher-forcing@al]
related_to: [transformer, recurrent-neural-network, long-short-term-memory, convolutional-neural-network, residual-connection, autoencoder, variational-autoencoder, normalizing-flow, diffusion-model, generative-adversarial-network, flow-matching, speculative-decoding, adam]
history:
  year: 2011
  people: [Hugo Larochelle, Iain Murray, Mathieu Germain, Karol Gregor, Aäron van den Oord, Nal Kalchbrenner, Koray Kavukcuoglu, Sander Dieleman]
  note: "Frey's fully visible sigmoid belief network (1998) and Bengio and Bengio's neural network over discrete variables (2000) wrote a joint distribution as a chain of learned conditionals. Larochelle and Murray made the conditionals share one hidden layer in NADE (2011), Germain and colleagues computed them all in one masked pass in MADE (2015), and van den Oord and colleagues scaled the idea to images with PixelRNN and PixelCNN and to raw audio with WaveNet (2016)."
prereqs: [autoencoder, convolutional-neural-network, conditional-probability@m2, cross-entropy@m2, maximum-likelihood-estimation@al]
sources:
  - {paper: "Germain, Gregor, Murray and Larochelle (2015)", title: "MADE: Masked Autoencoder for Distribution Estimation", locator: "ICML 2015; arXiv 1502.03509v2, pp. 1 to 8", note: "autoencoders as density estimators, the degree masks, direct connections, order-agnostic training, connectivity-agnostic training, timing and the binarized MNIST results"}
  - {paper: "Larochelle and Murray (2011)", title: "The Neural Autoregressive Distribution Estimator", locator: "AISTATS 2011, PMLR 15, pp. 3, 4, 6 of the proceedings PDF", note: "the tied hidden layer, its O(HD) running sum, untied output weights, the effect of the ordering and the MNIST results"}
  - {paper: "van den Oord, Kalchbrenner and Kavukcuoglu (2016)", title: "Pixel Recurrent Neural Networks", locator: "ICML 2016; arXiv 1601.06759v3, pp. 2 to 8", note: "the raster-scan factorization, colour channels, discrete 256-way softmax outputs, masks A and B, PixelCNN and the image results"}
  - {paper: "van den Oord, Kalchbrenner, Vinyals, Espeholt, Graves and Kavukcuoglu (2016)", title: "Conditional Image Generation with PixelCNN Decoders", locator: "arXiv 1606.05328v2, pp. 3, 5", note: "the blind spot, the vertical and horizontal stacks, the gated activation unit and class-conditional models"}
  - {paper: "van den Oord, Dieleman, Zen, Simonyan, Vinyals, Graves, Kalchbrenner, Senior and Kavukcuoglu (2016)", title: "WaveNet: A Generative Model for Raw Audio", locator: "arXiv 1609.03499v2, pp. 2 to 6", note: "causal and dilated convolutions, the receptive field, mu-law quantization, gated units and the listening tests"}
family: autoregressive-models
inputs: "for training and evaluation a batch of D-dimensional data vectors as a (B, D) array of binary values, or of integer codes for a softmax model; for sampling only the number of vectors B"
outputs: "for training and evaluation the logits of all D conditionals as a (B, D) array, the log-likelihoods as a (B,) array and a scalar loss; for sampling B vectors as a (B, D) array, drawn in D passes"
components: []
inductive_biases: ["the joint distribution is factored in one fixed ordering of the variables, so every variable is predicted from those before it and the order is part of the model", "every conditional is a normalized distribution, so the product is normalized for any parameters and the likelihood is exact", "the conditionals share parameters, through one network evaluated at every position, so a pattern learned at one position transfers to the others", "convolutional versions see a bounded, local past: each conditional depends only on the receptive field before it"]
params: "2DH + H + D + D^2 for the MADE of this page with D binary inputs, one masked hidden layer of H units and a masked direct connection, masked entries included"
flops_per_example: "2(2DH + D^2) for one pass, which gives all D conditionals, counting a multiply-add as 2 FLOPs and only the matrix products; drawing one sample costs D passes"
---

An autoregressive model is a generative model that predicts a vector one variable at a time.
It fixes an ordering of the $D$ variables and writes the probability of the vector as a product of $D$ conditionals, each giving the probability of one variable given all the variables before it.
A neural network computes the conditionals, and training maximizes the exact log-likelihood of the data.
Sampling runs the chain: draw the first variable, feed it back, draw the second, and so on.
The model has no latent variables and no partition function, and every probability it assigns is exact.
NADE, MADE, PixelCNN and WaveNet are autoregressive models over binary vectors, images and audio. The language models that predict the next token are autoregressive models over text.

## In plain words

Guessing a whole picture at once is hard, but guessing one pixel given all the pixels above and to the left of it is a smaller problem.
An autoregressive model breaks the hard problem into a chain of small ones.
To score a picture, it asks: how likely is the first pixel, how likely is the second given the first, how likely is the third given the first two, and so on, and multiplies the answers.
To draw a new picture, it guesses the first pixel, writes it down, guesses the second given what it wrote, and keeps going until the picture is full.
Each guess is a probability over a small set of values, which is easy to normalize, so the product is a proper probability of the whole picture.
The price is speed: writing a picture of a thousand pixels takes a thousand guesses in a row.

## Overview

Let $\mathbf{x} = (x_1, \ldots, x_D)$ and let $o$ be an ordering, a permutation of $1, \ldots, D$.
The chain rule of probability gives, for any distribution,

$$ p(\mathbf{x}) = \prod_{d=1}^{D} p\big(x_{o_d} \mid x_{o_1}, \ldots, x_{o_{d-1}}\big) . $$ {#eq-chain}

An autoregressive model chooses the conditionals instead of the joint distribution.
If each conditional $p(x_d \mid \mathbf{x}_{<d})$ is a probability distribution over the values of $x_d$ for every value of $\mathbf{x}_{<d}$, the product of the conditionals in any fixed ordering is a probability distribution over $\mathbf{x}$, and drawing $x_1$, then $x_2$ given $x_1$, and so on, draws exactly from it. No partition function is needed. {#c-normalization}
The derivation `autoregressive-normalization` proves it, and `test_probabilities_sum_to_one` sums a model over all $2^D$ binary vectors.
The drawing procedure is [[al:ancestral-sampling|ancestral sampling]].
With no restriction on the conditionals, eq-chain can represent every distribution. Restrictions come from the network: the time-series [[m2:autoregressive-model|autoregressive model]] of statistics makes each conditional a linear function of the last $p$ values, while a neural autoregressive model lets a network compute it from the whole past.

Training is [[al:maximum-likelihood-estimation|maximum likelihood]]. The loss is the negative log-likelihood, which splits into one term per conditional:

$$ \mathcal{L}(\theta) = -\mathbb{E}_{\mathbf{x} \sim q} \log p_\theta(\mathbf{x}) = -\sum_{d=1}^{D} \mathbb{E}_{\mathbf{x} \sim q} \log p_\theta(x_d \mid \mathbf{x}_{<d}) . $$ {#eq-nll}

Each term is the [[al:cross-entropy-loss|cross-entropy loss]] of a classifier that predicts $x_d$ from the true values before it, the setting called [[al:teacher-forcing|teacher forcing]] in sequence models.
In terms of the data distribution $q$,

$$ -\mathbb{E}_{q} \log p(\mathbf{x}) = H(q) + \mathrm{KL}(q \,\|\, p) . $$ {#eq-kl}

The expected negative log-likelihood of a model $p$ under the data distribution $q$ equals the entropy of $q$ plus the KL divergence from $q$ to $p$, so minimizing it minimizes $\mathrm{KL}(q \,\|\, p)$, which is zero only at $p = q$. The log of the mean probability over an ensemble of $R$ mask sets is at least the mean of their log probabilities. {#c-kl}
The derivation `likelihood-and-kl` proves both parts with [[m2:gibbs-inequality|Gibbs' inequality]] and Jensen's inequality.
Results are reported as the negative log-likelihood in nats for binary images, or in bits per dimension, $-\log_2 p(\mathbf{x}) / D$, for images and audio.

## Components

The notation has vectors as rows. A layer maps $\mathbf{h}$ to $\mathbf{h} W + \mathbf{b}$, with $W$ of shape $(n_\text{in}, n_\text{out})$, and $\odot$ is the elementwise product.
The models below differ in how one network produces all $D$ conditionals while keeping conditional $d$ blind to $x_d$ and to everything after it.

### Autoencoders that are not density models

An [[autoencoder]] maps $\mathbf{x}$ to a reconstruction, and with binary data its cross-entropy loss looks like a log-likelihood.
If the hidden layer of an ordinary autoencoder is at least as large as the input, hidden units can each learn to copy a single input dimension and reconstruct all inputs perfectly, so its cross-entropy loss is not a proper log-likelihood: the implied distribution could be 1 for any $\mathbf{x}$ and not be properly normalized. {#c-ae-not-likelihood}
Germain and colleagues mask the autoencoder's parameters to respect autoregressive constraints, so that each input is reconstructed only from previous inputs in a given ordering. Constrained this way, the autoencoder outputs can be interpreted as a set of conditional probabilities, and their product as the full joint probability. {#c-made-def}

### MADE

The masked autoencoder for distribution estimation, MADE, with one hidden layer of $H$ units and a direct connection from the inputs to the outputs, computes the logits

$$ \mathbf{h} = \mathrm{relu}\big(\mathbf{x}(W \odot M^W) + \mathbf{b}\big), \qquad \mathbf{a} = \mathbf{h}(V \odot M^V) + \mathbf{x}(A \odot M^A) + \mathbf{c}, \qquad p(x_d = 1 \mid \mathbf{x}_{<d}) = \sigma(a_d) . $$ {#eq-made}

Each conditional is a [[m2:bernoulli-distribution|Bernoulli distribution]] with a [[m2:logistic-regression|logistic]] link, and all $D$ of them come out of one pass.
The masks are built from integer degrees.
MADE assigns each hidden unit an integer $m(k)$ between 1 and $D - 1$, the maximum number of input units it can be connected to. $m(k) = D$ is disallowed because the unit would depend on all inputs and could not be used in any conditional, and $m(k) = 0$ because it would be constant. In the experiments $m(k)$ was sampled from a uniform discrete distribution on 1 to $D - 1$, independently for each hidden unit. {#c-made-degrees}
The input degrees $m^0$ are the ordering: $m^0(i)$ is the position of $x_i$ in it. Then

$$ M^W_{ik} = \mathbb{1}\big[m^1(k) \ge m^0(i)\big], \qquad M^V_{kd} = \mathbb{1}\big[m^0(d) > m^1(k)\big], \qquad M^A_{id} = \mathbb{1}\big[m^0(d) > m^0(i)\big] . $$ {#eq-made-masks}

With input degrees $m_0$ a permutation of $1..D$, hidden degrees in $1..D-1$, hidden masks $\mathbb{1}[m_l(k') \ge m_{l-1}(k)]$, output mask $\mathbb{1}[m_0(d) > m_L(k)]$ and direct mask $\mathbb{1}[m_0(d) > m_0(i)]$, every path from input $i$ to output $d$ passes through units whose degrees never decrease and must then strictly increase. So output $d$ depends only on the inputs that come before $d$ in the ordering, and the outputs define a valid autoregressive model for any values of the weights. {#c-made-masks}
The derivation `made-masks` proves it, and `test_masks_make_the_jacobian_triangular_in_the_ordering` checks that the Jacobian of the logits in the inputs is strictly triangular in the ordering, for zero to two hidden layers.
The masks multiply the weights and do not change the arithmetic, so a masked layer is an ordinary [[m2:matrix-multiplication|matrix product]] with a [[m2:hadamard-product|Hadamard product]] in front of it.
Earlier autoregressive neural networks found it advantageous to use direct connections between the input and output layers. In MADE their mask simply needs to be a strictly lower diagonal matrix filled otherwise with ones, and Germain and colleagues used such direct connections in their experiments. {#c-made-direct}

### Orderings and mask ensembles

The ordering is a choice, and a MADE can learn many at once.
Order-agnostic training samples an ordering before each stochastic or minibatch update. It lets missing values be imputed by invoking an ordering where observed dimensions all come before unobserved ones, and it gives an ensemble on the fly, by sampling a set of orderings, computing the probability of $\mathbf{x}$ under each and averaging. {#c-order-agnostic}
Germain and colleagues found that sampling masks for every example could over-regularize MADE and provoke underfitting. They therefore also sampled from a finite list of masks, cycling through it during training and averaging the probabilities of all masks in the list at test time. {#c-overregularize}
The ensemble of $R$ mask sets is the mixture

$$ p_\text{ens}(\mathbf{x}) = \frac{1}{R} \sum_{r=1}^{R} p_r(\mathbf{x}), \qquad \log p_\text{ens}(\mathbf{x}) \ge \frac{1}{R} \sum_{r=1}^{R} \log p_r(\mathbf{x}) , $$ {#eq-ensemble}

a normalized distribution whose log-likelihood is at least the average of its members', by the second part of c-kl. `test_mask_ensemble_is_normalized_and_beats_the_mean_log_prob` checks both.

### NADE

The neural autoregressive distribution estimator came before MADE and computes the same kind of conditionals in a different way.
Larochelle and Murray model high-dimensional vectors of discrete variables by decomposing the joint distribution into tractable conditional distributions and modeling each conditional with a non-linear function similar to a conditional of an RBM, which avoids the RBM's intractable partition function. The model can also be read as an autoencoder wired so that its output assigns valid probabilities. {#c-nade-def}
Conditional $d$ has its own hidden layer, computed from the inputs before $d$ with a shared matrix $W$ of shape $(D, H)$:

$$ \mathbf{s}_1 = \mathbf{c}, \qquad \mathbf{s}_{d+1} = \mathbf{s}_d + x_d W_{d,:}, \qquad p(x_d = 1 \mid \mathbf{x}_{<d}) = \sigma\big(b_d + \sigma(\mathbf{s}_d)\, V_{:,d}\big) . $$ {#eq-nade}

The tied connections of NADE let the conditionals share computation: the activations of the $i$-th and $(i+1)$-th hidden layers differ by $W_{(\cdot,i+1)} v_{i+1}$, computable in $O(H)$, so the complete cost of computing $p(\mathbf{v})$ is $O(HD)$ instead of the cost, quadratic in $D$, of a naive procedure that does not take advantage of the weight sharing. {#c-nade-sharing}
In NADE the hidden pre-activation of conditional $i + 1$ equals that of conditional $i$ plus $W_{(\cdot,i)} v_i$, so all $D$ hidden vectors follow from one running sum in $O(HD)$ operations, the same values that $D$ separate passes compute in $O(HD^2)$. {#c-nade-cost}
The derivation `nade-incremental` proves it, and `test_nade_recursion_matches_the_naive_passes_and_normalizes` compares the running sum with $D$ separate passes.
Larochelle and Murray found that untying the connections going in and out of the hidden units, learning a separate output matrix $V$, gave better distribution estimation and did not make NADE slower. {#c-nade-untied}
The running sum is sequential in $d$, which is why MADE, one masked pass, runs faster on parallel hardware.

### Causal and dilated convolutions

For a sequence, such as audio, one network evaluated at every position gives every conditional.
WaveNet factorises the joint probability of a waveform $\mathbf{x} = (x_1, \ldots, x_T)$ as the product of the conditional probabilities $p(x_t \mid x_1, \ldots, x_{t-1})$, so each audio sample is conditioned on the samples at all previous timesteps, and is built from causal convolutions, which keep the prediction at timestep $t$ from depending on any future timestep. {#c-wavenet-def}
A causal convolution with kernel size $k$ and dilation $d$ maps a sequence of vectors $\mathbf{h}_t$ to

$$ \mathbf{y}_t = \sum_{j=0}^{k-1} \mathbf{h}_{t - (k-1-j)d}\, W_j , $$ {#eq-causal-conv}

with zeros before the start, so $\mathbf{y}_t$ uses no input after $t$.
The input is the sequence shifted by one step, so the output at $t$ sees $x_{t-1}$ and earlier but not $x_t$.
A stack of $L$ such layers sees

$$ R = 1 + (k - 1) \sum_{l=1}^{L} d_l $$ {#eq-receptive-field}

past values.
A stack of causal convolutions with kernel size $k$ and dilations $d_1, \ldots, d_L$, applied to the input shifted by one step, makes the prediction at time $t$ depend on exactly the $1 + (k - 1)(d_1 + \cdots + d_L)$ values before $t$. For $k = 2$ and dilations $1, 2, \ldots, 512$ this is 1024. {#c-receptive-field}
The derivation `dilated-receptive-field` proves it, together with the condition that the field has no gaps: $d_1 = 1$ and each $d_l$ at most $1 + (k-1)$ times the sum of the earlier dilations. `test_receptive_field_of_the_dilated_causal_stack` perturbs one input at a time and finds which outputs move.
In WaveNet the dilation is doubled for every layer up to a limit and then repeated, as $1, 2, 4, \ldots, 512, 1, 2, 4, \ldots, 512$. Each $1, 2, 4, \ldots, 512$ block has a receptive field of size 1024, while causal convolutions without dilation need many layers or large filters, a receptive field of the number of layers plus the filter length minus 1. {#c-dilations}
The field grows exponentially with depth when the dilations double, and linearly when they do not.

### Gated units and residual stacks

Between the convolutions WaveNet and Gated PixelCNN use a gate instead of a ReLU:

$$ \mathbf{z} = \tanh(W_f * \mathbf{h}) \odot \sigma(W_g * \mathbf{h}) . $$ {#eq-gated}

Gated PixelCNN replaces the rectified linear units between the masked convolutions with the gated activation unit $\mathbf{y} = \tanh(W_{k,f} * \mathbf{x}) \odot \sigma(W_{k,g} * \mathbf{x})$, elementwise, where $\sigma$ is the sigmoid and $*$ the convolution. {#c-gated-unit}
WaveNet uses the gated activation unit of the gated PixelCNN, and in initial experiments this non-linearity worked significantly better than the rectified linear activation function for modeling audio signals. {#c-gated-audio}
Each layer adds $\mathbf{z} W_r$ to its input through a [[residual-connection]] and sends $\mathbf{z} W_s$ to a sum of skip connections. A ReLU and an output layer turn the skip sum into logits.

### Discrete outputs and mu-law

Images and audio are quantized, and a softmax over the levels needs no assumption about the shape of the conditional.
Each pixel is conditioned colour by colour, red, then green given red, then blue given both, and each channel variable takes one of 256 distinct values with a softmax. During training and evaluation the distributions are computed in parallel, while generation is sequential. {#c-pixel-discrete}
For the Row LSTM model with a softmax output distribution, van den Oord and colleagues obtained 3.06 bits/dim on the CIFAR-10 validation set, against 3.22 bits/dim for the same model with a mixture of conditional Gaussian scale mixtures. {#c-softmax-better}
Audio has more levels, so WaveNet compresses the range before [[m2:quantization|quantizing]] it:

$$ f(x) = \operatorname{sign}(x) \frac{\ln(1 + \mu |x|)}{\ln(1 + \mu)}, \qquad -1 < x < 1 . $$ {#eq-mu-law}

Raw audio stored as 16-bit integers would need a softmax over 65,536 values per timestep, so WaveNet applies the mu-law companding transformation $f(x) = \operatorname{sign}(x) \ln(1 + \mu |x|) / \ln(1 + \mu)$, with $\mu = 255$, and quantizes the result to 256 values. {#c-mu-law}
The transformation spends more levels near zero, where most audio samples lie. `test_mu_law_codes` checks the codes of a few values and the round trip.

### Masked convolutions for images

PixelRNN scans an image row by row and pixel by pixel within each row, writing $p(\mathbf{x})$ as the product of the conditional distributions $p(x_i \mid x_1, \ldots, x_{i-1})$ over the pixels, with the parameters of the predictions shared across all pixel positions. {#c-pixelrnn-def}
PixelRNN computes the conditionals with two-dimensional [[long-short-term-memory|LSTM]] layers, the Row LSTM and the Diagonal BiLSTM. PixelCNN replaces them with a [[convolutional-neural-network]] whose kernels are masked so that no pixel sees itself or the pixels after it.
Mask A is applied only to the first convolutional layer and restricts the connections to the neighboring pixels and to the colours of the current pixel already predicted. Mask B is applied to all subsequent input-to-state convolutions and also allows the connection from a colour to itself. {#c-masks-ab}
The masks leave a hole.
A stack of masked convolutions ignores a significant portion of the input image: the blind spot can cover as much as a quarter of the potential receptive field with $3 \times 3$ filters. Gated PixelCNN removes it with two stacks, a horizontal stack conditioned on the current row so far and an unmasked vertical stack on all rows above, whose output feeds the horizontal stack and never the reverse. {#c-blind-spot-attr}
For $L$ layers of $3 \times 3$ masked convolutions, mask A then mask B, an output sees row $i$ above it only out to column $-i$ on the right. So of the $2L(L + 1)$ positions of the causal box of rows $-L..-1$, columns $-L..L$ and the current row to the left, it misses $L(L - 1)/2$, a fraction that rises toward one quarter as $L$ grows. The vertical and horizontal stacks see the whole box. {#c-blind-spot}
The derivation `masked-convolution-blind-spot` proves it. `test_masked_convolutions_have_a_blind_spot` finds the hole by perturbing pixels, and `test_blind_spot_size_and_the_two_stack_fix` counts it for $L = 1, 2, 3, 5$ and 8.

## Shapes and parameters

The shape contract is the MADE of eq-made with $D$ binary inputs, one hidden layer of $H$ units and a direct connection. The example is $D = 4$, $H = 6$.
The hidden layer has $W$ of shape $(D, H)$ and $\mathbf{b}$; the output has $V$ of shape $(H, D)$, $A$ of shape $(D, D)$ and $\mathbf{c}$. So

$$ P = 2DH + H + D + D^2 , $$ {#eq-params}

masked entries included, since they are stored even though the masks zero them.
In the contract this is
{shape:params}
and the example has {shape:params@example} parameters.
The MADE of `test_learns_a_binary_mixture`, `mixture`, has {shape:params@mixture}. A one-hidden-layer MADE of 500 units on binarized MNIST, `binarized_mnist`, has {shape:params@binarized_mnist}.
The masks, the degrees and the ordering are not parameters: they are fixed integers, drawn once or cycled through a list.
The derivation `parameter-and-flop-count` proves the count.

## Computation and cost

One pass on $B$ vectors costs

$$ C = 2B(2DH + D^2) $$ {#eq-flops}

FLOPs in its three products, a multiply-add counted as two.
A MADE with $D$ inputs, one hidden layer of $H$ units and direct connections has $2DH + H + D + D^2$ parameters, masked entries included, and one pass on $B$ vectors costs $2B(2DH + D^2)$ FLOPs counting a multiply-add as 2, which gives all $D$ conditionals. Ancestral sampling takes $D$ passes, and NADE computes the same $D$ conditionals in $O(BDH)$ with its running sum. {#c-params}
In the contract,
{shape:flops_forward}
which is {shape:flops_forward@example} FLOPs for the example and {shape:flops_forward@binarized_mnist} for `binarized_mnist` at its batch of 100.
Training and evaluation are parallel, because the inputs to every conditional are known data. Sampling is sequential, because each variable must be drawn before the next conditional can be computed.
MADE only requires one pass through the autoencoder rather than the $D$ passes required by NADE, and in practice a single-layer MADE was an order of magnitude faster than a one-layer NADE of the same hidden layer size, because NADE's computations cannot be vectorized as efficiently. {#c-made-speed}
On a K20 NVIDIA GPU with a single mask, one training epoch on binarized MNIST took about 14 and 44 seconds for one- and two-hidden-layer MADEs of 8000 units, and 33 and 100 seconds with 32 sampled masks, against about 130 seconds for a 500-unit NADE. {#c-made-timing}
At training time the conditional predictions for all timesteps of WaveNet can be made in parallel because all timesteps of the ground truth are known. When generating, the predictions are sequential, each sample being fed back into the network to predict the next. {#c-wavenet-parallel}
The advantage of parallelization of the PixelCNN over the PixelRNN is only available during training or during evaluation of test images. Image generation is sequential for both kinds of networks, as each sampled pixel needs to be given as input back into the network. {#c-pixelcnn-sequential}
A sample of a 16 kHz second of audio is 16,000 network evaluations in a row; a $32 \times 32$ colour image is 3072.
Generation can reuse work: a convolutional model can cache the activations of earlier positions so that each step computes only the new column, and a [[transformer]] caches its keys and values.

## Training

Training is the same as for a classifier:

1. take a batch of data vectors;
2. compute all conditionals in one pass, with the true values as inputs;
3. take a gradient step on the summed cross-entropy of eq-nll, with [[al:backpropagation|backpropagation]] and an optimizer such as [[adam]].

There is no sampling during training, no encoder, no adversary and no bound.
Order-agnostic training adds one step: draw a new ordering or a new mask set, by c-order-agnostic, before each update.
The NumPy reference writes the backward pass of eq-made by hand and checks it against finite differences in `test_gradients_match_finite_differences`.
Regularization is the usual: weight decay, early stopping on held-out likelihood, and [[dropout]] in the larger models.

## Inductive biases

The ordering is the main assumption.
Eq-chain holds for every ordering, but a network of limited size fits some orderings better than others, and each conditional must predict its variable from the variables before it, however far the relevant ones are.
Raster order for images makes a pixel's context the rows above and the left part of its row; the pixels below and to the right are only reached through what they share with that context.
A dozen NADE models trained on different random orderings had test log-likelihood standard deviations of 0.045, 0.050 and 0.150 on mushrooms, dna and nips-0-12, small next to the uncertainty of a finite test set, so it did not seem necessary to optimize the ordering. {#c-nade-ordering}
Parameter sharing across positions, by one convolution or one recurrent network, assumes that the same rule of prediction holds everywhere, as a convolution assumes for features.
The receptive field of a convolutional model is a hard limit: by eq-receptive-field it sees a fixed number of past values and nothing older.
Unlike a [[variational-autoencoder]] or a [[generative-adversarial-network]], the model has no latent code, so it has no single variable that controls the global layout of a sample; global structure must be decided by the early variables and carried forward.

## Evidence

On binarized MNIST, NADE with 500 hidden units obtained an average test log-likelihood of $-88.86$, against $-86.34$ reported for an RBM with 500 hidden units trained with 25 steps of contrastive divergence and evaluated with an estimated partition function, $-137.64$ for a mixture of 500 Bernoullis and $-97.45$ for FVSBN. {#c-nade-mnist}
On binarized MNIST the two-hidden-layer MADE with 32 masks reached a test negative log-likelihood of 86.64 nats, against 88.40 for a one-layer MADE with one mask, 88.33 for NADE with one hidden layer, 85.10 for EoNADE with two hidden layers and 128 orderings, and about 84.55 for an intractable two-layer DBN. {#c-made-mnist}
On binarized MNIST a Diagonal BiLSTM with 7 layers and $h = 16$ reached a test negative log-likelihood of 79.20 nats, against 81.30 for PixelCNN, 86.64 for MADE with two hidden layers and 32 masks, and 88.33 for NADE. {#c-pixelrnn-mnist}
On the CIFAR-10 test set the PixelCNN reached 3.14 bits/dim, the Row LSTM 3.07 and the Diagonal BiLSTM 3.00, and on ImageNet a PixelRNN reached 3.86 bits/dim at $32 \times 32$ and 3.63 at $64 \times 64$. {#c-pixelrnn-results}
Gated PixelCNN reached 3.03 bits/dim on the CIFAR-10 test set against 3.00 for PixelRNN, and 3.83 and 3.57 bits/dim on $32 \times 32$ and $64 \times 64$ ImageNet against 3.86 and 3.63. It matched the Row LSTM PixelRNN in less than half the training time, 60 hours using 32 GPUs. {#c-gated-results}
In subjective 5-scale mean opinion scores of naturalness, WaveNet conditioned on linguistic features and log F0 scored 4.21 in North American English and 4.08 in Mandarin Chinese, against 3.67 and 3.79 for an LSTM-RNN statistical parametric synthesizer and 3.86 and 3.47 for HMM-driven concatenative synthesis. {#c-wavenet-mos}
The likelihoods of the binarized MNIST results are exact for the autoregressive models and estimated for the RBM and the DBN, whose partition functions are intractable.

## Limitations

Sampling is sequential: $D$ passes for $D$ variables, with no parallelism across them, by c-params.
Training never sees the model's own samples. A conditional trained on true prefixes is applied at sampling time to prefixes the model drew itself, and an early mistake becomes context for every later variable.
The receptive field bounds what a convolutional model can use.
The multi-speaker WaveNet had a receptive field of about 300 milliseconds and could only remember the last 2 to 3 phonemes it produced, which partly explains its lack of long-range coherence. {#c-wavenet-range}
The masks of PixelCNN leave a blind spot, by c-blind-spot, unless the two stacks are used.
The ordering is arbitrary for data without a natural one, such as images, and filling in a missing variable in the middle of the ordering needs every variable after it, unless the model was trained order-agnostically.
A high likelihood does not guarantee good samples, and the converse also fails: likelihood weighs every bit of every variable, including detail the eye ignores.
Conditioning a Gated PixelCNN on a one-hot ImageNet class gives it only about 0.003 bits per pixel of information; van den Oord and colleagues did not observe big differences in log-likelihood, but observed great improvements in the visual quality of the samples. {#c-class-conditional}

## Variants and descendants

- **FVSBN.** The fully visible sigmoid belief network makes each conditional a logistic regression on the variables before it, which is MADE with only the direct connection.
- **NADE** and its relatives: RNADE for real values with mixture-of-Gaussian conditionals, deep and order-agnostic NADE, and EoNADE, an ensemble over orderings.
- **MADE**, with order-agnostic and connectivity-agnostic training. MADE is the building block of the masked autoregressive flow and the inverse autoregressive flow, two kinds of [[normalizing-flow]].
- **PixelRNN and PixelCNN.** Row LSTM and Diagonal BiLSTM layers, or masked convolutions, over images in raster order; Gated PixelCNN with two stacks and conditioning; PixelCNN++ with a discretized logistic mixture output.
- **WaveNet.** Dilated causal convolutions over audio, conditioned on text features or a speaker. Parallel WaveNet distills it into an inverse autoregressive flow that samples in one pass.
- **Recurrent models.** A [[recurrent-neural-network]] or an LSTM carries the whole prefix in its state, so its receptive field is unbounded, at the cost of sequential training.
- **Autoregressive transformers.** A decoder-only [[transformer]] with a causal attention mask computes every conditional of a sequence in one parallel pass and attends to the whole prefix.
- **Discrete latents.** An autoregressive prior over the codes of a vector-quantized [[autoencoder]] models images and audio at a coarser scale.

## Notable models

PixelCNN, Gated PixelCNN and PixelCNN++ were the likelihood baselines for images, and WaveNet the reference model for neural speech synthesis.
The large language models are autoregressive transformers over tokens: each predicts the next token from the ones before it, with the loss of eq-nll.
They sample with temperature, top-$k$ or nucleus truncation, or search with [[al:beam-search|beam search]], and speed up decoding with [[speculative-decoding]], which drafts several tokens and verifies them in one pass.
Autoregressive priors also generate the discrete codes of image and audio tokenizers, and [[diffusion-model|diffusion models]], [[flow-matching|flow matching]] and autoregressive models are combined in several image and video systems.

## Implementations

The NumPy reference `impl/numpy/autoregressive.py` implements:

- the degrees and masks of eq-made-masks, for any number of hidden layers, with and without the direct connection;
- MADE by eq-made, its log-likelihood, a hand-written backward pass of eq-nll, Adam, ancestral sampling and the mask ensemble of eq-ensemble;
- NADE by the running sum of eq-nade, and by $D$ separate passes for comparison;
- a WaveNet-style stack of eq-causal-conv with the gates of eq-gated, residual and skip connections and a softmax output, with its sampler;
- the mu-law code of eq-mu-law and its inverse;
- the masks A and B of PixelCNN, a masked convolution, and the fields of a masked stack and of the two stacks of Gated PixelCNN.

The PyTorch version `impl/pytorch/autoregressive.py` has a `MADE` module built from a masked linear layer, NADE, and a WaveNet stack of left-padded `Conv1d` layers. `test_gradcheck` runs `torch.autograd.gradcheck` through MADE.

The JAX version `impl/jax/autoregressive.py` is a set of pure functions of a parameter dictionary. NADE's running sum is a `jnp.cumsum`, ancestral sampling is a `jax.lax.fori_loop`, and the gradients come from `jax.grad`.

All three load the shared fixture of `fixtures/make.py`. It fixes a MADE with $D = 4$, $H = 6$ and the ordering $(3, 1, 4, 2)$, a NADE, and a WaveNet stack with dilations 1 and 2, and checks the logits, the log-likelihoods, the loss and its gradients, the NADE logits, the WaveNet logits and the mu-law codes.

### Worked example

Take $D = 3$ binary variables, no hidden layer and the natural ordering, so the model is eq-made with only the direct connection: $\mathbf{a} = \mathbf{x}(A \odot M^A) + \mathbf{c}$.
Let $\mathbf{c} = (0, -1, 1)$, $A_{12} = 2$, $A_{13} = -1$, $A_{23} = 1$, and let the other entries be zero or masked.
For $\mathbf{x} = (1, 0, 1)$:

- $a_1 = c_1 = 0$, so $p(x_1 = 1) = \sigma(0) = 0.5$;
- $a_2 = x_1 A_{12} + c_2 = 2 - 1 = 1$, so $p(x_2 = 0 \mid x_1) = 1 - \sigma(1) = 0.2689$;
- $a_3 = x_1 A_{13} + x_2 A_{23} + c_3 = -1 + 0 + 1 = 0$, so $p(x_3 = 1 \mid x_1, x_2) = 0.5$.

By eq-chain, $p(1, 0, 1) = 0.5 \times 0.2689 \times 0.5 = 0.0672$.
The negative log-likelihood is $2.6996$ nats, or $1.2982$ bits per dimension.
The eight probabilities of the model sum to 1, whatever the weights.
`test_worked_example` checks these numbers.

### Pitfalls

> [!pitfall]
> An off-by-one in the shift or the mask leaks the answer. If conditional $d$ can see $x_d$, through mask B in the first layer, an unshifted input, or a direct mask with $\ge$ in place of $>$, the training loss falls toward zero while samples are noise. Check that the Jacobian of the logits in the inputs is strictly triangular, as `test_masks_make_the_jacobian_triangular_in_the_ordering` does.

> [!pitfall]
> Causal convolutions need padding on the left only. A framework's `padding="same"` pads both sides and lets every output see the future; pad $(k - 1)d$ zeros in front and none behind.

> [!pitfall]
> Bits per dimension depend on how the data were discretized. A likelihood over 256 levels, a dequantized density and a binarized image are not comparable, and binarizing MNIST by thresholding gives other numbers than sampling the binary values.

> [!pitfall]
> A masked MADE with too few hidden units of low degree cannot model the late conditionals well, and units of degree $D$ or 0 are wasted, by c-made-degrees. When the hidden layer is small, cycle the degrees evenly through $1..D-1$ instead of drawing them.

> [!pitfall]
> Sampling uses the conditionals in the order the model was trained with. Sampling a MADE in the natural order when its masks were built for another ordering gives wrong samples with no error.

## Experiments

This page has no experiment yet.
Some tests are small experiments.

`test_learns_a_binary_mixture` trains the NumPy MADE with $D = 8$ and $H = 48$ for 800 Adam steps on 4000 draws from a mixture of three noisy binary prototypes. Its loss falls below 0.8 of its starting value. Summed over all 256 vectors, its KL divergence from the true distribution is below 0.1 nats, while the product of the data's marginals, which ignores the dependence between variables, is above 1. The PyTorch and JAX versions run the same test.

`test_ancestral_sampling_draws_from_the_model` draws 40,000 samples from a random MADE and compares their frequencies with the model's probabilities.

`test_doubling_dilations_cover_1024_samples` builds the dilations $1, 2, \ldots, 512$ and finds the field of c-dilations. `test_wavenet_probabilities_sum_to_one` sums a small WaveNet over every sequence.

The checks of the seven derivations compare:

- the product of random conditionals with a normalized joint distribution, and ancestral samples with it;
- the dependencies of MADE outputs on inputs with the ordering;
- NADE's running sum with $D$ separate passes;
- the expected negative log-likelihood with entropy plus KL, and an ensemble with its members;
- the receptive field of dilated stacks with eq-receptive-field;
- the blind spot of masked stacks with its formula;
- the parameter and FLOP counts with the code.

## History

Frey's fully visible sigmoid belief network of 1998 wrote a joint distribution over binary variables as a chain of logistic regressions. Bengio and Bengio replaced the logistic regressions with a neural network in 2000.
Larochelle and Murray introduced NADE at AISTATS 2011, with one hidden layer shared by all conditionals and computed by a running sum.
Germain, Gregor, Murray and Larochelle introduced MADE at ICML 2015, which computes the conditionals in one pass of a masked autoencoder.
In 2016 van den Oord and colleagues introduced PixelRNN and PixelCNN (ICML 2016), Gated PixelCNN (NeurIPS 2016) and WaveNet, whose speech listeners rated above the parametric and concatenative synthesizers of c-wavenet-mos.
In the following years the decoder-only transformer, trained with the same factorization over text, became the architecture of large language models.

## References

- B. J. Frey. Graphical Models for Machine Learning and Digital Communication. MIT Press, 1998.
- Y. Bengio and S. Bengio. Modeling High-Dimensional Discrete Data with Multi-Layer Neural Networks. NIPS 12, 2000.
- H. Larochelle and I. Murray. The Neural Autoregressive Distribution Estimator. AISTATS 2011.
- M. Germain, K. Gregor, I. Murray and H. Larochelle. MADE: Masked Autoencoder for Distribution Estimation. ICML 2015.
- A. van den Oord, N. Kalchbrenner and K. Kavukcuoglu. Pixel Recurrent Neural Networks. ICML 2016.
- A. van den Oord, N. Kalchbrenner, O. Vinyals, L. Espeholt, A. Graves and K. Kavukcuoglu. Conditional Image Generation with PixelCNN Decoders. NeurIPS 2016.
- A. van den Oord, S. Dieleman, H. Zen, K. Simonyan, O. Vinyals, A. Graves, N. Kalchbrenner, A. Senior and K. Kavukcuoglu. WaveNet: A Generative Model for Raw Audio. arXiv 1609.03499, 2016.
