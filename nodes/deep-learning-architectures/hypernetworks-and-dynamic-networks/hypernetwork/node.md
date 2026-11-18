---
tag: NEW-hypernetwork
kind: architecture
slug: hypernetwork
name: Hypernetwork
aliases: [HyperNetwork, HyperNetworks, hypernetworks, hyper network, static hypernetwork, dynamic hypernetwork, HyperRNN, HyperLSTM, weight generation, weight scaling vector]
area: deep-learning-architectures/hypernetworks-and-dynamic-networks
level: core
summary: >-
  A small network that generates the weights of a larger one, from a learned embedding per layer for a convolutional network or anew at every step of a recurrent network.
uses: [matrix-multiplication@m2, backpropagation@al]
related_to: [linear-attention, long-short-term-memory, recurrent-neural-network, convolutional-neural-network, resnet, layer-normalization, batch-normalization, weight-normalization, lora, sparse-mixture-of-experts, multilayer-perceptron, dropout]
history:
  year: 2016
  people: [David Ha, Andrew Dai, Quoc V. Le]
  note: "Ha, Dai and Le posted HyperNetworks to arXiv (1609.09106) in September 2016. HyperNEAT (Stanley et al., 2009) had evolved networks that map the coordinates of a weight to its value, and Schmidhuber (1992, 1993) had proposed fast weights, one network producing the weight changes of another; Ha et al. generated the weights from learned embeddings and trained both networks end to end by gradient descent."
prereqs: [multilayer-perceptron, convolutional-neural-network, recurrent-neural-network, long-short-term-memory]
same_as:
  arxiv: ["1609.09106"]
sources:
  - {paper: "Ha, Dai and Le (2016)", title: "HyperNetworks", locator: "arXiv 1609.09106v4, Sections 2 to 4, Tables 1 to 7 and Appendices A.1, A.2.2, A.2.3 and A.3", note: "the static hypernetwork, the HyperRNN and the HyperLSTM, their experiments and their initialization"}
  - {paper: "Schlag, Irie and Schmidhuber (2021)", title: "Linear Transformers Are Secretly Fast Weight Programmers", locator: "ICML 2021; arXiv 2102.11174, abstract", note: "the fast weight programmers of the early nineties, in which one network programs the weights of another"}
family: hypernetwork
inputs: "for the static hypernetwork, E learned layer embeddings (E, N_z), and the input of the main network, here a batch of images (B, C, S, S); for the dynamic one, a batch of sequences (B, T, N_x)"
outputs: "the generated kernels, E basic kernels (E, N, N, f, f) tiled into the layers of the main network, and the output of the main network, here class scores (B, K); for the HyperRNN, the states (B, T, N_h)"
components: [convolutional-neural-network, long-short-term-memory, multilayer-perceptron]
inductive_biases: ["the kernels of all the layers come from one shared generator: every slice of every generated kernel lies in one affine subspace of dimension d, so the layers share structure and differ by their embeddings of N_z numbers", "the number of parameters grows with the number of layers only by N_z per basic kernel, not by the size of the kernels", "in the dynamic form the weights of a recurrent network change with the time step and the input, a relaxed weight sharing between a recurrent network, which shares all its weights over time, and a feedforward one, which shares none", "the memory-efficient dynamic form can only rescale the rows of fixed matrices, an input-dependent gain like that of a normalization layer"]
params: "N_z E + d (N_z + 1) N + f N f (d + 1) for the static hypernetwork that generates E basic kernels of N x N channels and f x f filters, with the hidden size d = N_z, against E N^2 f^2 weights generated; plus the layers that are not generated"
flops_per_example: "2 E N d (N_z + f^2 N) to generate the kernels, once per batch, plus the cost of the main network with the generated kernels, the same as with ordinary ones"
---

A hypernetwork is a neural network that outputs the weights of another network, the main network.
Ha, Dai and Le use a small network, the hypernetwork, to generate the weights for a larger one, the main network: the main network maps its raw inputs to their targets like any network, while the hypernetwork takes inputs that contain information about the structure of the weights and generates the weights of that layer. {#c-def}
They introduced two kinds in 2016.
A static hypernetwork turns a learned embedding of a few numbers into each kernel of a deep convolutional network, so that the layers share a generator and differ by their embeddings.
A dynamic hypernetwork is a small recurrent network that, at every step of a sequence, rewrites the weights of a larger recurrent network, so that the weights change over time and with the input. {#c-embedding}
Both are trained end to end with the main network by [[al:backpropagation|backpropagation]], since the weights are a differentiable function of the hypernetwork's parameters.

## In plain words

An ordinary network stores every one of its weights as a number to be learned.
A hypernetwork stores a recipe instead.
For a deep convolutional network, each layer gets a short code of a few numbers, and one small shared network turns any code into the full set of filters for a layer; to learn, the network adjusts the codes and the shared network, never the filters directly.
Layers end up similar to each other, since they come from the same recipe, but each can differ through its code, and the network has far fewer numbers to learn.
For a recurrent network, which normally applies the same weights at every step of a sequence, a second, smaller recurrent network watches the sequence as it goes and keeps adjusting the weights of the first one, step by step, so that the big network can behave differently between words than inside a word.

## Overview

Ha et al. take the view that convolutional and recurrent networks are two ends of a spectrum: a recurrent network imposes weight sharing across layers, which makes it inflexible and hard to train because of vanishing gradients, while a deep convolutional network shares no weights between its layers, at the cost of redundant parameters, and a hypernetwork is a relaxed form of weight sharing that strikes a balance between the two. {#c-relaxed}
The input of the hypernetwork is an embedding vector that describes the whole weights of a layer, rather than the coordinates of each weight as in HyperNEAT; the embeddings can be learned parameters, which allows approximate weight sharing within a layer and across the layers of the main network. {#c-embedding-static}
The embeddings can also be generated by the hypernetwork itself, which lets the weights of a recurrent network change over the time steps and adapt to the input sequence. {#c-embedding-dynamic}
The authors' main result is that hypernetworks can generate non-shared weights for an LSTM that work better than the standard LSTM on character-level language modelling, handwriting generation and neural machine translation, while on image recognition the static form reaches respectable results with fewer learnable parameters. {#c-main-result}
They compare the hypernetwork and the main network to a genotype and a phenotype. {#c-genotype}

## Components

### The static hypernetwork

A layer $j$ of a convolutional network has a kernel of $N_{\text{in}} \times N_{\text{out}}$ filters of $f \times f$, which the paper stores as a matrix $K^j \in \mathbb{R}^{N_{\text{in}} f \times N_{\text{out}} f}$, made of $N_{\text{in}}$ slices $K^j_i \in \mathbb{R}^{f \times N_{\text{out}} f}$.
The hypernetwork receives a layer embedding $z^j \in \mathbb{R}^{N_z}$ and predicts $K^j = g(z^j)$, as a two-layer linear network:

$$ a^j_i = W_i z^j + B_i, \qquad K^j_i = \langle W_{\text{out}}, a^j_i \rangle + B_{\text{out}}, \qquad K^j = \begin{pmatrix} K^j_1 \\ \vdots \\ K^j_{N_{\text{in}}} \end{pmatrix}, \qquad i = 1, \dots, N_{\text{in}} . $$ {#eq-static}

The first layer projects $z^j$ into $N_{\text{in}}$ hidden vectors with $N_{\text{in}}$ different matrices $W_i \in \mathbb{R}^{d \times N_z}$ and biases $B_i \in \mathbb{R}^d$; the second projects each hidden vector into a slice with a tensor $W_{\text{out}} \in \mathbb{R}^{f \times N_{\text{out}} f \times d}$ and a bias $B_{\text{out}}$ common to all slices; the paper fixes the hidden size $d$ to $N_z$, although they can differ. {#c-two-layer}
The learnable parameters are $W_i$, $B_i$, $W_{\text{out}}$, $B_{\text{out}}$ and all the embeddings $z^j$; at inference the model reproduces the kernels from the learned embeddings. {#c-learnable}
A two-layer linear network is no more expressive than a one-layer one, but because $W_{\text{out}}$ and $B_{\text{out}}$ are shared, the two-layer form is more compact: a one-layer hypernetwork would have $N_z \times N_{\text{in}} \times f \times N_{\text{out}} \times f$ learnable parameters. {#c-compact}
The authors relate this construction to the hierarchically semiseparable matrices of Xia et al. (2010). {#c-hss}

The kernel is an affine function of the embedding, and every slice of every generated kernel lies in the same affine subspace $B_{\text{out}} + \operatorname{span}\{W_{\text{out}}[:,:,1], \dots, W_{\text{out}}[:,:,d]\}$, of dimension at most $d$, whatever the layer. {#c-affine}
A one-layer map could give each slice up to $\min(N_z, f \cdot N_{\text{out}} f)$ dimensions of its own; the two-layer form trades that freedom for its smaller size. {#c-subspace}
The derivation `linear-hypernetwork` proves both statements.

### Kernels of several sizes

When the numbers of channels of the kernels are multiples of a basic size, as in the residual networks of the paper, where $N_{\text{in}}$ and $N_{\text{out}}$ are multiples of 16, the hypernetwork generates basic kernels of that size and a larger kernel is the concatenation of several; a kernel with $N_{\text{in}} = 32$ and $N_{\text{out}} = 64$ tiles eight basic kernels, each from its own embedding. {#c-basic-size}
For a kernel $m_{\text{in}}$ times wider in its input and $m_{\text{out}}$ in its output,

$$ K = \begin{pmatrix} K^{(1)} & K^{(2)} & \cdots & K^{(m_{\text{out}})} \\ \vdots & & & \vdots \\ K^{((m_{\text{in}} - 1) m_{\text{out}} + 1)} & \cdots & & K^{(m_{\text{in}} m_{\text{out}})} \end{pmatrix}, $$ {#eq-tile}

where block $(r, c)$ maps input channels $rN$ to $(r + 1)N - 1$ to output channels $cN$ to $(c+1)N - 1$ and depends on one embedding only. {#c-tile-layout}
Larger kernels therefore need proportionally more embeddings.

### The dynamic hypernetwork: HyperRNN

A basic RNN computes $h_t = \phi(W_h h_{t-1} + W_x x_t + b)$ with the same $W_h \in \mathbb{R}^{N_h \times N_h}$, $W_x \in \mathbb{R}^{N_h \times N_x}$ and $b$ at every step.
In a HyperRNN, at every step a small recurrent cell reads the concatenation of the input $x_t$ and the main network's state $h_{t-1}$ and outputs a vector $\hat h_t$, which generates the weights of the main RNN at the same time step; both networks are trained jointly. {#c-hyperrnn-def}
In general form the weights are linear in three embeddings $z_h$, $z_x$, $z_b \in \mathbb{R}^{N_z}$:

$$ h_t = \phi\big(W_h(z_h) h_{t-1} + W_x(z_x) x_t + b(z_b)\big), \qquad W_h(z_h) = \langle W_{hz}, z_h \rangle, \quad W_x(z_x) = \langle W_{xz}, z_x \rangle, \quad b(z_b) = W_{bz} z_b + b_0 , $$ {#eq-full}

with $W_{hz} \in \mathbb{R}^{N_h \times N_h \times N_z}$, $W_{xz} \in \mathbb{R}^{N_h \times N_x \times N_z}$, $W_{bz} \in \mathbb{R}^{N_h \times N_z}$ and $b_0 \in \mathbb{R}^{N_h}$. {#c-eq5}
The embeddings are computed by a recurrent hyper cell from $x_t$ and $h_{t-1}$:

$$ \hat x_t = \begin{pmatrix} h_{t-1} \\ x_t \end{pmatrix}, \qquad \hat h_t = \phi\big(W_{\hat h} \hat h_{t-1} + W_{\hat x} \hat x_t + \hat b\big), \qquad z_h = W_{\hat h h} \hat h_t + b_{\hat h h}, \quad z_x = W_{\hat h x} \hat h_t + b_{\hat h x}, \quad z_b = W_{\hat h b} \hat h_t . $$ {#eq-hyper-cell}

The hyper cell has $\hat N$ units, typically far fewer than $N_h$, and since the embeddings of $N_z$ numbers are typically smaller than $\hat N$, a linear network projects its state into them. {#c-eq6}

The paper found the general form impractical: its memory is $N_z$ times that of a basic RNN, which limits the number of hidden units. {#c-memory-paper}

### Weight scaling vectors

The practical form keeps fixed matrices and lets the hyper cell scale their rows: an intermediate vector $d(z) \in \mathbb{R}^{N_h}$, a linear projection of $z$ called a weight scaling vector, multiplies row $n$ of $W$ by $d_n(z)$. {#c-scaling-def}

$$ h_t = \phi\big(d_h(z_h) \odot W_h h_{t-1} + d_x(z_x) \odot W_x x_t + b(z_b)\big), \qquad d_h(z_h) = W_{hz} z_h, \quad d_x(z_x) = W_{xz} z_x, \quad b(z_b) = W_{bz} z_b + b_0 , $$ {#eq-scaled}

now with $W_{hz}, W_{xz} \in \mathbb{R}^{N_h \times N_z}$.
The authors give up building a whole matrix as a combination of $N_z$ matrices, and in exchange the memory is of the order of $N_z$ times the number of hidden units, and the row scaling is an elementwise product, which is much faster in practice. {#c-tradeoff}
Scaling the rows of $W$ by $\delta$ and then multiplying by a vector is the elementwise product $\delta \odot (Wv)$. {#c-rowscale}
eq-scaled is the special case of eq-full whose tensors have the slices $W^{(k)}_{hz} = \operatorname{diag}(W_{hz}[:, k])\, W_h$ and $W^{(k)}_{xz} = \operatorname{diag}(W_{xz}[:, k])\, W_x$, so it computes the same states as that general HyperRNN. {#c-scaled-is-full}
The derivation `weight-scaling` proves this and counts the memory of both forms.

The authors compare this form with recurrent batch normalization and [[layer-normalization|layer normalization]], which scale the inputs of the activation function by statistics of those inputs, and with the multiplicative RNN and multiplicative integration RNN; the difference is that the scaling of a HyperRNN can change at every time step and for every sample. {#c-norm-similarity}
Normalization can be combined with a HyperRNN, the hyper cell then having to find a scaling policy that complements it, and the paper tests the HyperLSTM both with and without layer normalization. {#c-ln-combo}

### The HyperLSTM

The experiments use the LSTM version, the HyperLSTM.
Its hyper cell is itself an LSTM with layer normalization, reading $\hat x_t = [h_{t-1}; x_t]$; each of the four gates $y \in \{i, g, f, o\}$ of the main LSTM has its own embeddings $z^y_h$, $z^y_x$, $z^y_b$, projected from the hyper cell's state, its own weight scaling vectors, and a layer normalization on its preactivation:

$$ y_t = \operatorname{LN}\big(d^y_h \odot W^y_h h_{t-1} + d^y_x \odot W^y_x x_t + b^y(z^y_b)\big), \qquad c_t = \sigma(f_t) \odot c_{t-1} + \sigma(i_t) \odot \operatorname{DropOut}(\phi(g_t)), \qquad h_t = \sigma(o_t) \odot \phi(\operatorname{LN}(c_t)) . $$ {#eq-hyper-lstm}

The hyper cell is a normal LSTM cell with layer normalization whose input is the concatenation of the input signal and the hidden units of the main LSTM, and the embeddings of each gate are linear projections of its hidden state. {#c-hyperlstm}

The only dropout is the one of the main LSTM's candidate, as in recurrent dropout without memory loss; it is not applied in the hyper cell, except for larger systems. {#c-dropout}

## Shapes and parameters

The static hypernetwork has

$$ N_{\text{hyper}} = N_z E + d(N_z + 1) N_{\text{in}} + f N_{\text{out}} f (d + 1) $$ {#eq-hyper-params}

parameters for $E$ basic kernels, against $E N_{\text{in}} f N_{\text{out}} f$ weights that it generates. {#c-params}
The paper writes this count with $D$, one embedding per layer, against $D \times N_{\text{in}} \times f \times N_{\text{out}} \times f$ for the kernels of the main network. {#c-count-paper}
The network of this page's shape contract has an ordinary stem convolution from $C$ channels to $M = mN$, $D$ convolutions of $M$ channels whose kernels are tiled from $m^2$ basic kernels each, so $E = Dm^2$, a global average pool and a linear layer to $K$ classes:

$$ N_{\text{params}} = f^2 C M + M + N_{\text{hyper}} + DM + MK + K . $$ {#eq-params}

In the shape contract this is

{shape:params}

and the example network, 1 input channel, basic kernels of 2 channels tiled $2 \times 2$, filters of $3 \times 3$, 2 generated layers and $N_z = 2$, has {shape:params@example} parameters, of which the hypernetwork has {shape:hyper_params@example}, for {shape:generated_weights@example} generated weights.

For the MNIST layer of the paper, a $7 \times 7$ kernel of 16 by 16 channels from an embedding of $N_z = 4$, the hypernetwork has {shape:hyper_params@mnist} parameters, including the embedding, for {shape:generated_weights@mnist} weights; a one-layer hypernetwork would have {shape:one_layer_params@mnist}, four times the kernel itself. {#c-mnist-count}
In general a one-layer map from $z$ to a basic kernel has $N_z N_{\text{in}} f N_{\text{out}} f$ parameters, $N_z$ times the kernel, so it pays for itself only when more than $N_z$ basic kernels share it, while the shared part of the two-layer form, $d(N_z + 1)N_{\text{in}} + fN_{\text{out}}f(d + 1)$, is smaller than one kernel when $d + 1$ is well below $N_{\text{in}}$. {#c-one-layer-count}
For the wide residual networks WRN 40-1 and WRN 40-2, whose 36 convolutions of $3 \times 3$ are tiled from basic kernels of 16 channels, the paper's $N_z = 64$ gives 242 and 966 basic kernels and hypernetworks of 91,408 and 137,744 parameters. {#c-wrn-count}
The derivation `parameter-count` proves these counts.

A HyperRNN of $N_h$ units with a hyper cell of $\hat N$ units has $N_h(N_h + N_x) + 3N_zN_h + N_h$ parameters in its main network in the scaled form and $N_zN_h(N_h + N_x) + N_zN_h + N_h$ in the general form, plus $\hat N(\hat N + N_h + N_x + 1) + 3N_z\hat N + 2N_z$ in the hyper cell; per sequence and step the general form builds $N_h(N_h + N_x)$ numbers of weights and the scaled one $2N_h$. {#c-rnn-params}
The weights of the general form take $N_z$ times the parameters of those of a basic RNN, and it builds $(N_h + N_x)/2$ times as many numbers per step as the scaled form. {#c-memory}

## Computation and cost

Generating the $E$ basic kernels costs

$$ C_{\text{hyper}} = 2 E N_{\text{in}} d \,(N_z + f^2 N_{\text{out}}) $$ {#eq-flops}

FLOPs, counting a multiply-add as 2, once per forward pass whatever the size of the batch; the main network then costs what it would cost with ordinary kernels. {#c-flops}
The example costs {shape:flops_forward@example} FLOPs in all, and the MNIST config of the contract, a stem convolution and one generated layer of $7 \times 7$ on an image of $28 \times 28$, costs {shape:flops_forward@mnist}.
A static hypernetwork thus saves parameters, and the memory and communication they need, not computation: the generated network is as large as an ordinary one, and at inference the kernels can be generated once and stored.
The scaled HyperRNN adds the hyper cell, of $\hat N$ units, and two elementwise products per step to the cost of the main RNN.

## Training

Both networks are trained together by gradient descent; the gradient of a hypernetwork parameter sums the gradients of all the weights it generates, through every layer that uses them, and through every step for a HyperRNN.
For MNIST the paper trains with Adam at a learning rate of 0.001 on mini-batches of 1000, on a split of 55000, 5000 and 10000 images, padding the training images to 30 by 30 and cropping them at random to 28 by 28. {#c-mnist-setup}
For the residual networks it trains the hypernetwork version with Adam and the ordinary one with Nesterov momentum SGD, both on mini-batches of 128 with a learning rate schedule, with L2 regularization on the generated kernels as on the ordinary ones. {#c-wrn-train}
For character-level Penn Treebank the HyperLSTM is trained on mini-batches of 128 sequences of length 100, with Adam at a learning rate of 0.001, gradient clipping of 1.0, and dropout with a keep probability of 90% on the input, the output and the recurrent layer. {#c-ptb-setup}

The HyperLSTM has an initialization of its own.
The biases of the hyper cell start at zero and its weights are orthogonal; the projections to $z_h$ and $z_x$ start with zero weights and biases of one, and that to $z_b$ with normal weights of standard deviation 0.01; $W_h$ and $W_x$ are orthogonal, $b_0$ and $W_{bz}$ zero, and $W_{hz}$ and $W_{xz}$ the constant $0.1/N_z$, so that the scaling vectors start at 0.1 rather than 1.0, following recurrent batch normalization. {#c-init}
With these values every weight scaling vector equals 0.1 in every entry, for every input, until training moves the projections to $z_h$ and $z_x$ away from zero. {#c-init-scale}

## Inductive biases

**One generator for all the layers.**
Every slice of every kernel of a static hypernetwork lies in the same affine subspace of dimension $d$, so the layers share their filters up to $N_z$ numbers each.
The authors explain the loss of accuracy of their residual networks this way: different layers extract different levels of features and need different kinds of filters, and the hypernetwork enforces some commonality between all of them while offering each layer 64 degrees of freedom. {#c-64dof}
`test_kernels_are_an_affine_function_of_the_embedding` checks the rank of the generated slices.

**Parameters that do not grow with the kernels.**
Each basic kernel costs $N_z$ numbers, whatever its size; the shared part of the hypernetwork is paid once.

**Weights that move with time and input.**
A recurrent network shares its weights over all time steps, a deep feedforward network shares none across layers; a dynamic hypernetwork lets them vary, with the variation itself a learned function of the sequence.

**An input-dependent gain.**
In the scaled form the hyper cell can only rescale the rows of fixed matrices and shift the bias, the operations of a normalization layer, which is why the paper compares the two.

## Evidence

- **MNIST.** Generating the $7 \times 7 \times 16 \times 16$ kernel of the second layer of a small two-layer convolutional network from an embedding of $N_z = 4$, the hypernetwork reached a test accuracy of 99.24%, against 99.28% for the ordinary network; the 12,544 weights of the kernel were represented by an embedding of 4 numbers and a hypernetwork of 4240 parameters. {#c-mnist}
- **CIFAR-10.** With one hypernetwork generating the kernels of the 36 layers of conv2, conv3 and conv4 of WRN 40-1 and WRN 40-2, with $N_z = 64$, the relaxed weight sharing cost about 1.25 to 1.5% of classification accuracy and drastically reduced the number of parameters. {#c-wrn-cost}
  The test errors were 8.02% with 0.097 M parameters against 6.73% with 0.563 M for WRN 40-1, and 7.23% with 0.148 M against 5.66% with 2.236 M for WRN 40-2. {#c-wrn}
- **Character-level Penn Treebank.** With 1000 units in the main network, a hyper cell of 128 units and embeddings of 4, the HyperLSTM reached 1.265 bits per character on the test set with 4.91 M parameters, against 1.312 for an LSTM of 1000 units, 1.306 for one of 1250 units, 1.281 for a 2-layer LSTM and 1.267 for a layer-normalized LSTM; with layer normalization the HyperLSTM reached 1.250, with an embedding of 16 1.233, and with 2 layers 1.219. {#c-ptb}
  The authors read this as the hyper cell having learned a policy of scaling the inputs of the activations as efficient as layer normalization, and the further gain of combining the two as a policy beyond normalization by moments. {#c-ptb-reading}
- **enwik8.** With 1800 units, the HyperLSTM reached 1.391 bits per character on the test set with 18.71 M parameters, against 1.470 for an LSTM without recurrent dropout and 1.402 for a layer-normalized LSTM; with layer normalization it reached 1.353, and 1.340 with 2048 units and 26.54 M parameters. {#c-enwik8}
  The HyperLSTM converged faster per training step than the LSTM and the layer-normalized LSTM. {#c-converge}
- **What the hyper cell does.** In a passage sampled from the model trained on enwik8, the weights of the main network changed little inside words such as "Europeans" and changed most between words and sometimes at brackets. {#c-weights-change}
  In the histograms of $\phi(c_t)$ layer normalization reduces the saturation of the LSTM, while the HyperLSTM cell is saturated most of the time: its policy works differently from normalization by statistics, and reaches a similar performance. {#c-saturated}
- **Handwriting.** On the IAM online handwriting database, 12179 lines from 221 writers, the HyperLSTM of 900 units reached a log-loss of -1,162 nats on the validation set with 3.94 M parameters, the best of the table, against -1,055 for an LSTM of 900 units, -1,135 for a 2-layer LSTM of 650 units, -1,096 for a layer-normalized LSTM and -1,067 for a layer-normalized HyperLSTM. {#c-iam}
  On this task layer normalization did not combine well with the HyperLSTM. {#c-iam-ln}
  The changes of the weights were concentrated at discrete instants, many between written words and some between characters, which the authors read as regime changes rather than gradual adjustments. {#c-regime}
- **Translation.** Replacing the LSTM cells of the GNMT wordpiece model by HyperLSTM cells on WMT'14 English to French gave 40.03 BLEU against 38.95, and a log perplexity of 0.993 against 1.027; an ensemble of 8 LSTM models had 40.35. {#c-nmt}
- **Coordinates as inputs.** A hypernetwork that maps the coordinates of a pixel and of a hidden unit to the weight between them, an 801-parameter 4-layer network generating the 784 × 256 = 200704 weights of a fully connected MNIST classifier, learned filters that look convolutional but reached only 93.5% accuracy, against 98.5% for the ordinary network, which led the authors to embeddings. {#c-coord}

## Limitations

- **Fewer parameters, not less computation.** The static hypernetwork generates a network as large as an ordinary one and runs it at the same cost; only the parameters to learn and store shrink.
- **Accuracy for compression.** On CIFAR-10 the shared generator cost 1.25 to 1.5 points of accuracy for a sixth or less of the parameters.
- **Shared subspace.** All the generated slices lie in one subspace of dimension $d$; layers that need very different filters cannot all have them.
- **Basic sizes.** The tiling of eq-tile needs the widths of all the generated layers to be multiples of the basic size $N$, and each tile is generated without regard to its neighbours.
- **The general dynamic form does not scale.** eq-full needs $N_z$ times the memory of the main RNN; the practical form only rescales rows, a much smaller family of weight matrices.
- **Interaction with normalization is task dependent.** Layer normalization helped the HyperLSTM in language modelling and hurt it in handwriting generation.

## Variants and descendants

- **Fast weights.** Schmidhuber (1992, 1993) proposed fast weights, in which one network produces context-dependent weight changes for a second network; small experiments were run for feedforward networks, and the recurrent version was mainly a thought experiment. {#c-fastweights}
  Schlag et al. show the formal equivalence of linearized self-attention and the fast weight controllers of the early nineties, in which a slow network learns by gradient descent to program the fast weights of another network through additive outer products of keys and values; see [[linear-attention|linear attention]]. {#c-schlag}
- **HyperNEAT and its relatives.** HyperNEAT evolves networks whose inputs are the virtual coordinates of each weight of the main network; Compressed Weight Search evolves the coefficients of a discrete cosine transform of the weights; differentiable pattern producing networks evolve the structure and learn the weights. {#c-hyperneat}
  The main difference of Ha et al.'s hypernetworks is that they are trained end to end by gradient descent together with the main network. {#c-end-to-end}
- **Predicted parameters and dynamic filters.** Earlier work predicted some parameters of a convolutional network with another network (Denil et al., 2013; Bertinetto et al., 2016; De Brabandere et al., 2016); applying the idea to recurrent networks is the paper's main contribution. {#c-related-conv}
- **Low-rank and multiplicative adaptation.** The weight scaling vectors of eq-scaled are an input-dependent diagonal factor; [[lora|LoRA]] adds a learned low-rank term to fixed weights, and [[sparse-mixture-of-experts|mixtures of experts]] choose among weights per input instead of generating them.
- **Later uses.** Hypernetworks have been used to search architectures by generating the weights of candidate networks (SMASH, Brock et al., 2018) and to learn sequences of tasks by generating the weights of each task from its embedding (von Oswald et al., 2020).

## Notable models

- **Hyper Residual Network 40-1 and 40-2**, the wide residual networks of the paper with generated kernels, 0.097 M and 0.148 M parameters.
- **HyperLSTM**, with and without layer normalization, in the paper's language models, its handwriting model and its GNMT translation model.

## Implementations

The NumPy reference `impl/numpy/hypernetwork.py` implements eq-static and eq-tile for rectangular grids, the convolutional network of the shape contract with a forward pass and a backward pass written by hand through the convolutions and the hypernetwork down to the embeddings, the HyperRNN in both forms, eq-full and eq-scaled, with backpropagation through time by hand for the scaled one, the HyperLSTM of eq-hyper-lstm and its initialization, and the counts eq-params and eq-flops.
The PyTorch version `impl/pytorch/hypernetwork.py` has the same networks as `nn.Module`s with the same parameter names, the dropout of the HyperLSTM, and `reset_parameters` for the initialization; its tests run `torch.autograd.gradcheck` and a small experiment.
The JAX version `impl/jax/hypernetwork.py` is a set of pure functions of a parameter dictionary, with the recurrences as `jax.lax.scan`, tested under `jax.jit`, `jax.vmap` and `jax.grad`.
All three load the shared fixture of `fixtures/`: the example network on two images, a HyperRNN and a HyperLSTM on two sequences of 4 steps, with their outputs, losses and gradients.

The training loss of the convolutional network is the cross-entropy of the labels averaged over a batch of $B$ images,

$$ \mathcal{L} = -\frac{1}{B} \sum_{b=1}^{B} \log \operatorname{softmax}\big(f(x_b)\big)_{y_b} , $$ {#eq-loss}

and that of the HyperRNN in the fixture half the mean squared error of its states against a target.

### Worked example

Take one basic kernel of 1 input and 2 output channels of size $1 \times 1$, with $N_z = d = 1$, $W_1 = 2$, $B_1 = 1$, $W_{\text{out}} = (1, -1)$ and $B_{\text{out}} = (0.5, 0)$.
eq-static gives $a = 2z + 1$ and the kernel $K = a\,(1, -1) + (0.5, 0)$.
For $z = 0$, $a = 1$ and $K = (1.5, -1)$; for $z = 1$, $a = 3$ and $K = (3.5, -3)$; for $z = -0.5$, $a = 0$ and $K = B_{\text{out}} = (0.5, 0)$.
All three kernels, and every other one this hypernetwork can produce, lie on the line $(0.5, 0) + t(1, -1)$, the subspace of dimension $d = 1$ of the statement of `linear-hypernetwork`; the kernel $(1, 1)$, for instance, is out of reach, though a one-layer hypernetwork with a bias could produce it.
The test `test_worked_example` checks these numbers.

### Pitfalls

> [!pitfall]
> **The layout of the kernel.** The paper's matrix $K^j$ has its entry $(i f + u, o f + v)$ for input channel $i$, output channel $o$ and filter position $(u, v)$, while a convolution library wants $(N_{\text{out}}, N_{\text{in}}, f, f)$. A reshape without the transpose gives kernels of the right shape and the wrong meaning; `test_kernel_layout` checks the conversion both ways.

> [!pitfall]
> **The order of the tiles.** Which embedding goes to which block of a tiled kernel is a convention; the gradient of each embedding must come back from the block it generated. `untile` inverts `tile`, and `test_tiling_of_eq3` checks a grid of $2 \times 4$ blocks.

> [!pitfall]
> **$\hat h_t$ or $\hat h_{t-1}$.** The text of Section 3.2 says that the hyper cell's output at a step generates the main network's weights at the same step, but eqs. 6 and 11 of the paper project the embeddings from $\hat h_{t-1}$. This page follows the text and uses $\hat h_t$, which lets the weights at step $t$ depend on $x_t$; an implementation that follows the equations is a different network and will not match the fixture.

> [!pitfall]
> **The printed eq. 5.** The paper's eq. 5 writes $W_x(z_x)$ without $x_t$, and gives $W_{\hat x}$ a shape of $N_{\hat h} \times (N_h + N_z)$ where $\hat x_t = [h_{t-1}; x_t]$ has $N_h + N_x$ entries; read them as $W_x(z_x)\,x_t$ and $N_{\hat h} \times (N_h + N_x)$.

> [!pitfall]
> **Scaling vectors at random.** With $W_{hz}$ and $W_{xz}$ at the usual random initialization, the scaling vectors start at random values of either sign, which flip and shrink the rows of the main weights at random; the paper's constant $0.1/N_z$ with biases of one in the projections makes every scaling vector start at 0.1. `test_the_initialization_of_appendix_a23` checks it.

> [!pitfall]
> **Dropout in the wrong network.** The paper's dropout is on the candidate $\phi(g_t)$ of the main LSTM only; dropping the hyper cell's units as well changes all the weights of the main network at once.

## Experiments

This page has no experiment directory yet; the PyTorch test `test_generated_kernels_learn_with_fewer_parameters` is a small experiment.
Images of $8 \times 8$ pixels hold a bar of 3 pixels at a random place, horizontal, vertical or along either diagonal, the class, under Gaussian noise of standard deviation 0.3.
A network with a stem convolution and four convolutions of 8 channels whose kernels are tiled from $2 \times 2$ basic kernels of 4 channels, generated from embeddings of $N_z = 4$, has 472 parameters; the same network with ordinary kernels has 2,452.
Trained for 300 steps of Adam at a learning rate of 0.003 on batches of 64, and tested on 512 new images where chance is 0.25, the network with generated kernels reached 0.896, 0.939 and 0.715 with seeds 0, 1 and 2, and the ordinary one 0.762, 0.783 and 0.838: with a fifth of the parameters, the hypernetwork did as well on this task, with a larger spread over seeds.
With noise of standard deviation 0.6 and a learning rate of 0.01, neither network rose above chance in 300 steps.
The test runs seed 0 and asserts that both networks pass 0.7.
The checks of the three derivations verify the counts, the affine subspace of the generated kernels and the equivalence of the two forms of the HyperRNN.

## History

The paper places hypernetworks in a line from evolutionary computation, where it is easier to evolve a small network that generates the weights of a large one than to search the large weight space directly. {#c-hist-evolution}
HyperNEAT, Compressed Weight Search and differentiable pattern producing networks were reported mostly at small scales, which the authors attribute to slow training and to the heuristics they need. {#c-hist-small}
Fast weights (Schmidhuber, 1992, 1993) were an earlier form of one network writing the weights of another.
Ha, Dai and Le, at Google Brain, posted HyperNetworks to arXiv (1609.09106) in September 2016, with a blog post and a TensorFlow implementation of the HyperLSTM.
Their contribution was to make the idea work at the scale of deep convolutional networks and long recurrent networks, trained end to end, and in particular to generate the weights of recurrent networks.

## References

- Ha, D., Dai, A. and Le, Q. V. (2016). HyperNetworks. arXiv 1609.09106.
- Schlag, I., Irie, K. and Schmidhuber, J. (2021). Linear Transformers Are Secretly Fast Weight Programmers. ICML 2021. arXiv 2102.11174.
- Stanley, K. O., D'Ambrosio, D. B. and Gauci, J. (2009). A Hypercube-Based Encoding for Evolving Large-Scale Neural Networks. Artificial Life 15(2).
- Schmidhuber, J. (1992). Learning to Control Fast-Weight Memories: An Alternative to Dynamic Recurrent Networks. Neural Computation 4(1).
- Brock, A., Lim, T., Ritchie, J. M. and Weston, N. (2018). SMASH: One-Shot Model Architecture Search through HyperNetworks. ICLR 2018. arXiv 1708.05344.
- von Oswald, J., Henning, C., Sacramento, J. and Grewe, B. F. (2020). Continual Learning with Hypernetworks. ICLR 2020. arXiv 1906.00695.
