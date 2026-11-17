---
tag: NEW-multilayer-perceptron
kind: architecture
slug: multilayer-perceptron
name: Multilayer perceptron
aliases: [MLP, feedforward neural network, fully connected network, deep feedforward network, multilayer feedforward network]
area: deep-learning-architectures/feedforward-networks
level: core
summary: >-
  A stack of fully connected layers, each an affine map followed by a fixed nonlinearity, trained end to end with backpropagation; the basic neural network from which most others are built.
uses: [backpropagation@al, relu@al, activation-function@al, matrix-multiplication@m2, stochastic-gradient-descent@m2]
related_to: [convolutional-neural-network, transformer, perceptron@al, linear-separability@al, he-initialization@al, xavier-initialization@al, vanishing-gradient-problem@al]
history:
  year: 1986
  people: [Alexey Ivakhnenko, Valentin Lapa, Seppo Linnainmaa, Paul Werbos, David Rumelhart, Geoffrey Hinton, Ronald Williams]
  note: "Ivakhnenko and Lapa's GMDH networks (1965) were deep multilayer feedforward networks grown layer by layer by regression; Linnainmaa described efficient backpropagation in 1970 and Werbos applied it to neural networks in 1981; Rumelhart, Hinton and Williams popularised it in 1986. The dates follow Schmidhuber's overview, arXiv 1404.7828, Sections 5.3 and 5.5."
prereqs: [perceptron@al, backpropagation@al, matrix-multiplication@m2]
same_as:
  wikipedia: Multilayer_perceptron
sources:
  - {paper: "Kidger and Lyons (2020)", title: "Universal Approximation with Deep Narrow Networks", locator: "arXiv 1905.08539v2, the abstract, Sections 1 and 2", note: "the classical theorem as they state it, and the deep narrow case"}
  - {paper: "Park, Yun, Lee and Shin (2021)", title: "Minimum Width for Universal Approximation", locator: "arXiv 2006.08859v1, the abstract"}
  - {paper: "Eldan and Shamir (2016)", title: "The Power of Depth for Feedforward Neural Networks", locator: "arXiv 1512.03965v4, the abstract"}
  - {paper: "Telgarsky (2016)", title: "Benefits of Depth in Neural Networks", locator: "arXiv 1602.04485v2, the abstract and Section 1"}
  - {paper: "Glorot and Bengio (2010)", title: "Understanding the Difficulty of Training Deep Feedforward Neural Networks", locator: "PMLR 9, the abstract, Sections 3.1 and 4.2.1"}
  - {paper: "Glorot, Bordes and Bengio (2011)", title: "Deep Sparse Rectifier Neural Networks", locator: "PMLR 15, the abstract"}
  - {paper: "He, Zhang, Ren and Sun (2015)", title: "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification", locator: "arXiv 1502.01852v1, Section 2.2"}
  - {paper: "Hendrycks and Gimpel (2016)", title: "Gaussian Error Linear Units (GELUs)", locator: "arXiv 1606.08415v5, the abstract"}
  - {paper: "Schmidhuber (2015)", title: "Deep Learning in Neural Networks: An Overview", locator: "arXiv 1404.7828v4, Sections 5.3, 5.5 and 5.5.1", note: "the history of deep feedforward networks and of backpropagation"}
family: feedforward
inputs: "a batch of feature vectors as a (B, d_in) array"
outputs: "a (B, d_out) array: regression values, or logits for a softmax"
components: [relu@al, activation-function@al]
inductive_biases: ["none on the structure of the input: every unit of a layer reads every unit of the layer below, so the network has no notion of position, neighbourhood or order", "composition: the function is built from simple layers, and depth lets some functions be written with far fewer units than one wide layer needs"]
params: "(n_0 + 1) d + (L - 1)(d + 1) d + (d + 1) k"
flops_per_example: "2 (n_0 d + (L - 1) d^2 + d k), counting a multiply-add as 2 FLOPs and leaving out biases and activations"
---

A multilayer perceptron, or MLP, is a neural network made of fully connected layers: each layer multiplies its input by a weight matrix, adds a bias, and applies a fixed nonlinearity, and the last layer is affine.
The nonlinearity is what makes the stack more than one linear map; with it, one hidden layer of enough units can approximate any continuous function on a compact set, and more layers can express some functions with exponentially fewer units.
All the weights are learned together by gradient descent, with the gradient computed by backpropagation.
The MLP is the base case of deep learning: a convolutional network is an MLP with shared, local weights, and every block of a Transformer contains one.

## In plain words

A single artificial neuron takes a weighted sum of its inputs and fires if the sum is large enough; it can only draw a straight line through the input space.
An MLP puts many such neurons side by side in a layer, and stacks layers so that each layer reads the outputs of the one before.
The first layer draws many lines; the next combines them into corners, bands and regions; later ones combine those again.
Because each neuron bends its output with a nonlinearity, the stack can carve out shapes that no single line can, such as the XOR pattern below.
Training adjusts every weight a little at a time to reduce the error, and backpropagation is the bookkeeping that says how much each weight is to blame.

## Overview

The single-layer [[al:perceptron|perceptron]] is a linear classifier, so it can only separate classes that a hyperplane separates; the page on [[al:linear-separability|linear separability]] makes this precise.
XOR is the smallest problem that is not linearly separable: no affine function is positive on $(0, 1)$ and $(1, 0)$ and non-positive on $(0, 0)$ and $(1, 1)$. {#c-xor-linear}
The least-squares fit of a linear model predicts $1/2$ at every point. {#c-xor-lstsq}
One hidden layer of two ReLU units fixes this (the worked example below).

The hidden layers must be nonlinear. If every activation is the identity, the whole network is one affine map, $y = W_L \cdots W_0 x + b$, whatever its depth. {#c-collapse}
Its rank is then at most the smallest width, so a narrow layer only restricts it. {#c-collapse-rank}
With a nonlinearity that is not a polynomial, one hidden layer is already enough in principle: Kidger and Lyons state the classical universal approximation theorem (Cybenko 1989, Hornik 1991, Pinkus 1999) as follows: for a continuous activation $\rho$, networks with one hidden layer of arbitrary width are dense in $C(K)$ for every compact $K \subseteq \mathbb{R}^n$ if and only if $\rho$ is nonpolynomial. {#c-uat}
They note that the theorem extends to any bounded number of hidden layers by letting the extra layers approximate the identity, so it is a statement about arbitrary width and bounded depth. {#c-uat-depth}

The theorem says that a good network exists, not how large it is or how to find it.
Both questions are what the rest of this page is about: width and depth in Shapes and parameters and in Inductive biases, and finding the weights in Training.

## Components

### The layer

A network with $L$ hidden layers has widths $n_0, n_1, \dots, n_{L+1}$, from the input to the output. Hidden layer $l$, for $l = 0, \dots, L - 1$, computes

$$ z_l = W_l h_l + b_l, \qquad h_{l+1} = \sigma(z_l), \qquad h_0 = x , $$ {#eq-layer}

with a weight matrix $W_l$ of shape $n_{l+1} \times n_l$, a bias $b_l$ of length $n_{l+1}$ and an [[al:activation-function|activation function]] $\sigma$ applied element by element. The output layer is affine:

$$ y = W_L h_L + b_L . $$ {#eq-output}

For classification a softmax turns $y$ into probabilities; that belongs to the loss, not to the network.
Each unit of a layer is connected to every unit of the layer below, hence the name fully connected, or dense.

### Activations

The [[al:relu|ReLU]], $\operatorname{relu}(z) = \max(0, z)$, is the default in deep networks.
Glorot, Bordes and Bengio find that rectifying units yield equal or better performance than hyperbolic tangent networks in spite of the hard non-linearity and the non-differentiability at zero, and that deep rectifier networks reach their best performance on purely supervised tasks with large labelled datasets without unsupervised pre-training. {#c-rectifier}
The logistic sigmoid was the classic choice and is now avoided in hidden layers: Glorot and Bengio find it unsuited for deep networks with random initialization, because its mean value can drive the top hidden layer into saturation. {#c-sigmoid-saturation}
The hyperbolic tangent is symmetric around zero and avoids that particular problem.
The GELU of Hendrycks and Gimpel is $x \Phi(x)$, with $\Phi$ the standard Gaussian distribution function; it weights inputs by their value rather than gating them by their sign as the ReLU does. {#c-gelu}
They report performance improvements over the ReLU and the ELU across all the computer vision, natural language processing and speech tasks they consider. {#c-gelu-results}

## Shapes and parameters

Layer $l$ has $n_l n_{l+1}$ weights and $n_{l+1}$ biases, so the network has

$$ P = \sum_{l=0}^{L} \big( n_l n_{l+1} + n_{l+1} \big) $$ {#eq-params}

parameters. {#c-params}
With $L$ hidden layers of width $d$, input width $n_0$ and output width $k$, that is

{shape:params}

parameters, linear in the depth and quadratic in the width. {#c-width-depth-params}
The shape contract has three blocks, the first hidden layer, each further hidden layer and the output layer; its example network, 3 inputs, two hidden layers of 4 and 2 outputs, has {shape:params@example} parameters, and the XOR network of the worked example has {shape:params@xor}.
An MNIST-sized network with three hidden layers of 512 has 932,362: the first layer alone holds 401,920, because it reads all $784$ pixels.
The quadratic growth is the reason an MLP is a poor fit for raw images, and the reason the [[convolutional-neural-network|convolutional neural network]] shares a small kernel instead.

## Computation and cost

The forward pass over a batch of $B$ examples is one matrix product per layer:

$$ C_\text{fwd} = 2 B \sum_{l=0}^{L} n_l n_{l+1} $$ {#eq-flops}

FLOPs, counting a multiply-add as two and leaving out the biases and the activations. {#c-flops}
In the shape contract the forward pass of the example network over a batch of 5 costs {shape:flops_forward@example} FLOPs, and the XOR network over its 4 points {shape:flops_forward@xor}.
The backward pass costs twice as much, $2 C_\text{fwd}$, because each layer makes two products of the size of its forward product, one for the gradient of its weights and one for the gradient of its input. {#c-backward-cost}
A training step therefore costs about three forward passes, $6$ FLOPs per weight per example, since the forward pass makes one multiply-add per weight.
The memory of training is dominated by the activations $h_l$ and $z_l$ that the backward pass needs, $B \sum_l n_l$ numbers for each kind.

## Training

The network is trained by gradient descent on a loss. The reference uses half the mean squared error over the batch,

$$ E = \frac{1}{2B} \sum_{b=1}^{B} \lVert y_b - t_b \rVert^2 , $$ {#eq-loss}

and classification uses the cross-entropy of a softmax; the backward pass below only needs $\partial E / \partial y$.

### Backpropagation

[[al:backpropagation|Backpropagation]] is the chain rule applied layer by layer from the output down. Write $\delta_l = \partial E / \partial z_l$ for the error of the pre-activations of layer $l$, with $\delta_L = \partial E / \partial y$. For one example, as column vectors,

$$ \delta_{l-1} = \big( W_l^\top \delta_l \big) \odot \sigma'(z_{l-1}) , $$ {#eq-delta}

and the gradients of the parameters are outer products summed over the batch,

$$ \frac{\partial E}{\partial W_l} = \sum_{b} \delta_{l,b}\, h_{l,b}^\top , \qquad \frac{\partial E}{\partial b_l} = \sum_{b} \delta_{l,b} . $$ {#eq-grads}

The recursion of eq-delta follows because $z_{l-1}$ reaches the loss only through $h_l$ and then $z_l$. {#c-delta}
Eq-grads follows because each weight enters every example with the input of its layer as the derivative. {#c-grads}
The derivation `backpropagation` proves both and its check compares them with finite differences.
A step of gradient descent with learning rate $\eta$ then updates every parameter,

$$ \theta \leftarrow \theta - \eta\, \frac{\partial E}{\partial \theta} , $$ {#eq-sgd}

and [[m2:stochastic-gradient-descent|stochastic gradient descent]] computes the gradient on a random minibatch instead of the whole data.

### Initialization

The weights must start random, since units with equal weights in a layer receive equal gradients and stay equal; the scale of the random draw decides whether the signal survives many layers.
For a unit with $n_l$ inputs, zero-mean independent weights and zero biases,

$$ \operatorname{Var}[z_l] = n_l \operatorname{Var}[w_l]\, \mathbb{E}[h_l^2] , $$ {#eq-var}

and for a ReLU layer whose input is symmetric about zero, $\mathbb{E}[h_l^2] = \tfrac{1}{2} \operatorname{Var}[z_{l-1}]$. {#c-variance-recursion}
The variance is therefore kept from layer to layer exactly when $\operatorname{Var}[w_l] = 2 / n_l$, which is [[al:he-initialization|He initialization]]:

$$ w_l \sim \mathcal{N}\!\left(0, \frac{2}{n_l}\right), \qquad b_l = 0 . $$ {#eq-he}

That is the condition He et al. derive for rectifiers, $\tfrac{1}{2} n_l \operatorname{Var}[w_l] = 1$, a zero-mean Gaussian with standard deviation $\sqrt{2 / n_l}$ and zero biases. {#c-he-init}
With half that variance each ReLU layer halves the signal, $2^{-m}$ after $m$ layers. {#c-half-variance}
He et al. note that the derivation of Glorot and Bengio assumes linear activations, which is invalid for the ReLU {#c-he-xavier-linear}, and find that their own initialization lets a 30-layer ReLU model converge while the Glorot initialization stalls it completely, with diminishing gradients. {#c-he-30-layers}

[[al:xavier-initialization|Xavier initialization]], the normalized initialization of Glorot and Bengio, comes from asking for constant variance both forward and backward: the forward condition is $n_l \operatorname{Var}[W] = 1$ and the backward one $n_{l+1} \operatorname{Var}[W] = 1$, and as a compromise they take $\operatorname{Var}[W] = 2 / (n_l + n_{l+1})$, drawn from a uniform distribution. {#c-glorot-init}

$$ w_l \sim U\!\left[ -\sqrt{\frac{6}{n_l + n_{l+1}}},\ \sqrt{\frac{6}{n_l + n_{l+1}}} \right] $$ {#eq-glorot}

The uniform distribution on $[-a, a]$ has variance $a^2 / 3$, so this bound gives exactly that variance. {#c-glorot-variance}
The derivation `variance-propagation` proves eq-var and these conditions, and its check measures them on networks of width 512.
In the backward pass the same arithmetic applies with $n_{l+1}$ in place of $n_l$, and a product of many factors below 1 is the [[al:vanishing-gradient-problem|vanishing gradient problem]] of deep plain networks.

## Inductive biases

An MLP assumes almost nothing about its input. Each unit reads every input, so a permutation of the input features, applied together with the same permutation of the columns of $W_0$, gives the same function: the network has no notion of which features are neighbours.
That is a strength for tabular data, where the columns have no order, and a weakness for images, sound and text, where it must learn from data what a convolution or an attention layer builds in.

Depth is the one structural assumption, and the theory supports it. Eldan and Shamir prove that a simple, approximately radial function on $\mathbb{R}^d$ that a small 3-layer network expresses cannot be approximated by any 2-layer network to more than a constant accuracy unless its width is exponential in the dimension, for virtually all known activations. {#c-eldan}
Telgarsky proves that for every positive integer $k$ there are networks with $\Theta(k^3)$ layers, $\Theta(1)$ nodes per layer and $\Theta(1)$ distinct parameters that networks with $O(k)$ layers cannot approximate unless they have $\Omega(2^k)$ nodes, for ReLU, maximization and other semi-algebraic gates. {#c-telgarsky}
His key idea is that a few compositions build a highly oscillatory function, which a shallow network can only match with many pieces. {#c-telgarsky-oscillation}
Width cannot be cut to nothing in exchange. Kidger and Lyons prove that networks of arbitrary depth and width $n + m + 2$, for $n$ inputs and $m$ outputs, are dense in $C(K; \mathbb{R}^m)$ for any nonaffine continuous activation with a continuous nonzero derivative at some point, polynomial activations included. {#c-deep-narrow}
For the ReLU, Park et al. prove that the minimum width for universal approximation of $L^p$ functions is exactly $\max\{d_x + 1, d_y\}$, and that the same does not hold for uniform approximation. {#c-min-width}

## Evidence

The architecture rests on the approximation theorems above and on experience more than on any single benchmark.
On the training side, the evidence that matters is about depth: Glorot and Bengio write that before 2006 deep multilayer networks appear not to have been trained successfully, and that the later successes all came with new initialization or training mechanisms. {#c-before-2006}
The ReLU (Glorot et al. 2011) and the initializations of Glorot and Bengio (2010) and He et al. (2015) are two of those mechanisms; residual connections and normalization layers are the others.
He et al. also find that deeper is not automatically better: their 30-layer model, though it converges, has 38.56/16.59 top-1/top-5 error on ImageNet against 33.82/13.34 for their 14-layer model. {#c-he-depth-degrades}
These are reported results on convolutional networks, whose dense layers are MLPs; this page has not reproduced them.

## Limitations

- The parameter count grows with the product of adjacent widths, and a dense first layer on raw high-dimensional input is large. The network has no built-in invariance, so it must see translated, rotated or reordered versions of the data to treat them alike.
- The universal approximation theorem is an existence result. Eldan and Shamir's separation means a shallow network may need a width exponential in the dimension, and nothing in the theorem says gradient descent will find the weights.
- Plain deep MLPs are hard to train: saturating activations, a badly scaled initialization or many layers make the gradient vanish or explode. In the experiments of Glorot and Bengio the saturation of sigmoid units can last very long in deeper networks, and their depth-five model never escaped it during training. {#c-sigmoid-depth-five}
- The loss is not convex in the weights, and the hidden units can be permuted without changing the function, so there are many equivalent minima and no guarantee of reaching the global one.

## Variants and descendants

- **Residual MLPs** add the input of each block to its output, $h_{l+1} = h_l + F(h_l)$, which keeps an identity path for the gradient.
- **Normalized MLPs** put a batch or layer normalization in each block.
- **Dropout** zeroes random hidden units during training as a regularizer.
- **The feedforward sublayer of the [[transformer|Transformer]]** is a two-layer MLP applied at each position separately, with a hidden width a few times the model width; gated variants multiply two linear maps of the input, one passed through the activation.
- **Convolutional networks** are MLPs whose weight matrices are constrained to be local and shared; the final layers of a classic convolutional network are an MLP on the flattened feature maps.

## Notable models

There is no single landmark MLP model. The MLP is the classifier head of LeNet, AlexNet and VGG, the feedforward half of every Transformer block, and the whole model in many tabular and reinforcement learning systems.

## Implementations

The NumPy reference `impl/numpy/multilayer_perceptron.py` computes eq-layer, eq-output, eq-loss, the backward pass of eq-delta and eq-grads, the step of eq-sgd, the initializations eq-he and eq-glorot, and the counts eq-params and eq-flops, with five activations: ReLU, tanh, sigmoid, GELU and the identity.
The PyTorch version `impl/pytorch/multilayer_perceptron.py` builds the same network from `torch.nn.Linear` with a `from_params` constructor and is tested with `gradcheck`; the JAX version `impl/jax/multilayer_perceptron.py` is a pure function differentiated with `jax.grad` and tested under `jax.jit`.
All three are tested against the shared fixture of `fixtures/`: a tanh network of the example config with its outputs, loss and every gradient.

### Worked example

Take the 2-2-1 ReLU network with

$$ W_0 = \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}, \quad b_0 = \begin{pmatrix} 0 \\ -1 \end{pmatrix}, \quad W_1 = \begin{pmatrix} 1 & -2 \end{pmatrix}, \quad b_1 = 0 . $$

Both hidden units see $x_1 + x_2$; the second is shifted down by 1. On the four points of XOR:

| $x$ | $z_0$ | $h_1$ | $y$ |
|---|---|---|---|
| (0, 0) | (0, −1) | (0, 0) | 0 |
| (0, 1) | (1, 0) | (1, 0) | 1 |
| (1, 0) | (1, 0) | (1, 0) | 1 |
| (1, 1) | (2, 1) | (2, 1) | 0 |

So $y = \operatorname{relu}(x_1 + x_2) - 2 \operatorname{relu}(x_1 + x_2 - 1)$ is XOR on $\{0, 1\}^2$. {#c-xor-network}
The first unit counts the ones, the second fires only when both are on, and the output subtracts twice the second from the first.
The derivation `xor` proves this and the impossibility for a linear model, and the test `test_worked_example` checks the table in all three implementations.

### Pitfalls

> [!pitfall]
> **No nonlinearity.** Without an activation between the layers the network collapses to one affine map. The test `test_linear_collapse` checks that an identity-activation network equals the product of its matrices, and `test_nonlinearity_breaks_collapse` that a ReLU network is not affine.

> [!pitfall]
> **The wrong initialization scale.** With variance $1 / n$ in a deep ReLU network each layer halves the signal. The test `test_he_variance_preserved` checks that eq-he keeps the variance within 15% per layer over ten layers, and `test_half_variance_shrinks` that half of it gives the factor $2^{-9}$.

> [!pitfall]
> **A backward pass that disagrees with the forward pass.** Hand-written gradients are easy to get wrong in a transpose or a missing $\sigma'$. The test `test_gradient_finite_differences` compares every gradient with finite differences for each activation.

> [!pitfall]
> **Rows that see each other.** A layer must act on each example of the batch alone. The test `test_batch_independence` checks that running a batch equals running its rows one at a time.

## Experiments

This page has no experiment yet. The checks of its five derivations play the part of small experiments: they solve the XOR least-squares problem, collapse linear networks, count parameters and multiply-adds, compare backpropagation with finite differences, and measure the variance of the signal through 12 ReLU layers of width 512.

## History

Schmidhuber names the networks of the Group Method of Data Handling of Ivakhnenko and Lapa (1965) as perhaps the first deep learning systems of the feedforward multilayer perceptron type, grown layer by layer by regression, and a 1971 paper already described one with 8 layers. {#c-hist-gmdh}
In his account a 1969 book of Minsky and Papert on the limitations of single-layer linear perceptrons discouraged some researchers from further studying neural networks. {#c-hist-minsky}
Efficient error backpropagation appeared in Linnainmaa's 1970 master's thesis without reference to neural networks; the first application to neural networks he knows of is Werbos (1981), and Rumelhart, Hinton and Williams (1986) popularised it by demonstrating useful internal representations in the hidden layers. {#c-hist-backprop}
Training networks with many hidden layers stayed hard until the late 2000s; the rectifier and the scaled initializations of 2010 to 2015 made deep MLPs trainable from scratch with plain backpropagation.

## References

- P. Kidger and T. Lyons. Universal Approximation with Deep Narrow Networks. COLT 2020.
- S. Park, C. Yun, J. Lee and J. Shin. Minimum Width for Universal Approximation. ICLR 2021.
- R. Eldan and O. Shamir. The Power of Depth for Feedforward Neural Networks. COLT 2016.
- M. Telgarsky. Benefits of Depth in Neural Networks. COLT 2016.
- X. Glorot and Y. Bengio. Understanding the Difficulty of Training Deep Feedforward Neural Networks. AISTATS 2010.
- X. Glorot, A. Bordes and Y. Bengio. Deep Sparse Rectifier Neural Networks. AISTATS 2011.
- K. He, X. Zhang, S. Ren and J. Sun. Delving Deep into Rectifiers. ICCV 2015.
- D. Hendrycks and K. Gimpel. Gaussian Error Linear Units (GELUs). arXiv 1606.08415, 2016.
- J. Schmidhuber. Deep Learning in Neural Networks: An Overview. Neural Networks 61, 2015.
