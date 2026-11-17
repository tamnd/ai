---
tag: GNXZCP
kind: architecture
slug: convolutional-neural-network
name: Convolutional neural network
aliases: [CNN, ConvNet, convolutional network, convolutional net, LeNet]
area: deep-learning-architectures/convolutional-networks
level: core
summary: >-
  A neural network for images and other grids whose layers slide one small kernel of shared weights over every position, so that each unit sees a local window and a feature is found anywhere.
uses: [backpropagation@al, relu@al, softmax-regression@al, matrix-multiplication@m2]
related_to: [resnet, dropout, transformer, batch-normalization@al]
history:
  year: 1989
  people: [Kunihiko Fukushima, Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner, Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton, Karen Simonyan, Andrew Zisserman]
  note: "Fukushima's neocognitron (1980) had local, replicated feature detectors without a supervised learning procedure; LeCun and colleagues trained convolutional networks with back-propagation on zip codes from 1989; LeNet-5 (LeCun, Bottou, Bengio and Haffner, Proceedings of the IEEE, 1998) fixed the standard form; AlexNet (2012) and VGG (2014) scaled it to ImageNet."
prereqs: [backpropagation@al, relu@al, stochastic-gradient-descent@m2]
same_as:
  wikipedia: Convolutional_neural_network
sources:
  - {paper: "LeCun, Bottou, Bengio and Haffner (1998)", title: "Gradient-Based Learning Applied to Document Recognition", locator: "Proceedings of the IEEE 86(11), pp. 5 to 8, 10, 11, 13, 15", note: "the three ideas, LeNet-5, its parameter counts and its MNIST results, LeNet-1"}
  - {paper: "Fukushima (1980)", title: "Neocognitron: A Self-organizing Neural Network Model for a Mechanism of Pattern Recognition Unaffected by Shift in Position", locator: "Biological Cybernetics 36, the abstract and p. 3"}
  - {paper: "LeCun, Boser, Denker, Henderson, Howard, Hubbard and Jackel (1989)", title: "Backpropagation Applied to Handwritten Zip Code Recognition", locator: "Neural Computation 1(4)", note: "listed for the history; no text copy was read, and the page cites LeCun et al. (1998) for the 1989 work"}
  - {paper: "Krizhevsky, Sutskever and Hinton (2012)", title: "ImageNet Classification with Deep Convolutional Neural Networks", locator: "the abstract, Sections 1, 3.4, 3.5 and 7"}
  - {paper: "Simonyan and Zisserman (2015)", title: "Very Deep Convolutional Networks for Large-Scale Image Recognition", locator: "arXiv 1409.1556v6, Sections 2.1 and 2.3, Table 2"}
  - {paper: "Zhang (2019)", title: "Making Convolutional Networks Shift-Invariant Again", locator: "arXiv 1904.11486v2, the abstract, Sections 1 and 3.1"}
  - {paper: "Luo, Li, Urtasun and Zemel (2016)", title: "Understanding the Effective Receptive Field in Deep Convolutional Neural Networks", locator: "arXiv 1701.04128v2, the abstract, Section 1, p. 4"}
family: convolutional
inputs: "an image as a (B, C, R, R) array, R - 12 a positive multiple of 4"
outputs: "class logits as a (B, K) array"
components: [relu@al, softmax-regression@al]
inductive_biases: ["locality: each unit depends on a small window of the layer below", "weight sharing: one kernel is applied at every position, so a shift of the input shifts the feature maps", "a hierarchy of feature maps whose resolution falls and whose receptive field grows with depth"]
params: "25 C C1 + C1 + 25 C1 C2 + C2 + C2 ((R - 12)/4)^2 F1 + F1 + F1 F2 + F2 + F2 K + K"
flops_per_example: "2 (25 C C1 (R - 4)^2 + 25 C1 C2 ((R - 12)/2)^2 + C2 ((R - 12)/4)^2 F1 + F1 F2 + F2 K), counting a multiply-add as 2 FLOPs and leaving out biases, ReLUs and pooling"
models: [LeNet-1, LeNet-5, AlexNet, VGG-16, VGG-19]
---

A convolutional neural network, CNN or ConvNet, is a neural network whose main layers are convolutions: each layer slides a small kernel of weights over every position of its input and computes the same weighted sum of a local window everywhere.
The same feature detector is reused across the whole image, so the layer has few parameters, and a shifted input gives shifted feature maps.
Stacked with nonlinearities and pooling, such layers build a hierarchy in which early units see small patches and later units see most of the image.
Fukushima's neocognitron had this structure in 1980; LeCun and colleagues trained it with back-propagation from 1989, and LeNet-5 (1998), AlexNet (2012) and VGG (2014) made it the standard architecture for vision.

## In plain words

To find a cat's ear in a photograph, it does not matter where the ear is: the same pattern of edges means "ear" in the top left as in the bottom right.
A convolutional layer builds this in. It has a small template, say 3 by 3 pixels, and slides it across the image; at each position it scores how well the patch below matches. The scores form a feature map, bright where the pattern occurs.
A layer has many templates and so produces many feature maps, and the next layer slides its templates over those maps, finding patterns of patterns.
Pooling shrinks the maps by keeping the strongest response in each small block, so later units summarise larger parts of the image.
The templates are not designed by hand; they are the weights of the network, learned by gradient descent like any other.

## Overview

LeCun et al. name three ideas that together give a convolutional network some invariance to shifts, scaling and distortions: local receptive fields, shared weights, and sub-sampling in space or time. {#c-three-ideas}
The motivation is that a fully connected network has no built-in invariance to translations or local distortions {#c-no-builtin-invariance}, and it ignores the topology of the input, while nearby pixels are highly correlated. {#c-topology}
Krizhevsky et al. put the same point as assumptions: CNNs assume stationarity of statistics and locality of pixel dependencies, and so have much fewer connections and parameters than feedforward networks with similarly-sized layers. {#c-alexnet-priors}

A typical network alternates convolutions, pointwise nonlinearities and pooling, which reduce the resolution and increase the number of feature maps, and ends with one or more fully connected layers that produce class scores.
This page covers the convolutional layer, its arithmetic and its symmetry. The page [[resnet]] covers residual connections, [[al:batch-normalization|batch normalization]] the normalization used in modern CNNs, [[dropout]] the regularizer that AlexNet used in its fully connected layers, and [[transformer]] the attention architecture that competes with CNNs in vision.

## Components

### The convolutional layer

A layer arranges its units in planes, and every unit of one plane uses one shared set of weights; what a plane outputs is a feature map, and each layer holds several maps, each with its own weights. {#c-feature-map}
Sliding one unit over the input and recording its state at each place amounts to a convolution, plus a bias, passed through a squashing nonlinearity; the shared weights of the map are the convolution kernel, hence the name. {#c-equivalent-convolution}
In LeNet-5, a unit of the first layer has 25 inputs from a 5 by 5 area, its receptive field, and so 25 coefficients and a bias, shared by all the units of the map. {#c-lenet-c1-unit}

With an input $x$ of $C$ channels, a kernel $w$ of $D$ output channels and size $k \times k$, a stride $s$ and a bias $b$, output channel $d$ at position $(i, j)$ is

$$ y_{d, i, j} = b_d + \sum_{c=1}^{C} \sum_{u=0}^{k-1} \sum_{v=0}^{k-1} w_{d, c, u, v}\; x_{c,\, s i + u,\, s j + v} . $$ {#eq-conv}

Deep learning libraries compute this cross-correlation and call it convolution. A true convolution indexes the input as $x_{i - u}$; it equals the cross-correlation with the kernel flipped, shifted by $k - 1$, so the two define the same family of layers and learning absorbs the flip. {#c-cross-correlation}

**Padding** adds $p$ positions around the input, zeros or a wrap-around copy (circular padding), so that windows can be centred on the border. **Stride** $s > 1$ evaluates only every $s$-th window. With both, an axis of $n$ positions gives

$$ n' = \left\lfloor \frac{n + 2p - k}{s} \right\rfloor + 1 $$ {#eq-out-size}

positions. {#c-out-size}
With stride 1 and odd $k$, padding $p = (k-1)/2$ keeps the size: $p = 1$ for $k = 3$. {#c-same-padding}
The floor matters: AlexNet's first layer applies 96 kernels of $11 \times 11 \times 3$ with stride 4 to a $224 \times 224 \times 3$ image {#c-alexnet-first-layer}, and the formula gives 54 positions, not the 55 of the familiar figure; 55 needs a 227-pixel input or a padding of 2. {#c-alexnet-size}
The derivation `output-size` proves the formula by counting the windows that fit.

### Pooling

A pooling layer replaces each window by a summary, channel by channel and without weights. Max pooling with window $z$ and stride $s$ is

$$ y_{c, i, j} = \max_{0 \le u, v < z} x_{c,\, s i + u,\, s j + v} , $$ {#eq-maxpool}

and its output size is eq-out-size with $k = z$. LeNet-5 used sub-sampling layers that average and sub-sample, which lowers the resolution and makes the output less sensitive to small shifts and distortions. {#c-subsampling}
In AlexNet, overlapping max pooling ($z = 3$, $s = 2$) lowered the top-1 and top-5 errors by 0.4% and 0.3% against $z = s = 2$. {#c-overlap-pool}
Global pooling takes the maximum or mean over all positions of a map and gives one number per channel.

### The network

LeNet-5 has 7 layers with weights after a $32 \times 32$ input: C1 with 6 maps of $28 \times 28$, S2 with 6 maps of $14 \times 14$, C3 with 16 maps, S4 with 16 maps of $5 \times 5$, C5 with 120 maps, F6 with 84 units, and one radial basis function unit per class. {#c-lenet5}
With 5x5 kernels without padding and $2 \times 2$ sub-sampling, the sizes are 32, 28, 14, 10, 5 and 1. {#c-lenet-sizes}
The shape contract of this page is LeNet-5 with the substitutions of later practice: ReLU instead of the scaled tanh, max pooling instead of trainable averaging, a full connection in the second convolution, and a linear output layer.

## Shapes and parameters

The sizes are the input channels $C$, the maps $C_1$ and $C_2$ of the two convolutions, the widths $F_1$ and $F_2$ of the fully connected layers, the classes $K$, the batch $B$ and the image side $R$.
A $k \times k$ convolution from $C_\text{in}$ to $C_\text{out}$ channels has

$$ P_\text{conv} = k^2 C_\text{in} C_\text{out} + C_\text{out} $$ {#eq-conv-params}

parameters, whatever the size of its input. {#c-conv-params}
The network has

{shape:params}

parameters, {shape:params@lenet5} for the sizes of LeNet-5.
LeCun et al. report 156 parameters and 122,304 connections for C1 {#c-lenet-reported}, and 1,516 parameters and 151,600 connections for C3, 48,120 for C5 and 10,164 for F6. {#c-lenet-reported-later}
The counting rule reproduces each one; the C3 of the paper connects each map to only some maps of S2, and a full connection would have 2,416. {#c-lenet-params}
The contract's total exceeds the paper's 60,000 by that difference and by the linear output layer.

Sharing is where the savings come from. C1 has 156 parameters for 122,304 connections; a fully connected layer from the $32 \times 32$ image to the same six $28 \times 28$ maps would have 4,821,600. {#c-dense-comparison}
LeCun et al. note that the sharing reduces the capacity and the gap between test and training error: the whole network has 340,908 connections but only 60,000 free parameters. {#c-weight-sharing-capacity}

## Computation and cost

Each output entry of a convolution costs $k^2 C_\text{in}$ multiply-adds, so a layer with an $n' \times n'$ output costs $n'^2 C_\text{out} k^2 C_\text{in}$ of them, quadratic in the side of the map and in the width.
In the shape contract, the forward pass of LeNet-5 on one image is {shape:flops_forward@lenet5} FLOPs, counting a multiply-add as two.
The two convolutions take 715,200 FLOPs, 86%, with 4% of the parameters; the fully connected layers hold 96% of the parameters (these shares are computed from the contract). The pattern is general: convolutions dominate the compute and dense layers the parameters.
AlexNet has 60 million parameters and 650,000 neurons in five convolutional and three fully connected layers {#c-alexnet}, and the VGG networks have 133 to 144 million. {#c-vgg-size}

In practice the windows are gathered into a matrix, one row per output position (im2col), and the layer becomes one matrix multiplication by the kernel reshaped to $(C_\text{out}, k^2 C_\text{in})$. The NumPy reference does this with a strided view, without copying the input.

## Training

The network is trained end to end with [[al:backpropagation|backpropagation]]. The gradient of eq-conv with respect to a kernel entry sums over all positions where it was applied, $\partial L / \partial w_{d,c,u,v} = \sum_{i,j} \delta_{d,i,j}\, x_{c, si+u, sj+v}$, which is how weight sharing appears in learning; the gradient with respect to the input is a convolution of the output gradient with the flipped kernel.
The reference implementation has this backward pass and checks it against finite differences.
On MNIST, LeNet-5's test error stabilised after about 10 passes at 0.95%, and 0.8% with distorted training images. {#c-lenet5-error}

## Inductive biases

### Equivariance and invariance

LeCun et al. note that if the input is shifted, the feature map is shifted by the same amount and otherwise unchanged, and place the robustness of convolutional networks to shifts on this property. {#c-shift-feature-map}
The precise statement is a commutation. Let $T_t$ shift a map by $t$ positions. With stride 1 and circular padding,

$$ (T_t x) \star w = T_t (x \star w), $$ {#eq-equivariance}

since $\sum_u w_u x_{i + u - t}$ is the old output at $i - t$. A pointwise nonlinearity commutes with $T_t$ as well, so any stack of such layers is equivariant. {#c-equivariance}
Real networks use zero padding or none. Without padding, every output whose window lies inside the shifted part of the input moves with it; with zero padding the outputs whose window crosses the border do not, since the zeros stay put while the signal moves. {#c-crop-equivariance}

Strides and pooling weaken the symmetry. A layer with stride $s$ commutes only with shifts by multiples of $s$, and a shift by $st$ moves its output by $t$; the same holds for pooling. {#c-stride-periodic}
After $L$ layers of stride 2, only shifts by multiples of $2^L$ are guaranteed to move the output cleanly.
Invariance comes at the end: a global maximum or mean ignores the order of the positions, so global pooling after equivariant layers gives a shift-invariant output. {#c-global-pool-invariance}
Local pooling gives only an approximate invariance: a small shift changes which entry wins a window less often than it changes the entries themselves.
The derivation `equivariance` proves these statements and its check tests them on random images.

### The receptive field

A unit's receptive field is the patch of the input it depends on. Let $r_l$ be its side at layer $l$ and $j_l$ the distance in input pixels between adjacent units. With $r_0 = j_0 = 1$,

$$ r_l = r_{l-1} + (k_l - 1)\, j_{l-1}, \qquad j_l = j_{l-1}\, s_l , $$ {#eq-rf}

and the field has no gaps when $s_l \le k_l$. {#c-rf-recursion}
With stride 1, $m$ layers of size $k$ see $1 + m(k-1)$ inputs, so the field grows linearly with depth: two $3 \times 3$ layers see $5 \times 5$ and three see $7 \times 7$. {#c-rf-linear}
Strides multiply the growth: in LeNet-5 the fields of C1, S2, C3, S4 and C5 are 5, 6, 14, 16 and 32, so each unit of C5 sees the whole image. {#c-rf-lenet}
The derivation `receptive-field` proves the recursion, and its check measures fields as the support of a gradient.

### Small kernels

Simonyan and Zisserman built VGG from $3 \times 3$ convolutions with stride 1 and padding 1, five $2 \times 2$ max poolings with stride 2, and three fully connected layers. {#c-vgg-design}
A stack of two $3 \times 3$ layers has the effective receptive field of a $5 \times 5$ and three of a $7 \times 7$, with three rectifications instead of one, which makes the decision function more discriminative. {#c-vgg-stack}
It also has fewer weights: with $C$ channels and no biases, $18C^2$ against $25C^2$ for two layers, and $27C^2$ against $49C^2$ for three, 81% more for the single layer. {#c-vgg-params}
They see this as a regularisation of the $7 \times 7$ filters, forced to decompose through $3 \times 3$ ones with nonlinearities in between. {#c-vgg-regularisation}

## Evidence

The architecture is supported by results more than by theory.
LeNet-5 reached 0.95% test error on MNIST (above). AlexNet had top-1 and top-5 errors of 37.5% and 17.0% on the ILSVRC-2010 test set, and a variant won ILSVRC-2012 with 15.3% top-5 error against 26.2% for the second entry. {#c-alexnet-results}
Removing any middle convolutional layer of AlexNet cost about 2% top-1, which Krizhevsky et al. read as evidence that depth matters. {#c-alexnet-depth}
These are reported results; this page has not reproduced them.

## Limitations

- Shift invariance is weaker than the symmetry suggests. Zhang reports that modern CNNs are not shift-invariant, small shifts can change the output drastically, and max pooling, strided convolution and average pooling ignore the sampling theorem. {#c-not-invariant}
  Max pooling is a dense max followed by naive subsampling, and Zhang low-pass filters between the two. {#c-maxpool-decomposition}
  Integrated this way, anti-aliasing increased ImageNet accuracy across ResNet, DenseNet and MobileNet and improved robustness to corruptions. {#c-blurpool}
- The theoretical receptive field overstates what a unit uses. Luo et al. find that the effective receptive field is Gaussian and covers only a fraction of the theoretical one, and that it starts small at initialization and grows in training. {#c-erf}
  In their analysis it grows as $O(\sqrt{n})$ with depth while the theoretical field grows linearly, so the ratio shrinks as $O(1/\sqrt{n})$. {#c-erf-sqrt}
- Convolution is equivariant to translations only. Rotations and changes of scale must be learned from data or augmentation.
- The cost grows with the square of the image side, and long-range interactions need depth or pooling; attention layers ([[transformer]]) connect all positions in one step instead.

## Variants and descendants

Residual networks ([[resnet]]) add identity shortcuts so that networks of a hundred layers train. Other variants change the layer itself: $1 \times 1$ convolutions mix channels only, depthwise convolutions use one kernel per channel, dilated convolutions space the kernel taps to enlarge the receptive field, and transposed convolutions upsample. The same layer applies to one-dimensional signals such as audio and to three-dimensional volumes.

## Notable models

LeNet-1 and LeNet-5 read handwritten digits; AlexNet (model card `alexnet`, GW532P) and VGG-16 and VGG-19 (cards `vgg16` and `vgg19`) are the ImageNet networks of 2012 and 2014. LeNet-5 also has a card, `lenet-5`.

## Implementations

The NumPy reference `impl/numpy/cnn.py` computes eq-conv with im2col and an einsum, with stride, zero or circular padding and several channels, and has an explicit six-loop version as the test oracle, the backward pass, max pooling, eq-out-size, eq-rf, eq-conv-params and the network of the shape contract.
The PyTorch version `impl/pytorch/cnn.py` has the same layer through `unfold` and a `torch.nn.Module` with a `from_params` constructor, tested against `torch.nn.functional.conv2d`, `nn.Conv2d` with circular padding and `gradcheck`. The JAX version `impl/jax/cnn.py` uses `lax.conv_general_dilated` and `jax.grad`.
All three are tested against the fixture of `fixtures/`: a small network with random weights, and one convolution with stride 2 and padding 1 in both padding modes.

### Worked example

Take the $3 \times 3$ input with entries $0, 1, \dots, 8$ row by row and the $2 \times 2$ kernel $w = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$.
With stride 1 and no padding, eq-out-size gives $\lfloor (3 - 2)/1 \rfloor + 1 = 2$, and the top-left output is $0 \cdot 1 + 1 \cdot 2 + 3 \cdot 3 + 4 \cdot 4 = 27$; the full output is $\begin{pmatrix} 27 & 37 \\ 57 & 67 \end{pmatrix}$.
The true convolution, with the kernel flipped, gives $\begin{pmatrix} 13 & 23 \\ 43 & 53 \end{pmatrix}$.
With padding 1 and stride 2, $\lfloor (3 + 2 - 2)/2 \rfloor + 1 = 2$ again, and the output is $\begin{pmatrix} 0 & 11 \\ 30 & 67 \end{pmatrix}$: the top-left window holds three zeros and the input 0, and the top-right holds $1 \cdot 3 + 2 \cdot 4 = 11$.
The tests of all three implementations check these numbers.

## Experiments

This page has no experiment yet. The checks of its four derivations play the part of small experiments: they measure output sizes, parameter counts, receptive fields and equivariance on random inputs with the reference implementation.

## History

Fukushima's neocognitron of 1980 was a cascade of S-cells, like simple cells, and C-cells, like complex cells, self-organized without a teacher, whose last layer responded to a pattern regardless of its position. {#c-neocognitron}
All cells of a cell-plane had input synapses of the same spatial distribution, shifted in parallel from cell to cell: weight sharing. {#c-neocognitron-planes}
LeCun et al. credit it with the convolution and sub-sampling combination, but no globally supervised learning procedure such as back-propagation was available then. {#c-neocognitron-no-backprop}
In 1989 LeCun and colleagues trained convolutional networks with back-propagation to read zip codes. LeNet-1 had only about 2,600 free parameters {#c-lenet1}, and LeCun et al. call it appropriate to the technology of 1989, when a network like LeNet-5 would have needed weeks of training and more data than there was. {#c-lenet1-1989}
LeNet-5 followed in 1998, with the form that later networks kept.
Convolutional networks then waited for data and compute: AlexNet won ILSVRC-2012 by a wide margin, and VGG (2014) went deeper with small kernels.
ResNet (2015) took the depth to a hundred layers and more.

## References

- Y. LeCun, L. Bottou, Y. Bengio and P. Haffner. Gradient-Based Learning Applied to Document Recognition. Proceedings of the IEEE, 1998.
- K. Fukushima. Neocognitron. Biological Cybernetics 36, 1980.
- Y. LeCun et al. Backpropagation Applied to Handwritten Zip Code Recognition. Neural Computation, 1989.
- A. Krizhevsky, I. Sutskever and G. E. Hinton. ImageNet Classification with Deep Convolutional Neural Networks. NIPS 2012.
- K. Simonyan and A. Zisserman. Very Deep Convolutional Networks for Large-Scale Image Recognition. ICLR 2015.
- R. Zhang. Making Convolutional Networks Shift-Invariant Again. ICML 2019.
- W. Luo, Y. Li, R. Urtasun and R. Zemel. Understanding the Effective Receptive Field in Deep Convolutional Neural Networks. NIPS 2016.
