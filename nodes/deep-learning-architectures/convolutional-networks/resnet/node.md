---
tag: NEW-resnet
kind: architecture
slug: resnet
name: ResNet
aliases: [residual network, residual neural network, deep residual network, deep residual learning]
area: deep-learning-architectures/convolutional-networks
level: core
summary: >-
  A deep convolutional network of residual blocks, each adding its input to the output of a few layers through a shortcut, so that networks of a hundred or more layers train as well as shallow ones.
uses: [batch-normalization@al, relu@al, backpropagation@al, he-initialization@al, matrix-multiplication@m2, binomial-distribution@m2]
related_to: [dropout, vanishing-gradient-problem@al, exploding-gradient-problem@al, layer-normalization@al, gradient-checkpointing@al, stochastic-gradient-descent@m2]
history:
  year: 2015
  people: [Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun]
  note: "He et al., arXiv 1512.03385 (10 Dec 2015), from Microsoft Research; published at CVPR 2016."
prereqs: [batch-normalization@al, relu@al, backpropagation@al, stochastic-gradient-descent@m2]
same_as:
  wikipedia: Residual_neural_network
  arxiv: ["1512.03385"]
sources:
  - {paper: "He, Zhang, Ren and Sun (2016)", title: "Deep Residual Learning for Image Recognition", locator: "arXiv 1512.03385v1, Sections 1 to 4, Tables 1 to 8", note: "the original paper: the degradation problem, the residual block, the bottleneck networks and the results"}
  - {paper: "He, Zhang, Ren and Sun (2016)", title: "Identity Mappings in Deep Residual Networks", locator: "arXiv 1603.05027v3, Sections 2 and 3, Table 4", note: "the propagation analysis and the pre-activation unit"}
  - {paper: "Srivastava, Greff and Schmidhuber (2015)", title: "Highway Networks", locator: "arXiv 1505.00387v2, Section 3.1", note: "gated shortcuts, and the degradation of plain networks on MNIST"}
  - {paper: "He, Zhang, Ren and Sun (2015)", title: "Delving Deep into Rectifiers", locator: "Section 2.2", note: "the initialization used for ResNets"}
  - {paper: "Ioffe and Szegedy (2015)", title: "Batch Normalization", locator: "the abstract"}
  - {paper: "Simonyan and Zisserman (2015)", title: "Very Deep Convolutional Networks for Large-Scale Image Recognition", locator: "the abstract"}
  - {paper: "Goyal et al. (2017)", title: "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour", locator: "Section 5.1", note: "zero-gamma initialization and the stride in the 3x3 convolution"}
  - {paper: "He, Zhang, Zhang, Zhang, Xie and Li (2019)", title: "Bag of Tricks for Image Classification with Convolutional Neural Networks", locator: "the abstract, Section 4.2"}
  - {paper: "Veit, Wilber and Belongie (2016)", title: "Residual Networks Behave Like Ensembles of Relatively Shallow Networks", locator: "the abstract, Sections 3 and 4.1"}
  - {paper: "Balduzzi et al. (2017)", title: "The Shattered Gradients Problem: If resnets are the answer, then what is the question?", locator: "the abstract"}
  - {paper: "Li, Xu, Taylor, Studer and Goldstein (2018)", title: "Visualizing the Loss Landscape of Neural Nets", locator: "Section 6"}
  - {paper: "Hardt and Ma (2017)", title: "Identity Matters in Deep Learning", locator: "the abstract"}
  - {paper: "Zhang, Dauphin and Ma (2019)", title: "Fixup Initialization: Residual Learning Without Normalization", locator: "the abstract"}
  - {paper: "Huang, Sun, Liu, Sedra and Weinberger (2016)", title: "Deep Networks with Stochastic Depth", locator: "the abstract"}
  - {paper: "Zagoruyko and Komodakis (2016)", title: "Wide Residual Networks", locator: "the abstract"}
  - {paper: "Xie, Girshick, Dollár, Tu and He (2017)", title: "Aggregated Residual Transformations for Deep Neural Networks", locator: "the abstract"}
  - {paper: "Huang, Liu, van der Maaten and Weinberger (2017)", title: "Densely Connected Convolutional Networks", locator: "the abstract"}
  - {paper: "Bello et al. (2021)", title: "Revisiting ResNets: Improved Training and Scaling Strategies", locator: "the abstract"}
  - {paper: "Wightman, Touvron and Jégou (2021)", title: "ResNet strikes back: An improved training procedure in timm", locator: "the abstract"}
  - {paper: "Liu, Mao, Wu, Feichtenhofer, Darrell and Xie (2022)", title: "A ConvNet for the 2020s", locator: "the abstract, Section 2.1"}
family: convolutional
inputs: "an image as a (B, 3, R, R) array, R a multiple of 32"
outputs: "class logits as a (B, K) array"
components: [batch-normalization@al, relu@al, softmax-regression@al]
inductive_biases: ["locality and weight sharing of convolutions", "a hierarchy of feature maps whose resolution halves and whose channels double from stage to stage", "each block starts close to the identity and learns a residual correction"]
params: "1950 C^2 + 449 C + (n1 - 1)(17 C^2 + 12 C) + (n2 - 1)(68 C^2 + 24 C) + (n3 - 1)(272 C^2 + 48 C) + (n4 - 1)(1088 C^2 + 96 C) + 32 C K + K"
flops_per_example: "C^2 R^2 (17 N + 19) / 8 + 147 C R^2 / 2 + 64 C K, with N = n1 + n2 + n3 + n4, counting a multiply-add as 2 FLOPs"
introduced_by: he-2016-deep-residual-learning
models: [ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152, ResNet-1001, ResNeXt, Wide ResNet, ResNet-RS]
---

ResNet, the residual network, is a deep convolutional network built from residual blocks: each block computes a correction $F(x)$ with a few convolutions and adds its input $x$ back through a shortcut connection, so that its output is $F(x) + x$.
He, Zhang, Ren and Sun introduced it in 2015 to fix a problem that had stopped convolutional networks from getting deeper: beyond some depth, adding layers made even the training error worse.
With shortcuts, networks of 50 to 152 layers trained without that degradation, won the ImageNet and COCO competitions of 2015, and became the default backbone of computer vision for years.
The residual connection itself spread far beyond convolutional networks; every Transformer block uses it.

## In plain words

A deep network is a long chain of layers, and each layer transforms what the previous one gives it.
One would expect a longer chain never to do worse than a shorter one, since the extra layers could simply pass their input along unchanged.
In practice it did worse: past twenty or thirty layers, plain networks fit even their training data less well.
ResNet gives every few layers a bypass.
The input goes around the layers and is added to what they compute, so the layers only need to learn what to change, and learning to change nothing is easy.
With these bypasses, networks of a hundred layers or more train well, and depth again buys accuracy.

## Overview

He et al. observed that as the depth of a plain network increases, accuracy saturates and then degrades, and that adding more layers to a suitably deep model leads to a higher training error, so the cause is not overfitting. {#c-degradation}
A deeper model should not do worse in principle: there is a solution by construction in which the added layers are identity mappings and the other layers are copied from the learned shallower model, so the deeper model should have no higher training error. The degradation suggests that solvers have difficulty approximating identity mappings by several nonlinear layers. {#c-construction-argument}

Their answer is residual learning.
Let $H(x)$ be the mapping that a few stacked layers should fit.
Instead, the layers fit the residual $F(x) := H(x) - x$, and a shortcut connection that performs the identity adds $x$ back, so the block outputs $F(x) + x$. {#c-resnet}
The hypothesis is that the residual is easier to optimize: if an identity mapping were optimal, it would be easier to push the residual to zero than to fit an identity with a stack of nonlinear layers. {#c-hypothesis}

The identity shortcut adds no parameter and no multiply-add, only an element-wise addition, so a residual network can be compared with a plain network that has exactly the same layers. {#c-no-extra-parameters}
On ImageNet, the 34-layer plain network has a higher training error than the 18-layer one; with shortcuts the 34-layer network has a considerably lower training error than its plain counterpart and a top-1 error 3.5% lower. {#c-shortcuts-fix-degradation}
The 34-layer residual network is also 2.8% better than the 18-layer one, so accuracy is again gained from depth. {#c-depth-gains}

## Components

### The residual block

The basic block of He et al. is

$$ y = F(x, \{W_i\}) + x , \qquad F = W_2\, \sigma(W_1 x) , $$

where $\sigma$ is the [[al:relu|ReLU]], the addition is element-wise, channel by channel on feature maps, and a second ReLU is applied after the addition. {#c-block}
In a convolutional network $W_1$ and $W_2$ are convolutions, each followed by batch normalization.
Writing $\mathrm{BN}$ for batch normalization,

$$ y = \operatorname{relu}\big(F(x) + x\big) $$ {#eq-residual}

is the block with the identity shortcut, the one used wherever the input and the output have the same shape.
He et al. use residual functions of two or three layers; with a single layer the block is close to a linear layer $y = W_1 x + x$, for which they observed no advantage. {#c-single-layer}

When the shape changes, because the number of channels grows or the feature map shrinks, the shortcut is a linear projection $W_s$:

$$ y = \operatorname{relu}\big(F(x) + \mathrm{BN}(W_s * x)\big) , $$ {#eq-projection}

where $W_s$ is a 1x1 convolution with the stride of the block. He et al. use $W_s$ only to match dimensions, since the identity suffices against the degradation. {#c-projection}
They compare three kinds of shortcut: (A) the identity with zero-padding for the extra channels, so that every shortcut is parameter-free; (B) projections where the dimensions increase and the identity elsewhere; (C) projections everywhere. ResNet-50, -101 and -152 use option B. {#c-options}

### Convolution, normalization and the head

Every weight layer is a convolution without bias,

$$ z[o, i, j] = \sum_{c} \sum_{u, v} W[o, c, u, v]\; x[c,\, s i + u - p,\, s j + v - p] , $$ {#eq-conv}

with stride $s$ and zero padding $p$, followed by [[al:batch-normalization|batch normalization]] over the batch and the positions of each channel,

$$ \mathrm{BN}(z) = \gamma\, \frac{z - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta , $$ {#eq-bn}

with $\epsilon = 10^{-5}$ in the implementations of this page.
Batch normalization normalizes each activation over the mini-batch and then applies a learned scale and shift; with it, an image classifier of 2015 reached the same accuracy in 14 times fewer training steps. {#c-bn}
ResNet puts it right after each convolution and before the activation.

The head averages each of the $4 m_3 = 32\, C$ channels of the last feature map over its positions and applies a fully connected layer:

$$ \text{logits} = W_\text{fc}\, \frac{1}{H W} \sum_{i, j} y[:, i, j] + b . $$ {#eq-head}

### The bottleneck block

For ResNet-50, -101 and -152, each residual function is a stack of three convolutions, 1x1, 3x3 and 1x1: the first 1x1 reduces the channels, the 3x3 works at the reduced width, and the last 1x1 restores them. He et al. chose it because of the training time they could afford. {#c-bottleneck}
With mid width $m$ and output width $4m$,

$$ F(x) = \mathrm{BN}_3\Big(W_3 * \operatorname{relu}\big(\mathrm{BN}_2(W_2 * \operatorname{relu}(\mathrm{BN}_1(W_1 * x)))\big)\Big) , $$ {#eq-bottleneck}

with $W_1$ of shape $(m, 4m, 1, 1)$, $W_2$ of shape $(m, m, 3, 3)$ and $W_3$ of shape $(4m, m, 1, 1)$, except in the first block of a stage, where $W_1$ takes the channels of the stage before.
The identity shortcut matters most here: in a 256-64-256 block the three convolutions have 69,632 weights, and a projection from 256 to 256 channels would add 65,536, nearly doubling the size and the cost of the block. {#c-bottleneck-projection-doubles}

### The network

The plain baselines follow two rules taken from VGG nets: layers with the same output feature map size have the same number of filters, and when the feature map size is halved the number of filters is doubled, to preserve the time complexity per layer. Downsampling is by convolutions with stride 2, and the network ends with global average pooling and a 1000-way fully connected layer. {#c-design-rules}
ResNet-50, -101 and -152 start with a 7x7 convolution with stride 2 and a 3x3 max pooling with stride 2, then have four stages of 3, 4, 6 and 3; 3, 4, 23 and 3; and 3, 8, 36 and 3 bottleneck blocks with 256, 512, 1024 and 2048 output channels, and end with average pooling and a 1000-way fully connected layer with softmax. {#c-table1}
The depth counts the weighted layers: $3 N + 2$ for $N$ bottleneck blocks, the stem convolution and the fully connected layer.

The networks of He et al. for CIFAR-10 are smaller and use two-layer blocks. They have $6n + 2$ weighted layers: a 3x3 convolution, then $2n$ layers of 3x3 convolutions on each of the feature map sizes 32, 16 and 8 with 16, 32 and 64 filters, then global average pooling and a 10-way fully connected layer. They use option A everywhere, so each residual network has the same depth, width and number of parameters as its plain counterpart. {#c-cifar-arch}

## Shapes and parameters

The shape contract of this page is the bottleneck network with option B shortcuts and the stride in the first 1x1 convolution, as in the paper.
Its sizes are the base width $C$, the numbers of blocks $n_1$ to $n_4$, the number of classes $K$, the batch $B$ and the image side $R$; the example configuration is small enough for the tests to run the forward pass.
A block that is not the first of stage $s$ has mid width $2^s C$ and takes and returns $4 \cdot 2^s C$ channels on a map of side $R / 2^{s + 2}$.
The number of parameters is

{shape:params}

which gives {shape:params@resnet50} for ResNet-50, {shape:params@resnet101} for ResNet-101 and {shape:params@resnet152} for ResNet-152, counting the convolution weights, the scale and shift of each batch normalization, and the weights and bias of the fully connected layer. {#c-params}
The derivation `parameter-count` builds the formula block by block.
The same counting gives the CIFAR-10 networks $464 + 4{,}672\, n + 13{,}952 + 18{,}560\,(n - 1) + 55{,}552 + 73{,}984\,(n - 1) + 650$ parameters: 269,722 for the 20 layers of $n = 3$ and 19,421,274 for the 1202 layers of $n = 200$, the 0.27M and 19.4M of Table 6 of He et al. {#c-cifar-params}

## Computation and cost

A multiply-add is counted once below, as He et al. count FLOPs; the shape contract counts it as two operations.
A network of base width $w$ with $N$ bottleneck blocks, on an $R \times R$ image with $K$ classes, takes

$$ \frac{w^2 R^2}{16}\,(17 N + 19) + \frac{147\, w R^2}{4} + 32\, w K $$

multiply-adds: 3.86, 7.57 and 11.28 billion for ResNet-50, -101 and -152 at 224 by 224. Table 1 of He et al. gives 3.8, 7.6 and 11.3 billion, so the count of ResNet-50 is the one that does not round to the published figure. {#c-flops}
In the shape contract, the forward pass of ResNet-50 on one image is {shape:flops_forward@resnet50} FLOPs.

The design rule of doubling the channels when the map is halved makes every block but the first of a stage cost the same $17 w^2 R^2 / 16$ multiply-adds in all four stages: doubling the channels multiplies the cost per position by four, and halving the map divides the positions by four. The cost therefore grows linearly in the number of blocks, whichever stage they are added to. {#c-stage-invariance}
That is why ResNet-152, with three times the blocks of ResNet-50, costs about three times as much.
He et al. count 3.6 billion FLOPs for their 34-layer networks, 18% of the 19.6 billion of VGG-19, and 11.3 billion for ResNet-152, which is 8 times deeper than VGG nets but cheaper than VGG-16 and VGG-19, at 15.3 and 19.6 billion. {#c-complexity}

Most implementations now put the stride of 2 of a downsampling block in the 3x3 convolution instead of the first 1x1, as in the ResNet-50 of Goyal et al. {#c-v15} This tweak, called ResNet-B by He et al. (2019), first appeared in a Torch implementation; the reason is that a 1x1 convolution with stride 2 ignores three-quarters of its input feature map. {#c-resnet-b}
This keeps the parameters and raises the cost of ResNet-50 from 3.86 to 4.09 billion multiply-adds, since the first 1x1 convolution of the three downsampling blocks then runs on four times as many positions. {#c-stride-cost}
The derivation `flop-count` proves both counts and the invariance; the reference implementation supports both placements through `stride_on`.

Memory in training is dominated by the activations kept for the backward pass, which grow with the batch, the image area and the depth; the shortcut adds no stored activation beyond the block input that the first convolution already keeps.
[[al:gradient-checkpointing|Gradient checkpointing]] keeps only some of these activations and recomputes the others during the backward pass, which trades compute for memory.

## Training

The ImageNet networks of He et al. put batch normalization after each convolution and before the activation, are initialized with the [[al:he-initialization|He initialization]], and are trained from scratch with [[m2:stochastic-gradient-descent|SGD]], a mini-batch of 256, a learning rate of 0.1 divided by 10 when the error plateaus, a weight decay of 0.0001, a momentum of 0.9, and no dropout. {#c-training-recipe} Training runs for up to 600,000 iterations.
That initialization draws each weight from a zero-mean Gaussian with standard deviation $\sqrt{2 / n_l}$, where $n_l = k^2 c$ is the number of input connections of a response, and sets the biases to zero. {#c-he-init}
The 110-layer CIFAR-10 network needs a warm-up: its learning rate is 0.01 until the training error is below 80%, then 0.1.

Goyal et al. initialize the scale $\gamma$ of the last batch normalization of each block to zero, so that every block starts as the identity and the signal initially goes through the shortcuts; they found it eases optimization early in training and helps most with large minibatches. {#c-zero-gamma}
With it, a learning rate scaled linearly with the minibatch and a warm-up, ResNet-50 trains on 256 GPUs with a minibatch of 8192 in one hour and matches the accuracy of small minibatches. {#c-one-hour}
The reference implementations of this page take `zero_gamma` as an option of `init`.

Training procedures improved long after the architecture was fixed.
A set of refinements, among them cosine learning rate decay, label smoothing and mixup, together with small changes to the stem and the downsampling blocks, raises the top-1 accuracy of ResNet-50 on ImageNet from 75.3% to 79.29%. {#c-bag-of-tricks}
With a modern procedure and no extra data, a plain ResNet-50 reaches 80.4% top-1 at 224 by 224. {#c-strikes-back}
Bello et al. conclude that training and scaling strategies may matter more than architectural changes: better training alone raises a canonical ResNet from 79.0% to 82.2% top-1, and their ResNet-RS models are 1.7 to 2.7 times faster than EfficientNets on TPUs at similar accuracy. {#c-training-matters}
When comparing ResNet with a newer architecture, the training recipe has to be the same, or the comparison measures the recipe.

## Inductive biases

ResNet keeps the biases of convolutional networks: local receptive fields, shared weights across positions, and a hierarchy of feature maps whose resolution halves while the channels double.
The residual form adds one more: each block starts near the identity and learns a correction, so depth acts as a sequence of refinements of one representation rather than a sequence of new ones.
The highway networks that came before differ exactly here: their shortcuts have data-dependent gates with parameters, and when a gate closes the layer computes a non-residual function, while an identity shortcut is never closed and always passes all the information. {#c-highway-contrast}

### Why the shortcut helps

He et al. (2016, Identity Mappings) analyzed the network in which both the shortcut and the function after the addition are identities, $x_{l+1} = x_l + F(x_l)$.
Then, for any deeper unit $L$,

$$ x_L = x_l + \sum_{i=l}^{L-1} F(x_i) , $$ {#eq-residual-unroll}

and for a loss $E$,

$$ \frac{\partial E}{\partial x_l} = \frac{\partial E}{\partial x_L} \Big( 1 + \frac{\partial}{\partial x_l} \sum_{i=l}^{L-1} F(x_i) \Big) . $$ {#eq-residual-grad}

The gradient of the deeper unit reaches the shallower one directly, added to the terms that pass through the residual functions; when the weights are small the second term is small, and the gradient stays close to $\partial E / \partial x_L$ instead of vanishing, as it can in a plain network ([[al:vanishing-gradient-problem]]). {#c-propagation}
If the shortcut is scaled instead, $x_{l+1} = \lambda_l x_l + F(x_l)$, the direct term becomes the product of the $\lambda_i$, which explodes or vanishes exponentially with depth when they are all above or all below 1. {#c-scaled-shortcut}
Unrolled the other way, a stack of $n$ units with identity shortcuts is a sum over $2^n$ paths, one for each subset of the residual functions it passes through, and $\binom{n}{k}$ of them pass through exactly $k$. {#c-paths}
The derivation `residual-gradient` proves the three statements and checks them on random networks.

A residual block whose last batch normalization has $\gamma = 0$ and $\beta = 0$ computes the identity on any non-negative input, such as the output of a ReLU, so a deeper residual network of the same widths can compute everything a shallower one computes. {#c-identity-construction}
The derivation `identity-construction` inserts such blocks into a network and checks that the logits do not change.
This makes the construction argument of He et al. concrete for residual networks; a plain network has to learn the identity with its own weights.

The measured behaviour of trained networks agrees with the picture of small corrections.
The responses of the 3x3 layers of ResNets on CIFAR-10, after batch normalization and before the ReLU, have smaller standard deviations than those of plain networks, and deeper ResNets have smaller ones, so each layer tends to change the signal less. {#c-small-responses}
Veit et al. found that deleting a single block of a trained residual network at test time barely changes its error, except at the downsampling blocks, while deleting a layer of VGG drops it to chance. {#c-lesion} They read a residual network as a collection of many paths of differing length with ensemble-like behavior, and found that most of the gradient of a 110-layer residual network comes from paths only 10 to 34 layers deep. {#c-ensembles}
Balduzzi et al. found that the correlation between the gradients at nearby inputs decays exponentially with depth in feedforward networks, until the gradients look like white noise, while with skip connections it decays sublinearly. {#c-shattered}
Li et al. found that the loss surface of CIFAR-10 networks without shortcuts goes from nearly convex at 20 layers to chaotic at 56 and 110, and that with shortcuts it does not: the 0.1-level contour is almost the same at 20 and 110 layers. {#c-loss-landscape}
For linear networks, Hardt and Ma proved that arbitrarily deep residual networks have no spurious local optima, a property that is much more delicate to establish for linear networks in their standard parameterization. {#c-linear-no-spurious}

He et al. themselves argued that the degradation of their plain networks is unlikely to come from vanishing gradients, since batch normalization keeps the forward signals at non-zero variance and the backward gradients had healthy norms; they conjectured that deep plain networks may have exponentially low convergence rates. {#c-not-vanishing}
Training the plain networks for three times more iterations still showed the degradation. {#c-more-iterations}
So which of these mechanisms explains the degradation is not settled; what is settled is that the identity shortcut removes it.

## Evidence

On ImageNet with 10-crop testing, the top-1 errors are 27.94% for plain-18, 28.54% for plain-34, 27.88% for ResNet-18 and 25.03% for ResNet-34: the plain network gets worse with depth and the residual one better. {#c-table2}
At 18 layers the two are comparably accurate, but the residual network converges faster. {#c-faster-convergence}
All three kinds of shortcut beat the plain network by far; B is slightly better than A, at 24.52% against 25.03% top-1 for ResNet-34, and C marginally better than B at 24.19%, so projection shortcuts are not essential for addressing the degradation. {#c-options-result}

ResNet-50, -101 and -152 are better than ResNet-34 by considerable margins, with 10-crop top-5 errors of 6.71%, 6.05% and 5.71%, and ResNet-152 reaches a single-model top-5 validation error of 4.49%. {#c-deeper-better}
An ensemble of residual networks of different depths reached 3.57% top-5 error on the ImageNet test set and won the ILSVRC 2015 classification task; residual networks also won ImageNet detection and localization and COCO detection and segmentation that year. {#c-ilsvrc}
In Faster R-CNN, replacing VGG-16 with ResNet-101 raises the COCO mAP@[.5, .95] from 21.2% to 27.2%, a 28% relative improvement, with the rest of the system unchanged. {#c-coco}

On CIFAR-10, ResNet-110 reaches 6.43% test error at best and 6.61% ± 0.16 over 5 runs, with fewer parameters than other deep and thin networks of the time. {#c-cifar-110}
A 1202-layer network has no difficulty in optimization and reaches a training error below 0.1%, but its test error of 7.93% is worse than that of the 110-layer network, which He et al. attribute to overfitting 19.4M parameters to a small dataset. {#c-cifar-1202}

The degradation was seen before ResNet.
On MNIST, Srivastava et al. trained fully connected networks of 10 to 100 layers: plain networks did well at 10 layers but their training cross-entropy degraded significantly with depth, while highway networks did not seem to suffer from depth and converged faster. {#c-highway-mnist}
The two experiments of this page reproduce both effects on MNIST in a few minutes on a CPU.

## Limitations

- Depth stops paying at some point. The 1202-layer CIFAR-10 network trains to near-zero error but generalizes worse than the 110-layer one.
- The original block applies a ReLU after the addition, so the shortcut path is not a pure identity; at 1001 layers this costs accuracy, which the pre-activation unit recovers.
- The analysis of [[resnet#eq-residual-grad|the gradient equation]] fails at the blocks with projection shortcuts, two in the CIFAR-10 networks and three in the ImageNet ones; the gradient passes through their projections.
- Batch normalization ties the examples of a mini-batch together, so very small batches or batch-size changes between training and inference change the function. Fixup and similar initializations remove the normalization. [[al:layer-normalization|Layer normalization]] avoids the dependence by normalizing over the features of each example; Transformers use it, while ResNets keep batch normalization.
- Each 1x1 convolution with stride 2 in the original downsampling blocks reads only one position in four of its input.
- The published accuracy of a ResNet depends heavily on the training recipe: the same ResNet-50 is reported at 75.3%, 76.1%, 79.3% and 80.4% top-1 by different procedures.

## Variants and descendants

Moving batch normalization and the ReLU before the weight layers makes the shortcut path a pure identity, which eases training and improves generalization; a 1001-layer network reaches 4.62% error on CIFAR-10, and a 200-layer one improves on ImageNet. {#c-preact} At 1001 layers on CIFAR-10 the original design reaches 7.61% test error and the pre-activation one 4.92% in the same setting. {#c-preact-fig1}
Wide residual networks decrease the depth and increase the width; a 16-layer wide network outperforms all earlier deep residual networks, including thousand-layer ones, in accuracy and efficiency. {#c-wide}
ResNeXt splits each block into a set of transformations of the same topology and calls the size of the set the cardinality; at equal complexity, more cardinality improves accuracy, and it is more effective than going deeper or wider. {#c-resnext}
Stochastic depth drops a random subset of the blocks for each mini-batch and bypasses them with the identity, which shortens training and lets residual networks grow beyond 1200 layers with better test error, 4.91% on CIFAR-10. {#c-stochastic-depth}
Fixup rescales a standard initialization of the residual branches, which solves the [[al:exploding-gradient-problem|exploding]] and [[al:vanishing-gradient-problem|vanishing gradient problem]] at the start of training, so that networks without normalization train as stably as with it, even at 10,000 layers. {#c-fixup}
DenseNet replaces the addition with a concatenation: each layer takes the feature maps of all the layers before it, so a block of $L$ layers has $L(L + 1)/2$ direct connections. {#c-densenet}
ResNet-B of He et al. (2019) moves the stride to the 3x3 convolution, and ResNet-D adds an average pooling before the projection of the downsampling shortcut.
ConvNeXt modernizes a standard ResNet step by step toward the design of a vision Transformer, and the final family reaches 87.8% ImageNet top-1. {#c-convnext} The modern training recipe alone takes ResNet-50 from 76.1% to 78.8% top-1. {#c-convnext-recipe}

## Notable models

ResNet-50, -101 and -152, with {shape:params@resnet50}, {shape:params@resnet101} and {shape:params@resnet152} parameters, are the models of the original paper and the most used members of the family; ResNet-50 became a standard backbone for detection and segmentation and a standard benchmark for training speed.
ResNet-18 and ResNet-34 use two-layer blocks and are common where compute is small.
ResNet-1001 is the pre-activation network of He et al. (2016, Identity Mappings); Wide ResNets, ResNeXt and ResNet-RS are the descendants above.

## Implementations

The NumPy reference `impl/numpy/resnet.py` builds the bottleneck network of the shape contract with explicit convolutions, max pooling and batch normalization, returns the internals of every block, and supports both placements of the stride and zero-gamma initialization.
It also has a fully connected residual network with a hand-written backward pass, which the experiments use and the tests check against finite differences.
The PyTorch version `impl/pytorch/resnet.py` is a `torch.nn.Module` with a `from_params` constructor, and the JAX version `impl/jax/resnet.py` is a pair of pure functions `init` and `apply`; both load the same parameters as NumPy.
All three are tested against the fixture of `fixtures/`, a small network with random weights and statistics, in training and inference mode.

### Worked example

Take a two-layer block on vectors with $x = (2, -1)$,

$$ W_1 = \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}, \qquad W_2 = \begin{pmatrix} 0.5 & 0 \\ 0 & -1 \end{pmatrix} . $$

Then $W_1 x = (1, 3)$, the ReLU keeps it, and $F(x) = W_2 (1, 3) = (0.5, -3)$.
The residual block gives $\operatorname{relu}(F(x) + x) = \operatorname{relu}(2.5, -4) = (2.5, 0)$; the plain block gives $\operatorname{relu}(F(x)) = (0.5, 0)$.
With $W_2 = 0$, the residual block gives $\operatorname{relu}(x) = (2, 0)$, the input with its negative entry clipped, while the plain block gives $(0, 0)$ and loses the input.
For the gradient, take $\partial E / \partial y = (1, 1)$. The ReLU after the addition passes $(1, 0)$. Through $F$ this becomes $W_1^\top \big((W_2^\top (1, 0)) \odot (1, 1)\big) = W_1^\top (0.5, 0) = (0.5, 0.5)$, and the shortcut adds $(1, 0)$, so $\partial E / \partial x = (1.5, 0.5)$. With $W_2 = 0$ the path through $F$ gives zero but the shortcut still gives $(1, 0)$; the plain block would give $(0, 0)$.

## Experiments

Two experiments test the degradation and its fix at a scale that runs on a CPU in two minutes.
Fully connected networks with batch normalization after every layer and ReLU are trained on a subset of the MNIST training images with SGD, momentum 0.9 and weight decay $10^{-4}$, for the same number of steps at every depth.
Each network has an input layer, $n$ blocks of two layers and a classifier, so $2n + 2$ layers with weights; the residual network adds an identity shortcut around each block, as in [[resnet#eq-residual|the residual block]], and has exactly the same parameters as the plain one.
The training error is measured on the training subset with the running statistics.

`mnist-depth-22` uses width 32, 5,000 images and 300 steps. Over 10 seeds, the median training error is 0.8% for the plain network of 4 layers and 24.4% for that of 22 layers; with shortcuts, the 22-layer network has 2.8%. On the test images the plain 22-layer network has 26.3% error and the residual one 8.5%.
`mnist-depth-32` uses width 64, 10,000 images and 400 steps. The plain network goes from 0.7% training error at 4 layers to 36.5% at 32 layers, and the residual 32-layer network has 2.5%; on the test images, 37.0% against 7.0%.
Both experiments hold their expectations, with Mann-Whitney tests at $p = 0.0002$ after the Holm correction: the deeper plain network has the higher training error, and the shortcuts remove most of it.
Two secondary results show the limits of the small scale. At 4 layers the shortcut makes little difference, as He et al. found at 18 layers. And at these budgets the deep residual networks still train to a slightly higher error than the shallow ones, 2.8% against 0.7% and 2.5% against 0.8%, so the experiments do not reproduce the gain from depth of the 34-layer ResNet over the 18-layer one; they find only that the shortcuts remove most of the degradation.

## History

Deeper networks had driven the progress of image classification: the VGG paper reported in 2014 that 16 to 19 weight layers of small 3x3 filters gave a significant improvement over shallower designs. {#c-vgg}
Batch normalization and the initialization of He et al. (2015) made networks of that depth trainable from scratch, but going further still failed.
Srivastava, Greff and Schmidhuber, of the Swiss AI Lab IDSIA, posted highway networks, with gated shortcut connections, in 2015; version 2 of arXiv 1505.00387 is dated 3 Nov 2015. {#c-hist-highway}
He, Zhang, Ren and Sun, at Microsoft Research, posted deep residual learning as arXiv 1512.03385 on 10 Dec 2015, {#c-hist-resnet} after winning the ILSVRC and COCO 2015 competitions with it, and published it at CVPR 2016.
In 2016 they posted the pre-activation unit and the propagation analysis as Identity Mappings in Deep Residual Networks, arXiv 1603.05027, with version 3 dated 25 Jul 2016. {#c-hist-preact}
Wide ResNets, stochastic depth, ResNeXt and DenseNet followed within two years, and residual connections became part of nearly every deep architecture, including the Transformer of 2017.
From 2019, work such as Bag of Tricks, ResNet-RS, ResNet strikes back and ConvNeXt measured how much of ResNet's accuracy depends on the training procedure.

## References

The sources of this page are listed in its front matter and cited by its claims; the main ones are He et al. (2016) for the architecture and its results, He et al. (2016, Identity Mappings) for the propagation analysis, and Srivastava et al. (2015) for the earlier observation of the degradation.
