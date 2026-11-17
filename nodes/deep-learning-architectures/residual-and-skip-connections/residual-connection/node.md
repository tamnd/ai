---
tag: NEW-residual-connection
kind: method
slug: residual-connection
name: Residual connection
aliases: [skip connection, shortcut connection, identity shortcut, residual block, residual unit]
area: deep-learning-architectures/residual-and-skip-connections
level: core
summary: >-
  Add the input of a block of layers to its output, y = x + F(x), so that the block learns a change to its input and the gradient reaches earlier layers through an identity path.
purpose: optimization
uses: [backpropagation@al, matrix-multiplication@m2, binomial-distribution@m2]
related_to: [resnet, transformer, vanishing-gradient-problem@al, exploding-gradient-problem@al, batch-normalization@al, layer-normalization@al, he-initialization@al]
hyperparameters:
  - {name: shortcut scale, meaning: "the factor λ on the shortcut, x_{i+1} = λ x_i + F(x_i)", typical: "1, the identity"}
  - {name: normalization placement, meaning: "where the normalization sits: on the branch before the other layers (pre) or after the addition (post)", typical: "pre in deep stacks"}
  - {name: branch initialization, meaning: "the scale of the last layer of each branch at the start", typical: "zero or scaled down with depth"}
  - {name: branch depth, meaning: "the number of weight layers inside F", typical: "2 or 3"}
cost:
  train: "one element-wise addition per block in the forward pass and one in the backward pass; no parameters; the input of the block must be kept until the addition"
  inference: "one element-wise addition per block"
history:
  year: 2015
  people: [Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun]
  note: "Identity shortcuts in He et al., arXiv 1512.03385 (10 Dec 2015); gated shortcuts before them in Highway Networks, Srivastava et al., arXiv 1505.00387; the propagation analysis in He et al., arXiv 1603.05027."
prereqs: [backpropagation@al, vanishing-gradient-problem@al, matrix-multiplication@m2]
same_as:
  wikipedia: Residual_neural_network
  arxiv: ["1512.03385", "1603.05027"]
sources:
  - {paper: "He, Zhang, Ren and Sun (2016)", title: "Deep Residual Learning for Image Recognition", locator: "arXiv 1512.03385v1, Sections 2 and 3.1, p. 3", note: "the residual form and the identity shortcut"}
  - {paper: "He, Zhang, Ren and Sun (2016)", title: "Identity Mappings in Deep Residual Networks", locator: "arXiv 1603.05027v3, Sections 2 to 4 and 6, Tables 1 and 3", note: "the propagation analysis, the shortcut ablation and the pre-activation unit"}
  - {paper: "Srivastava, Greff and Schmidhuber (2015)", title: "Highway Networks", locator: "arXiv 1505.00387v2, Sections 2 and 2.2", note: "the gated precursor"}
  - {paper: "Veit, Wilber and Belongie (2016)", title: "Residual Networks Behave Like Ensembles of Relatively Shallow Networks", locator: "arXiv 1605.06431v2, the abstract, Sections 3 and 5"}
  - {paper: "Li, Xu, Taylor, Studer and Goldstein (2018)", title: "Visualizing the Loss Landscape of Neural Nets", locator: "arXiv 1712.09913v3, Section 6"}
  - {paper: "Xiong et al. (2020)", title: "On Layer Normalization in the Transformer Architecture", locator: "arXiv 2002.04745v2, the abstract, Section 3.3"}
  - {paper: "Huang, Liu, van der Maaten and Weinberger (2017)", title: "Densely Connected Convolutional Networks", locator: "arXiv 1608.06993v5, the abstract, Section 3"}
  - {paper: "Ronneberger, Fischer and Brox (2015)", title: "U-Net: Convolutional Networks for Biomedical Image Segmentation", locator: "arXiv 1505.04597v1, Sections 1 and 2"}
  - {paper: "Zhang, Dauphin and Ma (2019)", title: "Fixup Initialization: Residual Learning Without Normalization", locator: "arXiv 1901.09321v2, the abstract, p. 5"}
  - {paper: "Goyal et al. (2017)", title: "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour", locator: "arXiv 1706.02677v2, Section 5.1"}
---

A residual connection adds the input of a block of layers to the output of the block.
The block then learns a change to its input rather than a whole new representation.
In the backward pass the addition copies the gradient to the input unchanged, so a deep stack of such blocks always has one path on which the gradient is not multiplied by any weight.
Residual connections are the shortcut of [[resnet]] and the skeleton of the [[transformer]], and they are the main reason that networks of a hundred or a thousand layers can be trained.

## In plain words

Instead of asking a few layers to compute the output from scratch, let them compute a correction and add it to what came in.
If the layers output zero, the block passes its input through untouched.
A deep network built this way can always fall back on doing nothing in a block, and the error signal can always travel back along the chain of additions.

## Setting

The input is a vector or tensor $x$, and the branch $F$ is any small differentiable function whose output has the shape of $x$: two or three weight layers with a nonlinearity, an attention layer, or a feedforward layer.
A network is a stack of units $x_{i+1} = x_i + F_i(x_i)$ for $i = 0, \dots, n-1$.
Training is by backpropagation of a scalar loss $E$.
When $F$ changes the shape, as at the stride steps of a ResNet, the shortcut needs a projection; that case belongs to [[resnet]] and is not treated here.

## The method

He et al. (2015) write the target mapping of a few stacked layers as $H(x)$ and let the layers fit the residual $F(x) := H(x) - x$, so that the block computes $F(x) + x$, with input and output of the same dimensions. {#c-residual-form}

$$ y = x + F(x) $$ {#eq-residual}

In the reference implementation the branch is a two-layer perceptron,

$$ F(x) = W_2\, \phi(W_1 x + b_1) + b_2 , $$ {#eq-branch}

with $\phi$ a ReLU or tanh, and the plain counterpart of the same block computes $y = F(x)$.
He et al. argue that if an identity mapping is optimal, the solver can drive the weights of the branch to zero, and that small changes are easier to find with reference to the identity; they report that the learned residual functions have small responses. {#c-precondition}
The shortcut adds no parameters and no computation apart from the element-wise addition, so a plain network and a residual network of the same depth, width and cost can be compared directly. {#c-no-extra-params}
If the last layer of the branch starts at zero, $F(x) = 0$ for every input, the block is the identity, and its Jacobian is $I$. {#c-zero-branch-identity}

Shortcuts themselves are old: He et al. note that an early practice in training multilayer perceptrons was to add a linear layer from the input of the network to its output. {#c-early-shortcuts}
The direct precursor is the highway layer of Srivastava et al. (2015),

$$ y = H(x)\, T(x) + x\, C(x) , $$ {#eq-highway}

with a transform gate $T$ and a carry gate $C$; they set $C = 1 - T$ and require $x$, $y$, $H(x)$ and $T(x)$ to have the same size. {#c-highway}
With $C = 1 - T$, the layer gives $y = x$ with Jacobian $I$ when $T(x) = 0$ and $y = H(x)$ with Jacobian $H'(x)$ when $T(x) = 1$. {#c-highway-limits}
The gate is $T(x) = \sigma(W_T x + b_T)$, and Srivastava et al. initialize $b_T$ to a negative value such as $-1$ or $-3$, so that the layer starts biased toward carrying its input, after the gate biases of LSTMs. {#c-highway-bias}
He et al. (2015) contrast the two: highway gates depend on the data and have parameters, and a closed gate makes the layer non-residual, while an identity shortcut is never closed. {#c-highway-contrast}
Setting $T = C = 1$ in [[residual-connection#eq-highway|the highway layer]] gives exactly $y = x + H(x)$, but with the coupled gate $C = 1 - T$ this point is out of reach, since $T = 1$ forces $C = 0$. {#c-highway-reduces}

## Derivation

The derivation `gradient-flow` proves the following and checks each step numerically.
Unrolling the units from $l$ to $L$ gives

$$ x_L = x_l + \sum_{i=l}^{L-1} F_i(x_i) . $$ {#eq-unroll}

So the deeper feature is the shallower one plus a sum of residuals (He et al. 2016, Identity Mappings, Section 2). {#c-unroll}
With $J_i$ the Jacobian of $F_i$ at $x_i$, the chain rule gives

$$ \frac{\partial x_L}{\partial x_l} = (I + J_{L-1}) \cdots (I + J_l) , $$ {#eq-jacobian}

$$ \frac{\partial E}{\partial x_l} = \frac{\partial E}{\partial x_L} \Big( I + \frac{\partial}{\partial x_l} \sum_{i=l}^{L-1} F_i(x_i) \Big) . $$ {#eq-grad}

The first term carries the gradient at the top to unit $l$ without passing through any weight. {#c-gradient}
In the backward pass of one unit, with $g_i = \partial E / \partial x_i$,

$$ g_i = g_{i+1} + g_{i+1} J_i . $$ {#eq-backward}

He et al. argue that for a mini-batch the gradient is unlikely to be canceled, because the second term cannot be $-1$ for all samples, so the gradient of a layer does not vanish even when the weights are arbitrarily small. {#c-not-cancel}
Their argument stops there; the bounds below are this page's own.
If every $\lVert J_i \rVert_2 \le \varepsilon < 1$, the triangle inequality on [[residual-connection#eq-backward|the backward step]] gives $(1 - \varepsilon)^n \lVert g_L \rVert \le \lVert g_l \rVert \le (1 + \varepsilon)^n \lVert g_L \rVert$ over $n$ units, while a plain stack only has $\lVert g_l \rVert \le \prod_i \lVert J'_i \rVert_2 \lVert g_L \rVert$, which goes to zero geometrically when the factors are below 1. {#c-gradient-bounds}

The identity matters in this argument.
With a scaled shortcut, $x_{i+1} = \lambda_i x_i + F_i(x_i)$, the term of the Jacobian that goes through no branch is

$$ \Big( \prod_{i=l}^{L-1} \lambda_i \Big) I , $$ {#eq-scaled}

which shrinks or grows exponentially with depth when all $\lambda_i$ are below or above 1 (He et al. 2016, Section 3). {#c-scaled-shortcut}

Two more forms keep the identity path.
The pre-activation unit of He et al. (2016, Section 4.1) applies the activation only on the branch,

$$ x_{i+1} = x_i + F_i\big(\hat f(x_i)\big) , $$ {#eq-preact}

which is the same form with branch $F_i \circ \hat f$, so the unrolled sum and the gradient hold unchanged; the pre-LN Transformer layer is the case $\hat f = \mathrm{LN}$. {#c-preact-form}
The post-LN unit normalizes after the addition,

$$ x_{i+1} = \mathrm{LN}\big(x_i + F_i(x_i)\big) , $$ {#eq-postln}

and its Jacobian is $J_{\mathrm{LN}}(I + J_i)$, so the term that goes through no branch is the product of the normalization Jacobians, not $I$. {#c-postln-jacobian}

**Paths.** The derivation `path-count` expands the product of [[residual-connection#eq-jacobian|the Jacobian]] by picking $I$ or $J_i$ from each factor:

$$ \prod_{i=1}^{n} (I + J_i) = \sum_{S \subseteq \{1, \dots, n\}} \prod_{i \in S}^{\rightarrow} J_i . $$ {#eq-paths}

There are $2^n$ terms, one per subset of branches, and $\binom{n}{k}$ of them go through $k$ branches, so the path length is binomial with mean $n/2$; Veit et al. (2016, Section 3) count the paths the same way, with binary codes. {#c-path-count}
For 54 units the share of paths through 19 to 35 units is 98.0%, and through 5 to 17 units, the paths of 10 to 34 layers, it is 0.45%; both numbers are computed here. {#c-path-share}

## Worked example

Take two units with the ReLU branch of [[residual-connection#eq-branch|the branch]], $x = (2, -1)$ and the fixture of the tests:

$$ W_1^{(0)} = \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}, \; W_2^{(0)} = \begin{pmatrix} 0.5 & 0 \\ 0 & -1 \end{pmatrix}, \; W_1^{(1)} = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \; b_1^{(1)} = (0.5, 0), \; W_2^{(1)} = \begin{pmatrix} 1 & 0 \\ 0 & 0.5 \end{pmatrix}, \; b_2^{(1)} = (0, 0.25) , $$

and zero for the other biases.
Unit 0: $W_1 x = (1, 3)$, the ReLU keeps it, $F = (0.5, -3)$, and $x_1 = (2.5, -4)$.
Unit 1: $z = (-3.5, 2.5)$, $h = (0, 2.5)$, $F = (0, 1.5)$, and $y = (2.5, -2.5)$.
The plain stack, $y = F_1(F_0(x))$, gives $(0, 0.5)$.

For the backward pass take $\partial E / \partial y = (1, 1)$.
Unit 1 passes $(1, 1)$ along the shortcut and $(0.5, 0)$ through the branch, so $g_1 = (1.5, 1)$.
Unit 0 passes $(1.5, 1)$ and adds $W_1^\top \big((0.75, -1) \odot (1, 1)\big) = (-0.25, 1.75)$, so $\partial E / \partial x = (1.25, 2.75)$.
The weight gradients of unit 0 are $\partial E / \partial W_1 = \begin{pmatrix} 1.5 & -0.75 \\ -2 & 1 \end{pmatrix}$ and $\partial E / \partial W_2 = \begin{pmatrix} 1.5 & 4.5 \\ 1 & 3 \end{pmatrix}$.
In the plain stack, $\partial E / \partial x = (0.25, 0.25)$ and $\partial E / \partial W_1 = \begin{pmatrix} 0.5 & -0.25 \\ 0 & 0 \end{pmatrix}$: the gradient to the first layer is five times smaller in the first entry and zero in the second row, after only two units.
The tests `test_worked_example` of the NumPy, PyTorch and JAX implementations check every number here.

The same effect at depth, measured for this page and not taken from a source: stacks of width 16 with tanh branches whose last layer has scale 0.1, a batch of 8 and fixed seeds.
The ratio $\lVert \partial E / \partial x_0 \rVert / \lVert \partial E / \partial y \rVert$ is

| units | residual | plain |
|---|---|---|
| 10 | 1.026 | $3.9 \times 10^{-10}$ |
| 25 | 1.049 | $1.0 \times 10^{-23}$ |
| 50 | 1.183 | $2.4 \times 10^{-45}$ |
| 100 | 1.298 | $2.3 \times 10^{-89}$ |

The residual ratio stays near 1, inside the bounds of the derivation, and the plain one falls by about ten orders of magnitude every ten units.

## Hyperparameters

The shortcut itself has no parameters; the choices are around it.
The scale of the shortcut should be 1: in the ablation below, a scale of 0.5 fails.
The normalization can sit on the branch, before the other layers (pre), or after the addition (post); He et al. place both the batch normalization and the ReLU on the branch for very deep networks, and Xiong et al. put the layer normalization inside the residual connection before all other nonlinear transformations, with a final layer normalization before the prediction. {#c-ln-placement}
The last layer of each branch can start at zero or small; see Variants.

## Cost

The addition costs one operation per element in the forward pass and one copy of the gradient in the backward pass.
The input of each unit must stay in memory until the addition, which a plain stack can free earlier.
There are no parameters.

## Evidence

On CIFAR-10 with ResNet-110, an identity shortcut gives 6.61% test error, while scaling it by 0.5 fails or gives 12.35%, gates give 8.70% to 12.86% unless biased toward carry (6.91%), a 1x1 convolution gives 12.22%, and dropout on it fails. {#c-shortcut-ablation}
He et al. conclude that identity shortcuts and an identity after the addition are essential for smooth propagation. {#c-identity-essential}
Pre-activation lowers CIFAR-10 error from 6.61% to 6.37% at 110 layers, 5.93% to 5.46% at 164, and 7.61% to 4.92% at 1001. {#c-preact-results}
Batch normalization after the addition is considerably worse, and a ReLU before the addition gives 7.84%. {#c-after-addition}
Without shortcuts the loss surface turns chaotic from 20 to 110 layers; with them it stays nearly convex. {#c-loss-landscape}
A DenseNet surface shows no noticeable non-convexity. {#c-loss-densenet}
Lesion studies find that the paths of a residual network behave like an ensemble, and that most of the gradient of a 110-layer network comes from paths 10 to 34 layers deep. {#c-ensemble}

## Limitations and failure modes

The identity path holds only when nothing sits on it.
A ReLU or normalization after the addition, a scaled or gated shortcut, or a projection puts a factor on every term of [[residual-connection#eq-paths|the path expansion]], and the derivation no longer gives an unchanged gradient.
He et al. (2016) find the impact of pre-activation twofold, easier optimization because the function after the addition is the identity, and better regularization from batch normalization on the branch; with a ReLU after the addition the effect is not severe at 164 layers but large at 1000. {#c-preact-why}
Xiong et al. prove that at initialization the expected gradients near the output of a post-LN Transformer are large, which makes a learning-rate warm-up helpful, while pre-LN gradients are well behaved, and that pre-LN without warm-up reaches comparable results with less training time and tuning. {#c-xiong}
The identity path also sends the activations up: by [[residual-connection#eq-unroll|the unrolled sum]] the residuals add up, so without normalization or a small branch scale the forward signal can grow with depth.
The shortcut is therefore no cure for the exploding gradient problem: the upper bound $(1 + \varepsilon)^n$ of the backward step still grows with depth, so the identity removes the vanishing direction but not the growing one.
He initialization keeps the variance of each layer of a plain stack near one, but in a residual stack the variances of the branches add up on top of the shortcut, which is why Fixup and zero-γ below start the branches smaller than He initialization would.
Short paths do most of the work: by Veit et al., a deep residual network is not one deep computation but many shallow ones in parallel.

## Variants

- **Highway connections** gate the shortcut, as in [[residual-connection#eq-highway|the highway layer]] above.
- **Pre-activation and pre-LN** put the normalization and activation on the branch, [[residual-connection#eq-preact|the pre-activation unit]]; post-LN, [[residual-connection#eq-postln|the post-LN unit]], is the order of the original Transformer.
- **Dense connections** concatenate instead of adding: layer $\ell$ of a DenseNet block receives $[x_0, \dots, x_{\ell-1}]$, because, by Huang et al., the summation of ResNets may impede the information flow. {#c-dense}

$$ x_\ell = H_\ell\big([x_0, x_1, \dots, x_{\ell-1}]\big) $$ {#eq-dense}

  A block of $L$ layers then has $L(L+1)/2$ direct connections instead of $L$. {#c-dense-connections}
  Huang et al. state that this alleviates vanishing gradients, strengthens feature propagation, encourages feature reuse and reduces the number of parameters. {#c-dense-claims}
- **Long skips between an encoder and a decoder.** In U-Net, each step of the expansive path upsamples the feature map and concatenates the correspondingly cropped feature map of the contracting path. {#c-unet}
  Ronneberger et al. combine the high-resolution features of the contracting path with the upsampled output in order to localize. {#c-unet-why}
- **Fixup** zero-initializes the classifier and the last layer of each branch, scales the other branch layers by $L^{-1/(2m-2)}$, and adds a scalar multiplier and biases; Zhang et al. find the rescaling essential. {#c-fixup}
  With it, residual networks without normalization train as stably as with it, up to 10,000 layers. {#c-fixup-result}
- **Zero-γ.** Goyal et al. set the scale γ of the last batch normalization of each block to zero, so the signal first flows through the shortcut; they found that this eases optimization early in training. {#c-zero-gamma}

## When to use it

Use residual connections in any network deeper than a handful of layers whose blocks keep the shape of their input: convolutional networks, Transformers, deep perceptrons.
Keep the shortcut an identity, put normalization on the branch in deep stacks, and start the branches small.
Use concatenation, as in DenseNet or U-Net, when later layers need the earlier features themselves rather than their sum.

## Pitfalls

> [!pitfall]
> **A function on the shortcut path.** A ReLU, normalization or scale after the addition breaks the identity term of the gradient. The test `test_zero_branch_is_identity` checks that a block with a zero branch is exactly the identity, and its Jacobian is $I$.

> [!pitfall]
> **A backward pass that drops the shortcut gradient.** Hand-written backward code must add the incoming gradient to the gradient through the branch. The test `test_backward_matches_finite_differences` compares the NumPy backward pass with finite differences, and `test_jacobian_is_product` compares the stack Jacobian with the product of the $I + J_i$.

> [!pitfall]
> **Branches that start too large.** With full-size branches the residuals add up and the signal grows with depth. The test `test_gradient_norm_bounded_vs_plain` checks that the gradient ratio of a residual stack with small branches stays between 0.8 and 1.6 at 10 and 50 units while the plain one decays, and `test_gradient_within_derivation_bounds` checks the bounds of the derivation.

> [!pitfall]
> **Gates that cannot open fully.** A sigmoid gate is never exactly 0 or 1, so a highway layer with $C = 1 - T$ never becomes a residual block. The test `test_highway_gates` checks both limits and that $T = C = 1$ gives the residual block. The test `test_path_count` checks the $2^n$ expansion.

## History

Srivastava, Greff and Schmidhuber proposed highway networks in May 2015 and posted the second version, arXiv 1505.00387v2, on 3 Nov 2015. {#c-hist-highway}
Ronneberger, Fischer and Brox posted U-Net, with its long concatenated skips, as arXiv 1505.04597v1 on 18 May 2015. {#c-hist-unet}
He, Zhang, Ren and Sun posted Deep Residual Learning for Image Recognition, with the identity shortcut, as arXiv 1512.03385v1 on 10 Dec 2015. {#c-hist-resnet}
Their follow-up on identity mappings in 2016 gave the propagation analysis and the pre-activation unit, and Xiong et al. carried the same question to the Transformer, posting the second version of their paper, arXiv 2002.04745v2, on 29 Jun 2020. {#c-hist-xiong}
