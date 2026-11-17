---
tag: 1REXCC
kind: architecture
slug: group-equivariant-convolutional-network
name: Group equivariant convolutional network
aliases: [G-CNN, group CNN, group convolutional network, p4-CNN, p4m-CNN, group equivariant CNN, G-convolution]
area: deep-learning-architectures/equivariant-and-geometric-networks
level: core
summary: >-
  A CNN whose layers correlate feature maps with filters over a symmetry group, such as translations and 90-degree rotations, so that turning the input turns every feature map.
uses: [backpropagation@al, relu@al, matrix-multiplication@m2]
related_to: [convolutional-neural-network, resnet, graph-neural-network]
history:
  year: 2016
  people: [Taco S. Cohen, Max Welling]
  note: "Cohen and Welling posted Group Equivariant Convolutional Networks as arXiv 1602.07576 in February 2016, and it appeared at ICML 2016. Earlier networks had rotated feature maps by hand (Dieleman et al., 2015); later work generalized the construction to steerable filters, continuous groups and graphs."
prereqs: [convolutional-neural-network, backpropagation@al]
same_as:
  arxiv: ["1602.07576"]
sources:
  - {paper: "Cohen and Welling (2016)", title: "Group Equivariant Convolutional Networks", locator: "arXiv 1602.07576v3, the abstract and Sections 2 to 9, Tables 1 and 2", note: "the G-correlation, its layers, the filter-bank implementation and the experiments"}
  - {paper: "Bronstein, Bruna, Cohen and Veličković (2021)", title: "Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges", locator: "arXiv 2104.13478v2, Section 3.5", note: "why invariant linear maps are weak, and the geometric blueprint"}
  - {paper: "Kondor and Trivedi (2018)", title: "On the Generalization of Equivariance and Convolution in Neural Networks to the Action of Compact Groups", locator: "arXiv 1802.03690v3, the abstract", note: "convolution is necessary for equivariance"}
  - {paper: "Weiler and Cesa (2019)", title: "General E(2)-Equivariant Steerable CNNs", locator: "arXiv 1911.08251v2, the abstract, Section 3.5 and Tables 5 and 6"}
  - {paper: "Zaheer, Kottur, Ravanbakhsh, Póczos, Salakhutdinov and Smola (2017)", title: "Deep Sets", locator: "arXiv 1703.06114v3, Section 2, Theorem 2 and Lemma 3", note: "the permutation group"}
  - {paper: "Satorras, Hoogeboom and Welling (2021)", title: "E(n) Equivariant Graph Neural Networks", locator: "arXiv 2102.09844v3, the abstract"}
family: convolutional
inputs: "a batch of images as a (B, C0, n, n) array"
outputs: "logits as a (B, M) array, invariant to rotations of the input by multiples of 90 degrees"
components: []
inductive_biases: ["equivariance to a chosen group: a translation or a rotation by 90 degrees of the input moves and turns every feature map in the same way", "weight sharing over the group: one stored filter is applied at every position and in every rotation", "locality, as in a planar CNN: each output depends on a k by k window of the layer below"]
params: "C1 C0 k^2 + C1 + 4 C2 C1 k^2 + C2 + M C2 + M for the lifting layer, one p4 layer and the linear head"
flops_per_example: "2 n^2 k^2 (4 C1 C0 + 16 C2 C1) + 2 C2 M, counting a multiply-add as 2 FLOPs and leaving out biases, ReLUs and pooling"
---

A group equivariant convolutional network, or G-CNN, is a convolutional network whose layers are equivariant to a group of transformations larger than the translations.
For the group p4 of translations and rotations by 90 degrees, each feature map has one planar map per rotation, and rotating the input image rotates every feature map and cycles its rotation planes, in every layer.
The layers correlate a feature map with filters moved by every element of the group, which is computed by one ordinary planar convolution with a bank of rotated and permuted copies of the stored filters.
Cohen and Welling introduced the model in 2016 and reported lower test errors than all earlier published results on rotated MNIST and on plain CIFAR-10.

## In plain words

A convolutional network looks for the same pattern everywhere in an image, so if the image moves, its features move with it.
It does not look for the pattern in every orientation: a detector of vertical edges says nothing about horizontal ones, and a network trained on upright digits may fail on tilted ones.
A G-CNN looks for each pattern in every orientation of a chosen set, for example the four quarter turns, and keeps a separate answer for each orientation.
When the image is turned a quarter of the way round, the answers are the same numbers moved to new places, so the network does not have to learn four separate versions of each pattern from data.
Deeper layers combine features together with their relative orientations, and at the end the network takes the strongest orientation at each place, so its prediction does not change when the image is turned.

## Overview

A map $\Phi$ between feature spaces is equivariant to a group $G$ when transforming the input and then applying $\Phi$ gives the same result as applying $\Phi$ and then transforming the output:

$$ \Phi(T_g x) = T'_g\, \Phi(x) \qquad \text{for all } g \in G . $$ {#eq-equivariance}

Invariance is the special case where $T'_g$ is the identity.
Cohen and Welling argue that in deep learning general equivariance is more useful than invariance, because it is impossible to determine if features are in the right spatial configuration if they are invariant. {#c-equivariance-more-useful}

A planar correlation of a stack of maps $f_k$ with filters $\psi_k$,

$$ [f \star \psi](x) = \sum_{y \in \mathbb{Z}^2} \sum_k f_k(y)\, \psi_k(y - x) , $$ {#eq-planar-corr}

is equivariant to translations: shifting the image shifts the output. {#c-planar-translation}
It is not equivariant to rotations. Writing $L_r f$ for the image turned by $r$,

$$ [L_r f] \star \psi = L_r \big[ f \star L_{r^{-1}} \psi \big] : $$ {#eq-planar-rotation}

the turned image correlated with a filter is the turned output of the original image with the filter turned back. {#c-planar-rotation}
Cohen and Welling draw the consequence that if an ordinary CNN learns rotated copies of the same filter, the stack of feature maps is equivariant, although individual feature maps are not. {#c-cnn-rotated-copies}
A G-CNN builds in those rotated copies, and keeps track of which copy produced which map.
Cohen and Welling describe G-convolutions as a layer that enjoys a substantially higher degree of weight sharing than regular convolution layers, and that increases the expressive capacity of the network without increasing the number of parameters. {#c-cohen-weight-sharing}
The rest of this page derives the layers for p4, proves that they and the usual nonlinearities and poolings are equivariant, counts their parameters and cost, and states what the original and later experiments measured.

## Components

### The group p4 and its feature maps

The group p4 consists of all compositions of translations and rotations by 90 degrees about any center of rotation in a square grid. {#c-p4}
Every element is a rotation $r \in \{0, 1, 2, 3\}$ quarter turns followed by a translation $t \in \mathbb{Z}^2$, $g = (t, r)$, acting on the plane by $g x = R^r x + t$.
A feature map on p4 is a function $F(t, r)$: for each channel, one planar map per rotation, an array of shape $(B, C, 4, n, n)$.
The rotation $u$ by a quarter turn acts on it by turning every plane and moving plane $r - 1$ to plane $r$:

$$ [L_u F](t, r) = F(R^{-1} t,\ r - 1) . $$ {#eq-p4-action}

Applying it four times gives back $F$, as it must for an element of order four.

### The lifting correlation

The first layer takes a planar image to a p4 map. It reads the filter at $g^{-1} y$ instead of $y - x$:

$$ [f \star \psi](g) = \sum_{y \in \mathbb{Z}^2} \sum_k f_k(y)\, \psi_k(g^{-1} y), \qquad [f \star \psi](t, r) = [f \star L_r \psi](t) . $$ {#eq-lift}

At rotation $r$ it is the planar correlation with the filter turned $r$ times, so the layer runs each stored filter in its four orientations.
It is equivariant: $[L_u f] \star \psi = L_u [f \star \psi]$ for every $u \in G$, with $L_u$ acting by eq-p4-action on the output. {#c-lift-equivariant}

### The group correlation

The later layers take a p4 map to a p4 map. The sum runs over the group, and the filters are themselves functions on p4, an array $(C_{\text{out}}, C_{\text{in}}, 4, k, k)$:

$$ [f \star \psi](g) = \sum_{h \in G} \sum_k f_k(h)\, \psi_k(g^{-1} h) . $$ {#eq-gconv}

This is the group correlation, and it is equivariant for every filter: $[L_u f] \star \psi = L_u [f \star \psi]$ for every $u \in G$. {#c-gconv-equivariant}
The proof is one substitution, $h = u h'$, which reindexes the sum; it uses nothing about $\psi$.
The derivation `group-correlation-equivariance` proves eq-planar-rotation and both equivariances, and its check confirms them on the NumPy reference with zero padding and with wrapping, on grids of odd and even side.

### Biases, nonlinearities and normalization

A pointwise nonlinearity acts on each value alone, so it commutes with the action: $\nu(L_u F) = L_u\, \nu(F)$. {#c-pointwise-commutes}
A bias commutes with the action when it has one value per G-feature map, shared by its four rotation planes, and not when it has one value per plane. {#c-bias-per-map}
Cohen and Welling state the same rule: there should be only one bias per G-feature map, and batch normalization should be implemented with a single scale and bias parameter per G-feature map in order to preserve equivariance. {#c-cohen-bias}
They also note that the sum of two G-equivariant feature maps is also G-equivariant, so G-conv layers can be used in highway networks and residual networks. {#c-cohen-residual}

### Pooling

Max pooling over a window that moves with the group commutes with the action. {#c-cohen-pooling}
Subsampling is another matter: Cohen and Welling generalize the stride to subsampling on a subgroup $H$, and note that the subsampled feature map is then equivariant to $H$ but not $G$. {#c-cohen-subsampling}
On a square grid this has a concrete form: subsampling with stride 2, keeping the pixels with even indices, keeps rotation equivariance on a grid of odd side and breaks it on a grid of even side. {#c-stride-even-grid}
Pooling over all four rotations at each position, coset pooling,

$$ [P F](t) = \max_{r} F(t, r) , $$ {#eq-rotpool}

gives a planar map that turns with the input, $P L_u = L_u P$: a function on $\mathbb{Z}^2 \cong p4 / R$ that transforms like the input image. {#c-coset-pooling}

### The whole network

Since every layer type is equivariant, any stack of lifting correlations, group correlations, shared biases, pointwise nonlinearities and window poolings is equivariant to p4. {#c-stacking}
Cohen and Welling write that since all layer types are equivariant, they can freely stack them into deep networks and expect G-conv parameter sharing to be effective at arbitrary depth. {#c-cohen-stack}
The network of this page is a lifting layer, ReLU, a group correlation, ReLU, coset pooling, the mean over the plane and a linear head.
The mean over the plane of a map that turns with the input is invariant, so the logits of this network are invariant to the four rotations of the input, and to cyclic translations when the correlations wrap around. {#c-model-invariant}
The derivation `nonlinearity-and-pooling` proves the statements of this section, and its check confirms each on the reference.

## Shapes and parameters

The lifting filters are $W_1 \in \mathbb{R}^{C_1 \times C_0 \times k \times k}$, the group filters $W_2 \in \mathbb{R}^{C_2 \times C_1 \times 4 \times k \times k}$, and the head is $M \times C_2$, so the network has

$$ P = C_1 C_0 k^2 + C_1 + 4 C_2 C_1 k^2 + C_2 + M C_2 + M $$ {#eq-params}

parameters. {#c-params}
In the shape contract the formula is

{shape:params}

and the example config, one input channel, $C_1 = C_2 = 4$, $k = 3$ and three classes, has {shape:params@example} parameters; with three input channels and six channels per layer, the config `rgb`, it has {shape:params@rgb}.
The filter side $k$ is odd, so that each filter has a centre pixel to turn about.
A planar CNN with as many planar maps, $4 C_1$ and $4 C_2$, has 4 times as many convolution weights, $4 C_1 C_0 k^2 + 16 C_2 C_1 k^2$, and the same FLOPs: the G-CNN shares each weight across the four rotations. {#c-weight-sharing}
For the example config that is {shape:planar_weights_same_width@example} weights against 612.

The usual comparison holds the parameters fixed instead. A group layer with a stabilizer of $|S|$ elements and $C$ channels in and out stores $|S| C^2 k^2$ weights, so matching a planar layer of $C'$ channels takes $C = C' / \sqrt{|S|}$ channels; the G-CNN then has $\sqrt{|S|}$ times as many planar maps and costs $|S|$ times the FLOPs. {#c-sqrt-rule}
This is what Cohen and Welling do: they divide the number of filters by $\sqrt 4 = 2$ in each p4 layer and by roughly $\sqrt 8 \approx 3$ in each p4m layer, so that the number of parameters is left approximately invariant while the size of the internal representation is increased. {#c-cohen-sqrt}

## Computation and cost

The group correlation does not need a new kernel. Write $g = t s$ with $t$ a translation and $s$ in the stabilizer of the origin, the four rotations; then $[f \star \psi](t s)$ is the planar correlation of $f$ with the filter $L_s \psi$, read at $t$.
For p4 the filter of output plane $(o, r)$ on input plane $(c, s)$ is the stored filter of rotation $s - r$, turned $r$ times:

$$ F^+[o, r, c, s] = L_r\, W[o, c, s - r \bmod 4] . $$ {#eq-bank}

Reshaped to $(4 C_{\text{out}}, 4 C_{\text{in}}, k, k)$, $F^+$ is an ordinary planar filter bank on $4 C_{\text{in}}$ input planes, and one call of a planar correlation computes eq-gconv. {#c-filter-bank}
Each filter of $F^+$ is a permutation of the entries of a stored filter, so the bank holds each stored weight exactly 4 times and building it costs no arithmetic. {#c-bank-permutation}
The layer then costs as much as a planar correlation with $4 C_{\text{out}}$ outputs and $4 C_{\text{in}}$ inputs. {#c-bank-cost}
Cohen and Welling put it the same way: the computational cost is roughly equal to that of a planar convolution with a filter bank that is the same size as the augmented filter bank, because the cost of the filter transformation is negligible, and the filter transformation amounts to a permutation of the entries of each filter. {#c-cohen-cost}
The derivation `split-filter-bank` proves eq-bank from the action property, and its check builds the bank from distinct numbers and compares it with the sum over the group written out.

Over a batch of $B$ images of $n \times n$ with 'same' padding, the forward pass costs

$$ C_\text{fwd} = 2 B n^2 k^2 (4 C_1 C_0 + 16 C_2 C_1) + 2 B C_2 M $$ {#eq-flops}

FLOPs, counting a multiply-add as two and leaving out the biases, ReLUs and poolings. {#c-flops}
For the example config and a batch of 2 images of $5 \times 5$ that is {shape:flops_forward@example} FLOPs, nearly all of it in the group correlation, whose 16 is the 4 output rotations times the 4 input rotations.
The derivation `parameter-and-flop-count` counts the arrays and the multiply-adds of the reference.

## Training

A G-CNN is trained like a CNN, by backpropagation and a gradient method on the stored filters.
Because the bank is a linear copy of the stored weights, the gradient of a stored weight is the sum of the gradients of its four copies in the bank, each turned back:

$$ \frac{\partial \mathcal{L}}{\partial W[o, c, q]} = \sum_{r=0}^{3} L_{r^{-1}}\, \frac{\partial \mathcal{L}}{\partial F^+[o, r, c, q + r]} . $$ {#eq-shared-gradient}

{#c-shared-gradient}
Frameworks with automatic differentiation do this by themselves when the bank is built from the stored filters inside the forward pass; the NumPy reference does it by hand, and its test checks the result against finite differences.
The gradient with respect to the input is equivariant too: the gradient at a turned image is the turned gradient.
Training cannot break the symmetry, because equivariance holds for every value of the filters.

## Inductive biases

The bias of a G-CNN is a chosen symmetry, built into every layer rather than learned from augmented data.
A first reason to want equivariant layers and not only an invariant output is that linear invariant maps are weak.
If $f$ is linear and invariant to a finite group $G$, then $f(x) = f\big(\frac{1}{|G|} \sum_{g} g . x\big)$: it sees only the average of $x$ over the group. {#c-linear-invariant-average}
Bronstein et al. state this for any group and note that in the case of images and translation, this would entail using only the average RGB color of the input. {#c-bronstein-average}
Equivariant linear layers followed by pointwise nonlinearities avoid that: the composition is equivariant, and an invariant pooling at the end gives an invariant that is not a function of the group average alone. {#c-why-equivariant-layers}
The derivation `linear-invariant-is-group-average` proves both statements and its check finds two images with the same rotation average that the reference network tells apart.
Bronstein et al. make this the geometric deep learning blueprint, whose main building blocks are (i) a local equivariant map, (ii) a global invariant map, and (iii) a coarsening operator. {#c-bronstein-blueprint}

A second reason is that convolution is the only way to get equivariance with a linear layer.
Kondor and Trivedi prove that, given some natural constraints, convolutional structure is not just a sufficient but also a necessary condition for equivariance to the action of a compact group. {#c-kondor}
So a [[convolutional-neural-network|convolutional network]] is the G-CNN for the translation group, and the G-CNN is what equivariance to a larger group forces on a linear layer.

The third point is when to stop being equivariant. Pooling over rotations too early throws away the relative orientation of features, which later layers need.
Cohen and Welling tested this: replacing each layer of their rotated MNIST baseline by a p4 correlation followed by coset pooling over rotations gives a test error of 3.21%, against 2.28% for the P4CNN that pools over rotations only after the last layer. {#c-premature-invariance}
The symmetry does not have to be exact to help.
Cohen and Welling note that the CIFAR dataset is not actually symmetric, since objects typically appear upright, yet they see substantial increases in accuracy on it, indicating that there need not be a full symmetry for G-convolutions to be beneficial. {#c-cifar-not-symmetric}

## Evidence

On rotated MNIST, 10000 training and 50000 test digits, Cohen and Welling report a test error of 2.28% for the P4CNN, against 5.03% for a planar CNN of the same depth and about the same number of parameters and 3.98% for the best earlier result, of Schmidt and Roth (2012). {#c-mnist-rot}
On CIFAR-10 they replace the convolutions of All-CNN-C and of a 44-layer ResNet by p4 or p4m correlations with about the same number of parameters.
The test errors without and with moderate augmentation are 9.44% and 8.86% for the planar All-CNN, 8.84% and 7.67% with p4, and 7.59% and 7.04% with p4m; for the ResNet44 they are 9.45% and 5.61% planar and 6.46% and 4.94% with p4m. {#c-cifar}
A wider 26-layer ResNet trained with moderate augmentation has a test error of 5.27% with planar convolutions and 4.19% with p4m convolutions, which they compare with the 4.17% of a wide ResNet of 36.5 M parameters, against their 7.2 M. {#c-wide-resnet}
Weiler and Cesa compare many equivariant models on rotated MNIST and bring the error from 2.28% for the P4CNN down to 0.682% for a model equivariant to 16 rotations and reflections whose symmetry is restricted to the 16 rotations at a later depth. {#c-weiler-mnist}
On CIFAR, equivariant wide ResNets 28/10 of about the same number of parameters reach test errors of 2.91% on CIFAR-10 and 16.22% on CIFAR-100, against 3.87% and 18.80% for the planar model; with AutoAugment the errors are 2.05% and 14.30%, against 2.6% and 17.1%. {#c-weiler-cifar}
These are reported results; this page has not reproduced them.

## Limitations

- **Discrete groups only, in this form.** Cohen and Welling write that one limitation of the method as presented is that it only works for discrete groups, since convolution on continuous groups is well defined but may be hard to approximate in an equivariant manner. {#c-discrete-only}
- **The size of the group.** The bank and the feature maps grow with the stabilizer: p4 multiplies the planar maps by 4 and the cost of a layer of fixed width by 16. Cohen and Welling note the challenge, identified by Gens and Domingos, that a full enumeration of transformations in a group may not be feasible if the group is large. {#c-enumeration}
- **Grid artifacts.** Only the symmetries of the pixel grid are exact. Rotations by 45 degrees need interpolation, and strides, padding and even image sides can break even the exact ones, as the pitfalls below show.
- **The wrong symmetry hurts.** Invariance to a transformation that changes the label, such as a 6 turned into a 9, removes information the task needs; the network cannot learn to break a symmetry built into its layers.

## Variants and descendants

- **p4m and other wallpaper groups.** Adding the reflections gives p4m, with 8 planes per feature map; the same filter-bank construction applies to any discrete group that splits into translations and a finite stabilizer.
- **Steerable CNNs.** Weiler and Cesa give a general description of E(2)-equivariant convolutions in the framework of steerable CNNs, where the feature spaces carry group representations and the kernel constraints reduce to constraints under irreducible representations. {#c-weiler-def}
- **Deep Sets.** For the permutation group, Zaheer et al. prove that a function on a set from a countable universe is invariant to the permutation of its elements if and only if it can be decomposed as $\rho\big(\sum_{x} \phi(x)\big)$. {#c-zaheer-invariant}
  They also prove that a linear layer $\Theta x$ is permutation equivariant if and only if $\Theta = \lambda I + \gamma \mathbf{1}\mathbf{1}^\top$. {#c-zaheer-equivariant}
- **Graph networks.** A [[graph-neural-network|graph neural network]] is equivariant to permutations of the nodes. Satorras et al. build graph networks equivariant to rotations, translations, reflections and permutations, E(n) equivariant graph neural networks, without the higher-order representations of earlier methods. {#c-satorras}
- **Equivariant residual networks.** Since sums of equivariant maps are equivariant, the p4 and p4m versions of [[resnet|residual networks]] are built by replacing every convolution, as in the CIFAR experiments above.
- **Augmentation.** Cohen and Welling report that G-CNNs benefit from data augmentation in the same way as convolutional networks, as long as the augmentation comes from a group larger than G. {#c-cohen-augmentation}

## Notable models

The P4CNN of Cohen and Welling for rotated MNIST and their p4m ResNets for CIFAR-10 are the original examples. The E(2)-steerable wide ResNets of Weiler and Cesa, rotation-equivariant networks for histopathology and microscopy images, where orientation carries no meaning, and E(3)-equivariant networks for molecules and point clouds are the main descendants.

## Implementations

The NumPy reference `impl/numpy/gcnn.py` implements the group action of eq-p4-action, the planar correlation of eq-planar-corr with zero padding or wrapping, the banks of eq-lift and eq-bank, the lifting and group correlations and their backward passes, coset pooling (eq-rotpool), the whole network with its cross-entropy gradient, and the counts eq-params and eq-flops.
It also has `gconv_direct`, the sum over the group of eq-gconv written out, against which the tests check the bank.
The PyTorch version `impl/pytorch/gcnn.py` is a function of a dict of tensors that builds the bank from the stored filters in every forward pass and calls `F.conv2d`, so autograd sums the gradients of the copies; its tests train it in float32 with Adam.
The JAX version `impl/jax/gcnn.py` is a pure function on a parameter dict that calls `lax.conv_general_dilated`, tested under `jax.jit` and `jax.vmap`; `test_gradient_of_the_shared_weights_sums_the_copies` checks that its gradient of a stored filter is the sum of eq-shared-gradient.
All three are tested against the shared fixture of `fixtures/`: the example config, a batch of 2 images of $5 \times 5$ with their labels, the logits, the loss and the gradients.

### Worked example

Take the example config: one input channel, 4 channels in each p4 layer, $3 \times 3$ filters and 3 classes.
The lifting layer stores $4 \cdot 1 \cdot 9 = 36$ weights and 4 biases, and its bank has $4 \cdot 4 = 16$ filters, each stored filter in its four orientations.
The group layer stores $4 \cdot 4 \cdot 4 \cdot 9 = 576$ weights and 4 biases. Its bank has 16 output planes and 16 input planes: the filter from input plane $(c, s)$ to output plane $(o, r)$ is the stored plane $s - r$ of filter $(o, c)$, turned $r$ times. For output rotation $r = 1$ and input rotation $s = 0$ that is the stored plane $3$, turned once.
The head has $3 \cdot 4 + 3 = 15$ parameters, so the network has $40 + 580 + 15 = 635$ in all, as eq-params says.
A planar CNN with 16 and 16 planar maps would store $16 \cdot 9 + 256 \cdot 9 = 2448$ convolution weights, four times the $36 + 576 = 612$ of the G-CNN, and cost the same FLOPs.
Turn the input image a quarter of the way round: each of the 16 maps of the first layer turns, and plane $r$ moves to plane $r + 1$; the same happens after the second layer; the maximum over the planes then gives 4 planar maps that turn with the image, and their means, and the logits, do not change.
The tests `test_lift_is_rotation_equivariant`, `test_gconv_is_rotation_equivariant` and `test_logits_are_invariant_to_rotations` check each step, and `test_planar_cnn_of_the_same_shape_is_not_invariant` checks that the same network without the rotated copies is not invariant.

### Pitfalls

> [!pitfall]
> **A bias per rotation plane.** Giving each of the 4 planes of a G-feature map its own bias breaks equivariance, because turning the input moves each plane under another plane's bias. The test `test_bias_per_rotation_plane_breaks_equivariance` checks that a per-plane bias breaks it and a per-map bias keeps it; the same holds for the scale and shift of batch normalization.

> [!pitfall]
> **Rotating the filters without cycling them.** In a group correlation the filter is itself a p4 map, so moving it by a rotation turns each of its planes and also cycles them, $s - r$ in eq-bank. The test `test_rotating_filters_without_cycling_breaks_equivariance` builds the bank with the turn but without the cycle and finds the layer no longer equivariant.

> [!pitfall]
> **Strides on an even grid.** Rotation about the centre of an $n \times n$ grid maps the pixels with even indices to themselves only when $n$ is odd. The test `test_stride_two_on_an_even_grid_breaks_rotation_equivariance` checks stride 2 on grids of side 7 and 6: equivariant on the first, not on the second. Networks that downsample should keep odd sides or subsample on a subgroup that the grid respects.

> [!pitfall]
> **Pooling over rotations too early.** Coset pooling makes a map invariant to rotations about each point, and with it every later layer forgets how features were oriented relative to each other. The test `test_pooling_rotations_early_loses_relative_orientation` takes two p4 maps, one with two features at right angles and one with the same features parallel: pooled at once they are equal, while a group correlation before pooling scores them 2 and 1.

## Experiments

This page has no experiment yet. The checks of its five derivations play the part of small experiments: they test the equivariance of each layer type on random inputs, build the filter bank from distinct numbers, compare the gradient of the shared weights with finite differences, find two images that a linear invariant cannot tell apart and the G-CNN can, and count the parameters and multiply-adds against a planar CNN of the same width.

## History

Cohen and Welling place G-CNNs among earlier equivariant models, including transforming autoencoders, equivariant Boltzmann machines and the networks of Dieleman et al. (2015), which exploited rotation symmetry for galaxy morphology prediction by rotating feature maps. {#c-hist-dieleman}
Cohen and Welling presented Group Equivariant Convolutional Networks at ICML 2016. {#c-hist-icml}
Steerable CNNs, which work with representations of the group instead of enumerating its elements, were posted by Cohen and Welling at the end of 2016; Deep Sets treated the permutation group the same year; Kondor and Trivedi proved in 2018 that equivariant linear layers must be convolutions; Weiler and Cesa gave the general solution for planar rotations and reflections in 2019; and the geometric deep learning proto-book of Bronstein, Bruna, Cohen and Veličković collected grids, groups, graphs, geodesics and gauges in one framework in 2021.

## References

- T. S. Cohen and M. Welling. Group Equivariant Convolutional Networks. ICML 2016.
- M. M. Bronstein, J. Bruna, T. Cohen and P. Veličković. Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges. arXiv 2104.13478, 2021.
- R. Kondor and S. Trivedi. On the Generalization of Equivariance and Convolution in Neural Networks to the Action of Compact Groups. ICML 2018.
- M. Weiler and G. Cesa. General E(2)-Equivariant Steerable CNNs. NeurIPS 2019.
- M. Zaheer, S. Kottur, S. Ravanbakhsh, B. Póczos, R. Salakhutdinov and A. J. Smola. Deep Sets. NIPS 2017.
- V. G. Satorras, E. Hoogeboom and M. Welling. E(n) Equivariant Graph Neural Networks. ICML 2021.
