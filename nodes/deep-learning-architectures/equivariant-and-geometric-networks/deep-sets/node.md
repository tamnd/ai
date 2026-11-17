---
tag: NEW-deep-sets
kind: architecture
slug: deep-sets
name: Deep Sets
aliases: [DeepSets, Deep Sets network, sum-decomposition, permutation-invariant network, permutation-equivariant layer, set pooling]
area: deep-learning-architectures/equivariant-and-geometric-networks
level: core
summary: >-
  A network for sets: a shared network phi maps each element, the results are summed, and a network rho reads the sum, so the output does not depend on the order of the elements.
uses: [matrix-multiplication@m2, permutation-matrix@m2, backpropagation@al]
related_to: [group-equivariant-convolutional-network, graph-neural-network, transformer, multi-head-attention, multilayer-perceptron, long-short-term-memory, gated-recurrent-unit, vision-transformer, latent-dirichlet-allocation@al, dropout@al]
history:
  year: 2017
  people: [Manzil Zaheer, Satwik Kottur, Siamak Ravanbakhsh, Barnabás Póczos, Ruslan Salakhutdinov, Alexander J. Smola]
  note: "Zaheer et al. posted Deep Sets to arXiv (1703.06114) in March 2017 and published it at NIPS 2017. PointNet (Qi et al.), posted in December 2016, pooled the points of a cloud by a maximum; Wagstaff et al. (2019) showed how large the summed representation must be, and the Set Transformer (Lee et al., 2019) replaced the sum by attention."
prereqs: [multilayer-perceptron, permutation-matrix@m2]
same_as:
  arxiv: ["1703.06114"]
sources:
  - {paper: "Zaheer, Kottur, Ravanbakhsh, Póczos, Salakhutdinov and Smola (2017)", title: "Deep Sets", locator: "NIPS 2017; arXiv 1703.06114v3, Sections 2 to 4, Tables 1 to 6 and Appendices A to C, H and I", note: "the theorem on invariant functions, the lemma on equivariant layers, the architecture and its applications"}
  - {paper: "Wagstaff, Fuchs, Engelcke, Posner and Osborne (2019)", title: "On the Limitations of Representing Functions on Sets", locator: "ICML 2019; arXiv 1901.09006v2, Sections 3 and 4", note: "discontinuity in the countable case, and the latent dimension needed for continuous decompositions"}
  - {paper: "Qi, Su, Mo and Guibas (2017)", title: "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation", locator: "arXiv 1612.00593v2, Sections 4.2 and 4.3, Table 1", note: "max pooling as the symmetric function, and its approximation theorem"}
  - {paper: "Lee, Lee, Kim, Kosiorek, Choi and Teh (2019)", title: "Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks", locator: "ICML 2019; arXiv 1810.00825v3, Sections 1 and 3", note: "attention between the elements, inducing points and pooling by attention"}
family: set
inputs: "a batch of sets as a (B, M, D) array of M elements of D features, with a (B, M) mask that marks the real elements when the sets have different sizes"
outputs: "a (B, K) array of class scores invariant to the order of the elements, or a (B, M) array of element scores that are permuted with the elements"
components: [multilayer-perceptron]
inductive_biases: ["the order of the elements carries no information: the output of the invariant model is the same for every ordering, and that of the equivariant model is permuted with them", "every element is processed by the same weights, so the number of parameters does not depend on the size of the set and the network accepts sets of any size", "elements interact only through a pooled summary of the whole set, a sum, a mean or a maximum, never pair by pair"]
params: "2 D H + H + (L - 1)(2 H^2 + H) + H^2 + H + H K + K for L equivariant layers of width H on elements of D features, a hidden layer of rho of width H and K classes; independent of the set size M"
flops_per_example: "2 (M + 1) D H + (L - 1) 2 (M + 1) H^2 + 2 H^2 + 2 H K for a set of M elements, counting a multiply-add as 2 FLOPs and only the matrix products"
---

Deep Sets is a design for neural networks whose input is a set: a bag of points, of words, of images or of measurements, with no order among them.
Each element $x$ goes through the same network $\phi$, the outputs are added, and a second network $\rho$ reads the sum, $f(X) = \rho\big(\sum_{x \in X} \phi(x)\big)$.
Because a sum does not depend on the order of its terms, neither does $f$.
Zaheer et al. introduced it in 2017 with a theorem that this form characterizes the permutation invariant functions on sets, and a lemma that gives the necessary and sufficient conditions for a layer to be permutation equivariant. {#c-def}
The same idea, a shared network on each element and a symmetric pooling, runs through point-cloud networks, multiple-instance learning, the readout of [[graph-neural-network|graph neural networks]] and the analysis of attention.

## In plain words

A list has a first element and a last one; a set does not.
A network that reads a set as a list, one element after another as a [[long-short-term-memory|recurrent network]] would, gives different answers for the same set written in different orders, and must learn from data that the order does not matter.
Deep Sets builds that fact into the network.
It looks at each element alone, through the same small network, and writes down a summary of a few numbers for it.
It then adds up the summaries of all the elements, which is the step that forgets the order, and a second network turns the total into the answer.
When each element needs its own answer, such as which member of a group is the odd one out, the layers let each element see the total of the others as well as itself, and the answers move with the elements when they are reordered.

## Overview

A function on sets must be permutation invariant: for any permutation $\pi$, $f(\{x_1, \dots, x_M\}) = f(\{x_{\pi(1)}, \dots, x_{\pi(M)}\})$. {#c-property}
When each element has its own label, the task is instead to learn a permutation equivariant function, which permutes its outputs when its inputs are permuted. {#c-equivariant-def}
The paper's Theorem 2 states that a function on sets whose elements come from a countable universe is a valid set function, that is, invariant to the permutation of the elements, if and only if it can be decomposed as $\rho\big(\sum_{x \in X} \phi(x)\big)$ for suitable transformations $\phi$ and $\rho$. {#c-theorem2}
For an uncountable universe such as the real numbers, the authors could prove the decomposition only for sets of a fixed size, and conjectured that it holds in general. {#c-uncountable}
For layers, Lemma 3 states that a layer $\sigma(\Theta x)$ on $M$ inputs is permutation equivariant if and only if all the diagonal elements of $\Theta$ are equal and all the off-diagonal elements are tied together, $\Theta = \lambda I + \gamma \mathbf{1}\mathbf{1}^\top$. {#c-lemma3}

The architecture follows from the two results.
In the invariant model each element is transformed, possibly by several layers, into a representation $\phi(x_m)$; the representations are added up, and the sum is processed by the network $\rho$ like the input of any deep network; the key is to add up all representations and then apply nonlinear transformations. {#c-invariant-model}
In the equivariant model, layers of the form of Lemma 3 are stacked, since the composition of permutation equivariant functions is also permutation equivariant. {#c-stacking}
The authors name as their contributions the universality result of Theorem 2, the permutation equivariant layer, which identifies the necessary and sufficient form of parameter sharing in a standard neural layer, and new applications. {#c-contributions}

## Components

### The invariant model

$$ f(X) = \rho\Big(\sum_{m=1}^{M} \phi(x_m)\Big), \qquad \phi : \mathbb{R}^D \to \mathbb{R}^H, \quad \rho : \mathbb{R}^H \to \mathbb{R}^K . $$ {#eq-invariant}

$\phi$ and $\rho$ are [[multilayer-perceptron|multilayer perceptrons]].
Replacing $\phi$ and $\rho$ by universal approximators leaves the theorem's conclusion unchanged, since they can approximate arbitrary polynomials. {#c-universal-approximators}
The gradient of the loss with respect to the weights of $\phi$ is a sum over the elements of the gradients through each copy of $\phi$, the usual rule for tied weights; the paper writes it out and argues that its result proves parameter tying to be the only way to handle irrelevant order. {#c-gradient}
If extra information $z$ comes with the set, such as an image whose tags form the set, the networks can be conditioned on it, $\phi(x_m \mid z)$. {#c-conditioning}

### The equivariant layer

For one feature per element, Lemma 3 says the weight matrix of an equivariant layer is

$$ \Theta = \lambda I + \gamma\, \mathbf{1}\mathbf{1}^\top, \qquad \sigma(\Theta x) = \sigma\big(\lambda x + \gamma\, (\textstyle\sum_m x_m)\, \mathbf{1}\big), $$ {#eq-lemma3}

a nonlinearity applied to a weighted combination of the input and of the sum of the inputs. {#c-lemma3-reading}
The proof shows that $\Theta$ must commute with every permutation matrix: a transposition of two elements forces two diagonal entries to be equal, and a product of transpositions moves any off-diagonal entry onto any other. {#c-lemma3-proof}
With $D$ input and $D'$ output features per element, $x \in \mathbb{R}^{M \times D}$, the layer becomes $\sigma(x\Lambda - \mathbf{1}\mathbf{1}^\top x\, \Gamma)$ with $\Lambda, \Gamma \in \mathbb{R}^{D \times D'}$. {#c-multichannel}
The page adds a bias and lets the pooled summary be a sum, a mean or a maximum:

$$ h' = \sigma\big(h\Lambda - \mathbf{1}\,\operatorname{pool}(h)\,\Gamma + \mathbf{1}\beta\big), \qquad \operatorname{pool}(h) \in \mathbb{R}^{1 \times D} . $$ {#eq-equivariant}

The derivation `equivariant-layers` checks that the linear maps commuting with every permutation of $M \geq 2$ elements of $D$ features to $D'$ features form a space of dimension exactly $2DD'$, the pairs $(\Lambda, \Gamma)$, and that eq-equivariant is equivariant for each of the three pools. {#c-commutant}
The paper's variant with a maximum, $\sigma(\lambda I x + \gamma \operatorname{maxpool}(x) \mathbf{1})$, performs better in some applications, which the authors suggest may be because for $\lambda = \gamma$ the input to the nonlinearity is max-normalized. {#c-maxpool}
To reduce the number of parameters further they tie the two matrices, $\sigma\big(\beta + (x - \mathbf{1}\operatorname{maxpool}(x))\Gamma\big)$, which subtracts the maximum of each feature over the set before one linear map. {#c-tied}
Any commutative pooling over the elements of the output of equivariant layers gives a permutation invariant function. {#c-pool-invariant}

### Examples of decompositions

The appendix gives decompositions of simple symmetric functions. {#c-examples-intro}
With $\phi(x) = (x, x^2, x^3)$ and $\rho(u, v, w) = uv - w + \tfrac{3}{2}(u^2 - v)$, $\rho(\phi(x_1) + \phi(x_2)) = x_1 x_2 (x_1 + x_2 + 3)$; with $\phi(x) = (1, x)$ and $\rho(u, v) = v/u$ the decomposition is the mean; and with $\phi(x) = (e^{\alpha x}, x e^{\alpha x})$ and the same $\rho$, a softmax-weighted mean, it approaches the maximum as $\alpha \to \infty$. {#c-examples}
The appendix also offers $\rho(u, v) = \big(v - (v/u) e^{\alpha v/u}\big) / \big(u - e^{\alpha v/u}\big)$ for the second largest element, but since $v - (v/u)e^{\alpha v/u} = (v/u)(u - e^{\alpha v/u})$, this expression equals $v/u$ wherever it is defined and tends to the largest element, not the second. {#c-second-largest}
The derivation `decompositions` checks all of these numerically, the last one included.

### Variable set sizes in a batch

Sets of different sizes are padded to a common $M$ and a mask marks the real elements.
Every pool must skip the padding: a sum that includes padded zeros changes by $M - |X|$ copies of $\phi(0)$, which is not zero for a network with biases, and a maximum must ignore the padded rows. {#c-padding}
The reference implementation pools over the masked elements only and sets the outputs of the padding to zero.

## Shapes and parameters

An equivariant layer from $D_\ell$ to $D_{\ell+1}$ features has $2 D_\ell D_{\ell+1} + D_{\ell+1}$ parameters, $\Lambda$, $\Gamma$ and the bias; the readout has a hidden layer of $\rho$ of width $H$ and a linear layer to $K$ classes, so the network of this page has

$$ N_{\text{params}} = 2DH + H + (L - 1)(2H^2 + H) + H^2 + H + HK + K $$ {#eq-params}

parameters, whatever the size of the sets. {#c-params}
In the shape contract this is

{shape:params}

and the example network, elements of 3 features, 2 layers of width 8 and 5 classes, has {shape:params@example} parameters.

For point clouds the paper uses 3 permutation-equivariant layers with 256 channels followed by max-pooling over the set, a fully connected layer with 256 units and a 40-way softmax, with tanh activations and 50% dropout on the layers after the pooling; for 5,000 points it uses 512 units in all layers. {#c-pointcloud-arch}
It does not say which of its equivariant layers it used; with the form of eq-equivariant, eq-params gives {shape:params@pointcloud_1000} parameters for the network of 256 channels and {shape:params@pointcloud_5000} for that of 512. {#c-params-models}
The tied layer of the paper's eq. 23 would have $DD' + D'$ parameters per layer instead of $2DD' + D'$.
The derivation `parameter-and-flop-count` proves these counts.

## Computation and cost

A forward pass multiplies each element by $\Lambda$ and the pooled summary of each set by $\Gamma$, once per set; only the first grows with the set:

$$ C = 2(M + 1) D H + (L - 1)\, 2(M + 1) H^2 + 2H^2 + 2HK $$ {#eq-flops}

FLOPs per set of $M$ elements, counting a multiply-add as 2. {#c-flops}
The cost is linear in $M$, the cost of applying an elementwise network to each element.
The point-cloud network of 256 channels costs {shape:flops_forward@pointcloud_1000} per cloud of 1,000 points, of which the products with $\Gamma$ are {shape:flops_pool_terms@pointcloud_1000}, about 0.1%. {#c-flops-models}
One layer of self-attention of the same width over the same 1,000 points would spend $4M^2H = 1{,}024{,}000{,}000$ FLOPs in the products $QK^\top$ and $AV$ alone, about 3.9 times the whole Deep Sets network. {#c-attention-compare}
The derivation `parameter-and-flop-count` counts the products of the reference implementation and checks both numbers.

## Training

Deep Sets is trained like any feedforward network, by [[al:backpropagation|backpropagation]] through the copies of $\phi$ and the pool.
The point-cloud network normalizes each set in its first layer to have zero mean along each axis and unit global variance. {#c-normalize}
Dropout applied to the permutation-equivariant layers deteriorated the performance on point clouds, and is used only after the pooling; for the outlier detector, a dropped feature is dropped in all the members of a set at once. {#c-dropout}
Different types of equivariant layer, and as few as 64 channels in the set layers, changed the point-cloud accuracy by less than 5%. {#c-robust-width}
With 5,000 points the inputs were rotated at random around the vertical axis and scaled by a factor drawn from $U(0.8, 1/0.8)$, and the optimizer was Adamax with a learning rate of 0.0005 instead of Adam with 0.001. {#c-pointcloud-train}
For set expansion the score of a candidate given a set is trained with a margin loss, so that a candidate that belongs to the set scores higher than one that does not by a margin. {#c-margin}

## Inductive biases

**The order of the elements carries no information.**
The invariant model gives one output for every ordering of a set, and the equivariant layers permute their outputs with their inputs, so no amount of data can teach the network to use the order, and none is needed to teach it to ignore the order.
`test_invariance_and_equivariance` checks this to rounding error for every pool.

**Shared weights and any set size.**
Every element goes through the same weights, so the number of parameters is fixed and the network reads sets of any size, larger at test time than in training.

**Interactions only through a pooled summary.**
In an equivariant layer an element sees the others only through the sum, mean or maximum of their features, never pair by pair.
Lee et al. argue that since every element is processed independently in set pooling, some information about the interactions between the elements is necessarily discarded, which can make some problems unnecessarily difficult, such as amortized clustering. {#c-set-transformer-motivation}

**A summary of fixed width.**
The whole set passes through a vector of $H$ numbers before $\rho$ reads it.
Wagstaff et al. prove that for every $M > N$ there are continuous permutation invariant functions of $M$ real numbers that are not continuously sum-decomposable through $\mathbb{R}^N$: representing arbitrary continuous functions on sets of size $M$ needs a latent space of dimension at least $M$. {#c-wagstaff-lower}
They also prove that dimension $M$ suffices, for sets of exactly $M$ real numbers and for sets of at most $M$. {#c-wagstaff-upper}
The fixed-size proof of Zaheer et al. uses $\phi(x) = (1, x, x^2, \dots, x^M)$, whose sum, the power sums of the set, determines the set through the Newton-Girard formulae; its latent space grows with the set. {#c-power-sums}
Fewer power sums do not suffice: the sets $\{1, 5, 6\}$ and $\{2, 3, 7\}$ have the same sum, 12, and the same sum of squares, 62. {#c-power-sums-collision}
The derivation `power-sums` checks that the first $M$ power sums recover random sets of $M$ numbers and the collision.

## Evidence

The paper evaluates Deep Sets on seven tasks: estimating statistics of populations, summing digits, classifying point clouds, estimating the red-shift of galaxies, retrieving sets of words, tagging images and finding the outlier of a set. {#c-applications}
In a test of generalization, models trained to sum sets of at most 10 digits were tested on sets of up to 100; Deep Sets generalized much better than LSTMs and GRUs with a similar number of layers and parameters. {#c-digits}
On sets of MNIST images the accuracy falls with the number of images, as it must: with a single-digit error of about $p = 0.01$, at least one of 50 images is misclassified with probability $1 - (1 - p)^{50}$, about 40%, which the authors find matches the observed accuracy. {#c-digits-mnist}
On point clouds from ModelNet40, Deep Sets reached 90 ± 0.3% with 5,000 points and 82 ± 2% with only 100, against 90.1% for MVCNN on multi-view images and 95.54% for the VRN ensemble on voxels. {#c-modelnet}
In the appendix's table, Deep Sets with 1,000 points reached 87 ± 1%, the same network with pooling alone 83 ± 1%, and a graph convolution on a nearest-neighbour graph of the points 58 ± 2%; the error bars come from the choice of points at test time, over three trials. {#c-modelnet-appendix}
PointNet, a network of the same family that pools its points by a maximum, reported 89.2% overall accuracy on ModelNet40. {#c-pointnet}
On estimating the red-shift of each galaxy of a cluster, with the cluster as the set, Deep Sets had a scatter of 0.023, against 0.025 for redMaPPer and 0.026 for an MLP. {#c-redshift}
In text concept set retrieval, where each set is the 50 top words of a topic of a [[al:latent-dirichlet-allocation|latent Dirichlet allocation]] model fit to Wikipedia, Deep Sets outperformed the other approaches on LDA-3k and LDA-5k by any metric, while on the smaller LDA-1k nearest neighbours in word2vec space were better and Deep Sets was 0.5% lower in Recall@10. {#c-lda}
On CelebA, 18,000 sets of 16 face images in which all but one share at least two attributes, an equivariant network found the outlier in 75% of the test sets; a baseline with set pooling followed by fully connected layers, not equivariant, reached about 6.3%, the accuracy of a random choice. {#c-celeba}
The paper's claims of universality concern what the form $\rho(\sum \phi)$ can represent, not what a network of given width learns; Wagstaff et al. found that on the median of a set, a model's error falls as the latent dimension grows, and the latent dimension at which it stops falling grows roughly linearly with the set size. {#c-median}

## Limitations

- **The countable case needs discontinuous maps.** The proof of Theorem 2 maps each element to $4^{-c(x)}$ for an enumeration $c$ of the universe, so that the sum encodes the set as the digits of one real number. {#c-countable-proof} Wagstaff et al. prove that there are functions on sets of rational numbers for which every sum-decomposition through $\mathbb{R}$ has a $\phi$ discontinuous at every rational number. {#c-wagstaff-discontinuous} A network with continuous activations is a continuous function, so it cannot be such a $\phi$.
- **Fixed-width summaries.** A latent space smaller than the set size cannot represent every continuous function of the set; see the inductive biases above.
- **No pairwise interactions.** The pooled summary is the only channel between elements; relational tasks need several equivariant layers or attention.
- **The variable-size case is open.** The paper's theorem for uncountable universes covers sets of one fixed size; for sets of all sizes, what continuity means is not clear, as the space of all subsets of $[0, 1]$ has no natural topology. {#c-topology}

## Variants and descendants

- **Max pooling.** PointNet uses a single symmetric function, max pooling, over per-point features, and proves that a continuous set function can be approximated arbitrarily well by a continuous per-point map followed by an elementwise maximum and a continuous function. {#c-pointnet-theorem}
- **Set Transformer** (Lee et al., 2019). Self-attention between the elements of the set, built from [[multi-head-attention|multi-head attention]] blocks without position encodings, which encodes pairwise and higher-order interactions; induced set attention with $m$ learned inducing points reduces its cost from $O(n^2)$ to $O(nm)$; pooling by multihead attention with learned seed vectors. {#c-set-transformer}
- **Transformers without positions.** Self-attention with no position encoding is itself permutation equivariant, which is why a [[vision-transformer|Vision Transformer]] without position embeddings reads an image as a bag of patches.
- **Graph neural networks.** Message passing applies the idea to the neighbours of each node: a shared message function and a symmetric aggregation.
- **Group equivariant networks.** Deep Sets is equivariance to the symmetric group; [[group-equivariant-convolutional-network|group equivariant CNNs]] build the same kind of parameter sharing for rotations and reflections.

## Notable models

- **The point-cloud classifier** of the paper, 3 equivariant layers of 256 channels with a max readout, and its version with 512 units on 5,000 points.
- **PointNet**, the per-point network with max pooling used in much later work on 3D point clouds.
- **The Set Transformer**, the attention-based successor.

## Implementations

The NumPy reference `impl/numpy/deepsets.py` implements eq-equivariant with sum, mean and max pooling, the layers without $\Gamma$ that make eq-invariant, a set head and an element head, masks for sets of different sizes, eq-loss and its gradient by hand through every layer down to the input, and the counts eq-params and eq-flops.
The PyTorch version `impl/pytorch/deepsets.py` is an `nn.Module` with the same parameters; the tests compare an equivariant layer with the dense layer $\lambda I + \gamma\mathbf{1}\mathbf{1}^\top$ of Lemma 3 on the flattened set, check `torch.nn.MultiheadAttention` without positions for the same symmetry, run `torch.autograd.gradcheck`, and train two small networks.
The JAX version `impl/jax/deepsets.py` is a pure function of a parameter dictionary, tested under `jax.jit` and `jax.vmap` and with `jax.grad` against the NumPy backward pass.
All three load the shared fixture of `fixtures/`: the example network on two random sets, one of them padded, with its scores, loss and gradients.

The training loss for a set head is the cross-entropy of the labels averaged over a batch of $B$ sets,

$$ \mathcal{L} = -\frac{1}{B} \sum_{b=1}^{B} \log \operatorname{softmax}\big(f(X_b)\big)_{y_b} , $$ {#eq-loss}

and for an element head the same with the softmax taken over the elements of the set and $y_b$ the position of the outlier.

### Worked example

Take one feature per element and the set $\{1, 2, 4\}$ in that order, $x = (1, 2, 4)$.
An equivariant layer with $\lambda = 1$, $\gamma = -1/3$ and no activation subtracts the mean, $7/3$, and gives $(-4/3, -1/3, 5/3)$.
Its matrix is $\Theta = I - \tfrac{1}{3}\mathbf{1}\mathbf{1}^\top$, with $2/3$ on the diagonal and $-1/3$ elsewhere, as Lemma 3 requires.
Reorder the input to $(4, 1, 2)$ and the output is $(5/3, -4/3, -1/3)$, the same numbers in the new order.
Sum-pooling the output gives 0 in both orders, and max-pooling gives $5/3$.
With $\phi(x) = (1, x)$ and $\rho(u, v) = v/u$ the invariant model returns the mean $7/3$, and with $\phi(x) = (x, x^2, x^3)$ the power sums are $(7, 21, 73)$, from which the Newton-Girard formulae recover the elementary symmetric polynomials $e_1 = 7$, $e_2 = 14$, $e_3 = 8$, the polynomial $t^3 - 7t^2 + 14t - 8 = (t - 1)(t - 2)(t - 4)$, and so the set.
The test `test_worked_example` checks these numbers.

### Pitfalls

> [!pitfall]
> **Pooling over the padding.** When sets of different sizes are padded into one array, a sum or mean that includes the padded rows, or a maximum that can pick them, makes the output depend on the amount of padding. Pass the mask to every pool, not only the last one; `test_padding_does_not_change_the_output` checks that a set gives the same scores alone and padded.

> [!pitfall]
> **Pooling over the wrong axis.** In a (B, M, D) array the set axis is 1. Pooling over the batch axis mixes sets, and pooling over the features gives a network that runs and is no longer equivariant. The invariance tests catch both.

> [!pitfall]
> **Batch normalization across the elements.** Normalizing with statistics of the batch makes each set's output depend on the other sets in the batch; the paper uses batch normalization only in the convolutional layers of the outlier detector, before the set layers. {#c-batchnorm}

> [!pitfall]
> **Taking a small latent width for granted.** A sum through a vector of $H$ numbers cannot represent every continuous function of sets larger than $H$; when the target depends on many elements individually, such as a median or a sorting, widen $H$ or add interactions.

> [!pitfall]
> **Elementwise scores for a set-level choice.** A network that scores each element alone, with no pooled term, cannot tell which element is the odd one out of its set; the paper's non-equivariant baseline was at chance. `test_the_interaction_is_needed` reproduces this on a small task.

## Experiments

This page has no experiment yet.
The PyTorch tests `test_sum_of_digits_generalizes` and `test_the_interaction_is_needed` are small ones.
In the first, a network with sum pooling and no interaction learns the sum of a set of at most 10 digits, given as one-hot vectors, by least squares, and is tested on sets of 50 digits: rounded to the nearest integer, its output is exactly right on more than 95% of the test sets.
In the second, each set has 8 points in the plane around a random centre, 7 on a circle of radius 2 and one near the centre, which is the one to find.
An equivariant network with max pooling, trained for 500 steps of Adam, finds it in more than 90% of new sets; the same network with $\Gamma$ removed, whose score for a point depends on that point alone, finds it in fewer than 20%, near the 12.5% of a random choice, since the centre of a set can be anywhere.
The checks of the four derivations verify Lemma 3, the decompositions of the appendix, the power sums and the counts.

## History

Pooling a function over the members of a set was not new in 2017: the paper cites pooling over a set of samples for causal discovery, over panoramic projections and multiple views of 3D objects, and over the rows and columns of the payoff matrix of a normal-form game, the actions of its players. {#c-hist-pooling}
A special case of Lemma 3 had been used as the architecture of networks for sensor networks and multi-agent settings. {#c-hist-multiagent}
The paper relates Theorem 2 to de Finetti's theorem, under which an exchangeable model factors through a latent parameter, and shows that for exponential families with conjugate priors the marginal likelihood of a set depends on it only through the sum of the sufficient statistics of its elements. {#c-de-finetti}
Zaheer et al.'s paper, arXiv 1703.06114, was published at the 31st Conference on Neural Information Processing Systems (NIPS 2017). {#c-hist-nips}
PointNet had been posted in December 2016; in 2019 Wagstaff et al. bounded the latent dimension and Lee et al. introduced the Set Transformer.

## References

- Zaheer, M., Kottur, S., Ravanbakhsh, S., Póczos, B., Salakhutdinov, R. and Smola, A. J. (2017). Deep Sets. NIPS 2017. arXiv 1703.06114.
- Wagstaff, E., Fuchs, F. B., Engelcke, M., Posner, I. and Osborne, M. (2019). On the Limitations of Representing Functions on Sets. ICML 2019. arXiv 1901.09006.
- Qi, C. R., Su, H., Mo, K. and Guibas, L. J. (2017). PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation. arXiv 1612.00593.
- Lee, J., Lee, Y., Kim, J., Kosiorek, A. R., Choi, S. and Teh, Y. W. (2019). Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks. ICML 2019. arXiv 1810.00825.
