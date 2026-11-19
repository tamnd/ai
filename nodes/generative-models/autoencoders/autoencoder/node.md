---
tag: NEW-autoencoder
kind: architecture
slug: autoencoder
name: Autoencoder
aliases: [auto-encoder, autoassociator, auto-associative network, deep autoencoder, denoising autoencoder, DAE, contractive autoencoder, CAE, stacked denoising autoencoder, SDAE]
area: generative-models/autoencoders
level: core
summary: >-
  A network trained to reproduce its input: an encoder maps the input to a code, a decoder maps the code back, and training lowers the reconstruction error.
uses: [matrix-multiplication@m2, frobenius-norm@m2, cross-entropy@m2, activation-function@al, principal-component-analysis@al]
related_to: [multilayer-perceptron, sparse-autoencoders, weight-decay, dropout, adam]
history:
  year: 1986
  people: [David E. Rumelhart, Geoffrey E. Hinton, Ronald J. Williams, Pierre Baldi, Kurt Hornik]
  note: "Rifai et al. date the auto-encoder to the late 1980s, citing Rumelhart, Hinton and Williams (1986) and Baldi and Hornik (1989); Hinton and Salakhutdinov (Science, 2006) trained deep autoencoders by pretraining them layer by layer, Vincent et al. (2008, 2010) added the denoising criterion and Rifai et al. (2011) the contractive penalty."
prereqs: [backpropagation@al, multilayer-perceptron, principal-component-analysis@al, cross-entropy@m2]
same_as:
  wikipedia: Autoencoder
sources:
  - {paper: "Hinton and Salakhutdinov (2006)", title: "Reducing the Dimensionality of Data with Neural Networks", locator: "Science 313, pp. 504 to 507", note: "the deep autoencoder, its pretraining and its comparison with PCA and LSA"}
  - {paper: "Vincent, Larochelle, Lajoie, Bengio and Manzagol (2010)", title: "Stacked Denoising Autoencoders: Learning Useful Representations in a Deep Network with a Local Denoising Criterion", locator: "JMLR 11, pp. 3376 to 3382, 3387 and 3394", note: "the encoder, the decoder and the losses, the denoising criterion, the corruptions, stacking, the filters and Table 3"}
  - {paper: "Rifai, Vincent, Muller, Glorot and Bengio (2011)", title: "Contractive Auto-Encoders: Explicit Invariance During Feature Extraction", locator: "ICML 2011, pp. 1 to 5", note: "the contractive penalty, its sigmoid form and cost, its relation to weight decay and to denoising, and Table 1"}
  - {paper: "Plaut (2018)", title: "From Principal Subspaces to Principal Components with Linear Autoencoders", locator: "arXiv 1804.10253v3, pp. 1, 3 and 4", note: "the linear autoencoder spans the principal subspace but does not find the principal directions"}
  - {paper: "Kunin, Bloom, Goeva and Seed (2019)", title: "Loss Landscapes of Regularized Linear Autoencoders", locator: "ICML 2019; arXiv 1901.08168v2, pp. 1 to 4", note: "the symmetry of the loss, the Transpose Theorem and the principal directions under weight decay"}
family: autoencoder
inputs: "a batch of inputs as a (B, D) array, and for a denoising autoencoder a corruption process"
outputs: "the code, a (B, C) array, and the reconstruction, a (B, D) array"
components: []
inductive_biases: ["a bottleneck or a penalty: the network must not be able to copy its input exactly", "the data lie near a low-dimensional set that a smooth code can parametrize", "the same loss for every component of the input", "with denoising, the clean input is predictable from a corrupted one; with contraction, the code should not move when the input moves a little"]
params: "2 sum_k n_k n_{k+1} weights plus sum_{k>=1} n_k + sum_{k<K} n_k biases for an encoder of widths n_0 = D, ..., n_K = C and a mirrored decoder; tying the weights halves the weights"
flops_per_example: "4 sum_k n_k n_{k+1} for one encoding and decoding, counting a multiply-add as 2 FLOPs and only the matrix products"
---

An autoencoder is a neural network trained to reproduce its input at its output.
It has two halves: an encoder that maps the input $\mathbf{x}$ to a code $\mathbf{y}$, and a decoder that maps the code back to a reconstruction $\mathbf{z}$; training lowers a reconstruction error between $\mathbf{x}$ and $\mathbf{z}$, so no labels are needed.
Hinton and Salakhutdinov put the idea in one sentence: high-dimensional data can be converted to low-dimensional codes by training a multilayer neural network with a small central layer to reconstruct high-dimensional input vectors, and gradient descent can fine-tune the weights of such autoencoder networks, but this works well only if the initial weights are close to a good solution. {#c-hinton-def}
Copying is easy if nothing stops it, so every useful autoencoder is constrained: a code narrower than the input, noise on the input that the network must remove, or a penalty on how fast the code changes with the input.
This page covers the deterministic autoencoder and these three constraints; the [[sparse-autoencoders|sparse autoencoder]] and the [[al:variational-autoencoder|variational autoencoder]] are variants with their own pages.

## In plain words

Describe a face in thirty numbers so that someone else can redraw it from the numbers alone.
You would have to find what matters, the shape, the lighting, the expression, and drop what does not.
An autoencoder learns such a description by trial: one network writes the numbers, another redraws from them, and both are corrected until the drawings look like the originals.
If the description were allowed as many numbers as there are pixels, the networks could simply pass the pixels through and learn nothing; that is why the middle is kept narrow, or why the input is smudged before the first network sees it, so that the second has to know what faces look like to clean it up.
The numbers in the middle, the code, are what is kept: a compressed description, and often a better starting point for other tasks than the raw input.

## Overview

The encoder $f$ and the decoder $g$ are usually stacks of fully connected layers, a [[multilayer-perceptron]] each, the decoder mirroring the encoder's widths.
In the deep autoencoder of Hinton and Salakhutdinov an encoder network transforms the data into a low-dimensional code and a decoder network recovers the data from the code; the two networks are trained together by minimizing the discrepancy between the original data and its reconstruction, with the gradients obtained by the chain rule, backpropagating error derivatives first through the decoder and then through the encoder. {#c-hinton-trained-together}
After training, the encoder alone is the useful part for most uses: the code is a low-dimensional representation for visualization, retrieval or compression, or the encoder's layers initialize a network that is then trained on labels.
The decoder alone maps codes to inputs, so the model gives maps in both directions; Hinton and Salakhutdinov note that, unlike nonparametric methods, autoencoders give mappings in both directions between the data and code spaces, and can be applied to very large data sets because both pretraining and fine-tuning scale linearly in time and space with the number of training cases. {#c-hinton-both-directions}
The name autoassociator is also used: autoencoders are also often called autoassociators in the literature. {#c-autoassociator}

The constraint is what makes an autoencoder learn something.
An ordinary autoencoder whose representation is as wide as the input, or wider, can reconstruct perfectly by learning an identity mapping; with a linear encoder and decoder of equal width, any invertible $W$ with $W' = W^{-1}$ does it. {#c-identity}
The classical constraint is the bottleneck, a code of $C < D$ units.
With an affine encoder and decoder, no nonlinearity, a bottleneck and a squared error, the autoencoder essentially performs [[al:principal-component-analysis|principal component analysis]], as Baldi and Hornik showed in 1989. {#c-vincent-pca}
Nonlinear layers make it a nonlinear generalization of PCA, which is how Hinton and Salakhutdinov present it.
The denoising and contractive autoencoders replace or complement the bottleneck by a requirement on how the map behaves near the data.

## Components

The notation has vectors as rows: a layer maps $\mathbf{h}$ of width $n_\text{in}$ to $n_\text{out}$ with a weight matrix $W$ of shape $(n_\text{in}, n_\text{out})$ and a bias $\mathbf{b}$, followed by an [[al:activation-function|activation]] $s$ applied coordinate by coordinate.
Vincent et al. write the same layer with column vectors, $s(W\mathbf{x} + \mathbf{b})$ with $W$ of shape $d' \times d$, which is the transpose of the $W$ here.

**The layer.**

$$ \mathbf{h}_{k+1} = s_k(\mathbf{h}_k W_k + \mathbf{b}_k) . $$ {#eq-layer}

**The encoder.** $K$ layers of widths $n_0 = D, n_1, \dots, n_K = C$:

$$ \mathbf{y} = f(\mathbf{x}) = \mathbf{h}_K, \qquad \mathbf{h}_0 = \mathbf{x} . $$ {#eq-encoder}

Vincent et al. define the encoder as an affine mapping followed by a nonlinearity, f(x) = s(Wx + b) with W a d' x d weight matrix, and the decoder as an affine mapping optionally followed by a squashing nonlinearity, g(y) = W'y + b' or s(W'y + b'), with the reconstruction error L(x, z) proportional to -log p(x|z). {#c-vincent-def}
The hidden layers usually have a sigmoid or tanh, and Hinton and Salakhutdinov made the code layer linear in their deep autoencoders, so that the code is not confined to a bounded box.

**The decoder.** $K$ layers of widths $C = n_K, \dots, n_0 = D$, with decoder matrices $V_j$ and biases $\mathbf{c}_j$; the last layer produces the logits $\mathbf{a}$ and the reconstruction:

$$ \mathbf{z} = g(\mathbf{y}) = s_\text{out}(\mathbf{a}), \qquad \mathbf{a} = \mathbf{u}_{K-1} V_{K-1} + \mathbf{c}_{K-1} , $$ {#eq-decoder}

where $\mathbf{u}_0 = \mathbf{y}$ and each earlier decoder layer is eq-layer; $s_\text{out}$ is the sigmoid for inputs in $[0, 1]$ and the identity for real-valued inputs.

**Tied weights.** The decoder may reuse the encoder's matrices transposed, $V_j = W_{K-1-j}^\top$, with its own biases.
This halves the weights, and Hinton and Salakhutdinov's unrolled networks start this way: after layer-by-layer pretraining, the model is unfolded into encoder and decoder networks that initially use the same weights, and a global fine-tuning stage uses backpropagation through the whole autoencoder to fine-tune the weights for optimal reconstruction. {#c-hinton-unfold}

**The loss.** The reconstruction error is a negative log-likelihood of $\mathbf{x}$ under a distribution with parameters $\mathbf{z}$.
A Gaussian X|z ~ N(z, sigma^2 I) yields the squared error up to a constant, and a Bernoulli X|z ~ B(z) with a sigmoid decoder yields the cross-entropy loss -sum_j [x_j log z_j + (1 - x_j) log(1 - z_j)], which can also be used when x is not strictly binary but in [0, 1]^d. {#c-vincent-losses}

$$ L_2(\mathbf{x}, \mathbf{z}) = \lVert \mathbf{x} - \mathbf{z} \rVert^2 , $$ {#eq-squared-error}

$$ L_H(\mathbf{x}, \mathbf{z}) = -\sum_{j=1}^{D} \big[ x_j \log z_j + (1 - x_j) \log(1 - z_j) \big] = \sum_{j=1}^{D} \big[ \operatorname{softplus}(a_j) - x_j a_j \big] , $$ {#eq-cross-entropy}

the second form, in the logits, being the one to compute.
Both are summed over the $D$ components and averaged over the examples.

## Shapes and parameters

The shape contract has $L$ hidden layers of $H$ units on each side of a code of $C$ units, on an input of $D$; the example is $D = 6$, $H = 5$, $L = 1$, $C = 2$.
For an encoder of widths $n_0, \dots, n_K$ and a mirrored decoder, every layer with a bias,

$$ P = 2 \sum_{k=0}^{K-1} n_k n_{k+1} + \sum_{k=1}^{K} n_k + \sum_{k=0}^{K-1} n_k $$ {#eq-params}

counts the weights of both halves, the encoder's biases and the decoder's. {#c-params}
In the shape contract this is

{shape:params}

and the example has {shape:params@example} parameters.
Tying the decoder weights to the transposes of the encoder weights halves the weights and keeps all the biases, {#c-params-tied}

{shape:params_tied}

which is {shape:params_tied@example} in the example.
The deep autoencoders of Hinton and Salakhutdinov have, untied, 2,837,314 parameters for MNIST (784-1000-500-250-30), 842,340 for the curves (784-400-200-100-50-25-6), 7,537,655 for the faces (625-2000-1000-500-30) and 2,318,760 for the documents (2000-500-250-125-10), {#c-hinton-sizes}
and about half that when the weights are tied, as they are at the start of fine-tuning.
The config `curves_like`, 784 inputs, three hidden layers of 100 units and a code of 6, has {shape:params@curves_like} parameters.
The derivation `parameter-and-flop-count` proves these counts.

## Computation and cost

One encoding and decoding of a batch of $B$ makes one matrix product per layer, each of shape $(B, n_k)$ by $(n_k, n_{k+1})$ in the encoder and the transpose shape in the decoder, so the forward pass costs

$$ C_\text{fwd} = 4 B \sum_{k=0}^{K-1} n_k n_{k+1} $$ {#eq-flops}

FLOPs, a multiply-add counted as two, the same with tied weights; the backward pass of the reference costs twice as much. {#c-flops}
In the contract,

{shape:flops_forward}

which is {shape:flops_forward@example} FLOPs for the example on a batch of 4, and {shape:flops_forward@curves_like} for one input of `curves_like`.
Encoding alone costs half of $C_\text{fwd}$, which is all that is paid at inference time when only the code is wanted.
The denoising autoencoder adds the cost of drawing the corruption, linear in $BD$, and the contractive penalty of a one-layer sigmoid encoder adds $O(B\, d_x d_h)$, the order of the encoder's own product (eq-cae-sigmoid below).

## Training

The parameters are fitted by stochastic gradient descent or [[adam|Adam]] on the reconstruction loss, by [[al:backpropagation|backpropagation]] through the decoder and then the encoder.

### Depth and pretraining

Weights of nonlinear autoencoders with multiple hidden layers are difficult to optimize: with large initial weights they typically find poor local minima, and with small initial weights the gradients in the early layers are tiny, making it infeasible to train autoencoders with many hidden layers. {#c-hinton-init}
Hinton and Salakhutdinov's answer was to pretrain the network one layer at a time as a stack of restricted Boltzmann machines and then fine-tune the unrolled autoencoder.
Without pretraining, the very deep autoencoder always reconstructs the average of the training data, even after prolonged fine-tuning; shallower autoencoders with a single hidden layer between the data and the code can learn without pretraining, but pretraining greatly reduces their total training time. {#c-hinton-no-pretrain}
Later practice, with better initializations, normalization and optimizers such as [[adam|Adam]], trains deep autoencoders end to end, and pretraining by stacking autoencoders became mainly of historical interest.

### Denoising

A denoising autoencoder is trained to reconstruct the clean input x from a corrupted version x~ drawn from a corruption process q(x~|x), and each time a training example is presented a different corrupted version of it is generated. {#c-dae-def}

$$ L_\text{DAE} = \mathbb{E}_{\tilde{\mathbf{x}} \sim q(\tilde{\mathbf{x}} \mid \mathbf{x})}\, L\big(\mathbf{x}, g(f(\tilde{\mathbf{x}}))\big) . $$ {#eq-dae}

Vincent et al. consider additive isotropic Gaussian noise, x~|x ~ N(x, sigma^2 I); masking noise, which forces a fraction nu of the elements of x, chosen at random for each example, to 0; and salt-and-pepper noise, which sets such a fraction to their minimum or maximum possible value according to a fair coin flip. {#c-corruptions}
The emphasized denoising autoencoder weights the reconstruction error by alpha on the corrupted components and beta on the others, and full emphasis, alpha = 1 and beta = 0, only takes into account the error on the prediction of the corrupted elements. {#c-emphasis}
Full emphasis makes the network predict each missing component from the others, which is in spirit the masked prediction of later self-supervised models, and masking noise on the hidden units rather than the input is close to [[dropout]].

For small Gaussian noise the denoising criterion becomes an explicit penalty.
With Gaussian corruption of variance $\sigma^2$, the expected squared denoising error is $\lVert \mathbf{x} - r(\mathbf{x}) \rVert^2 + \sigma^2 \big( \lVert J_r(\mathbf{x}) \rVert_F^2 - (\mathbf{x} - r(\mathbf{x})) \cdot \Delta r(\mathbf{x}) \big) + O(\sigma^4)$, a penalty on the Jacobian of the whole reconstruction $r = g \circ f$ to first order in $\sigma^2$. {#c-dae-small-noise}
For an affine reconstruction $r(\mathbf{x}) = A\mathbf{x} + \mathbf{c}$ the identity is exact, $\lVert \mathbf{x} - A\mathbf{x} - \mathbf{c} \rVert^2 + \sigma^2 \lVert A \rVert_F^2$: Gaussian denoising of a linear autoencoder is [[weight-decay|weight decay]] on its composite matrix. {#c-dae-linear-exact}
The derivation `small-noise-denoising` proves both.
With large noise this equivalence breaks down, and the denoising autoencoder learns something weight decay does not (Evidence below).

### Contraction

The contractive auto-encoder adds to the reconstruction cost a penalty equal to the squared Frobenius norm of the Jacobian matrix of the encoder activations with respect to the input. {#c-cae-def}

$$ L_\text{CAE} = L\big(\mathbf{x}, g(f(\mathbf{x}))\big) + \lambda \lVert J_f(\mathbf{x}) \rVert_F^2, \qquad J_f(\mathbf{x})_{ji} = \frac{\partial f_j(\mathbf{x})}{\partial x_i} . $$ {#eq-cae}

For a one-layer sigmoid encoder h = s(xW + b) the contractive penalty is sum_j (h_j (1 - h_j))^2 sum_i W_ij^2, {#c-cae-sigmoid}

$$ \lVert J_f(\mathbf{x}) \rVert_F^2 = \sum_{j=1}^{d_h} \big(h_j (1 - h_j)\big)^2 \sum_{i=1}^{d_x} W_{ij}^2 , $$ {#eq-cae-sigmoid}

and it and its gradient over a batch of $B$ cost $O(B\, d_x d_h)$ operations, the order of the encoder's forward pass. {#c-cae-cost}
For a linear encoder the contractive penalty is ||W||_F^2 at every input, so the contractive autoencoder reduces to weight decay on the encoder. {#c-cae-linear}
A sigmoid encoder has a second way to contract, by saturating its units so that $h_j (1 - h_j) \to 0$, and it can do that only where the data are, which is the point of the penalty.
The derivation `contractive-penalty` proves these three statements.
Rifai et al. argue that the contractive penalty explicitly encourages robustness of the representation f(x), whereas the denoising criterion encourages robustness of the reconstruction g(f(x)), and that for small additive Gaussian noise the denoising criterion yields a term in the Jacobian of the reconstruction rather than of the representation. {#c-cae-vs-dae}

### Linear autoencoders and PCA

The linear case can be solved exactly, and it shows what the bottleneck does and does not determine.
A linear autoencoder with biases and $C < D$ code units trained to minimum mean squared error has error $s_{C+1} + \dots + s_D$, the variance PCA with $C$ components discards, where $s_1 \ge \dots \ge s_D$ are the eigenvalues of the sample covariance; its decoder rows span the principal subspace and the composite map is the orthogonal projection onto it. {#c-linear-pca}
But for every invertible $C \times C$ matrix $G$, replacing the encoder $A$ by $AG$ and the decoder $B$ by $G^{-1} B$ leaves the reconstruction unchanged, so a minimum fixes only the subspace and not the principal directions. {#c-gl-symmetry}
Plaut notes that a linear autoencoder trained to minimum error spans the principal subspace, but its weight vectors are not the principal directions: they are correlated and not sorted by variance, and the solutions for reduction to different dimensions are not nested. {#c-plaut-subspace}
Kunin et al. prove that the unregularized loss of a linear autoencoder is symmetric under the group of invertible matrices, so the principal directions cannot be learnt from it, and that L2 regularization of the encoder and decoder reduces the symmetry group to the orthogonal group, which preserves the structure of the SVD. {#c-kunin-symmetry}
With weight decay $\lambda > 0$ on both matrices, every critical point of the loss of a linear autoencoder has the encoder equal to the transpose of the decoder, {#c-transpose}
and at the minimum the left singular vectors of the decoder are the principal directions, each with squared singular value $1 - \lambda / s_i$, and a direction whose variance $s_i$ is at most $\lambda$ is dropped. {#c-shrink}
So weight decay ties the weights by itself and recovers the principal directions up to an orthogonal rotation within directions of equal shrinkage; with distinct variances that leaves only signs.
The derivation `linear-autoencoder-and-pca` proves the four statements.

## Inductive biases

The bottleneck assumes that the data lie near a set that $C$ numbers can parametrize, and the decoder assumes that the set is the image of a smooth map; with sigmoid or tanh units both maps are smooth and bounded in slope.
The loss assumes a noise model for the input, Gaussian for the squared error and Bernoulli for the cross-entropy, applied to each component independently given the code; the reconstruction is the mean of that noise model, so an autoencoder trained with the squared error reconstructs blurry averages where the data are uncertain.
Denoising assumes that the clean input can be predicted from a corrupted one, which holds when the components of the data are strongly dependent, as neighbouring pixels are.
Contraction assumes that the code should be insensitive to small moves of the input in most directions, and leaves it free to change along the few directions in which the data vary.
Fully connected layers assume nothing about the order of the components: permuting the pixels of every image the same way gives the same autoencoder up to a permutation of its first and last weights.

## Evidence

Hinton and Salakhutdinov compared their deep autoencoders with PCA on three kinds of images.
On synthetic curves the six-dimensional deep autoencoder had an average squared reconstruction error per test image of 1.44, against 7.64 for logistic PCA with six components, 2.45 for logistic PCA with 18 components and 5.90 for standard PCA with 18 components. {#c-hinton-curves}
On MNIST the 30-dimensional autoencoder had an average squared reconstruction error of 3.00 against 8.01 for 30-dimensional logistic PCA and 13.87 for standard PCA; on faces the 30-dimensional autoencoder had 126 against 135 for 30-dimensional PCA. {#c-hinton-mnist-faces}
When the number of parameters is the same, deep autoencoders can produce lower reconstruction errors on test data than shallow ones, but this advantage disappears as the number of parameters increases. {#c-hinton-depth}
A 2000-500-250-125-10 autoencoder trained on half of 804,414 newswire stories, each represented by the counts of 2000 common word stems, clearly outperformed [[al:latent-semantic-analysis|latent semantic analysis]] at document retrieval. {#c-hinton-documents}

Vincent et al. found that the denoising criterion changes what the first layer learns.
On natural image patches an under-complete autoencoder of 50 hidden units learnt local blob detectors and an over-complete one of 200 learnt filters with no recognizable structure; a denoising autoencoder of 200 units with Gaussian noise of sigma = 0.5 learnt Gabor-like local oriented edge detectors, which an L2 weight decay autoencoder never produced over a wide range of the regularization hyperparameter. {#c-vincent-filters}
When denoising autoencoders are stacked, input corruption is only used for the initial denoising training of each individual layer; the learnt encoder is then applied to uncorrupted inputs to produce the input of the next layer. {#c-stacking}
Three stacked denoising autoencoders (SDAE-3) reached 1.28 +/- 0.22% test error on MNIST against 1.24 for DBN-3, 1.40 for SAE-3 and 1.40 for an RBF SVM, and 43.76% on bg-img-rot against 47.39, 51.93 and 55.18, and the authors report it as superior or equivalent to every other model of their comparison on all problems except bg-rand. {#c-sdae-results}
With one hidden layer of 1000 units used as pretraining for a classifier, the test error on MNIST was 1.14% for the CAE against 1.18 for DAE-g, 1.30 for RBM-binary, 1.57 for DAE-b, 1.68 for AE+wd and 1.78 for the AE, and on CIFAR-bw 47.86% for the CAE against 55.47 for the AE. {#c-cae-results}
These comparisons are of pretraining methods at the scale of 2010, on small images, and each is one paper's tuning of its own method against baselines.

## Limitations

The code of a plain or denoising autoencoder is not a generative model: nothing says which codes are typical, so decoding a random code gives no sample of the data.
The [[al:variational-autoencoder|variational autoencoder]] adds a prior over codes and an encoder that outputs a distribution for this reason.
The reconstruction loss weights every component equally, so the model spends its capacity on what is large in the loss, not on what matters to a downstream task; with the squared error, uncertain details become blurred averages.
The constraint has to be chosen, and the right strength of each, the width of the code, the noise level, $\lambda$, depends on the data and on the use; a plain over-complete autoencoder learns nothing useful.
The code is not identified: even in the linear case any invertible change of coordinates gives the same reconstruction, so individual code units carry no meaning unless a regularizer, such as weight decay or sparsity, picks a basis.
For classification, autoencoder pretraining has been overtaken since about 2012 by supervised training from a good initialization and, more recently, by other self-supervised objectives.

## Variants and descendants

- **Sparse autoencoders** penalize the activity of the code instead of its width; the [[sparse-autoencoders]] of interpretability research use a wide, sparse code to decompose a network's activations into features.
- **Denoising autoencoders** (Vincent et al., 2008, 2010), eq-dae, and **stacked denoising autoencoders**, pretrained layer by layer.
- **Contractive autoencoders** (Rifai et al., 2011), eq-cae.
- **Variational autoencoders** (Kingma and Welling, 2014) make the encoder output a distribution over codes and add a prior, which turns the autoencoder into a generative model.
- **Convolutional and recurrent autoencoders** use convolutional or recurrent encoders and decoders for images and sequences.
- **Masked autoencoders** for images and **masked language models** train an encoder to fill in masked parts of the input, a large-scale form of denoising with masking noise and full emphasis.

## Notable models

The four deep autoencoders of Hinton and Salakhutdinov (2006), for curves, MNIST digits, faces and newswire documents, made deep networks trainable on these tasks and found nonlinear codes that beat PCA and latent semantic analysis.
The stacked denoising autoencoders of Vincent et al. (2010) were among the standard ways to pretrain deep networks before supervised training from scratch took over.
Sparse autoencoders trained on the activations of language models are now a common tool of interpretability research.

## Implementations

The NumPy reference `impl/numpy/autoencoder.py` implements eq-layer, eq-encoder and eq-decoder for any widths, tied or untied, with sigmoid, tanh, relu or linear units; the losses eq-cross-entropy, from the logits, and eq-squared-error, with optional per-component weights for the emphasized denoising autoencoder; the backward pass by hand; the three corruptions of eq-dae; the Jacobian of the code or of the reconstruction; eq-cae-sigmoid and its gradient; training with Adam; PCA; and the counts eq-params and eq-flops.
The PyTorch version `impl/pytorch/autoencoder.py` is an `nn.Module` with the same parameters; `test_matches_a_stack_of_linear_layers` checks it against a `torch.nn.Sequential` of `torch.nn.Linear` layers, `test_gradcheck` runs `torch.autograd.gradcheck` with and without tied weights, and `test_contractive_penalty_is_the_jacobian_norm` checks eq-cae-sigmoid against the Jacobian from `torch.func.jacrev`.
The JAX version `impl/jax/autoencoder.py` is a pure function, tested with `jax.grad` against the gradients of the NumPy backward pass.
All three load the shared fixture of `fixtures/`: the example config with nonzero biases, a batch of 4, its codes, logits, reconstructions and cross-entropy loss, and the gradients of the loss.

### Worked example

Take two-dimensional data with variances 4 and 1 along two orthogonal directions, and a linear autoencoder with one code unit.
Without weight decay the best error is 1, the variance of the dropped direction, and the encoder and decoder are $g\,\mathbf{u}_1$ and $\mathbf{u}_1 / g$ for any $g \ne 0$, $\mathbf{u}_1$ the first principal direction.
With $\lambda = 1$, eq-squared-error plus $\lambda (\lVert A \rVert_F^2 + \lVert B \rVert_F^2)$ is minimized by an encoder and a decoder both equal to $m\,\mathbf{u}_1$ with $m^2 = 1 - \lambda / 4 = 0.75$; the reconstruction is $0.75$ times the projection onto $\mathbf{u}_1$, and the error is $0.25^2 \times 4 + 1 = 1.25$.
With $\lambda = 5 \ge 4$ the code unit dies, the reconstruction is the mean, and the error is $4 + 1 = 5$.
The test `test_worked_example` trains the NumPy reference by gradient descent from random weights in the three cases and checks these numbers.

### Pitfalls

> [!pitfall]
> **The cross-entropy of a sigmoid output, computed from the output.** $\log z_j$ with $z_j = \sigma(a_j)$ underflows to $-\infty$ when $a_j$ is very negative, and the gradient through the sigmoid vanishes when it saturates. Compute eq-cross-entropy from the logits, as softplus(a) − x·a, whose gradient is simply $\sigma(\mathbf{a}) - \mathbf{x}$. The test `test_cross_entropy_from_logits` checks the two forms against each other and that the logit form stays finite at logits of ±800.

> [!pitfall]
> **A code that is too wide.** With a code as wide as the input and no other constraint the autoencoder can learn the identity, and a low reconstruction error then means nothing. The test `test_an_over_complete_autoencoder_can_learn_the_identity` builds such a network exactly; in `test_denoising_fills_in_what_the_mask_removed` the over-complete plain autoencoder reconstructs its input but cannot fill in masked components, while the denoising one can.

> [!pitfall]
> **Reading principal directions off a linear autoencoder.** A trained linear autoencoder spans the principal subspace, but its weight vectors are any basis of it. The test `test_linear_autoencoder_finds_the_principal_subspace_but_not_the_directions` checks both halves, and `test_weight_decay_recovers_the_principal_directions` checks that weight decay ties the weights and recovers the directions with the shrinkage of c-shrink.

> [!pitfall]
> **Comparing with the corrupted input.** The denoising loss compares the reconstruction with the clean input, and the corruption must be drawn afresh each time an example is seen; a fixed corrupted copy of the data set is a different, weaker method, and comparing with the corrupted input trains an ordinary autoencoder on noisy data. With full emphasis the uncorrupted components must not contribute at all; `test_full_emphasis_ignores_the_uncorrupted_components` checks that changing them changes neither the loss nor its gradient.

> [!pitfall]
> **The gradient of tied weights.** With tied weights the gradient of $W_k$ is the sum of its gradient as an encoder matrix and the transpose of its gradient as a decoder matrix; keeping only one of the two trains, but not on the loss. The tied cases of `test_gradients_match_finite_differences` in NumPy and of `test_gradcheck` in PyTorch check the sum.

> [!pitfall]
> **The orientation of the weights.** This page stores $W$ as $(n_\text{in}, n_\text{out})$, torch.nn.Linear stores its weight as (out_features, in_features), and Vincent et al. write $W\mathbf{x}$ with $W$ of shape $d' \times d$. Tied weights and square layers make a wrong orientation run without error. `test_matches_a_stack_of_linear_layers` copies the transposed weights into torch.nn.Linear layers and checks that the outputs agree.

## Experiments

This page has no experiment yet.
Some tests are small experiments.
`test_a_nonlinear_code_follows_a_curve_that_pca_cannot` fits an 8-16-1 autoencoder with tanh units to points on a curve in $\mathbb{R}^8$; its error is less than a hundredth of that of one-component PCA.
`test_denoising_fills_in_what_the_mask_removed` trains a plain and a denoising over-complete autoencoder on noisy binary patterns and checks that, on test inputs with a quarter of their components masked, the denoising one fills them in with an error below 0.15 and less than half that of the plain one.
`test_contraction_shrinks_the_jacobian` in PyTorch checks that training with the contractive penalty at $\lambda = 1$ leaves the encoder's Jacobian at less than half the norm it has without it.
The checks of the four derivations train linear autoencoders by gradient descent to the minima of c-linear-pca and c-shrink, count the products of a forward and backward pass with an instrumented array, compare eq-cae-sigmoid with finite-difference Jacobians, and compute the denoising error by Gauss–Hermite quadrature.

## History

Rifai et al. write that the auto-encoder was introduced in the late 80's, citing Rumelhart et al. (1986) and Baldi and Hornik (1989). {#c-rifai-history}
Baldi and Hornik's result that the linear autoencoder finds the principal subspace dates from 1989, and nonlinear autoencoders for dimensionality reduction were studied through the 1990s, but deep ones were hard to train.
Hinton and Salakhutdinov write that it had been obvious since the 1980s that backpropagation through deep autoencoders would be very effective for nonlinear dimensionality reduction, provided that computers were fast enough, data sets were big enough and the initial weights were close enough to a good solution, and that all three conditions were then satisfied. {#c-hinton-1980s}
Their 2006 paper in Science, with its layer-by-layer pretraining, was one of the papers that started the revival of deep learning.
Vincent et al. introduced the denoising autoencoder in 2008 and the stacked version in the JMLR paper of 2010, and Rifai et al. the contractive autoencoder in 2011.
The question of what a linear autoencoder learns came back later.
An earlier version of Plaut's paper attempted to prove the hypothesis that the first m loading vectors are the first m left singular vectors of the matrix W2 that minimizes the loss, but the attempt was found to be erroneous. {#c-plaut-retracted}
Kunin et al. note that the theorem was retracted following their correspondence. {#c-kunin-retracted}
Kunin et al. then proved that with weight decay the principal directions are recovered.

## References

- G. E. Hinton and R. R. Salakhutdinov. Reducing the Dimensionality of Data with Neural Networks. Science 313(5786), 504–507, 2006.
- P. Vincent, H. Larochelle, I. Lajoie, Y. Bengio and P.-A. Manzagol. Stacked Denoising Autoencoders: Learning Useful Representations in a Deep Network with a Local Denoising Criterion. JMLR 11, 3371–3408, 2010.
- S. Rifai, P. Vincent, X. Muller, X. Glorot and Y. Bengio. Contractive Auto-Encoders: Explicit Invariance During Feature Extraction. ICML 2011.
- E. Plaut. From Principal Subspaces to Principal Components with Linear Autoencoders. arXiv 1804.10253, 2018.
- D. Kunin, J. M. Bloom, A. Goeva and C. Seed. Loss Landscapes of Regularized Linear Autoencoders. ICML 2019.
- P. Baldi and K. Hornik. Neural networks and principal component analysis: Learning from examples without local minima. Neural Networks 2(1), 1989.
- D. E. Rumelhart, G. E. Hinton and R. J. Williams. Learning representations by back-propagating errors. Nature 323, 1986.
