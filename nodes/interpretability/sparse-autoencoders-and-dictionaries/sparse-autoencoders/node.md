---
tag: ZTPNXN
kind: method
slug: sparse-autoencoders
name: Sparse autoencoders
aliases: [SAE, sparse autoencoder, k-sparse autoencoder, TopK SAE, sparse dictionary learning on activations]
area: interpretability/sparse-autoencoders-and-dictionaries
level: core
summary: >-
  A one-hidden-layer autoencoder that rewrites a network's activation vectors as sparse, nonnegative sums of many more learned directions than the layer has dimensions, each a candidate feature.
purpose: feature discovery
uses: [autoencoder@al, relu@al, mean-squared-error@al, lasso@m2, proximal-operator@m2, orthonormal-basis@m2]
related_to: [sparse-pca@al, principal-component-analysis@al, independent-component-analysis@al, compressed-sensing@m2, johnson-lindenstrauss-lemma@al]
hyperparameters:
  - {name: n, meaning: "the number of latents, the size of the dictionary", typical: "a small multiple R of the input width d in Cunningham et al., where R = 2 in their Figure 2; up to 16 million latents in Gao et al."}
  - {name: lambda, meaning: "the L1 coefficient of a ReLU SAE, which sets the sparsity only indirectly", typical: "tuned per model and layer; 0.00086 in Cunningham et al.'s Figure 2"}
  - {name: k, meaning: "the number of active latents of a TopK SAE", typical: "set directly; Gao et al. report most comparisons at k = 32 or L0 = 128"}
  - {name: k_aux and alpha, meaning: "the number of dead latents the AuxK loss uses and its weight", typical: "512 and 1/32"}
  - {name: dead window, meaning: "the number of tokens without firing after which a latent counts as dead", typical: "10 million tokens"}
  - {name: learning rate, meaning: "the Adam step size", typical: "1e-3 in Cunningham et al.; in Gao et al. the largest that converges falls as 1/sqrt(n)"}
cost:
  train: "per activation vector, about 2nd FLOPs for the encoder and 2nd for the decoder (2kd with a sparse TopK decoder), times three for the backward pass, on top of running the language model to produce the activations"
  inference: "one encoder and one decoder pass per activation vector; 2nd weights in memory"
history:
  year: 2023
  note: "Elhage et al. (2022) framed superposition with toy models and named finding an overcomplete basis as a sparse coding problem; Cunningham et al. (2023) trained ReLU SAEs with an L1 penalty on Pythia activations; Rajamanoharan et al. (2024) and Gao et al. (2024) attacked the L1 shrinkage with gated and TopK encoders and scaled SAEs to GPT-4."
prereqs: [autoencoder@al, regularization@al, principal-component-analysis@al]
same_as:
  arxiv: ["2209.10652", "2309.08600", "2404.16014", "2406.04093"]
sources:
  - {paper: "Elhage, Hume, Olsson, Schiefer, Henighan, Kravec, Hatfield-Dodds, Lasenby, Drain, Chen, Grosse, McCandlish, Kaplan, Amodei, Wattenberg and Olah (2022)", title: "Toy Models of Superposition", locator: "arXiv 2209.10652v1, the opening sections, the toy model and the section on strategic approaches", note: "superposition, why features outnumber neurons, and decoding it as a sparse coding problem"}
  - {paper: "Cunningham, Ewart, Riggs, Huben and Sharkey (2023)", title: "Sparse Autoencoders Find Highly Interpretable Features in Language Models", locator: "arXiv 2309.08600v3, Sections 1 to 4, 7 and Appendix B", note: "the tied ReLU SAE with an L1 penalty and its interpretability evaluation"}
  - {paper: "Rajamanoharan, Conmy, Smith, Lieberum, Varma, Kramár, Shah and Nanda (2024)", title: "Improving Dictionary Learning with Gated Sparse Autoencoders", locator: "arXiv 2404.16014v2, Sections 2 to 4", note: "shrinkage, the relative reconstruction bias and the gated encoder"}
  - {paper: "Gao, Dupré la Tour, Tillman, Goh, Troll, Radford, Sutskever, Leike and Wu (2024)", title: "Scaling and evaluating sparse autoencoders", locator: "arXiv 2406.04093v1, Sections 2 to 5 and Appendix A", note: "TopK SAEs, dead latents and AuxK, scaling laws and evaluation metrics"}
  - {paper: "Makhzani and Frey (2014)", title: "k-Sparse Autoencoders", locator: "ICLR 2014, arXiv 1312.5663v2, pp. 2 and 5", note: "the original TopK autoencoder and its sparsity schedule against dead units"}
---

A sparse autoencoder (SAE) takes the activation vectors of one layer of a trained network and learns to rewrite each of them as a sum of a few directions chosen from a much larger learned set.
The encoder maps an activation $x \in \mathbb{R}^d$ to a code $z \in \mathbb{R}^n$ with $n \gg d$ and few nonzero entries; the decoder maps $z$ back to $\hat{x} = W_{\text{dec}} z + b$.
Each decoder column is a candidate feature; sparsity comes from an L1 penalty on $z$, which biases activations low, or from keeping only the $k$ largest pre-activations.

## In plain words

A neuron in a language model often fires for several unrelated things.
One explanation is that the model stores more features than it has dimensions, each as a direction, which works because only a few are active at once.
An SAE looks for those directions without labels, by reconstructing each activation from a wide hidden layer in which only a few units may be on.
Afterwards a person or a language model reads the inputs that turn each unit on and tries to name it.
The original network is left untouched.

## Setting

Elhage et al. (2022) describe polysemanticity, neurons that pack many unrelated concepts, and give a toy model in which it arises because the model stores additional sparse features in superposition. {#c-el-polysemantic}
They point to two facts that make this plausible: an $n$-dimensional space has only $n$ orthogonal vectors but $\exp(n)$ almost orthogonal ones, by the Johnson-Lindenstrauss lemma, and compressed sensing can recover a vector from a lower-dimensional projection when the vector is known to be sparse. {#c-el-almost-orthogonal}
Since the directions are only almost orthogonal, one feature firing looks like others firing slightly, and for highly sparse features the cost of this interference may be outweighed by the benefit of representing more features. {#c-el-interference}
Their toy model compresses features with $h = Wx$ and reconstructs with $x' = \mathrm{ReLU}(W^\top W x + b)$, trained on a mean squared error weighted by feature importance. {#c-el-toy-model}
A linear version of it learns the most important features, as principal components would, while the ReLU version, as the features get sparser, represents more features than dimensions by placing them non-orthogonally, starting with antipodal pairs. {#c-el-pca-vs-superposition}

Elhage et al. list three ways out: build models without superposition, find an overcomplete basis that describes the features of a model that has it, or a hybrid of the two. {#c-el-three-ways}
The second way is, in their words, a relatively standard sparse coding problem over the activations of a layer. {#c-el-sparse-coding}
Cunningham et al. make the same reduction: finding directions from which each activation is a sparse linear combination is the problem of sparse dictionary learning. {#c-cu-dictionary-learning}

The input is a large set of activation vectors $x \in \mathbb{R}^d$ from one site of a trained model, such as a residual stream; the output is a dictionary of $n$ unit-norm directions and an encoder that gives each $x$ a sparse nonnegative code.
Rajamanoharan et al. state the target as $x \approx x_0 + \sum_{i=1}^{M} f_i(x)\, d_i$, with $M \gg n$ unit-norm directions $d_i$ and sparse coefficients $f_i(x) \ge 0$, where their $n$ is the activation width; the decoder's columns give the $d_i$ and its bias gives $x_0$. {#c-ra-decomposition}

## The method

### The ReLU SAE with an L1 penalty

Cunningham et al. train an autoencoder with one hidden layer of size $R \cdot d_{\text{in}}$, a ReLU, and tied weights, so the decoder is the transpose of the encoder. {#c-cu-architecture}
Their encoder is $c = \mathrm{ReLU}(Mx + b)$ and their decoder $\hat{x} = M^\top c$, with the rows of $M$ normalized, trained on $\lVert x - \hat{x} \rVert_2^2 + \alpha \lVert c \rVert_1$. {#c-cu-equations}
They tie the weights because the directions that detect and define a feature should agree, because tying halves the memory, and because it removes the question of which direction to interpret, while reporting some loss of performance on MLP data. {#c-cu-tied}

The untied form used as the baseline by Gao et al., following Bricken et al., has encoder and decoder {#c-ga-relu}

$$ z = \mathrm{ReLU}\big(W_{\text{enc}} (x - b_{\text{pre}}) + b_{\text{enc}}\big) $$ {#eq-relu-encoder}

$$ \hat{x} = W_{\text{dec}} z + b_{\text{pre}} $$ {#eq-decoder}

with $W_{\text{enc}} \in \mathbb{R}^{n \times d}$, $W_{\text{dec}} \in \mathbb{R}^{d \times n}$, and the loss

$$ L = \lVert x - \hat{x} \rVert_2^2 + \lambda \lVert z \rVert_1 . $$ {#eq-l1-loss}

The L1 term stands in for the number of active latents, $\lVert z \rVert_0$, which cannot be optimized directly because it is not differentiable. {#c-ra-l0-nondiff}
Without a constraint the L1 term can be made as small as wished with no change to the reconstructions, by shrinking the encoder outputs and growing the decoder norms, so the decoder columns are held at exactly unit norm. {#c-ra-norm-constraint}

### The TopK SAE

Gao et al. use the k-sparse autoencoder of Makhzani and Frey, whose activation keeps only the $k$ largest latents and zeroes the rest; the decoder is unchanged and the loss is the reconstruction error alone. {#c-ga-topk}

$$ z = \mathrm{TopK}\big(W_{\text{enc}} (x - b_{\text{pre}})\big), \qquad L = \lVert x - \hat{x} \rVert_2^2 $$ {#eq-topk-encoder}

Their code also applies a ReLU so the activations are nonnegative, and the training curves do not change because the $k$ largest are almost always positive. {#c-ga-topk-relu}
They list four benefits: no L1 penalty and so no shrinkage of the positive activations toward zero, an L0 set directly instead of through $\lambda$, a better trade-off between sparsity and reconstruction, with a gap that grows with scale, and small activations clamped to zero. {#c-ga-topk-benefits}
Makhzani and Frey had built the k-sparse autoencoder on tied weights and linear activations, so that picking the $k$ largest units is the only nonlinearity; at test time they kept the $\alpha k$ largest, with $\alpha \ge 1$ chosen on validation data. {#c-mf-linear}

### Dead latents

In large SAEs a growing share of latents stops firing altogether; Gao et al. cite a 34-million-latent SAE with only 12 million alive and find up to 90% dead without mitigation. {#c-ga-dead}
A latent counts as dead if it has not fired in 10 million tokens. {#c-ga-dead-def}
Two things fix most of it: starting the encoder as the transpose of the decoder, and an auxiliary loss that reconstructs the error from the dead latents; with both, their 16-million-latent SAE has 7% dead. {#c-ga-dead-fixes}
The auxiliary loss, AuxK, takes the error $e = x - \hat{x}$, reconstructs it as $\hat{e} = W_{\text{dec}} z$ from the top $k_{\text{aux}}$ dead latents, typically 512, and adds $\alpha \lVert e - \hat{e} \rVert_2^2$ with $\alpha$ typically 1/32, for about 10% more compute. {#c-ga-auxk}
The problem is older than SAEs on language models: Makhzani and Frey found that too much sparsity early in training leaves some hidden units dead, never reached by the gradient, and fixed it by starting with a large $k$ and lowering it, from $k = 100$ to $k = 15$ over the first half of the epochs. {#c-mf-dead}

### Training details

Gao et al. subtract the mean over the model dimension and scale each input to unit norm, and report errors as normalized MSE, divided by the error of always predicting the mean activation. {#c-ga-inputs}
They start $b_{\text{pre}}$ at the geometric median of a data sample, the encoder directions parallel to the decoder directions, and the decoder columns at unit norm, restoring unit norm after every step. {#c-ga-init}
They also remove the gradient component parallel to each decoder column, to account for the interaction of Adam with the normalization. {#c-ga-project}
The projected gradient $g - w (w^\top g)$ of a unit column $w$ is orthogonal to $w$, so a step keeps the norm to first order.

## Derivation

Three derivations support the main formulas; each lists its assumptions and has a numeric check.

**L1 shrinkage** (`l1-shrinkage`).
Fix one latent with unit decoder direction $d$, take $x$ with the decoder bias already subtracted, and minimize $g(f) = \lVert x - f d \rVert^2 + \lambda f$ over $f \ge 0$.
Since $g(f) = f^2 - 2f \langle d, x \rangle + \lambda f + \lVert x \rVert^2$, the minimum is the soft threshold $f^\star = \max(0, \langle d, x \rangle - \lambda/2)$: every active latent is lowered by exactly $\lambda/2$, and latents with $\langle d, x \rangle \le \lambda/2$ are switched off. {#c-shrink-soft-threshold}
A ReLU encoder with weight row $d$ and bias $-\lambda/2$ computes $f^\star$ exactly, so the ReLU SAE can represent this optimum and the bias it learns carries the shrinkage. {#c-shrink-relu-encoder}
With $x = d$ and $\lambda = 1$, $f^\star = 1/2$: the reconstruction is half the input although a perfect one exists, the example of Rajamanoharan et al.'s Figure 2. {#c-shrink-figure}
If the decoder norm is free, replacing $W_{\text{dec}}$ by $s W_{\text{dec}}$ and the encoder output by $z/s$ leaves $\hat{x}$ unchanged and divides the L1 term by $s$, so it can be made as small as wished. {#c-norm-scale-symmetry}

**TopK as an exact L0 constraint** (`topk-l0`).
$\mathrm{ReLU}(\mathrm{TopK}(a))$ has at most $k$ nonzero entries for every input, with equality when $k$ entries are positive; no coefficient is involved. {#c-topk-exact-l0}
It is also the closest nonnegative vector to $a$ with at most $k$ nonzeros: putting entry $i$ in the support lowers the squared distance by $\max(a_i, 0)^2$, so the best support is the $k$ largest entries. {#c-topk-projection}
If the decoder has orthonormal columns and the encoder is its transpose, $\lVert x - W v \rVert^2 = \lVert x - W W^\top x \rVert^2 + \lVert W^\top x - v \rVert^2$, so the TopK code is the best nonnegative $k$-sparse code. {#c-topk-orthonormal}
The kept values equal their pre-activations, with nothing subtracted. {#c-topk-no-shrink}
An overcomplete dictionary cannot be orthonormal, so this only motivates the tied start.

**Reconstruction bias and FVU** (`reconstruction-bias`).
Rajamanoharan et al. measure shrinkage with the relative reconstruction bias, the $\gamma$ for which rescaling the reconstructions by $1/\gamma$ minimizes the squared error; $\gamma < 1$ means shrinkage. {#c-ra-gamma}
Minimizing $\mathbb{E}\lVert c \hat{x} - x \rVert^2$ over the scalar $c$ gives $c = \mathbb{E}[\hat{x} \cdot x] / \mathbb{E}\lVert \hat{x} \rVert^2$, so {#c-gamma-def}

$$ \gamma = \frac{\mathbb{E} \lVert \hat{x} \rVert_2^2}{\mathbb{E}[\hat{x} \cdot x]} = \frac{2\, \mathbb{E} \lVert \hat{x} \rVert_2^2}{\mathbb{E} \lVert \hat{x} \rVert_2^2 + \mathbb{E} \lVert x \rVert_2^2 - \mathbb{E} \lVert \hat{x} - x \rVert_2^2} , $$ {#eq-gamma}

the second form by $2 a \cdot b = \lVert a \rVert^2 + \lVert b \rVert^2 - \lVert a - b \rVert^2$.
In the one-latent case with $x = s d$ and $s > \lambda/2$, $\gamma = 1 - \lambda/(2s)$, which is $1/2$ for $s = \lambda = 1$. {#c-gamma-shrink-example}
The fraction of variance unexplained is the normalized MSE of Gao et al., computed over a batch:

$$ \mathrm{FVU} = \frac{\sum \lVert x - \hat{x} \rVert^2}{\sum \lVert x - \bar{x} \rVert^2} , $$ {#eq-fvu}

which is 1 for the predictor that always outputs the batch mean and 0 for perfect reconstruction. {#c-fvu-mean-one}

## Worked example

Take the two inputs $(3, 1)$ and $(1, 3)$, a decoder with columns $e_1$ and $e_2$, a tied encoder and no decoder bias.
The batch mean is $(2, 2)$, so the denominator of the FVU is 4.
An L1 SAE with $\lambda = 1$ has encoder bias $-1/2$ and codes $(2.5, 0.5)$ and $(0.5, 2.5)$: L0 is 2, the squared error is 1, the FVU is 0.25 and $\gamma = 13/16$. {#c-worked-l1}
With $\lambda = 2$ the codes become $(2, 0)$ and $(0, 2)$, so L0 is 1, but the squared error is 4, the FVU is 1.0, no better than predicting the mean, and $\gamma = 2/3$.
A TopK SAE with $k = 1$ gives $(3, 0)$ and $(0, 3)$: L0 is 1, the squared error is 2, the FVU is 0.5 and $\gamma = 1$. {#c-worked-topk}
At the same L0 the L1 SAE has twice the error, all from the $\lambda/2$ subtracted from the kept latents.
The test `test_worked_example` of the NumPy implementation and `test_losses_and_metrics_on_the_worked_example` of the PyTorch one run this example.

## Hyperparameters

The dictionary size $n$ is the main lever.
Gao et al. find that the largest learning rate that converges scales as $1/\sqrt{n}$, and that the number of tokens to convergence grows as about $\Theta(n^{0.6})$ for GPT-2 small and $\Theta(n^{0.65})$ for GPT-4. {#c-ga-lr-tokens}
With an L1 penalty, $\lambda$ sets the sparsity only indirectly and must be swept; Cunningham et al. find a smooth trade-off between sparsity and accuracy as $\alpha$ varies, with no knee that would mark a single right decomposition. {#c-cu-tradeoff}
With TopK, $k$ is set directly.
Cunningham et al. train with Adam at a learning rate of 1e-3 on 5 to 50 million activation vectors for 1 to 3 epochs, under an hour on one A40 GPU per run. {#c-cu-training}
Gao et al. use an Adam $\epsilon$ of $6.25 \times 10^{-10}$, a batch of 131,072 tokens and an exponential moving average of the weights with coefficient 0.999. {#c-ga-adam}

## Cost

An SAE with $n$ latents on $d$-dimensional activations has about $2nd$ weights and costs about $2nd$ FLOPs per vector in the encoder; the decoder costs as much when dense and about $2kd$ when only $k$ latents are active.
Collecting the activations means running the language model over the training tokens, often the larger cost; Rajamanoharan et al. note that much of the wall-clock time goes to generating activations or to disk input and output. {#c-ra-wallclock}
Cunningham et al. describe the approach as using orders of magnitude less compute than training the original model. {#c-cu-compute}
The cost does not stay small at scale: Gao et al. trained a 16-million-latent SAE on GPT-4 activations for 40 billion tokens. {#c-ga-16m}

## Evidence

On Pythia residual streams, Cunningham et al.'s features score higher on automated interpretability than the default basis, random directions, PCA and ICA. {#c-cu-interp}
The advantage shrinks with depth: comparable to ICA in layer 4 and minimal in the final layer. {#c-cu-layers}
On indirect object identification, patching their dictionary features reaches a given KL divergence with fewer patches than PCA components. {#c-cu-ioi}
Gao et al. fit a joint law $L(n, k) = \exp(\alpha + \beta_k \log k + \beta_n \log n + \gamma \log k \log n) + \exp(\zeta + \eta \log k)$ to GPT-4 SAEs with $\alpha = -0.50$, $\beta_k = 0.26$, $\beta_n = -0.017$, $\gamma = -0.042$, $\zeta = -1.32$ and $\eta = -0.085$, valid for small $k$. {#c-ga-joint-law}
With $n = 32768$ latents, TopK reconstructs better than ReLU and ProLU at equal L0 and about as well as Gated, and at $L0 = 128$ its error falls faster with $n$ than ReLU's. {#c-ga-tradeoff}
Substituting their 16-million-latent SAE into GPT-4 gives a language-modeling loss that corresponds to 10% of GPT-4's pretraining compute. {#c-ga-gpt4-loss}
Refining the activations on a frozen support raises ReLU activations on average but not TopK ones, and closes only part of the gap between the two. {#c-ga-refine}
Gated SAEs have $\gamma \approx 1$, while baseline SAEs have $\gamma < 1$, getting worse as $\lambda$ grows. {#c-ra-gamma-results}
On models of up to 7B parameters, Gated SAEs need half as many firing features for comparable reconstruction fidelity. {#c-ra-half}
In a blinded human rating, the test that Gated SAEs are at most as interpretable as baseline ones gives $p = .060$ with an estimated difference of .13 and interval [0, .26], so the two are comparably interpretable. {#c-ra-human}

Interpretability here is rated automatically or by a few humans, and reconstruction metrics say how much of the activation is explained, not whether the features are the model's own.

## Limitations and failure modes

**Reconstruction is lossy.**
Replacing the layer 2 residual stream of Pythia-70M with its reconstruction raised perplexity on the Pile from 25 to 40. {#c-cu-perplexity}
Cunningham et al.'s pipeline also could not robustly learn overcomplete bases in the MLP intermediate layers. {#c-cu-mlp}

**Good metrics, bad features.**
An infinitely wide SAE with $k = 1$ can reconstruct perfectly by tiling the space with latents while being structureless. {#c-ga-k1}
Zero-ablation fidelity, the share of the loss gap to a zeroed layer that an SAE recovers, can be high even when the SAE explains the behavior poorly, since zeroing the residual stream does so much damage; Gao et al.'s 16-million-latent SAE scores 98.2% on it. {#c-ga-zero-ablation}

**Shrinkage is not only a rescaling.**
Rescaling shrunk activations after training is not necessarily enough, because training with L1 can also leave the encoder and decoder directions suboptimal. {#c-ra-rescale}
An unbiased but imperfect SAE still has a mean squared reconstruction norm below that of its inputs, so a small $\lVert \hat{x} \rVert$ alone does not prove shrinkage; $\gamma$ separates the two. {#c-ra-unbiased-norm}

**Fixed k.**
TopK forces exactly $k$ latents on every token, which is likely suboptimal, and a large share of random activations of GPT-4 features are not yet adequately monosemantic. {#c-ga-limits}
A TopK SAE gives a progressive code only up to the $k$ it was trained with; summing the losses $L(k) + L(4k)/8$ (Multi-TopK) fixes this at a small cost at $k$. {#c-ga-progressive}

**Decoding after the fact.**
Elhage et al. note that it is hard to know how many features to look for, that the problem may be a factorization of a millions-by-billions matrix into one a thousand or more times larger, and that interference no longer pushes training toward less superposition. {#c-el-challenges}

## Variants

**Gated SAEs.**
Rajamanoharan et al. note that the encoder both detects which features are active and estimates their magnitudes, and that the L1 penalty is needed only for the first. {#c-ra-two-roles}
Their gated encoder multiplies a Heaviside gate on $W_{\text{gate}}(x - b_{\text{dec}}) + b_{\text{gate}}$ by a ReLU magnitude path whose weights share directions with the gate, at only $2M$ more parameters than the baseline. {#c-ra-gated}
The loss needs a second decoder pass, for 50% more compute per training step. {#c-ra-compute}

**JumpReLU and ProLU.**
These zero every value below a positive threshold, $J_\theta(x) = x \cdot \mathbb{1}(x > \theta)$, and need an approximate gradient for $\theta$. {#c-ga-jumprelu}

## When to use it

Use an SAE to propose many candidate features for one site of a trained model without labels, when neurons are polysemantic or the basis is not privileged.
Prefer TopK to fix L0 and compare models at equal sparsity, and an L1 or gated SAE when the number of active features should vary by token.
Confirm features with causal tests such as patching, and check downstream loss with the reconstruction spliced in, not only MSE.
For a few known concepts, a supervised probe is cheaper.
Sparse PCA is a different tool: it makes the loadings of a few components sparse, while an SAE makes each input's code sparse over a dictionary wider than the input.

## Pitfalls

> [!pitfall]
> **Unconstrained decoder norms.** If the decoder columns can grow, the L1 term can be shrunk at no cost to the reconstruction. The tests `test_scaling_the_decoder_lowers_the_l1_term_without_changing_the_output` and `test_scale_symmetry_lowers_l1_only` show the symmetry, and `test_normalize_decoder_and_project_gradient` and `test_normalize_and_project` check the unit columns.

> [!pitfall]
> **Reading L1 activations as magnitudes.** The L1 optimum subtracts $\lambda/2$ from every active latent. The tests `test_soft_threshold_matches_a_brute_force_minimum` and `test_one_latent_example_of_figure_2` check the soft threshold, and `test_relative_reconstruction_bias` checks $\gamma$ in both forms of eq-gamma.

> [!pitfall]
> **A TopK that is not an exact L0.** Thresholding at the $k$-th value instead of indexing can keep more than $k$ latents when values tie. The tests `test_topk_is_an_exact_l0_constraint` and `test_topk_encoder_is_an_exact_l0_constraint` check at most $k$ nonzeros, and `test_topk_gradient_reaches_only_kept_latents` checks that no gradient reaches the dropped ones.

> [!pitfall]
> **AuxK reaching live latents or the main error.** The auxiliary loss must use only dead latents and treat the error as a constant. The tests `test_auxk_loss` of both implementations check that it is zero with no dead latents and that its gradient reaches only dead ones.

> [!pitfall]
> **FVU against zero instead of the mean.** Dividing by $\sum \lVert x \rVert^2$ instead of the variance about the mean makes a constant predictor look good. The test `test_fvu` checks that the mean predictor scores 1.

## History

Elhage et al. (2022) gave superposition a toy model and named the search for an overcomplete basis a sparse coding problem, after noting that an L1 penalty on the hidden layer removes superposition in their toy models at the cost of a higher loss. {#c-el-l1}
Cunningham et al. (2023) built on Sharkey et al. and, like Yun et al., applied sparse dictionary learning to language model activations, in their case with a ReLU SAE trained with an L1 penalty. {#c-cu-builds}
Rajamanoharan et al. (2024) place their work after Bricken et al. and Cunningham et al., who used SAEs to find sparse decompositions of activations in an overcomplete dictionary. {#c-ra-history}
Gao et al. (2024) replaced the L1 penalty with TopK, fixed most dead latents and fit scaling laws up to GPT-4.
Cunningham et al. close with the hope of enumerative safety, a complete list of a model's features that would rule out dangerous behavior such as deception. {#c-cu-enumerative}
