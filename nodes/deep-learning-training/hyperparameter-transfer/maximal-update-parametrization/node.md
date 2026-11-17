---
tag: 4VJECR
kind: method
slug: maximal-update-parametrization
name: Maximal update parametrization
aliases: [muP, μP, MUP, maximal update parametrisation, muTransfer, μTransfer, zero-shot hyperparameter transfer]
area: deep-learning-training/hyperparameter-transfer
level: core
summary: >-
  Scales each layer's initialization and learning rate with width so every layer keeps learning features, letting hyperparameters tuned on a small model transfer to a large one.
purpose: choosing per-layer initialization variances and learning rates as functions of width so that the optimal hyperparameters stay put as a network is made wider
uses: [stochastic-gradient-descent@m2, backpropagation@al, he-initialization@al, central-limit-theorem@m2, law-of-large-numbers@m2]
related_to: [neural-tangent-kernel@al, mean-field-approximation@al, xavier-initialization@al, hyperparameter-tuning@al, layer-normalization@al]
hyperparameters:
  - {name: base width n0, meaning: "the width at which muP and SP coincide and at which the base hyperparameters are tuned", typical: "the proxy width, 128 or 256 in Yang et al."}
  - {name: base learning rate eta, meaning: "the master learning rate, tuned on the proxy and copied to the target", typical: "whatever the proxy sweep finds; it then transfers across width"}
  - {name: initialization scale sigma, meaning: "a width-independent constant in front of every initialization standard deviation", typical: "tuned on the proxy; fixed when transferring across depth"}
  - {name: output multiplier alpha_output, meaning: "a constant in front of the readout layer's output", typical: "tuned on the proxy"}
  - {name: attention multiplier alpha_attn, meaning: "a constant in front of the 1/d attention logit", typical: "tuned on the proxy"}
cost:
  train: "the same as the network under the standard parametrization: muP changes only constants in the initialization and the per-layer learning rates"
  inference: "none; at inference a muP network is an ordinary network"
history:
  year: 2020
  note: "Yang and Hu (arXiv 2020) classified abc-parametrizations, proved the Dynamical Dichotomy and defined muP for SGD; Yang et al. (arXiv 2022) derived muP for Adam and Transformers and proposed muTransfer, tuning GPT-3 6.7B on a 40M-parameter proxy; Everett et al. (ICML 2024) showed that other parametrizations with per-layer learning rates can transfer too."
prereqs: [stochastic-gradient-descent@m2, backpropagation@al, he-initialization@al, central-limit-theorem@m2, law-of-large-numbers@m2]
same_as:
  arxiv: ["2011.14522", "2203.03466", "2407.05872"]
sources:
  - {paper: "Yang and Hu (2020)", title: "Feature Learning in Infinite-Width Neural Networks", locator: "arXiv 2011.14522v3, Sections 1 to 5 and 9", note: "abc-parametrizations, the Dynamical Dichotomy, why SP and NTP cannot learn features, the definition and uniqueness of muP, Word2Vec and Omniglot"}
  - {paper: "Yang, Hu, Babuschkin, Sidor, Liu, Farhi, Ryder, Pachocki, Chen and Gao (2022)", title: "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer", locator: "arXiv 2203.03466v2, Sections 1 to 9 and Appendices B and J", note: "Table 3 for SGD and Adam, the base-width form, 1/d attention, muTransfer and its experiments, the limitations, and the derivation"}
  - {paper: "Everett, Xiao, Wortsman, Alemi, Novak, Liu, Gur, Sohl-Dickstein, Kaelbling, Lee and Pennington (2024)", title: "Scaling Exponents Across Parameterizations and Optimizers", locator: "arXiv 2407.05872v2, Sections 1 and 2", note: "the alignment assumption, hyperparameter transfer in all parametrizations, a per-layer SP prescription, and Adam's epsilon"}
---

The maximal update parametrization (μP) is a rule for how each layer's initialization variance and learning rate should change with a network's width.
It keeps every layer's features moving by the same amount per step however wide the network is.
Its practical payoff is μTransfer: tune hyperparameters on a narrow proxy, copy them to the wide target, and skip tuning the target.

## In plain words

When a network is made wider, the usual recipe keeps the starting weights at a sensible scale and keeps the learning rate the same.
That is not enough.
Once training starts, a weight update lines up with the inputs it multiplies, so its effect adds up across the width.
In a wider network some layers then move far too much and others hardly at all, and the best learning rate drifts as the width grows.
μP rescales the readout layer's initialization and the per-layer learning rates so that, as the width grows, every layer changes its outputs by the same amount per step.
The best settings found on a small model then stay good on a big one.

## Setting

Hyperparameter tuning is expensive, and for networks with billions of parameters it is prohibitively so. {#c-tp5-tuning-cost}
Yang et al. find that in μP many optimal hyperparameters remain stable as the model size changes. {#c-tp5-stable}
They turn this into μTransfer: parametrize the target model in μP, tune on a smaller model, and zero-shot transfer the hyperparameters to the full-sized model without tuning it at all. {#c-tp5-mutransfer}
The standard parametrization (SP) keeps activations of unit order at initialization, but makes them blow up in wide models during training, essentially because the per-layer learning rates are out of balance. {#c-tp5-sp-imbalance}
In a Transformer trained with Adam, the optimal learning rate is stable across width in μP and far from stable in SP. {#c-tp5-fig1}

Yang and Hu start from a weakness of the NTK: in its limit, the last-layer features after pretraining are essentially those of random initialization. {#c-tp4-ntk-features}
They study abc-parametrizations of an $L$-hidden-layer perceptron. Each weight is $W^l = n^{-a_l} w^l$ for a trainable $w^l$ with entries drawn from $\mathcal{N}(0, n^{-2 b_l})$, and SGD uses the learning rate $\eta n^{-c}$ with $\eta$ independent of the width $n$. {#c-tp4-abc}
The NTK parametrization has $a_1 = 0$, $a_l = 1/2$ for $l \ge 2$, $b_l = 0$ and $c = 0$. SP, the PyTorch default, has $a_l = 0$, $b_1 = 0$, $b_l = 1/2$ for $l \ge 2$ and $c = 0$. {#c-tp4-sp-ntp}
If $c$ is too small, the learning rate is too large and SGD blows up preactivations or logits. If $c$ is too large, the network's function does not change in finite time. {#c-tp4-c-too-small}
Shifting $a_l \to a_l + \theta$, $b_l \to b_l - \theta$ and $c \to c - 2\theta$ leaves the trained function unchanged at every step. Moving $a_l$ against $b_l$ amounts to a per-layer learning rate. {#c-tp4-symmetry}

Their central result is the Dynamical Dichotomy. Any nontrivial stable abc-parametrization has an infinite-width limit that either lets the embedding evolve, a feature learning limit, or is described by kernel gradient descent, a kernel limit, but not both. {#c-tp4-dichotomy}
The two cases are told apart by an exponent $r$. Feature learning holds if and only if $r = 0$. {#c-tp4-thm36}
Kernel behaviour holds if and only if $r > 0$; NTP, with $r = 1/2$, is the typical example. {#c-tp4-thm38}
In any feature learning limit the output at initialization goes to 0. {#c-tp4-f0}
SP itself stays bounded only with an SGD learning rate of order $O(1/n)$, and then it is in the kernel regime and learns no features. {#c-tp4-thm41}

## The method

### Why SP fails

After one update the last hidden activations $x^L$ are correlated with the readout $W^{L+1}$. Their product then grows like $n^{1 - a - b}$ by the law of large numbers, while at initialization, when the two are independent, it grows only like $n^{1/2 - a - b}$ by the central limit theorem. {#c-tp4-lln-readout}
The first layer has the opposite problem: unless it is reparametrized, its preactivations effectively stay fixed throughout training even when its weights are trained. {#c-tp4-first-layer}
Tensor Programs V shows the mechanism on a one-hidden-layer linear perceptron. In SP, one SGD step with learning rate 1 makes the output blow up, because $U^\top U = \Theta(n)$ by the law of large numbers. {#c-tp5-perceptron}
The central limit theorem says $1/\sqrt{n}$ is the right scale for a sum of $n$ independent zero-mean terms: $1/n$ sends it to 0 and $1$ makes its variance blow up. {#c-tp5-clt}

### The definition

For an $L$-hidden-layer MLP, Yang and Hu define μP by $c = 0$, $b_l = 1/2$ for all $l$, $a_1 = -1/2$, $a_l = 0$ for the hidden layers and $a_{L+1} = 1/2$. {#c-tp4-mup-def}
Under μP every weight matrix is updated maximally and the last layer is also initialized maximally, and μP is the only stable abc-parametrization with this property. {#c-tp4-unique}
Yang and Hu covered only SGD; Tensor Programs V derives μP for Adam as well. {#c-tp5-adam-new}

Table 3 of Tensor Programs V gives μP for general networks without multipliers. Input weights and all biases have variance $1/\text{fan\_in}$, SGD learning rate $\text{fan\_out}$ and Adam learning rate 1. Output weights have variance $1/\text{fan\_in}^2$ and learning rate $1/\text{fan\_in}$ under both SGD and Adam. Hidden weights have variance $1/\text{fan\_in}$, SGD learning rate 1 and Adam learning rate $1/\text{fan\_in}$. {#c-tp5-table3}
In SP every learning rate in that table is 1 and the output variance is $1/\text{fan\_in}$. {#c-tp5-table3-sp}
In practice the factors of $n$ are written relative to a base width $n_0$, with $\tilde n = n/n_0$. For the MLP, the output layer gets variance $1/(n \tilde n)$, and the SGD learning rates are $\eta \tilde n$ for the input layer and biases, $\eta$ for the hidden layer and $\eta/\tilde n$ for the output layer. {#c-tp5-eq4}
At $n = n_0$ every extra factor is 1 and μP is SP. As $n$ grows, μP follows a different family of networks and trajectories. {#c-tp5-base-equal}

Writing $\sigma$ for the base initialization scale and $d$ for the input dimension, the initialization is

$$ W^1_{ij} \sim \mathcal{N}\!\left(0, \tfrac{\sigma^2}{d}\right), \quad W^l_{ij} \sim \mathcal{N}\!\left(0, \tfrac{\sigma^2}{n}\right), \quad W^{L+1}_{ij} \sim \mathcal{N}\!\left(0, \tfrac{\sigma^2}{n \tilde n}\right), \quad b = 0, $$ {#eq-mup-init}

and the per-layer learning rates for SGD and for Adam are

$$ \eta_{W^1} = \eta_b = \eta \tilde n, \qquad \eta_{W^l} = \eta, \qquad \eta_{W^{L+1}} = \eta / \tilde n , $$ {#eq-mup-sgd}

$$ \eta_{W^1} = \eta_b = \eta, \qquad \eta_{W^l} = \eta / \tilde n, \qquad \eta_{W^{L+1}} = \eta / \tilde n . $$ {#eq-mup-adam}

In SP the readout variance is $\sigma^2/n$ and every learning rate is $\eta$.

### Attention

μP for a Transformer is Table 3 plus 1/d attention: the logit is $q^\top k / d$ for a query and key of dimension $d$, not the usual $q^\top k / \sqrt{d}$. {#c-tp5-attn-def}
The reason is that during training $q$ and $k$ become correlated, so $q^\top k$ grows like $d$ by the law of large numbers, whereas the $1/\sqrt{d}$ rule assumed they stay uncorrelated as at initialization. {#c-tp5-attn-why}
With a base head dimension $d_0$ the logit is

$$ \text{logit}(q, k) = \alpha_{\text{attn}} \frac{\sqrt{d_0}}{d}\, q^\top k , $$ {#eq-attn}

which equals the SP logit at $d = d_0$.

### An equivalent form with multipliers

Lemma J.1 of Tensor Programs V lets a change of multiplier, initialization and learning rate be traded against each other, which turns Table 3 into Tables 8 and 9. {#c-tp5-lemma}
Table 8 is the form that allows input and output weights to be tied, as Transformers commonly do. {#c-tp5-table8}

## Derivation

Each derivation states its assumptions and has a numeric check.

**Feature updates** (`feature-update`).
Take a two-hidden-layer tanh MLP with fixed input and output dimension and hidden width $n$, and one step from initialization at a fixed $\eta$.
A gradient is built from the layer's own input: $\Delta W = -\eta\, a\, x^\top$ moves $\Delta W x$ by $\eta |x|^2 a$. That is $\Theta(n \eta)$ per coordinate, against $\Theta(\sqrt n\, \eta)$ for a random matrix with entries of the same size. {#c-aligned-lln}
Under SP, one SGD step moves the output by $\Theta(n\eta)$. {#c-sp-readout-sgd}
It moves the hidden preactivations by $\Theta(\sqrt n\, \eta)$, not $\Theta(n\eta)$, because the backpropagated signal through a readout with entries of size $1/\sqrt n$ is itself $\Theta(1/\sqrt n)$. {#c-sp-hidden-sgd}
Tensor Programs V agrees: a $\Theta(1)$ SGD rate makes SP preactivations grow like $\sqrt n$, and a $\Theta(1/n)$ rate lands in the kernel regime. {#c-tp5-sp-sgd}
Under SP with Adam the update has entries $\pm\eta$ whatever the gradient's size, so the hidden preactivations move by $\Theta(n\eta)$ and the output by at least that. {#c-sp-hidden-adam}
That is the $\Theta(n)$ blow-up that Tensor Programs V gives for Adam with a $\Theta(1)$ rate. {#c-tp5-sp-adam}
SGD in SP moves the first layer by only $\Theta(\eta/\sqrt n)$, so it stops learning as $n$ grows. {#c-sp-input-sgd}
Under μP with either optimizer the first layer, the last hidden layer and the output all move by $\Theta(\eta)$, independent of $n$. {#c-mup-constant}
Fitting log-log slopes over widths 64 to 4096 in the check gives input, hidden and output slopes of $-0.52$, $0.44$ and $0.98$ for SP with SGD, a hidden slope of $0.95$ for SP with Adam, and slopes within 0.03 of zero for μP.

**Attention scaling** (`attention-scaling`).
For independent $q$ and $k$ with coordinate variance $s^2$, $q^\top k$ has mean 0 and standard deviation $\sqrt d\, s^2$, so $q^\top k/\sqrt d$ stays $\Theta(1)$ and $q^\top k/d \to 0$. {#c-attn-random}
If instead $k = \rho q + \sqrt{1 - \rho^2}\, z$ with $\rho \ne 0$, then $q^\top k / d \to \rho s^2$ and $q^\top k/\sqrt d$ grows like $\sqrt d$. {#c-attn-aligned}
The μP scale $\alpha\sqrt{d_0}/d$ equals the SP scale $\alpha/\sqrt d$ at $d = d_0$ and is $\sqrt{d_0/d}$ times it elsewhere. {#c-attn-base}

**Multipliers** (`multiplier-equivalence`).
If a parameter enters only through $A W$, with $W_0 \sim \mathcal{N}(0, B^2)$ and learning rate $C$, then $(A\theta, B/\theta, C/\theta^2)$ trains the same function as $(A, B, C)$ under SGD. {#c-lemma-sgd}
Under Adam with $\epsilon = 0$ the matching change is $(A\theta, B/\theta, C/\theta)$. {#c-lemma-adam}
So the Table 3 readout (multiplier 1, variance $1/n^2$, rate $\eta/n$) is the same as a readout with multiplier $1/n$, variance 1, and rate $\eta n$ under SGD or $\eta$ under Adam. {#c-table8-readout}

## Worked example

Tune a proxy of width $n_0 = 256$ with Adam, base rate $\eta = 0.01$, $\sigma = 1$ and input dimension 32, then move to width $n = 4096$, so $\tilde n = 16$.

| layer | init std, μP | init std, SP | Adam rate, μP | Adam rate, SP | SGD rate, μP |
|---|---|---|---|---|---|
| input | $1/\sqrt{32} = 0.1768$ | 0.1768 | 0.01 | 0.01 | 0.16 |
| hidden | $1/\sqrt{4096} = 0.015625$ | 0.015625 | $0.01/16 = 0.000625$ | 0.01 | 0.01 |
| output | $1/\sqrt{4096 \cdot 16} = 0.00390625$ | 0.015625 | 0.000625 | 0.01 | 0.000625 |
| biases | 0 | 0 | 0.01 | 0.01 | 0.16 |

Under Adam, μP keeps the input and bias rates and divides the hidden and output rates by 16.
Under SGD it multiplies the input and bias rates by 16, keeps the hidden rate and divides the output rate by 16.
With a base head dimension of 64 and a target of 256, the attention scale is $\sqrt{64}/256 = 1/32$ under μP against $1/\sqrt{256} = 1/16$ under SP.
At $n = 256$ all three columns of μP coincide with SP.

## Hyperparameters

Under μP the transferable hyperparameters are those of optimization, initialization and parameter multipliers. Regularization, such as dropout and weight decay, does not transfer. Transfer works across width and, empirically, across depth, batch size, training time and sequence length. {#c-tp5-table1}
Regularization does not transfer because how much of it is needed depends on both model size and data size, and μP accounts only for model size. {#c-tp5-regularization}
The base width $n_0$ is not tuned; it fixes where μP and SP agree.
The Transformer experiments also sweep $\alpha_{\text{output}}$ and $\alpha_{\text{attn}}$.

## Cost

μP changes only constants, so a training step costs the same as in SP.
The saving is in tuning.
For GPT-3 6.7B the proxy had width 256 and about 40 million parameters, 168 times fewer than the target, and tuning cost 7% of pretraining. {#c-tp5-gpt3-proxy}

## Evidence

On IWSLT14 De-En, μTransfer from a proxy of a quarter the width gives validation BLEU of 35.27, 35.33, 35.45 and 35.53 at the 25th, 50th, 75th and 100th percentiles over 25 trials. Tuning the full model with the same compute gives 33.62, 35.00, 35.35 and 35.45. {#c-tp5-iwslt}
Copying the hyperparameters across in SP instead fails: training diverged. {#c-tp5-naive}
For WMT14 En-De the target is a 211M-parameter post-layernorm Transformer tuned through a 15M-parameter proxy. {#c-tp5-wmt-setup}
μTransfer reaches BLEU 25.94, 26.34 and 26.42 (worst, median and best trial), matching the fairseq default of 26.40. Tuning the target directly with three samples diverged in the worst trial and peaked at 25.69. {#c-tp5-wmt}
For BERT the proxy, BERT-prototype, has about 13M parameters and is shrunk in depth as well as width. {#c-tp5-bert-proxy}
BERT-large tuned this way reaches test loss 1.683 and MNLI 87.0/86.5, against 1.731 and 86.3/86.2 for the Megatron defaults, for a tuning cost of one BERT-large pretraining. {#c-tp5-bert}
The μTransferred GPT-3 6.7B outperforms the published 6.7B model and is comparable to the 13B model on their evaluation suite. {#c-tp5-gpt3-13b}
Its validation loss is 1.98 against 2.03 for a re-run with the original settings. Its LAMBADA zero-shot accuracy is 73.5, against 70.3 for the published 6.7B and 72.5 for the 13B model. {#c-tp5-table7}
It is not better everywhere: on LAMBADA few-shot it scores 74.7, below the re-run's 77.1 and the published 79.1. {#c-tp5-few-shot}
The comparison has two confounds: the re-run baseline mistakenly used absolute attention, and the μP model was trained in FP32 after frequent divergences, while the baselines used FP16. {#c-tp5-gpt3-caveats}

After one step in SP, a Transformer's logits and attention logits blow up with width but its word embeddings do not. In μP all three stay well-behaved. {#c-tp5-fig5}
With fixed hyperparameters, wider is better in training loss throughout training in μP, but not in SP. {#c-tp5-wider}
On Word2Vec and few-shot Omniglot, Yang and Hu's feature learning limits beat NTK baselines and finite-width networks, which approach the limit as width grows. {#c-tp4-word2vec}
On Omniglot the μP limit of a one-hidden-layer linear MLP reaches 66.42% meta-test accuracy. Its GP and NTK limits reach 41.68%, and relu GP and NTK limits 47.60% and 47.82%. {#c-tp4-omniglot}

## Limitations and failure modes

**Minimum scale.**
For Transformer language models, transfer generally holds only once the proxy reaches roughly width 256, depth 4, batch 32, sequence length 128 and 5000 steps. {#c-tp5-minimums}

**Depth.**
The optimal initialization scale shifts when depth changes, and transferring across depth works only for pre-layernorm Transformers. {#c-tp5-depth-init}

**Regularization-bound settings.**
When regularization is the bottleneck, as in finetuning on a small dataset, μTransfer may not be enough. {#c-tp5-finetune}

**An open puzzle.**
It is unclear in theory why the optimum converges quickly with width while the network and its loss converge slowly; μTransfer depends on that gap. {#c-tp5-puzzle}

**The alignment assumption.**
Everett et al. point out that μP-style derivations assume that updates are correlated enough with the data to change the activations' scaling exponent, which they call alignment. {#c-ev-alignment}
They trained tens of thousands of models with three optimizers, four parametrizations and fourteen sizes up to 26.8B parameters. {#c-ev-scale}
They find that all parametrizations, not just μP, can transfer hyperparameters, and that a per-layer learning rate rule for SP beats μP. {#c-ev-all}
For Adam their best rule keeps SP initialization and uses an embedding rate of $O(1)$ and hidden and readout rates of $O(1/n)$. {#c-ev-sp-adam}
For SGD in SP the hidden rate would ideally scale like $O(1/\sqrt n)$ and the readout rate like $O(1/n)$. {#c-ev-sgd}
This matches the $\sqrt n$ and $n$ growth derived above.
Adam's $\epsilon$ must also be scaled with width to avoid gradient underflow; their Adam-atan2 removes it. {#c-ev-epsilon}

## Variants

**Multiplier forms.**
Table 8 moves the readout's width dependence into a $1/\text{fan\_in}$ multiplier, and Table 9 follows Yang and Hu; both equal Table 3 by the multiplier lemma.

**Glorot initialization.**
In the linear perceptron example, Glorot initialization scales like μP, but with layernorm or batchnorm it causes logit blow-up like SP while μP does not. {#c-tp5-glorot}

**Lowering the SP learning rate.**
Scaling the global learning rate down with width does not fix SP. The word embeddings update by a width-independent amount per step, so at a smaller rate they effectively stop learning. {#c-tp5-sp-lower-lr}

## When to use it

Use μP when a model will be trained at a width too large to tune directly and a family of narrower versions can be trained cheaply.
Tune the learning rate, initialization scale and multipliers on the proxy, then copy them.
Do not expect regularization to transfer, and keep the proxy above the minimum sizes.
Use pre-layernorm if depth will change too.
A model already tuned in SP can be widened with $n_0$ set to its width.
An SP baseline with per-layer rates is a fair comparison.

The mean-field approximation of one-hidden-layer networks also gives a feature learning limit; μP extends it to every layer.

## Pitfalls

> [!pitfall]
> **Changing only the learning rates.** μP also shrinks the readout initialization by $\sqrt{n/n_0}$. The tests `test_worked_example_adam` and `test_initialization_matches_the_hparams` check the readout std.

> [!pitfall]
> **Using the SGD table with Adam.** The two tables differ for input, hidden and bias rates. The tests `test_worked_example_sgd` and `test_width_exponents` check both.

> [!pitfall]
> **Losing the base-width anchor.** Writing μP with bare powers of $n$ changes the model at $n_0$. The test `test_mup_equals_sp_at_the_base_width` checks that both rules agree there.

> [!pitfall]
> **Keeping $1/\sqrt d$ attention.** Aligned queries and keys then grow with $d$. The tests `test_attention_logits` and `test_attention_of_aligned_vectors_stays_bounded_under_mup` check the $1/d$ scale.

> [!pitfall]
> **Not testing the scaling.** A μP implementation can be checked cheaply by measuring how much features move in one step at several widths. The test `test_mup_keeps_the_feature_update_constant_and_sp_does_not` does this. Yang et al. suggest checking that wider is better early in training. {#c-tp5-debug}

## History

Yang and Hu posted abc-parametrizations, the Dynamical Dichotomy and μP in 2020.
Yang et al. (2022) extended μP to Adam and Transformers and proposed μTransfer.
They argued that μP is the unique feature learning parametrization that updates every parameter maximally, so each learning rate plays roughly the same role at finite width as in the limit. {#c-tp5-maximal}
Everett et al. (2024) revisited its alignment assumption at scale.
