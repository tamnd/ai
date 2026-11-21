---
tag: N79XJF
kind: architecture
slug: discrete-diffusion-model
name: Discrete diffusion model
aliases: [discrete denoising diffusion, D3PM, structured denoising diffusion, multinomial diffusion, masked diffusion model, absorbing diffusion, masked diffusion language model, MDLM, MD4, score entropy discrete diffusion, SEDD, masked generative transformer, MaskGIT]
area: generative-models/discrete-and-masked-generative-models
level: core
summary: >-
  A diffusion model for tokens: a Markov chain corrupts a sequence step by step, by random replacement or masking, and a network learns to undo each step.
uses: [markov-chain@m2, transition-matrix@m2, absorbing-markov-chain@m2, continuous-time-markov-chain@m2, stationary-distribution@m2, conditional-probability@m2, bayes-theorem@m2, joint-probability-distribution@m2, independence-of-random-variables@m2, expected-value@m2, beta-distribution@m2, kullback-leibler-divergence@m2, cross-entropy@m2, total-variation-distance-between-probability-measures@m2, rao-blackwell-theorem@m2, matrix-multiplication@m2, evidence-lower-bound@al, cross-entropy-loss@al, ancestral-sampling@al, backpropagation@al]
related_to: [diffusion-model, autoregressive-model, bert, transformer, vision-transformer, energy-based-model, normalizing-flow, flow-matching, variational-autoencoder, adam]
history:
  year: 2021
  people: [Emiel Hoogeboom, Max Welling, Jacob Austin, Daniel D. Johnson, Jonathan Ho, Daniel Tarlow, Rianne van den Berg, Huiwen Chang, Han Zhang, Aaron Lou, Chenlin Meng, Stefano Ermon, Subham Sekhar Sahoo, Volodymyr Kuleshov, Jiaxin Shi, Michalis Titsias]
  note: "Sohl-Dickstein and colleagues (2015) defined diffusion for binary data alongside the Gaussian case. Hoogeboom and colleagues' multinomial diffusion (2021) noised categories by uniform resampling, and Austin and colleagues' D3PM (2021) allowed any transition matrix, with an absorbing mask state that tied diffusion to BERT and to autoregressive models. MaskGIT (2022) generated images by iterative parallel unmasking, SEDD (2023) learned ratios of probabilities with score entropy, and MDLM and MD4 (2024) reduced masked diffusion to a weighted sum of masked-token cross-entropies that approaches autoregressive perplexity."
prereqs: [diffusion-model, transformer, markov-chain@m2, kullback-leibler-divergence@m2, evidence-lower-bound@al]
sources:
  - {paper: "Austin, Johnson, Ho, Tarlow and van den Berg (2021)", title: "Structured Denoising Diffusion Models in Discrete State-Spaces", locator: "NeurIPS 2021; arXiv 2107.03006v3, pp. 1 to 7", note: "transition matrices, the posterior, the x0-parameterization, the auxiliary loss, BERT and autoregressive models as diffusion, and the text8 and LM1B results"}
  - {paper: "Hoogeboom, Nielsen, Jaini, Forré and Welling (2021)", title: "Argmax Flows and Multinomial Diffusion: Learning Categorical Distributions", locator: "NeurIPS 2021; arXiv 2102.05379, pp. 5, 7", note: "the multinomial diffusion process, its marginal and posterior, and the text8 result"}
  - {paper: "Chang, Zhang, Jiang, Liu and Freeman (2022)", title: "MaskGIT: Masked Generative Image Transformer", locator: "arXiv 2202.04200v1, pp. 2, 4, 5, 8", note: "iterative parallel decoding by confidence, the mask schedule and the ImageNet results"}
  - {paper: "Lou, Meng and Ermon (2023)", title: "Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution", locator: "ICML 2024, PMLR 235; pp. 1, 3, 6, 7", note: "the concrete score, score entropy and its denoising form, and the language modelling results"}
  - {paper: "Sahoo, Arriola, Schiff, Gokaslan, Marroquin, Chiu, Rush and Kuleshov (2024)", title: "Simple and Effective Masked Diffusion Language Models", locator: "NeurIPS 2024; arXiv 2406.07524, pp. 4 to 9", note: "the SUBS parameterization, the continuous-time bound as a weighted average of masked language modelling losses, caching, semi-autoregressive decoding and the results"}
  - {paper: "Shi, Han, Wang, Doucet and Titsias (2024)", title: "Simplified and Generalized Masked Diffusion for Discrete Data", locator: "NeurIPS 2024; arXiv 2406.04329v4, pp. 1, 3, 4, 8, 9", note: "the bound as a weighted integral of cross-entropies, its invariance to the schedule, state-dependent schedules and the text and image results"}
family: diffusion-models
inputs: "for training, a batch of token sequences as a (B, L) integer array with entries in 0, ..., K - 1; for sampling, the number of sequences B, their length L and the number of denoising steps T"
outputs: "for training, an upper bound on the negative log-likelihood of each sequence in nats; for the denoiser, logits over the K data tokens at every position as a (B, L, K) array; for sampling, B sequences as a (B, L) array"
components: [transformer]
inductive_biases: ["a sequence is built by refining all positions in parallel, so every token can depend on context on both sides, unlike a left-to-right model", "under masking noise a token, once revealed, is never changed, so the order of generation is a random permutation drawn by the noise schedule", "the denoiser predicts each masked token independently given the visible ones, so tokens revealed in the same step are drawn from a product of marginals", "the corruption process encodes which tokens are close: uniform noise treats all tokens alike, absorbing noise hides them, and structured matrices prefer near neighbours"]
params: "the parameters of the denoiser; for the one-hidden-layer denoiser of this page with L tokens, K data symbols and H hidden units, L(K + 1)H + H + HLK + LK"
flops_per_example: "one denoiser call per sampling step, at most T calls for T steps; 2L(K + 1)H + 2HLK for one call of the denoiser of this page on one sequence, counting a multiply-add as 2 FLOPs"
---

A discrete diffusion model generates sequences of tokens, such as text, image codes or molecules, by learning to reverse a process that corrupts them.
The forward process is a [[m2:markov-chain|Markov chain]] that changes each token a little at a time: it resamples tokens uniformly at random, moves them to nearby tokens, or replaces them with a special mask token, until nothing of the original remains.
A network, usually a [[transformer]], learns to predict the original tokens from a corrupted sequence, and the reverse chain built from its predictions turns pure noise, or a sequence of masks, into data in a fixed number of parallel steps.
Training minimizes a variational bound on the negative log-likelihood, and for masking noise the bound reduces to a weighted sum of the cross-entropies of predicting masked tokens, the loss of [[bert]] with a random masking rate.
Masked diffusion language models of this kind come within about 15% of the perplexity of [[autoregressive-model|autoregressive models]] of the same size, and decode many tokens at once.

## In plain words

Take a sentence and hide its words one at a time behind a blank, in a random order, until only blanks are left.
Now play the game backwards: start from a row of blanks and fill some of them in, look at what you have, fill in a few more, and stop when no blanks are left.
A discrete diffusion model learns the backwards game.
Its network looks at a partly hidden sentence and guesses, for every blank, which word belongs there.
To generate, it starts from all blanks and at each step reveals some of the blanks using its guesses, keeping the words it has already revealed.
Unlike a model that writes left to right, it can fill in the end of a sentence before the beginning, and it can fill in many words in one step.
The price of filling in many words at once is that the guesses for those words are made independently, so two words chosen in the same step may not fit each other; more, smaller steps fix this.
Instead of blanks, the corruption can also replace words with random words, and the model then learns to spot and correct the wrong ones.

## Overview

A token takes one of $K$ values and is written as a one-hot row vector $\mathbf{x}$.
The forward process corrupts a token $\mathbf{x}_0$ through $T$ steps with [[m2:transition-matrix|transition matrices]] $Q_t$, where $[Q_t]_{ij} = q(x_t = j \mid x_{t-1} = i)$:

$$ q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \operatorname{Cat}(\mathbf{x}_t ;\, \mathbf{x}_{t-1} Q_t) , \qquad q(\mathbf{x}_t \mid \mathbf{x}_0) = \operatorname{Cat}(\mathbf{x}_t ;\, \mathbf{x}_0 \bar{Q}_t) , \quad \bar{Q}_t = Q_1 Q_2 \cdots Q_t . $$ {#eq-d3pm-forward}

Each token of a sequence is corrupted independently with the same matrices.
Austin and colleagues introduced Discrete Denoising Diffusion Probabilistic Models (D3PMs) for discrete data, which generalize the multinomial diffusion model of Hoogeboom and colleagues by going beyond corruption processes with uniform transition probabilities, to matrices that mimic Gaussian kernels, matrices based on nearest neighbours in embedding space, and matrices that introduce absorbing states; the third connects diffusion models with autoregressive and mask-based generative models. {#c-d3pm-def}
The two matrices that matter most are the uniform one and the absorbing one, with $\beta_t \in [0, 1]$:

$$ Q_t^{\text{uniform}} = (1 - \beta_t) I + \frac{\beta_t}{K} \mathbf{1}\mathbf{1}^\top , \qquad Q_t^{\text{absorb}} = (1 - \beta_t) I + \beta_t \mathbf{1} \mathbf{e}_m^\top , $$

where the absorbing matrix acts on $K + 1$ states and $\mathbf{e}_m$ is the one-hot of the mask state $m$.
The uniform chain has the uniform distribution as its [[m2:stationary-distribution|stationary distribution]], and the absorbing chain is an [[m2:absorbing-markov-chain|absorbing Markov chain]] whose only absorbing state is the mask.
With $\bar\alpha_t = \prod_{s \le t} (1 - \beta_s)$, the products have the closed forms $\bar\alpha_t I + (1 - \bar\alpha_t)\mathbf{1}\mathbf{1}^\top/K$ and $\bar\alpha_t I + (1 - \bar\alpha_t)\mathbf{1}\mathbf{e}_m^\top$: after $t$ steps a token is intact with probability $\bar\alpha_t$ and otherwise uniform, or masked.
Hoogeboom and colleagues defined multinomial diffusion by a categorical distribution that has a $\beta_t$ chance of resampling a category uniformly, with the marginal $q(\mathbf{x}_t \mid \mathbf{x}_0) = \operatorname{Cat}(\bar\alpha_t \mathbf{x}_0 + (1 - \bar\alpha_t)/K)$ and a categorical posterior in closed form. {#c-multinomial-def}

### The posterior and the reverse process

By [[m2:bayes-theorem|Bayes' rule]] and the Markov property, the forward step given both ends is

$$ q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) = \operatorname{Cat}\!\left(\mathbf{x}_{t-1} ;\, \frac{\mathbf{x}_t Q_t^\top \odot \mathbf{x}_0 \bar{Q}_{t-1}}{\mathbf{x}_0 \bar{Q}_t \mathbf{x}_t^\top}\right) . $$ {#eq-d3pm-posterior}

The reverse process does not predict $\mathbf{x}_{t-1}$ directly. A network predicts a distribution $\tilde p_\theta(\tilde{\mathbf{x}}_0 \mid \mathbf{x}_t)$ over the clean token, and the reverse step averages the posterior over it:

$$ p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) \propto \sum_{\tilde{\mathbf{x}}_0} q(\mathbf{x}_{t-1}, \mathbf{x}_t \mid \tilde{\mathbf{x}}_0)\, \tilde p_\theta(\tilde{\mathbf{x}}_0 \mid \mathbf{x}_t) . $$ {#eq-x0-param}

Under this x0-parameterization the KL divergence from the posterior is zero when $\tilde p_\theta$ puts all its mass on the original $\mathbf{x}_0$; the sparsity pattern of $Q_t$ fixes the sparsity pattern of the ideal reverse transitions, which the parameterization respects automatically, and it allows steps of $k$ at a time by predicting $p_\theta(\mathbf{x}_{t-k} \mid \mathbf{x}_t) \propto \sum q(\mathbf{x}_{t-k}, \mathbf{x}_t \mid \tilde{\mathbf{x}}_0)\tilde p_\theta(\tilde{\mathbf{x}}_0 \mid \mathbf{x}_t)$. {#c-x0-param}
The uniform case of eq-d3pm-posterior is exactly the posterior of multinomial diffusion:

For the uniform matrices, eq-d3pm-posterior equals the multinomial diffusion posterior $\tilde\theta \propto [\alpha_t \mathbf{x}_t + (1 - \alpha_t)/K] \odot [\bar\alpha_{t-1}\mathbf{x}_0 + (1 - \bar\alpha_{t-1})/K]$ with $\alpha_t = 1 - \beta_t$, for a one-hot $\mathbf{x}_0$ and for any probability vector in its place, so multinomial diffusion is the uniform D3PM. {#c-multinomial-is-uniform}
The derivation `multinomial-posterior` proves it, and `test_posterior_is_bayes_rule_and_matches_multinomial_diffusion` checks both forms against Bayes' rule.

### The variational bound

The model is trained by the [[al:evidence-lower-bound|evidence lower bound]] of a latent-variable model whose latents are $\mathbf{x}_{1:T}$:

$$ L_{\text{vb}} = \mathbb{E}_q\Big[ \mathrm{KL}\big(q(\mathbf{x}_T \mid \mathbf{x}_0) \,\|\, p(\mathbf{x}_T)\big) + \sum_{t=2}^{T} \mathrm{KL}\big(q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) \,\|\, p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)\big) - \log p_\theta(\mathbf{x}_0 \mid \mathbf{x}_1) \Big] \ \ge\ -\log p_\theta(\mathbf{x}_0) . $$ {#eq-vlb}

Each term is a [[m2:kullback-leibler-divergence|KL divergence]] between categorical distributions and can be computed exactly.
Austin and colleagues add an auxiliary [[m2:cross-entropy|cross-entropy]] on the prediction of $\mathbf{x}_0$, $L_\lambda = L_{\text{vb}} + \lambda\, \mathbb{E}_{q(\mathbf{x}_0)}\mathbb{E}_{q(\mathbf{x}_t \mid \mathbf{x}_0)}[-\log \tilde p_\theta(\mathbf{x}_0 \mid \mathbf{x}_t)]$; under the x0-parameterization both the auxiliary term and the KL terms are minimized exactly when $\tilde p_\theta$ has all its mass on the data point $\mathbf{x}_0$, and training with this loss improved the quality of their image samples. {#c-aux-loss}

The x0-parameterization has a trap for uniform noise. The best $\tilde p_\theta$ is not the posterior over the clean token:

For a single token whose last step reaches the stationary distribution, eq-x0-param with $\tilde p_\theta(\cdot \mid \mathbf{x}_t) = q(\mathbf{x}_0)$, the data distribution regardless of $\mathbf{x}_t$, gives exactly the true reverse step $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$, so the expected bound equals the entropy of the data and no prediction does better. The posterior mean $q(\mathbf{x}_0 \mid \mathbf{x}_t)$ gives the same reverse step under absorbing noise but not under uniform noise, because eq-x0-param already multiplies by $q(\mathbf{x}_t \mid \mathbf{x}_0)$ and the posterior counts it a second time. {#c-x0-optimum}
The derivation `x0-parameterization-optimum` proves it, and `test_d3pm_bound_equals_the_entropy_at_its_optimum` finds the posterior mean worse than the entropy by more than 0.05 nats under uniform noise and equal to it under absorbing noise.
In a sequence the network sees the other tokens too, and the same argument makes the best prediction the distribution of the clean token given the others, not given the noisy token itself.
Masking avoids the trap entirely, because a masked token carries no information about its own value.

## Components

### Masked diffusion in continuous time

Masked diffusion uses only the absorbing matrix, which leads to the simplest and strongest discrete diffusion models.
In continuous time, a schedule $\alpha_t$ decreasing from $\alpha_0 = 1$ to $\alpha_1 = 0$ gives each token its probability of still being visible at time $t \in [0, 1]$, and

$$ q(\mathbf{z}_t \mid \mathbf{x}) = \operatorname{Cat}\big(\mathbf{z}_t ;\, \alpha_t \mathbf{x} + (1 - \alpha_t)\mathbf{m}\big) , $$ {#eq-mask-marginal}

where $\mathbf{m}$ is the one-hot of the mask.
A token is hidden at a random time, its masking time, and stays hidden; with the linear schedule $\alpha_t = 1 - t$ that time is uniform on $[0, 1]$, and the cosine schedule $\alpha_t = 1 - \sin(\pi t/2) = 1 - \cos(\pi(1 - t)/2)$ masks fewer tokens early.
The posterior from $t$ back to $s < t$ is

$$ q(\mathbf{z}_s \mid \mathbf{z}_t, \mathbf{x}) = \begin{cases} \operatorname{Cat}(\mathbf{z}_s ;\, \mathbf{z}_t) & \mathbf{z}_t \ne \mathbf{m} , \\ \operatorname{Cat}\!\Big(\mathbf{z}_s ;\, \dfrac{(1 - \alpha_s)\mathbf{m} + (\alpha_s - \alpha_t)\mathbf{x}}{1 - \alpha_t}\Big) & \mathbf{z}_t = \mathbf{m} : \end{cases} $$ {#eq-mask-posterior}

a visible token stays as it is, and a masked token is revealed with probability $(\alpha_s - \alpha_t)/(1 - \alpha_t)$, as its true value.

### The SUBS parameterization

The reverse step replaces $\mathbf{x}$ in eq-mask-posterior with a network's prediction $\mathbf{x}_\theta(\mathbf{z}_t)$, a probability vector over the $K$ data tokens.
Sahoo and colleagues impose two substitutions, called SUBS: zero masking probabilities, setting the logit of the mask to $-\infty$ so that $\langle \mathbf{x}_\theta(\mathbf{z}_t), \mathbf{m}\rangle = 0$ as for the clean data; and carry-over unmasking, copying an unmasked $\mathbf{z}_t$ to the output so that unmasked latents are carried over. {#c-subs}
With SUBS,

$$ p_\theta(\mathbf{z}_s \mid \mathbf{z}_t) = q\big(\mathbf{z}_s \mid \mathbf{z}_t, \mathbf{x} = \mathbf{x}_\theta(\mathbf{z}_t)\big) , \qquad \mathbf{x}_\theta(\mathbf{z}_t) = \begin{cases} \mathbf{z}_t & \mathbf{z}_t \ne \mathbf{m} , \\ \operatorname{softmax}(f_\theta(\mathbf{z}_t)) & \mathbf{z}_t = \mathbf{m} . \end{cases} $$ {#eq-subs}

The KL term of a step is then nonzero only at masked positions, and there it is a cross-entropy:

$$ \mathrm{KL}\big(q(\mathbf{z}_s \mid \mathbf{z}_t, \mathbf{x}) \,\|\, p_\theta(\mathbf{z}_s \mid \mathbf{z}_t)\big) = \frac{\alpha_s - \alpha_t}{1 - \alpha_t} \big(-\log \langle \mathbf{x}_\theta(\mathbf{z}_t), \mathbf{x}\rangle\big) \quad \text{for } \mathbf{z}_t = \mathbf{m} . $$ {#eq-mask-kl}

`test_masked_posterior_and_kl_term` checks eq-mask-posterior against the general D3PM posterior and eq-mask-kl against the KL computed directly.

### The bound as weighted masked-token cross-entropy

Summing eq-mask-kl over steps and letting $T \to \infty$ gives the continuous-time bound on the negative log-likelihood of a sequence $\mathbf{x}^{1:L}$, whose tokens are masked independently:

$$ \mathcal{L}_\infty = \int_0^1 \frac{-\alpha_t'}{1 - \alpha_t}\, \mathbb{E}_{q(\mathbf{z}_t \mid \mathbf{x})}\Big[\sum_{\ell :\, \mathbf{z}_t^\ell = \mathbf{m}} -\log \langle \mathbf{x}_\theta^\ell(\mathbf{z}_t), \mathbf{x}^\ell\rangle\Big]\, dt \ \ge\ -\log p_\theta(\mathbf{x}) . $$ {#eq-mdlm}

Sahoo and colleagues note that the objective is a weighted average of masked language modelling losses, which connects generative diffusion models with encoder-only BERT models, allows a principled choice of a randomized masking rate, and gives BERT-style models principled generation. {#c-mdlm-mlm}
Shi and colleagues reached the same form independently: the continuous-time variational objective of masked diffusion models is a simple weighted integral of cross-entropy losses. {#c-md4-ce}
The integral does not depend on the shape of the schedule.
The diffusion loss is invariant to the functional form of $\alpha_t$, which Sahoo and colleagues derived by a change of variables to $\gamma = \log(1 - \alpha_t)$ and verified empirically. {#c-mdlm-invariant}
Shi and colleagues stated the same result through the log-SNR $\lambda_t = \log(\alpha_t/(1 - \alpha_t))$: as for continuous state-space diffusions, whose bound depends on the signal-to-noise ratio only at its endpoints, the bound is otherwise invariant to the noise schedule. {#c-md4-snr}
When the network ignores $t$, the integral has a closed form that makes this exact:

For a denoiser that does not depend on $t$ and a schedule decreasing from $\alpha_0 = 1$ to $\alpha_1 = 0$, eq-mdlm equals $\sum_{M} \frac{(L - |M|)!\,(|M| - 1)!}{L!} \sum_{\ell \in M} -\log \langle \mathbf{x}_\theta^\ell(\mathbf{x}_M), \mathbf{x}^\ell\rangle$, a sum over the nonempty sets $M$ of masked positions, where $\mathbf{x}_M$ is $\mathbf{x}$ with the positions in $M$ masked; the coefficient is a Beta integral and does not depend on the schedule. {#c-mask-coefficient}
The derivation `masked-bound-coefficients` proves it, and `test_nelbo_is_the_any_order_chain_rule_and_ignores_the_schedule` computes eq-mdlm by Gauss-Legendre quadrature under the linear and cosine schedules and finds both equal to the closed form to $10^{-10}$.
The coefficients give each mask size $m$ total weight $\binom{L}{m}\frac{(L - m)!(m - 1)!}{L!} = 1/m$ per masked token, so every size contributes equally, and with the exact denoiser the bound is tight:

If $\mathbf{x}_\theta^\ell(\mathbf{x}_M)$ is the true conditional distribution of the token at $\ell$ given the visible tokens, eq-mdlm equals $-\log p(\mathbf{x})$ exactly, for every $\mathbf{x}$: the bound averages the chain rule of probability over all orders in which the tokens could be revealed. {#c-any-order}
The derivation `any-order-chain-rule` proves it, and the same test checks it for every sequence of Markov data with $L$ up to 4.
So masked diffusion is an any-order autoregressive model: the loss of eq-mdlm averages, over random orders, the losses of [[autoregressive-model|autoregressive models]] that reveal the tokens in that order, and the network must solve every one of those prediction problems at once.

### Sampling

Ancestral sampling starts from all masks at $t = 1$ and applies eq-subs at times $1 = t_T > t_{T-1} > \dots > t_0 = 0$.
At each step every masked token is revealed with probability $(\alpha_s - \alpha_t)/(1 - \alpha_t)$, and the revealed tokens are drawn from $\mathbf{x}_\theta$ independently.
This is [[al:ancestral-sampling|ancestral sampling]] of the reverse chain.
Because the network does not see $t$ in MDLM, a step that reveals nothing leaves its input unchanged, and the previous prediction can be reused.
When no token changes in a step the denoiser's output can be cached and the step skipped, saving a call of the network; SEDD cannot cache this way because its network models time-dependent rates and has to be conditioned on time. {#c-mdlm-caching}
Tokens revealed in the same step are drawn independently given the visible ones, which is the source of the error of few-step sampling:

With the exact denoiser, ancestral sampling returns the data distribution exactly whenever no two tokens are revealed in the same step, so its total variation distance from the data is at most the probability that some step reveals two or more tokens; under the linear schedule with $T$ equal steps each token's step is uniform and independent, and that probability is $1 - T!/((T - L)!\,T^L)$, which tends to zero as $T$ grows. With $T = 1$ the sampler draws every token from its marginal. {#c-sampler-collisions}
Shi and colleagues observed the same effect from the schedule side: the cosine schedule slows down unmasking at the start of generation, which lowers the chance that conflicting tokens are unmasked at the same time and improves the samples. {#c-md4-cosine}
The derivation `ancestral-sampler-collisions` proves it, and `test_ancestral_sampler_distribution` computes the exact output distribution of the sampler for $T = 1$ to 32 and finds the [[m2:total-variation-distance-between-probability-measures|total variation distance]] falling from the product of marginals to below 0.025.

### Iterative decoding by confidence

MaskGIT decodes differently. At each of $T$ iterations a bidirectional transformer predicts probabilities for every masked location in parallel; a token is sampled at each masked location and its probability is used as a confidence score, set to 1 for already unmasked positions; the number of tokens to keep masked is $n = \lceil \gamma(t/T) N \rceil$ for the mask schedule $\gamma$ and $N$ tokens; and the $n$ least confident predictions are masked again. {#c-maskgit-decode}
Chang and colleagues note that the model could in theory infer all tokens and generate the whole image in a single pass, but found this challenging due to its inconsistency with the training task. {#c-maskgit-one-pass}
The mask scheduling function is used in both training and iterative decoding; concave functions generally obtained better FID and IS than the linear one, followed by the convex ones, and cosine slightly edged out square in all scores, which made it the default. {#c-maskgit-cosine}
Choosing tokens by confidence instead of at random is not ancestral sampling of eq-subs, and it favours likely tokens: `test_maskgit_decoding` finds that decoding one token per step, the most confident first, gives samples whose mean log-probability exceeds the negative entropy of the data by more than one nat.

### Score entropy

Lou, Meng and Ermon learn ratios instead of a denoiser.
The concrete score of a distribution $p$ at $x$ is the vector of ratios $p(y)/p(x)$ over the neighbours $y$ of $x$, the discrete counterpart of the gradient of $\log p$, and a network $s_\theta(x)_y$ learns it by minimizing the score entropy

$$ \mathcal{L}_{\text{SE}} = \mathbb{E}_{x \sim p}\Big[\sum_{y \ne x} w_{xy}\Big(s_\theta(x)_y - \frac{p(y)}{p(x)} \log s_\theta(x)_y + K\Big(\frac{p(y)}{p(x)}\Big)\Big)\Big] , \qquad K(a) = a(\log a - 1) , $$ {#eq-score-entropy}

where $K$ is a normalizing constant function that ensures $\mathcal{L}_{\text{SE}} \ge 0$; score entropy is a Bregman divergence with $F = -\log$, so it is non-negative, symmetric and convex, and it generalizes cross-entropy to positive values. {#c-sedd-score-entropy}
Its minimum is at the true ratios:

Each term of eq-score-entropy is $a\,(r - \log r - 1)$ with $r = s/a$ and $a$ the true ratio, which is non-negative and zero exactly at $s = a$; so for positive weights score entropy is zero exactly at the concrete score and positive elsewhere, and the denoising form differs from it by a constant that does not depend on $s$, so it has the same minimizer. {#c-score-entropy-min}
The derivation `score-entropy-minimum` proves it, and `test_score_entropy` checks both statements and that the gradient of the denoising form vanishes at the true ratios.
The ratios are unknown, so training uses the denoising form: if $p$ is a perturbation of $p_0$ by a kernel, $p(x) = \sum_{x_0} p(x \mid x_0) p_0(x_0)$, then score entropy equals, up to a constant independent of $\theta$, the denoising score entropy $\mathbb{E}_{x_0 \sim p_0,\, x \sim p(\cdot \mid x_0)}\big[\sum_{y \ne x} w_{xy}\big(s_\theta(x)_y - \frac{p(y \mid x_0)}{p(x \mid x_0)} \log s_\theta(x)_y\big)\big]$. {#c-sedd-dse}
For masking noise the concrete score and the denoiser carry the same information:

Under masking noise with visibility $\alpha$, the ratio of the marginal probability of a partly masked sequence with the masked position $i$ filled in with $y$ to that of the sequence itself is $\frac{\alpha}{1 - \alpha}\, p(x^i = y \mid \text{visible tokens})$, the true denoiser times a factor that depends only on time. {#c-absorbing-ratio}
The derivation `absorbing-concrete-score` proves it, and `test_absorbing_concrete_score_is_the_denoiser_times_alpha_ratio` checks it for every partly masked sequence of a small distribution.
This is why MDLM and MD4 can drop the time input and SEDD cannot: the time dependence of the masked concrete score is a known factor, which a score network has to learn and a denoiser does not.

## Shapes and parameters

The shape contract is the denoiser of the implementations: a network that reads a partly masked sequence of $L$ tokens, each one of $K$ data symbols or the mask, as a one-hot vector of length $L(K + 1)$, passes it through one tanh hidden layer of width $H$, and returns logits over the $K$ data tokens at every position.
The mask gets no logit, which is the zero-masking half of eq-subs; the carry-over half happens after the network.
With weights $W_1$ of shape $(L(K + 1), H)$ and $W_2$ of shape $(H, LK)$ and their biases, the parameter count is

$$ P = L(K + 1)H + H + HLK + LK . $$ {#eq-params}

In the contract this is
{shape:params}
and the example with $L = 3$, $K = 3$ and $H = 5$ has {shape:params@example} parameters.
The denoiser trained in the tests, `markov`, has {shape:params@markov}, and the same form on a 256-character chunk of text8, `text8_chunk`, would have {shape:params@text8_chunk}.
Real masked diffusion models replace this network with a bidirectional [[transformer]], whose parameter count does not grow with $L$; MDLM, MD4 and SEDD use transformers of the size of GPT-2 small and medium, and MaskGIT one of 24 layers on $16 \times 16$ grids of image tokens.
A transformer with $d$ channels and a vocabulary of $K + 1$ inputs and $K$ outputs adds one embedding row for the mask and nothing else, so a masked diffusion model has the same size as an autoregressive model of the same architecture.
The derivation `parameter-and-flop-count` proves the counts.

## Computation and cost

One call of the denoiser on $B$ sequences costs

$$ C = 2BL(K + 1)H + 2BHLK $$ {#eq-flops}

FLOPs for its two matrix products, a multiply-add counted as two, leaving out the biases, the tanh and the softmax.
For the denoiser of this page with $L$ tokens, $K$ symbols and $H$ hidden units, the parameter count is $L(K + 1)H + H + HLK + LK$, one call on $B$ sequences costs $2BL(K + 1)H + 2BHLK$ FLOPs, and the exact bound of one sequence by c-mask-coefficient costs $2^L - 1$ calls, one per nonempty set of masked positions. {#c-params}
In the contract,
{shape:flops_forward}
which is {shape:flops_forward@example} FLOPs for the example batch and {shape:flops_forward@markov} for the 1215 rows of the exact bound of the `markov` test.
Training costs one forward and one backward pass per sequence, the same as for an autoregressive model: the one-sample estimate of eq-mdlm draws one $t$ and one mask per sequence and computes the cross-entropy at the masked positions.
The estimate has more variance than an autoregressive loss, which predicts every token every time, while a masked diffusion step predicts only the masked tokens of one random mask, about half of them on average.
Sampling costs at most $T$ network calls, each on the whole sequence.
An autoregressive transformer with a key-value cache costs about one sequence's worth of compute in total for $L$ tokens, while a masked diffusion model with $T$ steps costs about $T$ full passes; the diffusion model is faster in wall-clock time when $T \ll L$, because each pass is parallel over positions, and slower in FLOPs.
With caching, steps that reveal no token are free, so with $T \ge L$ the number of calls is at most $L$.

## Training

The recipe of MDLM, which the implementations follow, is:

1. take a batch $\mathbf{x}$ of sequences;
2. draw a time $t \sim U(0, 1]$ per sequence and mask each token independently with probability $1 - \alpha_t$ to get $\mathbf{z}_t$;
3. compute the denoiser's logits on $\mathbf{z}_t$ and their log-softmax over the data tokens;
4. take the [[al:cross-entropy-loss|cross-entropy]] of the true tokens at the masked positions, sum per sequence, and multiply by $-\alpha_t'/(1 - \alpha_t)$, which is $1/t$ for the linear schedule;
5. average over the batch, and take a gradient step with [[al:backpropagation|backpropagation]] and an optimizer such as [[adam]].

This is an unbiased estimate of eq-mdlm, and `test_monte_carlo_estimate_is_unbiased` checks it against the exact bound with 40,000 draws under both schedules.
The tests also train by the exact bound, enumerating every sequence of a small distribution and every mask with the weights of c-mask-coefficient, which removes the sampling noise; `test_trained_denoiser_reaches_the_entropy_and_samples_the_data` trains a 32-unit denoiser on four tokens over three symbols for 600 Adam steps and finds the bound within 0.01 nats of the entropy of the data.
Computing the expectations that SUBS fixes analytically, such as the zero probability of the mask, instead of leaving the network to learn them, is what Sahoo and colleagues call, somewhat loosely, Rao-Blackwellization, after the [[m2:rao-blackwell-theorem|Rao-Blackwell theorem]], because it lowers the variance of the estimate; the exact training of the tests goes further and also sums over the masks and integrates over time.
Removing the carry-over substitution from SUBS raised the perplexity of MDLM by 1.5 points, while further removing zero masking made no meaningful difference, and a model trained with $T = 1000$ discrete steps was worse by 0.1 than one trained in continuous time. {#c-mdlm-ablation}
For D3PMs with uniform noise, the auxiliary cross-entropy of c-aux-loss with a small $\lambda$, such as 0.01, helps the network learn to predict $\mathbf{x}_0$ early in training.

## Inductive biases

A discrete diffusion model assumes nothing about the order of the tokens.
An autoregressive model fixes one order and conditions each token on the ones before it; masked diffusion trains on all orders at once, by c-any-order, and can condition on any subset of tokens, which suits infilling, editing and data such as images and molecules with no natural order.
The cost of order-freedom is that the network must solve many more prediction problems than an autoregressive model, which is one reason it needs more training to reach the same likelihood.
The denoiser predicts the masked tokens independently given the visible ones, a factorization that is exact one token at a time and approximate when several tokens are revealed together, by c-sampler-collisions.
The noise process is a choice of which tokens are close: uniform noise treats all tokens alike, so the model must learn to detect and correct substitutions anywhere; absorbing noise hides tokens without misleading, so a visible token can always be trusted; and nearest-neighbour or Gaussian-like matrices encode a similarity between tokens, such as between nearby pixel intensities.
Austin and colleagues found that the choice of transition matrix matters, and that absorbing noise did best on text.

## Evidence

On text8, character-level text with 27 symbols in chunks of 256, D3PM with 1000 steps reached at most 1.61 bits per character with uniform noise, 1.59 with nearest-neighbour noise and 1.45 with absorbing noise trained by $L_{\lambda = 0.01}$, against 1.23 for a transformer decoder, all of them 12-layer transformers; with 20 steps the bounds were 1.79, 1.75 and 1.56. {#c-d3pm-text8}
Multinomial diffusion had reached at most 1.72 bits per character on text8 and 1.75 bits per byte on enwik8. {#c-hoog-text8}
On LM1B, D3PM with 1000 steps reached perplexities of 137.9 with uniform noise, 149.5 with nearest-neighbour noise and 76.9 with absorbing noise, against 43.6 for their autoregressive transformer. {#c-d3pm-lm1b}
On LM1B, MDLM trained for 33B tokens reached a perplexity bound of at most 27.04 against at most 32.79 for SEDD trained as long, a 17% improvement, and at most 23.00 after 327B tokens, against 22.32 and 20.86 for an autoregressive transformer trained for 33B and 327B tokens, getting within 14% of the autoregressive baseline; on OpenWebText it reached at most 23.21 against 24.10 for SEDD and 17.54 for an autoregressive transformer. {#c-mdlm-results}
On text8, MD4 reached at most 1.37 bits per character and its generalization with state-dependent schedules, GenMD4, at most 1.34, against 1.39 for SEDD with absorbing noise and 1.45 for D3PM; on pixel-level images MD4 reached 2.75 bits per dimension on CIFAR-10 and 3.40 on ImageNet $64 \times 64$, better than autoregressive models of similar sizes. {#c-md4-results}
SEDD, for comparable model sizes, reduced perplexity by 25 to 75% relative to earlier language diffusion models, was competitive with autoregressive models, beat GPT-2 on a majority of the zero-shot perplexity tasks, and matched the sample quality of GPT-2 with 32 times fewer network evaluations. {#c-sedd-results}
On ImageNet $256 \times 256$, MaskGIT reached an FID of 6.18 against 15.78 for VQGAN, without beam search, top-$k$ or nucleus sampling or classifier guidance, and 7.32 at $512 \times 512$; it decoded an image in 8 steps instead of the 256 of autoregressive decoding and was 30 to 64 times faster than VQGAN's decoding. {#c-maskgit-results}
The tests of this page reproduce the main identities on small problems: the exact bound equals $-\log p(\mathbf{x})$ with the exact denoiser, it does not depend on the schedule, the sampler's error falls with the number of steps, and a trained denoiser reaches the entropy and samples the data within a total variation of 0.06 in 16 steps.

## Limitations

Masked diffusion language models still trail autoregressive models of the same size in perplexity, by about 14% on LM1B for MDLM by c-mdlm-results, and the gap was larger for uniform noise.
The reported likelihoods of diffusion models are upper bounds, so the comparison with exact autoregressive likelihoods is conservative for the diffusion model, but the gap is not an artefact of the bound: with the exact denoiser the bound is tight, by c-any-order.
Sampling in few steps draws several tokens at once from a product of marginals, which loses the dependencies between them, by c-sampler-collisions; with many steps the cost approaches one network pass per token, which is more than an autoregressive model with a key-value cache needs.
A token revealed by masked diffusion can never be changed, so an early mistake stays; uniform noise allows correction but trains worse, and remasking samplers trade the guarantee of eq-subs for the ability to revise.
Confidence-based decoding as in MaskGIT is not a sampler of the trained model: it concentrates on likely tokens and loses diversity, which helps image quality and can hurt likelihood-based evaluation.
Shi and colleagues observed that SEDD has training instabilities, which they attribute to the score parameterization breaking the consistency between the forward and reverse processes, and that the learned schedules of GenMD4 can overfit to dataset statistics and transfer less well zero-shot. {#c-md4-instability}
Generating a sequence of unknown length needs a fixed length, padding, or the semi-autoregressive scheme of MDLM, which generates blocks one after another and conditions each on the last part of the previous one. {#c-mdlm-sar}

## Variants and descendants

- **Noise processes.** Uniform noise from multinomial diffusion, absorbing noise, nearest-neighbour and discretized Gaussian matrices for ordinal data such as pixels, and mixtures of absorbing and uniform noise.
- **BERT and autoregressive models as special cases.** Austin and colleagues note that BERT is a one-step diffusion model: a transition matrix that mixes an absorbing mask with uniform noise, applied once so that 10% of tokens become [MASK] and 5% are replaced uniformly at random, gives exactly the BERT denoising objective. {#c-bert-one-step}
  A process that masks the tokens one by one deterministically, from the end of the sequence, has a delta posterior at every position but one, and its bound is the cross-entropy loss of an autoregressive model, so autoregressive models are discrete diffusion models. {#c-ar-diffusion}
  A D3PM with absorbing noise trained on the ELBO with the x0-parameterization reduces to a reweighted version of the objective of generative masked language models, which mask $k$ tokens according to a schedule and predict them from the context. {#c-mlm-reweighted}
  The [[bert]] page describes the masked language model itself.
- **Masked diffusion language models.** MDLM, MD4 and the closely related RADD, LLaDA and later large diffusion language models, which scale the transformer denoiser of eq-subs to billions of parameters and fine-tune it to follow instructions.
- **State-dependent schedules.** GenMD4 learns a different masking rate for each token value, so that some tokens are revealed earlier than others.
- **Ratio-based models.** SEDD with uniform and absorbing transition rates of a [[m2:continuous-time-markov-chain|continuous-time Markov chain]], sampled by $\tau$-leaping or by the reverse of the chain in closed form.
- **Masked generative transformers for images and audio.** MaskGIT and its successors decode image and audio tokens of a vector-quantized autoencoder by confidence, with a [[vision-transformer]]-style backbone over the token grid.
- **Continuous relaxations.** Diffusion on token embeddings or on the probability simplex, and discrete [[flow-matching]], which defines the corruption by a probability path between noise and data instead of a transition matrix.

## Notable models

D3PM established the framework of transition matrices, the x0-parameterization and the connection to BERT and autoregressive models, and was the first discrete diffusion model with competitive text and image likelihoods.
MaskGIT made parallel masked decoding the standard for generating images from discrete tokens, with 8 to 12 steps per image, and its successors extended it to text-to-image and video.
SEDD was the first discrete diffusion language model to match GPT-2 on zero-shot perplexity.
MDLM and MD4 reduced masked diffusion to a weighted masked-language-modelling loss, made it simple to train and scale, and are the basis of the large diffusion language models that followed.

## Implementations

The NumPy reference `impl/numpy/discrete_diffusion.py` implements:

- D3PM: the uniform and absorbing matrices of eq-d3pm-forward and their products, the posterior of eq-d3pm-posterior and the reverse step of eq-x0-param, the multinomial diffusion posterior, and the exact bound of eq-vlb for one token;
- masked diffusion: the linear and cosine schedules, forward masking, the posterior of eq-mask-posterior, the SUBS reverse step of eq-subs and its KL term eq-mask-kl, the one-sample estimate of eq-mdlm, its exact value by c-mask-coefficient, and its value by Gauss-Legendre quadrature over time;
- sampling: ancestral sampling and its exact output distribution, and MaskGIT's decoding by confidence with the cosine mask schedule;
- score entropy (eq-score-entropy), its denoising form and the concrete score of the masked process;
- the denoiser with a hand-written gradient of the weighted cross-entropy, trained with Adam by the exact bound.

The PyTorch version `impl/pytorch/discrete_diffusion.py` has the denoiser as an `nn.Module` with its gradient from autograd and training with `torch.optim.Adam`; `test_gradcheck_through_the_bound` runs `torch.autograd.gradcheck` through the exact bound.
The JAX version `impl/jax/discrete_diffusion.py` is a set of pure functions of a parameter dictionary, with the gradient from `jax.grad`, the schedule weight from `jax.grad` of $\alpha_t$, training as a `jax.lax.scan` of Adam steps and ancestral sampling as a `jax.lax.fori_loop`.

All three load the shared fixture of `fixtures/make.py`, a denoiser with $L = 3$, $K = 3$ and $H = 5$, and check its logits and SUBS probabilities, forward masking under the cosine schedule, the one-sample bound and its gradient in every parameter, the per-sequence estimates, a reverse step of uniform D3PM, the exact bound of one sequence and a score entropy.

### Worked example

Take two tokens over $\{0, 1\}$ that agree with probability 0.8: $p(0,0) = p(1,1) = 0.4$ and $p(0,1) = p(1,0) = 0.1$, and the exact denoiser.
By c-mask-coefficient with $L = 2$, each of the three masks has coefficient $1/2$:

- position 1 masked, position 2 visible as 0: the denoiser gives $p(x^1 = 0 \mid x^2 = 0) = 0.8$, a loss of $-\log 0.8 = 0.2231$;
- position 2 masked: the same, 0.2231;
- both masked: each marginal is $0.5$, a loss of $2 \times 0.6931 = 1.3863$.

The bound for $\mathbf{x} = (0, 0)$ is $\frac{1}{2}(0.2231 + 0.2231 + 1.3863) = 0.9163 = -\log 0.4$, the exact negative log-likelihood, by c-any-order.
Sampling in one step draws both tokens from their marginals and gives every pair probability 0.25.
In two steps under the linear schedule each token is revealed at the first step with probability 1/2: both at once with probability 1/4, giving $(0, 0)$ with probability 1/4; one in each step with probability 1/2, giving $(0,0)$ with probability $0.5 \times 0.8 = 0.4$; and both at the last step with probability 1/4, again 1/4.
So $p_2(0, 0) = 0.0625 + 0.2 + 0.0625 = 0.325$, against 0.4 for the data and 0.25 for one step.
`test_worked_example` checks these numbers.

### Pitfalls

> [!pitfall]
> The weight of eq-mdlm diverges as $t \to 0$: it is $1/t$ for the linear schedule. Few tokens are masked at small $t$, so the product stays finite, but drawing $t$ exactly 0 divides by zero, and low-discrepancy or antithetic draws of $t$ reduce the variance that the large weights cause.

> [!pitfall]
> Forgetting the carry-over half of SUBS lets the network change visible tokens and puts loss on them; forgetting zero masking lets it predict the mask. In both cases eq-mask-kl no longer holds and the bound is no longer the weighted cross-entropy.

> [!pitfall]
> Under uniform noise the network sees a corrupted version of the token it predicts. Predicting the posterior mean of the clean token from it, as a classifier trained on $(\mathbf{x}_t, \mathbf{x}_0)$ pairs would, counts that evidence twice under eq-x0-param, by c-x0-optimum. Train on the bound, not on a plain classification loss.

> [!pitfall]
> Sampling in fewer steps than tokens draws several tokens per step independently, by c-sampler-collisions, and the samples lose the dependencies between those tokens even with a perfect network; report the number of steps with any sample-quality metric.

> [!pitfall]
> Confidence-based decoding and low-temperature sampling raise sample quality metrics such as generative perplexity while losing diversity; compare samplers at matched entropy, or report both.

## Experiments

This page has no experiment yet.
Some tests are small experiments.

`test_trained_denoiser_reaches_the_entropy_and_samples_the_data`, described in the Training section, trains a denoiser end to end by the exact bound, then computes the exact output distribution of its sampler: the total variation distance from the data is above 0.6 in one step and below 0.06 in 16. The PyTorch and JAX versions run the same training, and the JAX version samples 20,000 sequences with its `fori_loop` sampler.

`test_ancestral_sampler_distribution` measures the error of few-step sampling exactly, without sampling noise, for 1 to 32 steps.

`test_maskgit_decoding` checks MaskGIT's mask counts and that a token, once kept, is never changed, and measures the bias of confidence-based decoding towards likely sequences.

The checks of the eight derivations compare:

- the uniform D3PM posterior with the multinomial diffusion posterior;
- the D3PM bound of a single token under eq-x0-param with the entropy of the data, for the data distribution and the posterior mean as predictions;
- the integral of eq-mdlm under several schedules with the closed-form coefficients;
- the exact bound with the exact denoiser with $-\log p(\mathbf{x})$;
- the error of the ancestral sampler with the probability that two tokens are revealed in the same step;
- the ratios of masked marginals with the denoiser;
- score entropy and its denoising form at and away from the true ratios;
- the parameter and FLOP counts with the code.

## History

Sohl-Dickstein and colleagues' diffusion probabilistic models of 2015 included a binomial diffusion for binary data alongside the Gaussian case.
After Ho and colleagues' denoising diffusion of 2020 revived Gaussian diffusion for images, Hoogeboom and colleagues' multinomial diffusion of 2021 corrupted categorical data by uniform resampling, and Austin and colleagues' D3PM of the same year allowed any transition matrix and found absorbing noise best for text.
Masked generative models had developed in parallel, from BERT's masked language modelling of 2018 and Ghazvininejad and colleagues' Mask-Predict for translation of 2019 to MaskGIT for images in 2022, which decoded by iterative unmasking.
Campbell and colleagues put discrete diffusion in continuous time in 2022, and Lou, Meng and Ermon's SEDD of 2023 learned ratios of probabilities with score entropy and matched GPT-2.
In 2024 MDLM and MD4, and Ou and colleagues' RADD, reduced masked diffusion to a weighted masked-language-modelling loss with a time-independent denoiser, and large diffusion language models trained this way followed in 2025.
Discrete diffusion relates to the other generative families: it is a [[diffusion-model]] on a finite state space, an any-order [[autoregressive-model]], a latent-variable model trained by the same kind of bound as a [[variational-autoencoder]], and a model of probability ratios like an [[energy-based-model]]; unlike a [[normalizing-flow]], it needs no invertible map, which does not exist on a finite set.

## References

- J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan and S. Ganguli. Deep Unsupervised Learning using Nonequilibrium Thermodynamics. ICML 2015.
- E. Hoogeboom, D. Nielsen, P. Jaini, P. Forré and M. Welling. Argmax Flows and Multinomial Diffusion: Learning Categorical Distributions. NeurIPS 2021.
- J. Austin, D. D. Johnson, J. Ho, D. Tarlow and R. van den Berg. Structured Denoising Diffusion Models in Discrete State-Spaces. NeurIPS 2021.
- H. Chang, H. Zhang, L. Jiang, C. Liu and W. T. Freeman. MaskGIT: Masked Generative Image Transformer. CVPR 2022; arXiv 2202.04200.
- A. Campbell, J. Benton, V. De Bortoli, T. Rainforth, G. Deligiannidis and A. Doucet. A Continuous Time Framework for Discrete Denoising Models. NeurIPS 2022.
- A. Lou, C. Meng and S. Ermon. Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution. ICML 2024.
- S. S. Sahoo, M. Arriola, Y. Schiff, A. Gokaslan, E. Marroquin, J. T. Chiu, A. Rush and V. Kuleshov. Simple and Effective Masked Diffusion Language Models. NeurIPS 2024.
- J. Shi, K. Han, Z. Wang, A. Doucet and M. K. Titsias. Simplified and Generalized Masked Diffusion for Discrete Data. NeurIPS 2024.
