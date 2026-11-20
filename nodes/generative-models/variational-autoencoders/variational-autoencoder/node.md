---
tag: NEW-variational-autoencoder
kind: architecture
slug: variational-autoencoder
name: Variational autoencoder
aliases: [VAE, variational auto-encoder, auto-encoding variational Bayes, AEVB, SGVB, deep latent Gaussian model, DLGM, importance weighted autoencoder, IWAE, beta-VAE]
area: generative-models/variational-autoencoders
level: core
summary: >-
  A latent-variable generative model whose encoder outputs a distribution over codes; encoder and decoder maximize a reconstruction term minus a KL term to the prior, a bound on the likelihood.
uses: [kullback-leibler-divergence@m2, multivariate-normal-distribution@m2, linearity-of-expectation@m2, bayes-theorem@m2, cross-entropy@m2, matrix-multiplication@m2, evidence-lower-bound@al, variational-inference@al]
related_to: [autoencoder, multilayer-perceptron, long-short-term-memory, dropout, adam]
history:
  year: 2013
  people: [Diederik P. Kingma, Max Welling, Danilo J. Rezende, Shakir Mohamed, Daan Wierstra]
  note: "Kingma and Welling posted Auto-Encoding Variational Bayes as arXiv 1312.6114 in December 2013, and Rezende, Mohamed and Wierstra posted the deep latent Gaussian models with stochastic backpropagation as arXiv 1401.4082 a month later, at ICML 2014; each paper notes that the two were developed independently. Burda et al. (2015) tightened the bound with importance weights, Bowman et al. (2015) met posterior collapse with recurrent decoders, and Alemi et al. (2017) read the two terms of the bound as a rate and a distortion."
prereqs: [autoencoder, multilayer-perceptron, kullback-leibler-divergence@m2, multivariate-normal-distribution@m2, evidence-lower-bound@al, backpropagation@al]
same_as:
  wikipedia: Variational_autoencoder
sources:
  - {paper: "Kingma and Welling (2013)", title: "Auto-Encoding Variational Bayes", locator: "arXiv 1312.6114v11, pp. 1 to 8, 10 and 11", note: "the bound, the reparameterization trick, the estimators A and B, the Gaussian and Bernoulli MLPs, the experiments against wake-sleep and appendix B"}
  - {paper: "Rezende, Mohamed and Wierstra (2014)", title: "Stochastic Backpropagation and Approximate Inference in Deep Generative Models", locator: "ICML 2014; arXiv 1401.4082v3, pp. 1 to 4 and 6", note: "deep latent Gaussian models, stochastic backpropagation, the free energy and Table 1 on binarized MNIST"}
  - {paper: "Burda, Grosse and Salakhutdinov (2015)", title: "Importance Weighted Autoencoders", locator: "arXiv 1509.00519v4, pp. 1, 3, 4, 6 to 8", note: "the k-sample bound, Theorem 1, the active-unit statistic and Tables 1 and 2"}
  - {paper: "Bowman, Vilnis, Vinyals, Dai, Jozefowicz and Bengio (2015)", title: "Generating Sentences from a Continuous Space", locator: "arXiv 1511.06349v4, pp. 1, 3 to 5", note: "posterior collapse with an LSTM decoder, KL cost annealing, word dropout and the Penn Treebank results"}
  - {paper: "Alemi, Poole, Fischer, Dillon, Saurous and Murphy (2017)", title: "Fixing a Broken ELBO", locator: "arXiv 1711.00464v3, pp. 1 to 5", note: "rate and distortion, H - D <= I <= R, the ELBO as -(D + R), the beta-VAE objective and the auto-decoding limit"}
family: variational-autoencoder
inputs: "a batch of inputs as a (B, D) array, and S draws of standard normal noise per example as an (S, B, J) array"
outputs: "the mean and the log-variance of q(z|x), each a (B, J) array, the S codes, and the parameters of p(x|z) for each code: logits for a Bernoulli decoder, a mean and a log-variance for a Gaussian one"
components: []
inductive_biases: ["the data are generated from a few continuous latent variables through a smooth map", "the prior is a standard normal, the same for every example", "given the code, the components of the input are independent", "the posterior of every example is close to a Gaussian with a diagonal covariance, whose mean and variance one network can predict from the input"]
params: "(D + 1)H + 2(H + 1)J + (J + 1)H + (H + 1)D for one hidden layer of H units on each side, J latent dimensions and a Bernoulli decoder on D inputs; a Gaussian decoder adds (H + 1)D"
flops_per_example: "2DH + 4HJ + S(2JH + 2HD) for one encoding and S decodings, counting a multiply-add as 2 FLOPs and only the matrix products"
---

A variational autoencoder (VAE) is a generative model with continuous latent variables, together with the way it is trained.
A decoder network defines the model: a code $\mathbf{z}$ is drawn from a fixed prior, usually the standard normal, and the decoder turns it into the parameters of a distribution over the data.
An encoder network, the recognition model, maps an input $\mathbf{x}$ to a distribution $q(\mathbf{z} \mid \mathbf{x})$ over the codes that could have produced it, and both networks are trained together on a lower bound on $\log p(\mathbf{x})$.
Kingma and Welling put the method in two parts: a reparameterization of the variational lower bound gives an estimator that can be optimized with standard stochastic gradient methods, and an approximate inference model fitted to the intractable posterior makes inference efficient; when a neural network is used for that recognition model, the result is the variational auto-encoder. {#c-kw-def}
It looks like an [[autoencoder]] with noise in the middle, and the bound is a reconstruction error plus a penalty that keeps the codes close to the prior; unlike a plain autoencoder it is a generative model, because decoding a draw from the prior gives a sample of the data.

## In plain words

Suppose every handwritten digit were made by choosing a few hidden numbers, how slanted, how thick, which digit, and then drawing from them.
A VAE learns both directions of that story at once.
The decoder learns to draw from the hidden numbers; the encoder learns to guess, from a finished drawing, which hidden numbers probably made it, and it answers with a range of plausible values, a mean and a spread, not a single guess.
Training picks a value from that range, asks the decoder to redraw the input from it, and scores two things: how well the redrawing matches, and how far the encoder's range strays from the range the hidden numbers are supposed to have in general.
The second score is what makes the model generative: because every input's range sits inside the common one, any point picked from the common range decodes to something that looks like data.

## Overview

The model has a prior $p(\mathbf{z})$, a decoder $p_\theta(\mathbf{x} \mid \mathbf{z})$ and an encoder $q_\phi(\mathbf{z} \mid \mathbf{x})$.
Kingma and Welling call $q_\phi(\mathbf{z} \mid \mathbf{x})$ a probabilistic encoder, because given a datapoint it produces a distribution over the possible values of the code from which the datapoint could have been generated, and $p_\theta(\mathbf{x} \mid \mathbf{z})$ a probabilistic decoder, because given a code it produces a distribution over the possible corresponding values of $\mathbf{x}$. {#c-kw-encoder-decoder}
The marginal likelihood $p_\theta(\mathbf{x}) = \int p(\mathbf{z})\, p_\theta(\mathbf{x} \mid \mathbf{z})\, d\mathbf{z}$ and the posterior $p_\theta(\mathbf{z} \mid \mathbf{x})$ are intractable once the decoder is a neural network.
Burda et al. describe the VAE as pairing a generative network, which runs from the code to the data, with a recognition network, which runs from the data to the code, both trained jointly to maximize a variational lower bound on the data log-likelihood. {#c-iwae-def}

The method is [[al:variational-inference|variational inference]] with two changes.
The approximate posterior of every datapoint is computed by one network from the datapoint itself, so inference on a new input is a single forward pass, not an optimization; this is amortized inference.
And the expectation in the bound is rewritten as an expectation over noise that does not depend on the parameters, so that one sample gives an unbiased, differentiable estimate of the bound and of its gradient; this is the reparameterization trick.
Rezende, Mohamed and Wierstra arrived at the same model, which they call a deep latent Gaussian model, and the same gradient, which they call stochastic backpropagation, at the same time.

## Components

The notation has vectors as rows; a layer maps $\mathbf{h}$ to $\mathbf{h} W + \mathbf{b}$ with $W$ of shape $(n_\text{in}, n_\text{out})$.
The model of the shape contract is the one of Kingma and Welling's experiments: a standard normal prior, an encoder and a decoder of one tanh hidden layer each.
They take the prior over the latent variables to be the centred isotropic Gaussian N(z; 0, I), which has no parameters, let p(x|z) be a multivariate Gaussian for real-valued data or a Bernoulli for binary data, with its parameters computed from z by an MLP with one hidden layer, and let the approximate posterior be a multivariate Gaussian with a diagonal covariance whose mean and standard deviation are outputs of the encoding MLP. {#c-kw-model}
They add that the diagonal Gaussian is a simplifying choice, not a limitation of the method. {#c-kw-diagonal}

**The encoder.** A hidden layer of $H$ units and two heads of $J$ units, the mean and the log-variance of a diagonal Gaussian:

$$ \mathbf{h} = \tanh(\mathbf{x} W_1 + \mathbf{b}_1), \qquad \boldsymbol\mu = \mathbf{h} W_\mu + \mathbf{b}_\mu, \qquad \log\boldsymbol\sigma^2 = \mathbf{h} W_v + \mathbf{b}_v, \qquad q_\phi(\mathbf{z} \mid \mathbf{x}) = \mathcal{N}\big(\boldsymbol\mu, \operatorname{diag}\boldsymbol\sigma^2\big) . $$ {#eq-encoder}

The log-variance is the output, not the variance, so that any real output gives a positive variance.

**The reparameterization.** A code is drawn as a deterministic function of the encoder's outputs and of noise that does not depend on any parameter:

$$ \mathbf{z} = \boldsymbol\mu + \boldsymbol\sigma \odot \boldsymbol\varepsilon, \qquad \boldsymbol\varepsilon \sim \mathcal{N}(0, I) . $$ {#eq-reparam}

For a fixed $\boldsymbol\varepsilon$ the code, and everything computed from it, is a differentiable function of $\phi$.
Kingma and Welling list three ways to find such a transformation for other families: a tractable inverse CDF applied to uniform noise, the location and scale of any location-scale family applied to its standard member, as for the Gaussian, and compositions of such transformations, as for the log-normal, gamma, Dirichlet and beta distributions. {#c-kw-reparam-families}

**The decoder.** A hidden layer of $H$ units and an output of $D$ units, applied to each code:

$$ \mathbf{g} = \tanh(\mathbf{z} V_1 + \mathbf{c}_1), \qquad \mathbf{a} = \mathbf{g} V_2 + \mathbf{c}_2, \qquad \log p_\theta(\mathbf{x} \mid \mathbf{z}) = \sum_{d=1}^{D} \big[x_d a_d - \operatorname{softplus}(a_d)\big] , $$ {#eq-decoder}

the log-likelihood of independent Bernoulli pixels with logits $\mathbf{a}$, the negative of the cross-entropy computed from the logits.
In the paper the Bernoulli decoder computes the probabilities as y = sigmoid(W2 tanh(W1 z + b1) + b2) and scores x with sum_i x_i log y_i + (1 - x_i) log(1 - y_i); the Gaussian MLP, used as the encoder and as the decoder for real-valued data, has one tanh layer and two affine heads, the mean and the log-variance. {#c-kw-mlps}
A Gaussian decoder adds the second head, $\log\boldsymbol\sigma_x^2 = \mathbf{g} V_3 + \mathbf{c}_3$, and scores $\mathbf{x}$ with the density of $\mathcal{N}(\mathbf{a}, \operatorname{diag}\boldsymbol\sigma_x^2)$; for the Frey Face data Kingma and Welling constrained its means to (0, 1) with a sigmoid. {#c-kw-frey-sigmoid}

**The KL term.** With a standard normal prior and a diagonal Gaussian encoder the divergence from the prior has a closed form,

$$ \mathrm{KL}\big(q_\phi(\mathbf{z} \mid \mathbf{x}) \,\Vert\, p(\mathbf{z})\big) = \frac12 \sum_{j=1}^{J} \big(\mu_j^2 + \sigma_j^2 - 1 - \log\sigma_j^2\big) , $$ {#eq-kl}

which is zero exactly when $\boldsymbol\mu = 0$ and $\boldsymbol\sigma = 1$, the prior. {#c-kl}
Its gradient is $\mu_j$ in $\mu_j$ and $\frac12(\sigma_j^2 - 1)$ in $\log\sigma_j^2$, which stays finite as $\sigma_j \to 0$: the term pulls each mean to 0 and each variance to 1. {#c-kl-grad}
With a full covariance $C$ the same divergence is $\frac12(\operatorname{tr} C - \log\lvert C \rvert + \boldsymbol\mu^\top\boldsymbol\mu - J)$, the form Rezende et al. use. {#c-kl-full}
The derivation `gaussian-kl` proves the three statements.

**The bound.** For any encoder,

$$ \log p_\theta(\mathbf{x}) = \mathcal{L}(\mathbf{x}) + \mathrm{KL}\big(q_\phi(\mathbf{z} \mid \mathbf{x}) \,\Vert\, p_\theta(\mathbf{z} \mid \mathbf{x})\big), \qquad \mathcal{L}(\mathbf{x}) = \mathbb{E}_{q_\phi(\mathbf{z} \mid \mathbf{x})}\big[\log p_\theta(\mathbf{x}, \mathbf{z}) - \log q_\phi(\mathbf{z} \mid \mathbf{x})\big] , $$ {#eq-elbo}

so $\mathcal{L}(\mathbf{x}) \le \log p_\theta(\mathbf{x})$, with equality exactly when the encoder's distribution is the true posterior. {#c-elbo}
This is the [[al:evidence-lower-bound|evidence lower bound]], ELBO, eq. 1 of Kingma and Welling.
Splitting $\log p_\theta(\mathbf{x}, \mathbf{z}) = \log p_\theta(\mathbf{x} \mid \mathbf{z}) + \log p(\mathbf{z})$ gives

$$ \mathcal{L}(\mathbf{x}) = \mathbb{E}_{q_\phi(\mathbf{z} \mid \mathbf{x})}\big[\log p_\theta(\mathbf{x} \mid \mathbf{z})\big] - \mathrm{KL}\big(q_\phi(\mathbf{z} \mid \mathbf{x}) \,\Vert\, p(\mathbf{z})\big) , $$

an expected reconstruction log-likelihood minus the KL term of eq-kl. {#c-elbo-split}
Kingma and Welling interpret the KL term as regularizing $\phi$, encouraging the approximate posterior to be close to the prior, and the first term as an expected negative reconstruction error, which is the connection with autoencoders. {#c-kw-regularizer}
For a linear Gaussian decoder every term has a closed form, the posterior is Gaussian, and an encoder that outputs its mean and covariance makes the bound exact. {#c-elbo-linear}
The derivation `elbo-identity` proves these statements.

**The estimators.** With $S$ draws $\boldsymbol\varepsilon^{(s)}$ and $\mathbf{z}^{(s)} = \boldsymbol\mu + \boldsymbol\sigma \odot \boldsymbol\varepsilon^{(s)}$ the bound has two unbiased estimates, Kingma and Welling's estimator A,

$$ \tilde{\mathcal{L}}^{A}(\mathbf{x}) = \frac1S \sum_{s=1}^{S} \big[\log p_\theta(\mathbf{x}, \mathbf{z}^{(s)}) - \log q_\phi(\mathbf{z}^{(s)} \mid \mathbf{x})\big] , $$ {#eq-sgvb-a}

and estimator B, which takes the KL term from eq-kl and samples only the reconstruction,

$$ \tilde{\mathcal{L}}^{B}(\mathbf{x}) = -\mathrm{KL}\big(q_\phi(\mathbf{z} \mid \mathbf{x}) \,\Vert\, p(\mathbf{z})\big) + \frac1S \sum_{s=1}^{S} \log p_\theta(\mathbf{x} \mid \mathbf{z}^{(s)}) . $$ {#eq-sgvb-b}

Estimator B typically has less variance than the generic estimator A. {#c-kw-estimator-b}
Both are differentiable in $\theta$ and $\phi$ for fixed noise: the gradient of $\mathbb{E}_q[f(\mathbf{z})]$ in $\mu$ and $\sigma$ is the expectation over $\boldsymbol\varepsilon$ of the gradient of $f(\boldsymbol\mu + \boldsymbol\sigma \odot \boldsymbol\varepsilon)$. {#c-reparam}
The alternative is the score-function estimator of [[al:reinforce|REINFORCE]], $f(\mathbf{z})\, \nabla \log q(\mathbf{z})$, which is also unbiased; Kingma and Welling note that it exhibits very high variance and is impractical for their purposes. {#c-kw-naive-variance}
For $f(z) = z^2$ in one dimension, one draw of the reparameterization estimate of the gradient in $\mu$ has variance $4\sigma^2$, and one draw of the score-function estimate $\mu^4/\sigma^2 + 14\mu^2 + 15\sigma^2$: 4 against 15 at $\mu = 0$, $\sigma = 1$, and the second grows without bound as $\sigma \to 0$. {#c-reparam-variance}
The derivation `reparameterization-gradient` proves both statements.

**The importance weighted bound.** With $k$ draws $\mathbf{z}_1, \dots, \mathbf{z}_k$ from the encoder and the weights $w_i = p_\theta(\mathbf{x}, \mathbf{z}_i) / q_\phi(\mathbf{z}_i \mid \mathbf{x})$, Burda et al. train on

$$ \mathcal{L}_k(\mathbf{x}) = \mathbb{E}\Big[\log \frac1k \sum_{i=1}^{k} w_i\Big] , $$ {#eq-iwae}

which is a lower bound on $\log p_\theta(\mathbf{x})$ for every $k$ and equals the ELBO at $k = 1$. {#c-iwae}
Their Theorem 1 states that log p(x) >= L_{k+1} >= L_k for all k, and that L_k approaches log p(x) as k goes to infinity if p(h, x)/q(h|x) is bounded. {#c-iwae-theorem}
The derivation `iwae-bound` proves the same order, $\mathcal{L}_k \le \mathcal{L}_{k+1} \le \log p_\theta(\mathbf{x})$, and the limit. {#c-iwae-monotone}
The gradient of one estimate is $\sum_i \tilde{w}_i \nabla \log w_i$ with the normalized weights $\tilde{w}_i = w_i / \sum_j w_j$: each draw's ELBO gradient, weighted by how well that draw explains $\mathbf{x}$; with $k = 1$ it is the gradient of the ELBO. {#c-iwae-grad}
The encoder and the decoder are those of the VAE; only the objective changes.

## Shapes and parameters

The shape contract has one hidden layer of $H$ units on each side, a code of $J$ dimensions and an input of $D$; the example is $D = 6$, $H = 5$, $J = 2$.
The encoder's layer has $DH + H$ weights and biases and each of its heads $HJ + J$; the decoder's layer has $JH + H$ and its output $HD + D$, so

$$ P = (D + 1)H + 2(H + 1)J + (J + 1)H + (H + 1)D $$ {#eq-params}

for a Bernoulli decoder, and a Gaussian decoder adds a second output head of $(H + 1)D$. {#c-params}
In the shape contract this is

{shape:params}

and the example has {shape:params@example} parameters, {shape:params_gaussian@example} with a Gaussian decoder.
The MNIST model of Kingma and Welling with 500 hidden units and 20 latent dimensions has 815,824 parameters, and the Frey Face model with 200 hidden units, 20 latent dimensions and a Gaussian decoder has 349,560. {#c-params-models}
The configs `kw13_mnist` and `kw13_frey_face` have {shape:params@kw13_mnist} and {shape:params_gaussian@kw13_frey_face}.
The prior has no parameters, and neither does the noise.
The derivation `parameter-and-flop-count` proves the counts.

## Computation and cost

The encoder runs once per example and the decoder once per draw of the code.
A forward pass over $B$ examples with $S$ draws each makes, in its products,

$$ C = 2BDH + 4BHJ + S\,(2BJH + 2BHD) $$ {#eq-flops}

FLOPs, a multiply-add counted as two, and $2SBHD$ more with a Gaussian decoder; the cost of the importance weighted bound with $k = S$ samples is linear in $k$. {#c-flops}
In the contract,

{shape:flops_forward}

which is {shape:flops_forward@example} FLOPs for the example on a batch of 4 with 3 draws.
For the MNIST model with one draw this is 1,628,000 FLOPs per example, 824,000 in the encoder and 804,000 in the decoder, and each further draw adds 804,000. {#c-flops-models}
The config `kw13_mnist`, a minibatch of 100 with one draw, costs {shape:flops_forward@kw13_mnist} FLOPs.
The KL term, the reparameterization, the activations and the loss are linear in $BJ$, $SBJ$, $SBH$ and $SBD$ and are left out.
Sampling from the trained model is a decoder pass alone, half of the cost or less, and encoding a new input is an encoder pass, with no iterative inference.

## Training

Every step draws a minibatch, draws the noise, computes the codes by eq-reparam, evaluates the estimate of the bound and takes a gradient step on $\theta$ and $\phi$ together by [[al:backpropagation|backpropagation]]; Kingma and Welling's Algorithm 1 is this loop with estimator B.
The number of draws can be small: Kingma and Welling found that the number of samples L per datapoint can be set to 1 as long as the minibatch size M is large enough, for example M = 100. {#c-kw-one-sample}
In their experiments all parameters were initialized from N(0, 0.01) and optimized jointly with Adagrad, its global stepsize chosen from {0.01, 0.02, 0.1} on the training set in the first iterations, with minibatches of M = 100, L = 1 sample per datapoint and a small weight decay corresponding to a prior N(0, I) on the parameters; the encoders and decoders had 500 hidden units for MNIST and 200 for the smaller Frey Face data set, to prevent overfitting. {#c-kw-training}
Later work usually uses [[adam|Adam]].

### Posterior collapse and KL annealing

The KL term has a cheap minimum: an encoder that ignores its input and outputs the prior pays nothing for it, and if the decoder can model the data without the code, the bound is then as good as the decoder alone.
Bowman et al. met this with a recurrent decoder, an [[long-short-term-memory|LSTM]] language model over the words of a sentence: except in vanishingly rare cases, most training runs with most hyperparameters gave models that set q(z|x) equal to the prior, bringing the KL term to zero, so that the model behaved as an ordinary recurrent language model. {#c-bowman-collapse}
They proposed two remedies.
KL cost annealing puts a weight on the KL term that is zero at the start of training, so that the model encodes as much information in z as it can, and increases it gradually to 1, at which point the cost is the true variational lower bound again; they describe it as annealing from a vanilla autoencoder to a VAE. {#c-bowman-annealing}
Word dropout weakens the decoder: a fraction of the previous words it is conditioned on is replaced by the unknown-word token, which forces the model to rely on z, with a keep rate k that at k = 0 leaves the decoder no input at all; standard [[dropout]] on the decoder's word embeddings did not help the model use the latent variable. {#c-bowman-word-dropout}
The reference's `train(..., anneal=n)` implements the first remedy, a weight that rises linearly from 0 to 1 over $n$ steps.

### Rate and distortion

Alemi et al. read the two terms of the bound as a distortion $D$, the expected reconstruction negative log-likelihood, and a rate $R$, the expected KL divergence from the encoder's distribution to the marginal over codes, and prove that $H - D \le I(X; Z) \le R$ for the entropy $H$ of the data and the mutual information between data and code under the encoder. {#c-alemi-bounds}
The ELBO is $-(D + R)$, so it cannot tell apart models on the line $D + R = \text{const}$; in the limit of unrestricted model families the ELBO alone, and the marginal likelihood, cannot distinguish models that make no use of the latent variable, autodecoders, from models that use it heavily, autoencoders. {#c-alemi-elbo}
With a weight $\beta \ge 0$ on the rate the objective is that of the beta-VAE, which moves smoothly from auto-encoding behaviour at small $\beta$, low distortion and high rate, to auto-decoding behaviour at large $\beta$, high distortion and low rate, without a change of architecture. {#c-alemi-beta}
`loss(..., beta=b)` of the reference minimizes $D + \beta R$ with the prior in place of the marginal.

## Inductive biases

The model assumes that the data are generated from a few continuous causes through a smooth map, the decoder, and that the causes are independent and standard normal a priori.
Given the code the components of the input are independent, with a Bernoulli or Gaussian noise model; whatever correlation the data have must be carried by the code, which is why a weak decoder forces the code to be used and a powerful autoregressive one lets it be ignored.
The diagonal Gaussian encoder assumes that every posterior is close to a Gaussian whose coordinates are independent, and the amortization assumes that one network can predict its mean and variance from the input; Burda et al. note that the VAE objective encourages representations in which these assumptions hold. {#c-iwae-assumptions}
The KL term prefers to switch off a latent dimension entirely, making its posterior the prior, over using it a little, so a VAE uses fewer dimensions than it is given.

## Evidence

Kingma and Welling compared their method, AEVB, with the wake-sleep algorithm on MNIST and Frey Face, with the same encoder for both, for latent spaces of 3 to 200 dimensions.
Their method converged considerably faster and reached a better value of the lower bound in all experiments, and more latent variables did not result in more overfitting, which they explain by the regularizing effect of the bound. {#c-kw-wake-sleep}
Rezende et al. report a test negative log-likelihood on binarized MNIST of 87.30 nats for a deep latent Gaussian model with a diagonal covariance and 86.60 for a rank-one covariance, against 106.00 for factor analysis, 95.80 for a nonlinear Gaussian belief network and 91.3 for wake-sleep. {#c-rmw-mnist}

Burda et al. measure how many latent dimensions a model uses with the statistic A_u = Cov_x(E_{u ~ q(u|x)}[u]), calling a dimension active if A_u > 10^-2; removing the inactive dimensions changed the test log-likelihood by less than 0.06 nats in all cases. {#c-iwae-active}
On MNIST with one stochastic layer of 50 units the VAE reached a test negative log-likelihood of 86.76, 86.47 and 86.35 nats with k = 1, 5 and 50 samples, and the IWAE 86.76, 85.54 and 84.78; the active units were 19, 20 and 20 for the VAE against 19, 22 and 25 for the IWAE; with two stochastic layers the VAE reached 85.33, 85.01 and 84.78 and the IWAE 85.33, 83.89 and 82.90. {#c-iwae-results}
In all their conditions the number of active dimensions was far less than the total, and adding latent dimensions did not increase it; continuing to train a VAE with the IWAE objective increased the number of active dimensions and the log-likelihood, and continuing to train an IWAE with the VAE objective did the opposite. {#c-iwae-inactive}

Bowman et al. trained a VAE with an LSTM encoder and decoder on the Penn Treebank and compared it with a recurrent language model, the VAE's figure being a lower bound.
In the standard setting the VAE was slightly worse, a test negative log-likelihood of 101 nats with a KL term of 2 against 100 for the language model; with an inputless decoder, word dropout keep rate 0, it reached 125 with a KL term of 15 against 135 for the equally weakened language model. {#c-bowman-ptb}
Without both word dropout and KL annealing, training reliably gave models equivalent to the language model with a KL term of zero. {#c-bowman-zero-kl}
Alemi et al. trained 209 runs of 12 architectures on MNIST at values of beta from 0.1 to 10, and found that as the decoder grows more powerful even beta = 1 suffices to make the model collapse to the auto-decoding limit, an example of a VAE ignoring its latent space. {#c-alemi-collapse}
These numbers are log-likelihoods, or bounds on them, on small binary images and one text corpus; they compare objectives and inference methods, not sample quality, on which VAEs have generally been judged below adversarial and diffusion models.

## Limitations

The bound is not the likelihood: the gap is the KL divergence from the encoder's distribution to the true posterior, and a diagonal Gaussian encoder pays it wherever the posterior is skewed, multimodal or correlated.
Amortization adds a second gap, since one network has to fit every example's posterior; the two are called the approximation gap and the amortization gap.
With a factorized Gaussian or Bernoulli decoder the samples are blurry: the decoder's mean averages over what the code does not determine, and the noise of a Gaussian decoder is usually left out of the samples people look at.
With a powerful decoder the code is ignored, the posterior collapse of the section on training, and with the KL term weighted down to keep the code in use the model is no longer trained on a bound.
The ELBO alone does not say how much the code is used, as Alemi et al. prove, so two models with the same ELBO can have very different representations.
The reparameterization trick needs continuous latent variables; for discrete ones the estimators are the score function, with its variance, or continuous relaxations.

## Variants and descendants

- **Deep latent Gaussian models** (Rezende et al., 2014) stack several layers of Gaussian latent variables; **ladder VAEs**, **NVAE** and **very deep VAEs** are later hierarchical models.
- **Importance weighted autoencoders** (Burda et al., 2015) train the same networks on eq-iwae.
- **Conditional VAEs** condition the encoder, the decoder and the prior on a label or another input, and the semi-supervised models of Kingma et al. (2014) treat the label as a partly observed latent variable.
- **beta-VAE** (Higgins et al., 2017) weights the KL term by beta > 1 for more disentangled codes.
- **Normalizing flows** in the encoder replace the diagonal Gaussian with a learned invertible transformation of it, and **VampPrior** and other learned priors replace the standard normal.
- **VAEs with autoregressive decoders**, PixelVAE and the sentence VAE of Bowman et al., put a recurrent or masked convolutional decoder behind the code, and must then be kept from ignoring it.
- **VQ-VAE** replaces the Gaussian code by a discrete one from a learned codebook, trained with a straight-through gradient and an autoregressive prior fitted afterwards.
- **Latent diffusion** models train an autoencoder with a small KL term, or a VQ code, and fit a diffusion model in its latent space; their image autoencoders are VAEs in this sense.

## Notable models

The MNIST and Frey Face models of Kingma and Welling (2013), whose two-dimensional latent spaces decode to smooth manifolds of digits and faces, are the standard first examples.
The sentence VAE of Bowman et al. (2015) generated sentences from the prior and interpolated between sentences through the latent space.
NVAE (2020) and very deep VAEs (2020) are hierarchical VAEs with dozens of latent layers trained on natural images.
The KL-regularized autoencoder of Stable Diffusion (2022) compresses each 8 by 8 patch of an image to a few latent channels on which the diffusion model works.

## Implementations

The NumPy reference `impl/numpy/vae.py` implements eq-encoder, eq-reparam and eq-decoder with a Bernoulli or a Gaussian decoder; the KL term eq-kl, the log-densities of the prior and the encoder, and the log-weights; the three objectives `elbo` (eq-sgvb-b), `elbo_mc` (eq-sgvb-a) and `iwae` (eq-iwae), all with a weight beta on the KL term, and their backward pass written out by hand; training with Adam and KL annealing; the importance weighted bound with fresh draws, the rate and distortion, the active-unit statistic of Burda et al. and sampling from the prior; and the counts eq-params and eq-flops.
The PyTorch version `impl/pytorch/vae.py` is an `nn.Module` with the same parameters; `test_densities_and_kl_match_torch_distributions` checks the KL term, the log-densities and the Bernoulli likelihood against `torch.distributions`, and `test_gradcheck` runs `torch.autograd.gradcheck` on every objective with both decoders.
The JAX version `impl/jax/vae.py` is a pure function of a parameter dictionary and a key; its tests compare `jax.value_and_grad` with the shared fixture and check that the loss is the same under `jax.jit` and that `jax.vmap` over keys averages it.
All three load the shared fixture of `fixtures/`: the example config with a Gaussian decoder's parameters and nonzero biases, a batch of 4 binary inputs, 3 draws of noise per example, and for each decoder the encoder's outputs, the codes, the logits, the KL term, the log-weights and the loss and gradients of the three objectives.

### Worked example

Take an encoder output with $\boldsymbol\mu = (1, 0)$ and $\boldsymbol\sigma^2 = (1, \tfrac14)$.
By eq-kl the KL term is $\frac12\big[(1 + 1 - 1 - 0) + (0 + \tfrac14 - 1 - \log\tfrac14)\big] = \frac12(1 + 0.6363) = 0.8181$ nats: the first coordinate pays for its mean, the second for its narrow variance.
With the noise $\boldsymbol\varepsilon = (\tfrac12, -2)$, eq-reparam gives $\mathbf{z} = (1 + 1 \cdot \tfrac12,\; 0 + \tfrac12 \cdot (-2)) = (1.5, -1.0)$.
The single-draw estimate of the same divergence, $\log q(\mathbf{z} \mid \mathbf{x}) - \log p(\mathbf{z})$, is $-\frac12(0.25 + 4) - \frac12\log\tfrac14 + \frac12(2.25 + 1) = 0.1931$, the log-normalizers cancelling; its average over many draws is 0.8181.
Estimator A would use the 0.1931 of this draw, estimator B the exact 0.8181.
The test `test_worked_example` checks these numbers.

### Pitfalls

> [!pitfall]
> **The variance as the encoder's output.** An encoder that outputs $\sigma$ or $\sigma^2$ directly needs a positivity constraint and divides by numbers near zero in $\log q$; one that outputs $\log\sigma^2$ has neither problem, and the gradient of eq-kl in $\log\sigma^2$ stays bounded. The reference outputs $\log\sigma^2$ and draws $\mathbf{z} = \boldsymbol\mu + e^{\log\sigma^2 / 2}\boldsymbol\varepsilon$; `test_gradients_match_finite_differences` checks the gradient through that exponential for every objective.

> [!pitfall]
> **Sum over the dimensions, average over the examples.** The reconstruction term and the KL term are sums over the $D$ pixels and the $J$ latent dimensions; averaging the reconstruction over pixels, as a mean-reduced cross-entropy does, weights the KL term $D$ times too much and the code goes unused. The losses of the three implementations sum over dimensions and average over the batch, and `test_shared_fixture` and its PyTorch and JAX counterparts check them against one fixture.

> [!pitfall]
> **The averaging in the importance weighted bound.** $\mathcal{L}_k$ is the log of the mean of the weights, not the mean of their logs, which would be eq-sgvb-a, and the weights must be combined in the log domain: $\log w$ is a sum over hundreds of pixels and its exponential overflows. The reference computes a log-mean-exp after subtracting the maximum; `test_logmeanexp_is_stable` checks it at log-weights of ±1000, and `test_one_sample_iwae_is_the_sampled_elbo` checks that $k = 1$ gives eq-sgvb-a.

> [!pitfall]
> **Comparing estimator A with estimator B.** The two estimate the same bound with different variances, so a lower loss of one in a run says nothing about the model. `test_both_estimators_have_the_same_mean` checks that they agree in mean over many draws and that B has the lower variance.

> [!pitfall]
> **Reporting the training objective as the likelihood.** The ELBO is a lower bound, so comparing a VAE's ELBO with another model's exact log-likelihood penalizes the VAE, and comparing it with a bound computed with more samples flatters the second. Report $\mathcal{L}_k$ with a large $k$, as Burda et al. do with $k = 5000$; `test_the_bound_tightens_with_more_samples` checks that the estimate rises from $k = 1$ to 5 to 50.

> [!pitfall]
> **A weight on the KL term left in place.** KL annealing and beta change the objective; a model trained with a final weight other than 1 is not trained on a bound, and its loss is not an ELBO. `test_kl_annealing_starts_from_zero` checks that the first step of an annealed run is the reconstruction loss alone.

## Experiments

This page has no experiment yet.
Some tests are small experiments.
`test_training_raises_the_elbo_and_the_unneeded_latents_go_inactive` trains the NumPy reference with a Gaussian decoder, 16 hidden units and 6 latent dimensions on points near a two-dimensional surface in $\mathbb{R}^8$ for 3000 Adam steps; the bound rises by more than 10 nats, exactly two latent dimensions end up active by the statistic of Burda et al., and the encoder's distribution is the prior on the other four, with a KL term below 0.02.
`test_the_weight_of_the_kl_trades_rate_for_distortion` trains the same model at beta = 0.1, 1 and 10; the rate falls and the distortion rises with beta, beta = 10 leaves a rate below 0.01 nats and no active dimension, and beta = 0.1 uses all six.
`test_the_bound_tightens_with_more_samples` evaluates the trained model's importance weighted bound with $k = 1$, 5 and 50 and finds the order of Theorem 1, the bound rising by more than 0.1 nats from $k = 1$ to 5.
In PyTorch `test_training_raises_the_bound` trains the module and checks that its IWAE bound is above its ELBO, and in JAX `test_more_samples_tighten_the_bound` checks the same order with `jax.vmap` over 400 keys.
The checks of the five derivations compare the identity of eq-elbo with closed forms for linear Gaussian models, eq-kl with integrals and averages, the variances of the two gradient estimators with samples, the order of the importance weighted bounds with exact enumeration over a five-valued latent variable, and the counts with the arrays and an instrumented forward pass.

## History

Training a recognition network alongside a directed generative model goes back to the Helmholtz machine and the [[al:wake-sleep-algorithm|wake-sleep algorithm]] of the mid-1990s.
Kingma and Welling write that wake-sleep was the only other on-line learning method they knew for the same general class of continuous latent variable models; like their method it uses a recognition model, but it optimizes two objectives at once that together do not correspond to a bound on the marginal likelihood, while it has the advantage of applying to discrete latent variables. {#c-kw-wake-sleep-relation}
Their paper was posted to arXiv in December 2013 and presented at ICLR 2014, and Rezende, Mohamed and Wierstra posted theirs in January 2014 and presented it at ICML 2014.
Rezende et al. write that Kingma and Welling concurrently presented an alternative discussion of stochastic backpropagation, that the two approaches were developed simultaneously and that they give complementary perspectives. {#c-rmw-concurrent}
Kingma and Welling in turn write that the work of Rezende et al. was developed independently of theirs and gives an additional perspective on AEVB. {#c-kw-rmw}
The reparameterization of a Gaussian expectation's gradient has older roots, in the theorems of Bonnet (1964) and Price (1958) that Rezende et al. cite.
Kingma and Welling listed as future directions hierarchical generative models with deep, for example convolutional, encoders and decoders, time-series models, applying the estimator to the global parameters, and supervised models with latent variables. {#c-kw-future}
Burda et al. note that after publication they learned that Laurent Dinh and Vincent Dumoulin had independently explored the importance weighted bound for training VAEs. {#c-iwae-dinh}
The problems of the objective, an unused code and a bound that does not measure the representation, were named by Bowman et al. in 2015 and analysed by Alemi et al. in 2017; the beta-VAE of Higgins et al. (2017) turned the weight of the KL term into a control.
In 2024 ICLR gave Auto-Encoding Variational Bayes its Test of Time award.

## References

- D. P. Kingma and M. Welling. Auto-Encoding Variational Bayes. arXiv 1312.6114, 2013; ICLR 2014.
- D. J. Rezende, S. Mohamed and D. Wierstra. Stochastic Backpropagation and Approximate Inference in Deep Generative Models. ICML 2014.
- Y. Burda, R. Grosse and R. Salakhutdinov. Importance Weighted Autoencoders. arXiv 1509.00519, 2015; ICLR 2016.
- S. R. Bowman, L. Vilnis, O. Vinyals, A. M. Dai, R. Jozefowicz and S. Bengio. Generating Sentences from a Continuous Space. arXiv 1511.06349, 2015; CoNLL 2016.
- A. A. Alemi, B. Poole, I. Fischer, J. V. Dillon, R. A. Saurous and K. Murphy. Fixing a Broken ELBO. arXiv 1711.00464, 2017; ICML 2018.
- I. Higgins, L. Matthey, A. Pal, C. Burgess, X. Glorot, M. Botvinick, S. Mohamed and A. Lerchner. beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework. ICLR 2017.
- P. Dayan, G. E. Hinton, R. M. Neal and R. S. Zemel. The Helmholtz Machine. Neural Computation 7(5), 1995.
