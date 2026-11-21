---
tag: 05S18M
kind: architecture
slug: energy-based-model
name: Energy-based model
aliases: [EBM, energy-based generative model, unnormalized model, Boltzmann machine, restricted Boltzmann machine, RBM, harmonium, product of experts, contrastive divergence training, joint energy-based model, JEM]
area: generative-models/energy-based-models
level: core
summary: >-
  A generative model that assigns every input a scalar energy and gives it probability proportional to the exponential of minus that energy, leaving the normalizing constant uncomputed.
uses: [exponential-family@m2, markov-random-field@m2, markov-chain@m2, stationary-distribution@m2, detailed-balance@m2, simulated-annealing@m2, langevin-equation@m2, bernoulli-distribution@m2, normal-distribution@m2, multivariate-normal-distribution@m2, logistic-regression@m2, kullback-leibler-divergence@m2, expected-value@m2, maximum-likelihood-estimation@m2, consistent-estimator@m2, cramer-rao-bound@m2, newtons-method-in-optimization@m2, matrix-multiplication@m2, partition-function@al, contrastive-divergence@al, spectral-normalization@al, experience-replay@al, backpropagation@al, weight-decay@al]
related_to: [autoregressive-model, variational-autoencoder, generative-adversarial-network, normalizing-flow, diffusion-model, flow-matching, autoencoder, multilayer-perceptron, resnet, adam]
history:
  year: 1985
  people: [David Ackley, Geoffrey Hinton, Terrence Sejnowski, Paul Smolensky, Tijmen Tieleman, Aapo Hyvärinen, Michael Gutmann, Max Welling, Yee Whye Teh, Yilun Du, Igor Mordatch, Will Grathwohl]
  note: "Ackley, Hinton and Sejnowski's Boltzmann machine (1985) defined a distribution over binary states by an energy and learned it from equilibrium statistics. Smolensky's restricted version (1986) removed the connections within each layer, and Hinton's contrastive divergence (2000, published 2002) made it trainable with one Gibbs step. Score matching (2005) and noise-contrastive estimation (2010) fitted unnormalized models without sampling, and Du and Mordatch (2019) trained deep convolutional energies on images with Langevin dynamics, which Grathwohl and colleagues turned into the energy of a classifier the same year."
prereqs: [multilayer-perceptron, exponential-family@m2, markov-chain@m2, maximum-likelihood-estimation@m2, partition-function@al]
sources:
  - {paper: "Ackley, Hinton and Sejnowski (1985)", title: "A Learning Algorithm for Boltzmann Machines", locator: "Cognitive Science 9, pp. 4 to 12 of the PDF", note: "the energy of a global state, settling to a local minimum, the Boltzmann distribution, annealing, the learning rule from equilibrium co-occurrences and noisy clamping"}
  - {paper: "Hinton (2000)", title: "Training Products of Experts by Minimizing Contrastive Divergence", locator: "Gatsby technical report GCNU TR 2000-004, pp. 3 to 8", note: "contrastive divergence, the dropped third term, one reconstruction per data vector, the RBM as a product of experts and the digit experiments"}
  - {paper: "Tieleman (2008)", title: "Training Restricted Boltzmann Machines using Approximations to the Likelihood Gradient", locator: "ICML 2008, pp. 1, 2, 5", note: "persistent contrastive divergence, the bias of CD-1 and the role of weight decay in mixing"}
  - {paper: "Hyvärinen (2005)", title: "Estimation of Non-Normalized Statistical Models by Score Matching", locator: "JMLR 6, pp. 2 to 7 of the PDF", note: "the score, the partial-integration objective, consistency and the Gaussian case"}
  - {paper: "Gutmann and Hyvärinen (2010)", title: "Noise-contrastive estimation: A new estimation principle for unnormalized statistical models", locator: "AISTATS 2010, PMLR 9, pp. 1 to 4", note: "logistic regression against noise, the normalizer as a parameter, consistency and the choice of noise"}
  - {paper: "Welling and Teh (2011)", title: "Bayesian Learning via Stochastic Gradient Langevin Dynamics", locator: "ICML 2011, p. 2", note: "Langevin dynamics as a discretized diffusion and its step sizes"}
  - {paper: "Du and Mordatch (2019)", title: "Implicit Generation and Generalization in Energy-Based Models", locator: "NeurIPS 2019; arXiv 1903.08689v6, pp. 3 to 5", note: "Langevin sampling from a network energy, the replay buffer, spectral normalization and the energy penalty"}
  - {paper: "Grathwohl, Wang, Jacobsen, Duvenaud, Norouzi and Swersky (2019)", title: "Your Classifier is Secretly an Energy Based Model and You Should Treat it Like One", locator: "ICLR 2020; arXiv 1912.03263v3, pp. 2 to 4, 8", note: "the energy of a classifier's logits, the CIFAR-10 results and the instability of training"}
family: energy-based-models
inputs: "a batch of data vectors as a (B, D) array, binary for the RBM of this page and real for a network energy; for sampling, B starting points of the Markov chains"
outputs: "for each input a scalar energy, or for the RBM its free energy, as a (B,) array; for training an estimate of the gradient of the log-likelihood; for sampling B vectors as a (B, D) array after k steps of Gibbs sampling or Langevin dynamics"
components: []
inductive_biases: ["any function of the input can be the energy, so the model imposes no structure on the distribution beyond what the energy function can express", "probabilities are only defined up to a constant, so the model compares inputs rather than scoring them in absolute terms", "the RBM's bipartite graph makes the hidden units independent given the visible ones and the visible units independent given the hidden ones", "a smooth energy, as from a network with bounded Lipschitz constant, gives a smooth density whose gradient guides the sampler"]
params: "DH + D + H for the restricted Boltzmann machine of this page with D visible and H hidden binary units"
flops_per_example: "4DH for one full block Gibbs step v to h to v, the two products with W, counting a multiply-add as 2 FLOPs; CD-k costs 4DH(k + 1) per vector, and the free energy costs 2DH"
---

An energy-based model is a generative model that assigns each input $\mathbf{x}$ a real number $E_\theta(\mathbf{x})$, its energy, and gives it a probability proportional to $\exp(-E_\theta(\mathbf{x}))$.
Low energy means likely.
The normalizing constant, the sum or integral of $\exp(-E)$ over all inputs, is almost never computed, so the model can rank inputs and find likely ones but cannot say exactly how likely any one is.
Training raises the probability of the data by lowering its energy and raising the energy of the samples the model would draw, which usually requires sampling from the model with a Markov chain during training.
Boltzmann machines, restricted Boltzmann machines and deep network energies trained with Langevin dynamics are energy-based models, and score matching and noise-contrastive estimation fit them without the normalizing constant.

## In plain words

Picture a landscape over all possible inputs, with valleys at the inputs that look like the data and hills everywhere else.
The energy is the height of the landscape.
An energy-based model learns the landscape: it digs valleys where the training examples lie and raises the ground where they do not.
To draw a new example, it drops a ball somewhere and lets it roll downhill while shaking it a little, so the ball spends most of its time in the deep valleys but can still hop between them.
The model never needs the total area of the valleys, which is what an exact probability would require, because rolling downhill only needs the local slope.
The price is that training also needs the rolling balls: to know which hills to raise, the model has to find out where its own balls end up.

## Overview

An energy function $E_\theta$, any real-valued function of the input with parameters $\theta$, defines the distribution

$$ p_\theta(\mathbf{x}) = \frac{\exp(-E_\theta(\mathbf{x}))}{Z(\theta)} , \qquad Z(\theta) = \sum_{\mathbf{x}} \exp(-E_\theta(\mathbf{x})) , $$ {#eq-gibbs}

with an integral in place of the sum for continuous inputs.
This is the Boltzmann or Gibbs distribution at temperature 1, and $Z$ is the [[al:partition-function|partition function]].
Any distribution that is positive everywhere can be written this way, with $E = -\log p$ up to a constant, so eq-gibbs restricts nothing until the form of $E$ is chosen.
Ackley, Hinton and Sejnowski noted that the difference in the log probabilities of two global states is just their energy difference, and that the equilibrium distribution is independent of the path followed in reaching it. {#c-boltzmann-logprob}
Ratios of probabilities, and the gradient $\nabla_\mathbf{x} \log p_\theta = -\nabla_\mathbf{x} E_\theta$, do not need $Z$. Absolute probabilities do.

For a discrete input of $D$ binary variables the sum in $Z$ has $2^D$ terms, and for continuous inputs it is an integral over the whole space, so $Z$ is intractable except for small or special models.
The [[m2:exponential-family|exponential family]] is the special case where the energy is linear in the parameters, $E_\theta(\mathbf{x}) = -\theta \cdot T(\mathbf{x})$, and a [[m2:markov-random-field|Markov random field]] is one whose energy is a sum of terms over the cliques of a graph.

### Maximum likelihood and its two phases

The gradient of the log-likelihood of a data point has two terms:

$$ \nabla_\theta \log p_\theta(\mathbf{x}) = -\nabla_\theta E_\theta(\mathbf{x}) + \mathbb{E}_{\mathbf{x}' \sim p_\theta}\big[\nabla_\theta E_\theta(\mathbf{x}')\big] . $$ {#eq-ml-grad}

For a model of eq-gibbs with a differentiable energy and a finite partition function, the gradient of the log-likelihood of $\mathbf{x}$ is minus the energy gradient at $\mathbf{x}$ plus the expected energy gradient under the model, because the gradient of $\log Z$ is that expectation. Averaged over the data, it vanishes when the data and the model give the energy gradient the same expectation; for an exponential family this is moment matching, and it is the stationary condition of $\mathrm{KL}(q \,\|\, p_\theta)$. {#c-ml-grad}
The derivation `ml-gradient` proves it, and `test_exact_gradient_matches_finite_differences` checks it on a small RBM by enumerating all inputs.
The first term, the positive phase, lowers the energy of the data. The second, the negative phase, raises the energy of the model's own samples, and needs samples from $p_\theta$.
Every training method for energy-based models either estimates the negative phase with a Markov chain or avoids it with another objective.

## Components

The notation has vectors as rows, $\sigma(z) = 1 / (1 + e^{-z})$ and $\operatorname{softplus}(z) = \log(1 + e^z)$.

### Boltzmann machines

A Boltzmann machine has binary units $s_i \in \{0, 1\}$, symmetric weights $w_{ij} = w_{ji}$ and biases, and the energy

$$ E(\mathbf{s}) = -\sum_{i<j} w_{ij} s_i s_j - \sum_i \theta_i s_i . $$

Each global state of a network of symmetrically connected binary units can be assigned a single number called the energy; if the units make their decisions asynchronously and transmission times are negligible, the system always settles into a local energy minimum, as in Hopfield's network of 1982. {#c-boltzmann-energy}
A Boltzmann machine replaces the deterministic update with a stochastic one: unit $i$ turns on with probability $\sigma$ of its energy gap, which is Gibbs sampling, and its states then follow eq-gibbs at equilibrium.
The units split into visible ones, which carry the data, and hidden ones, which model the dependencies the pairwise weights cannot.
The learning rule follows from eq-ml-grad: the derivative of the log probability in $w_{ij}$ depends on $p_{ij}$, the probability of finding units $i$ and $j$ both on at the same time when the system is at equilibrium. Learning compares this co-occurrence with the visible units clamped to the data and with the network running freely. {#c-boltzmann-grad}
Reaching equilibrium was slow, and [[m2:simulated-annealing|annealing]] helped: a good way to beat the trade-off between a favourable bias at low temperature and fast equilibrium at high temperature is to start at a high temperature and gradually reduce it. {#c-annealing}

### Restricted Boltzmann machines

Smolensky introduced a restricted Boltzmann machine with one visible layer, one hidden layer and no intralayer connections. Freund and Haussler realised that in it the probability of a visible vector is proportional to the product of the probabilities that each hidden unit acting alone would generate it, so an RBM is a product of experts with one expert per hidden unit, and exact inference is tractable because the hidden states are conditionally independent given the data. {#c-rbm-poe}
With $D$ visible units $\mathbf{v}$, $H$ hidden units $\mathbf{h}$, weights $W$ of shape $(D, H)$ and biases $\mathbf{b}$ and $\mathbf{c}$, the energy is

$$ E(\mathbf{v}, \mathbf{h}) = -\mathbf{v} W \mathbf{h}^\top - \mathbf{b} \cdot \mathbf{v} - \mathbf{c} \cdot \mathbf{h} . $$ {#eq-rbm-energy}

Summing $\exp(-E)$ over the $2^H$ hidden vectors factorizes over the hidden units, which leaves the free energy of the visible vector:

$$ F(\mathbf{v}) = -\mathbf{b} \cdot \mathbf{v} - \sum_{j=1}^{H} \operatorname{softplus}\big(c_j + (\mathbf{v} W)_j\big) , \qquad p(\mathbf{v}) = \frac{e^{-F(\mathbf{v})}}{Z} . $$ {#eq-free-energy}

The conditionals factorize too:

$$ p(h_j = 1 \mid \mathbf{v}) = \sigma\big(c_j + (\mathbf{v} W)_j\big) , \qquad p(v_i = 1 \mid \mathbf{h}) = \sigma\big(b_i + (W \mathbf{h}^\top)_i\big) . $$ {#eq-rbm-conditionals}

In an RBM with energy eq-rbm-energy, summing over the hidden units gives the free energy eq-free-energy, so $p(\mathbf{v}) = e^{-F(\mathbf{v})}/Z$ costs $O(DH)$ up to $Z$; and the hidden units are independent given the visible units, and the visible units given the hidden units, with the sigmoid conditionals of eq-rbm-conditionals. {#c-rbm-conditionals}
The derivation `rbm-free-energy-and-conditionals` proves it, and `test_free_energy_sums_out_the_hidden_units` compares both with sums over all hidden vectors.
Block Gibbs sampling alternates the two: draw all of $\mathbf{h}$ given $\mathbf{v}$ at once, then all of $\mathbf{v}$ given $\mathbf{h}$.
For the RBM, the statistics in eq-ml-grad are $\mathbf{v}^\top \mathbf{h}$, $\mathbf{v}$ and $\mathbf{h}$, and the hidden ones can be replaced by their conditional means $p(\mathbf{h} = 1 \mid \mathbf{v})$, which lowers the variance.
The partition function still sums over $2^D$ visible vectors, and the model expectation still needs samples.

### Contrastive divergence

Hinton's contrastive divergence runs the Gibbs chain for one full step from each data vector instead of to equilibrium, and updates the parameters to reduce the tendency of the chain to wander away from the data on that step. Because the one-step distribution is one step closer to equilibrium, the contrastive divergence can never be negative, and the intractable expectation over the equilibrium distribution cancels out. {#c-cd-def}
The update for an RBM is

$$ \Delta W \propto \big\langle \mathbf{v}^\top p(\mathbf{h} \mid \mathbf{v}) \big\rangle_{\text{data}} - \big\langle \mathbf{v}^\top p(\mathbf{h} \mid \mathbf{v}) \big\rangle_{k} , $$ {#eq-cd}

where $\langle \cdot \rangle_k$ averages over the visible vectors after $k$ steps of block Gibbs sampling started at the data, and likewise for $\mathbf{b}$ and $\mathbf{c}$. This is CD-$k$.
The gradient of the contrastive divergence has a third term that is problematic to compute, but Hinton's simulations showed it can safely be ignored because it is small and it seldom opposes the resultant of the other two terms; the update works very well even with a single reconstruction of each data vector in place of the full distribution over reconstructions. {#c-cd-third-term}
CD-$k$ is biased for finite $k$ and exact in the limit:

For an RBM, one block Gibbs step is a Markov chain on the visible vectors whose transition probabilities are all positive and which leaves $p(\mathbf{v})$ invariant, so the distribution after $k$ steps from the data converges to $p(\mathbf{v})$ and the expected CD-$k$ update converges to the exact gradient of eq-ml-grad as $k \to \infty$; for finite $k$ it differs from it. {#c-cd-limit}
The derivation `cd-k-limit` proves it, and `test_cd_k_approaches_the_exact_gradient_as_k_grows` computes the expected CD-$k$ update exactly from the $2^D \times 2^D$ transition matrix for $k = 1, 3, 10$ and 60.
CD-1 is fast, has low variance, and is a reasonable approximation to the likelihood gradient, but it is still significantly different from the likelihood gradient when the mixing rate is low. {#c-cd1-bias}

### Persistent contrastive divergence

Tieleman keeps the negative chains across updates: the Markov chain is not reset between parameter updates, which is an approximation because the model changes slightly with each update; with an infinitesimally small learning rate it becomes exact, and it seems to work best with small learning rates. He called this Persistent Contrastive Divergence. {#c-pcd}
The persistent chains are a population of fantasy particles that keep moving under the current model, so they can travel further than $k$ steps from the data.
Tieleman compared it with standard Contrastive Divergence and Pseudo-Likelihood algorithms on test data likelihood and classification error rate, and found that the Persistent Contrastive Divergence algorithm outperforms the other algorithms, and is equally fast and simple. {#c-pcd-results}

### Network energies and Langevin dynamics

For continuous inputs the energy can be any network with a scalar output, such as a [[multilayer-perceptron]] or a [[resnet]], and the samples for the negative phase come from Langevin dynamics, a discretization of the [[m2:langevin-equation|Langevin equation]]:

$$ \tilde{\mathbf{x}}_k = \tilde{\mathbf{x}}_{k-1} - \frac{\lambda}{2} \nabla_\mathbf{x} E_\theta(\tilde{\mathbf{x}}_{k-1}) + \boldsymbol{\omega}_k , \qquad \boldsymbol{\omega}_k \sim \mathcal{N}(0, \lambda I) . $$ {#eq-langevin}

Each step moves downhill on the energy and adds Gaussian noise whose variance equals the step size.
Langevin dynamics take gradient steps but also inject Gaussian noise; the step sizes and the variances of the injected noise are balanced so that the variance of the samples matches that of the target distribution, and as the step size decreases the discretization error decreases so that the rejection rate of a Metropolis-Hastings correction approaches zero. {#c-sgld}
Du and Mordatch use the same procedure with a network energy: as the number of steps $K \to \infty$ and the step size $\lambda \to 0$, the distribution $q_\theta$ of $\tilde{\mathbf{x}}_K$ approaches $p_\theta$, so samples are generated implicitly by the energy function rather than explicitly by a feedforward network. {#c-du-langevin}
For a finite step the chain samples a distorted distribution:

For the quadratic energy $E(\mathbf{x}) = \|\mathbf{x}\|^2 / (2s^2)$, whose Gibbs distribution is $\mathcal{N}(0, s^2 I)$, the Langevin chain of eq-langevin with step $\lambda < 4s^2$ is a linear autoregression whose stationary distribution is Gaussian with variance $s^2 / (1 - \lambda/(4s^2))$ in each coordinate, larger than $s^2$ and equal to it only in the limit $\lambda \to 0$. {#c-langevin-variance}
The derivation `langevin-stationary-variance` proves it, and `test_langevin_stationary_variance_on_a_quadratic_energy` runs 20,000 chains at three step sizes.
In practice the chains are short, a few tens of steps, and start from a replay buffer of earlier samples, the role persistent chains played for the RBM:
Du and Mordatch keep a replay buffer of past generated samples and initialize Langevin dynamics from it 95% of the time and from uniform noise otherwise, which continues to refine past samples and increases the effective number of sampling steps and sample diversity. {#c-du-buffer}
The update is eq-ml-grad with the buffer samples in the negative phase, which is a form of [[al:experience-replay|experience replay]].
Training is unstable without constraints on the energy:
To constrain the Lipschitz constant of the energy network, Du and Mordatch add spectral normalization to all layers, and they weakly L2 regularize the energy magnitudes of both positive and negative samples, since otherwise the energies drift to numerically unstable values while their difference is preserved. {#c-du-reg}
The penalty is [[al:spectral-normalization|spectral normalization]] plus $\alpha(E(\mathbf{x}^+)^2 + E(\mathbf{x}^-)^2)$ added to the loss $E(\mathbf{x}^+) - E(\mathbf{x}^-)$.

### Score matching

Score matching avoids the partition function by fitting the gradient of the log-density in $\mathbf{x}$, the score $\psi_\theta(\mathbf{x}) = \nabla_\mathbf{x} \log p_\theta(\mathbf{x}) = -\nabla_\mathbf{x} E_\theta(\mathbf{x})$.
The point in using the score function is that it does not depend on the normalization constant $Z(\theta)$. {#c-score-z}
Hyvärinen's objective is the expected squared distance between the model score and the data score, which by a partial integration does not need the data score:

$$ J(\theta) = \mathbb{E}_{\mathbf{x} \sim q} \sum_{i=1}^{D} \Big[ \partial_i \psi_{\theta,i}(\mathbf{x}) + \tfrac{1}{2} \psi_{\theta,i}(\mathbf{x})^2 \Big] = \mathbb{E}_{\mathbf{x} \sim q} \Big[ -\Delta E_\theta(\mathbf{x}) + \tfrac{1}{2} \|\nabla_\mathbf{x} E_\theta(\mathbf{x})\|^2 \Big] . $$ {#eq-score-matching}

If the data density and the model score are differentiable and the product of the density and the score vanishes at infinity, $\tfrac{1}{2}\mathbb{E}_q \|\psi_\theta - \nabla \log q\|^2$ equals $J(\theta)$ of eq-score-matching plus a constant that does not depend on $\theta$. For the Gaussian energy $\tfrac{1}{2}(\mathbf{x} - \boldsymbol{\mu}) M (\mathbf{x} - \boldsymbol{\mu})^\top$ the minimizer of the sample version of $J$ is the sample mean and the inverse of the sample covariance, which is the maximum likelihood estimate for every sample. {#c-score-matching}
The derivation `score-matching-integration-by-parts` proves both parts, and `test_score_matching_drops_the_data_score_and_matches_maximum_likelihood` checks them on 5000 Gaussian points.
Hyvärinen proved the estimator locally consistent; in his simulations it gave slightly larger errors than maximum likelihood, which is to be expected from the efficiency of maximum likelihood estimation. {#c-sm-efficiency}
The Laplacian $\Delta E$ costs $D$ backward passes for a network energy, which is why denoising and sliced versions of score matching were developed. Denoising score matching is the objective of the [[diffusion-model|diffusion models]].

### Noise-contrastive estimation

Noise-contrastive estimation performs nonlinear logistic regression to discriminate between the observed data and artificially generated noise, using the model log-density in the regression nonlinearity; it gives a consistent estimator, works directly for unnormalized models, and estimates the normalization constant just like any other parameter. {#c-nce-def}
The model is $\log p_m(\mathbf{x}) = -E_\theta(\mathbf{x}) + c$ with $c$ a free parameter for $-\log Z$. With as many noise samples $\mathbf{y} \sim p_n$ as data points, the objective is the log-likelihood of [[m2:logistic-regression|logistic regression]] with the log-ratio as the logit:

$$ J(\theta) = \mathbb{E}_{\mathbf{x} \sim q} \log \sigma\big(G(\mathbf{x})\big) + \mathbb{E}_{\mathbf{y} \sim p_n} \log\big(1 - \sigma(G(\mathbf{y}))\big) , \qquad G(\mathbf{u}) = \log p_m(\mathbf{u}) - \log p_n(\mathbf{u}) . $$ {#eq-nce}

Maximum likelihood cannot treat the normalizer this way: the likelihood can be made arbitrarily large by making $Z$ go to zero. {#c-nce-mle-z}

Over all functions $f$ used in place of $\log p_m$, the objective eq-nce is maximized by $f = \log q$, the data log-density, and has no other maximum if the noise density is positive wherever the data density is; the maximizing density has unit integral without any constraint, so the fitted $c$ is the log-normalizer. {#c-nce-optimum}
The derivation `nce-optimum` proves it pointwise, and `test_nce_recovers_the_parameters_and_the_normalizer` fits a one-dimensional Gaussian with its normalizer by Newton's method.
The noise matters for efficiency: Gutmann and Hyvärinen argue that the noise distribution should be close to the data distribution, because otherwise the classification problem might be too easy; if the noise is equal to the data distribution, the asymptotic covariance of the estimator is two times the Cramér-Rao bound. {#c-nce-noise}

### Classifiers as energy-based models

Grathwohl and colleagues re-use the logits $f_\theta(\mathbf{x})$ of any classifier with $K$ classes to define an energy:

$$ E_\theta(\mathbf{x}) = -\log \sum_{y=1}^{K} \exp\big(f_\theta(\mathbf{x})[y]\big) , \qquad p_\theta(y \mid \mathbf{x}) = \frac{\exp(f_\theta(\mathbf{x})[y])}{\sum_{y'} \exp(f_\theta(\mathbf{x})[y'])} . $$ {#eq-jem}

The joint model is $p_\theta(\mathbf{x}, y) \propto \exp(f_\theta(\mathbf{x})[y])$, whose marginal has the energy of eq-jem and whose conditional is the usual softmax.
Shifting the logits of an ordinary classifier by an arbitrary scalar does not affect the model, but in the joint energy-based model shifting the logits for a data point changes $\log p_\theta(\mathbf{x})$, so the extra degree of freedom in the logits defines the density over inputs. {#c-jem-def}
Adding a constant $s(\mathbf{x})$ to all $K$ logits of a classifier lowers the energy of eq-jem by $s(\mathbf{x})$ and leaves the class probabilities unchanged. {#c-jem-shift}
The derivation `jem-logit-shift` proves it and `test_classifier_energy_moves_with_a_logit_shift_and_the_class_probabilities_do_not` checks it.
JEM trains $\log p(y \mid \mathbf{x})$ with the [[al:cross-entropy-loss|cross-entropy loss]] and $\log p(\mathbf{x})$ with eq-ml-grad and persistent Langevin chains, following Du and Mordatch.

## Shapes and parameters

The shape contract is the restricted Boltzmann machine of eq-rbm-energy with $D$ binary visible units and $H$ binary hidden units. The example is $D = 4$, $H = 3$.
The parameters are $W$ of shape $(D, H)$, the visible biases $\mathbf{b}$ of shape $(D,)$ and the hidden biases $\mathbf{c}$ of shape $(H,)$, so

$$ P = DH + D + H . $$ {#eq-params}

In the contract this is
{shape:params}
and the example has {shape:params@example} parameters.
The RBM of `test_rbm_learns_a_binary_mixture`, `mixture`, has {shape:params@mixture}. Hinton's RBM on $16 \times 16$ digits, `usps_digits`, has {shape:params@usps_digits}, and a 500-unit RBM on MNIST, `mnist`, has {shape:params@mnist}.
An RBM with 500 hidden units and 256 visible units was trained by contrastive divergence on 8000 real-valued $16 \times 16$ images of handwritten digits from all 10 classes of the USPS Cedar ROM training set, with the pixel intensities used as probabilities. {#c-usps}
A network energy has the parameters of its network, for example the {shape:params@example} of the example RBM against the $dH + 2H$ of the one-hidden-layer energy $E(\mathbf{x}) = \tanh(\mathbf{x} W_1 + \mathbf{b}_1) \cdot \mathbf{w}_2 + \gamma \|\mathbf{x}\|^2/2$ of the implementations, whose quadratic term has no parameter.
The derivation `parameter-and-flop-count` proves the count.

## Computation and cost

One full block Gibbs step $\mathbf{v} \to \mathbf{h} \to \mathbf{v}$ on $B$ vectors multiplies by $W$ and by $W^\top$:

$$ C = 4BDH $$ {#eq-flops}

FLOPs, a multiply-add counted as two, leaving out the biases, the sigmoids and the Bernoulli draws.
An RBM with $D$ visible and $H$ hidden units has $DH + D + H$ parameters; one full block Gibbs step on $B$ vectors costs $4BDH$ FLOPs counting a multiply-add as 2, the free energy of $B$ vectors costs $2BDH$, a CD-$k$ update costs $4BDH(k + 1)$ for its $k$ Gibbs steps and the up passes of the data and of the last sample, and the exact partition function costs $2^D$ free energies. {#c-params}
In the contract,
{shape:flops_forward}
which is {shape:flops_forward@example} FLOPs for the example and {shape:flops_forward@mnist} per image for `mnist`.
The cost of an energy-based model is dominated by the sampler: each training step of a network energy runs $K$ Langevin steps, each a forward and a backward pass of the network, so a step costs about $3K$ times a forward pass on the batch, against one forward and one backward pass for a classifier.
Exact likelihoods are out of reach above a few dozen binary units; reported RBM likelihoods use estimates of $\log Z$ such as annealed importance sampling.
Sampling a trained model costs as many sampler steps as the chain needs to mix, and is not bounded in advance.

## Training

The general recipe for a network energy, eq-ml-grad with a Langevin negative phase, is:

1. take a batch $\mathbf{x}^+$ of data;
2. take starting points $\tilde{\mathbf{x}}_0$, from a replay buffer most of the time and from uniform noise otherwise;
3. run $K$ steps of eq-langevin from them, with the parameters held fixed, to get $\mathbf{x}^-$;
4. take a gradient step on $E_\theta(\mathbf{x}^+) - E_\theta(\mathbf{x}^-)$ plus the regularizers, with [[al:backpropagation|backpropagation]] and an optimizer such as [[adam]];
5. put $\mathbf{x}^-$ back in the buffer.

For an RBM the gradient has the closed form of eq-cd and the sampler is block Gibbs; CD-$k$ restarts the chains at the batch, PCD keeps them.
In practice the step size and the noise of Langevin dynamics are tuned separately: Grathwohl and colleagues note that the step size and the standard deviation of the noise are often chosen separately, leading to a biased sampler which allows for faster training. {#c-jem-biased-sampler}
The NumPy reference does the same, with step 0.02 and noise 0.05 instead of $\sqrt{0.02} \approx 0.14$.
[[al:weight-decay|Weight decay]] helps even with unlimited data:
Tieleman found that all four algorithms worked best with some weight decay even on an artificially generated data set with an infinite amount of data, because CD algorithms depend on the mixing rate of the Gibbs sampler and the mixing rate is higher when the parameters of the model are smaller. {#c-weight-decay-mixing}
Monitoring is hard, because the training loss $E(\mathbf{x}^+) - E(\mathbf{x}^-)$ does not track the likelihood. Small models can be monitored by the exact likelihood, as in the tests; larger ones by samples, held-out estimates of $\log Z$, or a downstream task.

## Inductive biases

An energy-based model assumes the least of the generative models about the form of the distribution: any function with a scalar output is a valid energy, with no invertibility, no ordering of the variables and no latent code.
The assumptions sit in the energy function, its architecture and its smoothness, and in the sampler, whose short chains decide which regions of the energy the training ever probes.
The RBM's bipartite graph makes inference in one layer given the other exact and parallel, by c-rbm-conditionals, and makes the hidden units a set of binary features, each an expert that votes on the visible vector.
A smooth network energy, which spectral normalization encourages, gives a density that changes slowly, so that gradient-based samplers can move through it, and assigns some probability to regions between the modes.
Because only differences in energy matter, an energy-based model naturally expresses comparisons and constraints, such as the compatibility of an input with a label in eq-jem, and the sum of two energies is the product of their distributions, so models compose by addition.

## Evidence

On CIFAR-10, JEM reached 92.9% classification accuracy, an Inception Score of 8.76 and an FID of 38.4, against 95.8% accuracy for the same Wide-ResNet trained as an ordinary classifier, an Inception Score of 8.3 and an FID of 37.9 for the energy-based model of Du and Mordatch, and 8.59 and 25.5 for SNGAN. {#c-jem-results}
Tieleman's comparisons on MNIST used the official training set split into 50,000 training cases and 10,000 validation cases, with the pixel intensities treated as probabilities of binary data, and RBMs with 500 hidden units. {#c-pcd-setup}
The tests of this page reproduce the main effects on small problems.
`test_rbm_learns_a_binary_mixture` trains an RBM with $D = H = 8$ for 800 steps, with CD-1 and with PCD, on 4000 draws from a mixture of three noisy binary prototypes. Its exact mean log-likelihood rises by more than 2 nats, and its KL divergence from the true distribution, summed over all 256 vectors, ends below 0.1 nats, while the product of the data's marginals is above 1.
`test_energy_trained_on_langevin_samples_finds_four_modes` trains a 32-unit network energy with the replay buffer on four Gaussian clusters and finds that Langevin chains from uniform noise end near the clusters, with each cluster receiving at least a tenth of them.
`test_nce_recovers_the_parameters_and_the_normalizer` recovers the mean, the variance and the log-normalizer of a Gaussian from 40,000 data and 40,000 noise points.

## Limitations

The likelihood is not available, so models cannot be compared by it without an estimate of $\log Z$, and the estimates are themselves noisy and often biased.
Training depends on a sampler that is not run to convergence, so the gradient is biased in a way that is hard to control, and the learned energy can fit the sampler's short-run behaviour rather than the data.
Grathwohl and colleagues found the gradient estimators used to train JEM quite unstable and prone to diverging if the sampling and optimization parameters are not tuned correctly; the models used for their results regularly diverged throughout training, requiring them to be restarted with lower learning rates or with increased regularization. {#c-jem-unstable}
Sample quality metrics can mislead: Du and Mordatch strongly encourage comparing with the FID score rather than the Inception Score, since Langevin dynamics converges to minima that artificially inflate the Inception Score. {#c-du-is}
Mixing between distant modes is slow for both Gibbs sampling and Langevin dynamics, so the model can assign mass to modes that its chains never visit and fail to remove them.
Score matching needs the Laplacian of the energy, which is expensive in high dimension, and is sensitive to regions of low data density; noise-contrastive estimation needs a noise distribution close to the data, by c-nce-noise, which is hard to find in high dimension.
Sampling is iterative and has no fixed cost, unlike a [[generative-adversarial-network]] or a [[variational-autoencoder]], which sample in one pass.

## Variants and descendants

- **Boltzmann machines and RBMs.** Gaussian-Bernoulli RBMs for real-valued data, conditional and convolutional RBMs, and deep belief networks and deep Boltzmann machines stacked from RBMs, which pretrained the first deep networks.
- **Products of experts and fields of experts**, energies that sum the log-potentials of many simple models.
- **Training variants.** CD-$k$ and PCD, fast-weight PCD, parallel tempering, and annealed importance sampling for estimating $\log Z$.
- **Objectives without sampling.** Score matching with its denoising and sliced forms, noise-contrastive estimation, ratio matching for binary data and pseudo-likelihood.
- **Deep energies with Langevin sampling.** The models of Du and Mordatch, short-run MCMC, JEM and its successors, and compositional energies whose sum expresses a conjunction of concepts.
- **Score-based and diffusion models.** These learn the score of noised data at every noise level by denoising score matching and sample with Langevin dynamics or a reverse diffusion. The score is the negative gradient of an energy that is never computed, and the [[diffusion-model]] page covers them.
- **Models with a tractable density.** A [[normalizing-flow]] buys an exact $\log p$ with an invertible network and a Jacobian determinant, where an energy-based model leaves the network free and pays at training and sampling time instead. [[flow-matching]] regresses a velocity field that carries noise to data along a chosen path and samples by integrating an ODE, with no Markov chain and no normalizer.
- **Energy-based views of other models.** A [[generative-adversarial-network]] discriminator can be read as an energy, a classifier's logits define one by eq-jem, and the [[m2:exponential-family|exponential family]] and conditional random fields are energy-based models with tractable or structured normalizers.

## Notable models

The Boltzmann machine of Ackley, Hinton and Sejnowski is a network with hidden units that learns a distribution from local, pairwise statistics.
RBMs trained by contrastive divergence were the building block of the deep belief networks of 2006, which were among the first deep networks trained layer by layer.
The convolutional energies of Du and Mordatch and JEM brought deep energies to image generation and to hybrid models that classify and generate with one network.
The score-based and diffusion models, which learn an energy's gradient rather than the energy, became the dominant image and video generators, and are connected to energy-based models by eq-langevin and eq-score-matching.

## Implementations

The NumPy reference `impl/numpy/energy_based.py` implements:

- the RBM of eq-rbm-energy, its free energy eq-free-energy, the conditionals of eq-rbm-conditionals, block Gibbs sampling, the exact $\log Z$, likelihood and gradient by enumeration, and the $2^D \times 2^D$ Gibbs transition matrix;
- CD-$k$ and PCD training by eq-cd;
- a one-hidden-layer network energy with its input gradient, its Laplacian and its parameter gradient by hand, Langevin dynamics by eq-langevin, and training with a replay buffer, the energy penalty and Adam;
- score matching by eq-score-matching, for a Gaussian energy in closed form and for the network energy;
- noise-contrastive estimation by eq-nce for a one-dimensional Gaussian with a free normalizer, fitted by [[m2:newtons-method-in-optimization|Newton's method]];
- the classifier energy of eq-jem.

The PyTorch version `impl/pytorch/energy_based.py` has an `RBM` module and a network energy as an `nn.Module`, with the input gradient and the Laplacian of score matching from `torch.autograd.grad`. `test_gradcheck` runs `torch.autograd.gradcheck` through the free energy and the score-matching objective.

The JAX version `impl/jax/energy_based.py` is a set of pure functions of a parameter dictionary. The score is a `jax.grad` of the energy in its input, the Laplacian the trace of a `jax.hessian`, and Langevin dynamics a `jax.lax.fori_loop`.

All three load the shared fixture of `fixtures/make.py`. It fixes an RBM with $D = 4$, $H = 3$, a network energy with $d = 2$ and 5 hidden units, and a set of uniforms and normal draws, and checks the energies, the free energies, the conditionals, one Gibbs step, $\log Z$, the CD and exact gradients, one Langevin step, the score-matching objective, the NCE objective and its gradient, and the classifier energy.

### Worked example

Take an RBM with $D = 2$ visible units, $H = 1$ hidden unit, $W = (2, -1)^\top$, $\mathbf{b} = 0$ and $c = -1$.
By eq-free-energy, $e^{-F(\mathbf{v})} = 1 + e^{c + \mathbf{v} W}$:

- $\mathbf{v} = (0, 0)$: $1 + e^{-1} = 1.3679$;
- $\mathbf{v} = (0, 1)$: $1 + e^{-2} = 1.1353$;
- $\mathbf{v} = (1, 0)$: $1 + e^{1} = 3.7183$;
- $\mathbf{v} = (1, 1)$: $1 + e^{0} = 2$.

So $Z = 8.2215$, $p(1, 0) = 3.7183 / 8.2215 = 0.4523$ and $p(1, 1) = 0.2433$.
By eq-rbm-conditionals, the hidden unit is on with probability $\sigma(1) = 0.7311$ given $(1, 0)$ and $\sigma(0) = 0.5$ given $(1, 1)$.
The first visible unit raises the hidden unit, the second lowers it, so the model prefers $(1, 0)$, where the hidden unit is most likely on.
`test_worked_example` checks these numbers.

### Pitfalls

> [!pitfall]
> The sign of the update is easy to flip. Eq-ml-grad lowers the energy of the data and raises the energy of the samples; a loss of $E(\mathbf{x}^-) - E(\mathbf{x}^+)$ instead of $E(\mathbf{x}^+) - E(\mathbf{x}^-)$ trains the model to avoid the data, and a gradient flowing through the sampler into $\mathbf{x}^-$ changes the objective. Stop the gradient at the samples.

> [!pitfall]
> Without a penalty the energies drift. The loss only sees differences, so both energies can fall or rise together until they overflow; add the $\alpha(E^{+2} + E^{-2})$ term of c-du-reg, and keep the samples inside the data's range, for example by clipping.

> [!pitfall]
> Langevin noise and step size must match for a correct sampler. Eq-langevin has noise variance equal to the step $\lambda$ and the drift $\lambda/2$ times the gradient; a smaller noise, as most deep EBMs use, gives a sampler that concentrates in the modes and does not sample $p_\theta$, by c-jem-biased-sampler, and a large step inflates the variance, by c-langevin-variance.

> [!pitfall]
> CD-1 samples are not model samples. The negative phase of CD-$k$ stays near the data, so the model can keep modes far from the data that its chains never visit, by c-cd1-bias. Judge a model by long chains from noise or by its exact likelihood on small problems, not by CD reconstructions.

> [!pitfall]
> Use the conditional means for the hidden statistics and sample the hidden units when you propagate down. Using probabilities in the up pass of the statistics lowers the variance, but feeding probabilities instead of binary samples to the down pass of the chain changes the sampler.

## Experiments

This page has no experiment yet.
Some tests are small experiments.

`test_rbm_learns_a_binary_mixture` and `test_energy_trained_on_langevin_samples_finds_four_modes`, described in the Evidence section, train small models end to end. The PyTorch and JAX versions run them too.

`test_cd_k_approaches_the_exact_gradient_as_k_grows` measures the bias of CD-$k$ exactly, without sampling noise, and finds it falls below $10^{-8}$ by $k = 60$.

`test_gibbs_sampling_draws_from_the_model` runs 20,000 Gibbs chains for 30 steps and compares their frequencies with the exact probabilities.

The checks of the eight derivations compare:

- the maximum-likelihood gradient with finite differences of the exact log-likelihood;
- the free energy and the conditionals with sums over all hidden vectors;
- the expected CD-$k$ update with the exact gradient, through powers of the transition matrix;
- the explicit and implicit score-matching objectives, and the Gaussian score-matching estimate with maximum likelihood;
- the NCE objective over unconstrained log-densities with its pointwise maximizer;
- the stationary variance of discretized Langevin dynamics with its formula;
- the classifier energy and class probabilities under a shift of the logits;
- the parameter and FLOP counts with the code.

## History

Hopfield's network of 1982 gave each state of a symmetric binary network an energy that its asynchronous updates never raise.
Ackley, Hinton and Sejnowski made the units stochastic in the Boltzmann machine of 1985, which defines a distribution by eq-gibbs and learns it from the difference of clamped and free co-occurrences, with simulated annealing to reach equilibrium.
Smolensky's harmonium of 1986 removed the connections within each layer, the model later called the restricted Boltzmann machine.
Hinton's contrastive divergence, a technical report of 2000 published in Neural Computation in 2002, replaced equilibrium sampling by one Gibbs step and made RBMs practical, and in 2006 stacks of RBMs pretrained the deep belief networks that revived deep learning.
Hyvärinen's score matching (2005) and Gutmann and Hyvärinen's noise-contrastive estimation (2010) fitted unnormalized models without sampling, and Tieleman's persistent contrastive divergence (2008) kept the chains across updates.
Welling and Teh's stochastic gradient Langevin dynamics (2011) brought Langevin sampling to large-scale learning, and Du and Mordatch (2019) used it to train deep convolutional energies on images, with JEM following the same year.
Song and Ermon's score-based models of 2019 and the diffusion models after them learned the score of noised data instead of the energy and became the leading image generators.

## References

- J. J. Hopfield. Neural networks and physical systems with emergent collective computational abilities. PNAS 79, 1982.
- D. H. Ackley, G. E. Hinton and T. J. Sejnowski. A Learning Algorithm for Boltzmann Machines. Cognitive Science 9, 1985.
- P. Smolensky. Information processing in dynamical systems: Foundations of harmony theory. In Parallel Distributed Processing, vol. 1, MIT Press, 1986.
- G. E. Hinton. Training Products of Experts by Minimizing Contrastive Divergence. Gatsby technical report GCNU TR 2000-004, 2000; Neural Computation 14, 2002.
- A. Hyvärinen. Estimation of Non-Normalized Statistical Models by Score Matching. JMLR 6, 2005.
- T. Tieleman. Training Restricted Boltzmann Machines using Approximations to the Likelihood Gradient. ICML 2008.
- M. Gutmann and A. Hyvärinen. Noise-contrastive estimation: A new estimation principle for unnormalized statistical models. AISTATS 2010.
- M. Welling and Y. W. Teh. Bayesian Learning via Stochastic Gradient Langevin Dynamics. ICML 2011.
- Y. Du and I. Mordatch. Implicit Generation and Generalization in Energy-Based Models. NeurIPS 2019.
- W. Grathwohl, K.-C. Wang, J.-H. Jacobsen, D. Duvenaud, M. Norouzi and K. Swersky. Your Classifier is Secretly an Energy Based Model and You Should Treat it Like One. ICLR 2020.
