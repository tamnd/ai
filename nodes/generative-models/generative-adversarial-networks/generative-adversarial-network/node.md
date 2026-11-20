---
tag: NEW-generative-adversarial-network
kind: architecture
slug: generative-adversarial-network
name: Generative adversarial network
aliases: [GAN, GANs, adversarial nets, generative adversarial nets, adversarial training of generative models, DCGAN, Wasserstein GAN, WGAN, WGAN-GP, non-saturating GAN]
area: generative-models/generative-adversarial-networks
level: core
summary: >-
  A generative model trained as a two-player game: a generator maps noise to samples, a discriminator tells them from data, and each is trained against the other.
uses: [kullback-leibler-divergence@m2, f-divergence@m2, cross-entropy@m2, logistic-regression@m2, stochastic-gradient-descent@m2, eigenvalue@m2, saddle-point-of-a-function-on-a-product@m2, minimax@al, nash-equilibrium@al, zero-sum-game@al, backpropagation@al]
related_to: [variational-autoencoder, autoencoder, multilayer-perceptron, batch-normalization, layer-normalization, adam, dropout, label-smoothing]
history:
  year: 2014
  people: [Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio]
  note: "Goodfellow et al. posted Generative Adversarial Nets as arXiv 1406.2661 in June 2014 and presented it at NIPS 2014. Radford, Metz and Chintala made convolutional GANs train reliably with DCGAN (2015), Salimans et al. collected training techniques and the Inception score (2016), Arjovsky, Chintala and Bottou replaced the Jensen-Shannon divergence by the Wasserstein distance (2017), Gulrajani et al. replaced weight clipping by a gradient penalty (2017), and Mescheder, Geiger and Nowozin proved which regularizers make training converge locally (2018)."
prereqs: [multilayer-perceptron, cross-entropy@m2, kullback-leibler-divergence@m2, minimax@al, backpropagation@al]
same_as:
  wikipedia: Generative_adversarial_network
sources:
  - {paper: "Goodfellow et al. (2014)", title: "Generative Adversarial Nets", locator: "arXiv 1406.2661v1, pp. 1 to 8", note: "the game, Algorithm 1, the non-saturating loss, Proposition 1, Theorem 1, Proposition 2, the Parzen-window results and the advantages and disadvantages"}
  - {paper: "Radford, Metz and Chintala (2015)", title: "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks", locator: "arXiv 1511.06434v2, pp. 3 to 4, 6 to 9", note: "the DCGAN guidelines, the training details, the feature-extractor results, the vector arithmetic and the remaining instability"}
  - {paper: "Salimans, Goodfellow, Zaremba, Cheung, Radford and Chen (2016)", title: "Improved Techniques for Training GANs", locator: "arXiv 1606.03498v1, pp. 1 to 8", note: "non-convergence of gradient descent on the game, feature matching, minibatch discrimination, one-sided label smoothing, the Inception score and Table 3"}
  - {paper: "Arjovsky, Chintala and Bottou (2017)", title: "Wasserstein GAN", locator: "arXiv 1701.07875v3, pp. 4 to 12", note: "Example 1, Theorems 1 to 3, the Kantorovich-Rubinstein dual, weight clipping, Algorithm 1 and the experiments"}
  - {paper: "Gulrajani, Ahmed, Arjovsky, Dumoulin and Courville (2017)", title: "Improved Training of Wasserstein GANs", locator: "arXiv 1704.00028v3, pp. 1, 3 to 7", note: "Proposition 1, the failures of weight clipping, the gradient penalty, Algorithm 1 and the CIFAR-10 Inception scores"}
  - {paper: "Mescheder, Geiger and Nowozin (2018)", title: "Which Training Methods for GANs do actually Converge?", locator: "ICML 2018; arXiv 1801.04406v4, pp. 2 to 8", note: "the Dirac-GAN, Lemmas 2.2 to 3.3, the R1 and R2 regularizers, Theorem 4.1 and the high-resolution experiments"}
family: generative-adversarial-network
inputs: "a batch of noise vectors as a (B, J) array for the generator, and a batch of data points as a (B, D) array for the discriminator"
outputs: "B generated points as a (B, D) array, and one logit per point from the discriminator, a (2B,) array over the fakes and the data"
components: []
inductive_biases: ["the data are the image of a simple noise distribution under a smooth map, the generator", "a classifier that tells data from samples measures how far apart they are, and its gradient points the samples towards the data", "the distance being minimized is the Jensen-Shannon divergence for the original loss and the Wasserstein distance for the critic losses, each through the discriminator's function class", "no likelihood is needed, so the samples can lie on a thin set, a manifold of lower dimension than the data space"]
params: "(J + 1)H + (H + 1)D for a generator with one hidden layer of H units from J noise dimensions to D data dimensions, plus (D + 1)K + K + 1 for a discriminator with one hidden layer of K units"
flops_per_example: "2(JH + HD) for one generated point and 2(DK + K) for each discriminator evaluation, counting a multiply-add as 2 FLOPs and only the matrix products; a training step evaluates the discriminator on a fake and on a data point for each example"
---

A generative adversarial network (GAN) is a generative model trained by a game between two networks.
The generator maps a draw of simple noise, such as a standard normal vector, to a point in the data space, and so defines a distribution, the one its outputs follow.
The discriminator takes a point and returns the probability that it came from the training data and not from the generator.
Goodfellow et al. describe the method as estimating a generative model by training the two at once: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data, with G trained to maximize the probability of D making a mistake; the framework corresponds to a minimax two-player game. {#c-gf-def}
Training needs no likelihood, no Markov chain and no inference network, only samples of the data, samples of the generator and [[al:backpropagation|backpropagation]] through both networks.
The price is that training searches for an equilibrium of a game, not a minimum of a loss, and gradient methods do not always find one.

## In plain words

Goodfellow et al. compare the generator to a team of counterfeiters trying to produce fake currency and use it without detection, and the discriminator to the police trying to detect it; competition drives both teams to improve until the counterfeits are indistinguishable from the genuine articles. {#c-gf-counterfeiters}
In a training step the police look at a pile of real notes and a pile of fakes and get better at telling them apart; then the counterfeiters learn, from the way the police decide, which changes would make their fakes pass.
Nobody ever tells the counterfeiters what a real note looks like.
They learn it only through the police, which is why the method can learn sharp images without a formula for their probability, and also why it can fail: if the police become too good too fast, or the counterfeiters find one fake that fools the current police, the game stalls or cycles.

## Overview

The generator $G_\theta$ maps noise $\mathbf{z} \sim p_z$ to $G_\theta(\mathbf{z})$, and the distribution of its outputs is the model distribution $p_g$.
The discriminator $D_\psi$ maps a point to a probability.
The original objective is the value of a [[al:zero-sum-game|zero-sum game]],

$$ \min_G \max_D V(G, D) = \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}} \log D(\mathbf{x}) + \mathbb{E}_{\mathbf{z} \sim p_z} \log\big(1 - D(G(\mathbf{z}))\big) , $$ {#eq-minimax}

which the discriminator maximizes, as a classifier of data against fakes, and the generator minimizes, the [[al:minimax|minimax]] problem of the title of the section of Goodfellow et al. that introduces it.
For a fixed generator the best discriminator is the ratio of the densities, and against it the generator's criterion is the Jensen-Shannon divergence between the data and the model, up to constants; in the space of all functions, Goodfellow et al. note, a unique solution exists, with G recovering the training data distribution and D equal to 1/2 everywhere, and with multilayer perceptrons the whole system trains by backpropagation with no Markov chains or unrolled approximate inference networks. {#c-gf-unique}

In practice both players are neural networks and the game is played by alternating gradient steps.
Salimans et al. put the difficulty plainly: training requires finding a [[al:nash-equilibrium|Nash equilibrium]] of a non-convex game with continuous, high-dimensional parameters, but GANs are usually trained with gradient descent techniques designed to find a low value of a cost function, which may fail to converge when used to seek one. {#c-sal-nash}
Most of the work after 2014 is about that failure: losses whose gradients do not vanish, architectures that train, distances other than the Jensen-Shannon divergence, and regularizers that make the equilibrium attracting.
Unlike a [[variational-autoencoder]], which maximizes a bound on the likelihood and has an encoder, a GAN has neither: it compares samples, and inference of a code from a data point is not part of the model.

## Components

The notation has vectors as rows; a layer maps $\mathbf{h}$ to $\mathbf{h} W + \mathbf{b}$ with $W$ of shape $(n_\text{in}, n_\text{out})$.
The model of the shape contract is the smallest pair that shows every part: a [[multilayer-perceptron]] with one tanh hidden layer on each side.

**The generator.** Noise of $J$ dimensions, a hidden layer of $H$ units and an output in the data space of $D$ dimensions:

$$ \mathbf{t} = \tanh(\mathbf{z} U_1 + \mathbf{c}_1), \qquad G(\mathbf{z}) = \mathbf{t} U_2 + \mathbf{c}_2, \qquad \mathbf{z} \sim \mathcal{N}(0, I_J) . $$ {#eq-generator}

The output layer is linear so that the fakes can lie anywhere in $\mathbb{R}^D$; image generators end in a tanh to match pixels scaled to $[-1, 1]$.
Goodfellow et al. used generators with a mixture of rectifier linear and sigmoid activations, discriminators with maxout activations and [[dropout]], and fed the noise into the bottom layer of the generator only. {#c-gf-experiments}

**The discriminator.** A hidden layer of $K$ units and one logit:

$$ \mathbf{h} = \tanh(\mathbf{x} W_1 + \mathbf{b}_1), \qquad a(\mathbf{x}) = \mathbf{h} \mathbf{w}_2 + b_2, \qquad D(\mathbf{x}) = \sigma\big(a(\mathbf{x})\big) . $$ {#eq-discriminator}

The implementations work with the logit $a$ and never with $D$ itself, because $\log D = -\operatorname{softplus}(-a)$ and $\log(1 - D) = -\operatorname{softplus}(a)$ are exact for every real $a$, while $\log \sigma(a)$ computed in floating point is $-\infty$ once $\sigma(a)$ rounds to 0.
For the Wasserstein losses the same network, without the sigmoid, is called the critic.

**The discriminator's loss.** Maximizing eq-minimax in $D$ is minimizing the [[m2:cross-entropy|cross-entropy]] of a classifier with label 1 on the data and 0 on the fakes:

$$ L_D = \mathbb{E}_{\mathbf{x}} \operatorname{softplus}\big(-a(\mathbf{x})\big) + \mathbb{E}_{\mathbf{z}} \operatorname{softplus}\big(a(G(\mathbf{z}))\big) . $$ {#eq-dloss}

During the discriminator's step the fakes are held fixed: the step changes $\psi$ only.

**The optimal discriminator.** For a fixed generator,

$$ D^*_G(\mathbf{x}) = \frac{p_{\text{data}}(\mathbf{x})}{p_{\text{data}}(\mathbf{x}) + p_g(\mathbf{x})}, \qquad a^*(\mathbf{x}) = \log\frac{p_{\text{data}}(\mathbf{x})}{p_g(\mathbf{x})} , $$ {#eq-dstar}

so the optimal logit is the log-ratio of the densities, and maximizing $V$ in $D$ is fitting a classifier of the label, real or generated, on an equal mixture of the two by maximum likelihood. {#c-dstar}
The derivation `optimal-discriminator` proves it, and its check fits a logistic regression on samples whose logit approaches the log-ratio.

**The generator's criterion.** Against $D^*$,

$$ C(G) = V(G, D^*_G) = -\log 4 + 2\,\mathrm{JSD}(p_{\text{data}} \,\Vert\, p_g) , $$ {#eq-jsd}

so $C$ is minimized exactly when $p_g = p_{\text{data}}$, with the value $-\log 4$ and $D^* = \tfrac12$ everywhere. {#c-jsd}
This is Theorem 1 of Goodfellow et al.: the global minimum of the virtual training criterion is achieved if and only if $p_g = p_{\text{data}}$, and its value there is $-\log 4$. {#c-jsd-paper}
The derivation `jsd-identity` proves it.
Their Proposition 2 adds that if G and D have enough capacity, the discriminator reaches its optimum at every step and $p_g$ is updated to improve the criterion, then $p_g$ converges to $p_{\text{data}}$; they note that in practice adversarial nets represent a limited family of $p_g$ through the parameters of G and optimize those instead, so the proposition does not cover training. {#c-gf-prop2}

**The generator's loss.** Eq-minimax gives the generator the loss $\mathbb{E}_{\mathbf{z}} \log(1 - D(G(\mathbf{z}))) = -\mathbb{E}_{\mathbf{z}} \operatorname{softplus}(a)$.
Goodfellow et al. observe that early in learning, when G is poor, D can reject samples with high confidence and this term saturates; training G to maximize $\log D(G(\mathbf{z}))$ instead gives the same fixed point of the dynamics and much stronger gradients early in learning. {#c-gf-saturation}
That is the non-saturating loss,

$$ L_G = \mathbb{E}_{\mathbf{z}} \operatorname{softplus}\big(-a(G(\mathbf{z}))\big) = -\mathbb{E}_{\mathbf{z}} \log D(G(\mathbf{z})) , $$ {#eq-gloss}

used in nearly all later GANs with the cross-entropy loss.
In the logit, the minimax loss has the derivative $-\sigma(a) = -D$, which vanishes as $D(G(\mathbf{z})) \to 0$, and the non-saturating loss has $\sigma(a) - 1 = D - 1$, which tends to $-1$; both push the logit of a fake up, and they are stationary at the same points, where the fakes are not distinguished from the data. {#c-saturation}
The derivation `saturation` proves it.

**The Wasserstein critic.** Arjovsky et al. replace the Jensen-Shannon divergence by the Wasserstein-1 distance, the cost of the cheapest transport of one distribution onto the other.
By the Kantorovich-Rubinstein duality it is a supremum over 1-Lipschitz functions $f$ of $\mathbb{E}_{p_{\text{data}}} f - \mathbb{E}_{p_g} f$; over $K$-Lipschitz functions the supremum is $K$ times the distance, and for an optimal critic the gradient of the distance in the generator's parameters is $-\mathbb{E}_{\mathbf{z}} \nabla_\theta f(G_\theta(\mathbf{z}))$. {#c-wgan-duality}
The critic is the discriminator's network without the sigmoid, and the two losses are

$$ L_{\text{critic}} = \mathbb{E}_{\mathbf{z}}\, a(G(\mathbf{z})) - \mathbb{E}_{\mathbf{x}}\, a(\mathbf{x}), \qquad L_G = -\mathbb{E}_{\mathbf{z}}\, a(G(\mathbf{z})) , $$ {#eq-wgan}

with $a$ kept Lipschitz by some constraint, so that $-L_{\text{critic}}$ at the critic's optimum estimates the distance.

**The gradient penalty.** Gulrajani et al. prove that the optimal critic has gradient norm 1 on the straight line between any pair of points matched by the optimal transport, and gradient norm 1 almost everywhere under both distributions. {#c-gp-unit-gradient}
WGAN-GP enforces this softly:

$$ L_{\text{critic}} + \lambda\, \mathbb{E}_{\hat{\mathbf{x}}} \big(\lVert \nabla_{\hat{\mathbf{x}}}\, a(\hat{\mathbf{x}}) \rVert_2 - 1\big)^2, \qquad \hat{\mathbf{x}} = \alpha\mathbf{x} + (1 - \alpha)\, G(\mathbf{z}), \quad \alpha \sim U[0, 1] , $$ {#eq-gp}

one $\alpha$ per pair, with $\lambda = 10$, no batch normalization in the critic and Adam with learning rate 0.0001, $\beta_1 = 0$ and $\beta_2 = 0.9$, five critic steps per generator step. {#c-gp-penalty}
In place of [[batch-normalization]], which makes each example's gradient depend on the rest of the batch, they recommend [[layer-normalization]].
The penalty differentiates the gradient of the critic in its input, so its own gradient in the critic's weights contains second derivatives of the network.

**The R1 penalty.** Mescheder et al. penalize the gradient on the data only:

$$ L_D + \frac{\gamma}{2}\, \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}} \lVert \nabla_{\mathbf{x}}\, a(\mathbf{x}) \rVert^2 , $$ {#eq-r1}

with the non-saturating loss eq-gloss for the generator; their R2 penalizes the gradient on the generator's samples instead. {#c-msch-r1}
At an equilibrium where $p_g = p_{\text{data}}$ the discriminator is constant on the data, so the penalty is zero there and moves no equilibrium; away from it the penalty stops the discriminator from building up a slope that the generator can only chase.

## Shapes and parameters

The shape contract has noise of $J$ dimensions, a generator with $H$ hidden units, data of $D$ dimensions and a discriminator with $K$ hidden units; the example is $J = 2$, $H = 5$, $D = 3$, $K = 4$.
The generator has $JH + H$ and $HD + D$ weights and biases, the discriminator $DK + K$ and $K + 1$, so

$$ P_G = (J + 1)H + (H + 1)D, \qquad P_D = (D + 1)K + K + 1 . $$ {#eq-params}

In the contract these are

{shape:params_gen}

and

{shape:params_dis}

and the example has {shape:params_gen@example} parameters in the generator and {shape:params_dis@example} in the discriminator.
The pair of the mixture test, `toy_mixture`, has {shape:params@toy_mixture} parameters in all, and the pair of the tests in the plane, `toy_plane`, has {shape:params_gen@toy_plane} and {shape:params_dis@toy_plane}.
The noise has no parameters, and the losses and penalties add none.
The derivation `parameter-and-flop-count` proves the counts.

## Computation and cost

A training step runs the generator on $B$ noise vectors and the discriminator on the $B$ fakes and $B$ data points, which in the products costs

$$ C = 2B(JH + HD) + 4B(DK + K) $$ {#eq-flops}

FLOPs, a multiply-add counted as two; the generator and the discriminator of the contract have $P_G = (J + 1)H + (H + 1)D$ and $P_D = (D + 1)K + K + 1$ parameters, and one forward pass of a training step costs $2B(JH + HD) + 4B(DK + K)$ FLOPs. {#c-params}
In the contract,

{shape:flops_forward}

which is {shape:flops_forward@example} FLOPs for the example and {shape:flops_forward@toy_plane} for `toy_plane` at its batch of 128.
The backward passes cost about twice the forward ones.
With $k$ discriminator steps per generator step, as in WGAN with $k = 5$, the discriminator's share grows $k$-fold, and the generator's step needs a forward pass through both networks and a backward pass through both.
The gradient penalties of WGAN-GP and R1 need the gradient of the critic in its input and then the gradient of its norm in the weights: about one more forward and two more backward passes through the critic on the points of the penalty.
Sampling from a trained GAN is one pass of the generator, with no iteration, which is the property later compared with the many network evaluations of a diffusion model.

## Training

Algorithm 1 of Goodfellow et al. alternates $k$ steps on the discriminator, each a gradient step on a minibatch of $m$ noise samples and $m$ data examples, with one step on the generator; they used $k = 1$ and momentum. {#c-gf-algorithm}
The reference's `train(..., n_dis=k)` is this loop with [[adam|Adam]], and `d_step` and `g_step` are its two halves.
WGAN's Algorithm 1 takes $n_{\text{critic}} = 5$ RMSProp steps on the critic, each followed by clipping all its weights to $[-c, c]$, per RMSProp step on the generator, with the defaults learning rate 0.00005, $c = 0.01$ and batch size 64. {#c-wgan-algorithm}
Arjovsky et al. call weight clipping a clearly terrible way to enforce a Lipschitz constraint: a large clipping parameter makes the weights slow to reach their limit and the critic hard to train to optimality, and a small one makes gradients vanish in deep networks or without batch normalization. {#c-wgan-clipping}

### The DCGAN recipe

Radford et al. found architectural rules that let convolutional GANs train across datasets: replace pooling with strided convolutions in the discriminator and fractional-strided convolutions in the generator, use batch normalization in both networks, remove fully connected hidden layers, use ReLU in the generator except for a Tanh output, and LeakyReLU in every layer of the discriminator. {#c-dcgan-guidelines}
Batch normalization helped deep generators start learning and kept the generator from collapsing all samples to a single point, but applying it to every layer caused sample oscillation and instability, avoided by leaving it out of the generator's output layer and the discriminator's input layer. {#c-dcgan-batchnorm}
They trained with minibatches of 128, weights initialized from a normal distribution with standard deviation 0.02, a LeakyReLU slope of 0.2, and Adam with learning rate 0.0002, the suggested 0.001 being too high; the suggested $\beta_1 = 0.9$ caused oscillation and instability, and 0.5 helped stabilize training. {#c-dcgan-adam}
A momentum of 0.5 or less is still the usual choice for GANs; WGAN-GP sets it to 0.

### Mode collapse

Salimans et al. name one of the main failure modes: the generator collapses to a parameter setting where it always emits the same point; near collapse the discriminator's gradients point in similar directions for many points, because it processes each example independently and nothing tells the generator's outputs to become more dissimilar. {#c-sal-collapse}
Goodfellow et al. had already listed it as the Helvetica scenario among the disadvantages of the method: there is no explicit representation of $p_g(\mathbf{x})$, and D must be synchronized well with G during training, since G trained too much without updating D collapses too many values of $\mathbf{z}$ to the same value of $\mathbf{x}$. {#c-gf-disadvantages}
Salimans et al. propose feature matching, which trains the generator to match the expected value of an intermediate layer of the discriminator on the data; minibatch discrimination, which lets the discriminator look at the other examples of a minibatch; historical averaging; one-sided [[label-smoothing]], which softens only the targets of the real examples; and virtual batch normalization. {#c-sal-techniques}
Collapse is partial in practice: a GAN that covers most of the data well and drops some modes has a good sample quality and a poor coverage, and no single number from the discriminator reports which.

### Why gradient steps cycle

Mescheder et al. reduce the problem to two numbers.
In the Dirac-GAN the generator is the point $\theta$, the discriminator is the line $D(x) = \psi x$, and the data are the point 0, so the objective is $L(\theta, \psi) = f(\psi\theta) + f(0)$, with $f(t) = -\log(1 + e^{-t})$ for the original GAN.
The only equilibrium is $\theta = \psi = 0$, where the vector field of the game has the purely imaginary eigenvalues $\pm f'(0) i$ and the continuous dynamics circle the equilibrium, keeping $\theta^2 + \psi^2$ constant; simultaneous gradient steps of size $h$ multiply the distance to the equilibrium by $\sqrt{1 + h^2 f'(0)^2}$ near it; and a penalty $\tfrac{\gamma}{2}\psi^2$ on the discriminator moves the eigenvalues to $-\tfrac{\gamma}{2} \pm \sqrt{\tfrac{\gamma^2}{4} - f'(0)^2}$, with negative real parts for every $\gamma > 0$. {#c-msch-dirac}
The derivation `dirac-gan-dynamics` proves it.
They conclude that unregularized GAN training is not always locally convergent, and neither is WGAN training with a fixed number of critic updates nor WGAN-GP: on the Dirac-GAN simultaneous gradient descent spirals away from the equilibrium and alternating gradient descent oscillates around it in stable cycles. {#c-msch-nonconvergence}
On the Dirac-GAN the penalty $\tfrac{\gamma}{2}\psi^2$ is the R1 penalty of eq-r1, and under their assumptions, for small enough learning rates, simultaneous and alternating gradient descent with R1 or R2 converge locally to the equilibrium, at least at a linear rate. {#c-msch-theorem}

## Inductive biases

The generator assumes that the data are the image of a simple noise distribution under a smooth map, so the model distribution lives on a set of dimension at most $J$; nothing requires that set to fill the data space, which is why GANs can produce sharp samples and why the Jensen-Shannon divergence between the data and the model is often at its maximum.
The discriminator assumes that its function class can tell the two distributions apart, and the distance the game minimizes is only as good as that class: a discriminator that cannot see a difference cannot make the generator fix it.
The convolutional generators and discriminators of DCGAN and its descendants add the biases of a [[convolutional-neural-network]], locality and translation equivariance, to both players.
Goodfellow et al. list the advantages of the framework: Markov chains are never needed, only backpropagation is used to obtain gradients, no inference is needed during learning, and adversarial nets can represent very sharp, even degenerate distributions, while methods based on Markov chains require somewhat blurry distributions so that the chains can mix between modes. {#c-gf-advantages}

## Evidence

Goodfellow et al. estimated the test log-likelihood of their models with a Gaussian Parzen window fitted to samples: 225 ± 2 on MNIST and 2057 ± 26 on the Toronto Face Database, against 138 ± 2 and 1909 ± 66 for a deep belief network, 121 ± 1.6 and 2110 ± 50 for a stacked contractive autoencoder and 214 ± 1.1 and 1890 ± 29 for a deep generative stochastic network; they note that the estimator has a high variance and does not perform well in high-dimensional spaces. {#c-gf-parzen}
Radford et al. used the trained discriminator as a feature extractor: with a linear L2-SVM it reached 82.8% accuracy on CIFAR-10, against 80.6% for one-layer K-means and 84.3% for an Exemplar CNN, and a 22.48% test error on SVHN with 1000 labels, against 28.87% for a supervised CNN of the same architecture. {#c-dcgan-features}
Arithmetic on the generator's input vectors gave images that obeyed it semantically, smiling woman minus neutral woman plus neutral man giving a smiling man, but only when the vectors of three exemplars per concept were averaged; single samples were unstable. {#c-dcgan-arithmetic}

Salimans et al. introduced the Inception score, $\exp\big(\mathbb{E}_{\mathbf{x}} \mathrm{KL}(p(y \mid \mathbf{x}) \,\Vert\, p(y))\big)$ for the label distribution of a pretrained Inception network on generated images, high when each image has a confident label and the labels vary, evaluated on 50k samples. {#c-sal-inception}
On CIFAR-10 it was 11.24 ± .12 for real images and 8.09 ± .07 for their full model, falling to 7.54 with batch instead of virtual batch normalization, 6.86 with historical averaging instead of the labels, 6.83 without label smoothing, 4.36 without labels and 3.87 without minibatch features; human annotators on MTurk told generated CIFAR-10 images from real ones in 78.7% of cases, and in 71.4% when only the top 1% of samples by Inception score were shown. {#c-sal-results}
They caution that the score should be a rough guide for models trained by an independent criterion, since optimizing it directly produces adversarial examples. {#c-sal-caution}

Arjovsky et al. found that the critic's estimate of the Wasserstein distance correlated well with sample quality; that training became unstable with a momentum-based optimizer such as Adam with $\beta_1 > 0$ on the critic, or with high learning rates, so they used RMSProp; and that in no experiment did they see evidence of mode collapse. {#c-wgan-results}
Gulrajani et al. found that weight clipping biased 12-layer critics towards much simpler functions and made their gradients explode or vanish through the layers unless the threshold was tuned, while the gradient penalty did neither. {#c-gp-clipping-failures}
WGAN-GP trained a wide range of architectures with almost no hyperparameter tuning, including 101-layer ResNets and language models with continuous generators, and its ResNet reached Inception scores of 7.86 ± .07 unsupervised and 8.42 ± .10 with labels on CIFAR-10, against 6.16 for DCGAN and 8.59 for SGAN. {#c-gp-results}
Mescheder et al. trained ResNet GANs on all 1000 Imagenet classes and on celebA-HQ at 1024 by 1024 pixels: unregularized training quickly collapsed, and the R1 regularizer made training stable. {#c-msch-celeba}
These comparisons use the Parzen estimate, the Inception score and human judgement, none of which measures how much of the data distribution a model covers; later work moved to the Fréchet Inception distance and to precision and recall of samples, and found many reported gains between GAN variants to be within the spread of hyperparameter searches.

## Limitations

There is no likelihood: a GAN cannot score a data point, compare itself with likelihood models, or detect inputs unlike its training data, and the Parzen estimates of the first paper are known to be poor.
There is no encoder: finding the noise that generates a given image needs an optimization or a separately trained inverse.
Training is the search for an equilibrium of a game, and gradient steps can spiral away from it or cycle around it; the losses of the two players do not decrease monotonically and do not show progress, except for the critic's estimate in a Wasserstein GAN.
Mode collapse makes the model cover part of the data well and the rest not at all, and a sample-quality metric can rise while coverage falls.
The method is sensitive to architecture and hyperparameters: DCGAN found rules by search, and WGAN, WGAN-GP and R1 each add a constraint that changes the game.
Radford et al. note that some instability remains even with their recipe: as models are trained longer they sometimes collapse a subset of filters to a single oscillating mode. {#c-dcgan-instability}

## Variants and descendants

- **Conditional GANs** feed a label or another input $c$ to both players; Goodfellow et al. listed this extension, a conditional model $p(\mathbf{x} \mid c)$ obtained by adding $c$ as an input to both G and D, together with learned approximate inference by an auxiliary network that predicts $\mathbf{z}$ from $\mathbf{x}$. {#c-gf-extensions}
- **DCGAN** (Radford et al., 2015) fixed the convolutional architecture and training settings that most image GANs started from.
- **f-GAN** (Nowozin et al., 2016) generalizes eq-minimax to any f-divergence through its variational lower bound; the original GAN is the Jensen-Shannon case.
- **Wasserstein GAN** (Arjovsky et al., 2017) and **WGAN-GP** (Gulrajani et al., 2017) use the critic of eq-wgan, with weight clipping or the penalty of eq-gp; **spectral normalization** (Miyato et al., 2018) bounds each layer's Lipschitz constant instead.
- **Least squares GAN**, **hinge-loss GANs** and the **relativistic GAN** change the losses of eq-dloss and eq-gloss.
- **Bidirectional GANs** and **adversarially learned inference** train an encoder alongside, against a discriminator of pairs $(\mathbf{x}, \mathbf{z})$.
- **CycleGAN** and **pix2pix** train generators that map images of one domain to another with adversarial losses, unpaired and paired.
- **Progressive GANs**, **BigGAN** and **StyleGAN** scale the method to high-resolution images with growing resolutions, large batches and a mapping network for style; StyleGAN 2 trains with the R1 penalty.
- **Adversarial autoencoders** match the code distribution of an [[autoencoder]] to a prior with a discriminator, and VAE-GAN hybrids add an adversarial loss to the decoder of a VAE.
- Adversarial losses are also used outside generative modelling, in domain adaptation and as perceptual losses for super-resolution and image tokenizers.

## Notable models

The Wasserstein critic's example of two parallel segments is the standard illustration of why a distance matters: for the uniform distributions on the segments $\{0\} \times [0, 1]$ and $\{\theta\} \times [0, 1]$ the Wasserstein distance is $|\theta|$, while for $\theta \ne 0$ the Jensen-Shannon divergence is $\log 2$, both KL divergences are infinite and the total variation is 1, so only the Wasserstein distance goes to 0 with $\theta$ and only it gives a gradient. {#c-parallel-lines}
The derivation `wasserstein-parallel-lines` proves it.
Arjovsky et al. turn the example into a theorem: if the generator is continuous in its parameters so is the Wasserstein distance, and if it is locally Lipschitz and regular the distance is continuous everywhere and differentiable almost everywhere, while both statements are false for the Jensen-Shannon divergence and all the KL divergences. {#c-wgan-continuity}
Among trained models, the DCGAN bedroom generator of Radford et al. (2015) was the first widely shown GAN to produce recognizable scenes at 64 by 64 pixels; BigGAN (2018) generated class-conditional ImageNet images at 512 by 512; and StyleGAN (2018) and StyleGAN 2 (2019) generated faces at 1024 by 1024 that people could rarely tell from photographs.

## Implementations

The NumPy reference `impl/numpy/gan.py` implements eq-generator and eq-discriminator, the gradient of the logit in the input, and five losses: `"minimax"`, eq-dloss with the generator loss of eq-minimax; `"nonsat"`, eq-dloss with eq-gloss; `"wgan"`, eq-wgan; `"wgan_gp"`, eq-wgan with eq-gp; and `"r1"`, eq-dloss plus eq-r1 with eq-gloss.
Every backward pass is written by hand, including the second derivatives that the two penalties need.
It also has eq-dstar, $V(G, D)$ and the Jensen-Shannon divergence on a grid, the Dirac-GAN with simultaneous or alternating steps and the penalty $\tfrac{\gamma}{2}\psi^2$, Adam, RMSProp and weight clipping, the training loop of Algorithm 1 with $k$ discriminator steps, and the counts eq-params and eq-flops.
The PyTorch version `impl/pytorch/gan.py` is an `nn.Module` with the same parameters, autograd for every backward pass and `create_graph=True` for the penalties; `test_gradcheck_through_the_penalty` runs `torch.autograd.gradcheck` through both.
The JAX version `impl/jax/gan.py` is a set of pure functions of a parameter dictionary; the gradient in the input is `jax.grad` of the critic under `jax.vmap`, and `d_grads` and `g_grads` differentiate each player's loss in its own parameters only.
All three load the shared fixture of `fixtures/`: the example config's parameters with nonzero biases, 4 data points, 4 noise vectors and 4 mixing weights, and for each of the five losses the loss and gradients of both players.

### Worked example

Take a data space of two points with $p_{\text{data}} = (\tfrac12, \tfrac12)$ and a generator with $p_g = (0.9, 0.1)$.
By eq-dstar, $D^* = \big(\tfrac{0.5}{1.4}, \tfrac{0.5}{0.6}\big) = (0.3571, 0.8333)$; at the second point, where the generator puts too little mass, the logit is $\log(0.5 / 0.1) = \log 5 = 1.609$.
Then $V(G, D^*) = \tfrac12\log 0.3571 + \tfrac12\log 0.8333 + 0.9\log 0.6429 + 0.1\log 0.1667 = -1.1828$.
The Jensen-Shannon divergence of the two distributions is 0.1017 nats, and $-\log 4 + 2 \cdot 0.1017 = -1.3863 + 0.2035 = -1.1828$, as eq-jsd says; the constant discriminator $D = \tfrac12$ would score only $-\log 4 = -1.3863$.
Now suppose the discriminator gives a fake $D = 0.01$, a logit of $-4.595$: the minimax loss has the derivative $-0.01$ in the logit and the non-saturating loss $-0.99$, 99 times more.
The test `test_worked_example` checks these numbers.

### Pitfalls

> [!pitfall]
> **The fakes in the discriminator's step.** The discriminator's loss must treat the generated points as constants; letting its gradient flow into the generator turns the discriminator's step into a step of the generator against its own loss. The PyTorch version detaches the fakes and the JAX version differentiates in the discriminator's parameters only; `test_shared_fixture_gradients` in PyTorch checks that a discriminator step leaves no gradient on the generator.

> [!pitfall]
> **The minimax generator loss.** Minimizing $\log(1 - D(G(\mathbf{z})))$ gives almost no gradient while the discriminator rejects the fakes, which is the start of every run. Use eq-gloss. `test_minimax_generator_saturates_and_nonsaturating_does_not` trains a discriminator against an initial generator until $D(G(\mathbf{z}))$ is below 0.05 and finds the non-saturating gradient more than 20 times larger.

> [!pitfall]
> **The log of a sigmoid.** Computing $D = \sigma(a)$ and then $\log D$ gives $-\infty$, and a gradient of nan, as soon as the discriminator is sure; compute the losses from the logits with softplus, as the [[m2:cross-entropy|cross-entropy]] with logits does. `test_losses_are_finite_at_extreme_logits` sets the discriminator's bias to ±800 and checks that every loss and gradient is finite.

> [!pitfall]
> **The gradient penalty's interpolation and its second derivatives.** $\hat{\mathbf{x}}$ mixes one data point with one fake by one scalar $\alpha$, not a separate $\alpha$ per coordinate, and the penalty's gradient in the critic's weights runs through the input gradient, so the input gradient must stay in the graph, with `create_graph=True` in PyTorch. `test_gradients_match_finite_differences` checks the hand-written second derivatives of the reference for both penalties, and `test_gradcheck_through_the_penalty` the PyTorch version.

> [!pitfall]
> **Reading the losses as progress.** With the cross-entropy loss and a discriminator that keeps up, the discriminator's loss stays near $\log 4$ whether the samples are good or not, the value of eq-jsd at the equilibrium. `test_generator_learns_a_shifted_plane` checks that a run that learns the data ends with a discriminator loss within 0.05 of $\log 4 = 1.386$; only the Wasserstein critic's loss estimates a distance, which `test_gradient_penalty_critic_estimates_the_wasserstein_distance` checks.

> [!pitfall]
> **Simultaneous steps without a regularizer.** Updating both players from the same point moves them away from the equilibrium, even on a problem with one parameter each. `test_dirac_gan` checks that simultaneous steps on the Dirac-GAN grow the distance by $\sqrt{1 + h^2/4}$ per step, that alternating steps circle, and that the penalty $\tfrac{\gamma}{2}\psi^2$ with $\gamma = 1$ brings both to the equilibrium.

## Experiments

This page has no experiment yet.
Some tests are small experiments.
`test_trained_discriminator_approaches_the_optimal_one` trains the NumPy discriminator alone on $\mathcal{N}(0, 1)$ against fixed fakes from $\mathcal{N}(1.5, 0.7^2)$; after 2000 Adam steps its output is within 0.04 of eq-dstar on a grid from $-2$ to 3.
`test_gradient_penalty_critic_estimates_the_wasserstein_distance` trains a WGAN-GP critic between two Gaussians in the plane whose means are 2 apart; its estimate of the distance lands between 1.8 and 2.5 and its gradient norm on the segment between the means is within 0.3 of 1.
`test_generator_learns_a_gaussian` trains generators with the minimax and the non-saturating losses on $\mathcal{N}(2, 0.5^2)$; both reach a mean within 0.3 of 2.
`test_generator_covers_both_modes` trains on an equal mixture of two narrow Gaussians at $\pm 2$, with and without the R1 penalty, and finds between 35% and 65% of the samples on each side; this mixture is easy, and the test documents that the reference does not collapse on it, not that collapse cannot happen.
`test_generator_learns_a_shifted_plane` trains the non-saturating, WGAN and WGAN-GP losses on a Gaussian in the plane centred at $(2, -1)$; all three reach its mean within 0.5.
In PyTorch `test_training_moves_the_generator_onto_a_gaussian` trains the module on $\mathcal{N}(2, 0.5^2)$, and in JAX `test_dirac_gan` repeats the Dirac-GAN with `jax.lax.scan`.
The checks of the six derivations compare eq-dstar with a grid search and a logistic regression, eq-jsd with grid integrals for pairs of Gaussians and mixtures, the derivatives of the two generator losses with finite differences, the eigenvalues of the Dirac-GAN with finite-difference Jacobians of the reference's update, the parallel-segments example with couplings and a dual critic, and the counts with the arrays and an instrumented forward pass.

## History

Goodfellow et al. posted Generative Adversarial Nets as arXiv 1406.2661 in June 2014, and thanked Yann Dauphin for sharing his Parzen-window evaluation code. {#c-gf-history}
The paper appeared at NIPS 2014.
Training a network against a second network that judges it had precedents, among them Schmidhuber's predictability minimization of the early 1990s and noise-contrastive estimation, which fits a density by classifying data against samples of a fixed noise distribution; the GAN made the noise distribution a learned generator.
Radford, Metz and Chintala's DCGAN (2015) made convolutional GANs train reliably and drew attention to their latent spaces.
Salimans et al. (2016) collected training techniques and introduced the Inception score, and the Fréchet Inception distance followed in 2017.
Arjovsky and Bottou analysed the instability of the original objective in early 2017, and the Wasserstein GAN and WGAN-GP followed within months.
Mescheder, Geiger and Nowozin (2018) separated the training methods that converge locally from those that do not, and their R1 penalty became part of the StyleGAN family.
From about 2021, diffusion models replaced GANs as the usual choice for image generation, while adversarial losses stayed in image tokenizers, super-resolution and the distillation of diffusion models into few-step generators.

## References

- I. J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville and Y. Bengio. Generative Adversarial Nets. arXiv 1406.2661, 2014; NIPS 2014.
- A. Radford, L. Metz and S. Chintala. Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks. arXiv 1511.06434, 2015; ICLR 2016.
- T. Salimans, I. Goodfellow, W. Zaremba, V. Cheung, A. Radford and X. Chen. Improved Techniques for Training GANs. arXiv 1606.03498, 2016; NIPS 2016.
- M. Arjovsky, S. Chintala and L. Bottou. Wasserstein GAN. arXiv 1701.07875, 2017; ICML 2017.
- I. Gulrajani, F. Ahmed, M. Arjovsky, V. Dumoulin and A. Courville. Improved Training of Wasserstein GANs. arXiv 1704.00028, 2017; NIPS 2017.
- L. Mescheder, A. Geiger and S. Nowozin. Which Training Methods for GANs do actually Converge? ICML 2018, PMLR 80.
- S. Nowozin, B. Cseke and R. Tomioka. f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization. NIPS 2016.
- T. Miyato, T. Kataoka, M. Koyama and Y. Yoshida. Spectral Normalization for Generative Adversarial Networks. ICLR 2018.
- T. Karras, S. Laine and T. Aila. A Style-Based Generator Architecture for Generative Adversarial Networks. CVPR 2019.
