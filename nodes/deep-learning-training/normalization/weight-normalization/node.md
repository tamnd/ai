---
tag: 5PC0VR
kind: method
slug: weight-normalization
name: Weight normalization
aliases: [WeightNorm, WN, weight norm]
area: deep-learning-training/normalization
level: core
summary: >-
  Write each weight vector as w = g v / ||v|| and train the length g and the direction v by gradient descent, with no batch statistics and a data-dependent initialization.
purpose: a reparameterization of the weights that speeds up gradient descent without depending on the batch, for recurrent, generative and reinforcement learning models
uses: [orthogonal-projection@m2, singular-value-decomposition@m2, rayleigh-quotient@m2, backpropagation@al]
related_to: [batch-normalization@al, layer-normalization@al, spectral-normalization@al, variational-autoencoder@al, dqn@al, gradient-checking@al]
hyperparameters:
  - {name: init scale of v, meaning: "the standard deviation of the normal distribution that v is drawn from", typical: "0.05 in Salimans and Kingma"}
  - {name: init batch, meaning: "the single minibatch used to set g and b before training", typical: "one ordinary training minibatch; it must hold at least two examples"}
  - {name: parameterization of g, meaning: "g learned directly or as exp(s)", typical: "g directly; Salimans and Kingma found no advantage in the log scale"}
  - {name: mean-only BN, meaning: "whether the minibatch mean of each pre-activation is subtracted as well", typical: "on for convolutional classifiers in Salimans and Kingma; off for recurrent models and DQN"}
  - {name: learning rate, meaning: "the step size of the optimizer", typical: "0.003 with Adam on CIFAR-10 in Salimans and Kingma; the method tolerates a wide range"}
cost:
  train: "one norm per weight vector per step, O(number of weights), against O(number of pre-activations) for batch normalization; no running statistics and no memory per example"
  inference: "none: w = g v / ||v|| can be computed once and stored as an ordinary weight"
history:
  year: 2016
  people: [Tim Salimans, Diederik P. Kingma]
  note: "Salimans and Kingma, arXiv 1602.07868v3 (4 Jun 2016), from OpenAI."
prereqs: [backpropagation@al, batch-normalization@al, stochastic-gradient-descent@m2, orthogonal-projection@m2]
same_as:
  wikidata: null
  wikipedia: null
  arxiv: ["1602.07868"]
  al: weight-normalization
sources:
  - {paper: "Salimans and Kingma (2016)", title: "Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks", locator: "the abstract, pp. 1 to 8, Figures 2 and 6", note: "the reparameterization, the gradients, the norm growth, the initialization, mean-only batch normalization and the experiments"}
  - {paper: "Miyato et al. (2018)", title: "Spectral Normalization for Generative Adversarial Networks", locator: "pp. 3 to 9 and 15, Table 2", note: "spectral normalization by power iteration, and its comparison with weight normalization in GAN discriminators"}
  - {paper: "Qiao et al. (2019)", title: "Micro-Batch Training with Batch-Channel Normalization and Weight Standardization", locator: "p. 2", note: "weight standardization as a relative of weight normalization"}
---

Weight normalization (WN) changes how a network stores its weights, not what it computes.
Each weight vector $w$ is written as a length $g$ times a unit direction $v/\lVert v\rVert$, and gradient descent runs on $g$ and $v$.
The output of the network is unchanged, but the gradients are different, and Salimans and Kingma (2016) found that training is faster.
Unlike [[batch-normalization]], the method uses no statistics of the batch, so it works for recurrent networks, generative models and reinforcement learning.
It comes with a data-dependent initialization and an optional mean-only batch normalization.
Spectral normalization, a later relative, divides a whole matrix by its largest singular value instead.

## In plain words

A neuron multiplies its inputs by a vector of weights.
That vector has a direction and a length.

Weight normalization stores them apart.
One number, $g$, is the length.
A vector $v$ gives the direction; only $v/\lVert v\rVert$ matters, so the length of $v$ itself is free.
Gradient descent then moves the length and the direction on their own.

Two things follow.
The gradient for $v$ is always at right angles to $v$, so a plain step can only make $v$ longer.
A longer $v$ turns more slowly.
So a step size that is too large makes $v$ grow until the steps are moderate.

The method does not fix the scale of each layer's outputs as batch normalization does.
So the paper adds an initialization: run one batch through the network and set $g$ and the bias so that every unit starts with mean 0 and variance 1 on that batch.

## Setting

Plain gradient descent is fast or slow depending on the curvature of the loss, and the curvature depends on the parameterization: two ways to write the same model can differ in how easy they are to train. {#c-curvature}
One line of work keeps plain first-order steps and changes the parameterization so that the gradients look more like whitened natural gradients; batch normalization is in this line, and so is weight normalization. {#c-natural-gradient}

A neuron computes {#c-neuron}

$$ y = \phi(w\cdot x + b), $$ {#eq-neuron}

with a weight vector $w \in \mathbb{R}^k$, a bias $b$, an input $x \in \mathbb{R}^k$ and a nonlinearity $\phi$.
Salimans and Kingma present weight normalization as a reparameterization of $w$ that separates its length from its direction, and say it improves the conditioning of the problem and speeds up gradient descent. {#c-abstract}
It was inspired by batch normalization but adds no dependence between the examples of a batch, so it applies to LSTMs and to noise-sensitive settings such as reinforcement learning and generative models. {#c-no-batch-dependence}

## The method

**The reparameterization.**
Each weight vector is written as {#c-wn}

$$ w = \frac{g}{\lVert v\rVert}\,v , $$ {#eq-wn}

with $v \in \mathbb{R}^k$ and a scalar $g$, and gradient descent runs on $v$ and $g$.
Then $\lVert w\rVert = |g|$ whatever $v$ is.
Earlier work had normalized weight vectors too, but optimized in $w$ and only rescaled after each step; here the optimizer never sees $w$. {#c-earlier-work}
A log scale $g = e^s$ was no better on test data and slightly slower to optimize. {#c-log-scale}
For a layer, $v$ is a matrix with one row per output unit and $g$ is a vector, with one length per row.

**The gradients.**
With $\nabla_w L$ the usual gradient in $w$, the gradients in the new parameters are

$$ \nabla_g L = \frac{\nabla_w L\cdot v}{\lVert v\rVert}, \qquad \nabla_v L = \frac{g}{\lVert v\rVert}\nabla_w L - \frac{g\,\nabla_g L}{\lVert v\rVert^2}\,v . $$ {#eq-grad}

They can also be written as {#c-proj}

$$ \nabla_v L = \frac{g}{\lVert v\rVert} M_w \nabla_w L, \qquad M_w = I - \frac{w w^\top}{\lVert w\rVert^2} . $$ {#eq-proj}

So backpropagation runs as usual up to $\nabla_w L$, and two lines turn it into $\nabla_g L$ and $\nabla_v L$.
These formulas do not involve the batch size, so the overhead is small. {#c-batch-size-free-grad}
Salimans and Kingma read eq-proj as two effects: the gradient is scaled by $g/\lVert v\rVert$ and projected away from $w$, and they argue both bring the gradient covariance closer to the identity. {#c-two-effects}
If $C$ is the covariance of the gradient in $w$, the covariance in $v$ is $(g^2/\lVert v\rVert^2) M_w C M_w$; they report that $w$ is often close to a dominant eigenvector of $C$, so removing it may help. {#c-covariance}

**Relation to batch normalization.**
Batch normalization divides the pre-activation $t = v\cdot x$ by its standard deviation over the batch.
For one layer with whitened inputs, $t$ has mean 0 and standard deviation $\lVert v\rVert$, so the two methods coincide. {#c-bn-equivalence}
In deeper networks they differ.
Salimans and Kingma argue WN is a cheaper and less noisy stand-in: a convolutional layer has far fewer weights than pre-activations, and $\lVert v\rVert$ is not random. {#c-cheaper}

**Data-dependent initialization.**
Batch normalization fixes the scale of each layer's features; WN does not, so the initialization matters. {#c-init-needed}
The entries of $v$ are drawn from a normal with mean 0 and standard deviation 0.05. {#c-init-v}
Then one forward pass over one minibatch computes, at each neuron in turn, {#c-init-rule}

$$ t = \frac{v\cdot x}{\lVert v\rVert}, \qquad g \leftarrow \frac{1}{\sigma[t]}, \qquad b \leftarrow -\frac{\mu[t]}{\sigma[t]} , $$ {#eq-init}

where $\mu[t]$ and $\sigma[t]$ are the mean and the standard deviation of $t$ over the minibatch.
The pass continues to the next layer with the new $g$ and $b$.
This needs a forward pass over a batch, so it applies only where batch normalization would; RNNs and LSTMs need a standard initialization. {#c-init-recurrent}
Other authors proposed similar data-based initializations for the usual parameterization around the same time. {#c-init-prior}

**Mean-only batch normalization.**
WN fixes the scale of a unit roughly but not its mean.
So the paper also tries subtracting the minibatch mean without dividing by the standard deviation: {#c-mean-only}

$$ t = w\cdot x, \qquad \tilde{t} = t - \mu[t] + b, \qquad y = \phi(\tilde{t}) , $$ {#eq-mean-only}

with a running average of $\mu[t]$ at test time.
The backward pass centres the gradient: {#c-mean-only-grad}

$$ \nabla_t L = \nabla_{\tilde{t}} L - \mu[\nabla_{\tilde{t}} L] . $$ {#eq-mean-only-grad}

Salimans and Kingma argue that the noise of a minibatch mean is roughly normal, with much lighter tails than the noise from a minibatch variance. {#c-mean-only-noise}

## Derivation

Four facts carry the method.
Each is proved in a derivation file, with its assumptions and a numeric check.

**The gradients** (derivation `gradient`).
Write $\hat{v} = v/\lVert v\rVert$.
The chain rule through $w = g\hat{v}$ gives eq-grad. {#c-grad}
The Jacobian of $\hat{v}$ is $(I - \hat{v}\hat{v}^\top)/\lVert v\rVert$, and $\hat{v}\hat{v}^\top = ww^\top/\lVert w\rVert^2$ when $g \neq 0$, which gives eq-proj.
Since $M_w$ projects onto the complement of $v$, the gradient in $v$ is orthogonal to $v$. {#c-grad-orth}
The check compares eq-grad with central finite differences in 100 random cases, which is gradient checking.

**Norm growth** (derivation `norm-growth`).
For a plain step $v' = v - \eta\nabla_v L$, the cross term $v\cdot\nabla_v L$ is 0, so {#c-norm-growth}

$$ \lVert v'\rVert^2 = \lVert v\rVert^2 + \eta^2\lVert\nabla_v L\rVert^2 . $$ {#eq-norm-growth}

With $c = \eta\lVert\nabla_v L\rVert/\lVert v\rVert$ this is the paper's $\lVert v'\rVert = \sqrt{1+c^2}\,\lVert v\rVert$.
At fixed $w$, scaling $v$ by $s$ divides $\nabla_v L$ by $s$. {#c-scale-step}
So noisy or large steps grow $\lVert v\rVert$, which shrinks the later steps, and the effective learning rate settles.
Salimans and Kingma report that WN works over a much wider range of learning rates for this reason. {#c-lr-robust}
With momentum, the step is not a multiple of the current gradient, so it need not be orthogonal to $v$ and the identity fails. {#c-momentum-breaks}
The paper says the same of momentum and Adam, but reports the same behaviour qualitatively. {#c-adam-qualitative}
The check verifies eq-norm-growth at every step of 400-step runs and finds non-orthogonal steps under momentum.

**Initialization moments** (derivation `init-moments`).
After eq-init, $w\cdot x + b = (t - \mu[t])/\sigma[t]$, so the pre-activations have mean 0 and biased variance 1 on the initialization batch. {#c-init-moments}
This holds only for that batch; a new batch has other moments. {#c-init-batch-only}
With the unbiased standard deviation of $m$ values, the variance is $(m-1)/m$ instead. {#c-init-unbiased}

**Power iteration** (derivation `power-iteration`).
Spectral normalization (below) needs the largest singular value $\sigma_1$ of a matrix $W$.
From a start $u_0$, power iteration repeats

$$ v \leftarrow \frac{W^\top u}{\lVert W^\top u\rVert}, \qquad u \leftarrow \frac{W v}{\lVert W v\rVert}, \qquad \hat\sigma = u^\top W v . $$ {#eq-power}

After $k$ rounds, $\hat\sigma_k^2$ is a Rayleigh quotient of $W^\top W$, which gives {#c-power-converges}

$$ 0 \leq \sigma_1^2 - \hat\sigma_k^2 \leq \sigma_1^2\Big(\frac{\sigma_2}{\sigma_1}\Big)^{4k-2}\tan^2\theta_0 , $$ {#eq-power-bound}

where $\theta_0$ is the angle between $u_0$ and the top left singular vector.
Miyato et al. state the same conditions: a random start, no tie for the top singular value, and a start not orthogonal to the first left singular vector. {#c-power-conditions}
The estimate never exceeds $\sigma_1$, so $W/\hat\sigma$ has spectral norm at least 1 until the iteration converges. {#c-power-underestimates}
The check tests the bound on 80 matrices with known singular values and compares 400 rounds with `numpy.linalg.svd`.

## Worked example

All numbers in this section are my own computations, and the tests `test_worked_example_gradient`, `test_worked_example_init` and `test_worked_example_power` check them.

**Gradient and growth.**
Take $v = (3, 4)$ and $g = 2$, so $\lVert v\rVert = 5$ and $w = (1.2, 1.6)$.
Let $\nabla_w L = (1, 0)$.
Then $\nabla_g L = 3/5 = 0.6$ and $\nabla_v L = \frac{2}{5}(1, 0) - \frac{2 \cdot 0.6}{25}(3, 4) = (0.256, -0.192)$.
Its dot product with $v$ is $0.768 - 0.768 = 0$.
A step with $\eta = 1$ gives $v' = (2.744, 4.192)$, and $\lVert v'\rVert^2 = 25.1024 = 25 + 0.32^2$, since $\lVert\nabla_v L\rVert = 0.32$.

**Initialization.**
Take $v = (3, 4)$ and a batch of four inputs $(0,0)$, $(5,0)$, $(0,5)$ and $(5,5)$.
Then $t = (0, 3, 4, 7)$, $\mu[t] = 3.5$ and $\sigma[t] = 2.5$, so eq-init sets $g = 0.4$ and $b = -1.4$.
The pre-activations become $(-1.4, -0.2, 0.2, 1.4)$, with mean 0 and variance $(1.96 + 0.04 + 0.04 + 1.96)/4 = 1$.

**Power iteration.**
Take $W = \mathrm{diag}(3, 1)$ and $u_0 = (1, 1)/\sqrt{2}$.
One round gives $\hat\sigma_1^2 = 8.2$, so $\hat\sigma_1 \approx 2.86$ against $\sigma_1 = 3$.
In general $9 - \hat\sigma_k^2 = 8/(9^{2k-1} + 1)$, so two rounds give $\hat\sigma_2^2 = 6562/730 \approx 8.989$.
Here $\sigma_2/\sigma_1 = 1/3$ and $\tan\theta_0 = 1$, and the bound of eq-power-bound is $9 \cdot 9^{-(2k-1)}$, just above the exact gap.

## Hyperparameters

**Init scale of v.**
0.05 in the paper.
The output does not depend on $\lVert v\rVert$, but the step does, through $1/\lVert v\rVert$ in eq-grad, so a small start gives large early turns of the direction.

**Init batch.**
One ordinary minibatch.
With eq-init, every unit must have non-constant pre-activations on it, which needs at least two distinct examples.

**g.**
Learned directly; the log scale gave no gain, as above.

**Mean-only BN.**
On for the best CIFAR-10 result below, off where batch statistics hurt, as in DQN.

**Learning rate.**
On CIFAR-10 each variant got its best rate from {0.0003, 0.001, 0.003, 0.01}, with Adam for 200 epochs at minibatch size 100. {#c-cifar-setup}
The norm growth makes WN tolerant of the choice.

## Cost

Per step, WN computes one norm per weight vector and two extra vector operations in the backward pass, which is $O(\text{number of weights})$.
Batch normalization works on the pre-activations, which in a convolutional layer are far more numerous.
In the CIFAR-10 runs, batch normalization made training about 16% slower than the plain parameterization, while WN was not noticeably slower. {#c-bn-slower}
At inference, $w = g v/\lVert v\rVert$ can be computed once, so there is no cost.
Mean-only batch normalization adds one mean per unit per batch and a running average.

## Evidence

All numbers below are from the cited papers.

**CIFAR-10 classification.**
WN and batch normalization both trained much faster than the plain parameterization; batch normalization made slightly more progress per epoch early, and later the two kept pace. {#c-cifar-curves}
The test errors, without data augmentation, were 8.52% with mean-only batch normalization, 8.46% with WN, 8.43% with the plain parameterization, 8.05% with batch normalization and 7.31% with WN plus mean-only batch normalization. {#c-cifar-table}
So WN alone matched the plain parameterization in final error and won in speed.
Salimans and Kingma suggest the combination beats full batch normalization because its noise is close to Gaussian while that of full batch normalization has high kurtosis. {#c-kurtosis}

**Generative models.**
In convolutional [[variational-autoencoder]]s on MNIST and CIFAR-10, WN gave lower variance across seeds and a better final lower bound. {#c-cvae}
In DRAW, an LSTM model where batch normalization does not apply easily, WN on all weight vectors sped up convergence without retuning the initialization or the learning rate. {#c-draw}

**Reinforcement learning.**
For [[dqn]] on Atari games, the authors could not make batch normalization work without an impractically large minibatch. {#c-dqn-bn}
The evaluation scores, each the maximum over training, with the plain parameterization and with WN were 410 and 403 on Breakout, 1,250 and 1,448 on Enduro, 7,188 and 7,375 on Seaquest, and 1,779 and 2,179 on Space Invaders. {#c-dqn}

The authors concluded that WN was their default for new architectures. {#c-default}

**GAN discriminators.**
Miyato et al. compared WN, without its $g$, with spectral normalization in GAN discriminators. {#c-no-gamma}
With a standard CNN, the inception score was 6.84 with WN and 7.42 with spectral normalization on CIFAR-10, and 7.16 against 8.28 on STL-10; the CIFAR-10 FID was 34.7 against 29.3. {#c-sngan-table}
The singular values of the first five WN layers concentrated on a few components, while spectral normalization spread them. {#c-singular-values}

These results are from 2016 to 2018, with models of that era and one paper's runs per setting.

## Limitations and failure modes

**No fixed scale per layer.**
WN normalizes the weights, not the activations, so the scale and the mean of each unit can drift during training.
The initialization fixes them for one batch only.

**The initialization needs a batch.**
Models with recursion fall back to standard initializations.

**The norm argument is for plain steps.**
With momentum or Adam the growth of $\lVert v\rVert$ is not guaranteed.

**Rank in Lipschitz-constrained networks.**
With $g = 1$, the rows of $W$ have unit norm, so the squared singular values sum to the number of rows, and $\lVert Wh\rVert$ for a unit $h$ is largest only when $W$ has rank one. {#c-wn-rank}
Miyato et al. argue that this pushes a WN discriminator toward few features. {#c-wn-features}

**Final accuracy.**
On CIFAR-10, WN alone did not beat the plain parameterization in final test error; the gain was speed.

## Variants

**Spectral normalization.**
It divides each weight matrix by its spectral norm, so each layer has Lipschitz constant 1 and, with 1-Lipschitz activations, so does the network. {#c-sn-def}

$$ \bar{W}_{\mathrm{SN}} = \frac{W}{\sigma(W)} . $$ {#eq-sn}

An SVD per step would be heavy, so $\sigma(W)$ is estimated by eq-power. {#c-sn-power}
The vector $u$ from each step starts the next, since $W$ changes little per update, and one round per step was enough in practice. {#c-recycle}
It costs about 110 to 120% of WN's time on CIFAR-10. {#c-sn-time}
In PyTorch, `torch.nn.utils.parametrizations.spectral_norm` does the same; `SpectralNormLinear` in the reference matches it.

**Weight standardization.**
It drops the learnable length and standardizes each weight vector to zero mean and unit variance; with group normalization it narrows the gap to batch normalization. {#c-ws}

**Layer normalization** also avoids the batch, but it normalizes the activations of each example over the units of a layer, at a cost per example; WN touches only the weights.

**Mean-only batch normalization** is described above.
Data-dependent initialization can also be used without WN, with plain training in $w$ afterwards.

## When to use it

Use WN where batch statistics are unavailable or harmful: recurrent networks, small batches, reinforcement learning, and generative models.
Pair it with the data-dependent initialization when a forward pass over a batch is possible.
For convolutional classifiers with large batches, batch normalization was more accurate in the results above, and WN with mean-only batch normalization was better still.
For a Lipschitz bound, as in GAN discriminators, use spectral normalization.

## Pitfalls

> [!pitfall]
> **Normalizing the wrong axis.** In a layer, each output unit has its own $v$, so the norm is per row of the $(\text{out}, \text{in})$ matrix. PyTorch's `weight_norm` uses `dim=0` for this. `test_against_torch` in both implementations compares with it.

> [!pitfall]
> **Dropping the projection.** Treating $\lVert v\rVert$ as a constant in backpropagation gives $(g/\lVert v\rVert)\nabla_w L$ without the projection. `test_gradient_finite_differences` checks eq-grad, and `test_projection_orthogonality_and_growth` checks that $\nabla_v L \perp v$.

> [!pitfall]
> **Initializing with the wrong moments.** eq-init uses the biased standard deviation. `test_init_moments` checks mean 0 and variance 1 on the batch, and `test_worked_example_init` checks $g = 0.4$.

> [!pitfall]
> **Running power iteration from scratch.** One round from a random start can be far off; the reference keeps $u$ between steps. `test_power_iteration_recycle` checks that the recycled estimate converges.

> [!pitfall]
> **Expecting spectral norm exactly 1.** The estimate is at most $\sigma_1$, so the normalized matrix can have norm slightly above 1. `test_power_iteration_against_svd` checks the inequality.

## History

Tim Salimans and Diederik Kingma of OpenAI posted weight normalization on arXiv in 2016. {#c-hist-wn}
Miyato, Kataoka, Koyama and Yoshida posted spectral normalization in February 2018 and published it at ICLR 2018. {#c-hist-sn}
Qiao et al. proposed weight standardization in 2019.
