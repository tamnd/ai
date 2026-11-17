---
tag: 3SJCNH
kind: concept
slug: neural-tangent-kernel
name: Neural tangent kernel
aliases: [NTK, tangent kernel, empirical NTK, lazy training, kernel regime, linearized network, CNTK, convolutional NTK]
area: deep-learning-training/training-dynamics
level: core
summary: >-
  The inner product of a network's parameter gradients at two inputs; in the infinite-width NTK limit it stays fixed during training, so gradient descent becomes kernel regression.
uses: [backpropagation@al, relu@al, eigendecomposition@m2, law-of-large-numbers@m2, central-limit-theorem@m2]
related_to: [gaussian-process@al, kernel-trick@al, kernel-ridge-regression@al, double-descent@al, he-initialization@al, xavier-initialization@al]
history:
  year: 2018
  people: [Arthur Jacot, Franck Gabriel, Clément Hongler]
  note: "Jacot, Gabriel and Hongler (NIPS 2018) defined the NTK and proved its infinite-width limit; Lee et al. (NeurIPS 2019) showed wide networks train like their linearization; Chizat, Oyallon and Bach (NeurIPS 2019) traced lazy training to scaling; Arora et al. (NeurIPS 2019) computed the convolutional NTK exactly."
prereqs: [backpropagation@al, kernel-ridge-regression@al, eigendecomposition@m2, stochastic-gradient-descent@m2]
same_as:
  al: neural-tangent-kernel
  wikipedia: Neural_tangent_kernel
  arxiv: ["1806.07572", "1902.06720", "1812.07956", "1904.11955"]
sources:
  - {paper: "Jacot, Gabriel and Hongler (2018)", title: "Neural Tangent Kernel: Convergence and Generalization in Neural Networks", locator: "arXiv 1806.07572v4, Sections 2 to 6", note: "the NTK parametrization, kernel gradient descent, the deterministic limit at initialization, constancy during training, positive-definiteness, least squares and early stopping, the numerical study"}
  - {paper: "Lee, Xiao, Schoenholz, Bahri, Novak, Sohl-Dickstein and Pennington (2019)", title: "Wide Neural Networks of Any Depth Evolve as Linear Models Under Gradient Descent", locator: "arXiv 1902.06720v4, Sections 1 to 3 and Appendix D", note: "the linearized model, its closed-form dynamics, Theorem 2.1, the parameterization remark, the analytic ReLU kernel"}
  - {paper: "Chizat, Oyallon and Bach (2019)", title: "On Lazy Training in Differentiable Programming", locator: "arXiv 1812.07956v5, Sections 1 to 4", note: "lazy training as a consequence of scale, the O(1/alpha) bounds, the mean-field scaling, the CIFAR-10 experiments"}
  - {paper: "Arora, Du, Hu, Li, Salakhutdinov and Wang (2019)", title: "On Exact Computation with an Infinitely Wide Neural Net", locator: "arXiv 1904.11955v2, Sections 1 and 6", note: "the exact convolutional NTK and its CIFAR-10 accuracy"}
---

The neural tangent kernel (NTK) of a network is the inner product of its parameter gradients at two inputs.
It says how a gradient step taken to fit one input moves the output at another.
When the network is very wide and scaled the right way, this kernel barely changes during training, so training the network is the same as kernel regression with a fixed kernel.

## In plain words

A gradient step changes every weight a little.
Those changes move the network's output at the training point you stepped on, but they also move the output at every other input.
How much the output at $x'$ moves when you push on the output at $x$ is a number, $\Theta(x, x')$, and that number is the neural tangent kernel.

For a general network this number changes as the weights change, so it gives no shortcut.
In a very wide network each weight moves only a tiny amount, the gradients hardly change, and the kernel stays what it was at the start.
Training then follows a linear equation that can be solved in closed form: the fit improves fastest along the directions the kernel favours most, and the trained network is a kernel regression predictor.

## The idea

Take a network $f(x; \theta)$ with $P$ parameters and train it on inputs $x_1, \dots, x_n$ with targets $y_i$ by gradient descent.
By the chain rule, a small change $\delta\theta$ moves the output at $x$ by $\nabla_\theta f(x)^\top \delta\theta$.
A gradient step on the loss at $x_i$ sets $\delta\theta \propto -\nabla_\theta f(x_i)$, so it moves the output at $x$ in proportion to $\nabla_\theta f(x)^\top \nabla_\theta f(x_i)$.
That inner product is the NTK.
Jacot et al. show that during gradient descent the network function evolves along the negative kernel gradient of the cost with respect to it. {#c-ntk-def}

If $f$ were linear in $\theta$, its gradients would not depend on $\theta$ and the kernel would be fixed.
Gradient descent on such a model is kernel gradient descent with its tangent kernel, and as the number of parameters grows that random kernel tends to a fixed one by the law of large numbers. {#c-linear-model-kernel}
A real network is not linear, so its NTK depends on the parameters: it is random at initialization and varies during training. {#c-random-at-init}
The main result of Jacot et al. is that, in a suitable infinite-width limit, both kinds of randomness go away.
The NTK at initialization tends to a deterministic kernel, and it stays at that value while the network trains.
So a wide network behaves like a linear model in its parameters, and its training can be studied in function space rather than parameter space.

The limit needs a particular scaling.
In the NTK parametrization each layer multiplies its input by $1/\sqrt{n_\ell}$, with $n_\ell$ the width of the layer below, and all parameters start as independent $\mathcal{N}(0, 1)$. {#c-ntk-param}
Jacot et al. stress that these factors are what make the network behave consistently as the hidden widths grow. {#c-factors-key}
The sibling page on the [maximal update parametrization](../../hyperparameter-transfer/maximal-update-parametrization/node.md) explains why the same scaling also stops the hidden features from learning.

## Formal statement

Let $f(x; \theta) \in \mathbb{R}$ be a network with parameters $\theta \in \mathbb{R}^P$.
Its neural tangent kernel at $\theta$ is

$$ \Theta_\theta(x, x') = \sum_{p=1}^{P} \frac{\partial f(x; \theta)}{\partial \theta_p} \frac{\partial f(x'; \theta)}{\partial \theta_p} = \nabla_\theta f(x; \theta)^\top \nabla_\theta f(x'; \theta) . $$ {#eq-ntk}

On training inputs $X = (x_1, \dots, x_n)$, let $J \in \mathbb{R}^{n \times P}$ be the Jacobian with rows $\nabla_\theta f(x_i; \theta)^\top$.
The empirical NTK is the $n \times n$ Gram matrix $\hat\Theta = J J^\top$, symmetric and positive semidefinite.

**Kernel gradient flow.** Train on $L(\theta) = \frac{1}{2n} \sum_i (f(x_i; \theta) - y_i)^2$ by gradient flow $\dot\theta = -\eta \nabla L$, and write $r = f(X; \theta) - y$.
Then $\dot r = -\frac{\eta}{n} \hat\Theta_{\theta(t)}\, r$.
If the kernel is constant, $\hat\Theta_{\theta(t)} = \hat\Theta$, this integrates to

$$ r(t) = \exp\!\left( -\frac{\eta\, \hat\Theta\, t}{n} \right) r(0) , $$ {#eq-residual-flow}

so the residual's component along an eigenvector of $\hat\Theta$ with eigenvalue $\lambda_i$ decays like $e^{-\eta \lambda_i t / n}$. {#c-residual-flow}
The component in the null space of $\hat\Theta$ does not move. {#c-eigen-rates}
Gradient descent with step $\eta$ gives exactly $r_k = (I - \eta \hat\Theta / n)^k r_0$, which converges on every positive eigen-direction when $\eta < 2n / \lambda_{\max}$ and tends to the flow as $\eta \to 0$ at fixed $\eta k$. {#c-gd-discrete}
The derivation is [kernel-flow](derivations/kernel-flow.md).

**The linearized model.** The kernel is exactly constant for the first-order Taylor expansion around the initial parameters $\theta_0$,

$$ f_{\text{lin}}(x; \theta) = f(x; \theta_0) + \nabla_\theta f(x; \theta_0)^\top (\theta - \theta_0) , $$ {#eq-flin}

which is linear in $\theta$ and has the NTK of the network at $\theta_0$.
Gradient descent on it with $\omega = \theta - \theta_0$ is

$$ \omega_{k+1} = \omega_k - \frac{\eta}{n} J^\top r_k , \qquad r_{k+1} = \left(I - \frac{\eta}{n} \hat\Theta\right) r_k . $$ {#eq-gd-step}

Lee et al. argue, and prove for the squared loss, that in the infinite-width limit the learning dynamics of wide networks are governed by this linear model. {#c-lee-linear}

**The infinite-width limit.** For a depth-$L$ network in the NTK parametrization with a Lipschitz nonlinearity, as the hidden widths go to infinity the NTK at initialization converges in probability to a deterministic kernel $\Theta_\infty$, given by a recursion over the layers. {#c-thm-deterministic}
For a Lipschitz, twice differentiable nonlinearity with a bounded second derivative, and any horizon $T$ over which the training direction stays stochastically bounded, the NTK stays at $\Theta_\infty$ uniformly on $[0, T]$. {#c-thm-constant}
For a non-polynomial Lipschitz nonlinearity and $L \ge 2$, $\Theta_\infty$ is positive definite on the unit sphere, so the flow drives the training residual to zero. {#c-pd-sphere}

For the squared loss with $\lambda_{\min}(\Theta_\infty) > 0$ and $\eta$ below $2 / (\lambda_{\min} + \lambda_{\max})$, Lee et al. prove a quantitative version at width $n$: the gap between the network and $f_{\text{lin}}$, the weight change divided by $\sqrt n$, and the change of the empirical NTK are all $O(n^{-1/2})$ uniformly in time. {#c-lee-thm}

**The two-layer ReLU kernel.** Take

$$ f(x) = \frac{1}{\sqrt m} \sum_{j=1}^{m} a_j\, \sigma(w_j^\top x), \qquad \sigma(u) = \max(u, 0), \quad w_j \sim \mathcal{N}(0, I_d),\ a_j \sim \mathcal{N}(0, 1) . $$ {#eq-two-layer}

Its gradients are

$$ \frac{\partial f}{\partial a_j} = \frac{\sigma(w_j^\top x)}{\sqrt m}, \qquad \frac{\partial f}{\partial w_j} = \frac{a_j\, \sigma'(w_j^\top x)\, x}{\sqrt m} , $$ {#eq-jacobian}

so the empirical NTK is a mean of $m$ independent terms, and as $m \to \infty$ it tends to

$$ \Theta_\infty(x, x') = \underbrace{x^\top x' \, \frac{\pi - \varphi}{2\pi}}_{\text{first layer}} + \underbrace{\lVert x \rVert \lVert x' \rVert \, \frac{\sin\varphi + (\pi - \varphi)\cos\varphi}{2\pi}}_{\text{second layer}} , $$ {#eq-relu-ntk}

with $\varphi$ the angle between $x$ and $x'$. {#c-relu-ntk-limit}
Each term is unbiased, so the root-mean-square error of the empirical kernel is exactly proportional to $1/\sqrt m$. {#c-relu-ntk-rate}
The derivation is [relu-ntk](derivations/relu-ntk.md).

**Lazy weights.** Along gradient descent on $f_{\text{lin}}$ with $\eta \le n / \lambda_{\max}$, the displacement obeys $\lVert \omega_k \rVert^2 \le r_0^\top \hat\Theta^{-1} r_0 \le \lVert r_0 \rVert^2 / \lambda_{\min}$. {#c-lazy-bound}
In the network of eq-two-layer, $\lVert \theta_0 \rVert^2$ has mean $m(d+1)$ while that bound stays of order 1, so the relative change

$$ \frac{\lVert \theta_T - \theta_0 \rVert}{\lVert \theta_0 \rVert} = O\!\left( \frac{1}{\sqrt m} \right) . $$ {#eq-relative-change}

So the weights of a wide network in this regime change by a vanishing fraction of their size. {#c-lazy-relative}
The derivation is [lazy-weights](derivations/lazy-weights.md).

> [!note]
> Every number on this page that is not attributed to a paper comes from our own runs of the check scripts in [derivations/](derivations/) or of the tests in [impl/](impl/), and is labelled as such.

### Worked example

Take $x = (1, 0)$ and $x' = (1/2, \sqrt 3 / 2)$, two unit vectors at $\varphi = \pi/3$, so $x^\top x' = 1/2$.
The first-layer term is $\frac12 \cdot \frac{2\pi/3}{2\pi} = \frac16 \approx 0.1667$.
The second-layer term is $\frac{\sqrt 3/2 + (2\pi/3)(1/2)}{2\pi} = \frac{\sqrt3/2 + \pi/3}{2\pi} \approx 0.3045$.
So $\Theta_\infty(x, x') \approx 0.4712$. {#c-worked-example}
On the diagonal the kernel is $\lVert x \rVert^2$, half from each layer, so $\Theta_\infty(x, x) = 1$; for orthogonal unit vectors it is $1/(2\pi) \approx 0.159$, and for opposite ones it is 0.
These four values are the fixed points of `test_worked_example`, `test_special_angles` and the matching PyTorch tests.

In our run of [relu-ntk.check.py](derivations/relu-ntk.check.py), on six inputs in $\mathbb{R}^4$ with 30 draws per width, the root-mean-square error of the empirical kernel fell from 0.567 at $m = 64$ to 0.0187 at $m = 65536$, with a fitted log-log slope of $-0.505$.
In our run of [kernel-flow.check.py](derivations/kernel-flow.check.py), at width 512 the decay rate fitted to each eigen-component of the linearized residual matched $-\log(1 - \eta \lambda_i / n)$ to $10^{-9}$, for eigenvalues from 0.118 to 3.01.

### Implementations

[impl/numpy/neural_tangent_kernel.py](impl/numpy/neural_tangent_kernel.py) builds the two-layer network of eq-two-layer, its explicit Jacobian, the empirical NTK and its two layer terms, the analytic kernel of eq-relu-ntk, the closed-form flow of eq-residual-flow by eigendecomposition, gradient descent on $f_{\text{lin}}$ and on the network, and the relative weight change.
[impl/pytorch/neural_tangent_kernel.py](impl/pytorch/neural_tangent_kernel.py) does the same with `torch.func.jacrev` for the Jacobian and `torch.optim.SGD` for the network.
The tests check the Jacobian against finite differences and a hand computation, the analytic kernel at special angles and by Monte Carlo, the $1/\sqrt m$ convergence slope, that discrete gradient descent on $f_{\text{lin}}$ equals the matrix power exactly, that a wider network tracks $f_{\text{lin}}$ more closely, and that the relative weight change falls with width.

## Examples

**The linearized network fits along eigen-directions.**
In our run of [kernel-flow.check.py](derivations/kernel-flow.check.py), 200 steps of gradient descent on $f_{\text{lin}}$ at $\eta = 1$ matched $(I - \eta\hat\Theta/n)^{200} r_0$ to $8.9 \times 10^{-17}$, and the gap to the flow at time 100 halved each time $\eta$ was halved.
Jacot et al. read the same picture as a case for early stopping: convergence is faster along the eigenspaces with larger eigenvalues, so stopping early fits the main kernel principal components and leaves the low-eigenvalue ones, which tend to be the noisier ones. {#c-early-stopping}

**Wider networks move their weights less.**
In our run of [lazy-weights.check.py](derivations/lazy-weights.check.py), 300 steps of gradient descent on six points moved the weights of the actual network by a total of 2.0 to 2.6 at every width from 64 to 8192, while the relative change fell from 0.155 to 0.0116, a log-log slope of $-0.512$.
Lee et al. describe the same regime: individual weights and hidden activations move by vanishingly small amounts, but together they make a finite change in the output. {#c-lee-weights-vanish}
Jacot et al. saw it in the kernel itself: after 200 steps, the NTK of a depth-4 network of width 500 tends to inflate, while at width 10000 it stays almost fixed. {#c-inflate}

**Linearization in practice.**
Lee et al. find that the predictions of the network and its linearization agree well even at practical widths, across architectures, optimizers and losses. {#c-lee-agreement}
At finite width the empirical kernel at initialization predicts training better than the analytic infinite-width kernel. {#c-lee-empirical-kernel}
With cross-entropy the agreement is weaker: some logits grow without bound, and if the two models part before that, the gap grows much worse at late times. {#c-lee-cross-entropy}

**Non-example: the mean-field scaling.**
For a two-layer network with output $\alpha(m) \sum_j \phi(\theta_j)$, Chizat et al. prove that the lazy regime is reached whenever $m\,\alpha(m) \to \infty$, which includes the NTK choice $\alpha = 1/\sqrt m$, while $\alpha(m) = 1/m$ gives a different, non-degenerate limit, the mean-field limit. {#c-chizat-mean-field}
There the weights move by order 1, the features change, and the NTK is not constant.

**Non-example: networks that perform well.**
On CIFAR-10, Chizat et al. trained VGG-11 with an output scale $\alpha$ and saw a significant drop in accuracy as $\alpha$ grew toward the lazy regime; wide VGG-11 and ResNet models reached 61.7% and 56.7% test accuracy when trained linearized, against 89.7% and 91.0% when trained in the standard way. {#c-chizat-cnn}
They do not claim that wide networks are not lazy, only that the ones that perform well are far from that limit. {#c-chizat-far}

**Borderline: the convolutional NTK.**
Arora et al. gave the first efficient exact algorithm for the convolutional NTK and the first non-asymptotic proof that a fully trained, wide enough net equals kernel regression with the NTK. {#c-arora-algorithm}
The 11-layer CNTK with global average pooling reached 77.43% on CIFAR-10, 10% above earlier kernels but about 6% below the corresponding finite network with batch normalization turned off. {#c-arora-cifar}
So the kernel is a strong baseline, and the gap that remains is what finite networks gain by moving away from it.

## Where it is used

The NTK is a tool for analysis more than a training method.
It gives convergence proofs for over-parameterized networks: with a positive definite kernel, gradient descent drives the training loss to zero at a linear rate.
It turns generalization questions into kernel questions, so the eigenvalues of $\Theta_\infty$ say which functions a wide network learns quickly.
Infinite-width kernels such as the CNTK are used as fixed kernels for small-data regression and classification.
The empirical NTK of a finite network is used to diagnose training, by tracking how far $\hat\Theta$ moves from its initial value, and as the base of work on parametrizations, such as the [maximal update parametrization](../../hyperparameter-transfer/maximal-update-parametrization/node.md), that are built to escape the kernel regime.

## Related ideas

- **Gaussian processes** ([[al:gaussian-process]]): a wide network at initialization is a Gaussian process with the NNGP kernel; the NTK governs how it moves during training, and Jacot et al. note that $f_t$ stays Gaussian along the limiting flow.
- **The kernel trick** ([[al:kernel-trick]]) uses a kernel to stand for an inner product of feature maps that are never computed; the NTK is such a kernel, but its feature map is the parameter gradient $\nabla_\theta f(x)$, which is explicit and finite at any finite width.
- **He and Xavier initialization** ([[al:he-initialization]], [[al:xavier-initialization]]) put the factor $1/\sqrt{n}$ into the variance of the weights instead of a multiplier in front of them; the network computes the same functions, but its gradients, and so its NTK and its learning rates, scale differently.
- **Kernel ridge regression** ([[al:kernel-ridge-regression]]): the limit of the trained network, from $f_0 = 0$, is kernel regression with $\Theta_\infty$ and no ridge term; stopping early acts like a ridge.
- **The maximal update parametrization** ([maximal update parametrization](../../hyperparameter-transfer/maximal-update-parametrization/node.md)) scales the layers so that features keep learning as width grows, the opposite of the kernel regime.
- **Double descent** ([[al:double-descent]]): in the kernel regime, the test error of very wide networks can be studied as that of a fixed kernel method.

## Common confusions

**"Wide networks are always in the kernel regime."**
Width by itself does not decide it; the scaling does.
Lee et al. find that linearization holds in both the standard and the NTK parameterization, so in their setting the effect comes from width, not the parametrization. {#c-lee-param-independence}
The two parametrizations train identically up to a width-dependent learning rate for each parameter tensor. {#c-lee-same-up-to-lr}
Chizat et al. put the cause in scale: lazy training is not specific to over-parameterized networks but comes from a choice of scaling, often implicit, that makes the model behave like its linearization. {#c-chizat-scaling}
For any model $\alpha h$ trained on a loss divided by $\alpha^2$, the parameters stay within $O(1/\alpha)$ of their start. {#c-chizat-alpha}
In our run of [lazy-weights.check.py](derivations/lazy-weights.check.py) at a fixed width of 256, the weight change was 0.444, 0.111 and 0.0276 for $\alpha = 4$, 16 and 64, a slope of $-1.00$ against $\log \alpha$.
The two views agree: width at the usual scalings is one way to make the scale large, and the mean-field scaling is a way to be wide without being lazy.

**"The NTK is constant during training."**
Only in the limit, or for $f_{\text{lin}}$.
At finite width $\hat\Theta$ is random at initialization and moves during training; Lee et al. bound that change by $O(n^{-1/2})$, and Jacot et al. saw it at width 500.
Jacot et al.'s constancy theorem also assumes a twice differentiable nonlinearity, which ReLU is not, so for the ReLU network on this page the constancy is checked numerically rather than taken from that theorem.

**"The NTK is the NNGP kernel."**
The NNGP kernel is the covariance of the outputs at initialization, $\mathbb{E}[f(x) f(x')]$.
The NTK is the inner product of gradients.
For the two-layer ReLU network the NNGP kernel is the second-layer term of eq-relu-ntk alone, and the NTK adds the first-layer term.

**"The NTK explains why deep learning works."**
It explains how wide networks can reach zero training loss, but a network in the kernel regime does not learn features.
Its best results, such as the CNTK on CIFAR-10, stay below those of the finite networks they come from.

**"Small relative change means the weights do not matter."**
The relative change goes to zero, but the total displacement stays of order 1, and it is spread over many weights so that the output moves by order 1.

## History

Jacot, Gabriel and Hongler posted the NTK in June 2018 and published it at NIPS 2018.
They defined the kernel, proved its deterministic limit at initialization and its constancy during training, and related least-squares training to kernel principal components and early stopping.
Lee et al. (NeurIPS 2019) recast the result as training the first-order Taylor expansion of the network, proved a quantitative bound in the width, and computed analytic ReLU and erf kernels.
Chizat, Oyallon and Bach (NeurIPS 2019; online December 2018) named the phenomenon lazy training and argued that it comes from scale, with experiments in which lazy convolutional networks did worse.
Arora et al. (NeurIPS 2019) gave the exact convolutional NTK and its CIFAR-10 benchmark.
Later work on feature learning, including the maximal update parametrization, starts from the gap between these kernels and trained networks.
