---
tag: NEW-adam
kind: method
slug: adam
name: Adam
aliases: [Adam optimizer, adaptive moment estimation, AdamW, decoupled weight decay]
area: optimization-for-learning/adaptive-optimizers
level: core
summary: >-
  A stochastic gradient method that steps each parameter by the ratio of bias-corrected moving averages of the gradient and of its square, so every coordinate gets its own step size.
purpose: training neural networks and other stochastic objectives with little tuning of the learning rate
uses: [stochastic-gradient-descent@m2, expected-value@m2, moment@m2]
related_to: [adamw@al, amsgrad@al, radam@al, weight-decay@al, regularization@al, learning-rate-schedule@al, gradient-clipping@al, fisher-information@m2, quasi-newton-method@m2]
hyperparameters:
  - {name: learning rate, meaning: "the step size eta, alpha in Kingma and Ba; with the bias correction it is about the largest step of each coordinate in the common case", typical: "1e-4 to 3e-3", default_in: {pytorch: 1.0e-3, keras: 1.0e-3}}
  - {name: beta1, meaning: "the decay rate of the moving average of the gradient", typical: "0.9", default_in: {pytorch: 0.9, keras: 0.9}}
  - {name: beta2, meaning: "the decay rate of the moving average of the squared gradient; lower values react faster to a change of scale", typical: "0.95 to 0.999", default_in: {pytorch: 0.999, keras: 0.999}}
  - {name: epsilon, meaning: "the constant added to the root of the second moment; Keras adds it in the efficient form, where it is the epsilon hat of Kingma and Ba", typical: "1e-8 to 1e-6, larger in some settings", default_in: {pytorch: 1.0e-8, keras: 1.0e-7}}
  - {name: weight decay, meaning: "an L2 term added to the gradient in torch.optim.Adam, a decoupled decay multiplied by the learning rate in torch.optim.AdamW", typical: "0 for Adam; 0.01 to 0.1 for AdamW", default_in: {pytorch: 0.0}}
cost:
  train: "two buffers the size of the parameters (three with AMSGrad), and a few elementwise operations per parameter at each step"
  inference: "none; the optimizer state is not needed after training"
history:
  year: 2014
  people: [Diederik P. Kingma, Jimmy Ba]
  note: "Kingma and Ba, arXiv 1412.6980 (December 2014), published at ICLR 2015. Reddi, Kale and Kumar found the error in its convergence proof (ICLR 2018); Loshchilov and Hutter introduced AdamW (ICLR 2019)."
prereqs: [stochastic-gradient-descent@m2, expected-value@m2, moment@m2]
same_as:
  wikidata: null
  wikipedia: Stochastic_gradient_descent
  arxiv: ["1412.6980", "1711.05101", "1904.09237"]
  al: null
sources:
  - {paper: "Kingma and Ba (2015)", title: "Adam: A Method for Stochastic Optimization", locator: "the abstract, Algorithm 1, Sections 2 to 7", note: "the method, the bias correction, the step size, the regret bound, the experiments and AdaMax"}
  - {paper: "Reddi, Kale and Kumar (2018)", title: "On the Convergence of Adam and Beyond", locator: "Sections 1, 3 and 4", note: "the error in the convergence proof, the counterexample and AMSGrad"}
  - {paper: "Loshchilov and Hutter (2019)", title: "Decoupled Weight Decay Regularization", locator: "the abstract, Section 1, Algorithm 2", note: "L2 regularization against weight decay, and AdamW"}
  - {paper: "Zhang, Chen, Shi, Sun and Luo (2022)", title: "Adam Can Converge Without Any Modification On Update Rules", locator: "the abstract"}
  - {paper: "Wilson, Roelofs, Stern, Srebro and Recht (2017)", title: "The Marginal Value of Adaptive Gradient Methods in Machine Learning", locator: "the abstract, Section 1"}
  - {paper: "Choi, Shallue, Nado, Lee, Maddison and Dahl (2019)", title: "On Empirical Comparisons of Optimizers for Deep Learning", locator: "the abstract, Sections 4.3 and 5"}
  - {paper: "Liu et al. (2020)", title: "On the Variance of the Adaptive Learning Rate and Beyond", locator: "the abstract"}
  - {paper: "Vaswani et al. (2017)", title: "Attention Is All You Need", locator: "Section 5.3", note: "the optimizer settings of the Transformer"}
  - {paper: "Touvron et al. (2023)", title: "LLaMA: Open and Efficient Foundation Language Models", locator: "Section 2.3", note: "the optimizer settings of LLaMA"}
  - {url: "https://docs.pytorch.org/docs/2.15/generated/torch.optim.Adam.html", title: "torch.optim.Adam", locator: "the algorithm and the parameters lr, betas, eps, weight_decay, amsgrad and decoupled_weight_decay", note: "the PyTorch defaults, and weight_decay as an L2 penalty"}
  - {url: "https://docs.pytorch.org/docs/2.15/generated/torch.optim.AdamW.html", title: "torch.optim.AdamW", locator: "the algorithm and the parameter weight_decay", note: "the decay by lr times weight_decay, and its default of 1e-2"}
  - {url: "https://keras.io/api/optimizers/adam/", title: "Keras Adam", locator: "the arguments learning_rate, beta_1, beta_2 and epsilon", note: "the Keras defaults, and epsilon as the epsilon hat of Kingma and Ba"}
---

Adam is a stochastic gradient method that gives every parameter its own step size.
It keeps an exponential moving average of the gradient and one of the squared gradient, corrects both for their start at zero, and steps by their ratio.
The step of each coordinate is then about the learning rate in size, whatever the scale of its gradient.
It needs little tuning, and it is the usual optimizer for training neural networks.
Its convergence proof had an error, and its L2 regularization is not weight decay; AMSGrad and AdamW answer these two problems.

## In plain words

Plain stochastic gradient descent moves every parameter by the learning rate times its gradient.
If one parameter has gradients a thousand times larger than another, it moves a thousand times faster, and no single learning rate suits both.
Adam divides each gradient by a running estimate of its typical size, the root of the average squared gradient.
After this division every parameter moves by roughly the learning rate, in the direction of its averaged gradient.
When the gradient of a parameter keeps changing sign, the average is small against the typical size, and the step shrinks.
Both averages start at zero, so early on they are too small; Adam divides them by a factor that removes this bias.

## Setting

Adam minimizes an objective $f(\theta)$ of the parameters $\theta \in \mathbb{R}^d$ from stochastic gradients $g_t = \nabla f_t(\theta_{t-1})$, for example the gradients of the loss on successive mini-batches.
Kingma and Ba (2015) introduce it as an algorithm for first-order gradient-based optimization of stochastic objectives, based on adaptive estimates of lower-order moments. {#c-adam-definition}
They describe it as straightforward to implement, computationally efficient, light in memory and invariant to a diagonal rescaling of the gradients. {#c-adam-properties}
They state that it suits non-stationary objectives and very noisy or sparse gradients, and that its hyperparameters have intuitive meanings and typically need little tuning. {#c-adam-suited}
This page writes the learning rate as $\eta$; Kingma and Ba write it as $\alpha$.
All operations on vectors are elementwise.

## The method

Start from $m_0 = v_0 = 0$.
At step $t = 1, 2, \dots$, update the moving averages of the gradient and of its square,

$$ m_t = \beta_1 m_{t-1} + (1 - \beta_1)\, g_t, \qquad v_t = \beta_2 v_{t-1} + (1 - \beta_2)\, g_t^2 , $$ {#eq-moments}

correct them for their start at zero, and take the step

$$ \hat m_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat v_t = \frac{v_t}{1 - \beta_2^t}, \qquad \theta_t = \theta_{t-1} - \eta\, \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon} . $$ {#eq-update}

This is Algorithm 1 of Kingma and Ba. {#c-algorithm}
They give $\eta = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$ and $\epsilon = 10^{-8}$ as good defaults for the problems they tested. {#c-defaults}
$m_t$ estimates the mean of the gradient and $v_t$ its uncentered variance, and $\beta_1$ and $\beta_2$ set how fast the two estimates forget. {#c-moment-estimates}
Because both start at zero, they are biased towards zero, most of all in the first steps and when the $\beta$s are close to 1. {#c-init-bias}
The divisions by $1 - \beta^t$ remove this bias.

Kingma and Ba note that the same update can be computed with fewer operations, by folding the corrections into the step size: {#c-efficient-form}

$$ \eta_t = \eta\, \frac{\sqrt{1 - \beta_2^t}}{1 - \beta_1^t}, \qquad \theta_t = \theta_{t-1} - \eta_t\, \frac{m_t}{\sqrt{v_t} + \hat\epsilon} . $$ {#eq-efficient}

The two forms agree when $\hat\epsilon = \epsilon \sqrt{1 - \beta_2^t}$, so a fixed $\hat\epsilon$ is not a fixed $\epsilon$.

**Weight decay.**
`torch.optim.Adam` adds `weight_decay` times $\theta$ to the gradient before eq-moments, which is L2 regularization.
AdamW instead shrinks the parameters directly, beside the Adam step.
In the form of `torch.optim.AdamW`, with $w$ the `weight_decay`,

$$ \theta_t = (1 - \eta w)\, \theta_{t-1} - \eta\, \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon} . $$ {#eq-adamw}

Loshchilov and Hutter (2019) write AdamW in Algorithm 2 as a step of $\eta_t$, a schedule multiplier, times the sum of $\alpha \hat m_t / (\sqrt{\hat v_t} + \epsilon)$ and $\lambda \theta_{t-1}$. {#c-lh-alg2}

**AMSGrad.**
Reddi et al. (2018) keep the running maximum $v^{\max}_t = \max(v^{\max}_{t-1}, v_t)$ and divide by it instead of $v_t$, so the step size of each coordinate never increases. {#c-amsgrad}
`torch.optim.Adam` and `torch.optim.AdamW` offer it with `amsgrad=True`, and apply the bias correction to the maximum.

## Derivation

**The bias correction.**
Unrolling eq-moments down to $m_0 = 0$ gives

$$ m_t = (1 - \beta_1) \sum_{i=1}^{t} \beta_1^{t-i} g_i, \qquad v_t = (1 - \beta_2) \sum_{i=1}^{t} \beta_2^{t-i} g_i^2 , $$ {#eq-unrolled}

and the weights of each sum add up to $1 - \beta^t$.
Kingma and Ba take the expectation of $v_t$ in this form, find the true second moment times $1 - \beta_2^t$ plus a term $\zeta$ that vanishes for a stationary second moment, and divide by $1 - \beta_2^t$. {#c-bias-section3}
If every gradient has mean $\mu$ and second moment $s$, then $\mathbb{E}[\hat m_t] = \mu$ and $\mathbb{E}[\hat v_t] = s$. {#c-bias-expectation}
For a constant gradient the corrected averages are exact, and with $\epsilon = 0$ each step is exactly $\eta$ against the gradient. {#c-constant-gradient}
Without the correction and with $\epsilon = 0$, the first step is $\eta (1 - \beta_1)/\sqrt{1 - \beta_2}$, about $3.16\,\eta$ for the defaults. {#c-uncorrected-first-step}
Kingma and Ba point out that sparse gradients need $\beta_2$ close to 1, and that without the correction this makes the first steps much larger. {#c-sparse-large-steps}
A fixed $\hat\epsilon$ in eq-efficient acts as $\epsilon = \hat\epsilon / \sqrt{1 - \beta_2^t}$ in eq-update: for $\beta_2 = 0.999$ it is about 31.6 times larger at the first step, and it falls to $\hat\epsilon$ as $t$ grows. {#c-eps-hat}
The derivation `bias-correction` gives the steps.

**The size of the step.**
Write $r_t = \hat m_t / \sqrt{\hat v_t}$, so that with $\epsilon = 0$ the step of a coordinate is $\eta |r_t|$.
Kingma and Ba call $r_t$ a signal-to-noise ratio: when the gradient is noisy, $r_t$ and the step are close to zero. {#c-snr}
Scaling the gradients by $c$ scales $\hat m_t$ by $c$ and $\sqrt{\hat v_t}$ by $|c|$, so the step does not change. {#c-scale-invariant}
They state that the step is approximately bounded by $\eta$, which they read as a trust region around the current parameters. {#c-trust-region}
More precisely, they state the bound $\eta (1 - \beta_1)/\sqrt{1 - \beta_2}$ when $1 - \beta_1 > \sqrt{1 - \beta_2}$ and $\eta$ otherwise, and they say the first case arises only for a gradient that is zero at every step but the current one. {#c-paper-bounds}

The exact worst case is larger.
With $\epsilon = 0$ and $\gamma = \beta_1^2 / \beta_2$, the Cauchy–Schwarz inequality applied to eq-unrolled gives

$$ |r_t| \le B_t = \frac{1 - \beta_1}{\sqrt{1 - \beta_2}} \cdot \frac{\sqrt{1 - \beta_2^t}}{1 - \beta_1^t} \cdot \sqrt{\sum_{k=0}^{t-1} \gamma^k} , $$ {#eq-ratio-bound}

and gradients that grow as $g_{t-k} \propto (\beta_1 / \beta_2)^k$ reach it. {#c-step-bound-tight}
For $\gamma < 1$, $B_t$ tends to $(1 - \beta_1)/\sqrt{(1 - \beta_2)(1 - \gamma)}$, about 7.27 for the defaults and 2.35 for $\beta_2 = 0.99$, against the 3.16 and 1 of Section 2.1; for $\gamma \ge 1$ it grows without limit. {#c-step-bound-limits}
The first step is always exactly $\eta$ in size. {#c-first-step}
The single spike that Section 2.1 describes gives $|r_t|$ close to $(1 - \beta_1)/\sqrt{1 - \beta_2}$, which is its first bound. {#c-single-spike}
So the stated bound is the value of the spike, not the supremum.
The worst case needs gradients that grow by about 11% per step for many steps, which is rare, and in the common case $|r_t|$ stays below 1, since $|\mathbb{E}[g]| \le \sqrt{\mathbb{E}[g^2]}$.
The derivation `step-bound` gives the steps, and its check confirms the supremum numerically.

**Weight decay.**
For SGD with learning rate $\eta$, an L2 coefficient $\lambda/\eta$ gives exactly decoupled weight decay $\lambda$. {#c-sgd-equivalence}
In Adam the L2 term is divided by $\sqrt{\hat v_t} + \epsilon$ like the rest of the gradient, so coordinates with large gradients are decayed less, and no single L2 coefficient gives decoupled decay. {#c-adam-l2-not-decay}
At the first step, with $\epsilon = 0$, the step is $\eta$ times the sign of the penalised gradient, so the penalty does nothing unless it flips that sign. {#c-l2-first-step}
`torch.optim.AdamW` is Algorithm 2 of Loshchilov and Hutter with $\lambda = \eta w$, so its decay per step changes with the learning rate. {#c-torch-adamw-lr-wd}
The derivation `decoupled-weight-decay` gives the steps.

## Worked example

Take one parameter $\theta_0 = 1$ with $\eta = 0.1$, $\beta_1 = 0.5$, $\beta_2 = 0.96$ and $\epsilon = 0$.
These values keep the numbers exact: $1 - \beta_2 = 0.04$ and $1 - \beta_2^2 = 0.0784 = 0.28^2$.
The gradient is $g_1 = 2$, then $g_2 = -2$.

| step | $g_t$ | $m_t$ | $v_t$ | $\hat m_t$ | $\hat v_t$ | step $\eta \hat m_t / \sqrt{\hat v_t}$ | $\theta_t$ |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 1 | 0.16 | 2 | 4 | 0.1 | 0.9 |
| 2 | −2 | −0.5 | 0.3136 | −2/3 | 4 | −1/30 | 14/15 ≈ 0.9333 |

At step 1, $m_1 = 0.5 \cdot 2 = 1$ and $v_1 = 0.04 \cdot 4 = 0.16$.
The corrections divide by $0.5$ and $0.04$, so $\hat m_1 = 2$, $\hat v_1 = 4$, and the step is $0.1 \cdot 2/2 = 0.1$, exactly $\eta$.
At step 2, $m_2 = 0.5 - 1 = -0.5$ and $v_2 = 0.1536 + 0.16 = 0.3136$.
The corrections divide by $0.75$ and $0.0784$, so $\hat m_2 = -2/3$ and $\hat v_2 = 4$.
The gradient changed sign, so the average shrank against the typical size and the step is only $1/30$.

Without the correction, the first step would be $0.1 \cdot 1/0.4 = 0.25$, and $\theta_1 = 0.75$.
With AdamW and $w = 0.1$, the first step gives $\theta_1 = 1 \cdot (1 - 0.01) - 0.1 = 0.89$.
With an L2 coefficient of 0.1 instead, the gradient is $2.1$, and the first step is still exactly $0.1$, so $\theta_1 = 0.9$: the penalty has no effect.
The tests `test_worked_example` and `test_worked_example_weight_decay` check these numbers.

## Hyperparameters

**Learning rate.**
It sets the size of the steps, about $\eta$ per coordinate in the common case.
PyTorch and Keras default to $10^{-3}$, the value of Kingma and Ba.
Liu et al. (2020) find that the variance of the adaptive step size is very large early in training, explain warmup of the learning rate as a way to reduce it, and propose RAdam, which corrects this variance. {#c-liu-warmup}

**$\beta_1$ and $\beta_2$.**
$\beta_1 = 0.9$ averages the gradient over about 10 steps, and $\beta_2 = 0.999$ the squared gradient over about 1000.
A smaller $\beta_2$ reacts faster when the scale of the gradients changes, and lowers the worst-case step of eq-ratio-bound.
Zhang et al. (2022) prove that Adam converges to a neighborhood of critical points when $\beta_2$ is large and $\beta_1 < \sqrt{\beta_2}$, with a neighborhood that grows with the variance of the gradients. {#c-zhang-converge}
For small $\beta_2$ they find a large region of $(\beta_1, \beta_2)$ where Adam can diverge, and they advise raising $\beta_2$ when Adam does not work well. {#c-zhang-diverge}
PyTorch and Keras default to $\beta_1 = 0.9$ and $\beta_2 = 0.999$.

**Epsilon.**
It keeps the division stable when $\hat v_t$ is close to zero.
When it is large against $\sqrt{\hat v_t}$, Adam turns into SGD with momentum and learning rate $\eta/\epsilon$.
Choi et al. (2019) find that the best $\epsilon$ depends on the problem, and that studies that fix it at $10^{-8}$ should defend that choice. {#c-choi-eps}
Reddi et al. also note that the choice of $\epsilon$ appears critical in practice. {#c-reddi-epsilon}
PyTorch defaults to $10^{-8}$, added as in eq-update.
Keras defaults to $10^{-7}$, and its documentation states that this is the $\hat\epsilon$ of the efficient form, not the $\epsilon$ of Algorithm 1.

**Weight decay.**
`torch.optim.Adam` defaults to no weight decay, and its `weight_decay` is an L2 penalty.
`torch.optim.AdamW` defaults to $w = 10^{-2}$, decoupled and multiplied by the learning rate.
Loshchilov and Hutter find that the longer the training, the smaller the best weight decay, and that Adam gains from a learning-rate schedule such as cosine annealing. {#c-lh-runtime}

## Cost

Adam keeps two numbers per parameter, $m_t$ and $v_t$, so its state is twice the size of the model; AMSGrad keeps a third.
Each step costs a few elementwise operations and one square root per parameter, which is small next to the gradient.
Nothing is needed at inference.

## Evidence

On logistic regression on MNIST, Adam converges about as fast as SGD with Nesterov momentum and faster than AdaGrad; on sparse bag-of-words features of IMDB, it matches AdaGrad. {#c-mnist-logreg}
On a network with two hidden layers of 1000 units on MNIST, it makes faster progress than the quasi-Newton method SFO, and with dropout it converges better than the other first-order methods. {#c-mlp}
In convolutional networks on CIFAR-10, Kingma and Ba observe that $\hat v_t$ becomes very small after a few epochs and is dominated by $\epsilon$. {#c-cnn-eps}
On a variational autoencoder, they find that $\beta_2$ close to 1 makes training unstable without the bias correction, especially in the first epochs, and that Adam does as well as or better than RMSProp at every setting. {#c-bias-experiment}

Loshchilov and Hutter report that decoupled weight decay improves the test error of Adam by 15% relative to L2 regularization on CIFAR-10 and ImageNet32x32, for 100 to 1800 epochs of training. {#c-lh-results}
Choi et al. (2019) find that when all hyperparameters are tuned, adaptive methods such as Adam never do worse than momentum or plain SGD, and that the search space can explain the rankings of earlier comparisons. {#c-choi-inclusion}

## Limitations and failure modes

**The convergence proof was wrong.**
Kingma and Ba prove an $O(\sqrt{T})$ regret bound in online convex optimization, for bounded gradients, a learning rate decaying as $1/\sqrt{t}$ and a decaying $\beta_1$. {#c-regret}
Reddi et al. (2018) show that the proof is wrong, and give a simple convex problem on which Adam does not converge to the optimum. {#c-reddi-flaw}
The proof assumes that a quantity $\Gamma_t$, the change of the inverse step sizes, is positive semi-definite, while for Adam and RMSProp it can be negative. {#c-reddi-gamma}
In their example, on $[-1, 1]$, the loss is $C x$ at one step in three and $-x$ at the other two, with $C > 2$, $\beta_1 = 0$ and $\beta_2 = 1/(1 + C^2)$. Adam converges to $x = +1$, while the best point is $x = -1$. {#c-reddi-example}
The rare large gradient raises $v_t$, but $\beta_2$ forgets it within a few steps, so the two small gradients pull harder than the large one.
They prove that for any constant $\beta_1 < \sqrt{\beta_2}$ there are online and stochastic convex problems on which Adam fails to converge. {#c-reddi-any-beta}
AMSGrad fixes the proof, because the maximum of $v_t$ never forgets the large gradient.
The test `test_reddi_counterexample` runs the example with $C = 3$: Adam ends at 1, and AMSGrad near $-1$.

Zhang et al. (2022) point out that the counterexamples choose the problem after the hyperparameters, while in practice the problem comes first and $\beta_1$ and $\beta_2$ are tuned for it. {#c-zhang-order}
With $\beta_2$ chosen large for the problem, Adam converges without changes, in the sense of the previous section.

**L2 regularization is not weight decay.**
For SGD the two are the same up to a factor of the learning rate, but not for Adam. {#c-lh-not-equivalent}
With L2, weights with large past gradients are regularized less than with weight decay. {#c-lh-large-gradients}
Loshchilov and Hutter propose to decouple the decay from the gradient step, and report that this separates the best weight decay from the learning rate and improves generalization substantially. {#c-lh-decouple}

**Generalization.**
Wilson et al. (2017) build a classification problem on which SGD reaches zero test error while Adam, AdaGrad and RMSProp get close to one half, and report that adaptive methods generalize worse than SGD on several models. {#c-wilson-generalize}
They observe that adaptive methods progress faster on the training set at first, but plateau early on the test set. {#c-wilson-plateau}
Reproducing their VGG experiment on CIFAR-10, Choi et al. find that once the momentum and $\epsilon$ are tuned, all four optimizers reach nearly the same test error. {#c-choi-wilson}
So part of the gap comes from tuning, and the question depends on the search space.

**The second moment can vanish.**
When $\hat v_t$ becomes small against $\epsilon$, as in the CIFAR-10 networks of Kingma and Ba, Adam no longer adapts to the geometry of the loss.

## Variants

- **AdamW** decouples weight decay from the gradient step (Loshchilov and Hutter 2019), as in eq-adamw.
- **AMSGrad** divides by the running maximum of $v_t$, so the step sizes never increase (Reddi et al. 2018).
- **AdaMax** replaces $\sqrt{v_t}$ by $u_t = \max(\beta_2 u_{t-1}, |g_t|)$, a weighted infinity norm that needs no bias correction; its default learning rate is 0.002 (Kingma and Ba, Section 7.1). {#c-adamax}
- **Temporal averaging** keeps a moving average of the parameters, corrected for its start at zero, because the last iterate is noisy (Kingma and Ba, Section 7.2). {#c-temporal-averaging}
- **RAdam** corrects the variance of the adaptive step size early in training (Liu et al. 2020).

Kingma and Ba relate Adam to two earlier methods.
RMSProp with momentum applies the momentum to the rescaled gradient, while Adam averages the gradient before rescaling, and RMSProp has no bias correction, which with $\beta_2$ close to 1 leads to very large steps and often divergence. {#c-rmsprop-relation}
AdaGrad is Adam with $\beta_1 = 0$, an infinitesimal $1 - \beta_2$ and a learning rate that decays as $\eta / \sqrt{t}$. {#c-adagrad-relation}
They also note that $\hat v_t$ approximates the diagonal of the Fisher information matrix, and that dividing by its square root adapts more conservatively than natural gradient descent. {#c-fisher}

## When to use it

Adam, or AdamW, is the default optimizer for most neural networks, and the first one to try.
Wilson et al. note that Adam has become the default algorithm of many deep learning frameworks. {#c-wilson-default}
Loshchilov and Hutter call adaptive methods such as Adam the default choice for feed-forward and recurrent networks, while the best results on CIFAR-10 and CIFAR-100 were still obtained with SGD with momentum. {#c-lh-default}
The Transformer was trained with Adam, $\beta_1 = 0.9$, $\beta_2 = 0.98$ and $\epsilon = 10^{-9}$, with a learning rate that rises linearly for 4000 steps and then decays as the inverse square root of the step. {#c-transformer}
LLaMA was trained with AdamW, $\beta_1 = 0.9$, $\beta_2 = 0.95$, a weight decay of 0.1, gradient clipping at 1.0, a warmup and a cosine schedule down to 10% of the peak learning rate. {#c-llama}
Its warmup lasts 2000 steps.
Use AdamW rather than Adam with L2 whenever you want weight decay.
For image classifiers, SGD with momentum remains a strong choice when there is budget to tune its schedule, as the results above on CIFAR-10 and CIFAR-100 suggest.

## Pitfalls

> [!pitfall]
> **The bias correction left out.** Without it the first step is $(1 - \beta_1)/\sqrt{1 - \beta_2}$ times too large, about 3.16 for the defaults. The tests `test_uncorrected_first_step` and `test_worked_example` check the corrected and the uncorrected first step.

> [!pitfall]
> **L2 regularization used as weight decay.** `torch.optim.Adam(weight_decay=...)` adds an L2 term that the second moment rescales, so it is not AdamW. The tests `test_l2_is_not_decoupled` and `test_adam_weight_decay_is_not_adamw` check that the two differ and that L2 decays the coordinates with large gradients much less.

> [!pitfall]
> **The weight decay of PyTorch moves with the learning rate.** `torch.optim.AdamW` decays by $\eta w$ per step, so changing the learning rate or its schedule changes the decay too, and a $\lambda$ from Loshchilov and Hutter must be divided by $\eta$. The test `test_adamw_decay_scales_with_lr` checks this for the page's AdamW and for `torch.optim.AdamW`.

> [!pitfall]
> **Epsilon in the wrong place.** Adding $\epsilon$ inside the square root, $\sqrt{\hat v_t + \epsilon}$, gives a different optimizer whose $\epsilon$ acts like the square root of a much larger value. The test `test_eps_outside_the_square_root` checks that the implementation matches `torch.optim.Adam` and not the other form.

> [!pitfall]
> **The epsilon of another framework.** The Keras `epsilon` is $\hat\epsilon$ of eq-efficient, so the same number gives a larger effective $\epsilon$ in the first steps than in PyTorch. The tests `test_keras_epsilon_hat` and `test_eps_hat` check that the efficient form with $\hat\epsilon$ is eq-update with $\epsilon = \hat\epsilon / \sqrt{1 - \beta_2^t}$.

> [!pitfall]
> **Trusting the convergence proof.** Adam with a constant $\beta_2$ can converge to the wrong point of a convex problem. The tests `test_reddi_counterexample` and `test_amsgrad_max_is_nondecreasing` run the counterexample of Reddi et al. and check that AMSGrad's maximum never decreases.

## History

Diederik P. Kingma and Jimmy Ba posted Adam in December 2014, with the ninth version on 30 Jan 2017, and it appeared at ICLR 2015. {#c-hist-kingma}
It combined the momentum of SGD with the per-parameter scaling of AdaGrad and RMSProp, and added the bias correction.
Sashank Reddi, Satyen Kale and Sanjiv Kumar of Google showed the error in its convergence proof at ICLR 2018, and proposed AMSGrad. {#c-hist-reddi}
Ilya Loshchilov and Frank Hutter separated weight decay from the gradient step in AdamW, published at ICLR 2019. {#c-hist-lh}
PyTorch offers it as `torch.optim.AdamW`, and LLaMA, among other large language models, was trained with it.
