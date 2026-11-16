---
tag: DD5QVM
kind: method
slug: weight-decay
name: Weight decay
aliases: [L2 regularization, decoupled weight decay, SGDW, weight decay regularization, ridge penalty]
area: deep-learning-training/regularization
level: core
summary: >-
  Shrink the weights toward zero at every step, by an L2 penalty on the loss or by multiplying them by a factor below one; the two agree for plain SGD only.
purpose: regularization
uses: [stochastic-gradient-descent@m2, normal-distribution@m2]
related_to: [dropout, adamw@al, batch-normalization@al, layer-normalization@al, weight-normalization@al, ridge-regression@m2, maximum-a-posteriori-estimation@al, early-stopping@al, learning-rate-schedule@al, regularization@al, overfitting@al]
hyperparameters:
  - {name: weight decay, meaning: "the L2 coefficient lambda, the weight_decay of torch.optim.SGD and torch.optim.Adam, added as lambda theta to the gradient", typical: "1e-4 to 5e-4 for SGD on image classifiers", default_in: {pytorch: 0.0}}
  - {name: decoupled weight decay, meaning: "the weight_decay of torch.optim.AdamW, which multiplies the weights by 1 - lr * weight_decay at each step, so the decay per step moves with the learning rate", typical: "0.1 for large language models", default_in: {pytorch: 1.0e-2}}
cost:
  train: "one multiply-add per parameter at each step, and no extra memory"
  inference: "none"
history:
  year: 1991
  people: [Anders Krogh, John A. Hertz]
  note: "Hanson and Pratt (1988) described the decay; Krogh and Hertz (NIPS 1991) analysed it for neural networks; Loshchilov and Hutter (ICLR 2019) separated it from L2 regularization in adaptive optimizers."
prereqs: [stochastic-gradient-descent@m2, regularization@al, overfitting@al]
same_as:
  wikidata: null
  wikipedia: null
  arxiv: ["1711.05101", "1810.12281", "2310.04415", "1910.07454", "1706.05350"]
  al: weight-decay
sources:
  - {paper: "Krogh and Hertz (1991)", title: "A Simple Weight Decay Can Improve Generalization", locator: "the abstract, Sections 1, 3 to 6", note: "the penalty, the smallest-norm solution, the optimal decay and NetTalk"}
  - {paper: "Loshchilov and Hutter (2019)", title: "Decoupled Weight Decay Regularization", locator: "the abstract, Sections 2 to 4.2, page 12", note: "L2 against weight decay, Propositions 1 to 3, SGDW and AdamW, normalized weight decay"}
  - {paper: "van Laarhoven (2017)", title: "L2 Regularization versus Batch and Weight Normalization", locator: "the abstract, Sections 2 to 3.1", note: "no regularizing effect under normalization, and the effective learning rate"}
  - {paper: "Zhang, Wang, Xu and Grosse (2019)", title: "Three Mechanisms of Weight Decay Regularization", locator: "the abstract, Sections 1, 3 and 4.1", note: "the three mechanisms, and the effective learning rate in practice"}
  - {paper: "Li and Arora (2020)", title: "An Exponential Learning Rate Schedule for Deep Learning", locator: "pages 1 and 2, Theorem 1.1"}
  - {paper: "D'Angelo, Andriushchenko, Varre and Flammarion (2024)", title: "Why Do We Need Weight Decay in Modern Deep Learning?", locator: "the abstract, Sections 1, 1.1 and 3"}
  - {paper: "Krizhevsky, Sutskever and Hinton (2012)", title: "ImageNet Classification with Deep Convolutional Neural Networks", locator: "Section 5"}
  - {paper: "He, Zhang, Ren and Sun (2016)", title: "Deep Residual Learning for Image Recognition", locator: "Section 3.4"}
  - {paper: "Goyal et al. (2017)", title: "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour", locator: "Sections 3 and 5.1"}
  - {paper: "He, Zhang, Zhang, Zhang, Xie and Li (2019)", title: "Bag of Tricks for Image Classification with Convolutional Neural Networks", locator: "Section 3.1", note: "no bias decay"}
  - {paper: "Touvron et al. (2023)", title: "LLaMA: Open and Efficient Foundation Language Models", locator: "Section 2.3"}
  - {url: "https://docs.pytorch.org/docs/2.15/generated/torch.optim.SGD.html", title: "torch.optim.SGD", locator: "the algorithm and the parameter weight_decay", note: "weight_decay added to the gradient before the momentum buffer, default 0"}
  - {url: "https://docs.pytorch.org/docs/2.15/generated/torch.optim.AdamW.html", title: "torch.optim.AdamW", locator: "the algorithm and the parameter weight_decay", note: "the decay by lr times weight_decay, and its default of 1e-2"}
---

Weight decay pulls the weights of a model toward zero at every step of training.
It comes in two forms: an L2 penalty $\frac{\lambda}{2}\lVert\theta\rVert^2$ added to the loss, and a decoupled decay that multiplies the weights by a factor just below one beside the gradient step.
For plain stochastic gradient descent the two are the same; with momentum they differ a little, and with adaptive optimizers such as Adam they differ a lot, which is why AdamW exists.
The penalty is the negative log of a Gaussian prior, so L2 training finds a maximum a posteriori estimate.
In networks with batch or layer normalization the penalty no longer limits what the network computes; it controls the scale of the weights and through it the effective learning rate.

## In plain words

Gradient descent moves the weights wherever the loss pushes them, and nothing stops them from growing large.
Weight decay adds a steady pull back toward zero: at every step each weight loses a small fraction of its value.
A weight that the loss does not need fades away, while a weight that the loss keeps pushing settles where the push and the pull balance.
Smaller weights usually mean a smoother function that fits less of the noise in the training data, the overfitting that regularization fights.
The same pull can be written as a cost on large weights added to the loss; for plain gradient descent this gives exactly the same steps.
Optimizers that rescale each gradient also rescale that cost, so for them the two forms are no longer the same.

## Setting

The weights $\theta \in \mathbb{R}^d$ are trained on a loss $f(\theta)$ from stochastic gradients $g_t = \nabla f_t(\theta_{t-1})$ with learning rate $\eta$.
Krogh and Hertz add to the cost a term that penalizes large weights, $E(w) = E_0(w) + \frac{1}{2}\lambda\sum_i w_i^2$, with $\lambda$ setting how strongly large weights are penalized. {#c-kh-penalty}
With gradient descent this term adds $-\lambda w_i$ to the update, and without the gradient of $E_0$ it would make the weights decay exponentially. {#c-kh-decay-term}
Hanson and Pratt (1988), as summarized by Loshchilov and Hutter, instead define weight decay as $\theta_{t+1} = (1 - \lambda)\theta_t - \alpha\nabla f_t(\theta_t)$, with $\lambda$ the rate of decay per step. {#c-hanson-pratt}
This page writes $\lambda$ for the L2 coefficient, the `weight_decay` of `torch.optim.SGD`, and $\delta$ for the decay per step, which is the $\lambda$ of Loshchilov and Hutter.
Because the two are the same for SGD, L2 regularization is very often called weight decay, including in deep learning libraries. {#c-lh-called-wd}
Goyal et al., for example, describe weight decay as the gradient of an L2 term, computed separately and added to the aggregated gradients. {#c-goyal-l2}

## The method

**L2 regularization** adds $\lambda\theta$ to the gradient before the optimizer sees it.
For SGD with momentum $\mu$, as `torch.optim.SGD` computes it,

$$ g'_t = g_t + \lambda\theta_{t-1}, \qquad b_t = \mu b_{t-1} + g'_t, \qquad \theta_t = \theta_{t-1} - \eta\, b_t , $$ {#eq-l2-step}

with $b_1 = g'_1$.

**Decoupled weight decay** keeps the gradient of the loss alone in the optimizer and shrinks the weights beside the step,

$$ b_t = \mu b_{t-1} + g_t, \qquad \theta_t = (1 - \delta)\,\theta_{t-1} - \eta\, b_t . $$ {#eq-decay-step}

Loshchilov and Hutter call this SGDW; their Algorithm 1 adds $\lambda\theta_{t-1}$ to the gradient for L2 and decays the weights in line 9 for SGDW. {#c-lh-sgdw}
For an adaptive optimizer the same split gives AdamW, in which the gradient alone enters the moving averages and the weights are shrunk outside them; see adam for the algorithm and `torch.optim.AdamW`.
PyTorch writes the decay of AdamW as $\delta = \eta\cdot$`weight_decay`, so the decay per step moves with the learning rate and its schedule.

Biases and the scale and shift of normalization layers are usually left out of the decay, as described under Hyperparameters.

## Derivation

**L2 and decay under SGD.**
Loshchilov and Hutter's Proposition 1 states that SGD with decay $\delta$ takes the same steps as SGD on the loss plus an L2 term with coefficient $\delta/\eta$, so the best L2 coefficient is tied to the learning rate. {#c-lh-prop1}
With $\mu = 0$, eq-l2-step is $\theta_t = \theta_{t-1} - \eta g_t - \eta\lambda\theta_{t-1} = (1 - \eta\lambda)\theta_{t-1} - \eta g_t$, which is eq-decay-step with $\delta = \eta\lambda$, for every sequence of gradients. {#c-sgd-equivalence}
With no loss gradient, both give $\theta_k = (1 - \eta\lambda)^k\theta_0$, the discrete form of the exponential decay $\dot\theta = -\lambda\theta$ that gives the method its name. {#c-exponential-decay}
With momentum the L2 term enters the buffer and is carried into later steps, while the decoupled decay is not.
The two forms with $\delta = \eta\lambda$ agree at the first step and differ at the second by exactly $\eta\mu\lambda\theta_0$, so no constant $\delta$ reproduces L2 along the trajectory. {#c-momentum-differs}
On a fixed loss, L2 stops where $\nabla f(\theta) + \lambda\theta = 0$ and the decoupled form where $\nabla f(\theta) + \frac{\delta(1-\mu)}{\eta}\theta = 0$, so their fixed points agree when $\delta = \eta\lambda/(1 - \mu)$. {#c-momentum-fixed-point}
The derivation `sgd-equivalence` gives the steps, and its check runs both forms with and without momentum.

**Adaptive and preconditioned optimizers.**
Loshchilov and Hutter state that the two forms are equivalent for SGD when rescaled by the learning rate, but not for adaptive methods such as Adam. {#c-lh-not-equivalent}
Their Proposition 2 states that for a step $-\alpha M_t\nabla f_t$ with $M_t \neq kI$ there is no L2 coefficient that reproduces weight decay. {#c-lh-prop2}
With L2, the gradient of the penalty is divided by the typical size of the gradient along with the rest, so weights with large gradients are regularized relatively less, while decoupled decay shrinks all weights at the same rate. {#c-lh-large-s}
Their Proposition 3 states that with a fixed preconditioner $\operatorname{diag}(s)^{-1}$, weight decay is a scale-adjusted L2 penalty that regularizes coordinates in proportion to $\sqrt{s_i}$. {#c-lh-prop3}
Zhang et al. likewise read decay with a preconditioner $C^{-1}$ as preconditioned descent on a loss with the penalty $\theta^\top C\theta$, which holds only approximately when $C$ adapts, as in Adam or K-FAC. {#c-zhang-c-norm}
For a fixed symmetric positive definite preconditioner $P$, the L2 step $\theta - \eta P(g + \lambda\theta)$ and the decoupled step $(1-\delta)\theta - \eta P g$ agree for all $\theta$ only when $\eta\lambda P = \delta I$; with $P = \operatorname{diag}(1/s_i)$, L2 decays coordinate $i$ by $\eta\lambda\theta_i/s_i$. {#c-precond-not-equal}
The decoupled step is the preconditioned step on $f(\theta) + \frac{\delta}{2\eta}\theta^\top P^{-1}\theta$, which penalizes coordinate $i$ in proportion to $s_i$. {#c-precond-penalty}
On a fixed loss, L2 stops where $\nabla f + \lambda\theta = 0$ whatever $P$ is, while decoupled decay stops where $\nabla f + \frac{\delta}{\eta}P^{-1}\theta = 0$, which depends on $P$. {#c-precond-fixed-point}
So under an adaptive optimizer L2 finds the minimizer of the penalized loss, and decoupled decay finds something else, which in practice works better.
The derivation `preconditioned` gives the steps; the adam page derives the same split for Adam itself, including the first step, in which the L2 term has no effect.
Loshchilov and Hutter also cite the view of Aitchison (2018), in which adaptive methods are Bayesian filtering and a state transition $A = \lambda I$ multiplies the mean by $1 - \lambda$, exactly decoupled decay, whatever the uncertainty in each parameter. {#c-lh-bayesian}

**The Gaussian prior.**
Under the prior $\theta \sim \mathcal N(0, \sigma_p^2 I)$, Bayes' theorem gives

$$ \hat\theta_{\text{MAP}} = \arg\min_\theta \Big[ -\sum_{n=1}^{N}\log p(y_n \mid x_n, \theta) + \frac{1}{2\sigma_p^2}\lVert\theta\rVert^2 \Big] , $$ {#eq-map}

which is L2 regularization with $\lambda = 1/\sigma_p^2$ on the summed loss, or $\lambda = 1/(N\sigma_p^2)$ on the mean loss over $N$ examples. {#c-map-penalty}
For a linear model with Gaussian noise of variance $\sigma^2$, the MAP estimate is ridge regression, $(X^\top X + \lambda I)^{-1}X^\top y$ with $\lambda = \sigma^2/\sigma_p^2$, and it equals the posterior mean. {#c-map-ridge}
Averaged over the prior and the noise, the squared error of the ridge weights is smallest at the same $\lambda = \sigma^2/\sigma_p^2$, for every design matrix. {#c-map-optimal}
As $\lambda \to 0$ the ridge solution tends to the minimum-norm solution $X^+y$, and gradient descent with decay shrinks any part of the weights in the null space of $X$ by $1 - \eta\lambda$ per step. {#c-min-norm}
The derivation `map-gaussian-prior` gives the steps, and its check compares the MAP estimate with the posterior mean and scans $\lambda$.

These are the results of Krogh and Hertz for a linear network.
They prove that weight decay suppresses irrelevant components of the weight vector by choosing the smallest vector that solves the problem, and that if its size is chosen right it suppresses some of the effect of static noise on the targets. {#c-kh-two-effects}
The part of the weights outside the space of the training patterns decays to zero, so an infinitesimal decay picks the solution of smallest norm in the valley of solutions. {#c-kh-smallest}
With noise of variance $\sigma^2$ on the targets they find an optimal decay of $\sigma^2$ over the mean squared size of the teacher's weights, independent of the spectrum of the inputs. {#c-kh-optimal}
For non-linear networks they argue by a local linearization that a small decay picks the point of smallest norm in a valley of zero error, which cannot be proven best in general but is the simplest solution. {#c-kh-nonlinear}

**Scale invariance with normalization.**
If a layer is followed by batch normalization, layer normalization or weight normalization, scaling its weights by $c > 0$ does not change the output, $L(cw) = L(w)$.
van Laarhoven argues that L2 regularization then has no regularizing effect, and acts instead on the scale of the weights and through it on the effective learning rate. {#c-vl-no-effect}
With normalization $L_\lambda(\alpha w) = L_{\lambda\alpha^2}(w)$, so the penalty makes the weights small without making the computed function simpler, and $\lambda$ does not change the optimum. {#c-vl-penalty-scale}
The output does not change with the scale, but the gradient does: it scales like $1/\alpha$. {#c-vl-gradient-scale}
Differentiating $L(cw) = L(w)$ in $w$ gives $\nabla L(cw) = \nabla L(w)/c$, and differentiating in $c$ at $c = 1$ gives $w^\top\nabla L(w) = 0$. {#c-scale-gradient}
Without decay each gradient step is orthogonal to $w$, so it increases the squared norm by exactly $\eta^2\lVert g_t\rVert^2$, and the weights only grow. {#c-norm-grows}
Since the gradient at $w$ is the gradient at $u = w/\lVert w\rVert$ divided by $\lVert w\rVert$, a step of $\eta$ on $w$ moves the direction as a step of

$$ \eta_{\text{eff}} = \frac{\eta}{\lVert w\rVert^2} $$ {#eq-effective-lr}

on $u$. {#c-effective-lr}
van Laarhoven derives this effective learning rate and concludes that by shrinking the weights, weight decay raises the effective learning rate, and that the best learning rate is proportional to $1/\lambda$. {#c-vl-effective-lr}
Without any regularization the weights can grow without bound and the effective learning rate goes to 0, which is why L2 still helps with batch normalization. {#c-vl-unbounded}
He also finds that Adam only partly removes the effect of normalization on the learning rate. {#c-vl-adam}

Li and Arora push this further: for normalized networks, SGD with weight decay and momentum is equivalent to training with a learning rate that grows exponentially, by some factor $1 + \alpha$ per epoch, and the weights blow up while the network stays well behaved. {#c-li-arora}
Their Theorem 1.1 gives the schedule $\tilde\eta_t = \alpha^{-2t-1}\eta$, which without momentum is $(1 - \lambda\eta)^{-2t-1}\eta$, under the condition $\lambda\eta < (1 - \sqrt\gamma)^2$ for momentum $\gamma$. {#c-li-arora-theorem}
Without momentum this follows in one line: if $\tilde w_t = (1 - \eta\lambda)^{-t}w_t$, then the decayed step for $w$ is the undecayed step for $\tilde w$ with learning rate $\eta(1-\eta\lambda)^{-2t-1}$, so the two runs have the same directions and losses at every step. {#c-li-arora-identity}
With decay the growth of eq-effective-lr is held back, and the norm settles.
If the gradient has norm $G/\lVert w\rVert$ at every step, the squared norm tends to

$$ \lVert w\rVert^2 \to \sqrt{\frac{\eta G^2}{\lambda(2 - \eta\lambda)}}, \qquad \eta_{\text{eff}} \to \frac{\sqrt{\eta\lambda(2 - \eta\lambda)}}{G} , $$ {#eq-equilibrium}

and the direction turns by an angle of about $\sqrt{2\eta\lambda}$ per step. {#c-equilibrium}
At equilibrium the learning rate and the decay act only through their product $\eta\lambda$, and a larger decay means faster movement of the direction, not a simpler function.
The derivation `scale-invariance` gives the steps, and its check runs a scale-invariant loss to equilibrium.

## Worked example

Take one weight $\theta_0 = 1$ with $\eta = 0.1$ and $\lambda = 0.5$, so $\delta = \eta\lambda = 0.05$.
The loss gradient is $g_1 = 2$, then $g_2 = -1$.

| optimizer | step 1 | $\theta_1$ | step 2 | $\theta_2$ |
|---|---|---|---|---|
| SGD, L2 | $1 - 0.1(2 + 0.5)$ | 0.75 | $0.75 - 0.1(-1 + 0.375)$ | 0.8125 |
| SGD, decoupled | $0.95 \cdot 1 - 0.1 \cdot 2$ | 0.75 | $0.95 \cdot 0.75 + 0.1$ | 0.8125 |
| momentum 0.5, L2 | $b_1 = 2.5$ | 0.75 | $b_2 = 1.25 - 0.625 = 0.625$ | 0.6875 |
| momentum 0.5, decoupled | $b_1 = 2$ | 0.75 | $b_2 = 1 - 1 = 0$ | 0.7125 |

Without momentum the two forms give the same numbers, as eq-l2-step and eq-decay-step promise.
With momentum $\mu = 0.5$ they agree after the first step and differ by $0.025 = \eta\mu\lambda\theta_0$ after the second: the L2 buffer still carries $\mu\lambda\theta_0 = 0.25$ from the first step.
The tests `test_worked_example` and `test_worked_example_momentum` check these numbers, and the JAX test `test_worked_example` checks them again.

## Hyperparameters

**The coefficient.**
`torch.optim.SGD` defaults to no weight decay, and its `weight_decay` is the L2 coefficient $\lambda$, added to the gradient before the momentum buffer.
`torch.optim.AdamW` defaults to `weight_decay` $= 10^{-2}$, applied as $\delta = \eta\cdot$`weight_decay`.
Krogh and Hertz used $\lambda = 0.00008$ on NetTalk; Krizhevsky et al. used 0.0005 on ImageNet, He et al. and Goyal et al. 0.0001 for ResNets, and LLaMA 0.1 with AdamW, as the Evidence and Practice paragraphs below report.
Because of Proposition 1, the L2 coefficient and the learning rate must be tuned together under SGD with L2; Loshchilov and Hutter find that the best settings lie on a diagonal of the two. {#c-lh-diagonal}
Goyal et al. note that with weight decay, scaling the loss is not the same as scaling the learning rate. {#c-goyal-remark}
Loshchilov and Hutter find that the best decay depends on the length of training, and propose a normalized weight decay that sets $\delta$ from the batch size, the number of training points and the number of epochs. {#c-lh-normalized}

**Which parameters.**
Goyal et al. do not decay the scale $\gamma$ and shift $\beta$ of batch normalization. {#c-goyal-bn}
He et al. (2019) describe "no bias decay", which decays only the weights of convolution and fully connected layers and leaves biases and the batch normalization $\gamma$ and $\beta$ alone. {#c-no-bias-decay}
`param_groups` in the PyTorch code and `decay_mask` in the NumPy and JAX code build these groups.

## Cost

Decoupled decay costs one multiply per parameter at each step, and L2 one multiply-add into the gradient.
Neither needs extra memory, and nothing changes at inference.

## Evidence

On NetTalk, Krogh and Hertz found a clear improvement in generalization error with $\lambda = 0.00008$, less in the error rate, and basically the same curves for 0.00005 and 0.0001. {#c-kh-nettalk}
Krizhevsky et al. trained their ImageNet network with SGD, momentum 0.9 and weight decay 0.0005, and found that this small decay was important for the model to learn: it lowered the training error, not just the test error. {#c-krizhevsky}
Zhang et al. list this observation, and the fact that with batch normalization the decay should in principle have no effect, among the reasons to doubt the L2 reading of weight decay. {#c-zhang-train-acc}

Loshchilov and Hutter found that Adam did not benefit from L2 at all on CIFAR-10, while AdamW rivals SGD and SGDW. {#c-lh-adam-no-benefit}
Decoupled decay beat L2 in Adam for every learning rate schedule they tried, and made the hyperparameter search more separable. {#c-lh-schedules}
Zhang et al. found that weight decay consistently beat L2 where the two differ, and closed most of the gap between first- and second-order optimizers and between small and large batches. {#c-zhang-wd-beats-l2}
They identify three mechanisms of the effect: a higher effective learning rate, an approximate penalty on the norm of the input-output Jacobian, and a smaller effective damping for second-order methods. {#c-zhang-three}
For SGD and Adam on networks with batch normalization, almost all of the effect came from the layers followed by batch normalization, and training without decay at the effective learning rate of the run with decay gave the full effect. {#c-zhang-bn-layers}
Without decay the effective learning rate of a ResNet32 fell by a factor of 10 over the first 50 epochs; with decay it stayed roughly constant within each stage of the schedule. {#c-zhang-elr-curve}

D'Angelo et al. argue that weight decay is never useful as an explicit regularizer, but changes the training dynamics: in vision it strengthens the implicit regularization of SGD, and in language models trained for about one epoch it lowers the training loss and stabilizes training. {#c-dangelo-never}
For GPT-2-124M trained with AdamW, the final training loss was lower with decay 0.1 and 0.3 than with 0, and a plain $\ell_2$ penalty added to the loss was enough. {#c-dangelo-gpt2}
They also find that it enables stable training in bfloat16 precision. {#c-dangelo-bf16}

**Practice.**
He et al. trained ResNets with SGD, momentum 0.9, batch 256 and weight decay 0.0001. {#c-he-resnet}
LLaMA was trained with AdamW, $\beta_1 = 0.9$, $\beta_2 = 0.95$, a cosine schedule, weight decay 0.1 and gradient clipping at 1.0. {#c-llama}

## Limitations and failure modes

Under normalization the L2 penalty does not make the function simpler, so its value is not a measure of model complexity, and its effect is through the learning rate; changing either one changes the training dynamics.
The best coefficient depends on the optimizer, the learning rate, its schedule, the batch size and the length of training, so a value from one setup does not transfer to another.
`weight_decay` means L2 in `torch.optim.SGD` and `torch.optim.Adam` and decoupled decay in `torch.optim.AdamW`; the same number gives different training in each.
The MAP reading needs the penalty to match the loss scale: on a mean loss over $N$ examples, the prior variance is $1/(N\lambda)$, so the same coefficient means a different prior when the data set size changes.
With normalized layers and no decay the weights grow, and the effective learning rate decays on its own, which can look like a schedule that was never set.

## Variants

**L2 regularization** is the penalty form, the default meaning of `weight_decay` in most SGD and Adam implementations.
**Decoupled weight decay** is the multiplicative form: SGDW for SGD with momentum and AdamW for Adam; the adam page gives AdamW in full.
**Normalized weight decay** sets the decay from the number of batch passes, as above.
**Masked decay** applies the decay only to weight matrices, leaving biases and normalization parameters out.
Ridge regression is the same penalty on a linear model, and dropout is another regularizer that can be combined with it; see dropout.

## When to use it

Use some weight decay by default when training neural networks, alongside early stopping, which ends training early rather than pulling the weights back; the papers above report gains on image classifiers and language models, and stable training in low precision.
With Adam, use the decoupled form, AdamW, rather than `torch.optim.Adam(weight_decay=...)`.
With SGD without momentum the two forms are the same; with momentum they differ slightly and either works once tuned.
Leave biases and the parameters of normalization layers out of the decay.
For layers followed by normalization, tune the decay together with the learning rate, since together they set the effective learning rate.
For a Bayesian linear model, choose $\lambda = \sigma^2/\sigma_p^2$ from the noise and prior variances.

## Pitfalls

> [!pitfall]
> **Assuming L2 and decoupled decay are the same with momentum.** They are only for plain SGD; with momentum the L2 term enters the buffer, and the trajectories split after the first step. The tests `test_momentum_second_step_gap` and `test_momentum_l2_differs_from_decoupled` check the gap $\eta\mu\lambda\theta_0$ and the difference on a small network, and `test_matched_decay_fixed_point` checks that $\delta = \eta\lambda/(1-\mu)$ matches the fixed points.

> [!pitfall]
> **`torch.optim.Adam(weight_decay=...)` used as AdamW.** It adds an L2 term that the second moment rescales. The tests `test_adam_l2_is_not_adamw` and `test_adam_l2_is_not_decoupled` check that the two differ, and `test_matches_torch_adamw_and_adam` checks both forms against PyTorch.

> [!pitfall]
> **The decay of AdamW moves with the learning rate.** `torch.optim.AdamW` shrinks the weights by $\eta\cdot$`weight_decay` per step, so a learning rate schedule also schedules the decay, and a per-step decay from a paper must be divided by $\eta$. The test `test_torch_adamw_decay_is_lr_times_weight_decay` checks this for two learning rates.

> [!pitfall]
> **Decaying biases and normalization parameters.** Decaying the scale of a normalization layer shrinks its output, which the next layer must undo. The tests `test_param_groups` and `test_decay_mask` check that only weight matrices are decayed.

> [!pitfall]
> **Writing the penalty twice, or with the wrong scale.** `torch.optim.SGD(weight_decay=wd)` already equals the penalty $\frac{wd}{2}\lVert\theta\rVert^2$; adding the penalty to the loss as well doubles it, and a penalty of $wd\lVert\theta\rVert^2$ without the half doubles it again. On a mean loss the MAP coefficient is $1/(N\sigma_p^2)$, not $1/\sigma_p^2$. The tests `test_torch_sgd_weight_decay_is_l2_penalty`, `test_l2_penalty_gradient` and `test_map_ridge` check these scales.

> [!pitfall]
> **Reading the penalty as a complexity limit under normalization.** With batch normalization the loss does not depend on the scale of the weights, and the decay only changes the effective learning rate. The tests `test_batchnorm_scale_invariance` and `test_norm_grows_without_decay_and_settles_with_it` check the gradient scaling, and the growth of the norm without decay.

> [!pitfall]
> **Changing the decay without the learning rate under normalization.** The decay is equivalent to an exponentially growing learning rate, so it cannot be tuned independently of the schedule. The tests `test_li_arora_exponential_schedule` and `test_li_arora` check the schedule of Li and Arora against SGD with decay.

## History

The $\ell_2$ penalty goes back to Tikhonov's work on ill-posed problems in 1943 and to ridge regression in statistics by Hoerl and Kennard in 1970; Krogh and Hertz (1991) made one of the earliest systematic studies of weight decay for neural networks. {#c-dangelo-history}
Anders Krogh of CONNECT at the Niels Bohr Institute and John A. Hertz of Nordita, both in Copenhagen, wrote "A Simple Weight Decay Can Improve Generalization". {#c-hist-kh}
They noted that a weight decay of this form had long been known to improve generalization, but was not widely recognized. {#c-kh-known}
Twan van Laarhoven posted "L2 Regularization versus Batch and Weight Normalization" in 2017, showing that the penalty acts on the effective learning rate when the network is normalized. {#c-hist-vl}
Ilya Loshchilov and Frank Hutter of the University of Freiburg separated weight decay from L2 regularization in adaptive methods in "Decoupled Weight Decay Regularization", published at ICLR 2019. {#c-hist-lh}
PyTorch added the decoupled form as `torch.optim.AdamW`, and it became the usual optimizer for training large language models.
