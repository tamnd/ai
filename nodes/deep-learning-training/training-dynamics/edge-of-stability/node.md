---
tag: NEW-edge-of-stability
kind: finding
slug: edge-of-stability
name: Edge of stability
aliases: [EoS, progressive sharpening, edge of stability regime, self-stabilization]
area: deep-learning-training/training-dynamics
level: core
summary: >-
  Full-batch gradient descent on neural networks raises the sharpness of the loss to 2/η, the stability limit of a quadratic, then holds it there while the loss still falls, non-monotonically.
history:
  year: 2021
  note: "Named and measured by Cohen, Kaur, Li, Kolter and Talwalkar (ICLR 2021, arXiv 2103.00065), after the break-even point of Jastrzębski et al. (ICLR 2020) and the catapult phase of Lewkowycz et al. (arXiv 2003.02218)."
prereqs: [backpropagation@al, loss-function@al, mean-squared-error@al, learning-rate-schedule@al]
see_also: [sharpness-aware-minimization@al, neural-tangent-kernel@al, batch-normalization@al]
same_as:
  arxiv: ["2103.00065", "2003.02218", "2209.15594", "2002.09572"]
sources:
  - {paper: "Cohen et al. (2021)", title: "Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability", locator: "pp. 1-11, 17", note: "the name, progressive sharpening, the experiments, the momentum thresholds and the caveats"}
  - {paper: "Lewkowycz et al. (2020)", title: "The large learning rate phase of deep learning: the catapult mechanism", locator: "pp. 3-7", note: "the lazy, catapult and divergent phases and the solvable two-layer model"}
  - {paper: "Damian, Nichani and Lee (2023)", title: "Self-Stabilization: The Implicit Bias of Gradient Descent at the Edge of Stability", locator: "pp. 1-3, 6-8, 12-15", note: "self-stabilization through the cubic term, the two-variable model and its conserved quantity"}
  - {paper: "Jastrzębski et al. (2020)", title: "The Break-Even Point on Optimization Trajectories of Deep Neural Networks", locator: "pp. 1-5", note: "the break-even point and the conjectures about SGD"}
claim: >-
  Full-batch gradient descent on neural networks typically raises the largest Hessian eigenvalue of the training loss until it reaches 2/η, then keeps it hovering at or just above 2/η while the loss decreases over long timescales but not monotonically.
first_reported: {work: cohen-2021-gradient-descent-edge-stability, note: "named the regime and showed it for fully-connected, convolutional, VGG, ResNet and Transformer networks with MSE and cross-entropy, with and without momentum"}
replications:
  - {work: damian-2023-self-stabilization-edge-stability, note: "an MLP and a CNN on 5k CIFAR10 images and a 2-layer Transformer on SST2, with MSE, tracked by their predicted dynamics"}
scale_range:
  - {range: "networks on a 5k subset of CIFAR-10 up to a VGG with batch normalization on the full CIFAR-10", work: cohen-2021-gradient-descent-edge-stability}
  - {range: "3-layer MLP and CNN on a 5k CIFAR10 subset, 2-layer Transformer on SST2", work: damian-2023-self-stabilization-edge-stability}
failures:
  - {work: cohen-2021-gradient-descent-edge-stability, note: "with SGD the sharpness does not settle at a value predicted by the hyperparameters; wide, shallow or easy problems sharpen little"}
explanations:
  - {name: "the catapult", summary: "When the curvature at the start is above 2/η, the loss grows while the curvature falls below 2/η, and training then converges to a flatter minimum; the edge of stability may be a never-ending series of such small catapults.", sources: [lewkowycz-2020-large-learning-rate-phase, cohen-2021-gradient-descent-edge-stability]}
  - {name: "self-stabilization", summary: "The growing oscillation along the top eigenvector, through the cubic term of the loss, moves the weights against the gradient of the sharpness, which pulls the sharpness back below 2/η; gradient descent ends up following projected gradient descent under the constraint S ≤ 2/η.", sources: [damian-2023-self-stabilization-edge-stability]}
---

The edge of stability is the finding that full-batch gradient descent on a neural network does not stay in the region where classical optimization theory says it is safe.
The sharpness, the largest eigenvalue of the Hessian of the training loss, rises during training until it reaches $2/\eta$, where $\eta$ is the step size.
On a quadratic, gradient descent with that step size would start to diverge there.
On a network it does not: the sharpness stops rising and hovers at or just above $2/\eta$, and the training loss keeps going down over long stretches while it goes up and down from step to step.
Cohen, Kaur, Li, Kolter and Talwalkar named the regime in 2021 and measured it on many architectures; the catapult mechanism of Lewkowycz et al. (2020) and the self-stabilization analysis of Damian, Nichani and Lee (2023) are the two main accounts of why it holds.

## In plain words

Gradient descent takes steps downhill of a fixed length.
In a narrow valley, a step that is too long overshoots the floor and lands higher up on the other wall, and the next step overshoots back, each time further.
How narrow is too narrow depends on the step size: if the valley's curvature is above $2/\eta$, the bouncing grows.

Training a network, the valley tends to get narrower as training goes on.
With a small enough step, the narrowing never reaches the limit.
With a usual step, it does, and the bouncing starts.
What happens next is the surprise: the bouncing does not blow up.
It pushes the weights toward places where the valley is a little wider, which brings the curvature back to the limit, and the run goes on downhill with the curvature pinned near $2/\eta$ and the loss jittering.

## The finding

On a quadratic $\tfrac12 x^\top H x$, gradient descent multiplies the coordinate of the iterate along each eigenvector $q_i$ of $H$ by a fixed factor per step:

$$ \langle q_i, x_{t+1}\rangle = (1 - \eta\lambda_i)\,\langle q_i, x_t\rangle. $$ {#eq-mode-factor}

For gradient descent on $\tfrac12 x^\top H x$, the coordinate along each eigenvector is multiplied by $1 - \eta\lambda_i$ at every step. {#c-quad-modes}
With $H$ positive definite, the run converges from every start if and only if $\eta\lambda_{\max} < 2$; at equality the top coordinate flips sign without shrinking, and above it grows. {#c-quad-threshold}
With eigenvalues 20 and 1, the top factor is $-0.8$ at $\eta = 0.09$ and $-1.2$ at $\eta = 0.11$, so after 100 steps the top coordinate has shrunk by about $2.04 \cdot 10^{-10}$ or grown by about $8.28 \cdot 10^{7}$. {#c-quad-example}
The derivation `quadratic-stability` gives the steps.
Cohen et al. define sharpness strictly as the maximum eigenvalue of the training loss Hessian, and they do not tie it to generalization. {#c-cohen-sharpness}

Cohen et al. (2021) then report two things about networks.
So long as the sharpness is below $2/\eta$, gradient descent has an overwhelming tendency to keep increasing it; they call this progressive sharpening, and the brief decreases they see, mostly early, are followed by a return to the rise. {#c-cohen-progressive}
Their illustrations use full-batch gradient descent on a subset of 5,000 CIFAR-10 examples while the sharpness is monitored. {#c-cohen-subset}
For a fully-connected network with two hidden layers of width 200 and tanh activations, trained with mean squared error at the small step size $2/600$ to 99% training accuracy, the sharpness rises throughout except for a brief dip at the start. {#c-cohen-fc-setup}

At the larger step $\eta = 0.01$ the same network reaches sharpness $2/\eta = 200$, the breakeven point in the terms of Jastrzębski et al. (2020). {#c-cohen-breakeven}
From there the iterates oscillate along the top eigenvector $q_1$ with growing size, as on an unstable quadratic; the loss spikes around iteration 215, and then, instead of diverging, the run keeps reducing the loss, though not monotonically. {#c-cohen-oscillate}
In general, after the sharpness first crosses $2/\eta$, gradient descent enters the edge of stability: the sharpness hovers right at or just above $2/\eta$, and the loss is non-monotone over short timescales and decreases over long ones. {#c-cohen-regime}
Above $2/\eta$ means a minuscule amount for the mean squared error, and a small but noticeable gap for cross-entropy. {#c-cohen-gap}
The same is stated across the paper's experiments: full-batch gradient descent typically operates in this regime. {#c-cohen-eos}

The constraint behaves as if it were active.
Training at step $2/200$ to the edge of stability and cutting the step to $2/300$ at iteration 6,000 makes the sharpness rise again at once, until it stops at the new $2/\eta$. {#c-cohen-lr-drop}
With momentum, the limit moves to the maximum stable sharpness of a quadratic,

$$ \mathrm{MSS}_{\text{Polyak}} = \frac{2 + 2\beta}{\eta}, \qquad \mathrm{MSS}_{\text{Nesterov}} = \frac{2 + 2\beta}{\eta\,(1 + 2\beta)}. $$ {#eq-mss}

On quadratics, gradient descent with step $\eta$ and momentum $\beta$ diverges above these values. {#c-quad-momentum}
With $\eta = 0.01$ and several values of $\beta$ for each kind of momentum, the sharpness of the network rises to the corresponding threshold and plateaus there. {#c-cohen-momentum}
The next few Hessian eigenvalues also rise until they plateau near $2/\eta$. {#c-cohen-next-eigenvalues}

For many networks, gradient descent at any step size follows the gradient flow path, computed with a Runge-Kutta solver, until the point where the sharpness on that path reaches $2/\eta$; a step below $2/\lambda_{\max}$ of the whole path never reaches the edge of stability, and a step in $[2/\lambda_{\max}, 2/\lambda_0]$ leaves the path shortly after that point. {#c-cohen-gradient-flow}

## Evidence

Cohen et al. repeated the experiment over eleven networks on the 5k CIFAR-10 subset: fully-connected networks and convolutional networks with max or average pooling, with tanh, ReLU and ELU, plus softplus and hardtanh for the fully-connected ones, each with cross-entropy and with mean squared error, and saw the same behaviour in each case. {#c-cohen-architectures}
Outside images, a Transformer on WikiText-2, one-hidden-layer tanh networks on a synthetic one-dimensional regression and deep linear networks on Gaussian data all show the sharpness rising to and hovering at $2/\eta$. {#c-cohen-tasks}

The effect is large on standard networks.
A VGG with batch normalization trained on all of CIFAR-10 by gradient flow sharpens from 6.3 to 2227.6, so a stable step would have to be below 0.000897; gradient descent at $\eta = 0.16$ trains it in 329 iterations, so a stable step would be slower by a factor of at least 50.5. {#c-cohen-vgg}
On the 5k subset with cross-entropy, a stable step took 3700 iterations against 1000 at a larger step, and in every case they observed the stable step sizes to be the slower ones. {#c-cohen-stable-slow}

Damian, Nichani and Lee (2023) reproduced the regime.
Their MLP on CIFAR10 at $\eta = 2/100$ became unstable after about 2200 steps, and its sharpness then hovered at 100. {#c-damian-figure-one}
They also tested a 3-layer MLP and CNN on 5k CIFAR10 images and a 2-layer Transformer on SST2, all with mean squared error, and found the runs settling at the fixed point that their model predicts. {#c-damian-experiments}

## Scope and conditions

**Batch size.**
The finding is about full-batch gradient descent.
Under SGD the sharpness does not settle at a fixed value, although large steps and small batches still lead to regions of lower sharpness. {#c-cohen-sgd}
Jastrzębski et al. (2020) had described the SGD side: early in training, SGD reaches a break-even point where it oscillates along the most curved direction. {#c-jastr-break-even}
They conjectured that the largest sharpness and the largest spectral norm of the gradient covariance reached along the run are smaller for larger learning rates or smaller batches. {#c-jastr-conjecture}
Cohen et al. credit them with conjecturing the full-batch regime, but note that they ran no full-batch experiments and so did not observe it. {#c-cohen-on-jastr}

**Architecture and task.**
Progressive sharpening is weaker for wider networks, for shallow networks and on easy tasks. {#c-cohen-width}
With cross-entropy the sharpness usually falls at the end of training, because the curvature of the loss is small once the margins of the predictions are large. {#c-cohen-cross-entropy}
With ReLU or hardtanh, the plateau sometimes starts a little below $2/\eta$, and with batch normalization the sharpness at the iterates themselves can stay below $2/\eta$ while the sharpness between them sits just above. {#c-cohen-caveats}

**The quadratic model fails.**
At the edge of stability gradient descent on the network makes steady progress where gradient descent on its quadratic Taylor model would diverge, so no quadratic model explains the regime. {#c-cohen-taylor}

## Explanations

**The quadratic picture.**
On a quadratic, if the curvature $a$ exceeds $2/\eta$ the iterates oscillate around the minimum with growing size, and each eigen-coordinate evolves on its own. {#c-cohen-quadratic}
This explains the threshold and the oscillation along $q_1$, but not why the oscillation stops.

**The catapult.**
Lewkowycz et al. (2020) studied what happens when the curvature at the start is already above $2/\eta$.
Their curvature $\lambda$ is the largest eigenvalue of the neural tangent kernel, which for wide networks is close to the top Hessian eigenvalue. {#c-lewk-curvature}
In their catapult phase, $2/\lambda_0 < \eta < \eta_{\max}$, the loss first grows exponentially while the curvature drops fast, until it settles below $2/\eta$; the run then converges to a flatter minimum than in the lazy phase $\eta < 2/\lambda_0$, after a number of steps of order $\log n$ in the width $n$. {#c-lewk-phases}
In their solvable model, $f = n^{-1/2} v^\top u$ on one example with label 0, one gradient step gives exactly

$$ f_{t+1} = \Big(1 - \eta\lambda_t + \frac{\eta^2 f_t^2}{n}\Big) f_t, $$ {#eq-catapult-f}

$$ \lambda_{t+1} = \lambda_t + \frac{\eta f_t^2}{n}\,(\eta\lambda_t - 4). $$ {#eq-catapult-lambda}

These are the exact one-step updates of that model, with $\lambda = (\lVert u\rVert^2 + \lVert v\rVert^2)/n$. {#c-catapult-update}
So a step with $f \ne 0$ lowers the curvature when $\eta\lambda < 4$ and raises it when $\eta\lambda > 4$, which puts the edge of the divergent phase at $4/\lambda_0$. {#c-catapult-decrease}
The derivation `catapult-update` gives the steps.
For ReLU networks the measured limit was about $12/\lambda_0$, and the rise of the loss lasted fewer than 10 steps. {#c-lewk-max-rate}
Cohen et al. suggest viewing the edge of stability as a never-ending series of micro-catapults. {#c-cohen-catapult}

**Self-stabilization.**
Damian et al. (2023) give a mechanism with a cubic Taylor expansion: as the iterates move out along the top eigenvector, the cubic term lowers the curvature until stability returns, so that gradient descent in effect follows projected gradient descent under the constraint $S(\theta) \le 2/\eta$. {#c-damian-self-stabilization}
They write progressive sharpening as $\nabla L(\theta) \cdot \nabla S(\theta) < 0$. {#c-damian-progressive}
The key identity is that, when the top eigenvalue of $\nabla^2 L(\theta)$ is unique, the sharpness is differentiable and $\nabla S(\theta) = \nabla^3 L(\theta)(u, u)$, the third derivative along the top eigenvector $u$. {#c-damian-lemma}
With $x_t$ the deviation from a reference point $\theta^\star$ in the $u$ direction and $y_t$ the sharpness minus $2/\eta$, their simplified dynamics are

$$ x_{t+1} \approx -(1 + \eta y_t)\,x_t, \qquad y_{t+1} \approx y_t + \eta\Big(\alpha - \frac{\beta x_t^2}{2}\Big), $$ {#eq-eos-map}

with $\alpha = -\nabla L \cdot \nabla S$ and $\beta = \lVert\nabla S\rVert^2$, so the sharpness begins to decrease once $x_t^2 > 2\alpha/\beta$. {#c-damian-map}
In continuous time, $X' = XY$ and $Y' = \alpha - \beta X^2/2$ keep

$$ g(X, Y) = h\Big(\frac{\beta X^2}{2\alpha}\Big) + \frac{Y^2}{\alpha}, \qquad h(z) = z - \log z - 1, $$ {#eq-conserved}

constant, with the fixed point $(\sqrt{2\alpha/\beta}, 0)$. {#c-ode-conserved}
The derivation `self-stabilization-ode` gives the steps.
So the sharpness cycles above and below $2/\eta$ while the oscillation grows and shrinks, which is the jitter of the loss.
Damian et al. explain the long-run decrease of the loss by gradient descent staying close to the constrained path, and they report that full-batch gradient descent generalized better with large learning rates in a standard setting. {#c-damian-loss}

The two accounts are compatible: the catapult describes one escape from a region that is too sharp, and self-stabilization describes the repeated small escapes that keep the sharpness at the limit.

## Reproduction here

The sharpness of a network is computed here without forming the Hessian.
A central difference of two gradients gives a Hessian-vector product, exact on a quadratic and with error of order $\epsilon^2$ otherwise:

$$ H v \approx \frac{\nabla L(\theta + \epsilon v) - \nabla L(\theta - \epsilon v)}{2\epsilon}. $$ {#eq-hvp}

This central difference is exact for a quadratic loss and has an error of order $\epsilon^2$ otherwise. {#c-hvp-central}
Power iteration repeats $q \leftarrow Hq/\lVert Hq\rVert$ and reads off the Rayleigh quotient,

$$ \lambda \approx q^\top H q, $$ {#eq-power}

which converges to the eigenvalue of largest modulus, the sharpness when the top eigenvalue dominates the negative ones. {#c-power-iteration}
The derivation `power-iteration-hvp` gives the steps; the PyTorch code gets the products from autograd instead.

The NumPy and PyTorch implementations train a 5-16-16-1 tanh MLP on 64 points of a sine regression with full-batch gradient descent at $\eta = 0.25$, so $2/\eta = 8$, and measure the sharpness every 100 steps.
With seed 0 in NumPy, the sharpness starts near 3.2, overshoots to about 9.4 by step 400, and from step 1000 to 1600 stays between 7.8 and 8.1, while the loss goes up on 490 of 1600 steps and still falls from 0.59 to 0.0013.
At $\eta = 0.05$ the sharpness also rises, to about 8 by step 1600, far below $2/\eta = 40$, and the loss never goes up.
The tests check these runs with fixed seeds; across ten seeds tried while writing, the late sharpness stayed within 7% of $2/\eta$.

## Implications

Cohen et al. argue that analyses which assume L-smoothness, a sharpness below $2/\eta$ along the path, cannot cover gradient descent at step sizes that reach the edge of stability, nor can analyses that assert monotone descent. {#c-cohen-l-smooth}
The rule of thumb of setting the step to $1/\lambda_t$ for the current sharpness did worse than a fixed step of $1/\lambda_0$ set from the sharpness at the start. {#c-cohen-step-rule}
Lewkowycz et al. found that, at a fixed compute budget, models trained in the catapult phase matched or beat those trained in the lazy phase. {#c-lewk-budget}
For practice, the sharpness at the end of training is set mostly by the step size, not by the starting point, and a learning rate cut lets it climb again.

## Open questions

Cohen et al. write that they do not know why progressive sharpening happens, nor whether sharp solutions differ in any important way from flat ones. {#c-cohen-unknown}
The self-stabilization analysis covers one unstable eigenvalue; with several, the dynamics look much more chaotic. {#c-damian-multiple}
How the full-batch picture carries over to SGD, where the sharpness does not settle, is also open.

## History

Jastrzębski and colleagues presented the break-even point at ICLR 2020. {#c-hist-jastr}
Lewkowycz, Bahri, Dyer, Sohl-Dickstein and Gur-Ari posted the catapult paper as arXiv 2003.02218 on 4 March 2020. {#c-hist-lewk}
Cohen, Kaur, Li, Kolter and Talwalkar published the paper that named the edge of stability at ICLR 2021. {#c-hist-cohen}
Damian, Nichani and Lee posted the second version of the self-stabilization analysis, the one cited here, as arXiv 2209.15594v2 on 10 April 2023. {#c-hist-damian}

## References
