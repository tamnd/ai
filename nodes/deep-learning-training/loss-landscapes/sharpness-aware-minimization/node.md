---
tag: 8BNWST
kind: method
slug: sharpness-aware-minimization
name: Sharpness-aware minimization
aliases: [SAM, sharpness aware minimization, sharpness-aware training, m-sharpness]
area: deep-learning-training/loss-landscapes
level: core
summary: >-
  Train on the worst loss in a ball of radius rho around the weights: take a gradient, step rho along it, and apply the gradient found there at the original weights.
purpose: improving generalization by preferring parameters whose whole neighbourhood has low training loss, rather than only the parameters themselves
uses: [stochastic-gradient-descent@m2, minimax@al]
related_to: [weight-decay@al, adversarial-training@al, generalization-error@al, batch-normalization@al, label-smoothing@al]
hyperparameters:
  - {name: neighbourhood size rho, meaning: "the radius of the ball over which the loss is maximized; rho = 0 is the base optimizer", typical: "0.05, chosen by Foret et al. from the grid {0.01, 0.02, 0.05, 0.1, 0.2, 0.5}; 0.1 for training with noisy labels"}
  - {name: norm p, meaning: "the norm of the ball; p = 2 moves along the normalized gradient, p = infinity along its sign", typical: "2, which beat infinity in Foret et al.'s ablation"}
  - {name: m, meaning: "the number of examples that share one perturbation, for example the share of a batch on one accelerator", typical: "the per-accelerator batch; smaller m generalized better in Foret et al."}
  - {name: base optimizer, meaning: "the optimizer that takes the SAM gradient, with its own learning rate, momentum and weight decay", typical: "SGD with momentum 0.9"}
cost:
  train: "two forward and backward passes per step instead of one, so about twice the compute of the base optimizer per step, plus one extra copy of the parameters for the perturbation"
  inference: "none: SAM changes only how the weights are trained"
history:
  year: 2020
  note: "Foret, Kleiner, Mobahi and Neyshabur posted SAM as arXiv 2010.01412 in October 2020 (ICLR 2021), building on the link between flat minima and generalization that goes back to Hochreiter and Schmidhuber and was revived for deep networks by Keskar et al. (ICLR 2017)."
prereqs: [stochastic-gradient-descent@m2, generalization-error@al, overfitting@al]
same_as:
  wikidata: null
  wikipedia: null
  arxiv: ["2010.01412", "1609.04836", "1712.09913", "1703.04933"]
  al: sharpness-aware-minimization
sources:
  - {paper: "Foret et al. (2021)", title: "Sharpness-Aware Minimization for Efficiently Improving Generalization", locator: "pp. 1 to 9, 17 to 19", note: "the SAM objective, the dual-norm perturbation, Algorithm 1, the experiments, m-sharpness and the ablations"}
  - {paper: "Keskar et al. (2017)", title: "On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima", locator: "pp. 1 to 6", note: "sharp and flat minimizers, the large-batch generalization gap and the box sharpness metric"}
  - {paper: "Li et al. (2018)", title: "Visualizing the Loss Landscape of Neural Nets", locator: "pp. 3 to 4, 8", note: "scale invariance and filter-normalized directions for plotting loss surfaces"}
  - {paper: "Dinh et al. (2017)", title: "Sharp Minima Can Generalize For Deep Nets", locator: "pp. 1 to 6", note: "reparametrizations of rectifier networks that make a minimum arbitrarily sharp without changing the function"}
---

Sharpness-aware minimization (SAM) trains a network on the highest loss found in a small ball around its weights instead of on the loss at the weights themselves.
Pierre Foret, Ariel Kleiner, Hossein Mobahi and Behnam Neyshabur introduced it to seek parameters that lie in neighbourhoods having uniformly low loss, posed as a min-max problem on which gradient descent can be run efficiently. {#c-foret-abstract}
Each step costs two gradient computations: one to find the direction in which the loss rises fastest, and one at the point a distance $\rho$ along it.
It improved generalization in the vast majority of the image tasks Foret et al. tried, and gave robustness to label noise on par with methods built for noisy labels. {#c-label-noise-abstract}

## In plain words

Training looks for weights where the loss is low.
Many such weights exist, and they are not equally good: at some the loss sits at the bottom of a narrow pit, at others at the bottom of a wide basin.
If the test data shifts the loss surface a little, weights in a narrow pit can end up on a steep wall, while weights in a wide basin barely notice.
SAM asks, at every step, "how bad could the loss get if I nudged the weights a little in the worst way?" and lowers that number instead of the plain loss.
To answer the question cheaply it nudges the weights once, uphill along the gradient, measures the slope there, and uses that slope for the real step.

## Setting

A model has parameters $w \in \mathbb{R}^d$ and a per-example loss $l(w, x, y) \ge 0$.
On a training set $S$ of $n$ examples drawn from a distribution $D$, the training loss is $L_S(w) = \frac1n \sum_i l(w, x_i, y_i)$ and the population loss is $L_D(w) = \mathbb{E}_{(x, y) \sim D}\, l(w, x, y)$. {#c-setting}
For deep networks $L_S$ is non-convex, with many local and even global minima that can have similar training loss but quite different population loss. {#c-many-minima}
Training normally minimizes $L_S$ with [[m2:stochastic-gradient-descent|stochastic gradient descent]] or Adam, possibly with a regularizer, and leaves the choice among these minima to the optimizer.

Nitish Shirish Keskar and colleagues described the two kinds of minimum this page is about.
They found that large-batch training tends to converge to sharp minimizers of the training function, with many large positive eigenvalues of the Hessian, which generalize less well, while small-batch training converges to flat minimizers with numerous small eigenvalues. {#c-keskar-main}
Following Hochreiter and Schmidhuber, a flat minimizer is one where the function varies slowly in a relatively large neighbourhood, and a sharp one is where it rises rapidly in a small neighbourhood. {#c-flat-def}
The drop in test accuracy they saw with large batches, the generalization gap, was as high as 5% even for smaller networks. {#c-keskar-gap}

## The method

### The objective

Foret et al. start from a PAC-Bayesian bound, stated informally: for any $\rho > 0$, with high probability over $S$, $L_D(w) \le \max_{\lVert \epsilon \rVert_2 \le \rho} L_S(w + \epsilon) + h(\lVert w \rVert_2^2 / \rho^2)$, where $h$ is strictly increasing. {#c-bound}
Writing the right-hand side as $[\max_{\lVert \epsilon \rVert_2 \le \rho} L_S(w + \epsilon) - L_S(w)] + L_S(w) + h(\lVert w \rVert_2^2/\rho^2)$ exposes a sharpness term, the bracket, which measures how fast the training loss can rise when moving from $w$ to a nearby point. {#c-sharpness-term}
Replacing $h$ by $\lambda \lVert w \rVert_2^2$ and allowing any $p$-norm gives the SAM problem

$$ \min_w \; L_S^{\mathrm{SAM}}(w) + \lambda \lVert w \rVert_2^2, \qquad L_S^{\mathrm{SAM}}(w) = \max_{\lVert \epsilon \rVert_p \le \rho} L_S(w + \epsilon), $$ {#eq-sam-objective}

with $\rho \ge 0$ a hyperparameter and $p \in [1, \infty]$; in their experiments $p = 2$ is typically best. {#c-objective}
The $\lambda$ term is ordinary [[al:weight-decay|weight decay]]; the new part is the inner maximum.
It resembles [[al:adversarial-training|adversarial training]], but the worst case is taken over a perturbation of the weights, not of the inputs.

### The perturbation

The inner maximum has no closed form, so it is approximated by a first-order Taylor expansion of $L_S(w + \epsilon)$ around $\epsilon = 0$, which turns it into the maximum of $\epsilon^\top \nabla_w L_S(w)$ over the ball. {#c-taylor}
That linear problem is a classical dual-norm problem, whose solution, with $1/p + 1/q = 1$ and the absolute value and power taken entry by entry, is eq-eps-hat. {#c-eps-formula}

$$ \hat\epsilon(w) = \rho \, \operatorname{sign}\!\big(\nabla_w L_S(w)\big) \, \lvert \nabla_w L_S(w) \rvert^{q-1} \Big/ \big( \lVert \nabla_w L_S(w) \rVert_q^q \big)^{1/p} . $$ {#eq-eps-hat}

For $p = 2$ this just rescales the gradient to norm $\rho$, $\hat\epsilon = \rho\, \nabla L_S / \lVert \nabla L_S \rVert_2$. {#c-p2-footnote}

### The gradient

Substituting $\hat\epsilon(w)$ back gives $\nabla_w L_S^{\mathrm{SAM}}(w) \approx \nabla_w L_S(w + \hat\epsilon(w))$, whose exact derivative also contains a term through $d\hat\epsilon/dw$ that depends on the Hessian.
That term enters only through Hessian-vector products, which can be computed without forming the Hessian, but to further speed up the computation Foret et al. drop the second-order terms, which gives their final approximation eq-sam-grad. {#c-drop-second-order}

$$ \nabla_w L_S^{\mathrm{SAM}}(w) \approx \nabla_w L_S(w) \big|_{w + \hat\epsilon(w)} . $$ {#eq-sam-grad}

In a first experiment, keeping the second-order terms surprisingly made results worse. {#c-second-order-worse}

### The algorithm

Algorithm 1 applies SGD to the SAM objective: sample a batch $B$, compute $\nabla_w L_B(w)$, compute $\hat\epsilon(w)$ from it by eq-eps-hat, compute $g = \nabla_w L_B(w)|_{w + \hat\epsilon(w)}$, and update $w_{t+1} = w_t - \eta g$. {#c-algorithm}
In code: a forward and backward pass at $w$ gives $g_1$; move to $w + \rho g_1 / \lVert g_1 \rVert$, with the norm over all parameters at once; a second pass there gives $g_2$; move back to $w$ and hand $g_2$ to the base optimizer, which applies momentum, weight decay and the learning rate as usual.

$$ w_{t+1} = w_t - \eta\, \nabla L_B\!\Big(w_t + \rho \frac{\nabla L_B(w_t)}{\lVert \nabla L_B(w_t) \rVert}\Big) . $$ {#eq-update}

The step starts from $w_t$, not from $w_t + \hat\epsilon$: the perturbed point is only a probe.

### Many devices and m-sharpness

When training is split across accelerators, Foret et al. give each accelerator an equal share of the batch, compute the SAM gradient on each share independently, and average the results. {#c-parallel}
This changes the objective: it becomes a sum of independent maximizations, each over the loss of a disjoint subset of $m$ examples, instead of one maximization over the whole training set, and they call the resulting measure m-sharpness. {#c-m-sharpness}
Smaller $m$ gave models that generalized better, and m-sharpness also correlated better with the actual generalization gap as $m$ decreased. {#c-m-smaller}

## Derivation

Two derivations support the formulas; each lists its assumptions and has a numeric check.

**The dual-norm perturbation** (`dual-norm-perturbation`).
With $g = \nabla L_S(w) \ne 0$, the vector of eq-eps-hat has $p$-norm $\rho$ and $\hat\epsilon^\top g = \rho \lVert g \rVert_q$, which by Hölder's inequality is the largest value of $\epsilon^\top g$ on the $p$-ball, so it solves the linearized inner problem. {#c-eps-dual}
At $p = 2$ it is $\rho g / \lVert g \rVert_2$ and at $p = \infty$ it is $\rho \operatorname{sign}(g)$. {#c-eps-p2}
It depends only on the direction of $g$: multiplying $g$ by any $c > 0$ leaves it unchanged, so scaling the loss does not change the perturbation. {#c-eps-scale}
The linearized SAM objective is therefore $L_S(w) + \rho \lVert \nabla L_S(w) \rVert_q$, a penalty on the size of the gradient, and it differs from the exact inner maximum by $O(\rho^2)$ when the Hessian is bounded. {#c-first-order-penalty}

**SAM on a quadratic** (`sam-quadratic`).
For $L(w) = \frac12 w^\top H w$ and $p = 2$ the SAM gradient eq-sam-grad is $H(w + \rho Hw / \lVert Hw \rVert) = Hw + \rho H^2 w / \lVert Hw \rVert$. {#c-quad-sam-grad}
This is the exact gradient of $L(w) + \rho \lVert Hw \rVert$, so on a quadratic SAM is gradient descent with a gradient-norm penalty. {#c-quad-penalty}
In the eigenbasis of $H$, with eigenvalues $\lambda_i$ and $c_i = v_i^\top w$, the $i$-th component of the SAM gradient is $\lambda_i c_i (1 + \rho \lambda_i / \lVert Hw \rVert)$: SAM amplifies the gradient most along the sharpest directions, so with a small step it removes the sharp components of $w$ faster than gradient descent does. {#c-quad-eigen}
If $H$ is positive definite the SAM gradient is never zero away from the minimum, so the update never comes to rest there. {#c-quad-no-rest}
In one dimension with $0 < \eta\lambda < 2$ the update $w \leftarrow w - \eta\lambda(w + \rho \operatorname{sign} w)$ settles into a 2-cycle between $\pm\, \eta\lambda\rho/(2 - \eta\lambda)$, so with fixed $\eta$ and $\rho$ the iterates bounce around the minimum at a distance that scales with $\rho$. {#c-quad-cycle}
On the valley $L(a, b) = \frac12(ab - 1)^2$ every point with $ab = 1$ is a global minimum, with Hessian eigenvalues $0$ and $a^2 + b^2 \ge 2$, so the flattest minima are $a = b = \pm 1$. {#c-valley}

## Worked example

Take $L(w) = \frac12 w^\top H w$ with $H = \operatorname{diag}(4, 1)$, $w = (1, 2)$, $\rho = 0.5$ and learning rate $\eta = 0.1$.
The gradient is $g = Hw = (4, 2)$ with norm $\sqrt{20} = 4.4721$, so $\hat\epsilon = 0.5\, g / 4.4721 = (0.4472, 0.2236)$.
The probe point is $w + \hat\epsilon = (1.4472, 2.2236)$ and the gradient there is $H(w + \hat\epsilon) = (5.7889, 2.2236)$.
Compared with $g$, the sharp coordinate's gradient grew by the factor $1 + 0.5 \cdot 4 / 4.4721 = 1.4472$ and the flat coordinate's by only $1 + 0.5 \cdot 1 / 4.4721 = 1.1118$, as the eigen-component formula says.
The SAM step gives $w - 0.1 \cdot (5.7889, 2.2236) = (0.4211, 1.7776)$ with loss $1.9347$; plain gradient descent gives $(0.6, 1.8)$ with loss $2.34$, starting from loss $4$.
The linearized SAM objective at $w$ is $4 + 0.5 \cdot 4.4721 = 6.2361$, while the loss at the probe point is $6.6611$ and the true maximum on the circle of radius $0.5$ is about $6.678$; the gaps are the second-order terms that the linearization ignores.

The valley $\frac12(ab - 1)^2$ shows the effect SAM is built for.
Starting from $(a, b) = (3, 0.4)$ with $\eta = 0.02$, gradient descent stops at the first minimum it reaches, $(2.992, 0.334)$, where the top Hessian eigenvalue $a^2 + b^2$ is $9.06$.
SAM with $\rho = 0.05$ also reaches the valley but keeps sliding along it: after 20,000 steps it is at $(1.272, 0.787)$ with $ab = 1.0008$ and top eigenvalue $2.24$, close to the flattest value $2$.
Both runs are in the check of `sam-quadratic` and in the tests of the implementations.

## Hyperparameters

The neighbourhood size $\rho$ is SAM's one new hyperparameter; Foret et al. tuned it by grid search over $\{0.01, 0.02, 0.05, 0.1, 0.2, 0.5\}$ on a validation split of 10% of the training set. {#c-rho-grid}
They report $\rho = 0.05$ as a solid default and give the scores of all their experiments at that value without further tuning. {#c-rho-default}
For ImageNet they used $\rho = 0.05$, picked by searching over values for a 100-epoch ResNet-50. {#c-rho-imagenet}
With noisy labels they used $\rho = 0.1$ for every noise level except 80%, where $\rho = 0.05$ gave more stable convergence. {#c-rho-noise}

The norm $p$ is almost always 2.
In an ablation on a wide ResNet on CIFAR-10, $p = 2$ beat $p = \infty$, and adversarial perturbations beat random perturbations of the same Euclidean norm. {#c-p-ablation}

Because $\hat\epsilon$ has a fixed length, $\rho$ is measured in the units of the weights; the right value depends on the parametrization, and it interacts with the learning rate as the 2-cycle in the derivation shows.
The base optimizer's learning rate, momentum and weight decay are tuned as without SAM.

## Cost

Each SAM update needs two backpropagations, one to compute $\hat\epsilon(w)$ and one for the final gradient. {#c-two-backprop}
So a SAM step costs about two steps of the base optimizer, and the extra memory is one copy of the perturbation.
For a fair comparison Foret et al. let each non-SAM run train for twice as many epochs as the SAM run and reported its best score over the standard and the doubled epoch count. {#c-fair-epochs}
Compared with earlier sharpness measures, which are hard to compute and differentiate, SAM scales because it needs only two gradient computations per iteration. {#c-scalable}
Inference is unchanged.
SAM combines with other regularizers: the ImageNet runs below used it together with [[al:label-smoothing|label smoothing]] of 0.1, which changes the targets while SAM changes where the gradient is taken.

## Evidence

**CIFAR.**
On CIFAR-10 and CIFAR-100 SAM improved generalization in every setting tested; a WideResNet reached 1.6% test error with SAM against 2.2% without, and a PyramidNet with ShakeDrop reached 10.3% error on CIFAR-100. {#c-cifar}

**ImageNet.**
SAM improved ResNets on ImageNet, lowering the top-1 error of ResNet-152 from 20.3% to 18.4%. {#c-imagenet}
SAM kept improving as training grew from 200 to 400 epochs, while standard training overfitted. {#c-epochs}

**Finetuning.**
Finetuning EfficientNets with SAM improved on finetuning without it in every case, reaching 0.30% error on CIFAR-10, 3.92% on CIFAR-100 and 11.39% on ImageNet. {#c-finetune}

**Label noise.**
On CIFAR-10 with randomly flipped labels, SAM alone beat every prior method aimed at label noise except MentorMix, and bootstrapped SAM was comparable to MentorMix. {#c-label-noise}

**Curvature.**
For a WideResNet40-10 on CIFAR-10 trained without batch normalization, the largest Hessian eigenvalue at convergence was about 24 without SAM and 1.0 with it, and the ratio $\lambda_{\max}/\lambda_5$ was up to 11.4 without SAM and 2.6 with it. {#c-hessian}

**Inner steps.**
For most of training one projected gradient step, as in standard SAM, approximates well the perturbation found with several steps of inner maximization. {#c-one-step}
Near convergence the one-step approximation becomes weaker, but test accuracy was not strongly affected by the number of inner steps. {#c-inner-steps}

**Seeing flatness.**
Plots of the loss along random directions are a common way to look at sharpness, and Hao Li and colleagues proposed a way to make them comparable.
With filter-normalized directions, the sharpness of the plots correlates well with generalization error, while plots without the normalization can be very misleading. {#c-li-correlates}
In their plots, visually flatter minimizers consistently corresponded to lower test error. {#c-li-flatter}

## Limitations and failure modes

**Sharpness depends on the parametrization.**
Laurent Dinh and colleagues proved that for deep networks with rectifier units, the symmetries of parameter space give equivalent models with arbitrarily sharper minima. {#c-dinh-abstract}
For a one-hidden-layer rectifier network the map $T_\alpha: (\theta_1, \theta_2) \mapsto (\alpha\theta_1, \alpha^{-1}\theta_2)$, $\alpha > 0$, changes the parameters without changing the function. {#c-dinh-alpha}
As a result any minimum with a non-zero Hessian is observationally equivalent to a minimum whose Hessian has an arbitrarily large spectral norm. {#c-dinh-spectral}
SAM's ball has a fixed radius $\rho$ in parameter space, so the sharpness it penalizes is of this kind: it changes under such rescalings even though the network computes the same function.

**The objective is approximated twice.**
The inner maximum is replaced by one linearized step, and the second-order terms of its gradient are dropped.
Near convergence the one-step perturbation finds a smaller loss increase than a few steps of projected gradient ascent would. {#c-weaker-near-convergence}

**It does not settle.**
With a fixed $\rho$ the SAM gradient is non-zero at every point near a strict minimum, as `sam-quadratic` shows, so the weights keep bouncing at a distance of order $\rho$ unless the learning rate decays.

## Variants

**Second-order SAM** keeps the term through $d\hat\epsilon/dw$, computed with Hessian-vector products; in Foret et al.'s first experiment it trained worse than the first-order version, and the model without the second-order terms reached a lower test error. {#c-second-order-appendix}

**Several inner steps** replace the single gradient step by a few steps of projected gradient ascent on $L(w + \epsilon)$.

**$p = \infty$** perturbs every weight by $\pm\rho$, the sign of its gradient; **random perturbations** draw $\hat\epsilon$ from a Gaussian rescaled to norm $\rho$; both did worse than $p = 2$ in the ablation.

## When to use it

Use SAM when generalization, not training speed, is the bottleneck: image classification from scratch, finetuning a pretrained network on a small dataset, or training with noisy labels, the settings where Foret et al. report gains.
It wraps any first-order optimizer, so it can be tried by changing the training step and tuning one number, $\rho$, starting at 0.05.
When compute is fixed, compare against the base optimizer trained for twice as many steps.
Skip it when a step is already as expensive as the budget allows and the model underfits, since SAM spends its second pass on generalization, not on fitting.

## Pitfalls

- **Stepping from the probe point.** The base optimizer must update the original weights $w$ with the gradient taken at $w + \hat\epsilon$; leaving the weights at $w + \hat\epsilon$ is a different and worse algorithm.
- **Per-tensor normalization.** $\hat\epsilon$ uses the norm of the whole gradient vector; normalizing each tensor separately gives each layer its own ball of radius $\rho$.
- **Batch-norm statistics.** The second forward pass runs at perturbed weights; if it updates batch-norm running statistics, they are computed twice per step and partly at the wrong weights.
- **Gradient synchronization.** Averaging the first gradient across devices before perturbing gives one perturbation per batch; computing it per device gives m-sharpness with $m$ equal to the per-device batch, which is what the reported results used.
- **A zero gradient.** $\hat\epsilon$ divides by $\lVert g \rVert$; add a small constant or skip the perturbation when the gradient vanishes.
- **Reading sharpness off plots.** Loss plots along random directions compare minima only if the directions are filter-normalized, since rectifier networks can be rescaled without changing their function.

## History

The search for flat minima goes back to Hochreiter and Schmidhuber in 1995. {#c-history-hochreiter}
Keskar et al. revived the idea in 2016 (ICLR 2017) by linking the generalization gap of large-batch training to sharp minimizers, measured by the largest value of the loss in a small box around a solution relative to its value at the solution. {#c-keskar-metric}
Dinh et al. (2017) countered that common notions of flatness cannot by themselves explain generalization in deep networks, since a rescaling can make a minimum arbitrarily sharp.
Li et al. (2018) introduced filter normalization to compare loss surfaces despite such rescalings.
Given a random Gaussian direction $d$, they rescale each filter $d_{i,j}$, the $j$-th filter of the $i$-th layer, to the Frobenius norm of the matching filter $\theta_{i,j}$ of the weights, as in eq-filter-norm. {#c-li-filter}

$$ d_{i,j} \leftarrow \frac{d_{i,j}}{\lVert d_{i,j} \rVert} \lVert \theta_{i,j} \rVert $$ {#eq-filter-norm}

They treat a fully connected layer as a convolution with a $1 \times 1$ output, so that a filter is the set of weights producing one neuron. {#c-li-fc}
They kept batch-normalization parameters constant, applying no random direction to them. {#c-li-bn}
Foret et al. turned the flatness idea into a practical optimizer in 2020; their Figure 1 contrasts a sharp minimum to which a ResNet trained with SGD converged with a wide minimum reached by the same ResNet trained with SAM. {#c-figure1}
