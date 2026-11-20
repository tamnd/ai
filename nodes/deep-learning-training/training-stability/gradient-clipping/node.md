---
tag: WH0JRN
kind: method
slug: gradient-clipping
name: Gradient clipping
aliases: [gradient norm clipping, norm clipping, clip by norm, clipped gradient descent, value clipping, adaptive gradient clipping]
area: deep-learning-training/training-stability
level: core
summary: >-
  Before each update, scale the gradient down so that its norm is at most a threshold c, keeping its direction, so that one step stays bounded when the gradient suddenly explodes.
purpose: keeping each optimization step bounded when the gradient norm spikes, as with exploding gradients in recurrent networks and large models
uses: [stochastic-gradient-descent@m2]
related_to: [exploding-gradient-problem@al, vanishing-gradient-problem@al, backpropagation-through-time@al]
hyperparameters:
  - {name: threshold c, meaning: "the largest norm a gradient may have; a gradient with a larger norm is rescaled to norm c", typical: "1.0 on the global norm for GPT-3; 0.25 for the AWD-LSTM language model; 6, 8 and 45 in Pascanu et al., depending on the task and on whether the loss is a mean or a sum over time"}
  - {name: granularity, meaning: "what the norm is taken over: the whole gradient (global norm), each layer, each unit, or each entry (value clipping)", typical: "the global norm in most training code; unit-wise for adaptive gradient clipping"}
  - {name: AGC lambda, meaning: "in adaptive gradient clipping, the largest allowed ratio of the norm of a unit's gradient to the norm of its weights", typical: "0.01 to 0.16 in Brock et al., smaller for larger batches; 0.01 at batch size 4096"}
cost:
  train: "one reduction over the gradient to get its norm, and one multiply per parameter when the norm is above the threshold; the global norm needs every parameter's gradient before any update"
  inference: "none: clipping acts only on training updates"
history:
  year: 2012
  note: "Clipping was introduced for recurrent language models by Mikolov et al. (2010) and used element-wise in Mikolov's 2012 PhD thesis; Pascanu, Mikolov and Bengio (arXiv 1211.5063, November 2012; ICML 2013) proposed rescaling by the norm and justified it with the geometry of the loss surface."
prereqs: [stochastic-gradient-descent@m2, backpropagation-through-time@al, exploding-gradient-problem@al]
same_as:
  wikidata: null
  wikipedia: null
  arxiv: ["1211.5063", "1905.11881", "2102.06171"]
  al: gradient-clipping
sources:
  - {paper: "Pascanu, Mikolov and Bengio (2013)", title: "On the difficulty of training Recurrent Neural Networks", locator: "pp. 1 to 12", note: "exploding gradients, the wall in the loss surface, norm clipping (Algorithm 1) and its experiments"}
  - {paper: "Zhang, He, Sra and Jadbabaie (2020)", title: "Why gradient clipping accelerates training: A theoretical justification for adaptivity", locator: "pp. 1 to 8, 18 to 21", note: "(L0, L1)-smoothness and the convergence rates of clipped and fixed-step gradient descent"}
  - {paper: "Brock, De, Smith and Simonyan (2021)", title: "High-Performance Large-Scale Image Recognition Without Normalization", locator: "pp. 3 to 5", note: "adaptive gradient clipping (AGC) and its ablations"}
  - {paper: "Brown et al. (2020)", title: "Language Models are Few-Shot Learners", locator: "p. 43", note: "clipping the global norm at 1.0 when training GPT-3"}
  - {paper: "Chowdhery et al. (2022)", title: "PaLM: Scaling Language Modeling with Pathways", locator: "arXiv 2204.02311, pp. 10 and 11", note: "global norm clipping at 1.0 and the loss spikes of the 540B model that occurred despite it"}
---

Gradient clipping caps the size of a gradient before the optimizer uses it.
If the norm of the gradient is above a threshold $c$, the gradient is scaled down to norm $c$; otherwise it is left alone.
Razvan Pascanu, Tomas Mikolov and Yoshua Bengio proposed clipping the norm of the gradient against exploding gradients in recurrent networks, together with a soft constraint against vanishing gradients. {#c-pascanu-abstract}
Clipping is standard practice in tasks such as language modelling, yet Jingzhao Zhang and colleagues note that it long lacked a firm theoretical footing, with earlier work treating it as an intuitive workaround. {#c-standard-practice}
Their analysis proves that, when the curvature of the loss grows with the size of its gradient, clipped gradient descent can be arbitrarily faster than gradient descent with a fixed step. {#c-arbitrarily-faster}

## In plain words

Training takes many small steps downhill on a loss surface.
The gradient says which way is downhill and how steep it is, and the step is proportional to it.
Sometimes the surface has a cliff: the gradient there is huge, so a normal step throws the parameters far away and undoes much of the training so far.
Clipping puts a ceiling on the step.
The direction stays the same; only the length is cut back when it is too long.
Most steps are below the ceiling and are not touched at all.

## Setting

There are parameters $\theta \in \mathbb{R}^d$, a loss $E(\theta)$ or $f(\theta)$, and an iterative optimizer such as [[m2:stochastic-gradient-descent|stochastic gradient descent]] that updates $\theta$ from a gradient estimate $g = \nabla_\theta E$ on a mini-batch.
The problem clipping addresses is the exploding gradient: a large increase in the norm of the gradient during training, caused by long-term components that can grow exponentially faster than short-term ones. {#c-exploding-def}
Its counterpart, the vanishing gradient, is when the long-term components shrink exponentially fast to norm 0, so that the model cannot learn correlations between events far apart in time. {#c-vanishing-def}
Both arise in recurrent networks trained with [[al:backpropagation-through-time|backpropagation through time]], where the gradient contains products of many Jacobian matrices, one per time step.

Clipping is not a regularizer and not an optimizer of its own.
It is a transformation of the gradient, $g \mapsto \operatorname{clip}(g)$, placed between the backward pass and the optimizer's update.

## The method

**Norm clipping.**
Pascanu et al. compute the gradient $\hat g = \partial E / \partial \theta$ and, whenever $\lVert \hat g \rVert \ge c$, replace it with $c\,\hat g / \lVert \hat g \rVert$. {#c-algorithm}
In one line,

$$ \operatorname{clip}_c(g) = \min\!\left(1, \frac{c}{\lVert g \rVert}\right) g , $$ {#eq-clip}

and the update is $\theta \leftarrow \theta - \eta\, \operatorname{clip}_c(g)$ for learning rate $\eta$.
Andrew Brock and colleagues write the same rule with threshold $\lambda$ for the vector $G = \partial L / \partial \theta$ of all model parameters: $G \to \lambda G / \lVert G \rVert$ if $\lVert G \rVert > \lambda$, and $G$ otherwise. {#c-brock-standard}

**Value clipping.**
The older rule, from Mikolov's PhD thesis, clips the temporal components of the gradient entry by entry: an entry is clipped when its absolute value exceeds a fixed threshold. {#c-mikolov}
Pascanu et al. moved to rescaling the norm so that the step is always a descent direction for the current mini-batch, and they report that in practice the two rules behave alike. {#c-descent-direction}

**Global norm.**
In code, $\lVert g \rVert$ is usually the norm of all gradients concatenated, $\lVert g \rVert = \sqrt{\sum_\ell \lVert g_\ell \rVert^2}$ over the parameter tensors $\ell$.
GPT-3 was trained with Adam, $\beta_1 = 0.9$, $\beta_2 = 0.95$ and $\epsilon = 10^{-8}$, with the global norm of the gradient clipped at 1.0. {#c-gpt3}

**Adaptive gradient clipping (AGC).**
Brock et al. clip relative to the size of the weights.
Their motivation is that the ratio $\lVert G^\ell \rVert_F / \lVert W^\ell \rVert_F$ of the gradient norm to the weight norm of layer $\ell$ measures how much one gradient step will change those weights. {#c-agc-ratio}
They apply the rule to each unit $i$, the $i$-th row of the layer's gradient:

$$ G_i^\ell \to \begin{cases} \lambda \dfrac{\lVert W_i^\ell \rVert_F^\star}{\lVert G_i^\ell \rVert_F} G_i^\ell & \text{if } \dfrac{\lVert G_i^\ell \rVert_F}{\lVert W_i^\ell \rVert_F^\star} > \lambda , \\[1ex] G_i^\ell & \text{otherwise,} \end{cases} \qquad \lVert W_i \rVert_F^\star = \max(\lVert W_i \rVert_F, \epsilon), $$ {#eq-agc}

with default $\epsilon = 10^{-3}$, which keeps the gradients of zero-initialized parameters from always being clipped to zero. {#c-agc}

## Derivation

**Why gradients explode.**
Write the recurrent state as $x_t = W_{\text{rec}}\,\sigma(x_{t-1}) + W_{\text{in}} u_t + b$.
The gradient of the cost at step $t$ with respect to the parameters sums temporal contributions, each containing the product of Jacobians $\partial x_t / \partial x_k = \prod_{t \ge i > k} W_{\text{rec}}^\top \operatorname{diag}(\sigma'(x_{i-1}))$.
A product of $t - k$ matrices can shrink to zero or blow up, just as a product of $t - k$ numbers can.
For the linear model, it is sufficient for the largest eigenvalue of $W_{\text{rec}}$ to be below 1 for the long-term components to vanish, and necessary for it to be above 1 for the gradients to explode. {#c-linear-condition}
With a nonlinearity whose derivative is bounded by $\gamma$, the necessary condition for explosion becomes $\lambda_1 > 1/\gamma$, where $\gamma = 1$ for tanh and $\gamma = 1/4$ for the sigmoid. {#c-gamma}

**Why the curvature explodes with the gradient.**
In the one-unit linear model $x_t = x_0 w^t$, the first derivative is $\partial x_t / \partial w = t x_0 w^{t-1}$ and the second is $\partial^2 x_t / \partial w^2 = t(t-1) x_0 w^{t-2}$, so when the first derivative explodes, so does the second. {#c-linear-derivatives}
With $|w| > 1$, $|t x_0 w^{t-1}|$ grows geometrically in $t$, and the ratio of the second derivative to the first is $(t - 1)/w$, so the curvature grows at least as fast as the slope. {#c-m-ratio}
Pascanu et al. generalize this to a hypothesis: when the gradient explodes along a direction $v$, the curvature along $v$ explodes too, which makes a wall in the error surface. {#c-wall}

**Why a bounded step helps.**
At such a wall, a plain gradient step is huge and points across the valley, so SGD jumps over it and may leave the valley, which disrupts learning. {#c-jump}
A step of bounded length in the same direction only pushes the parameters back into the smoother region beside the wall, where SGD can continue along other descent directions. {#c-bounded-step}
Because the valley is assumed to be wide, a first-order method is enough once it is back there, so clipping alone may suffice and no second-order method is needed. {#c-wide-valley}

**What clipping does to one step.**
If $\lVert g \rVert > c > 0$, the clipped vector $(c / \lVert g \rVert)\, g$ has norm exactly $c$ and inner product $c \lVert g \rVert > 0$ with $g$, so it points the same way as $g$ and is a descent direction whenever $g$ is. {#c-m-direction}
By eq-clip, the update $\eta\,\operatorname{clip}_c(g)$ never has length above $\eta c$, and clipped SGD with learning rate $\eta$ is plain SGD with the step size $\eta \min(1, c / \lVert g \rVert)$, which falls as $1/\lVert g \rVert$ once the norm passes $c$. {#c-m-effective-lr}
This is why Pascanu et al. describe clipping as adapting the learning rate to the norm of the gradient. {#c-adaptive-lr}
Unlike adaptation schemes that collect statistics of the gradient over time, it uses the current gradient alone, so it can handle very abrupt changes in the norm. {#c-instantaneous}
Value clipping keeps each entry in $[-c, c]$, so the clipped vector has norm at most $c\sqrt{d}$ in $d$ dimensions, and it changes the direction of $g$ whenever some entries are cut and others are not. {#c-m-elementwise}

**Relaxed smoothness.**
The textbook analysis of gradient descent assumes $f$ is $L$-smooth, $\lVert \nabla f(x) - \nabla f(y) \rVert \le L \lVert x - y \rVert$ for all $x, y$, which for twice differentiable $f$ means $\lVert \nabla^2 f(x) \rVert \le L$. {#c-l-smooth}
A single global $L$ is restrictive: even $f(x) = x^3$ breaks the assumption. {#c-x3}
Zhang et al. measured the local smoothness along training trajectories and found that it varies greatly and correlates positively with the gradient norm, especially in language modelling. {#c-smoothness-varies}
They call $f$ $(L_0, L_1)$-smooth if the norm of its Hessian is bounded by $L_0$ plus $L_1$ times the norm of its gradient, which strictly relaxes $L$-smoothness: {#c-l0l1}

$$ \lVert \nabla^2 f(x) \rVert \le L_0 + L_1 \lVert \nabla f(x) \rVert . $$ {#eq-l0l1}

Every univariate polynomial of degree $d \ge 3$ is $(L_0, L_1)$-smooth for some $L_0, L_1$ but not $L$-smooth. {#c-polynomial}

Under eq-l0l1, the curvature near $x$ is bounded by $L_0 + L_1 \lVert \nabla f(x) \rVert$, so the classic safe step $1/L$ suggests the step $1/(L_0 + L_1 \lVert g \rVert)$.
For $L_0 > 0$, $L_1 \ge 0$ and $a \ge 0$, $1/(L_0 + L_1 a)$ lies between $\tfrac12 \min\{1/L_0,\ 1/(L_1 a)\}$ and $\min\{1/L_0,\ 1/(L_1 a)\}$, because $\max(u, v) \le u + v \le 2 \max(u, v)$ for $u, v \ge 0$. {#c-m-sandwich}
The right-hand form is the step size of clipped gradient descent, so up to a factor of 2, clipping takes the step that the local curvature bound allows.
Zhang et al. write clipped gradient descent as $x_{k+1} = x_k - h_c \nabla f(x_k)$ with $h_c = \min\{\eta_c,\ \gamma \eta_c / \lVert \nabla f(x) \rVert\}$. {#c-clipped-gd}
Normalized gradient descent, with step $h_n = \eta_n / (\lVert \nabla f(x) \rVert + \beta)$, is the same method up to a constant factor: with matched constants, $\tfrac12 h_c \le h_n \le 2 h_c$. {#c-ngd-equivalent}

**The rates.**
With $\eta_c = 1/(10 L_0)$ and $\gamma = \min\{1/\eta_c,\ 1/(10 L_1 \eta_c)\}$, clipped gradient descent finds an $\epsilon$-stationary point within $20 L_0 (f(x_0) - f^*)/\epsilon^2 + 20 \max\{1, L_1^2\}(f(x_0) - f^*)/L_0$ iterations. {#c-thm3}
Let $M$ bound the gradient norm on the sublevel set $\{x : f(x) \le f(x_0)\}$; over objectives with $L_0, L_1 \ge 1$ and $M > 1$, gradient descent with any fixed step needs at least $L_1 M (f(x_0) - f^* - 5/8) / (8 \epsilon^2 (\log M + 1))$ iterations. {#c-thm4}
A fixed step size of $1/(2(M L_1 + L_0))$ reaches the matching upper bound $4 (M L_1 + L_0)(f(x_0) - f^*)/\epsilon^2$. {#c-thm6}
So when $L_1 M$ is large, which Zhang et al. read as a poor initialization, clipped gradient descent can be arbitrarily faster. {#c-remark5}
The fixed step has to be small enough for the steepest point the run might visit, and it pays for that everywhere; the clipped step is small only where the gradient is large.

## Worked example

Take the gradient $g = (3, 4)$, so $\lVert g \rVert = 5$, and the threshold $c = 1$.
Norm clipping gives $(3, 4)/5 = (0.6, 0.8)$, of norm 1 and pointing exactly the way $g$ points.
Value clipping at 1 gives $(1, 1)$, of norm $\sqrt 2 \approx 1.414$, and the angle between $(1, 1)$ and $(3, 4)$ is $\arccos\big(7 / (5\sqrt 2)\big) \approx 8.13°$. {#c-we-value-clip}

Now let a gradient spike to norm 1000 at learning rate $\eta = 0.1$.
The plain SGD step has length $0.1 \times 1000 = 100$; the step clipped at $c = 1$ has length $0.1 \times 1 = 0.1$, a thousand times shorter, and its direction is unchanged. {#c-we-spike}
A gradient of norm 0.5 is below the threshold and passes through untouched.

Global and per-layer norms differ.
Let a network have two layers with gradient norms 3 and 4, so the global norm is $\sqrt{3^2 + 4^2} = 5$.
Clipping the global norm at 1 multiplies both layers by $1/5$, to norms 0.6 and 0.8; clipping each layer at 1 gives norms 1 and 1, so the first layer's share of the step grows from $3/5$ to $1/\sqrt 2 \approx 0.707$ and the direction changes. {#c-we-per-layer}

For the exploding derivative in the linear model, take $x_0 = 1$, $w = 1.1$ and $t = 50$: $\partial x_{50} / \partial w = 50 \times 1.1^{49} \approx 50 \times 106.7 \approx 5336$, while at $t = 10$ it is $10 \times 1.1^9 \approx 23.6$. {#c-we-linear}

Zhang et al. run gradient descent and clipped gradient descent on $f(x) = x^4$ from $x_0 = 30$.
On the sublevel set $[-30, 30]$, $f''(x) = 12x^2 \le 12 \times 30^2 = 1.08 \times 10^4$, and also $f''(x) \le 10 |f'(x)| + 0.1$, so the function is $(0.1, 10)$-smooth there, while the best global constant for gradient descent is $L = 1.08 \times 10^4$. {#c-x4-curvature}
The best gradient descent run ends with $f'(x_T) = 0.36$, and clipped gradient descent with $1.3 \times 10^{-8}$. {#c-x4-result}
For comparison, $f'(30) = 4 \times 30^3 = 108{,}000$ at the start, so a fixed step that does not diverge there, at most $2 / f''(30) \approx 1.85 \times 10^{-4}$, makes progress of only $4 h x^3 \approx 7.4 \times 10^{-4}$ per step at $x = 1$. {#c-we-x4}

## Hyperparameters

The threshold $c$ is the only new hyperparameter, and Pascanu et al. note that it must be chosen. {#c-new-hyperparameter}
They suggest setting it from statistics of the average gradient norm over a sufficiently large number of updates. {#c-threshold-heuristic}
For a given task and model size, they found training not very sensitive to the threshold, and the method worked even with rather small thresholds. {#c-insensitive}
Brock et al. found the opposite for plain norm clipping of normalizer-free ResNets at large batch sizes: stability was extremely sensitive to the threshold, which needed fine tuning whenever the depth, the batch size or the learning rate changed. {#c-brock-sensitive}

The threshold's scale depends on how the loss is reduced.
Pascanu et al. clipped the norm at 6 on the synthetic addition and temporal order problems, {#c-threshold-6}
at 8 for polyphonic music, where the gradient is a mean over the sequence, and at 45 for character-level language modelling, where the cost is summed over the sequence. {#c-threshold-8-45}
The AWD-LSTM language model recipe uses clipped SGD with clip value 0.25 and learning rate 30. {#c-awd}
For AGC, Brock et al. searched $\lambda \in \{0.01, 0.02, 0.04, 0.08, 0.16\}$ {#c-agc-range}
and found that $\lambda$ should be smaller for larger batches. {#c-agc-lambda}

## Cost

Clipping costs one pass over the gradient to compute its norm, a scalar comparison, and, when the norm is above $c$, one multiplication per parameter.
It adds no parameters and no memory beyond the norm.
In data-parallel or sharded training, the global norm is a sum over all parameter tensors, so it needs every gradient before any of them can be scaled.
AGC needs the norms of the weights as well as of the gradients, one per unit.
At inference there is nothing to do.

## Evidence

**Recurrent networks (Pascanu et al.).**
On the temporal order problem, for sequences longer than 20 the vanishing gradient stopped both SGD and SGD with clipping (SGD-C) from solving the task. {#c-sgd-c-fails}
With clipping plus their regularization term (SGD-CR), the task was solved 100% of the time for sequences up to 200 steps. {#c-temporal-order}
On polyphonic music, clipping lowered the negative log likelihood per time step on Nottingham from 3.80 to 3.48 on the test set, and on MuseData from 8.25 to 6.54 on the training set. {#c-music}
On Penn Treebank characters, clipping lowered the test entropy from 1.50 to 1.42 bits per character; for predicting the fifth character ahead, SGD without clipping gave no result, while SGD-C reached 3.89. {#c-ptb}
Because both training and test error improve, they conclude that clipping fixes an optimization problem and does not act as a regularizer. {#c-not-regularizer}

**Language models and image classifiers (Zhang et al.).**
The correlation between smoothness and gradient norm appeared in the default training for language modelling but not in the default training for image classification, which matches the wide use of clipping in language modelling and its rarity in ResNet training. {#c-lm-vs-cv}
In training curves for an LSTM language model and for ResNet20 on CIFAR-10, Zhang et al. observe that clipping accelerates convergence. {#c-accelerates}
For image classification, clipped gradient descent also converged fastest and matched the test performance of SGD with momentum. {#c-cv-matches}
Clipped training of ResNet20 on CIFAR-10 reached 95.2% test accuracy in 200 epochs, {#c-resnet-952}
against 95.0% for their tuned baseline of SGD with momentum. {#c-baseline-950}
The AWD-LSTM model, trained with clipping for 500 epochs, reached a test perplexity of 56.5 on Penn Treebank. {#c-awd-ppl}

**Large-batch training without normalization (Brock et al.).**
AGC let normalizer-free ResNets scale to batch sizes up to 4096 while matching or beating batch-normalized networks, with smaller gains at small batch sizes. {#c-agc-scale}
It also let them train with strong augmentations such as RandAugment, under which they fail to train without AGC. {#c-agc-randaugment}

## Limitations and failure modes

**It does nothing for vanishing gradients.**
Clipping only shortens long steps.
Pascanu et al. needed a separate regularization term for the vanishing problem, and the gated units of the LSTM, which address vanishing gradients, do not explicitly address exploding ones. {#c-lstm-no-explode}

**It does not stop loss spikes at scale.**
PaLM clipped the global gradient norm at 1.0 for every model size. {#c-palm-clip}
The 540B model still saw its loss spike about 20 times, at irregular points and sometimes late in training, while the smaller models did not. {#c-palm-spikes}
The fix the authors used was not a tighter clip: they restarted from a checkpoint about 100 steps before each spike and skipped 200 to 500 batches. {#c-palm-restart}
They did not blame bad data: the same batches, replayed from an earlier checkpoint, caused no spike, so the spikes came from particular batches meeting a particular state of the model. {#c-palm-not-data}
Clipping bounds each step, but it has no view of that interaction.

**It needs a threshold.**
The threshold is an extra hyperparameter, and for plain norm clipping its best value can move with depth, batch size and learning rate, as Brock et al. found.

**It biases stochastic gradients.**
Clipping is not linear, so the mean of clipped mini-batch gradients is not the clipped mean gradient.
In one dimension, if $g = 0$ or $g = 4$ with probability $\tfrac12$ each and $c = 1$, then $\mathbb{E}[\operatorname{clip}_1(g)] = 0.5$ while $\operatorname{clip}_1(\mathbb{E}[g]) = 1$. {#c-m-bias}
The stochastic analysis of Zhang et al. assumes the noise of the stochastic gradient is bounded almost surely. {#c-bounded-noise}

**The theory is loose.**
Zhang et al. call their own bounds correct but conservative on the $x^4$ example. {#c-conservative}
On a smaller LSTM language model, the theory predicted a ratio of about 5 between the gradient norms of gradient descent and clipped gradient descent, and they measured about 3. {#c-ratio-5-3}
They leave open whether a better condition than $(L_0, L_1)$-smoothness would match the experiments and still allow a clean analysis. {#c-open}

**The alternatives restrict the model.**
An L1 or L2 penalty on the recurrent weights also prevents explosion, but it confines the model to a regime with a single point attractor at the origin, where any information put into the network dies out exponentially fast. {#c-l2-penalty}

## Variants

- *Norm clipping.* Rescale the whole gradient to norm $c$ when it is longer (eq-clip). It keeps the direction.
- *Value clipping.* Clip each entry to $[-c, c]$, as in Mikolov's thesis. It is cheaper to reason about per entry but can change the direction.
- *Global-norm clipping.* Norm clipping with the norm taken over all parameters at once, as in GPT-3.
- *Per-layer clipping.* Clip each parameter tensor's norm separately; the relative size of the layers' steps changes.
- *Adaptive gradient clipping (AGC).* Clip each unit's gradient relative to the norm of its weights (eq-agc). Brock et al. interpret it as a relaxation of normalized optimizers that caps the update size relative to the parameter norm, without a lower bound on the update and without ignoring the gradient's size. {#c-agc-lars}
- *Normalized gradient descent.* Divide the gradient by its norm plus a constant; up to a factor of 2 in the step size, it is clipped gradient descent.

## When to use it

Use clipping when the gradient norm can spike: recurrent networks over long sequences, language models, and training with large learning rates or large batches.
Gradient clipping is often used in language modelling to stabilize training, and Brock et al. cite evidence that it allows larger learning rates. {#c-brock-larger-lr}
Zhang et al. find that clipping lets the training trajectory pass stably through non-smooth regions, while without clipping the optimizer must keep a small learning rate. {#c-traverse}
Use AGC when training deep networks without normalization layers at large batch sizes.
Clipping is cheap, so it is often left on as a guard even where it rarely triggers.

## Pitfalls

- *Value clipping when norm clipping was meant.* Clipping entries changes the update direction (see the worked example); `clip_grad_value` and `clip_grad_norm` in PyTorch are different rules.
- *A threshold copied across loss reductions.* A loss summed over a sequence has gradients that grow with its length, while a mean does not; Pascanu et al. used 8 for a mean and 45 for a sum.
- *Clipping per layer when the global norm was meant.* It changes the relative steps of the layers.
- *Clipping before the gradients are complete.* With gradient accumulation or data parallelism, the norm should be computed on the full gradient of the step, after accumulation and averaging.
- *AGC on the classifier.* Brock et al. found it always better not to clip the final linear layer. {#c-agc-final-layer}
- *Reading clipping as regularization.* It changes optimization, not the model; Pascanu et al. saw training and test error fall together.

## History

Clipped gradient descent was introduced in training recurrent neural network language models (Mikolov et al., 2010) to relieve the exploding gradient problem, and later carried over to LSTMs and sequence-to-sequence models. {#c-history-mikolov}
Mikolov's 2012 thesis clipped the entries of the gradient, and the clipping in his Penn Treebank language models reached results that Pascanu et al. matched with norm clipping. {#c-ptb-sota}
In 2012, Pascanu, Mikolov and Bengio analysed exploding and vanishing gradients from analytical, geometric and dynamical-systems views and proposed rescaling by the norm, motivated by the assumption that the curvature explodes with the gradient, giving a valley with a single steep wall. {#c-conclusion}
In 2020, GPT-3 was trained with the global norm clipped at 1.0, and Zhang et al. gave a convergence theory for clipping under $(L_0, L_1)$-smoothness.
In 2021, Brock et al. introduced adaptive gradient clipping and used it to train normalizer-free networks that match the accuracy of an EfficientNet-B7 on ImageNet while being up to 8.7 times faster to train. {#c-nfnet}
