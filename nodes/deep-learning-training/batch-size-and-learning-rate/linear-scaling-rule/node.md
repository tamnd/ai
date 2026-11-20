---
tag: JDMWH4
kind: method
slug: linear-scaling-rule
name: Linear scaling rule
aliases: [linear learning rate scaling, learning-rate scaling rule, large-batch scaling]
area: deep-learning-training/batch-size-and-learning-rate
level: core
summary: >-
  When the minibatch grows k times, multiply the SGD learning rate by k; this holds while the batch is small compared with the gradient noise scale and stops paying off beyond the critical batch size.
purpose: moving a learning rate tuned at one batch size to a larger batch size, and knowing how large the batch can usefully grow
uses: [stochastic-gradient-descent@m2]
related_to: [batch-normalization@al, gradient-accumulation@al]
hyperparameters:
  - {name: base learning rate and base batch size, meaning: "a learning rate tuned at a reference batch size, from which the rule extrapolates", typical: "0.1 per 256 images for ResNet-50 with SGD and momentum"}
  - {name: target batch size, meaning: "the batch size after scaling, usually set by the number of workers times the per-worker batch", typical: "below the critical batch size; 8k was the edge for ResNet-50 on ImageNet"}
  - {name: scaling rule, meaning: "how the rate moves with the batch: linear, square root, or a curve that saturates at the noise scale", typical: "linear for SGD and momentum; between square root and linear for Adam"}
  - {name: warmup, meaning: "a ramp from the base rate to the scaled rate at the start of training, which the rule needs at large k", typical: "gradual warmup over 5 epochs; see learning-rate-warmup"}
  - {name: noise-scale estimator, meaning: "the two batch sizes and the moving-average decay used to measure B_simple during training", typical: "one worker's batch and the global batch; decay tuned until the estimate is stable"}
cost:
  train: "none for the rule itself, which changes one scalar; measuring the noise scale with the two-batch estimator costs two squared norms per step in data-parallel training, and from per-example gradients one extra vectorized backward pass"
  inference: "none: the rule acts only on training"
history:
  year: 2017
  note: "Krizhevsky (2014) already scaled the learning rate linearly with the batch; Goyal et al. (arXiv 1706.02677, June 2017) named the linear scaling rule and used it with gradual warmup to train ResNet-50 on ImageNet in one hour. Hoffer et al. (2017) argued for square-root scaling, Smith et al. (ICLR 2018) traded learning-rate decay for batch growth, and McCandlish et al. (arXiv 1812.06162, December 2018) explained when linear scaling stops through the gradient noise scale and the critical batch size."
prereqs: [stochastic-gradient-descent@m2, learning-rate-schedule@al, variance@m2]
same_as:
  wikidata: null
  wikipedia: null
  arxiv: ["1706.02677", "1812.06162", "1711.00489", "1705.08741"]
  al: null
sources:
  - {paper: "Goyal et al. (2017)", title: "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour", locator: "pp. 1 to 4, 6, 9 and 10", note: "the linear scaling rule, its interpretation and limits, the remarks on distributed SGD, and the minibatch-size experiments"}
  - {paper: "McCandlish et al. (2018)", title: "An Empirical Model of Large-Batch Training", locator: "pp. 1, 3, 5 to 11, 13, 17 and 25", note: "the gradient noise scale, the optimal step, the steps-examples trade-off, the critical batch size, the two-batch estimator and the experiments"}
  - {paper: "Smith et al. (2018)", title: "Don't Decay the Learning Rate, Increase the Batch Size", locator: "pp. 1 to 3", note: "the SGD noise scale and batch growth in place of learning-rate decay"}
  - {paper: "Hoffer et al. (2017)", title: "Train longer, generalize better: closing the generalization gap in large batch training of neural networks", locator: "pp. 1 and 5", note: "the square-root scaling rule and the number of updates as the cause of the generalization gap"}
  - {paper: "Shallue et al. (2019)", title: "Measuring the Effects of Data Parallelism on Neural Network Training", locator: "JMLR 20, pp. 4, 19 and 42", note: "learning rate heuristics measured against retuned learning rates across many workloads"}
---

The linear scaling rule says how to change the learning rate of stochastic gradient descent when the batch size changes: when the minibatch size is multiplied by $k$, multiply the learning rate by $k$. {#c-goyal-rule}
Priya Goyal and colleagues stated it in this form and kept all other hyperparameters, such as weight decay, unchanged. {#c-goyal-unchanged}
The rule holds only up to a point, and Sam McCandlish and colleagues located that point: a statistic that is cheap to measure, the gradient noise scale, forecasts in many domains the biggest batch that still helps. {#c-mccandlish-abstract}
This page covers the rule, the noise scale that bounds it, and the critical batch size that follows; the warmup that the rule needs at the start of training has its own page, [[learning-rate-warmup]].

## In plain words

A minibatch gradient is an average of per-example gradients, and an average of more examples is a less noisy estimate of the true gradient.
Doubling the batch halves the variance of the estimate, so each step can safely move further, and the rule lets it move twice as far.
If the gradient hardly changes over a few steps, one double-length step on twice the data lands where two ordinary steps would, and training needs half as many steps.

That works only while noise is what limits the step.
Once the batch gradient is almost exact, a bigger batch adds little, and doubling the step overshoots.
The batch size where this happens is the noise scale: how noisy the per-example gradients are compared with how large the true gradient is.
Below it, more batch buys almost proportionally fewer steps; above it, mostly more compute.

## Setting

Minibatch SGD with a minibatch $\mathcal B$ of size $n$ and learning rate $\eta$ updates $w_{t+1} = w_t - \eta \frac1n \sum_{x \in \mathcal B} \nabla l(x, w_t)$. {#c-sgd-minibatch}
Write $G$ for the true gradient and $G_{\mathrm{est}}$ for the mean of the per-example gradients of a batch of $B$ examples drawn from the data distribution; then $\mathbb{E}[G_{\mathrm{est}}] = G$ and $\operatorname{cov}(G_{\mathrm{est}}) = \Sigma / B$, where $\Sigma$ is the covariance of the per-example gradients. {#c-gest}
If the batches are drawn without replacement from $D$ examples, the factor $1/B$ becomes $1/B - 1/D$. {#c-without-replacement}
The question is how $\eta$ should depend on $B$, and how large $B$ can grow before it stops shortening training.

## The method

### The rule

Given a learning rate $\eta_0$ tuned at batch size $B_0$, use

$$ \eta(B) = \eta_0 \, \frac{B}{B_0} $$ {#eq-linear-rule}

at batch size $B$, with the same number of epochs, the same schedule shape expressed in epochs, and the same weight decay.
Goyal et al. justify it by comparing $k$ SGD steps at rate $\eta$ on $k$ minibatches with one step at rate $\hat\eta$ on their union: if $\nabla l(x, w_t) \approx \nabla l(x, w_{t+j})$ for $j < k$, then $\hat\eta = k\eta$ makes the large step approximately equal to the $k$ small ones. {#c-goyal-interpretation}
They name two cases where that assumption fails: early in training, when the network changes rapidly, and at very large minibatches, where accuracy degrades rapidly beyond a point that is about 8k for their ImageNet experiments. {#c-goyal-fails}
The first is what [[learning-rate-warmup]] repairs; the second is what the noise scale explains.

### Making it work in distributed training

In data-parallel training with $k$ workers of $n$ examples each, Goyal et al. keep the per-worker batch $n$ fixed at 32 and change only $k$; $n$ is also the [[al:batch-normalization|batch normalization]] batch. {#c-goyal-n-fixed}
Three of their remarks are easy to get wrong.
Scaling the cross-entropy loss is not equivalent to scaling the learning rate, because weight decay is not scaled with it. {#c-remark1}
Momentum needs a correction after a change of learning rate when the learning rate is folded into the momentum buffer. {#c-remark2}
The per-worker loss should be normalized by the total minibatch $kn$, not by the per-worker $n$. {#c-remark3}

### The gradient noise scale

McCandlish et al. ask what step size is best for one step along $G_{\mathrm{est}}$ on a loss with Hessian $H$, to second order.
They obtain the optimal step and improvement

$$ \epsilon_{\mathrm{opt}}(B) = \frac{\epsilon_{\max}}{1 + B_{\mathrm{noise}}/B}, \qquad \Delta L_{\mathrm{opt}}(B) = \frac{\Delta L_{\max}}{1 + B_{\mathrm{noise}}/B}, $$ {#eq-eps-opt}

where $\epsilon_{\max}$ and $\Delta L_{\max}$ are the step and the improvement with the exact gradient. {#c-mc-eps-opt}
The gradient noise scale is

$$ B_{\mathrm{noise}} = \frac{\operatorname{tr}(H\Sigma)}{G^\top H G}, $$ {#eq-b-noise}

and it does not depend on the size of the training set. {#c-noise-scale-def}
With a step larger than twice $\epsilon_{\mathrm{opt}}$ the loss may increase. {#c-mc-twice}
When $B \ll B_{\mathrm{noise}}$, increasing the batch linearly increases the progress per step, the small-batch regime; when $B \gg B_{\mathrm{noise}}$ it barely helps. {#c-small-batch-regime}
The linear scaling rule is eq-eps-opt in its small-batch regime, where $\epsilon_{\mathrm{opt}} \approx \epsilon_{\max} B / B_{\mathrm{noise}}$.

$H$ is rarely available. If it is a multiple of the identity, the noise scale reduces to the simple noise scale

$$ B_{\mathrm{simple}} = \frac{\operatorname{tr}(\Sigma)}{\lvert G \rvert^2}, $$ {#eq-b-simple}

the sum of the variances of the gradient components divided by the squared norm of the gradient. {#c-b-simple-def}
In practice $B_{\mathrm{simple}}$ and $B_{\mathrm{noise}}$ typically differ only by a small constant factor, particularly with training schemes that improve conditioning. {#c-simple-vs-noise}

### The critical batch size

Summing the per-step improvements over a run gives a relation between the steps $S$ and the examples $E$ needed to reach a target loss,

$$ \Big(\frac{S}{S_{\min}} - 1\Big)\Big(\frac{E}{E_{\min}} - 1\Big) = 1, \qquad B_{\mathrm{crit}} = \frac{E_{\min}}{S_{\min}}, $$ {#eq-tradeoff}

where $S_{\min}$ is the fewest steps (infinite batch) and $E_{\min}$ the fewest examples (tiny batch), and McCandlish et al. define $B_{\mathrm{crit}}$ by an empirical fit to this relation. {#c-mc-tradeoff}
Their model predicts $B_{\mathrm{crit}} \approx B_{\mathrm{noise}}$ averaged over training, and at $B = B_{\mathrm{crit}}$ a run takes twice as many passes through the data as the most data-efficient run and twice as many steps as the fastest. {#c-mc-bcrit-noise}
So training should parallelize almost linearly up to a batch equal to the noise scale, then switch smoothly but quickly to a regime where more parallelism gives little. {#c-mc-parallelize}

### Measuring the noise scale

From the squared norms of gradients at two batch sizes $B_{\mathrm{small}} < B_{\mathrm{big}}$,

$$ \lvert\mathcal G\rvert^2 = \frac{B_{\mathrm{big}} \lvert G_{B_{\mathrm{big}}}\rvert^2 - B_{\mathrm{small}} \lvert G_{B_{\mathrm{small}}}\rvert^2}{B_{\mathrm{big}} - B_{\mathrm{small}}}, \qquad \mathcal S = \frac{\lvert G_{B_{\mathrm{small}}}\rvert^2 - \lvert G_{B_{\mathrm{big}}}\rvert^2}{1/B_{\mathrm{small}} - 1/B_{\mathrm{big}}} $$ {#eq-two-batch}

are unbiased estimates of $\lvert G\rvert^2$ and $\operatorname{tr}(\Sigma)$. {#c-two-batch}
In data-parallel training one worker's gradient and the all-reduced gradient give both norms for free.
Their ratio is not an unbiased estimate of the noise scale, so McCandlish et al. keep separate exponentially weighted moving averages of the two and take the ratio of the averages. {#c-ratio-biased}

## Derivation

Three derivations support the formulas; each lists its assumptions and has a numeric check.

**k small steps against one scaled step** (`k-small-steps`).
For least squares with plain SGD, $k$ steps at rate $\eta$ on disjoint batches end at $w_t - \eta \sum_j g_j(w_{t+j})$, and one step at rate $k\eta$ on their union differs from them by exactly $\eta^2 \sum_{j \ge 1} H_j \sum_{i<j} g_i(w_{t+i})$. {#c-k-step-identity}
The norm of that gap is at most $\frac{k-1}{2}\, \eta\lambda_{\max} \cdot k\eta \max_i \lVert g_i \rVert$, so relative to the step it is first order in $\eta$ and vanishes as $\eta \to 0$. {#c-k-step-second-order}
Any other rate, such as $\sqrt k\, \eta$ or $\eta$, leaves a gap proportional to $(k\eta - \hat\eta)$ times the mean gradient, so only the linear rule matches the small steps as $\eta \to 0$. {#c-k-step-only-k}
The scaled step is stable only if $k\eta\lambda_{\max} < 2$, while the small steps need only $\eta\lambda_{\max} < 2$; for $2/k < \eta\lambda_{\max} < 2$ the small-batch run converges and the scaled run diverges. {#c-k-step-fails}

**The optimal step and the noise scale** (`optimal-step-noise-scale`).
On a quadratic, the expected loss after the step $\theta - \epsilon G_{\mathrm{est}}$ is $L(\theta) - \epsilon\lvert G\rvert^2 + \frac12 \epsilon^2 \big(G^\top H G + \operatorname{tr}(H\Sigma)/B\big)$. {#c-expected-loss}
It is smallest at $\epsilon_{\mathrm{opt}}(B) = \epsilon_{\max}/(1 + B_{\mathrm{noise}}/B)$ with $\epsilon_{\max} = \lvert G\rvert^2 / G^\top H G$. {#c-eps-opt}
The best improvement is $\Delta L_{\max}/(1 + B_{\mathrm{noise}}/B)$ with $\Delta L_{\max} = \lvert G\rvert^4 / (2 G^\top H G)$. {#c-dl-opt}
For $B \ll B_{\mathrm{noise}}$ the optimal step is about $\epsilon_{\max} B / B_{\mathrm{noise}}$, linear in $B$; for $B \gg B_{\mathrm{noise}}$ it tends to $\epsilon_{\max}$; at $B = B_{\mathrm{noise}}$ it is $\epsilon_{\max}/2$. {#c-eps-regimes}
If $H = hI$, then $B_{\mathrm{noise}} = B_{\mathrm{simple}}$, and the expected squared error of $G_{\mathrm{est}}$ relative to $\lvert G\rvert^2$ is $B_{\mathrm{simple}}/B$. {#c-b-simple-reduction}
The expected loss is back at $L(\theta)$ at $\epsilon = 2\epsilon_{\mathrm{opt}}$ and higher beyond it. {#c-twice-eps}

**The steps-examples trade-off** (`steps-examples-tradeoff`).
If every step at batch $B$ achieves that improvement, the steps to a target loss are $S(B) = S_{\min} + E_{\min}/B$, with $S_{\min} = \int dL/\Delta L_{\max}$ and $E_{\min} = \int B_{\mathrm{noise}}\, dL/\Delta L_{\max}$. {#c-steps-form}
With $E = BS$ this gives eq-tradeoff for every $B$. {#c-tradeoff-product}
$B_{\mathrm{crit}} = E_{\min}/S_{\min}$ is the average of $B_{\mathrm{noise}}$ over the steps of an infinite-batch run. {#c-bcrit-average}
At $B = B_{\mathrm{crit}}$ a run takes $2S_{\min}$ steps and $2E_{\min}$ examples. {#c-bcrit-twice}

## Worked example

Goyal et al.'s ResNet-50 recipe uses $\eta = 0.1$ per 256 images.
At a minibatch of 8,192, $k = 32$, so the linear rule gives $\eta = 3.2$ and the square-root rule $0.1\sqrt{32} = 0.5657$.
With a gradual warmup hook over 5 epochs that starts at the base rate, the rate is $0.1$ at epoch 0, $1.65$ at epoch 2.5 and $3.2$ from epoch 5 until the first decay to $0.32$ at epoch 30.

Next, the noise scale from four per-example gradients in two dimensions, $(3, 1)$, $(1, -1)$, $(2, 2)$ and $(2, -2)$.
Their mean is $(2, 0)$ with squared norm 4.
The squared deviations from the mean are 2, 2, 4 and 4, so the unbiased $\operatorname{tr}(\Sigma)$ is $12/3 = 4$.
Since $\mathbb{E}\lvert\bar g\rvert^2 = \lvert G\rvert^2 + \operatorname{tr}(\Sigma)/n$, the unbiased $\lvert G\rvert^2$ is $4 - 4/4 = 3$, and $B_{\mathrm{simple}} \approx 4/3$.

The two-batch estimator does the same from norms alone.
Say a worker's batch of 32 has $\lvert G_{32}\rvert^2 = 3$ and the global batch of 256 has $\lvert G_{256}\rvert^2 = 1.25$.
Then eq-two-batch gives $\lvert\mathcal G\rvert^2 = (256 \cdot 1.25 - 32 \cdot 3)/224 = 1$ and $\mathcal S = 1.75/(1/32 - 1/256) = 64$, so $B_{\mathrm{simple}} = 64$.

What does $B_{\mathrm{noise}} = 64$ mean for the rule?
With $\epsilon_{\max} = 1$, eq-eps-opt gives optimal steps $0.2$, $0.5$ and $0.8$ at $B = 16$, $64$ and $256$.
From 16 to 64 the batch grows 4 times but the best step only 2.5 times; the linear rule would set $0.8$, 1.6 times too large.
From 64 to 256 it would set $2.0$, more than twice the optimum of $0.8$, and the expected loss would rise.
If $S_{\min} = 1{,}000$ and $B_{\mathrm{crit}} = 64$, a run at $B = 64$ takes 2,000 steps and 128,000 examples, while at $B = 1{,}024$ it takes 1,062.5 steps and 1,088,000 examples: 17 times $E_{\min}$ for 47% fewer steps.
These numbers are tested in both implementations.

## Hyperparameters

The rule needs a base rate tuned at a base batch; Goyal et al. use $\eta = 0.1 \cdot kn/256$, which is 0.1 for $k = 8$ workers with $n = 32$ images. {#c-reference-lr}
They train every model for 90 epochs regardless of minibatch size, with Nesterov momentum 0.9 and weight decay 0.0001. {#c-goyal-setup}
The target batch should sit below the critical batch size, which can be measured during a small-batch run with the two-batch estimator.
The noise scale depends on the learning rate through the temperature $T = \epsilon/B$ of training: it is inflated if the rate is too small, and in equilibrium it varies as $1/T$. {#c-mc-temperature}

## Cost

The rule changes one scalar and costs nothing per step.
Measuring $B_{\mathrm{simple}}$ from the norms of a local and a global gradient costs two reductions per step; per-example gradients cost a vectorized backward pass.
Goyal et al. simulated minibatches of 12k and more on one server with [[al:gradient-accumulation|gradient accumulation]] between updates and found results equivalent to distributed training. {#c-goyal-accumulation}
The real cost is in the trade-off eq-tradeoff: past $B_{\mathrm{crit}}$ each halving of the step count needs much more than double the compute.

## Evidence

At minibatch 8k, $\eta = 0.1 \cdot 32$ gave ResNet-50 23.74% top-1 error against 23.60% at 256; keeping $\eta = 0.1$ gave 41.67% and the square-root rule 26.22%. {#c-goyal-table2a}
Error stayed stable from 64 to 8k, then rose, and beyond 64k training diverged. {#c-goyal-sweep}
ResNet-101 at 8k reached 22.36% against 22.08%, in 92.5 minutes on 256 GPUs. {#c-goyal-resnet101}
On the five-times-larger ImageNet-5k, 8k raised error by 0.26% and the maximum effective minibatch did not change significantly. {#c-goyal-5k}
Across tasks, the simple noise scale predicted the critical batch size to an order of magnitude while it ranged from 20 to over 10 million. {#c-mc-range}
With SGD and momentum the tuned learning rate grew linearly with batch size and then levelled off as eq-eps-opt predicts; with Adam it grew as $B^\alpha$, $0.5 < \alpha < 1$. {#c-mc-optimal-lr}
The critical batch size typically grew tenfold or more during training. {#c-mc-grows}

## Limitations and failure modes

The rule assumes the gradient changes little over $k$ steps, so it fails at the start of training without warmup, and it fails past the critical batch size however long the warmup.
Near the breaking point results are also more sensitive to the learning rate, and Goyal et al. advise staying away from it. {#c-goyal-breaking-point}
The noise scale is a property of the task and the current loss, not of the network size alone: for LSTM language models it was roughly independent of model size at a fixed loss, but larger models reach lower losses and so larger noise scales. {#c-mc-model-size}
A batch size chosen at the start of training is therefore often too small at the end.

A broad measurement by Christopher Shallue and colleagues, over many workloads with the learning rate retuned at each batch size, found that popular learning rate heuristics, such as linearly scaling the learning rate with the batch size, do not hold across all problems or across all batch sizes (p. 4). {#c-shallue-heuristics}
The best learning rates they found did not always follow linear or square root scaling (Figure 21, p. 42). {#c-shallue-fig21}
Momentum extended the region of perfect scaling to larger batches, but at batch sizes inside every optimizer's perfect scaling region, SGD with momentum performed the same as plain SGD (p. 19). {#c-shallue-momentum}

The noise-scale model is about optimizing the training loss and says nothing about generalization; McCandlish et al. cite work that found no generalization gap when hyperparameters are properly tuned. {#c-mc-generalization}
The older view that large batches generalize worse because they find sharp minima is covered on [[sharpness-aware-minimization]].
Elad Hoffer and colleagues argued instead that the gap stems from the small number of updates rather than from the batch size, and can be removed by adapting the training regime. {#c-hoffer-gap}

The linear form is derived for SGD; adaptive optimizers normalize the gradient, and for them the best rate grows more slowly than $B$ (see Evidence).

## Variants

**Square-root scaling.** To keep the covariance of the weight update the same for all minibatch sizes $M$, Hoffer et al. propose $\eta \propto \sqrt M$. {#c-hoffer-sqrt}
With SGD it did worse than the linear rule (see Evidence).

**Saturating scaling.** Eq-eps-opt itself is a rule: linear for $B \ll B_{\mathrm{noise}}$, constant for $B \gg B_{\mathrm{noise}}$.

**Increasing the batch instead of decaying the rate.** Smith et al. use the noise scale $g = \epsilon(N/B - 1) \approx \epsilon N/B$ for training-set size $N$, which implies an optimal batch size proportional to the learning rate. {#c-smith-noise}
Growing the batch where a schedule would decay the rate gave the same learning curves and test accuracy after the same epochs, with fewer updates, for SGD, momentum, Nesterov momentum and Adam. {#c-smith-abstract}
They grow $B$ at constant rate until $B \approx N/10$ and then decay the rate. {#c-smith-n10}
With momentum $m$ the noise scale becomes about $\epsilon N/(B(1-m))$; raising $m$ and scaling $B$ with $1/(1-m)$ slightly reduced test accuracy. {#c-smith-momentum}
Inception-ResNet-V2 reached 77% on ImageNet in under 2,500 updates with batches of 65,536, and ResNet-50 reached 76.1% in under 30 minutes. {#c-smith-results}

## When to use it

Use the linear rule when moving an SGD or momentum recipe to more workers, with the per-worker batch fixed, warmup at the start, and a target batch below the critical batch size.
If the critical batch size is unknown, measure $B_{\mathrm{simple}}$ during a short run at the base batch; batches up to about that size will cut steps almost proportionally.
For Adam start with square-root scaling and tune.
When the batch is already at or above the noise scale, spend extra hardware on a larger model or on more runs instead of a larger batch.

## Pitfalls

- **Scaling the loss instead of the rate.** Multiplying the loss by $k$ also changes the effective weight decay.
- **Normalizing by the local batch.** Each worker's loss must be divided by the global batch $kn$ before gradients are summed, or the effective rate changes with $k$.
- **Changing the batch-norm batch.** Growing the per-worker batch changes the batch-normalization statistics and so the objective; grow the number of workers instead.
- **Forgetting the schedule.** Decay points are in epochs; at a larger batch the same epoch is fewer steps, so step-based schedules must be divided by $k$.
- **Measuring at the wrong temperature.** A learning rate far below its tuned value inflates the measured noise scale.

## History

The rule was adopted by Krizhevsky, if not earlier, who reported a 1% increase of error when going from minibatch 128 to 1024. {#c-krizhevsky}
Bottou et al. found that with the linear scaling rule a solver follows the same training curve as a function of examples seen, and that the rate should not exceed a maximum independent of the minibatch size. {#c-bottou}
In June 2017 Goyal et al. named the rule, paired it with gradual warmup, and trained ResNet-50 with a minibatch of 8,192 on 256 GPUs in one hour, with about 90% scaling efficiency from 8 to 256 GPUs. {#c-goyal-hour}
McCandlish et al. tied these results together in December 2018 with the gradient noise scale and the critical batch size.
