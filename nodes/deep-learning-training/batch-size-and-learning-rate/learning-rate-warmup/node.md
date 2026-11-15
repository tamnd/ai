---
tag: NEW-learning-rate-warmup
kind: method
slug: learning-rate-warmup
name: Learning rate warmup
aliases: [lr warmup, warm-up, warm-up stage, gradual warmup, linear warmup, constant warmup]
area: deep-learning-training/batch-size-and-learning-rate
level: core
summary: >-
  Start training with a small learning rate and raise it, usually linearly, to the target over the first steps, so that early updates stay small while the network changes fastest.
purpose: stable early training at a large learning rate, as with large minibatches in SGD and with Adam on Post-LN Transformers
uses: [learning-rate-schedule@al, stochastic-gradient-descent@m2]
related_to: [radam@al, adamw@al, layer-normalization@al, gradient-clipping@al, batch-normalization@al]
hyperparameters:
  - {name: warmup length, meaning: "the number of steps (or epochs) over which the learning rate rises to its target", typical: "5 epochs for ResNet-50 at minibatch 8k (about 780 iterations); 4000 steps for the Transformer; 10k steps for BERT in Xiong et al."}
  - {name: start learning rate, meaning: "the learning rate at the first step of warmup", typical: "the small-batch rate 0.1 in Goyal et al.; zero, or one step's worth of the ramp, in linear warmup to a peak"}
  - {name: peak learning rate, meaning: "the learning rate reached at the end of warmup, where the main schedule takes over", typical: "0.1 times the minibatch size over 256 for ResNet-50 with SGD; 5e-4 or 1e-3 with Adam on IWSLT14 De-En"}
  - {name: shape, meaning: "how the rate rises during warmup: a constant low rate, a linear ramp, or a ramp built into a schedule such as the inverse square root", typical: "linear"}
cost:
  train: "none per step beyond computing a scalar learning rate; the cost is the steps spent at a reduced rate, and the tuning of the warmup length and the peak"
  inference: "none: warmup acts only on training"
history:
  year: 2017
  note: "A low constant learning rate at the start of training was used by He et al. (2016) for ResNets; in June 2017 Goyal et al. (arXiv 1706.02677) introduced gradual (linear) warmup for large-minibatch SGD, and Vaswani et al. (arXiv 1706.03762) trained the Transformer with a linear warmup followed by inverse-square-root decay. Liu et al. (2019, ICLR 2020) explained warmup for Adam as variance reduction, and Xiong et al. (2020) tied the need for it to the Post-LN placement of layer normalization."
prereqs: [stochastic-gradient-descent@m2, learning-rate-schedule@al, variance@m2]
same_as:
  wikidata: null
  wikipedia: null
  arxiv: ["1706.02677", "1706.03762", "1908.03265", "2002.04745"]
  al: null
sources:
  - {paper: "Goyal et al. (2017)", title: "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour", locator: "pp. 1 to 3, 6, 7 and 9", note: "the linear scaling rule, constant and gradual warmup, Table 1, and the minibatch-size sweep"}
  - {paper: "Vaswani et al. (2017)", title: "Attention Is All You Need", locator: "p. 7", note: "the warmup and inverse-square-root schedule of the Transformer"}
  - {paper: "Liu et al. (2020)", title: "On the Variance of the Adaptive Learning Rate and Beyond", locator: "pp. 1 to 8", note: "warmup as variance reduction for Adam, and RAdam"}
  - {paper: "Xiong et al. (2020)", title: "On Layer Normalization in the Transformer Architecture", locator: "the abstract, Sections 1 to 4.2", note: "warmup in the Post-LN Transformer, the gradient scale at initialization, and Pre-LN without warmup"}
  - {url: "https://docs.pytorch.org/docs/2.15/generated/torch.optim.RAdam.html", title: "torch.optim.RAdam", locator: "the algorithm", note: "the adaptive step is taken when rho_t > 5"}
---

Learning rate warmup starts training with a small learning rate and raises it to the intended one over the first part of training.
Priya Goyal and colleagues introduced their warmup to get past the optimization difficulties of early training by keeping the learning rate low at first. {#c-goyal-def}
In the form used for Transformers, the rate begins very small and climbs to a fixed maximum over a fixed number of iterations. {#c-xiong-def}
For adaptive optimizers such as Adam and RMSprop, Liyuan Liu and colleagues credit warmup with steadier training, faster convergence and better generalization. {#c-liu-abstract}
Ruibin Xiong and colleagues call the warmup stage crucial to the final performance of a Transformer, but note that it slows down optimization and brings more hyperparameters to tune. {#c-xiong-abstract}

## In plain words

At the start of training the network is at a random point, and the first few gradients point in directions that will not stay valid for long.
A large step in such a direction can throw the network into a bad region it never recovers from.
Warmup takes small steps first and lengthens them gradually, so that by the time the steps are large, the network has moved somewhere they are safe.
It is used in two common situations: training with a very large minibatch, where the learning rate has been scaled up to match, and training Transformers with Adam.

## Setting

There are parameters $w$, a loss averaged over minibatches, and an iterative optimizer, [[m2:stochastic-gradient-descent|stochastic gradient descent]] or an adaptive method such as Adam, with a learning rate $\eta_t$ at step $t$ set by a [[al:learning-rate-schedule|learning rate schedule]].
A schedule is usually constant or decreasing.
Liu et al. define warmup by the opposite: it sets the learning rate to smaller values in the first few steps, so the condition $\alpha_{t+1} \le \alpha_t$ for all $t$ fails, and linear warmup in particular sets $\alpha_t = t\,\alpha_0$ while $t < T_w$. {#c-liu-not-monotone}
For convolutional and recurrent networks, the common practice was the opposite pattern, a relatively large learning rate at the beginning that decreases during training, and warmup had been shown essential only for specific problems such as large-batch training. {#c-xiong-cnn-practice}

The large-batch case comes from the linear scaling rule: when the minibatch size is multiplied by $k$, multiply the learning rate by $k$. {#c-linear-scaling}
The scaled learning rate is what the warmup ramps up to.

## The method

**Constant warmup.**
The warmup of He et al. (2016), as Goyal et al. describe it, holds the rate at a low fixed value for the first few epochs, and Goyal et al. found it helpful when fine-tuning pre-trained layers together with new ones for object detection and segmentation. {#c-constant}

**Gradual warmup.**
Goyal et al. raise the rate step by step from a small value to a large one, so that it never jumps. {#c-gradual}
With a minibatch of size $kn$, they start from the small-batch rate $\eta$, increase it by a constant amount at every iteration until it reaches $\hat\eta = k\eta$ after 5 epochs, and then return to the original schedule; they report that the precise warmup length matters little. {#c-gradual-detail}
Written out for a warmup of $T_w$ iterations,

$$ \eta_t = \eta + (\hat\eta - \eta)\,\frac{t}{T_w}, \qquad 0 \le t \le T_w , $$ {#eq-gradual}

after which $\eta_t$ follows the main schedule, which for Goyal et al. is a step decay from $\hat\eta$.

**Linear warmup from zero.**
Xiong et al. write the warmup of the Transformer as $\mathrm{lr}(t) = \frac{t}{T_{\text{warmup}}}\,\mathrm{lr}_{\max}$ for $t \le T_{\text{warmup}}$, after which a classical scheduler takes over, such as linear decay, inverse square-root decay, or forced decay at given iterations. {#c-xiong-eq}
This is eq-gradual with $\eta = 0$ and $\hat\eta = \mathrm{lr}_{\max}$:

$$ \eta_t = \eta_{\max} \min\!\left(1, \frac{t}{T_w}\right) \cdot s(t) , $$ {#eq-linear}

where $s(t)$, equal to 1 during warmup, is the decay that follows.

**Warmup inside the schedule.**
The original Transformer was trained with Adam with $\beta_1 = 0.9$, $\beta_2 = 0.98$ and $\epsilon = 10^{-9}$, and with the learning rate {#c-noam-adam}

$$ \mathrm{lrate} = d_{\text{model}}^{-0.5} \cdot \min\!\left(\mathrm{step}^{-0.5},\ \mathrm{step} \cdot \mathrm{warmup}^{-1.5}\right) , $$ {#eq-noam}

which increases linearly for the first warmup steps and then decreases in proportion to the inverse square root of the step number, with 4000 warmup steps. {#c-noam}
For $s \le w$, $s\,w^{-1.5} \le s^{-0.5}$, so the rate is $d^{-0.5} s\, w^{-1.5}$, a linear ramp; for $s \ge w$ it is $d^{-0.5} s^{-0.5}$; the two branches meet at $s = w$, where the rate peaks at $(d\, w)^{-0.5}$. {#c-m-noam-peak}
So the warmup length also sets the peak: a longer warmup gives a lower peak.

**Warmup built into the optimizer.**
RAdam multiplies Adam's adaptive step by a rectification term $r_t$ that keeps the variance of the adaptive learning rate constant, and it switches the adaptive step off while that variance is divergent, which avoids instability in the first few updates. {#c-radam-summary}
The derivation below shows where the term comes from, and the worked example shows that $r_t$ rises from about 0 to 1 over the first few thousand steps.

## Derivation

**Why large minibatches need it (Goyal et al.).**
Compare $k$ steps of SGD on minibatches $B_0, \dots, B_{k-1}$ of size $n$ at learning rate $\eta$,
$w_{t+k} = w_t - \eta \frac{1}{n} \sum_{j<k} \sum_{x \in B_j} \nabla l(x, w_{t+j})$,
with one step on their union, of size $kn$, at learning rate $\hat\eta$,
$\hat w_{t+1} = w_t - \hat\eta \frac{1}{kn} \sum_{j<k} \sum_{x \in B_j} \nabla l(x, w_t)$.
If $\nabla l(x, w_t) \approx \nabla l(x, w_{t+j})$ for $j < k$, then setting $\hat\eta = k\eta$ gives $\hat w_{t+1} \approx w_{t+k}$, and the small- and large-minibatch updates are similar. {#c-goyal-interpretation}
The approximation fails in early training, when the network is changing rapidly, and Goyal et al. address this with the warmup phase. {#c-goyal-rapid}
Their argument says when the scaled rate is wrong; it does not by itself pick the shape of the ramp, which they chose by experiment.
A constant low rate followed by a jump to $k\eta$ was not enough for large $k$: the switch out of the low-rate phase can make the training error spike. {#c-constant-spike}
Goyal et al. also cite the analysis of Bottou et al., who suggest that the learning rate should not exceed a maximum rate that does not depend on the minibatch size, which justifies warmup. {#c-bottou}

**Why Adam needs it (Liu et al.).**
Adam divides each step by $\sqrt{v_t}$, where $v_t$ is an exponential moving average of squared gradients; call $\psi_t = \sqrt{(1-\beta_2^t)/v_t}$ the adaptive learning rate.
Liu et al. argue that the root cause of the early convergence problem is that the adaptive learning rate has undesirably large variance in the early stage of training, because only a few training samples have been used. {#c-liu-root-cause}
At $t = 1$ the adaptive learning rate is $\psi(g_1) = \sqrt{1/g_1^2}$; treating the gradients as i.i.d. draws from $N(0, \sigma^2)$, they note that the variance of $\sqrt{1/g_1^2}$ is divergent. {#c-liu-t1}
To see why, assume $g$ has the density $\phi$ of $N(0, \sigma^2)$, with $\phi(0) > 0$ and $\phi \ge \phi(0)/2$ on some interval $[-a, a]$; then $\mathbb{E}[1/g^2] \ge \frac{\phi(0)}{2} \int_{-a}^{a} g^{-2}\, dg = \infty$, and even $\mathbb{E}[1/|g|] = \infty$, since $\int_0^a g^{-1}\, dg$ diverges too. {#c-m-t1-divergent}
A small learning rate early on reduces this variance, because $\mathrm{Var}[\alpha x] = \alpha^2 \mathrm{Var}[x]$. {#c-liu-small-lr}
Under linear warmup, the step $\alpha_t \psi_t$ with $\alpha_t = \alpha_0\, t / T_w$ has variance $(t/T_w)^2 \alpha_0^2 \mathrm{Var}[\psi_t]$, so the first step's share is scaled by $1/T_w^2$. {#c-m-warmup-variance}

To measure how the variance falls, Liu et al. model $\psi_t^2$ as a scaled inverse chi-squared variable with $\rho$ degrees of freedom and prove that the variance of $\psi$ decreases monotonically as $\rho$ increases. {#c-liu-theorem1}
They take $\rho$ to be the length of the simple moving average with the same center of mass as Adam's exponential average, which gives $\rho_\infty = 2/(1-\beta_2) - 1$ and

$$ \rho_t = \rho_\infty - \frac{2 t \beta_2^t}{1 - \beta_2^t} . $$ {#eq-rho}

The variance at $\rho_t = 5$ is over 100 times the variance at $\rho_t = 500$. {#c-liu-100x}
To first order they approximate $\mathrm{Var}[\psi] \approx \rho_t / [2(\rho_t - 2)(\rho_t - 4)\sigma^2]$. {#c-liu-first-order}
Scaling $\psi_t$ by $r_t = \sqrt{\mathrm{Var}[\psi]|_{\rho_\infty} / \mathrm{Var}[\psi]|_{\rho_t}}$ makes its variance equal to the smallest one, reached at $\rho_\infty$; with the first-order approximation, $\sigma^2$ cancels and

$$ r_t = \sqrt{\frac{(\rho_t - 4)(\rho_t - 2)\,\rho_\infty}{(\rho_\infty - 4)(\rho_\infty - 2)\,\rho_t}} , $$ {#eq-rect}

which is the rectification term of RAdam. {#c-m-rect}
RAdam computes this term and takes the adaptive step only when $\rho_t > 4$; otherwise it updates with un-adapted momentum. {#c-radam-alg}
At $t = 1$, $\rho_1 = \frac{2}{1-\beta_2} - 1 - \frac{2\beta_2}{1-\beta_2} = 1$ for every $\beta_2 \in (0, 1)$, so RAdam's first steps are always momentum steps. {#c-m-rho1}
Expanding $\beta_2^t$ to second order in $\delta = 1 - \beta_2$ gives $\rho_t = t + O(t^2 \delta)$, so early in training $\rho_t$ is close to the number of steps taken. {#c-m-rho-t}
Liu et al. observe that $r_t$ has a form similar to linear warmup, which can be read as setting the rectification term to $\min(t, T_w)/T_w$, and take this as support for warmup working as variance reduction. {#c-liu-linear-form}
The two ramps differ in shape: when $4 \ll t$ and $t\delta \ll 1$, eq-rect gives $r_t \approx \sqrt{t / \rho_\infty}$, a square-root rise, not a straight line.

**Why Post-LN Transformers need it (Xiong et al.).**
Xiong et al. use mean field theory to prove that in a freshly initialized Post-LN Transformer, where layer normalization sits between the residual blocks, the parameters close to the output receive large expected gradients; a large learning rate on those gradients destabilizes training, and warmup prevents that. {#c-xiong-post-ln}
With high probability, under a concentration assumption on the hidden states, their Theorem 1 bounds the gradient of the last feed-forward layer by $O(d\sqrt{\ln d})$ for the Post-LN Transformer, independent of the number of layers $L$, and by $O\big(d\sqrt{\ln d / L}\big)$ for the Pre-LN Transformer, which puts the layer normalization inside the residual blocks. {#c-xiong-thm1}
Their hypothesis is that this scale is one reason why Post-LN models need a carefully chosen schedule: some layers get large gradients, so starting at a large rate with no warmup can destabilize training. {#c-xiong-hypothesis}
Their experiments also found that warmup helps SGD, not only Adam, which suggests its benefit is not tied to one optimizer. {#c-xiong-sgd}

The three explanations are not exclusive.
Goyal et al.'s concerns the scaled learning rate of SGD, Liu et al.'s the noise in Adam's denominator, and Xiong et al.'s the gradient scale of one architecture; each predicts that a smaller early step helps, and none of them derives the best warmup length.

## Worked example

**ResNet-50 at minibatch 8k.**
For minibatch sizes $kn$ above 256, Goyal et al. set the reference rate $\eta = 0.1 \cdot kn/256$ and used gradual warmup starting from $\eta = 0.1$ and rising linearly to the reference rate after 5 epochs. {#c-goyal-sweep-setup}
ImageNet has about 1.28 million training images. {#c-imagenet-size}
At $kn = 8192$ the reference rate is $0.1 \times 32 = 3.2$, one epoch is $1.28 \times 10^6 / 8192 = 156.25$ iterations, and 5 epochs are about 781 iterations, so eq-gradual adds $(3.2 - 0.1)/781.25 \approx 0.00397$ to the learning rate per iteration; after 1 epoch the rate is about 0.72, and after 2.5 epochs about 1.65. {#c-we-goyal}

**The Transformer schedule.**
In eq-noam with $d_{\text{model}} = 512$ and 4000 warmup steps, the peak at step 4000 is $1/\sqrt{512 \times 4000} \approx 6.99 \times 10^{-4}$.
The first step uses $1/4000$ of the peak, about $1.75 \times 10^{-7}$; step 1000 uses a quarter of it.
After the peak the rate is the peak times $\sqrt{4000 / s}$: half the peak at step 16,000, a fifth (about $1.40 \times 10^{-4}$) at step 100,000, and about $0.115$ of it (about $8.07 \times 10^{-5}$) at step 300,000. {#c-we-noam}
The base Transformer has $d_{\text{model}} = 512$. {#c-vaswani-dmodel}
Vaswani et al. trained it for 100,000 steps, about 12 hours on 8 P100 GPUs, so the run ends at a fifth of the peak rate. {#c-vaswani-base}

**RAdam's built-in ramp.**
With $\beta_2 = 0.999$, $\rho_\infty = 1999$ and, from eq-rho, $\rho_2 \approx 1.9995$, $\rho_3 \approx 2.9987$, $\rho_4 \approx 3.9975$ and $\rho_5 \approx 4.9960$, so the first adaptive step is at $t = 5$.
From eq-rect, $r_5 \approx 0.017$, $r_{100} \approx 0.215$, $r_{556} \approx 0.50$, $r_{1000} \approx 0.645$, $r_{2000} \approx 0.828$ and $r_{5000} \approx 0.983$: the rectification behaves like a warmup of a few thousand steps whose length is set by $\beta_2$. {#c-we-radam}
At $t = 100$, $\rho_{100} \approx 98.3$ and $\sqrt{100/1999} \approx 0.224$, close to $r_{100}$, as the square-root approximation predicts.
For $\beta_2 = 0.98$, $\rho_\infty = 99$, and for $\beta_2 = 0.9$, $\rho_\infty = 19$.
For $\beta_2 \le 0.6$, $\rho_\infty \le 4$ and RAdam reduces to SGD with momentum. {#c-radam-degenerate}
The first-order formula gives $\mathrm{Var}[\psi]$ at $\rho = 5$ about 820 times its value at $\rho = 500$, since $\frac{5}{2 \cdot 3 \cdot 1} \big/ \frac{500}{2 \cdot 498 \cdot 496} \approx 823$, in line with "over 100 times". {#c-we-ratio}

## Hyperparameters

**Length.**
Goyal et al. used 5 epochs and found the results robust to the exact duration.
For the Post-LN Transformer, Popel and Bojar found the final performance quite sensitive to both the maximum learning rate and the number of warmup iterations. {#c-popel}
With $T_{\text{warmup}} = 500$ in place of 4000, Xiong et al.'s Adam models reached only 31.16 and 2.77 BLEU at $\mathrm{lr}_{\max}$ of $5 \times 10^{-4}$ and $10^{-3}$. {#c-xiong-t500}
The original Transformer used 4000 steps.
For BERT, Xiong et al. trained the Post-LN model with 10k warmup steps and $\mathrm{lr}_{\max} = 10^{-4}$ and the Pre-LN model with a linear decay from $3 \times 10^{-4}$ and no warmup; Post-LN BERT diverged when they tried the larger rate. {#c-bert-warmup}

**Start.**
Goyal et al. start at the small-batch rate, so warmup interpolates between the rate that is known to work and the scaled one.
Linear warmup from zero (eq-linear) starts at $\eta_{\max}/T_w$ on the first step.

**Peak.**
The peak is the learning rate of the main schedule.
Warmup does not choose it, but it makes larger peaks usable; Xiong et al. read the faster training of Pre-LN BERT, a speed-up of 200k steps against a warmup of only 10k, as a sign that the Pre-LN Transformer is easier to optimize with larger learning rates. {#c-xiong-bert-lr}

## Cost

Warmup adds no parameters, no memory and no work per step: the learning rate is a scalar function of the step count.
Its cost is in steps.
During warmup the rate is below the target, so a run that would otherwise be stable makes less progress; Xiong et al. count this slowdown among its disadvantages.
The warmup length and its interaction with the peak rate are extra hyperparameters, and tuning them costs full or partial training runs.

## Evidence

**Large-minibatch SGD (Goyal et al.).**
In Table 1, ResNet-50 on ImageNet with minibatch 256 at $\eta = 0.1$ had a top-1 validation error of 23.60% ± 0.12; at minibatch 8k with $\eta = 3.2$, it was 24.84% ± 0.37 with no warmup, 25.88% ± 0.56 with constant warmup and 23.74% ± 0.09 with gradual warmup, over 5 trials each. {#c-goyal-table1}
The constant warmup used $\eta = 0.1$ for 5 epochs, and the gradual warmup rose linearly from $\eta = 0.1$ to 3.2 over 5 epochs. {#c-goyal-table1-setup}
The no-warmup run's validation error was about 1.2% higher than the baseline, likely because of an increase of about 2.1% in training error, and the gradual-warmup run was within 0.14% of the baseline. {#c-goyal-optimization}
Constant warmup made results worse: the error fell during warmup, spiked right after, and training never fully recovered. {#c-goyal-constant-worse}
With gradual warmup, the small- and large-minibatch training curves matched closely after about 20 epochs, which suggests that large minibatches are held back by optimization difficulties in early training. {#c-goyal-curves}
Across minibatch sizes, with gradual warmup and the linear scaling rule, validation error stayed stable from 64 to 8k and then rose, and beyond 64k training diverged. {#c-goyal-range}
The whole recipe trained ResNet-50 at minibatch 8k in one hour on 256 GPUs. {#c-goyal-1hour}

**Adam on Transformers and other models (Liu et al.).**
For Transformers on IWSLT'14 De-En, removing the warmup stage raised the training loss from 3 to around 10, and similar effects were seen in BERT pre-training. {#c-liu-3-10}
In their translation experiments, vanilla Adam converged to a training perplexity around 500 and needed warmup to stabilize. {#c-liu-ppl}
On CIFAR-10, Adam with warmup reached a best test accuracy of 91.29%, similar to RAdam's 91.38%; at learning rate 0.1, Adam with 100 warmup steps reached only 90.13%, while RAdam reached 91.06%. {#c-liu-cifar}

**Post-LN against Pre-LN (Xiong et al.).**
On IWSLT14 De-En, the Post-LN Transformer trained with Adam reached only 8.45 BLEU without warmup, against around 34 with it, and with SGD the BLEU score stayed just above zero for 15 epochs without warmup. {#c-xiong-845}
A fixed small learning rate of $10^{-4}$ without warmup also mitigated the problem, but the convergence was significantly slower. {#c-xiong-small-lr}
The Pre-LN Transformer without warmup reached a BLEU score around 34 and a validation loss around 4, comparable to the Post-LN Transformer with warmup. {#c-xiong-preln}
RAdam trained the Post-LN Transformer well without warmup, but differed little from Adam on the Pre-LN Transformer, so the position of the layer normalization dominated the choice of optimizer. {#c-xiong-radam}
For BERT pre-training, the Pre-LN model without warmup trained faster, with what Xiong et al. report as a 40% speed-up. {#c-xiong-bert}

## Limitations and failure modes

**It is a heuristic without a derived length.**
Liu et al. note that without a theoretical basis there is no guarantee that warmup helps consistently across settings and no guidance on how to set it, so practitioners fall back on trial and error. {#c-liu-no-theory}
None of the three analyses above gives the warmup length.

**It costs steps and tuning.**
Xiong et al. list two disadvantages: the warmup configuration significantly affects the final performance and must be tuned, which is expensive for large models, and warmup can slow optimization because the rate must climb from zero. {#c-xiong-disadvantages}

**It does not extend the scaling rule without limit.**
Even with gradual warmup, ImageNet training diverged beyond minibatch 64k (see Evidence), and at 8k the results were more sensitive to the learning rate; Goyal et al. suggest staying away from the breaking point. {#c-goyal-sensitive}

**Its explanations disagree.**
Liu et al. tie warmup to the variance of Adam's adaptive rate, but Xiong et al. found that warmup also helps other optimizers, which suggests Adam is not what makes it necessary. {#c-xiong-not-adam}
Goyal et al.'s use is with SGD and momentum, where there is no adaptive rate at all.

## Variants

- *Constant warmup.* A low fixed rate for a few epochs, then a jump to the target. Goyal et al. found that it degrades large-minibatch ResNet training.
- *Gradual (linear) warmup.* A straight ramp from a small rate to the target over $T_w$ steps (eq-gradual, eq-linear), followed by any decay. This is the common default.
- *Inverse-square-root schedule.* Linear warmup to a peak set by $d_{\text{model}}$ and the warmup length, then decay as $s^{-1/2}$ (eq-noam).
- *RAdam.* Adam with the rectification term of eq-rect, which acts like a warmup set by $\beta_2$, with no $T_w$ to tune. {#c-liu-no-tw}
  Liu et al. present the rectification as orthogonal to other stabilizers, such as [[al:gradient-clipping|gradient clipping]], initialization and normalization, and combinable with them. {#c-liu-orthogonal}
- *Pre-LN without warmup.* Moving [[al:layer-normalization|layer normalization]] inside the residual blocks removes the need for warmup in Xiong et al.'s experiments; Nguyen and Salazar (2019) independently observed the same. {#c-nguyen}

## When to use it

Use warmup when the target learning rate is large compared with what the freshly initialized network can take: large-minibatch SGD with the linear scaling rule, Adam or AdamW on Post-LN Transformers, and, by Liu et al.'s argument, any adaptive optimizer whose second-moment estimate starts from few samples.
Liu et al. note that the problem they analyse applies to other optimizers with a similar adaptive rate, such as RMSprop and Nadam. {#c-liu-other-optimizers}
Warmup is cheap insurance: a short ramp costs a small fraction of a long run.
It is less needed for small-batch SGD on convolutional networks, where a large initial rate that then decays is the standard practice, and for Pre-LN Transformers, which Xiong et al. trained without it.

## Pitfalls

- *Counting warmup in epochs when the batch size changes.* Goyal et al.'s 5 epochs are about 781 iterations at minibatch 8k but 25,000 at minibatch 256, so a warmup copied in iterations or in epochs is not the same warmup.
- *Gradient accumulation.* The schedule should advance once per optimizer update, not once per micro-batch, or the warmup ends early by the accumulation factor.
- *A jump at the end of a constant warmup.* Goyal et al. saw the training error spike after the switch from the low constant rate; ramp instead.
- *Changing the warmup length of eq-noam to change the warmup only.* In eq-noam the peak is $(d\,w)^{-1/2}$, so a longer warmup also lowers the peak and every later rate.
- *Expecting the paper's RAdam threshold in PyTorch.* Algorithm 2 adapts when $\rho_t > 4$; torch.optim.RAdam adapts when $\rho_t > 5$, so with $\beta_2 = 0.999$ the first adaptive step is $t = 6$, not $t = 5$, with $r_6 \approx 0.026$. {#c-torch-radam}
- *Relying on warmup alone in a large-batch ResNet.* Goyal et al. also set the scale $\gamma$ of the last [[al:batch-normalization|batch normalization]] layer of each residual block to 0 at initialization, so that the signal first passes through the identity shortcuts, which they found eases optimization at the start of training, especially with large minibatches. {#c-goyal-zero-gamma}
- *Warmup on top of RAdam.* RAdam already ramps its adaptive step; an outer warmup adds a second ramp, and the two lengths interact.
- *Expecting warmup to rescue any scaled rate.* At minibatch 8k, Goyal et al. found that both keeping the rate at 0.1 and the square-root scaling rule, which they also ran with gradual warmup, worked poorly. {#c-goyal-sqrt}

## History

In 2016, He et al. used a low constant learning rate at the start of training for ResNets, the strategy Goyal et al. later called constant warmup.
In June 2017, two papers made the ramp standard.
Goyal et al. combined the linear scaling rule with gradual warmup to train ResNet-50 at minibatch 8k, and Vaswani et al. trained the Transformer with a linear warmup over 4000 steps followed by inverse-square-root decay.
Liu et al. describe warmup as originally proposed to handle very large batches in SGD, and in 2019 they explained its role for Adam as variance reduction and proposed RAdam. {#c-liu-origin}
In 2020, Xiong et al. traced the need for warmup in Transformers to the Post-LN placement of layer normalization and found in experiments that Pre-LN Transformers train without it.
