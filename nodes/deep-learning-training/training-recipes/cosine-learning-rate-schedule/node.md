---
tag: FG7K47
kind: method
slug: cosine-learning-rate-schedule
name: Cosine learning-rate schedule
aliases: [cosine annealing, cosine decay, SGDR, warm restarts, learning-rate decay, step decay, linear decay, warmup-stable-decay, WSD, trapezoidal schedule]
area: deep-learning-training/training-recipes
level: core
summary: >-
  Lower the learning rate from a peak to a floor along half a cosine over a fixed number of steps; the page also covers step decay, linear decay, warm restarts and warmup-stable-decay.
purpose: lowering the learning rate over training so that the last steps are small and the loss settles, with one shape and two numbers to set
uses: [learning-rate-schedule@al, stochastic-gradient-descent@m2]
related_to: [adamw@al, ensemble-learning@al, simulated-annealing@m2]
hyperparameters:
  - {name: peak learning rate, meaning: "eta_max, the rate at the start of the decay, after any warmup", typical: "0.05 to 0.1 for SGD with momentum on CIFAR wide residual networks; 1e-4 to 2e-4 for AdamW on Chinchilla-scale language models"}
  - {name: final learning rate, meaning: "eta_min, the rate at the end of the cycle", typical: "0 in SGDR; 10% of the peak in many language model runs"}
  - {name: cycle length, meaning: "T, the number of steps or epochs over which the rate falls from eta_max to eta_min", typical: "the whole training run, after warmup"}
  - {name: restart period, meaning: "T_0 and T_mult of SGDR: the length of the first run and the factor by which each later run is longer", typical: "T_0 = 10 epochs, T_mult = 2; or no restarts"}
  - {name: cooldown length, meaning: "for warmup-stable-decay, the share of the steps spent decaying at the end", typical: "10% to 20%"}
cost:
  train: "none per step beyond one scalar cosine; the cost is that the run length must be fixed in advance, so stopping early or training longer means a new run"
  inference: "none: the schedule acts only on training"
history:
  year: 2016
  note: "Step decay by a factor of 10 was the usual recipe for convolutional networks such as ResNet (He et al., arXiv 1512.03385). Loshchilov and Hutter put cosine annealing with warm restarts on arXiv in August 2016 (1608.03983, ICLR 2017). Hoffmann et al. (2022) found that the cosine cycle must match the length of training. Hu et al. (2024) named the warmup-stable-decay schedule, and Hägele et al. (2024) compared it with the cosine at scale."
prereqs: [stochastic-gradient-descent@m2, learning-rate-schedule@al]
same_as:
  wikidata: null
  wikipedia: null
  arxiv: ["1608.03983", "2405.18392", "2404.06395"]
  al: null
sources:
  - {paper: "Loshchilov and Hutter (2017)", title: "SGDR: Stochastic Gradient Descent with Warm Restarts", locator: "pp. 1, 2, 4 and 6 to 10", note: "cosine annealing, warm restarts, T_mult, and the CIFAR, ensemble and downsampled ImageNet results"}
  - {paper: "He, Zhang, Ren and Sun (2016)", title: "Deep Residual Learning for Image Recognition", locator: "pp. 4 and 7", note: "step decay by a factor of 10 on ImageNet and CIFAR-10"}
  - {paper: "Hoffmann et al. (2022)", title: "Training Compute-Optimal Large Language Models", locator: "pp. 3, 22 and 24", note: "matching the cosine cycle to the number of training steps"}
  - {paper: "Hägele et al. (2024)", title: "Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations", locator: "pp. 1 to 4", note: "constant learning rate with cooldown against cosine, and the (1-sqrt) cooldown"}
  - {paper: "Hu et al. (2024)", title: "MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies", locator: "pp. 5 and 6", note: "the warmup-stable-decay scheduler"}
  - {paper: "Smith (2017)", title: "Cyclical Learning Rates for Training Neural Networks", locator: "pp. 1 to 3", note: "the triangular cyclical policy"}
---

A learning-rate schedule changes the step size of gradient descent as training goes on.
The cosine schedule starts at a peak rate and lowers it along half a cosine wave until it reaches a floor at the last step.
Ilya Loshchilov and Frank Hutter introduced it inside SGDR, a method of warm restarts for stochastic gradient descent whose aim was better anytime performance. {#c-sgdr-purpose}
Within each run, SGDR sets the rate by cosine annealing from a top value to a bottom value, as a function of the epochs since the last restart. {#c-sgdr-def}
This page covers the cosine and the schedules it is compared with: step decay, linear decay, warm restarts and warmup-stable-decay.
The ramp at the start of training is a separate topic, [[learning-rate-warmup]]; here it is only a prefix.

## In plain words

Early in training a large step size moves the weights quickly toward good regions.
Late in training the same step size makes the weights bounce around the bottom of a valley, because each minibatch gradient is noisy.
Lowering the rate at the end lets the weights settle.
The word annealing comes from simulated annealing, which lowers a temperature that scales random moves; here the randomness comes from the minibatches.

A schedule decides when and how fast to lower it.
Step decay keeps the rate flat and then cuts it, say by 10, at a few fixed points.
Linear decay lowers it by the same amount at every step.
The cosine stays near the peak for a while, drops fastest in the middle, and eases gently into the floor.
Warmup-stable-decay holds the peak for most of training and decays only in a short final phase.

## Setting

Training runs stochastic gradient descent, with or without momentum, or an adaptive method such as AdamW.
The update at step $t$ uses a learning rate $\eta_t$, and the schedule is a function $t \mapsto \eta_t$ fixed before training.
The schedule needs the length of the run, or of each cycle, in steps or epochs.
It does not look at the loss.
A common schedule before the cosine was to hold the rate constant and divide it by a fixed factor at roughly regular intervals. {#c-step-common}

## The method

### Cosine annealing

Fix a peak $\eta_{\max}$, a floor $\eta_{\min}$ and a cycle length $T$.
At time $T_{\mathrm{cur}}$ into the cycle, the rate is

$$ \eta = \eta_{\min} + \tfrac12 (\eta_{\max} - \eta_{\min}) \Big(1 + \cos\big(\pi \tfrac{T_{\mathrm{cur}}}{T}\big)\Big) . $$ {#eq-cosine}

SGDR counts $T_{\mathrm{cur}}$ in epochs but updates it at every batch, so it takes fractional values; the rate is $\eta_{\max}$ at $T_{\mathrm{cur}} = 0$ and $\eta_{\min}$ at $T_{\mathrm{cur}} = T$. {#c-tcur}
With one cycle over the whole run, the schedule has only two numbers to tune: the initial rate and the number of epochs. {#c-two-hyperparameters}
After any warmup, $T$ counts the steps that are left.

### Warm restarts (SGDR)

SGDR splits training into runs $i = 0, 1, \dots$ of length $T_i$ and repeats eq-cosine in each.
A restart keeps the weights and only raises the rate back to the peak, so it is warm, not a new start. {#c-restart-emulated}
The runs may grow: start with a short $T_0$ and multiply the length by $T_{\mathrm{mult}}$ at each restart, as in the settings $T_0 = 1, T_{\mathrm{mult}} = 2$ and $T_0 = 10, T_{\mathrm{mult}} = 2$. {#c-tmult}
Then restart $k$ falls at epoch

$$ t_k = T_0 \, \frac{T_{\mathrm{mult}}^k - 1}{T_{\mathrm{mult}} - 1} \quad (T_{\mathrm{mult}} > 1), \qquad t_k = k T_0 \quad (T_{\mathrm{mult}} = 1). $$ {#eq-restarts}

The authors keep the same $\eta_{\max}$ and $\eta_{\min}$ in every run to save hyperparameters, though they note that lowering them at each restart might help. {#c-same-range}
A restart often makes the model worse for a while, so after the first run SGDR returns the weights at the end of the last finished run, where the rate was at its floor. {#c-incumbent}

### Step decay

Pick milestones $m_1 < m_2 < \dots$ and a factor $\gamma$, and set

$$ \eta_t = \eta_0 \, \gamma^{\#\{j : m_j \le t\}} . $$ {#eq-step}

Kaiming He and colleagues trained ResNets on ImageNet from 0.1 and divided the rate by 10 each time the error stopped improving, for up to 600,000 iterations with a batch of 256. {#c-resnet-imagenet}
On CIFAR-10 they fixed the milestones: 0.1, divided by 10 at 32k and 48k iterations, stop at 64k. {#c-resnet-cifar}
The baseline of the SGDR paper, the wide residual network recipe of Zagoruyko and Komodakis, starts at 0.1 and multiplies by 0.2 at epochs 60, 120 and 160 of 200. {#c-wrn-default}

### Linear decay

$$ \eta_t = \eta_{\max} - (\eta_{\max} - \eta_{\min}) \, \tfrac{t}{T} . $$ {#eq-linear}

### Warmup-stable-decay

Shengding Hu and colleagues split training into three stages: a warmup up to step $W$, a stable stage at the peak up to step $T$, and a decay stage where the rate is the peak times a decreasing factor $f$ with $0 < f \le 1$. {#c-wsd-def}
Alexander Hägele and colleagues write the same schedule with $N$ steps in all, $N_{\mathrm{warmup}}$ of warmup and $N_{\mathrm{decay}}$ of cooldown: {#c-cooldown-def}

$$ \eta(n) = \begin{cases} \eta_{\max} \, n / N_{\mathrm{warmup}} & n < N_{\mathrm{warmup}} \\ \eta_{\max} & N_{\mathrm{warmup}} \le n \le N - N_{\mathrm{decay}} \\ \eta_{\max} \, f\big((n - N + N_{\mathrm{decay}})/N_{\mathrm{decay}}\big) & n > N - N_{\mathrm{decay}} \end{cases} $$ {#eq-wsd}

The same shape was called trapezoidal before it was called warmup-stable-decay, and the cooldown usually goes to zero. {#c-cooldown-names}
The cooldown $f(u)$ can be linear, $1 - u$, or a cosine, or $1 - \sqrt{u}$, which Hägele et al. call (1-sqrt). {#c-one-minus-sqrt}
Its main appeal is that the length of training need not be fixed in advance: the cooldown can start from any checkpoint. {#c-cooldown-advantage}

## Derivation

Four short results explain the shape.
Each has its assumptions and a numeric check in `derivations/`.

**The mean rate** (`mean-rate`).
Over one cycle the cosine term integrates to zero, so the mean of eq-cosine is $(\eta_{\max} + \eta_{\min})/2$. {#c-mean-rate}
At integer steps the mean over $t = 0, \dots, T$ is exactly the same, and over $t = 0, \dots, T - 1$, as a loop of $T$ steps reads it, it is higher by $(\eta_{\max} - \eta_{\min})/(2T)$. {#c-mean-discrete}
So a cosine cycle and a linear decay between the same ends spend the same total learning rate, and they differ only in where they spend it.

**The slope** (`slope`).
Differentiating eq-cosine gives $\eta'(t) = -\frac{\pi}{2T}(\eta_{\max} - \eta_{\min}) \sin(\pi t/T)$, which is zero at the start and at the end of the cycle. {#c-slope-ends}
The drop is fastest at $t = T/2$, where it is $\pi/2 \approx 1.57$ times the slope of the linear decay. {#c-slope-max}
The cosine stays above the linear decay for the first half of the cycle and below it for the second.

**The restart epochs** (`restart-epochs`).
Restart $k$ comes after runs $0$ to $k - 1$, whose lengths form a geometric series; summing it gives eq-restarts. {#c-restart-epochs}
For $T_0 = 10$ and $T_{\mathrm{mult}} = 2$ the restarts fall at epochs 10, 30, 70, 150 and 310. {#c-restart-example}

**Why decay at all** (`noise-floor`).
This toy model is the page's own, not a result of the cited papers.
Take SGD on $f(x) = \tfrac12 h x^2$ whose gradient carries independent noise of variance $\sigma^2$.
The second moment $v_t = \mathbb{E}[x_t^2]$ obeys $v_{t+1} = (1 - \eta_t h)^2 v_t + \eta_t^2 \sigma^2$. {#c-noise-recursion}
With a constant rate it settles at $\eta\sigma^2/(h(2 - \eta h))$, so the loss stalls at a floor of about $\eta\sigma^2/4$ that only a smaller rate can lower. {#c-noise-constant}
With $h = \sigma^2 = 1$ and 1,000 steps from $\eta = 0.1$, the constant rate stays at $v = 0.0526$; the cosine to 0 ends at 0.0036, linear decay at 0.0044, step decay at 0.0034 and a 20% linear cooldown at 0.0100. {#c-noise-decay}
The toy shows why every decay helps.
It does not rank them, since on a quadratic the final variance depends mostly on the last few rates.

## Worked example

Take $\eta_{\max} = 0.1$, $\eta_{\min} = 0$ and $T = 100$ steps.
The cosine gives 0.1 at step 0, 0.0854 at step 25, 0.05 at step 50, 0.0146 at step 75 and 0 at step 100.
Linear decay gives 0.1, 0.075, 0.05, 0.025 and 0.
Step decay with milestones at 50 and 75 and $\gamma = 0.1$ gives 0.1, 0.1, 0.01, 0.001 and 0.001.
WSD with a 20-step linear cooldown gives 0.1 up to step 80, 0.05 at step 90 and 0 at step 100.
The mean of the cosine over steps 0 to 99 is $0.05 + 0.1/200 = 0.0505$.

With a 10-step linear warmup in front and 110 steps in all, the rate is 0 at step 0, 0.05 at step 5, 0.1 at step 10, 0.05 at step 60 and 0 at step 110.

For SGDR with $T_0 = 10$, $T_{\mathrm{mult}} = 2$ and $\eta_{\max} = 0.05$, epoch 20 is halfway through the second run, which lasts from epoch 10 to 30, so the rate is 0.025.
At epoch 29.9 the rate is about $3 \times 10^{-6}$, and at epoch 30 it jumps back to 0.05.
The tests of both implementations check these numbers.

## Hyperparameters

The peak rate matters most, and it is the rate that a constant schedule would use.
SGDR used a peak of 0.05 and a floor of 0 for wide residual networks on CIFAR.
Hoffmann et al. decay the rate 10 times over the cosine, and they assume that a well trained model has its cycle matched to the number of steps of the run. {#c-chinchilla-10x}
In language model training the cosine usually follows a warmup and ends at about 10% of the peak, and it has been the default since GPT and models such as Gopher, PaLM 2 and LLaMA. {#c-llm-cosine-10}
For SGDR, a short $T_0$ and $T_{\mathrm{mult}} = 2$ give good models early; one long cycle gave the best final error in the CIFAR experiments.
For WSD the cooldown length is the main choice; see Evidence.
For cyclical schedules, Leslie Smith suggests a half cycle of 2 to 10 epochs. {#c-clr-stepsize}

## Cost

Computing a cosine per step costs nothing.
The real cost is that the cycle length is fixed before the run.
A run that stops early never reaches the low rates, and a run that goes on past its cycle sits at the floor.
Comparing models at several training lengths then needs one run per length.
WSD removes this cost: one stable run gives a checkpoint at each length, and each needs only a short cooldown.

## Evidence

On WRN-28-10, one cosine cycle of 200 epochs gave a median test error of 3.86% on CIFAR-10, against 4.24% and 4.13% for the step schedule at two starting rates; on CIFAR-100, SGDR with $T_0 = 10, T_{\mathrm{mult}} = 2$ gave 19.58% against 20.33% and 20.21%. {#c-sgdr-table1}
The schedules with restarts reached about 4% and 20% two to four times sooner than the step schedule. {#c-sgdr-faster}
The step schedule began to overfit after about epoch 120, while SGDR overfit only a little. {#c-sgdr-overfit}
Snapshots taken at epochs 30, 70 and 150, just before restarts, made an ensemble that lowered the error from 4.03% to 3.51% on CIFAR-10 and from 19.57% to 17.75% on CIFAR-100; 16 runs of 3 snapshots reached 3.14% and 16.21%. {#c-sgdr-ensemble}
On a downsampled ImageNet with about 50 times fewer pixels per image, SGDR reached 39.24% top-1 and 17.17% top-5 error, close to what AlexNet reached on full-size images. {#c-sgdr-imagenet32}

For language models, Jordan Hoffmann and colleagues found that the best final loss comes from a schedule whose length roughly matches the number of training tokens, at every model size. {#c-chinchilla-match}
A cosine cycle more than 25% longer than the run hurt the loss clearly. {#c-chinchilla-25}
Hu et al. saw the same: for each run length $S$, the cosine with cycle $T = S$ had the lowest loss, and both shorter and longer cycles did worse. {#c-minicpm-cosine-t}

In the WSD decay stage the loss fell quickly to the level of the matched cosine or below, and a checkpoint from before the decay could keep training at the peak and be decayed later. {#c-minicpm-decay-drop}
A decay over 10% of the tokens was enough, and 2.5% was too short. {#c-minicpm-10}
Hägele et al. found that a constant rate with a linear cooldown to zero over 20% of the steps matched the best cosine almost exactly at several training lengths, and was a little less sensitive to the peak rate. {#c-cooldown-match}
The gain from a longer cooldown levelled off at about 20% of the steps, and a (1-sqrt) cooldown over 5% nearly matched the cosine on a 20B-token run. {#c-cooldown-20}
The (1-sqrt) cooldown beat the linear one, by more as runs grew longer. {#c-cooldown-sqrt-better}

## Limitations and failure modes

The cosine is tied to the planned length of the run.
Hoffmann et al. point out that the loss of a run partway through a long cosine overestimates the loss of a run that is planned to stop there; reading scaling trends off such partial runs pushed Kaplan et al. toward models that grow faster than their data. {#c-chinchilla-kaplan}
Continuing a finished cosine helps little because the rate is already low, and raising the rate again causes loss spikes that heal slowly. {#c-cosine-continuation}
Hägele et al. argue that this tie has made scaling research needlessly complex, since each length needs its own run. {#c-hagele-abstract}

The warm restarts of SGDR are meant for anytime performance; Loshchilov and Hutter do not claim that they help the optimizer escape poor local minima. {#c-no-multimodality}
A restart raises the loss for a while, so the model at an arbitrary epoch can be worse than the one before the restart.

None of these schedules adapts to the loss.
A schedule tuned for one run length, batch size or optimizer needs retuning when they change.

## Variants

- **Step decay** (eq-step) cuts the rate by a constant factor at fixed points or on a plateau.
- **Linear decay** (eq-linear) spends the same total rate as the cosine (see Derivation) but drops it evenly.
- **Cosine with warm restarts** (SGDR) repeats the cycle, often with growing lengths.
- **Warmup-stable-decay** (eq-wsd) holds the peak and decays only at the end; the cooldown can be linear, cosine or (1-sqrt).
- **Cyclical learning rates** move the rate up and down between two bounds in straight lines, with a half cycle of a set number of iterations. {#c-clr-def} On CIFAR-10, Smith reached the final accuracy of a fixed schedule, 81.4%, in 25,000 iterations instead of 70,000. {#c-clr-cifar}
- **Snapshot ensembles** keep the model at the end of each cosine cycle and average their predictions. Unlike ordinary ensemble learning, the members come from one training run, so they cost no extra training.

## When to use it

Use a single cosine cycle, after a short warmup, when the number of training steps is known in advance; it has few knobs and is hard to get badly wrong.
Use WSD when the length is not known, when you want models at several lengths from one run, or when you plan to keep training later.
Use step decay to reproduce older vision recipes exactly.
Use SGDR with $T_{\mathrm{mult}} = 2$ when you want a good model early and snapshots for an ensemble.

## Pitfalls

> [!pitfall]
> **The cycle does not match the run.** A cycle set for 100 epochs in a 90-epoch run never reaches the floor, and a cycle shorter than the run sits at the floor. The test `test_cosine_reaches_floor` checks that the rate is $\eta_{\min}$ at the last step and stays there.

> [!pitfall]
> **PyTorch's `CosineAnnealingLR` is periodic.** Past `T_max` it does not stay at `eta_min`; it climbs back toward the peak. The test `test_cosine_annealing_lr_past_t_max` shows the rate back at the peak at step `2 * T_max`. Set `T_max` to the number of scheduler steps of the whole run.

> [!pitfall]
> **Steps and epochs are mixed.** A scheduler stepped once per batch with a length given in epochs finishes its cycle far too early. The PyTorch tests step once per optimizer update and give all lengths in updates.

> [!pitfall]
> **The warmup shifts the cycle.** After a warmup of $W$ steps the cosine must run over the remaining $T - W$ steps, or it ends early or late. The test `test_warmup_then_cosine` checks the rate at both joints.

> [!pitfall]
> **The off-by-one in the mean.** The mean of $T$ rates read at $t = 0, \dots, T-1$ is not exactly the midpoint; see the derivation. The test `test_mean_rate` checks both forms.

## History

Before the cosine, convolutional networks were usually trained with step decay, as in the ResNet recipes above.
Loshchilov and Hutter posted SGDR on arXiv in 2016, and their report led Huang et al. to build snapshot ensembles from the models just before each restart. {#c-snapshot-history}
The cosine cycle without restarts then spread to language models.
Hoffmann et al. (2022) made the match between cycle and run length a rule for scaling studies.
In 2024 Hu et al. named warmup-stable-decay for MiniCPM, and Hägele et al. found that it matches the cosine while freeing the run length.
