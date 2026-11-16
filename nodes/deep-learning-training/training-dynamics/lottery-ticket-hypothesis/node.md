---
tag: NEW-lottery-ticket-hypothesis
kind: finding
slug: lottery-ticket-hypothesis
name: Lottery ticket hypothesis
aliases: [winning ticket, lottery tickets, iterative magnitude pruning with reset, IMP]
area: deep-learning-training/training-dynamics
level: core
summary: >-
  A random dense network contains a sparse subnetwork that, trained alone from the same initial weights, matches the full network; at scale this needs a rewind to early training.
related_to: [dropout]
history:
  year: 2018
  note: "Stated by Frankle and Carbin in arXiv 1803.03635 (March 2018) and published at ICLR 2019."
prereqs: [neural-network-pruning@al, xavier-initialization@al, overfitting@al]
see_also: [learning-rate-schedule@al, pruning@al]
same_as:
  arxiv: ["1803.03635", "1912.05671", "1905.01067", "1810.05270", "1911.13299"]
sources:
  - {paper: "Frankle and Carbin (2019)", title: "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks", locator: "pp. 1-9", note: "the hypothesis, the pruning procedure, Lenet, Conv-2/4/6, VGG-19, Resnet-18 and the limitations"}
  - {paper: "Frankle, Dziugaite, Roy and Carbin (2020)", title: "Linear Mode Connectivity and the Lottery Ticket Hypothesis", locator: "pp. 1-4, 7", note: "instability analysis, the error barrier, rewinding and the failure of reset at scale"}
  - {paper: "Zhou, Lan, Liu and Yosinski (2019)", title: "Deconstructing Lottery Tickets: Zeros, Signs, and the Supermask", locator: "pp. 1, 4, 5", note: "ablations of the procedure: signs, zeros and supermasks"}
  - {paper: "Liu, Sun, Zhou, Huang and Darrell (2019)", title: "Rethinking the Value of Network Pruning", locator: "pp. 1, 11", note: "no gain from the ticket initialization at the best learning rate"}
  - {paper: "Ramanujan et al. (2019)", title: "What's Hidden in a Randomly Weighted Neural Network?", locator: "p. 1", note: "subnetworks of random networks that match trained ones without weight training"}
claim: >-
  Iterative magnitude pruning, with the surviving weights reset to their initial values after each round, finds subnetworks of 10-20% of the weights or less that train to the test accuracy of the full network on MNIST and CIFAR10, while the same subnetworks trained from a new random initialization do worse.
first_reported: {work: frankle-2018-lottery-ticket-hypothesis, note: "stated the hypothesis and found winning tickets for Lenet on MNIST and for Conv-2/4/6, VGG-19 and Resnet-18 on CIFAR10"}
replications:
  - {work: frankle-2019-linear-mode-connectivity-lottery, note: "reproduced the result for LeNet, found that reset fails for ResNet-20, VGG-16, ResNet-50 and Inception-v3, and that rewinding to an early iteration restores it"}
  - {work: zhou-2019-deconstructing-lottery-tickets, note: "reproduced the procedure on MNIST and CIFAR-10 and ablated the mask criterion, the reset values and the treatment of pruned weights"}
  - {work: liu-2018-rethinking-value-network-pruning, note: "repeated the comparison with random reinitialization on CIFAR-10 at a larger learning rate and found no gain"}
scale_range:
  - {range: "Lenet 300-100 with 266K weights on MNIST to VGG-19 with 20.0M parameters on CIFAR10", work: frankle-2018-lottery-ticket-hypothesis}
  - {range: "LeNet to ResNet-50 and Inception-v3 on ImageNet, with rewinding", work: frankle-2019-linear-mode-connectivity-lottery}
failures:
  - {work: frankle-2018-lottery-ticket-hypothesis, note: "no winning tickets for Resnet-18 at its original learning rate of 0.1, even with warmup"}
  - {work: liu-2018-rethinking-value-network-pruning, note: "at the optimal learning rate the ticket initialization gives no gain over a random one"}
  - {work: frankle-2019-linear-mode-connectivity-lottery, note: "with a reset to iteration 0, IMP subnetworks of ResNet-20, VGG-16, ResNet-50 and Inception-v3 do not match the full networks"}
explanations:
  - {name: "SGD finds a well-initialized subnetwork", summary: "A large network holds many candidate subnetworks, and training picks out and trains one that started well; this is a conjecture, not a tested result.", sources: [frankle-2018-lottery-ticket-hypothesis]}
  - {name: "stability to SGD noise", summary: "A subnetwork trains to full accuracy when its training outcome no longer depends on the noise of SGD, measured by a low error barrier between two trained copies; at scale this holds only after a short period of dense training.", sources: [frankle-2019-linear-mode-connectivity-lottery]}
  - {name: "zeros, signs and the mask", summary: "Much of what a ticket keeps from its initialization is the sign of each weight and the choice of which weights are zero; masks alone already compute useful functions.", sources: [zhou-2019-deconstructing-lottery-tickets, ramanujan-2019-hidden-randomly-weighted-network]}
---

The lottery ticket hypothesis says that a dense neural network, at its random initialization, already contains a small subnetwork that could be trained alone to the same accuracy.
Frankle and Carbin stated it in 2018 and found such subnetworks, which they called winning tickets, by pruning a trained network and resetting the surviving weights to their initial values.
On MNIST and CIFAR10 the tickets kept 10-20% of the weights or less, and the same sparse networks did worse when trained from a fresh random start.
The result at larger scale needs a change: the weights must be rewound to an early point of training rather than to the initialization, and finding a ticket costs many full trainings.

## In plain words

Train a big network, then cut away most of its weights, keeping the ones that grew largest.
Pruning of this kind is old and works well on a trained network.
The puzzle was that the small network left over usually trained badly when started again from scratch.

Frankle and Carbin tried one change.
Instead of starting the small network from new random numbers, they started it from the exact random numbers it had at the very beginning.
Then the small network trained as well as the big one, and often faster.
So the big network seemed to hold a lucky combination of a small wiring and its starting values: a winning ticket in a lottery.
A random start for the same wiring is like a new ticket, and it usually loses.

## The finding

Once a network is trained, more than 90% of its weights can often be removed with no loss of accuracy. {#c-prune-90}
Yet the same sparse networks, trained again from a new random start, usually train poorly. {#c-hard-to-train}
Frankle and Carbin's hypothesis is that a dense network, at its random initialization, holds a subnetwork that can be trained on its own, from those same initial weights, to the test accuracy of the full network, in no more iterations than the full network needs. {#c-hypothesis}
Formally, a mask $m$ applied to the initial weights $\theta_0$ gives a subnetwork that reaches its minimum validation loss no later than the dense network, with test accuracy at least as high, and with far fewer parameters. {#c-formal}

A winning ticket is identified by training a randomly initialized network, pruning a share of its parameters to create a mask, and resetting the remaining parameters to their values from the initialization. {#c-procedure}
The share pruned is the one with the smallest magnitudes.
In the iterative form, the cycle of training, pruning and resetting runs for several rounds, and each round removes a share of the weights still left; the one-shot form trains and prunes a single time. {#c-iterative}
The method is called iterative magnitude pruning, or IMP.

If each round removes the fraction $p$ of the weights still alive, the fraction left after $n$ rounds is $(1 - p)^n$. {#c-schedule}

$$ P_m(n) = (1 - p)^n, \qquad n^* = \left\lceil \frac{\ln s}{\ln (1 - p)} \right\rceil $$ {#eq-remaining}

Here $P_m$ is the fraction of weights that remain, the paper's notation, and $n^*$ is the least number of rounds that brings it to a target $s$.

$$ (1 - p)^{n^*} \le s < (1 - p)^{n^* - 1} $$ {#eq-rounds}

Reaching a remaining fraction $s$ takes $\lceil \ln s / \ln (1 - p) \rceil$ rounds; at $p = 0.2$, reaching 10% takes 11 rounds and reaching 3.6% takes 15. {#c-rounds}
Pruning the total fraction $P$ in $n$ equal rounds takes the rate $1 - (1 - P)^{1/n}$ per round, which is 20.6% for 90% in 10 rounds. {#c-per-round}
The page's derivation proves both and checks them numerically.

With this procedure, Frankle and Carbin found winning tickets with under 10-20% of the weights, again and again, in fully connected and convolutional networks on MNIST and CIFAR10. {#c-ticket-size}
Down to that size, the tickets trained faster than the full network and ended with better test accuracy. {#c-faster-higher}

## Evidence

**Lenet on MNIST.**
The main test bed was Lenet 300-100, a fully connected network trained with Adam.
With 20% per round on the hidden layers of Lenet 300-100-10 and 10% on its output layer, 7, 9, 15 and 16 rounds leave 21.1%, 13.5%, 3.6% and 2.9% of its 266,200 weights. {#c-lenet-schedule}
These are the pruning levels the paper reports, so its figures follow the schedule of the formula above.
On Lenet, tickets train faster as $P_m$ shrinks from 100% to 21%, where the early-stopping point comes 38% sooner than for the full network; past that, training slows again and is back to the full network's pace at $P_m$ = 3.6%. {#c-lenet-faster}
Test accuracy also rises as pruning goes on, by over 0.3 percentage points at $P_m$ = 13.5%, then falls back to that of the full network at $P_m$ = 3.6%. {#c-lenet-accuracy}

The control is the key part of the evidence.
Keep the mask, but draw new random initial weights for it.
Reinitialized this way, the Lenet subnetworks start losing test accuracy at $P_m$ = 21.1%, while the tickets hold it down to 2.9%; at $P_m$ = 21% a ticket hits its lowest validation loss 2.51x sooner than its reinitialized copy. {#c-reinit-worse}
So the structure of the mask alone does not explain the result; the initial values matter.
One-shot pruning also finds winning tickets, but iterative pruning extracts smaller ones, and repeated training makes them costly to find. {#c-one-shot}

**Convolutional networks on CIFAR10.**
The same pattern held for the small convolutional networks Conv-2, Conv-4 and Conv-6.
Trained with a dropout rate of 0.5, Conv-2, Conv-4 and Conv-6 still have winning tickets, and iterative pruning raises their test accuracy by up to an additional 2.3, 4.6 and 4.7 percentage points. {#c-dropout}
So [[dropout]] and pruning combine rather than cancel in these networks.

**Deeper networks.**
For the deeper Resnet-18 and VGG-19, Frankle and Carbin prune globally, removing the lowest-magnitude weights across all convolutional layers together instead of layer by layer. {#c-global}
Here the procedure was sensitive to the learning rate.
Training VGG-19 with warmup (k = 10000) at learning rate 0.1 makes it possible to find winning tickets that exceed the initial accuracy when Pm >= 1.5%. {#c-warmup}
Resnet-18 tickets with warmup over k = 20000 iterations at learning rate 0.03 got to 90.5% test accuracy at $P_m$ = 27.1%, yet at the network's usual learning rate of 0.1 no setting produced a ticket. {#c-resnet-lr}
A VGG-19 with up to 80% of its weights removed still reaches full accuracy from a random restart; tickets keep working up to 98.5% removed, where a random restart falls well short. {#c-vgg-985}

## Scope and conditions

**Learning rate.**
The deeper networks already suggest that the reset works only under some learning rates.
Liu et al. found that with the optimal learning rate, the winning ticket initialization brings no improvement over random initialization. {#c-liu-lr}
In Liu et al.'s experiments the winning ticket helps only at the small learning rate 0.01, which gives lower accuracy than the learning rate 0.1. {#c-liu-small-lr}
The two groups agree on the facts at moderate sparsity.
They differ on how much the small learning rate setting should count.

**Scale and rewinding.**
Frankle, Dziugaite, Roy and Carbin tested the reset on larger networks and on ImageNet.
Reset to iteration 0, the IMP subnetwork of LeNet reaches 98.2% against 98.3% for the full network, while those of ResNet-20 and ResNet-50 reach 88.5% and 73.7% against 91.7% and 76.1%. {#c-not-matching-k0}
So the original form of the hypothesis, with a reset to the initialization, fails beyond small networks.
Their fix is to rewind the surviving weights to their values $\theta_k$ at an early iteration $k$, instead of to $\theta_0$.
A subnetwork is called matching when it reaches the accuracy of the full network.

To explain when rewinding works, they measure stability to the noise of SGD.
Train two copies of a network from the same point with different data orders, and look at the error on the straight line between the two results.

$$ E_\alpha = E\big(\alpha W_1 + (1 - \alpha) W_2\big), \qquad B = \sup_{\alpha \in [0, 1]} E_\alpha - \tfrac12 \big(E(W_1) + E(W_2)\big) $$ {#eq-barrier}

The error barrier height between two networks is the highest error on the linear path between them minus the mean of the errors at its two ends, and a network is stable to SGD noise when the barrier between two trained copies is below 2%. {#c-barrier-def}
In practice the supremum is taken over a grid of evenly spaced values of $\alpha$.
The error barrier is at least half the gap between the two end errors, so it is never negative, and it equals that half gap when the error is convex along the path. {#c-barrier-bounds}
If the error is L-Lipschitz in alpha along the path, a grid of m evenly spaced alphas underestimates the barrier by at most L / (2 (m - 1)), which is L / 58 for 30 points. {#c-barrier-grid}
Two networks that differ only by a permutation of their hidden units have the same error, and the linear path between them can still have a positive barrier. {#c-barrier-permutation}
The page's derivation proves these facts and checks the last one on a small ReLU network, where the barrier is 0.237.
So a large barrier does not mean the two networks compute different functions, only that the straight path between them leaves the low-error region.

Dense networks become stable to SGD noise early in training: ResNet-20 on CIFAR-10 at 3% of training and ResNet-50 on ImageNet at 20%. {#c-stability-onset}
IMP subnetworks reach full accuracy only when they are stable to SGD noise, which happens at initialization only in small settings such as MNIST and otherwise early in training, for ResNet-50 and Inception-v3 on ImageNet. {#c-lmc-abstract}
With rewinding to an early iteration, IMP subnetworks are matching when they are stable, while randomly pruned and randomly reinitialized subnetworks are unstable and not matching, except for LeNet. {#c-matching-when-stable}
The finding that survives at scale is therefore weaker than the first statement.
A sparse subnetwork can match the full network if it starts from weights after a short stretch of dense training, not from the random initialization.

**Cost.**
IMP does not save training compute.
Iterative pruning is computationally intensive, requiring training a network 15 or more times consecutively for multiple trials, and Frankle and Carbin did not investigate Imagenet. {#c-cost-15}
By the formula above, each extra halving of the remaining weights at 20% per round costs about three more full trainings.
The sparse architectures that magnitude pruning produces are not optimized for modern libraries or hardware. {#c-hardware}
A ticket is a fact about what training found, not a cheaper way to train.

## Explanations

**A search inside the dense network.**
Frankle and Carbin conjecture, and say they do not test, that SGD finds and trains a subnetwork that happened to start well; a larger network would then be easier to train simply because it offers more candidate tickets. {#c-conjecture}
The winning ticket weights move further during training than the other weights. {#c-weights-move}
So a ticket is not a network that was already trained at initialization.

**Stability.**
The account of Frankle et al. ties a ticket's success to the point where training stops depending on SGD noise.
At initialization that point has not been reached for large networks, which is why a reset fails there and a rewind works.
It describes when tickets exist; it does not say why magnitude picks the right weights.

**Zeros, signs and masks.**
Zhou et al. took the procedure apart.
Zhou et al. found that reinitialized tickets work when they keep the sign of the original initial weights, and that even a constant with the right sign performs well. {#c-zhou-sign}
Setting pruned weights to zero matters: freezing them at zero works better than freezing them at their initial values. {#c-zhou-zero}
Masks applied to untrained networks, which Zhou et al. call supermasks, give test accuracy of up to 86% on MNIST and 41% on CIFAR-10. {#c-supermask}
Ramanujan et al. pushed this further.
Hidden in a randomly weighted Wide ResNet-50 is a subnetwork, with random weights, that is smaller than a trained ResNet-34 and matches its performance on ImageNet. {#c-ramanujan}
These results suggest that much of a ticket's value lies in which weights are zero and in their signs, and less in their exact starting values.
They use different procedures from IMP, so they extend the finding rather than test it directly.

## Reproduction here

This page runs its own small experiment, `imp-mlp-synthetic`; it is not a reproduction of the paper's networks or data.
A random teacher network labels 3,000 Gaussian points in 20 dimensions with two classes, and 4,000 more form the test set.
The student is a 20-100-100-2 ReLU MLP without biases, trained with Adam for 500 steps.
The ticket arm runs 4 rounds of layerwise magnitude pruning at 30% per round, 15% on the output layer, with a reset to $\theta_0$ each round.
That leaves 24.47% of the weights, as the schedule formula predicts.
The reinit arm trains the same final mask from a fresh random initialization.
The data are fixed, and 10 seeds vary $\theta_0$ and the minibatch order.

Over the 10 seeds, the mean test accuracy was 0.904 for the ticket, 0.897 for the dense network and 0.877 for the reinitialized mask.
The ticket beat its reinitialized mask in all 10 seeds, by 2.75 points on average (Welch t-test, Holm-corrected p below 1e-9).
It was within the preset margin of 2 points of the dense network (TOST, p below 1e-5).
The run takes about a minute on a laptop CPU.

The scope is narrow.
It is one synthetic task, one width, one pruning level and reset to $\theta_0$, at a scale where the original paper also found reset to work.
The design was fixed after pilot runs on other seeds: a first version that drew new data for every seed was too noisy to separate the arms.
It shows the mechanism of the claim on a CPU; it says nothing about large networks, where reset is known to fail.

## Implications

The finding changed how people think about overparameterization.
A large network may train well partly because it holds many small subnetworks, one of which starts well.
It also suggests that the initial values of weights carry information that pruning after training can exploit, even though training moves them far.

For practice, the gains are indirect.
Finding a ticket costs more than training the dense network once, so IMP is a tool for study, not for saving compute.
Its value is in what it reveals: how sparse a trainable network can be, and when in training the useful structure is fixed.
The rewinding result points to the first few percent of training as the period that sets the sparse structure.

## Open questions

Is there a way to find a winning ticket before or early in training, at a cost below one full training?
Why does weight magnitude after training pick out a subnetwork that trains well from its starting point?
Does the onset of stability depend mainly on the learning rate, the data or the architecture?
Do tickets found on one dataset or task transfer to others?

## History

The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks, by Jonathan Frankle and Michael Carbin of MIT CSAIL, published as a conference paper at ICLR 2019. {#c-hist-frankle}
It was first posted to arXiv in March 2018.
Liu et al. and Zhou et al. published their replications and ablations in 2019, and Ramanujan et al. posted their study of untrained subnetworks the same year.
Linear Mode Connectivity and the Lottery Ticket Hypothesis, by Frankle, Dziugaite, Roy and Carbin, appeared in the Proceedings of the 37th International Conference on Machine Learning in 2020. {#c-hist-lmc}
It turned rewinding from a fix into an account of when tickets exist.

## References
