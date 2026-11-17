---
tag: MDXMTA
kind: finding
slug: mode-connectivity
name: Mode connectivity
aliases: [linear mode connectivity, loss barrier, connected minima, LMC]
area: deep-learning-training/loss-landscapes
level: core
summary: >-
  Networks trained from different seeds have a high-loss barrier on the straight line between them, yet simple curves of low loss join them, and after a permutation of units often a straight line.
related_to: [ensemble-learning@al, neural-network-pruning@al, hungarian-algorithm@m2, convex-optimization@m2]
history:
  year: 2018
  note: "Found at the same time by Garipov, Izmailov, Podoprikhin, Vetrov and Wilson (NIPS 2018, arXiv 1802.10026) and Draxler, Veschgini, Salmhofer and Hamprecht (ICML 2018, arXiv 1803.00885); the linear form is from Frankle et al. (ICML 2020), the permutation form from Entezari et al. (ICLR 2022) and Ainsworth et al. (ICLR 2023)."
prereqs: [loss-function@al, stochastic-gradient-descent@m2, relu@al, permutation-matrix@m2]
same_as:
  arxiv: ["1802.10026", "1803.00885", "1912.05671", "2110.06296", "2209.04836", "1412.6544"]
sources:
  - {paper: "Garipov et al. (2018)", title: "Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs", locator: "pp. 1-6, 14", note: "curves of near-constant loss, the curve loss, Bezier and polygonal chain curves, Fast Geometric Ensembling"}
  - {paper: "Draxler et al. (2018)", title: "Essentially No Barriers in Neural Network Energy Landscape", locator: "pp. 1-2, 6", note: "minimum energy paths found with AutoNEB, and the trend with depth and width"}
  - {paper: "Frankle et al. (2020)", title: "Linear Mode Connectivity and the Lottery Ticket Hypothesis", locator: "pp. 1-5", note: "the error barrier, instability analysis, stability early in training, and IMP subnetworks"}
  - {paper: "Entezari et al. (2022)", title: "The Role of Permutation Invariance in Linear Mode Connectivity of Neural Networks", locator: "pp. 3-7", note: "the linear-reference barrier, width and depth, the permutation conjecture and the one-layer theorem"}
  - {paper: "Ainsworth et al. (2023)", title: "Git Re-Basin: Merging Models modulo Permutation Symmetries", locator: "pp. 1-8", note: "weight matching, zero-barrier ResNets on CIFAR-10, width, onset during training, and a counterexample"}
  - {paper: "Goodfellow, Vinyals and Saxe (2015)", title: "Qualitatively characterizing neural network optimization problems", locator: "pp. 2, 5, 7, 10", note: "the linear interpolation probe and the barrier between two solutions"}
claim: >-
  Two networks of the same architecture trained from different random seeds usually have a high loss barrier on the straight line between their weights, but are joined by simple curves of near-constant loss; once the hidden units of one are permuted to match the other, the straight-line barrier shrinks and, for wide enough networks, can vanish.
first_reported: {work: garipov-2018-loss-surfaces-mode-connectivity, note: "curves of near-constant train loss and test accuracy between independently trained VGG, ResNet and Wide ResNet networks on CIFAR-10 and CIFAR-100; Draxler et al. (2018) found the same independently"}
replications:
  - {work: draxler-2018-essentially-no-barriers, note: "AutoNEB paths between ten minima per architecture for CNNs, ResNets and DenseNets on CIFAR10 and CIFAR100"}
  - {work: frankle-2019-linear-mode-connectivity-lottery, note: "linear connectivity between copies trained from a shared early checkpoint, from LeNet to ResNet-50 and Inception-v3 on ImageNet"}
  - {work: ainsworth-2023-git-re-basin, note: "linear connectivity after weight matching for MLPs, VGG and ResNets on MNIST, CIFAR-10 and CIFAR-100"}
scale_range:
  - {range: "VGG-16 to ResNet-164 and Wide ResNet-28-10 on CIFAR-10 and CIFAR-100", work: garipov-2018-loss-surfaces-mode-connectivity}
  - {range: "LeNet on MNIST to ResNet-50 and Inception-v3 on ImageNet", work: frankle-2019-linear-mode-connectivity-lottery}
  - {range: "MLPs to ResNet20 at 32x width on CIFAR-10 and ResNet-50 on ImageNet", work: ainsworth-2023-git-re-basin}
failures:
  - {work: frankle-2019-linear-mode-connectivity-lottery, note: "copies split at initialization are not linearly connected except for LeNet on MNIST"}
  - {work: ainsworth-2023-git-re-basin, note: "1x-width models and ResNet-50 on ImageNet keep a barrier after matching; adversarial non-SGD solutions admit no permutation that connects them"}
  - {work: entezari-2022-permutation-invariance-mode-connectivity, note: "without permutation, barriers grow with depth and stay high for VGG and ResNet families"}
explanations:
  - {name: "permutation symmetry", summary: "Many apparent basins are copies of one another under reorderings of hidden units; the straight line between two copies crosses high loss, and undoing the permutation removes most of the barrier.", sources: [entezari-2022-permutation-invariance-mode-connectivity, ainsworth-2023-git-re-basin]}
  - {name: "a connected low-loss set", summary: "In large networks the minima are not isolated valleys but parts of a connected set of low loss, whose saddles get lower as the networks get deeper and wider.", sources: [draxler-2018-essentially-no-barriers, garipov-2018-loss-surfaces-mode-connectivity]}
  - {name: "the bias of SGD", summary: "Linear connectivity is a property of the solutions that SGD finds, not of every minimum: it appears during training, and adversarially built solutions lack it.", sources: [ainsworth-2023-git-re-basin, frankle-2019-linear-mode-connectivity-lottery]}
---

Mode connectivity is the finding that the minima found by training a neural network are not isolated.
Two networks trained from different random seeds reach similar loss at very different weights, and the straight line between them crosses a high barrier.
Yet a simple curve between them, with one bend, keeps the loss low all the way.
If the hidden units of one network are first reordered to line up with the other, even the straight line often keeps the loss low.
That last form is called linear mode connectivity.

## In plain words

A network has many weights, and training moves them downhill on the loss until it stops in a low region, a mode.
Start again from another random seed and training stops somewhere else, equally low.
Walk straight from one stopping point to the other and the loss rises in the middle, so they look like separate valleys.

The finding is that they are usually not separate.
A path that bends once can join them while staying low.
Part of the reason is simple bookkeeping.
The hidden units of a layer can be listed in any order without changing what the network computes, so the same solution sits at many places in weight space.
Two runs may have found the same kind of solution with its units listed in different orders.
Put the units in the same order and the straight line between them is often low too.

## The finding

The basic probe is old.
Goodfellow, Vinyals and Saxe (2015) evaluated the objective at points $\theta = (1-\alpha)\theta_0 + \alpha\theta_1$ along the line between two weight vectors and looked at the cross-section of the loss along it. {#c-goodfellow-probe}
Write $L(t)$ for the loss at the point

$$ \phi(t) = (1-t)\,w_1 + t\,w_2, \qquad t \in [0, 1]. $$ {#eq-linear}

The height of the bump on this path is the loss barrier.
Two definitions are in use:

$$ B_{\text{mean}} = \max_{t} L(t) - \tfrac12\big(L(0) + L(1)\big), $$ {#eq-barrier-mean}

$$ B_{\text{lin}} = \max_{t}\,\big[\,L(t) - (1-t)L(0) - tL(1)\,\big]. $$ {#eq-barrier-linear}

Frankle et al. (2020) use the first, on the error: the error barrier height is the supremum along the path minus the mean error of the two ends; two networks are mode connected if some path between them has a barrier near 0, and linearly mode connected if the straight path does. {#c-frankle-barrier}
Entezari et al. (2022) use the second; it equals the first when the two end losses are equal, and it gives no barrier to a loss that changes linearly along the path. {#c-entezari-barrier}
Ainsworth et al. (2023) note that barriers are non-negative, so a path that dips below both ends still gets a positive barrier, which they call a shortcoming of the measure. {#c-ainsworth-nonneg}
Both barriers are non-negative because the ends are on the path; for a loss that is convex along the segment, as in convex optimization, $B_{\text{lin}} = 0$ and $B_{\text{mean}}$ is half the difference of the end losses. {#c-barrier-convex}
The two differ by at most half the difference of the end losses, agree at $t = \tfrac12$, and are equal when the end losses are equal. {#c-barrier-relation}
The derivation `loss-barrier` gives the steps.

Garipov et al. (2018) found that the optima of deep networks are connected by simple curves, such as a polygonal chain with one bend or a quadratic Bezier curve, along which training and test accuracy are nearly constant. {#c-garipov-curves}
The curves have endpoints $w_1, w_2$ and a trainable bend $\theta$:

$$ \phi_\theta(t) = (1-t)^2 w_1 + 2t(1-t)\,\theta + t^2 w_2 . $$ {#eq-bezier}

This is the quadratic Bezier curve; the polygonal chain runs straight from $w_1$ to $\theta$ for $t \le 0.5$ and from $\theta$ to $w_2$ after. {#c-bezier-def}
The Bezier curve passes through $w_1$ and $w_2$ at the ends, and its midpoint is $(w_1 + 2\theta + w_2)/4$, not $\theta$. {#c-bezier-points}
To train the bend, Garipov et al. minimize the loss averaged over a uniform $t$,

$$ \ell(\theta) = \mathbb{E}_{t \sim U(0,1)}\, L(\phi_\theta(t)), $$ {#eq-curve-loss}

with one sampled $t$ per step; they chose this loss because, for Bezier curves, the gradient of the loss averaged uniformly over the length of the curve is intractable. {#c-curve-loss}
The gradient in $\theta$ at one $t$ is $2t(1-t)\nabla L(\phi_\theta(t))$, so one sample gives an unbiased estimate of the gradient of $\ell$. {#c-bend-gradient}
A toy case shows what a bend buys.
On the ring loss $(\lVert w\rVert^2 - 1)^2$ with minima $(-1, 0)$ and $(1, 0)$, the segment has barrier 1 at its midpoint, while the Bezier curve with bend $(0, 2)$ has barrier $1/16$, where $t(1-t) = 1/8$. {#c-ring-example}
The derivation `bezier-curve` gives the steps.

The linear form needs a second idea.
Permuting the hidden units of a one-hidden-layer network, the rows of $W_1$ and entries of $b_1$ with the matching columns of $W_2$, does not change its function, so every network has $h!$ weight settings with the same outputs: {#c-perm-invariance}

$$ (W_1, b_1, W_2, b_2) \;\mapsto\; (P W_1,\; P b_1,\; W_2 P^\top,\; b_2). $$ {#eq-permute}

The straight line between two such copies need not be low.
The two-unit ReLU network with input weights $(1, -1)$ and output weights $(1, 1)$ computes $\lvert x\rvert$, as does its copy with the units swapped, but the network halfway between is the constant 0, and the squared error on $x = \pm 1$ rises from 0 at the ends to 1 in the middle. {#c-perm-barrier}
Entezari et al. conjecture that most SGD solutions belong to a set whose elements can be permuted so that there is no barrier on the straight line between any two of them. {#c-entezari-conjecture}
Weight matching picks the permutation of network $B$ that brings its weights closest to network $A$.
With one hidden layer it is a single linear assignment problem, solved in $O(h^3)$ time by the Hungarian algorithm, maximizing $\sum_i C_{i,\pi(i)}$ for {#c-weight-matching-lap}

$$ C = W_1^A (W_1^B)^\top + b_1^A (b_1^B)^\top + (W_2^A)^\top W_2^B . $$ {#eq-matching}

With more layers, Ainsworth et al. prove that the problem is a sum of bilinear assignments, which is NP-hard, and they solve it approximately by coordinate descent, one layer's permutation at a time. {#c-ainsworth-sobla}
The derivation `permutation-symmetry` covers the invariance, the example and the one-layer matching.

## Evidence

Before 2018 the line between two solutions was known to be high.
For seven models of practical interest, Goodfellow et al. found a straight path from initialization to solution along which the objective falls smoothly and monotonically, at the resolution of their experiments. {#c-goodfellow-init}
Between two solutions from different random seeds, the line can cross a barrier of higher cost. {#c-goodfellow-two}
In a deep linear regression they gave two solutions on a connected manifold of global minima with a barrier on the straight line between them. {#c-goodfellow-linear}

Garipov et al. measured the segments directly.
For VGG-16 on CIFAR-10 the test error reaches 90% at the centre of the segment, and for ResNet-158 and Wide ResNet-28-10 the worst errors on the segment are 80% and 66%. {#c-garipov-segment}
The trained curves between the same pairs stay low.
The points on a curve are not copies of one network.
An ensemble of 50 networks spaced along a polygonal chain between two ResNet-164 on CIFAR-100 had 21.03% test error, against 22.0% for the two endpoints together and 21.01% for three independently trained networks. {#c-garipov-ensemble}
From this, Garipov et al. argue that the curves are not just degenerate reparametrizations, such as rescaling around a ReLU, and that points on them are meaningfully different representations. {#c-garipov-not-rescaling}

Draxler et al. (2018) looked for the path whose highest loss is lowest, the minimum energy path, and called the highest point on a path its saddle. {#c-draxler-mep}
They used the Automated Nudged Elastic Band algorithm, a method from molecular statistical mechanics. {#c-draxler-neb}
The paths they built between minima on CIFAR10 and CIFAR100 are essentially flat in both the training and the test landscape. {#c-draxler-flat}
Garipov et al. note that Draxler et al. found curves between optima simultaneously and independently, with a different method. {#c-garipov-simultaneous}

Frankle et al. (2020) asked when the straight line is low.
Instability analysis trains two copies of a network from the same weights $W_k$ under different SGD noise, such as data order and augmentation, and measures the error barrier between the results; an instability below 2% counts as stable. {#c-frankle-instability}
Split at initialization, only LeNet on MNIST is stable; for the others, train and test error rise to random guessing along the line. {#c-frankle-init}
Split a little later, they are stable: ResNet-20 and VGG-16 after 3% and 1.5% of training, ResNet-50 and Inception-v3 on ImageNet at epoch 18 (20% of training) and epoch 28 (16%). {#c-frankle-early}

Entezari et al. measured the unpermuted barrier between solutions from different initializations.
It rises with width up to a point and then falls, which they liken to double descent. {#c-entezari-width}
For MLPs of width $2^{10}$, it rises fast as layers are added. {#c-entezari-depth}
Ainsworth et al. then aligned the networks by matching.
Weight matching runs in seconds and needs no input data. {#c-ainsworth-seconds}
After matching, wide ResNets trained independently on CIFAR-10 show a barrier of zero, the first such result by their account. {#c-ainsworth-zero}

## Scope and conditions

The curved form is the robust one.
Garipov et al. and Draxler et al. found low curves across every architecture they tried, from VGG to DenseNet.
Deeper and wider networks have lower saddles, to the point where they essentially vanish; CIFAR100 raises them, and the error at the saddle grows by at most 0.5% on CIFAR10 and 2.2% on CIFAR100 for the deep architectures. {#c-draxler-depth-width}

The linear form depends on more things.
The two runs need a shared start late enough in training, or a permutation that aligns them.
Width matters: at 1x width, VGG-16 and ResNet20 did not seem linearly connected after matching, while wider versions brought the barrier down to zero. {#c-ainsworth-width}
On ImageNet, matching lowered the barrier of ResNet-50 by 67% against plain interpolation but did not remove it. {#c-ainsworth-imagenet}
Time matters too: Ainsworth et al. found linear connectivity after matching to be a property that emerges during training, and could not find it early. {#c-ainsworth-emergent}
The measured barrier also depends on the convention, $B_{\text{mean}}$ or $B_{\text{lin}}$, and on whether loss or error is measured, on the train or the test set.

## Explanations

Permutation symmetry accounts for much of the straight-line barrier.
The invariance above gives every solution $h!$ copies per layer of width $h$.
Entezari et al. prove that for a wide enough one-hidden-layer network at random initialization, a permutation exists that leaves no barrier on the line, with an error that shrinks as the width grows. {#c-entezari-theorem}
They test the conjecture by search, because brute force over permutations works in reasonable time only up to 24 neurons in one hidden layer, and not at all for deep networks. {#c-entezari-brute}

A second account is that the low-loss region is one connected set.
In the reading of Draxler et al., the minima may be better pictured as points of one connected low-loss manifold than as the floors of separate valleys. {#c-draxler-manifold}
This explains the curves but not why a straight line should work.

A third account puts the cause in the optimizer.
Ainsworth et al. construct adversarial, non-SGD solutions that no permutation connects linearly, and read this as evidence that SGD and its relatives are biased toward solutions that admit linear connectivity. {#c-ainsworth-counter}
The early stability that Frankle et al. measure fits the same picture: after a short time, the noise of SGD no longer changes which linearly connected region training ends in.

## Reproduction here

The NumPy and PyTorch implementations train two 2-64-2 ReLU MLPs by full-batch gradient descent on 256 points of two noisy half-moons, from two seeds, for 2,000 steps at learning rate 0.5.
Both reach a train loss below 0.01.
The demo then measures $B_{\text{mean}}$ of the train loss on 25 points of three paths: the plain segment, a Bezier curve whose bend is trained for 2,000 steps of the curve loss, and the segment after weight matching.
Over ten seed pairs in NumPy, the plain barrier ranged from 0.052 to 0.278 (median 0.133).
The Bezier barrier ranged from 0.009 to 0.017, and the matched barrier from 0.0007 to 0.0024.
Every pair showed both drops.
The PyTorch run, with its own random numbers, gave 0.043 to 0.220, 0.009 to 0.015, and 0.0002 to 0.0034.
The tests check one pair with fixed seeds.
A check at width 8 while writing the page was less clean: matching lowered the barrier in 9 of 10 pairs but raised it in one, and left barriers up to 0.22.
This fits the width effect above, but it is one small run, and the page does not claim it.
The tests also check the Hungarian solver against brute force and SciPy, and weight matching on a network permuted by a known permutation.

## Implications

Mode connectivity changed ensemble learning and the averaging of weights.
Garipov et al. used it to propose Fast Geometric Ensembling, which trains a good ensemble in the time of one model. {#c-garipov-fge}
Linear connectivity after matching is what makes it reasonable to average the weights of separately trained models.
In neural network pruning, Frankle et al. found that sparse subnetworks from iterative magnitude pruning reach full accuracy only when they are stable in this linear sense, which ties the finding to the [[lottery-ticket-hypothesis]]. {#c-frankle-imp}
It also bears on flatness: if minima form a connected set, the sharpness of one minimum, the quantity that [[sharpness-aware-minimization]] lowers, describes a point of that set and not an isolated valley.

## Open questions

Whether the permutation conjecture holds for deep, narrow, or very large networks is open; ImageNet models still keep a barrier after matching.
It is not settled whether the residual barrier comes from weak matching methods or from symmetries other than permutations.
The link between linear connectivity and the training dynamics that set the curvature of a minimum, as at the [[edge-of-stability]], has not been worked out.
Which measure of the barrier should be standard is also unsettled.

## History

Goodfellow, Vinyals and Saxe introduced straight-line interpolation as a probe at ICLR 2015.
Garipov et al. (NIPS 2018) and Draxler et al. (ICML 2018) found low curves between independent minima, independently and at the same time.
Frankle et al. (ICML 2020) defined linear mode connectivity and instability analysis.
Entezari et al. (ICLR 2022) proposed that permutations explain the remaining barriers.
Ainsworth et al. (ICLR 2023) gave fast matching methods and the first zero-barrier ResNets.

## References
