---
tag: NEW-flow-matching
kind: architecture
slug: flow-matching
name: Flow matching
aliases: [conditional flow matching, CFM, rectified flow, reflow, stochastic interpolant, InterFlow, independent conditional flow matching, I-CFM, optimal transport conditional flow matching, OT-CFM, minibatch optimal transport flow matching, Schrödinger bridge conditional flow matching]
area: generative-models/flow-matching
level: core
summary: >-
  A generative model that trains a velocity field by regressing it on the velocities of simple paths from noise to data, and samples by solving the learned ODE from noise.
uses: [multivariate-normal-distribution@m2, conditional-expectation@m2, mean-squared-error@m2, expected-value@m2, linearity-of-expectation@m2, independence-of-random-variables@m2, mixture-model@m2, trace-of-a-matrix@m2, coupling@m2, convex-optimization@m2, hungarian-algorithm@m2, matrix-multiplication@m2, maximum-likelihood-estimation@al, backpropagation@al]
related_to: [diffusion-model, normalizing-flow, neural-ordinary-differential-equation, variational-autoencoder, generative-adversarial-network, multilayer-perceptron, transformer, adam]
history:
  year: 2022
  people: [Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le, Xingchao Liu, Chengyue Gong, Qiang Liu, Michael S. Albergo, Eric Vanden-Eijnden]
  note: "Three groups posted the idea in the autumn of 2022: Lipman and colleagues as flow matching, Liu, Gong and Liu as rectified flow, and Albergo and Vanden-Eijnden as stochastic interpolants. It trains the continuous normalizing flows of Chen and colleagues (2018) without simulating them. Tong and colleagues generalized it to any source distribution and to optimal transport couplings (2023), and Esser and colleagues scaled rectified flow to an 8B text-to-image transformer (2024)."
prereqs: [diffusion-model, neural-ordinary-differential-equation, normalizing-flow, multivariate-normal-distribution@m2, conditional-expectation@m2]
same_as:
  wikipedia: Flow_matching
sources:
  - {paper: "Lipman, Chen, Ben-Hamu, Nickel and Le (2022)", title: "Flow Matching for Generative Modeling", locator: "arXiv 2210.02747v2, pp. 1 to 8", note: "continuous normalizing flows, the flow matching loss, Theorems 1 and 2, the Gaussian conditional paths, the optimal transport path and its field, and the image results"}
  - {paper: "Liu, Gong and Liu (2022)", title: "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow", locator: "arXiv 2209.03003v1, pp. 1, 4 to 6", note: "the linear interpolation, the causalized flow, non-crossing paths, marginal preservation, the transport cost theorem, reflow and one-step results"}
  - {paper: "Albergo and Vanden-Eijnden (2022)", title: "Building Normalizing Flows with Stochastic Interpolants", locator: "ICLR 2023; arXiv 2209.15571v3, pp. 1, 2", note: "the velocity from the probability current of an interpolating density, the quadratic loss and the trigonometric interpolant"}
  - {paper: "Tong, Fatras, Malkin, Huguet, Zhang, Rector-Brooks, Wolf and Bengio (2023)", title: "Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport", locator: "TMLR 03/2024; arXiv 2302.00482v4, pp. 1, 7, 8", note: "generalized conditional flow matching, I-CFM, minibatch OT-CFM and SB-CFM"}
  - {paper: "Esser and colleagues (2024)", title: "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis", locator: "arXiv 2403.03206v1, pp. 1, 3 to 6", note: "the rectified flow convention with data at t = 0, logit-normal timestep sampling and the comparison of 61 formulations"}
  - {paper: "Chen, Rubanova, Bettencourt and Duvenaud (2018)", title: "Neural Ordinary Differential Equations", locator: "NeurIPS 2018; arXiv 1806.07366v5, p. 4", note: "the instantaneous change of variables"}
family: flow-matching
inputs: "for training a batch of data points x_1 as a (B, D) array, with noise x_0 of shape (B, D) and times t of shape (B,) drawn inside the loss; for sampling a batch of Gaussian draws x_0 as a (B, D) array"
outputs: "for training the predicted velocities as a (B, D) array and a scalar loss; for sampling B points as a (B, D) array; for evaluation the log-likelihoods of B points as a (B,) array"
components: []
inductive_biases: ["data are reached from noise along a flow, the solution of an ordinary differential equation, so the map from noise to data is smooth and invertible", "the network learns the average velocity of simple, usually straight, paths from noise to data, and the paths of the learned flow are as straight as that average allows", "one network shared across all times, told the time through features of t, learns the velocity at every point of the path", "the loss weights the times by the distribution t is drawn from, chosen for sample quality rather than likelihood"]
params: "(D + E + 1)H + (H + 1)D for the velocity network of this page, one tanh hidden layer of H units on D data coordinates and E Fourier features of t"
flops_per_example: "2H(2D + E) for one evaluation of the velocity network, counting a multiply-add as 2 FLOPs and only the matrix products; a sample costs one evaluation per Euler step and two per midpoint step"
---

Flow matching is a way to train a generative model that turns noise into data by following a velocity field.
A neural network takes a point and a time $t \in [0, 1]$ and returns a velocity. Starting from a noise sample at $t = 0$ and moving along that velocity until $t = 1$ produces a data sample.
Training needs no simulation: for each data point the method fixes a simple path from noise to that point, usually a straight line, and fits the network by least squares to the velocity of the path.
Lipman and colleagues present Flow Matching as a simulation-free approach for training continuous normalizing flows based on regressing vector fields of fixed conditional probability paths. It is compatible with a general family of Gaussian probability paths for transforming between noise and data samples, which subsumes existing diffusion paths as specific instances. {#c-fm-def}
Rectified flow and stochastic interpolants are the same idea, found at the same time, and flow matching now trains text-to-image models in place of diffusion.

## In plain words

Picture a cloud of dust, the noise, and a statue, the data, standing next to each other.
You want a wind, a velocity at every point of space and every moment, that blows the dust cloud into the shape of the statue in one second.
Finding that wind directly is hard.
Pairing things up is easy: take one speck of dust and one point of the statue, and imagine the speck flying to the point in a straight line at constant speed.
Do this for millions of pairs, and at any place and moment some specks are passing through, each with its own heading.
The wind you want is the average heading of the specks that pass through that place at that moment.
A network can learn that average by least squares, guessing the velocity of a random speck at a random moment and being corrected, without ever simulating the wind.
To make a new statue point, drop a fresh speck of dust and let the learned wind carry it for one second.

## Overview

A time-dependent vector field $v_t : \mathbb{R}^D \to \mathbb{R}^D$, $t \in [0, 1]$, defines a flow by the ordinary differential equation

$$ \frac{d}{dt}\mathbf{x}_t = v_t(\mathbf{x}_t), \qquad \mathbf{x}_0 \sim p_0 = \mathcal{N}(0, I) . $$ {#eq-ode}

A time-dependent vector field $v_t$ defines a flow $\phi_t$ through the ODE $\frac{d}{dt}\phi_t(\mathbf{x}) = v_t(\phi_t(\mathbf{x}))$ with $\phi_0(\mathbf{x}) = \mathbf{x}$. Chen and colleagues suggested modeling the field with a neural network, which gives a continuous normalizing flow: a model that reshapes a simple prior density $p_0$, such as pure noise, into a more complicated one, $p_1$, through the push-forward equation. {#c-cnf}
The density $p_t$ of $\mathbf{x}_t$ then obeys the continuity equation $\partial_t p_t + \nabla \cdot (p_t v_t) = 0$, and along a path its log-density changes by minus the divergence of the field:

$$ \log p_1(\mathbf{x}_1) = \log p_0(\mathbf{x}_0) - \int_0^1 \nabla \cdot v_t(\mathbf{x}_t)\, dt . $$ {#eq-change-of-variables}

If a point moves by $d\mathbf{x}/dt = v_t(\mathbf{x})$ and its density is $p_t$, then $\frac{d}{dt}\log p_t(\mathbf{x}_t) = -\nabla \cdot v_t(\mathbf{x}_t)$, the trace of the Jacobian of $v_t$. The log-density of a sample at $t = 1$ is therefore $\log p_0(\mathbf{x}_0)$ minus the integral of the divergence along its path, with no determinant to compute. {#c-icov}
The derivation `change-of-variables` proves it.
Chen and colleagues state this instantaneous change of variables for a continuous-in-time transformation $d\mathbf{z}/dt = f(\mathbf{z}(t), t)$, with $f$ uniformly Lipschitz continuous in $\mathbf{z}$ and continuous in $t$: the log probability follows a differential equation, and instead of the log determinant of a discrete normalizing flow only a trace operation is required. {#c-chen-trace}

Before 2022, continuous normalizing flows were trained by maximum likelihood through eq-change-of-variables. Every training step solved the ODE, which made training slow.
Flow matching instead regresses the network on a field $u_t$ that is known to carry $p_0$ to the data:

$$ \mathcal{L}_\text{FM}(\theta) = \mathbb{E}_{t, \mathbf{x} \sim p_t} \lVert v_\theta(\mathbf{x}, t) - u_t(\mathbf{x}) \rVert^2 . $$ {#eq-fm-loss}

Lipman and colleagues note that upon reaching zero loss the learned flow generates $p_t$. Used naively, however, the loss is intractable, because there is no prior knowledge of what an appropriate $p_t$ and $u_t$ are, and in general no closed-form $u_t$ that generates the desired $p_t$. {#c-fm-intractable}

The way out is to build $p_t$ from simple pieces.
For each data point $\mathbf{x}_1$, take a conditional path $p_t(\mathbf{x} \mid \mathbf{x}_1)$ from the noise to a small blob around $\mathbf{x}_1$, with a field $u_t(\mathbf{x} \mid \mathbf{x}_1)$ that generates it.
Averaging over the data $q$ gives a marginal path and a marginal field:

$$ p_t(\mathbf{x}) = \int p_t(\mathbf{x} \mid \mathbf{x}_1)\, q(\mathbf{x}_1)\, d\mathbf{x}_1, \qquad u_t(\mathbf{x}) = \mathbb{E}\big[u_t(\mathbf{x} \mid \mathbf{x}_1) \,\big|\, \mathbf{x}_t = \mathbf{x}\big] . $$ {#eq-marginal-field}

If every conditional field $u_t(\mathbf{x} \mid \mathbf{x}_1)$ generates its conditional path $p_t(\mathbf{x} \mid \mathbf{x}_1)$, then for any data distribution $q$ the marginal field generates the marginal path. The marginal field is the average of $u_t(\mathbf{x} \mid \mathbf{x}_1)$ weighted by $p_t(\mathbf{x} \mid \mathbf{x}_1)\, q(\mathbf{x}_1) / p_t(\mathbf{x})$; equivalently, it is the conditional mean of the conditional field given that the path passes through $\mathbf{x}$ at time $t$. {#c-marginal}
The derivation `marginal-field` proves it. Its check confirms the continuity equation for a Gaussian mixture, where $u_t$ has a closed form.
The marginal field is still unknown for real data, but the regression does not need it:

$$ \mathcal{L}_\text{CFM}(\theta) = \mathbb{E}_{t, \mathbf{x}_1 \sim q, \mathbf{x} \sim p_t(\cdot \mid \mathbf{x}_1)} \lVert v_\theta(\mathbf{x}, t) - u_t(\mathbf{x} \mid \mathbf{x}_1) \rVert^2 . $$ {#eq-cfm-loss}

The conditional flow matching loss regresses $v_t(\mathbf{x})$ on the conditional field $u_t(\mathbf{x} \mid \mathbf{x}_1)$ at points drawn from $p_t(\mathbf{x} \mid \mathbf{x}_1)$. It differs from the flow matching loss, which regresses on the marginal field $u_t(\mathbf{x})$, by a constant that does not depend on the network. The two losses therefore have the same gradient and the same minimizer. {#c-cfm-grad}
The derivation `cfm-gradient` proves it. This is Theorem 2 of Lipman and colleagues and the reason the method works: a least-squares fit to a target that is noisy but unbiased converges to the conditional mean of that target.

## Components

The notation has vectors as rows. A layer maps $\mathbf{h}$ to $\mathbf{h} W + \mathbf{b}$, with $W$ of shape $(n_\text{in}, n_\text{out})$.
Time runs from noise at $t = 0$ to data at $t = 1$, as in Lipman and colleagues.

**The path.** The optimal transport path of Lipman and colleagues moves noise $\mathbf{x}_0 \sim \mathcal{N}(0, I)$ to the data point $\mathbf{x}_1$ along a straight line:

$$ \mathbf{x}_t = \big(1 - (1 - \sigma_\text{min})t\big)\mathbf{x}_0 + t\,\mathbf{x}_1 . $$ {#eq-ot-path}

At $t = 1$ the path ends in a Gaussian blob of standard deviation $\sigma_\text{min}$ around $\mathbf{x}_1$. With $\sigma_\text{min} = 0$ it is the straight line $(1 - t)\mathbf{x}_0 + t\mathbf{x}_1$ of rectified flow.
Its velocity is the same at every $t$:

$$ \frac{d}{dt}\mathbf{x}_t = \mathbf{x}_1 - (1 - \sigma_\text{min})\mathbf{x}_0 , $$ {#eq-cfm-target}

and this is the regression target of eq-cfm-loss.
Written as a field of the position, the velocity is

$$ u_t(\mathbf{x} \mid \mathbf{x}_1) = \frac{\mathbf{x}_1 - (1 - \sigma_\text{min})\mathbf{x}}{1 - (1 - \sigma_\text{min})t} . $$ {#eq-cond-field}

The Gaussian conditional path with mean $t\mathbf{x}_1$ and standard deviation $1 - (1 - \sigma_\text{min})t$ is the push-forward of the noise by $\mathbf{x}_t = (1 - (1 - \sigma_\text{min})t)\mathbf{x}_0 + t\mathbf{x}_1$. The time derivative of this map, $\mathbf{x}_1 - (1 - \sigma_\text{min})\mathbf{x}_0$, does not depend on $t$. Expressed at the point $\mathbf{x}$, it is the field $u_t(\mathbf{x} \mid \mathbf{x}_1) = (\mathbf{x}_1 - (1 - \sigma_\text{min})\mathbf{x}) / (1 - (1 - \sigma_\text{min})t)$, which is defined for all $t \in [0, 1]$. {#c-cond-ot}
The derivation `conditional-ot-field` proves it.
The conditional flow of the linear path is the optimal transport displacement map between the two Gaussians, and its particles always move in straight-line trajectories at constant speed, whereas the sampling trajectories of diffusion paths can overshoot the final sample. The authors note that although the conditional flow is optimal, this by no means implies that the marginal vector field is an optimal transport solution. {#c-ot-straight}
The diffusion paths of the [[diffusion-model]] fit the same template, as Gaussian conditional paths with other means and standard deviations. Lipman and colleagues note that these paths do not actually reach a true noise distribution in finite time.

**The coupling.** Eq-cfm-loss pairs each data point with an independent noise draw.
Any coupling of $p_0$ and $q$ works, as long as the pairs are drawn from it. Minibatch optimal transport pairs the $B$ noise draws with the $B$ data points of a batch so that the summed squared distance is smallest:

$$ \pi^* = \arg\min_{\pi \in S_B} \sum_{i=1}^{B} \lVert \mathbf{x}_0^{(i)} - \mathbf{x}_1^{(\pi(i))} \rVert^2 . $$ {#eq-ot-coupling}

The minimization is an assignment problem, solved exactly by the [[m2:hungarian-algorithm|Hungarian algorithm]].
Optimal transport conditional flow matching samples the pair $\mathbf{x}_0, \mathbf{x}_1$ jointly from the 2-Wasserstein optimal transport plan instead of independently. As $\sigma$ goes to 0, its marginal field solves the dynamic optimal transport problem. Because optimal transport has cubic time and quadratic memory complexity in the number of samples, the plan is computed on each minibatch. {#c-tong-ot}
Paired by optimal transport, the paths of a batch do not cross: `test_ot_pairs_shorten_and_uncross_the_paths` confirms that no swap of two partners shortens the total.

**The time features.** The network gets $t$ as a vector $\mathbf{e}(t) \in \mathbb{R}^E$ of sines and cosines:

$$ e_{2k}(t) = \sin\big((k + 1)\pi t\big), \qquad e_{2k+1}(t) = \cos\big((k + 1)\pi t\big) . $$ {#eq-time-features}

**The velocity network.** The network of this page is a [[multilayer-perceptron]] with one tanh hidden layer of $H$ units on the concatenation of $\mathbf{x}$ and $\mathbf{e}(t)$:

$$ v_\theta(\mathbf{x}, t) = \tanh\big([\mathbf{x}, \mathbf{e}(t)]\, W_1 + \mathbf{b}_1\big)\, W_2 + \mathbf{b}_2 . $$ {#eq-velocity-model}

$W_2$ and $\mathbf{b}_2$ start at 0, so the untrained network predicts zero velocity everywhere.
For images the network is a U-Net or a [[transformer]], as in diffusion. The flow matching machinery around it does not change.

**The sampler.** A sample solves eq-ode from $\mathbf{x}_0 \sim \mathcal{N}(0, I)$ with a numerical method.
The Euler step is $\mathbf{x} \leftarrow \mathbf{x} + h\, v_\theta(\mathbf{x}, t)$. The midpoint step is $\mathbf{x} \leftarrow \mathbf{x} + h\, v_\theta\big(\mathbf{x} + \tfrac{h}{2} v_\theta(\mathbf{x}, t), t + \tfrac{h}{2}\big)$, which costs two evaluations and has second-order error.
Image models use adaptive solvers such as dopri5, or a fixed number of steps.
The ODE is deterministic: the same noise always gives the same sample, and the map can be run backwards to encode data as noise.

## Shapes and parameters

The shape contract has data of $D$ dimensions, $E$ time features and a hidden layer of $H$ units. The example is $D = 3$, $E = 4$, $H = 5$.
The hidden layer has $W_1$ of shape $(D + E, H)$ and $\mathbf{b}_1$; the output layer has $W_2$ of shape $(H, D)$ and $\mathbf{b}_2$. So

$$ P = (D + E + 1) H + (H + 1) D . $$ {#eq-params}

In the contract this is
{shape:params}
and the example has {shape:params@example} parameters.
The network of the two-mode test, `two_modes`, with $D = 2$, $E = 16$ and $H = 64$, has {shape:params@two_modes}.
The path, the coupling and the time features have no parameters.
The derivation `parameter-and-flop-count` proves the count.

## Computation and cost

One evaluation of the velocity network on $B$ points costs

$$ C = 2BH(2D + E) $$ {#eq-flops}

FLOPs in its two products, a multiply-add counted as two.
The velocity network of this page, one tanh hidden layer of $H$ units on $D$ data coordinates and $E$ time features, has $(D + E + 1)H + (H + 1)D$ parameters and costs $2BH(2D + E)$ FLOPs on $B$ points. An Euler sample of $n$ steps costs $n$ evaluations and a midpoint sample $2n$. {#c-params}
In the contract,
{shape:flops_forward}
which is {shape:flops_forward@example} FLOPs for the example and {shape:flops_forward@two_modes} for `two_modes` at its batch of 256.
A training step evaluates the network once per point, at one random time, and the backward pass costs about twice that. The cost is the same as a diffusion training step.
Minibatch optimal transport adds an $O(B^3)$ assignment per batch. That is small next to an image network for batches of a few hundred, but it grows faster than the network cost as the batch grows.
A sample costs one evaluation per Euler step, run one after the other. Lipman and colleagues counted 142 evaluations of an adaptive solver on CIFAR-10, against 274 for DDPM under the same solver.
The exact log-likelihood of eq-change-of-variables also needs the divergence at every step. That costs $D$ backward passes with automatic differentiation, or one pass with a stochastic trace estimate. For this page's network the divergence has the closed form $\sum_j (1 - h_j^2) \sum_{i \le D} W_1[i, j]\, W_2[j, i]$.

## Training

The training loop is regression:

1. draw data $\mathbf{x}_1 \sim q$, noise $\mathbf{x}_0 \sim \mathcal{N}(0, I)$ and a time $t$;
2. optionally re-pair the batch by eq-ot-coupling;
3. form $\mathbf{x}_t$ by eq-ot-path and the target by eq-cfm-target;
4. take a gradient step on $\lVert v_\theta(\mathbf{x}_t, t) - \text{target} \rVert^2$.

There is no adversary and no encoder, and the ODE is never solved during training.

### Three papers, one loss

Rectified flow learns an ordinary differential equation $dZ_t = v(Z_t, t)\, dt$ whose drift is fitted by least squares to the direction $X_1 - X_0$ of the linear interpolation $X_t = tX_1 + (1 - t)X_0$. Following $X_t$ directly would be non-causal, because the update requires the final point $X_1$; fitting the drift causalizes the paths, so that the flow can be simulated without seeing the future. {#c-rf-def}
Albergo and Vanden-Eijnden infer the velocity of a continuous-time normalizing flow from the probability current of a time-dependent density that interpolates between base and target in finite time. This leads to a simple quadratic loss for the velocity and avoids costly backpropagation through ODE solvers. One instance of the interpolant is $\cos(\pi t/2)\mathbf{x}_0 + \sin(\pi t/2)\mathbf{x}_1$, and they demonstrate the method on images up to $128 \times 128$. {#c-si-def}
Lipman and colleagues write that, concurrently with their work, Liu and colleagues and Albergo and Vanden-Eijnden arrived at similar conditional objectives for simulation-free training of continuous normalizing flows. {#c-fm-concurrent}
Tong and colleagues introduce generalized conditional flow matching, a family of simulation-free training objectives. Unlike diffusion models and earlier training algorithms for continuous normalizing flows, it does not require the source distribution to be Gaussian or require evaluation of its density. Its optimal transport variant creates simpler flows that are more stable to train and lead to faster inference. {#c-tong-def}
When the conditional probability path is a Dirac, with $\sigma = 0$, independent conditional flow matching is equivalent to rectified flow. With the mean $\cos(\pi t/2)\mathbf{x}_0 + \sin(\pi t/2)\mathbf{x}_1$ in place of the linear interpolation, it would be equivalent to the variance-preserving stochastic interpolant of Albergo and Vanden-Eijnden. {#c-tong-equiv}

### What the loss can reach

The CFM loss does not go to zero.
Its minimum is the gap of c-cfm-grad, $\mathbb{E}\lVert u_t(\mathbf{x} \mid \mathbf{x}_1) \rVert^2 - \mathbb{E}\lVert u_t(\mathbf{x}) \rVert^2$: the variance of the target among all the pairs whose paths pass through the same $\mathbf{x}$ at the same $t$.
The gap is large where many pairs cross, which happens near $t = 0$ for an independent coupling, since every noise point heads toward every data point.
Better couplings shrink the gap. Under minibatch optimal transport, `test_ot_coupling_lowers_the_loss` finds the loss settles below half of its value under the independent coupling, because nearby noise goes to nearby data.
Under the independent coupling, the marginal field at $t = 0$ is $u_0(\mathbf{x}) = \mathbb{E}[\mathbf{x}_1] - (1 - \sigma_\text{min})\mathbf{x}$. A single Euler step from $t = 0$ to $t = 1$ therefore sends every noise point $\mathbf{x}$ to $\mathbb{E}[\mathbf{x}_1] + \sigma_\text{min}\mathbf{x}$, which for $\sigma_\text{min} = 0$ is the mean of the data. {#c-one-step}
The derivation `one-step-to-mean` proves it.

### Choosing the times

Which values of $t$ the training draws decides which part of the path the network fits well.
The velocity target is more difficult for $t$ in the middle of $[0, 1]$, since at either end the optimal prediction is the mean of the other distribution. Sampling $t$ from a logit-normal distribution, a normal variable with location $m$ and scale $s$ mapped through the standard logistic function, gives more weight to intermediate timesteps: {#c-sd3-middle}

$$ t = \frac{1}{1 + e^{-(m + s z)}}, \qquad z \sim \mathcal{N}(0, 1) . $$ {#eq-logit-normal}

Training each of 61 different formulations, Esser and colleagues observe that rf/lognorm(0.00, 1.00) consistently achieves a good rank and outperforms a rectified flow formulation with uniform timestep sampling. They also observe that only rectified flow formulations with modified timestep sampling perform better than the eps/linear formulation of LDM used previously. {#c-sd3-results}

### Reflow

Pairing by the learned flow straightens it.
Let $Z$ solve $dZ_t = v(Z_t, t)\, dt$ from $Z_0 = X_0$, with $v(\mathbf{x}, t) = \mathbb{E}[X_1 - X_0 \mid X_t = \mathbf{x}]$ for the linear interpolation $X_t$ of a coupling $(X_0, X_1)$. Then $Z_t$ has the law of $X_t$ at every $t$, and $\mathbb{E}\, c(Z_1 - Z_0) \le \mathbb{E}\, c(X_1 - X_0)$ for every convex cost $c$. {#c-rf-cost}
The derivation `reflow-transport-cost` proves it with Jensen's inequality, applied twice.
The paths of a well-defined ordinary differential equation cannot cross each other, while the paths of the interpolation process may intersect. The rectified flow rewires the individual trajectories that pass through the intersection points so that they do not cross, while tracing out the same density map as the linear interpolation. {#c-rf-noncross}
Applying the rectification recursively, which is called reflow, decreases the transport cost and straightens the paths of the flow; perfectly straight paths can be simulated exactly with a single Euler step. {#c-rf-reflow}
Reflow costs one full simulation of the ODE for every training pair. Each round also trains on the previous model's samples rather than on the data, so its errors carry forward.

## Inductive biases

The flow is a deterministic, invertible map from noise to data.
Every sample has a noise code, and nearby codes give nearby samples. The map can be run backwards to recover the code of a data point and to compute its exact likelihood by eq-change-of-variables.
The straight conditional paths make the target simple: a constant velocity per pair, with no schedule of noise levels to tune.
The learned field is the average of straight lines, and it bends where lines from different pairs cross, so its paths are straighter when fewer pairs cross. The choice of coupling and reflow both act on this.
Unlike a [[variational-autoencoder]], which samples with one pass of its decoder and only bounds the likelihood, a flow pays for each sample with many evaluations, but it gives the exact likelihood and needs no encoder.
Unlike a [[generative-adversarial-network]], which also maps noise to data in one pass, a flow is trained by a plain regression with no adversary, so its training does not oscillate or drop modes in the way adversarial training can.
A single network serves every $t$, so it learns how the velocity changes along the path. The distribution of $t$ in training decides where it learns best.

## Evidence

Lipman and colleagues compared paths with the same U-Net architecture.
With the same U-Net architecture, Flow Matching with the optimal transport path reached a negative log-likelihood of 2.99 bits per dimension, an FID of 6.35 and 142 function evaluations of the adaptive solver on CIFAR-10, against 3.12, 7.48 and 274 for DDPM, and an FID of 20.9 on ImageNet 128 x 128. {#c-fm-results}
For ImageNet-128, where Dhariwal and Nichol train for 4.36m iterations with batch size 256, the Flow Matching model, 25% larger, used 500k iterations with batch size 1.5k, which is 33% less image throughput. {#c-fm-train-cost}
With one step of reflow the flow becomes nearly straight and gives good results with a single Euler discretization step. For one-step fast diffusion and flow models on CIFAR10, Liu and colleagues report an FID of 4.85 and a recall of 0.51. {#c-rf-results}
Esser and colleagues improve noise sampling techniques for training rectified flow models by biasing them towards perceptually relevant scales, and train an 8B rectified flow model for text-to-image synthesis. {#c-sd3-def}

## Limitations

Sampling still solves an ODE.
Straight conditional paths do not make the marginal paths straight. A flow trained with the independent coupling needs tens of steps: one Euler step collapses every sample onto the mean, by c-one-step.
Reflow and distillation buy fewer steps with extra training. Minibatch optimal transport buys straighter paths at an $O(B^3)$ cost per batch, and its plan is only optimal within the batch.
The marginal field of the optimal transport path is not the optimal transport map between noise and data, as Lipman and colleagues caution in c-ot-straight. Reflow lowers every convex transport cost but has no guarantee of reaching the optimum in more than one dimension.
The exact likelihood is available but expensive, since it needs a full ODE solve with a divergence at each step.
The method fits a deterministic flow, so all the randomness of a sample comes from its noise. A field that is wrong in a region where samples pass sends them off course, with no noise injection to correct them as stochastic samplers do.

## Variants and descendants

- **Rectified flow** (Liu, Gong and Liu): the straight path with $\sigma_\text{min} = 0$, plus reflow.
- **Stochastic interpolants** (Albergo and Vanden-Eijnden): general interpolants, such as the trigonometric one of c-si-def. Later work adds noise to the interpolant.
- **I-CFM and OT-CFM** (Tong and colleagues): independent and minibatch optimal transport couplings, from any source distribution.
- **SB-CFM.** Schrödinger bridge conditional flow matching, an entropic variant of OT-CFM, pairs the samples by the solution of the entropy-regularized optimal transport problem. The flow still matches the marginals for any choice of the regularization $\varepsilon$; OT-CFM is recovered as $\varepsilon$ goes to 0 and I-CFM as $\varepsilon$ goes to infinity. {#c-tong-sb}
- **Diffusion paths.** The variance-exploding and variance-preserving paths of the [[diffusion-model]] are Gaussian conditional paths too. Training on them with eq-cfm-loss gives a probability flow ODE of diffusion.
- **Latent flow matching.** Like latent diffusion, it runs the flow in the latent space of an [[autoencoder]].

## Notable models

Rectified flows define the forward process as straight paths between the data distribution and a standard normal distribution, $\mathbf{z}_t = (1 - t)\mathbf{x}_0 + t\boldsymbol\epsilon$, with the data at $t = 0$, and the network output directly parameterizes the velocity. {#c-sd3-convention}
Esser and colleagues used this convention for the 8B transformer of c-sd3-def, trained on latents with logit-normal times. Note that their $t$ runs from data to noise, the reverse of this page.

## Implementations

The NumPy reference `impl/numpy/flow_matching.py` implements:

- eq-ot-path, eq-cfm-target and eq-cond-field;
- the network of eq-velocity-model with eq-time-features, with a hand-written backward pass of eq-cfm-loss;
- the Euler and midpoint samplers of eq-ode;
- the network's exact divergence and the log-likelihood of eq-change-of-variables;
- the Hungarian algorithm and eq-ot-coupling;
- reflow pairs and eq-logit-normal.

For data that are a mixture of isotropic Gaussians, the marginal field eq-marginal-field has a closed form, `mixture_field`. The tests drive the samplers with it, which separates the error of the sampler from the error of the network.

The PyTorch version `impl/pytorch/flow_matching.py` is a `VelocityModel` module with the same parameters and autograd for the backward pass. `test_gradcheck` runs `torch.autograd.gradcheck` through it, and `divergence` takes the trace of the Jacobian with one backward pass per coordinate.

The JAX version `impl/jax/flow_matching.py` is a set of pure functions of a parameter dictionary. The divergence is the trace of `jax.jacfwd`, the samplers and the log-likelihood are `jax.lax.fori_loop`s, and the Adam step is jitted.

All three load the shared fixture of `fixtures/make.py`, which fixes the parameters, the data, the noise and the times. It checks the time features, the path, the targets, the predictions, the loss and its gradients, the divergence, and an Euler and a midpoint sample.

### Worked example

Move noise $\mathcal{N}(0, 1)$ to data $\mathcal{N}(2, 1)$ in one dimension with $\sigma_\text{min} = 0$ and the independent coupling.
By eq-marginal-field the field is $u_t(x) = 2 + \kappa_t(x - 2t)$, where $\kappa_t = (2t - 1)/((1 - t)^2 + t^2)$.
For noise $N(0, 1)$ and data $N(2, 1)$ in one dimension with $\sigma_\text{min} = 0$, the 1-rectified flow is $x_t = 2t + x_0\sqrt{(1 - t)^2 + t^2}$. It ends at $x_0 + 2$, which is the translation and the optimal transport map, but it bends on the way. Euler's method with 1, 2 and 4 steps maps two points one apart at $t = 0$ to points 0, 0.5 and 0.72 apart, while one midpoint step is exact. {#c-translation}
The derivation `translation-flow` proves it.
For the two points $x_0 = 0$ and $x_0 = 1$:

- At $t = 1/2$ they sit at $1$ and $1 + \sqrt{0.5} = 1.7071$, closer together than at either end.
- One Euler step uses $u_0(x) = 2 - x$ and sends both to 2.
- Two Euler steps go through $1$ and $1.5$, then add the constant field $u_{1/2} = 2$, ending at $2$ and $2.5$.
- One midpoint step evaluates $u_{1/2} = 2$ at the midpoint and lands on $2$ and $3$, the exact answer.

`test_worked_example` checks these numbers.

### Pitfalls

> [!pitfall]
> The direction of time differs between papers. Lipman and colleagues and this page put the noise at $t = 0$, while Esser and colleagues put the data at $t = 0$ and the noise at $t = 1$. Under the second convention, the velocity target is $\boldsymbol\epsilon - \mathbf{x}_0$, the negative of eq-cfm-target. Mixing the conventions flips the sign of the field, and the sampler walks away from the data.

> [!pitfall]
> With $\sigma_\text{min} = 0$, eq-cond-field divides by $1 - t$, which is zero at $t = 1$. Training uses the target of eq-cfm-target, which has no division, and draws $t$ from $[0, 1)$ or from the open interval of eq-logit-normal.

> [!pitfall]
> The CFM loss is not a measure of fit. Its floor is the variance of the target given $(\mathbf{x}_t, t)$, which depends on the data and the coupling, so a loss of 1.5 may be perfect for one dataset and poor for another. Compare samples, or compare against the loss of the exact field where it is known, as `test_conditional_and_marginal_losses_have_the_same_gradient` does.

> [!pitfall]
> Few sampling steps fail in a specific way. By c-one-step, a single Euler step from the exact field of an independent coupling returns the data mean for every noise draw, and a handful of steps blurs the modes together. Use a midpoint or higher-order solver, straighten the flow with reflow or optimal transport, or distill before cutting the step count.

> [!pitfall]
> Minibatch optimal transport is only as good as the batch. With a small batch the plan is close to a random pairing, and the gain over the independent coupling disappears. With conditional generation, pairs must also be matched within the same condition.

> [!pitfall]
> A likelihood computed with a few fixed solver steps, or with a stochastic trace estimate, is an estimate and not a bound. Report the solver and its tolerance with any number in bits per dimension.

## Experiments

This page has no experiment yet.
Some tests are small experiments.

`test_learns_two_modes` trains the NumPy model, with $H = 64$ and $E = 16$, for 1500 Adam steps on two Gaussian modes in the plane at $(\pm 2, 0)$ with standard deviation 0.3. Its loss falls below 0.8 of its starting value. Of 2000 samples from 20 midpoint steps, between 40% and 60% fall on each side, and more than 90% lie within 1 of a mode. The PyTorch and JAX versions run the same test, PyTorch with logit-normal times.

`test_ot_coupling_lowers_the_loss` trains the same model twice for 200 steps at batch 32. With minibatch optimal transport, the loss settles below half of its value under the independent coupling.

`test_reflow_lowers_the_transport_cost` runs the exact field of the two modes and finds that the rectified coupling has a lower squared and absolute displacement than the independent one, while its paths still bend.

The checks of the eight derivations compare:

- the path with its Gaussian marginal;
- the marginal field with a Monte Carlo conditional mean and the continuity equation;
- the CFM and FM gradients for two networks;
- the one-step map with the mean;
- the translation flow with its closed form and its Euler gaps;
- the log-likelihood from eq-change-of-variables with the exact mixture density;
- the rectified coupling with the independent one under three convex costs;
- the parameter and FLOP counts with the code.

## History

Chen, Rubanova, Bettencourt and Duvenaud introduced neural ordinary differential equations and continuous normalizing flows at NeurIPS 2018. These models were trained by maximum likelihood through an ODE solver.
In September and October 2022, three groups posted the simulation-free alternative within weeks of each other:

- Liu, Gong and Liu, as rectified flow, arXiv 2209.03003;
- Albergo and Vanden-Eijnden, as stochastic interpolants, arXiv 2209.15571, published at ICLR 2023;
- Lipman, Chen, Ben-Hamu, Nickel and Le, as flow matching, arXiv 2210.02747.

Lipman and colleagues framed the method as a generalization of diffusion training to arbitrary probability paths. Liu and colleagues framed it as transport, with reflow to straighten the paths.
Tong and colleagues unified the variants under conditional flow matching and added minibatch optimal transport and Schrödinger bridge couplings (arXiv 2302.00482, TMLR 2024).
In 2024 Esser and colleagues trained Stable Diffusion 3 as a rectified flow transformer, and flow matching became a standard objective for large image, video and audio generators.

## References

- R. T. Q. Chen, Y. Rubanova, J. Bettencourt and D. Duvenaud. Neural Ordinary Differential Equations. NeurIPS 2018.
- X. Liu, C. Gong and Q. Liu. Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow. arXiv 2209.03003, 2022.
- M. S. Albergo and E. Vanden-Eijnden. Building Normalizing Flows with Stochastic Interpolants. ICLR 2023.
- Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel and M. Le. Flow Matching for Generative Modeling. arXiv 2210.02747, 2022.
- A. Tong, K. Fatras, N. Malkin, G. Huguet, Y. Zhang, J. Rector-Brooks, G. Wolf and Y. Bengio. Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport. TMLR, 2024.
- P. Esser and colleagues. Scaling Rectified Flow Transformers for High-Resolution Image Synthesis. arXiv 2403.03206, 2024.
