---
tag: WNCNQE
kind: architecture
slug: neural-ordinary-differential-equation
name: Neural ordinary differential equation
aliases: [neural ODE, NODE, ODE-Net, ODE network, continuous-depth network, continuous-depth residual network]
area: deep-learning-architectures/neural-odes-and-implicit-models
level: core
summary: >-
  A network whose hidden state follows an ODE with a neural network as its vector field, computed by a solver and trained by the adjoint method; the continuous-depth limit of a residual network.
uses: [backpropagation@al, matrix-multiplication@m2, stochastic-gradient-descent@m2]
related_to: [resnet, structured-state-space-model, exploding-gradient-problem@al, vanishing-gradient-problem@al]
history:
  year: 2018
  people: [Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud]
  note: "Residual networks were read as Euler discretizations of ODEs by Haber and Ruthotto and by Lu et al. in 2017; Chen et al. posted Neural Ordinary Differential Equations as arXiv 1806.07366 in June 2018, and it appeared at NeurIPS 2018. The adjoint method they use goes back to Pontryagin et al. (1962)."
prereqs: [resnet, backpropagation@al, matrix-multiplication@m2]
same_as:
  arxiv: ["1806.07366"]
sources:
  - {paper: "Chen, Rubanova, Bettencourt and Duvenaud (2018)", title: "Neural Ordinary Differential Equations", locator: "arXiv 1806.07366v5, the abstract and Sections 1 to 7, Tables 1 and 2", note: "the model, the adjoint method, continuous normalizing flows and latent ODEs"}
  - {paper: "Dupont, Doucet and Teh (2019)", title: "Augmented Neural ODEs", locator: "arXiv 1904.01681v3, Sections 3 to 5, Table 1", note: "what a flow cannot represent, and augmentation"}
  - {paper: "Haber and Ruthotto (2017)", title: "Stable Architectures for Deep Neural Networks", locator: "arXiv 1705.03341v3, Sections 2, 3 and 4.1", note: "ResNets as discretized ODEs, and the stability of the forward Euler method"}
  - {paper: "Lu, Zhong, Li and Dong (2018)", title: "Beyond Finite Layer Neural Networks", locator: "arXiv 1710.10121v3, the abstract"}
  - {paper: "Gholami, Keutzer and Biros (2019)", title: "ANODE: Unconditionally Accurate Memory-Efficient Gradients for Neural ODEs", locator: "arXiv 1902.10298v3, the abstract and Section I", note: "the instability of the backward solve and the gradient of the discrete forward pass"}
  - {paper: "Rubanova, Chen and Duvenaud (2019)", title: "Latent ODEs for Irregularly-Sampled Time Series", locator: "arXiv 1907.03907v1, the abstract"}
  - {paper: "Finlay, Jacobsen, Nurbekyan and Oberman (2020)", title: "How to Train Your Neural ODE", locator: "arXiv 2002.02798v3, the abstract"}
  - {paper: "Kidger (2022)", title: "On Neural Differential Equations", locator: "arXiv 2202.02435v1, the abstract", note: "a survey of the field"}
family: continuous-depth
inputs: "a batch of feature vectors as a (B, d) array, padded with zeros to the state width D"
outputs: "the state at the end time, a (B, D) array, read by a task head"
components: []
inductive_biases: ["continuity in depth: the state changes smoothly, so the map from input to output is a homeomorphism and trajectories never cross", "weight sharing across depth: one vector field, optionally depending on t, is used at every step", "the cost of evaluation adapts to the input when an adaptive solver is used"]
params: "2 D H + 2 H + D for the one-hidden-layer field of width H, whatever the number of solver steps"
flops_per_example: "4 S N D H for N steps of a solver with S evaluations per step, counting a multiply-add as 2 FLOPs and leaving out biases and activations"
---

A neural ordinary differential equation, or neural ODE, replaces a stack of layers by a differential equation: the hidden state $h(t)$ changes continuously in time $t$, its rate of change is given by a neural network $f$, and the output is the state at an end time $T$.
A numerical ODE solver computes that state, so the number of evaluations of $f$, the analogue of depth, is chosen by the solver rather than fixed in the architecture.
The gradient is computed by solving a second ODE backwards in time, the adjoint method, which needs memory independent of the number of solver steps.
Chen, Rubanova, Bettencourt and Duvenaud introduced the model in 2018, together with continuous normalizing flows and latent ODEs for irregularly sampled time series.

## In plain words

A residual network updates its features a little at each layer: the new features are the old ones plus a correction.
Imagine making the corrections smaller and the layers more numerous, without end: the features then flow smoothly, like a particle pushed along by a wind whose direction depends on where the particle is.
A neural ODE learns that wind.
To get the output, a solver follows the particle from where the input puts it to where it is at the end time, taking as many small steps as it needs to be accurate.
Because the particle moves continuously, two particles never pass through the same point at the same time; that is both what makes the model well behaved and what limits what it can learn.

## Overview

A [[resnet|residual network]] computes $h_{k+1} = h_k + f(h_k, \theta_k)$, which is one step of Euler's method for an ODE with step 1.
Chen et al. write that these iterative updates can be seen as an Euler discretization of a continuous transformation, and ask what happens as more layers and smaller steps are taken. {#c-chen-euler}
Lu et al. argue that ResNet, PolyNet, FractalNet and RevNet can be interpreted as different numerical discretizations of differential equations. {#c-lu-discretizations}
In the limit the network parameterizes the derivative of the hidden state:

$$ \frac{dh(t)}{dt} = f(h(t), t, \theta), \qquad h(0) = x, \qquad y = h(T) . $$ {#eq-ode}

Chen et al. define the model in these terms: instead of a discrete sequence of hidden layers, a neural network parameterizes the derivative of the hidden state, and the output is computed with a black-box differential equation solver. {#c-def}
They list three properties of such continuous-depth models: constant memory cost, an evaluation strategy that adapts to each input, and an explicit trade of numerical precision for speed. {#c-properties}
The rest of this page derives each of them, shows what the continuity costs in expressiveness, and states what the experiments of the original papers measured.

## Components

### The vector field

Any network that maps $(h, t)$ to a vector of the width of $h$ can be the field. The reference implementations use one hidden layer of width $H$ with the time as an extra input:

$$ f(h, t) = W_2 \tanh(W_1 h + t\, u + b_1) + b_2 , $$ {#eq-field}

with $W_1$ of shape $H \times D$, $u$ and $b_1$ of length $H$, $W_2$ of shape $D \times H$ and $b_2$ of length $D$.
The same parameters are used at every time, so the model shares its weights across depth as a recurrent network shares them across time.
For image classification Chen et al. use convolutions in the field, with the time appended as a channel.

The solution exists and is unique when the field is well behaved.
Chen et al. cite Picard's existence theorem: the solution to an initial value problem exists and is unique if the differential equation is uniformly Lipschitz continuous in $z$ and continuous in $t$, which holds for their model if the network has finite weights and uses Lipschitz nonlinearities such as tanh or relu. {#c-picard}

### Augmentation

The state can be wider than the input. Dupont, Doucet and Teh concatenate every data point with a vector of $p$ zeros and solve the ODE on the augmented space: {#c-anode-def}

$$ h(0) = \begin{pmatrix} x \\ \mathbf{0}_p \end{pmatrix} \in \mathbb{R}^{d + p}, \qquad D = d + p . $$ {#eq-augment}

With $p = 0$ this is the plain model. Augmentation is what lets the flow represent maps that the plain model cannot (Inductive biases below).

### The solver

The fixed-step solvers take $N$ steps of size $\Delta t = T / N$. Euler's method,

$$ h_{k+1} = h_k + \Delta t\, f(h_k, t_k) , $$ {#eq-euler}

evaluates $f$ once per step and is exactly a residual block with the residual branch $\Delta t\, f$. {#c-euler-resnet}
The classical Runge-Kutta method of order 4 evaluates it four times:

$$ \begin{aligned} k_1 &= f(h_k, t_k), & k_2 &= f(h_k + \tfrac{\Delta t}{2} k_1, t_k + \tfrac{\Delta t}{2}), \\ k_3 &= f(h_k + \tfrac{\Delta t}{2} k_2, t_k + \tfrac{\Delta t}{2}), & k_4 &= f(h_k + \Delta t\, k_3, t_k + \Delta t), \\ h_{k+1} &= h_k + \tfrac{\Delta t}{6} (k_1 + 2 k_2 + 2 k_3 + k_4) . \end{aligned} $$ {#eq-rk4}

Its error at $T$ shrinks as $\Delta t^4$, against $\Delta t$ for Euler: halving the step divides the error by about 16 instead of 2. {#c-euler-order}
The derivation `euler-resnet` proves both statements for a linear field and its check measures the orders.
An adaptive solver such as the Dormand-Prince pair of orders 5 and 4 estimates its own error at each step and shrinks or grows the step to keep it under a tolerance, so the number of evaluations depends on the input and on the field.
Chen et al. note that there have been more than 120 years of development of efficient and accurate ODE solvers since Euler's method, starting with Runge (1895) and Kutta (1901). {#c-solvers-history}

## Shapes and parameters

The field eq-field has

$$ P = 2 D H + 2 H + D $$ {#eq-params}

parameters, whatever the number of solver steps. {#c-params}
In the shape contract the formula is

{shape:params}

and the example model, an input of width 2 augmented to 3 with a hidden width of 4, has {shape:params@example} parameters; the plain model of the same input and hidden width has {shape:params@plain}.
A residual network of $N$ blocks of the same shape would have $N$ times as many, which is the parameter saving of the continuous model: Chen et al. report an ODE-Net on MNIST with 0.22 M parameters against 0.60 M for a ResNet with the same error (Evidence below).

## Computation and cost

Each evaluation of eq-field over a batch of $B$ makes two matrix products of $B D H$ multiply-adds, so a fixed-step solve with $S$ evaluations per step ($S = 1$ for Euler, $S = 4$ for RK4) costs

$$ C_\text{fwd} = 4\, S N B D H $$ {#eq-flops}

FLOPs, counting a multiply-add as two and leaving out the biases and the activation. {#c-flops}
The example model takes 3 RK4 steps, and its forward pass over a batch of 5 costs {shape:flops_forward@example} FLOPs.
The number of function evaluations, NFE, plays the part of depth.
With an adaptive solver it is not fixed: Chen et al. find that the time spent by the forward call is proportional to the number of function evaluations, so tuning the tolerance trades accuracy against computational cost {#c-tolerance}, and that the number of function evaluations increases throughout training. {#c-nfe-training}
They report that the default tolerance of 1.5e-8 was used for sequence modelling, and that in classification and density estimation they could reduce it to 1e-3 and 1e-5 respectively without degrading performance. {#c-tolerances}
Minibatches are solved as one combined system of all the examples; Chen et al. note that this might in some cases need more evaluations than solving each system alone, but that in practice the number of evaluations did not increase substantially. {#c-minibatch}

## Training

The model is trained by gradient descent on a loss $L(h(T))$. There are two ways to get the gradient.

**Discretize, then optimize.** Fix the solver and its steps, and backpropagate through every operation of the forward pass, as for any network. This gives the exact gradient of the computed output, but it stores the input of every stage of every step: $S N B D$ numbers.

**Optimize, then discretize.** Derive the gradient of the exact solution as an ODE of its own and solve that ODE numerically. This is the adjoint method.

### The adjoint method

Write $a(t) = \partial L / \partial h(t)$ for the adjoint, a row vector. It follows

$$ \frac{da(t)}{dt} = -a(t)\, \frac{\partial f(h(t), t, \theta)}{\partial h}, \qquad a(T) = \frac{\partial L}{\partial h(T)} , $$ {#eq-adjoint}

backwards in time, with $a(0) = \partial L / \partial x$. {#c-adjoint}
The gradient of the parameters is an integral along the trajectory,

$$ \frac{\partial L}{\partial \theta} = \int_0^T a(t)\, \frac{\partial f(h(t), t, \theta)}{\partial \theta}\, dt . $$ {#eq-adjoint-theta}

{#c-adjoint-theta}
The derivation `adjoint` proves both from the chain rule over a step of length $\varepsilon \to 0$, and its check compares the result with finite differences.
The right-hand sides are vector-Jacobian products, which reverse-mode differentiation of $f$ computes at the cost of a few evaluations of $f$; Chen et al. compute all the integrals in a single call to an ODE solver, on a state that concatenates $h$, $a$ and the running gradient (their Algorithm 1). {#c-algorithm-1}
$h(t)$ is recomputed by solving eq-ode backwards from $h(T)$ next to $a$, so nothing of the forward pass is stored: the state of the backward solve is $2 B D + P$ numbers for any number of steps. {#c-adjoint-memory}
Chen et al. use the adjoint sensitivity method of Pontryagin et al. (1962), which they describe as scaling linearly with problem size, with low memory cost and explicit control of numerical error. {#c-pontryagin}
They contrast it with forward sensitivity analysis, which is quadratic-time in the number of variables, whereas the adjoint sensitivity analysis is linear. {#c-forward-sensitivity}

One evaluation of the adjoint system computes $f$ and its two vector-Jacobian products from one shared pre-activation, six matrix products against two for $f$ alone, so the backward solve costs three times the forward solve when both take the same steps. {#c-backward-cost}
Backpropagation through RK4 instead keeps $4 N B D$ numbers. {#c-memory}
The derivation `parameter-and-flop-count` counts both passes and both memories.
Chen et al. report that on a trained ODE-Net the number of evaluations in the backward pass is roughly half that of the forward pass. {#c-backward-nfe}

### What the adjoint gets wrong

The two gradients are not the same. The adjoint method computes the gradient of the exact solution and then makes the error of the solver; backpropagation computes the exact gradient of the solver's output.
They agree as the steps shrink, at the rate of the solver; the test `test_adjoint_matches_backprop_as_the_steps_shrink` measures the gap.
Chen et al. note that reconstructing the trajectory by running the dynamics backwards can introduce extra numerical error if it diverges from the original, which checkpointing addresses, though they did not find it a practical problem. {#c-reconstruction}
Gholami, Keutzer and Biros argue that it can be a problem: they report that solving the forward ODE backwards may be numerically unstable for ReLU and non-ReLU activations and general convolution operators, and that the optimize-then-discretize approach may lead to divergent training through inconsistent gradients for small time steps. {#c-gholami-unstable}
Their ANODE checkpoints the forward pass and differentiates the discrete steps, with a memory footprint of $O(L) + O(N_t)$ for $L$ ODE blocks of $N_t$ steps. {#c-gholami-memory}

## Inductive biases

The continuity that gives the model its constant memory also limits what it can represent.
In one dimension trajectories cannot cross, so the flow map is strictly increasing: $x_1 < x_2$ gives $h_1(T) < h_2(T)$. {#c-flow-order}
Dupont et al. take the function $g(-1) = 1$, $g(1) = -1$ and prove that the flow of an ODE cannot represent it. {#c-dupont-prop1}
On these two points every flow has a loss above $\frac12$ under half the mean squared error, the value of mapping both to 0. {#c-g1d}
Dupont et al. observe exactly that in training: since the trajectories cannot cross, the model maps all input points to zero to minimize the mean squared error. {#c-dupont-zero}
A single residual step $h + F(h)$ with $F(h) = -2h$ represents $g$, because a discrete step can jump over another trajectory. {#c-residual-g1d}
Dupont et al. read ResNets in this light, as ODE solutions with large errors that let them represent more functions. {#c-dupont-resnet}
One extra dimension is enough: the rotation by $\pi$ in the augmented plane maps $(x, 0)$ to $(-x, 0)$, so the augmented model represents $g$ on its first coordinate. {#c-augment-g1d}
The derivation `one-dimensional-flow` proves all four statements.
In higher dimensions the same argument applies to regions. Dupont et al. prove that neural ODEs cannot represent the function that maps all points inside a ball to $-1$ and all points in a surrounding annulus to $1$. {#c-dupont-prop2}
More generally they prove that the feature map of a neural ODE is a homeomorphism, so its features preserve the topology of the input space: the flow can only continuously deform the input space and cannot tear a connected region apart. {#c-homeomorphism}

The second bias is stability. Haber and Ruthotto study the linearized forward propagation $\dot y = K y$ and argue that stable forward propagation and well-posed learning problems are obtained when the real parts of the eigenvalues of the Jacobian are close to zero. {#c-haber-re0}
Antisymmetric $K$ has purely imaginary eigenvalues and preserves the norm exactly, but forward Euler multiplies the component along an eigenvalue $i\omega$ by $\sqrt{1 + \Delta t^2 \omega^2} > 1$ at every step. {#c-antisymmetric-unstable}
Euler steps do not grow along an eigenvalue $\lambda$ if and only if $\lvert 1 + \Delta t \lambda \rvert \le 1$. {#c-euler-stable}
Shifting to $K - \gamma I$ with $\gamma > 0$ makes Euler steps non-growing exactly when $\Delta t (\gamma^2 + \omega^2) \le 2\gamma$. {#c-diffusion}
Eigenvalues with a positive real part make the forward propagation and its gradient grow along the depth, the continuous form of the [[al:exploding-gradient-problem|exploding gradient problem]], and eigenvalues with a negative real part make them decay, the form of the [[al:vanishing-gradient-problem|vanishing gradient problem]]; a discrete network suffers both through products of layer Jacobians, the ODE through the Jacobian of its flow map.
Haber and Ruthotto propose antisymmetric weights, $\frac12 (K - K^\top) - \gamma I$, with the diffusion $\gamma \ge 0$ added to stabilize the discrete forward propagation based on the forward Euler method. {#c-haber-antisym}
The derivation `euler-stability` proves the three bounds.

## Evidence

On MNIST, Chen et al. report a test error of 0.42% for an ODE-Net with 0.22 M parameters and $O(1)$ memory, against 0.41% for a ResNet with 0.60 M parameters and $O(L)$ memory, 0.47% for the same network backpropagated through a Runge-Kutta integrator (RK-Net) and 1.60% for a one-layer MLP. {#c-mnist}
On a dataset of noisy spirals sampled at irregular times, their latent ODE has a predictive RMSE of 0.1642, 0.1502 and 0.1346 with 30, 50 and 100 observed points, against 0.3937, 0.3202 and 0.1813 for an RNN. {#c-latent-ode}
Dupont et al. report test accuracies of 98.2%, 60.6% and 83.5% for augmented neural ODEs on MNIST, CIFAR10 and SVHN, against 96.4%, 53.7% and 81.0% for plain neural ODEs, over 5 runs. {#c-anode-accuracy}
Trained on the ball and annulus function of Inductive biases, augmented models need a number of evaluations that hardly increases during training, while it nearly doubles for plain ones. {#c-anode-nfe}
They also find that the NFE of plain models could often become prohibitively large, in excess of 1000, roughly a 1000-layer ResNet. {#c-nfe-1000}
On 200 classes of 64 × 64 ImageNet, augmented models achieve lower losses and train almost 10 times faster. {#c-anode-imagenet}
Rubanova et al. report that ODE-RNNs and latent ODEs outperform their RNN-based counterparts on irregularly sampled data. {#c-rubanova}
Finlay et al. report that optimal transport and stability regularization lead to simpler dynamics, fewer discretizations of the solver and considerably less wall-clock time without loss in performance. {#c-finlay}
These are reported results; this page has not reproduced them.

## Limitations

- **Cost.** A solve takes as many evaluations as the solver needs, and an adaptive solver may need many more as training proceeds; the evidence above includes NFE in excess of 1000. Training is slower than for a residual network of the same accuracy, and the cost per example is not known in advance.
- **Expressiveness.** The plain model is a homeomorphism of its input space, so it cannot represent maps that reverse order in one dimension or tear regions apart in more; augmentation or a final linear layer is needed.
- **Gradient accuracy.** The adjoint gradient is the gradient of the exact solution, not of the computed one, and the backward reconstruction of $h$ can be unstable for some fields; the gap closes only as the solver's error does.
- **Tolerance as a hyperparameter.** The forward and backward tolerances trade accuracy against cost, must be set by the user, and interact with training.
- **Minibatches.** Solving a batch as one system makes all examples take the steps of the hardest one.

## Variants and descendants

### Continuous normalizing flows

A normalizing flow transforms a simple density by an invertible map and tracks the density with the change of variables formula, which needs $\log \lvert \det \partial F / \partial z \rvert$, a cubic cost in general.
For a flow defined by an ODE the log density follows an ODE of its own:

$$ \frac{\partial \log p(h(t))}{\partial t} = -\operatorname{tr}\!\left( \frac{\partial f}{\partial h(t)} \right) . $$ {#eq-cnf}

This is the instantaneous change of variables of Chen et al., stated for $f$ uniformly Lipschitz continuous in $h$ and continuous in $t$. {#c-cnf}
The derivation `instantaneous-change-of-variables` proves it from Jacobi's formula, and its check compares the integrated trace with the log determinant of the Jacobian of the flow map.
Chen et al. point out that, unlike finite flows, the field does not need to be bijective, since if uniqueness is satisfied the whole transformation is automatically bijective. {#c-cnf-bijective}
A trace is cheaper than a determinant. For eq-field, $\operatorname{tr}(\partial f / \partial h) = \sum_j (1 - s_j^2) (W_1 W_2)_{jj}$ with $s = \tanh(W_1 h + t u + b_1)$, which costs $O(H)$ per example once the diagonal of $W_1 W_2$ is known. {#c-trace-cost}
Chen et al. make the same point for a continuous planar flow, whose cost is linear in the number of hidden units $M$, against $O(M^3)$ for a standard normalizing flow, which is why those use many layers of a single hidden unit. {#c-cnf-linear}
They prove that a planar CNF with $M$ hidden units can be at least as expressive as a planar NF with $K = M$ layers. {#c-cnf-planar}
FFJORD replaces the exact trace with an unbiased stochastic estimate to scale to high dimensions.

### Other variants

- **Augmented neural ODEs** widen the state with zeros (eq-augment).
- **Latent ODEs and ODE-RNNs** model irregularly sampled time series with an ODE in a latent space.
- **Regularized neural ODEs** penalize the kinetic energy or the Jacobian of the field to make the dynamics easier to solve.
- **Stable architectures** constrain the field, with antisymmetric or Hamiltonian structure, so that the forward propagation neither explodes nor vanishes.
- **Neural controlled, stochastic and rough differential equations** extend the idea to driven and noisy dynamics; Kidger's thesis surveys them and notes that residual and recurrent networks are discretisations of such equations. {#c-kidger}
- **[[structured-state-space-model|Structured state space models]]** use linear ODEs with a structured state matrix, discretized exactly, as sequence layers.

## Notable models

The ODE-Net of Chen et al. for MNIST, the continuous normalizing flows of the same paper and FFJORD, and the latent ODE of Rubanova et al. are the standard examples. Neural ODE layers are used in physics-informed and hybrid mechanistic models more than in large-scale vision or language models.

## Implementations

The NumPy reference `impl/numpy/neural_ode.py` implements eq-field, eq-augment, the Euler, RK4 and adaptive Dormand-Prince solvers (eq-euler, eq-rk4), the forward pass of eq-ode, the adjoint backward pass of eq-adjoint and eq-adjoint-theta, backpropagation through the solver's steps, the trace and the change of log density of eq-cnf, and the counts eq-params and eq-flops.
The PyTorch version `impl/pytorch/neural_ode.py` is an `nn.Module` whose forward pass is differentiated by autograd through the steps, with an `AdjointODE` autograd function whose backward pass solves the adjoint system instead.
The JAX version `impl/jax/neural_ode.py` is a pure function, with a `jax.custom_vjp` for the adjoint backward pass, tested under `jax.jit` and `jax.vmap`.
All three are tested against the shared fixture of `fixtures/`: the example config with its output, loss and the gradients of the discrete RK4 map.

### Worked example

Take the plain one-dimensional model, $d = D = 1$, and the two points $x = -1$ and $x = 1$ with the targets $g(-1) = 1$ and $g(1) = -1$.
Whatever the field, $h(T)$ for $x = -1$ stays below $h(T)$ for $x = 1$, because the trajectories would have to meet to change order and two solutions that meet are the same solution.
With $u = h(T)$ at $-1$ and $v = h(T)$ at $1$, the loss is $\frac14 ((u - 1)^2 + (v + 1)^2)$, and on $u \le v$ its minimum is $\frac12$, at $u = v = 0$: the model can do no better than to collapse both points to the origin.
Add one zero coordinate and let the field be the rotation $f(h) = \pi \left(\begin{smallmatrix} 0 & -1 \\ 1 & 0 \end{smallmatrix}\right) h$ for $T = 1$. The point $(-1, 0)$ travels half a circle to $(1, 0)$ and $(1, 0)$ travels to $(-1, 0)$; the two trajectories pass each other through the second dimension instead of crossing, and the loss is 0.
The tests `test_one_dimensional_flow_cannot_fit_g1d` and `test_augmentation_can_fit_g1d` check both halves.

### Pitfalls

> [!pitfall]
> **Assuming the two gradients are equal.** The adjoint gradient and the gradient of the discrete forward pass differ by the error of the solver. The test `test_adjoint_matches_backprop_as_the_steps_shrink` checks that the gap shrinks by more than a factor of 8 per halving of the step with RK4, and `test_backprop_is_the_exact_gradient_of_the_solver` that backpropagation matches finite differences of the solver itself.

> [!pitfall]
> **Storing the forward pass in the adjoint method.** The point of the adjoint is that $h$ is recomputed backwards. The test `test_adjoint_memory_does_not_grow_with_the_steps` records the state of the backward solve and checks that it is $2 B D + P$ numbers for 2, 8 and 32 steps.

> [!pitfall]
> **A field that changes the batch.** Each example must follow its own ODE. The test `test_batch_independence` checks that solving a batch equals solving its rows one at a time.

> [!pitfall]
> **Comparing a log density without its trace.** The log density of a CNF must be integrated with the state. The test `test_instantaneous_change_of_variables` checks the integrated trace against the log determinant of the flow map's Jacobian.

## Experiments

This page has no experiment yet. The checks of its six derivations play the part of small experiments: they measure the orders of Euler and RK4, compare the adjoint gradient with finite differences, integrate the trace against a log determinant, train a one-dimensional flow against the bound of $\frac12$, run Euler steps on antisymmetric fields and count the multiply-adds of both passes.

## History

Chen et al. trace the reading of residual networks as Euler discretizations to Lu et al. and to Haber and Ruthotto in 2017. {#c-hist-euler}
They write that the use of the adjoint method for training continuous-time neural networks was previously proposed by LeCun et al. (1988) and Pearlmutter (1995), but not demonstrated practically. {#c-hist-adjoint}
Chen, Rubanova, Bettencourt and Duvenaud presented Neural Ordinary Differential Equations at NeurIPS 2018. {#c-hist-neurips}
Augmentation (Dupont et al., 2019), latent ODEs for irregular time series (Rubanova et al., 2019), the critique of the backward solve (Gholami et al., 2019) and regularized dynamics (Finlay et al., 2020) followed within two years, and Kidger's 2021 thesis collected the field.

## References

- R. T. Q. Chen, Y. Rubanova, J. Bettencourt and D. Duvenaud. Neural Ordinary Differential Equations. NeurIPS 2018.
- E. Dupont, A. Doucet and Y. W. Teh. Augmented Neural ODEs. NeurIPS 2019.
- E. Haber and L. Ruthotto. Stable Architectures for Deep Neural Networks. arXiv 1705.03341, 2017.
- Y. Lu, A. Zhong, Q. Li and B. Dong. Beyond Finite Layer Neural Networks: Bridging Deep Architectures and Numerical Differential Equations. arXiv 1710.10121, 2017.
- A. Gholami, K. Keutzer and G. Biros. ANODE: Unconditionally Accurate Memory-Efficient Gradients for Neural ODEs. arXiv 1902.10298, 2019.
- Y. Rubanova, R. T. Q. Chen and D. Duvenaud. Latent ODEs for Irregularly-Sampled Time Series. arXiv 1907.03907, 2019.
- C. Finlay, J.-H. Jacobsen, L. Nurbekyan and A. M. Oberman. How to Train Your Neural ODE: the World of Jacobian and Kinetic Regularization. ICML 2020.
- P. Kidger. On Neural Differential Equations. DPhil thesis, University of Oxford, 2022.
