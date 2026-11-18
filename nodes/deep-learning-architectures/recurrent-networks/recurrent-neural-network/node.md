---
tag: FKJKTR
kind: architecture
slug: recurrent-neural-network
name: Recurrent neural network
aliases: [RNN, simple recurrent network, SRNN, Elman network, vanilla RNN, Elman RNN]
area: deep-learning-architectures/recurrent-networks
level: core
summary: >-
  A network that reads a sequence step by step, computing each new hidden state as a nonlinearity of an affine map of the input and the previous state, with the same weights at every step.
uses: [matrix-multiplication@m2, activation-function@al]
related_to: [long-short-term-memory, gated-recurrent-unit, transformer, gradient-clipping, structured-state-space-model]
history:
  year: 1990
  people: [Jeffrey L. Elman]
  note: "Elman's Finding Structure in Time (Cognitive Science, 1990) introduced the simple recurrent network with context units, after the recurrent networks of Hopfield (1982) and Jordan (1986); Mikolov et al. (2010) made it a language model, and Le, Jaitly and Hinton (2015) revived the relu version with an identity initialization."
prereqs: [backpropagation-through-time@al, backpropagation@al, matrix-multiplication@m2, activation-function@al]
same_as:
  wikipedia: Recurrent_neural_network
sources:
  - {paper: "Lipton, Berkowitz and Elkan (2015)", title: "A Critical Review of Recurrent Neural Networks for Sequence Learning", locator: "arXiv 1506.00019v4, Sections 1.3, 3, 3.1 and 3.2", note: "the definition, the unfolding, the early designs and the difficulty of training"}
  - {paper: "Mikolov, Karafiát, Burget, Černocký and Khudanpur (2010)", title: "Recurrent neural network based language model", locator: "INTERSPEECH 2010, the abstract and Section 2", note: "the Elman network as a language model"}
  - {paper: "Pascanu, Mikolov and Bengio (2013)", title: "On the difficulty of training recurrent neural networks", locator: "ICML 2013, Sections 2.1, 3.2 and 4.2", note: "the vanishing and exploding gradients and norm clipping"}
  - {paper: "Le, Jaitly and Hinton (2015)", title: "A Simple Way to Initialize Recurrent Networks of Rectified Linear Units", locator: "arXiv 1504.00941v2, Sections 2 to 4, Tables 1, 3 and 4", note: "the IRNN"}
  - {paper: "Weiss, Goldberg and Yahav (2018)", title: "On the Practical Computational Power of Finite Precision RNNs for Language Recognition", locator: "ACL 2018; arXiv 1805.04908v1, the abstract and Section 4", note: "what a tanh and a relu RNN can count"}
  - {paper: "Hochreiter and Schmidhuber (1997)", title: "Long Short-Term Memory", locator: "Section 1", note: "the decaying error flow that motivated the LSTM"}
  - {paper: "Vaswani et al. (2017)", title: "Attention Is All You Need", locator: "Section 1", note: "the sequential computation of recurrent networks"}
  - {url: "https://docs.pytorch.org/docs/2.15/generated/torch.nn.RNN.html", title: "torch.nn.RNN", locator: "the equation, the parameters and the initialization", note: "the two biases, the layout of the weights and the two nonlinearities"}
family: recurrent
inputs: "a batch of sequences as a (B, T, I) array, and an optional initial state (B, H)"
outputs: "the state of every step, a (B, T, H) array, and the final state (B, H)"
components: []
inductive_biases: ["sequential order: one step function, shared over time, is applied from left to right", "a fixed-size state that summarizes the whole prefix", "the same weights at every position, so any length can be read", "a multiplicative path through time: the gradient is multiplied by the step Jacobian at every step"]
params: "H I + H^2 + 2 H for one layer with two biases, whatever the length of the sequence"
flops_per_example: "2 T H (I + H) for a sequence of T steps, counting a multiply-add as 2 FLOPs and only the two products"
---

A recurrent neural network (RNN) reads a sequence one element at a time and keeps a hidden state, a vector that it updates at every step from the new input and its previous value.
Lipton et al. describe recurrent networks as feedforward networks augmented by edges that span adjacent time steps, so that at time t the nodes with recurrent edges receive input from the current data point and from the hidden node values of the previous state. {#c-def}
The simplest kind, the subject of this page, computes the new state as one nonlinearity of one affine map of the input and the old state; it is called the simple recurrent network, the Elman network or the vanilla RNN, to tell it apart from gated units such as the [[long-short-term-memory|LSTM]] and the [[gated-recurrent-unit|GRU]].
The same weights are used at every step, so one set of parameters reads sequences of any length.

## In plain words

Reading a sentence, you do not keep every word you have seen; you keep a running sense of what it is about, and revise it with each new word.
An RNN does the same with a list of numbers, its state.
At every step it mixes the new input with its current state, squashes the result, and that is its new state.
The rule for mixing is learned, and it is the same rule at every step, which is why the network can read a sequence of any length.
The weakness is in the repetition: whatever the network knew about an early input has been mixed and squashed again at every later step, so it tends to fade, and the signal that should teach the network about that early input fades the same way, or blows up.
Most of the history of recurrent networks after 1990 is the search for ways around this.

## Overview

A recurrent layer reads a sequence $\mathbf{x}_1, \dots, \mathbf{x}_T$ of vectors of size $d_x$ and carries a state $\mathbf{h}_t$ of size $d_h$ from each step to the next; $\mathbf{h}_t$ is also the output of the step.
A readout on $\mathbf{h}_t$ gives a prediction at every step, as in a language model, or, from $\mathbf{h}_T$ alone, one prediction for the whole sequence, as in classification.
Layers can be stacked, the states of one being the inputs of the next.

The loop can be drawn out.
Lipton et al. point out that, unfolded across time steps, the network can be interpreted not as cyclic but as a deep network with one layer per time step and shared weights across time steps. {#c-unfolded}
This view is how the network is trained, by [[al:backpropagation-through-time|backpropagation through time]], and it is also the root of its trouble: a deep network with tied weights multiplies its gradient by the same kind of matrix once per layer, and the depth is the length of the sequence.

Mikolov et al. used what they call a simple recurrent neural network or Elman network, with an input layer x, a hidden or context layer s and an output layer y, the input x(t) being the concatenation of the vector w(t) of the current word and the output of the context layer at time t − 1, a sigmoid on the hidden layer and a softmax on the output. {#c-mikolov-elman}
Concatenating the input with the old state and applying one weight matrix is the same as applying two matrices, one to each, and adding, which is the form of eq-step below.

## Components

The notation is that of torch.nn.RNN, with vectors as rows: $W$ is the $d_x \times d_h$ input matrix, $U$ the $d_h \times d_h$ recurrent matrix, and $\mathbf{b}_i$, $\mathbf{b}_h$ two bias vectors of size $d_h$.

**The step.**
One affine map of the input, one of the previous state, added and passed through a nonlinearity $f$, an [[al:activation-function|activation function]] applied coordinate by coordinate:

$$ \mathbf{a}_t = \mathbf{x}_t W + \mathbf{b}_i + \mathbf{h}_{t-1} U + \mathbf{b}_h, \qquad \mathbf{h}_t = f(\mathbf{a}_t) . $$ {#eq-step}

torch.nn.RNN applies an Elman RNN with a tanh or relu nonlinearity, computing h_t = tanh(x_t W_ih^T + b_ih + h_{t-1} W_hh^T + b_hh) for each element of the input sequence, with relu used in place of tanh when the nonlinearity is 'relu'. {#c-torch-equations}
Its weight_ih_l0 and weight_hh_l0 are $W^\top$ and $U^\top$ in the notation of this page, and it initializes all weights and biases from $\mathcal U(-\sqrt{k}, \sqrt{k})$ with $k = 1/\text{hidden\_size}$. {#c-torch-init}
Lipton et al. write the same step with the logistic sigmoid and a single bias, h(t) = σ(W_hx x(t) + W_hh h(t−1) + b_h), followed by the readout ŷ(t) = softmax(W_yh h(t) + b_y). {#c-lipton-equations}

**The initial state.**
$\mathbf{h}_0$ is a fixed vector, usually zero, or a learned parameter, or the final state of a previous chunk of the same stream.
Mikolov et al. set s(0) to a vector of small values, like 0.1, and note that when processing a large amount of data the initialization is not crucial. {#c-mikolov-s0}

**The nonlinearity.**
tanh keeps every coordinate in $(-1, 1)$ and is the default; relu keeps the state nonnegative and unbounded; the logistic sigmoid of the older papers keeps it in $(0, 1)$.
The choice decides both what the state can represent (Inductive biases below) and how gradients behave (Training below).

**The readout.**
The recurrent layer only produces states; a task adds a map from $\mathbf{h}_t$ to its outputs, such as a linear layer and a softmax over a vocabulary, which is not counted among the parameters of the layer here.

## Shapes and parameters

A layer with input width $I = d_x$ and $H = d_h$ units has

$$ N = H I + H^2 + 2 H $$ {#eq-params}

parameters in the form of torch.nn.RNN, whatever the length of the sequence. {#c-params}
In the shape contract the formula is

{shape:params}

and the example layer, 3 inputs and 4 units, has {shape:params@example} parameters.
The two biases only ever enter eq-step as their sum $\mathbf{b}_i + \mathbf{b}_h$, so the layer computes exactly the same functions with one bias, {shape:one_bias_params@example} parameters in the example. {#c-one-bias}
A [[gated-recurrent-unit|GRU]] and an [[long-short-term-memory|LSTM]] layer of the same widths, in the same form with two biases per block, have exactly three and four times as many parameters and FLOPs, {shape:gru_params@example} and {shape:lstm_params@example} in the example, because they compute three and four such affine maps per step instead of one. {#c-gated-ratio}
The networks of Le et al. on pixel-by-pixel MNIST, 100 units reading one pixel per step, have {shape:params@pixel} parameters in the recurrent layer. {#c-pixel-size}
Le et al. note, for their adding experiments with 100 hidden units in every network, that the LSTMs had more parameters by a factor of about 4 and also took about 4 times as much computation per timestep. {#c-le-factor4}
The derivation `parameter-and-flop-count` proves these counts.

## Computation and cost

Each step makes two matrix products, $\mathbf{x}_t W$ of $B \times I$ by $I \times H$ and $\mathbf{h}_{t-1} U$ of $B \times H$ by $H \times H$, so a forward pass over a batch of $B$ sequences of length $T$ costs

$$ C = 2 B T H (I + H) $$ {#eq-flops}

FLOPs, counting a multiply-add as two and leaving out the nonlinearity. {#c-flops}
The example layer on a batch of 2 sequences of 5 steps costs {shape:flops_forward@example} FLOPs; a layer of 1024 units on 1024 input features costs {shape:flops_forward@wide} per batch of 2 sequences of 5 steps, against {shape:gru_flops@wide} for a GRU and {shape:lstm_flops@wide} for an LSTM of the same width.

The input products of all steps do not depend on the state and can be computed in one $(BT) \times I$ by $I \times H$ product before the loop.
Only the recurrent product has to wait: the $T$ products $\mathbf{h}_{t-1} U$ form a chain in which each needs the result of the one before, so the depth of the computation grows linearly with $T$ however many processors are available.
The state carried from step to step is $H$ numbers per sequence, {shape:state@example} in the example, against $2H$ for an LSTM; at inference time the memory of the model does not grow with the length of what it has read.
Vaswani et al. write that recurrent models generate a sequence of hidden states h_t as a function of the previous hidden state h_{t-1} and the input for position t, and that this inherently sequential nature precludes parallelization within training examples. {#c-sequential}
The [[transformer]] removed the recurrence for this reason, and pays for it with a cost and a memory that grow with the length of the context.

## Training

### Backpropagation through time

Lipton et al. write that the unfolded network can be trained across many time steps using backpropagation, an algorithm called backpropagation through time (BPTT) that was introduced by Werbos in 1990 and that all recurrent networks in common current use apply. {#c-bptt-werbos}
It is ordinary [[al:backpropagation|backpropagation]] on the unfolded graph.
Write $\mathbf{g}_t = \partial L / \partial \mathbf{h}_t$ for the whole gradient of the loss with respect to the state of step $t$, $\mathbf{e}_t$ for the part that comes from the loss at step $t$ directly, and $\boldsymbol\delta_t = \mathbf{g}_t \odot f'(\mathbf{a}_t)$.
Going backwards from $\mathbf{g}_T = \mathbf{e}_T$,

$$ \mathbf{g}_{t-1} = \mathbf{e}_{t-1} + \boldsymbol\delta_t U^\top, \qquad \frac{\partial L}{\partial U} = \sum_{t=1}^{T} \mathbf{h}_{t-1}^\top \boldsymbol\delta_t . $$ {#eq-bptt}

The first equation is a recursion over the steps, run in the opposite order of the forward pass, and the same recursion gives the gradient with respect to $\mathbf{h}_0$. {#c-bptt-recursion}
The second says that the gradient of a shared parameter is the sum, over the steps, of the gradients with respect to the copy of the parameter that each step uses in the unfolded network; the same holds for $W$ with $\mathbf{x}_t$ in place of $\mathbf{h}_{t-1}$ and for both biases with $\boldsymbol\delta_t$ alone. {#c-bptt-sum}
The backward pass costs $4BTH(I + H)$ FLOPs in its products, twice the forward pass. {#c-bptt-cost}
It needs the $T$ states of the forward pass, $BTH$ numbers that grow with the length of the sequence, which is the memory cost of training on long sequences. {#c-bptt-memory}
The derivation `bptt-sum` proves these four statements.

### Truncation

Truncated backpropagation through time (TBPTT), following Williams and Zipser (1989), sets a maximum number of time steps along which error can be propagated; Lipton et al. note that with a small cutoff it alleviates the exploding gradient problem but requires that one sacrifice the ability to learn long-range dependencies. {#c-tbptt}
In practice a long stream is cut into chunks of $k$ steps; the state is carried from one chunk to the next, but the gradient is not.
Mikolov et al. trained their language model with what they call truncated backpropagation through time with τ = 1, which is not optimal because the weights are updated based on the error vector computed only for the current time step. {#c-mikolov-tbptt}

### Vanishing and exploding gradients

The gradient that reaches step $k$ from the loss at a later step $t$ passes through every step in between.
With $\mathbf{h}_t$ as a column vector, the Jacobian of one step is

$$ J_t = \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}} = \operatorname{diag}(f'(\mathbf{a}_t))\, U^\top , $$ {#eq-jacobian}

for one sequence, with $\mathbf{a}_t$ the pre-activation of step $t$. {#c-jacobian}
The gradient over several steps follows by the chain rule: $\partial \mathbf{h}_t / \partial \mathbf{h}_k = J_t J_{t-1} \cdots J_{k+1}$ is a product of $t - k$ such factors. {#c-product}
If $|f'| \le \gamma$ everywhere, the [[m2:matrix-norm|spectral norm]] of the product is bounded by

$$ \Big\| \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_k} \Big\|_2 \le \big(\gamma\, \sigma_1(U)\big)^{t-k} , $$ {#eq-bound}

where $\sigma_1(U)$ is the largest [[m2:singular-value-decomposition|singular value]] of $U$. {#c-bound}
So $\sigma_1(U) < 1/\gamma$ is sufficient for the gradient to vanish exponentially in $t - k$. {#c-vanish-sufficient}
Conversely, $\sigma_1(U) > 1/\gamma$ is necessary for it to explode. {#c-explode-necessary}
$\gamma = 1$ for tanh and relu, and $\gamma = 1/4$ for the logistic sigmoid. {#c-gamma}
The derivation `gradient-bound` proves the bound.

This is the [[al:vanishing-gradient-problem|vanishing gradient problem]] and the [[al:exploding-gradient-problem|exploding gradient problem]] of recurrent networks.
Hochreiter and Schmidhuber wrote that with BPTT or real-time recurrent learning, error signals going backwards in time tend to either blow up or vanish, because the temporal evolution of the backpropagated error exponentially depends on the size of the weights. {#c-hochreiter-decay}
Lipton et al. illustrate it with a single recurrent node, whose contribution from an early input either explodes or approaches zero exponentially fast depending on whether the weight of its recurrent edge is above or below 1 in absolute value; with a sigmoid the vanishing problem is more pressing, and with a relu it is easier to imagine the exploding one. {#c-lipton-toy}
Pascanu, Mikolov and Bengio state the condition as follows: it is sufficient for the largest eigenvalue λ1 of the recurrent weight matrix to be smaller than 1 for long term components to vanish, and necessary for it to be larger than 1 for gradients to explode, with 1 replaced by 1/γ for a nonlinearity whose derivative is bounded by γ. {#c-pascanu-condition}
Their proof bounds the 2-norm of $U$, that is its largest singular value, and the two agree only for normal matrices; the derivation `gradient-bound` gives a non-normal $U$ whose eigenvalues are all 0.5 and whose products still grow for several steps.

> [!pitfall]
> **Reading the eigenvalue condition as a guarantee.** A recurrent matrix with spectral radius below 1 can still amplify gradients over many steps when it is far from normal, and one with spectral radius above 1 need not explode, because the derivative of the nonlinearity also enters every factor. The bound eq-bound is on the largest singular value, and it is an upper bound only.

### Gradient clipping

The standard remedy for exploding gradients is [[gradient-clipping]].
Pascanu et al. rescale the gradient whenever its norm goes over a threshold, $\hat{\mathbf{g}} \leftarrow \text{threshold}\, \hat{\mathbf{g}} / \|\hat{\mathbf{g}}\|$ if $\|\hat{\mathbf{g}}\| \ge \text{threshold}$ (their Algorithm 1), which they describe as very similar to the one proposed by Tomas Mikolov, who clipped the elements of the gradient instead. {#c-clip-alg}

$$ \hat{\mathbf{g}} \leftarrow \min\!\Big(1, \frac{\text{threshold}}{\|\hat{\mathbf{g}}\|}\Big)\, \hat{\mathbf{g}} $$ {#eq-clip}

They report that training is not very sensitive to the threshold and that the algorithm behaves well even for rather small thresholds, and suggest the average norm over a sufficiently large number of updates as a heuristic choice. {#c-clip-threshold}
From their experiments they conclude that clipping solves an optimization issue and does not act as a regularizer. {#c-clip-not-reg}
Clipping does nothing for vanishing gradients, which do not make training unstable but make it blind to long dependencies; for those, Pascanu et al. proposed a regularization term, and the field mostly adopted gated units instead.

### The identity initialization

Le, Jaitly and Hinton revisited the relu RNN with a single change: initialize the recurrent weight matrix to be the identity matrix and the biases to be zero. {#c-irnn-def}
They write that in the absence of input, an RNN composed of ReLUs and initialized with the identity matrix, which they call an IRNN, just stays in the same state indefinitely, and that its error derivatives remain constant when no extra error-derivatives are added. {#c-irnn-le}
Both statements are exact: with $U = I$, zero biases and zero inputs, a nonnegative state never changes. {#c-irnn-state}
If all its coordinates are positive, the Jacobian of eq-jacobian is the identity, so the gradient is carried back unchanged over any number of steps. {#c-irnn-grad}
With $U = cI$ and $0 < c < 1$ the state decays as $c^t$ and the gradient as $c^{t-k}$. {#c-irnn-scaled}
Le et al. note that scaling the identity matrix by a small scalar is an effective mechanism to forget long range effects. {#c-irnn-forget}
The trick needs the relu: with tanh and $U = I$, a state that starts in $(0, 1]$ shrinks at every step, roughly as $\sqrt{3/(2t)}$. {#c-tanh-identity}
The derivation `identity-initialization` proves these statements.
In their experiments the non-recurrent weights were drawn from a Gaussian with standard deviation 0.001, and all models were trained with SGD with a fixed learning rate and gradient clipping, selected by a grid search over learning rates from $10^{-9}$ to $10^{-1}$ and clipping thresholds 1, 10, 100 and 1000. {#c-le-setup}
Weiss et al. caution that ReLU-activated RNNs are known to be notoriously hard to train because of the exploding gradient problem. {#c-weiss-relu-hard}

## Inductive biases

**Order and weight sharing.** One step function is applied at every position, so the network handles sequences of any length with the same parameters, the order of the inputs matters, and a pattern learned at one position is recognized at every other.

**A fixed-size summary.** Everything the network knows about the prefix $\mathbf{x}_1, \dots, \mathbf{x}_t$ must fit in the $H$ numbers of $\mathbf{h}_t$, and an output at step $t$ depends on the past only through it.
This is the opposite choice to attention, which keeps every past state and lets each output look at all of them.

**A bounded or a nonnegative state.** With tanh every coordinate of $\mathbf{h}_t$ stays in $(-1, 1)$; with relu every coordinate is nonnegative and unbounded.
The tests `test_tanh_state_is_bounded_and_relu_state_is_nonnegative` check both.

**Exponential forgetting by default.** By eq-bound, a tanh network whose recurrent matrix has a largest singular value below 1 forgets its inputs, and the sensitivity of its gradient to them, at an exponential rate; long memory requires the network to sit near the edge, where the gradient can also explode.

**Expressive power.** Lipton et al. recall that a finite-sized recurrent network with sigmoidal activation functions can simulate a universal Turing machine (Siegelmann and Sontag, 1991), and point out that this says nothing about whether gradient descent can find such a network. {#c-turing}
With finite precision the picture changes.
Weiss et al. argue that the LSTM and the Elman RNN with relu activation are strictly stronger than the RNN with a squashing activation and the GRU, because LSTMs and ReLU RNNs can easily implement counting behavior. {#c-weiss-stronger}
For the tanh SRNN, $h_t = \tanh(W x + U h_{t-1} + b)$, they argue that the finite-precision network cannot designate unbounded counting dimensions, because its counting behavior happens inside a tanh, which makes it inherently unstable and bounded to a relatively narrow region. {#c-weiss-srnn}
A finite-precision IRNN, on the other hand, can perform unbounded counting conditioned on its input symbols, with two dimensions per counter. {#c-weiss-irnn}

## Evidence

**Language modelling.** Mikolov et al. trained Elman networks as language models for speech recognition, usually with 30 to 500 hidden units. {#c-mikolov-hidden}
They report that a mixture of several RNN language models obtained around 50% reduction of perplexity against a strong backoff language model, and, in speech recognition, around 18% reduction of word error rate on the Wall Street Journal task against a backoff model trained on the same amount of data, and around 5% on the much harder NIST RT05 task. {#c-mikolov-results}

**Long dependencies.** Le et al. compared tanh RNNs, LSTMs and IRNNs of 100 units on the adding problem, where the target is the sum of two marked numbers in a sequence of length $T$.
Both LSTMs and RNNs started to struggle when T was around 150, and for T = 200, 300 and 400 the grid search found no hyperparameter combination that gave a good result for the tanh RNN (Table 1), while the IRNN converged about as well as the LSTM. {#c-le-adding}
On MNIST read one pixel at a time, 784 steps per image, standard RNNs failed to work, even with ReLUs, whereas the IRNN achieved a 3% test error rate. {#c-le-mnist}
In their overview they put this against 34% for the LSTM. {#c-le-mnist-lstm}
With a fixed random permutation of the pixels, IRNNs were still better than LSTMs on the non-permuted pixels. {#c-le-permuted}

**At scale.** On the one billion word benchmark, with about the same number of parameters, the test perplexities in Le et al.'s Table 3 are 68.8 for an LSTM of 512 units, 69.4 for an IRNN of 4 layers of 512 units, 70.2 for an IRNN of 1024 units with a projection, 71.8 for a tanh RNN of 4 layers of 512 units and 72.5 for a tanh RNN of 1024 units with a projection. {#c-le-lm}
On TIMIT phoneme recognition they used $0.01 I$ instead of the identity, because the full identity led to slow convergence, worse results and sometimes divergence. {#c-le-speech-init}
The frame error rates (dev / test) of Table 4 are 35.0 / 36.2 for a tanh RNN of 500 neurons and 2 layers, 34.5 / 35.4 for an LSTM of 250 cells and 2 layers, 34.3 / 35.5 for the iRNN of 500 neurons and 2 layers, and 33.0 / 33.8 for the iRNN of 5 layers, against 35.6 / 37.0 for the tanh RNN of 5 layers. {#c-le-speech}

**The first experiments.** Lipton et al. recount that on a temporal version of XOR, where a perfect system reaches 66.7% accuracy, the simple network of Elman approached that maximum. {#c-lipton-xor}
These are reported results; this page has not reproduced them.

## Limitations

- **Short memory in practice.** Gradients through many steps vanish or explode (eq-bound), so a plain RNN trained by gradient descent rarely uses information more than a few dozen steps back unless it is initialized or constrained to do so.
- **Unstable training.** Exploding gradients make clipping necessary, and the relu version, which can remember longer, is the more prone to explode.
- **Sequential computation.** Each step waits for the previous one, so training cannot be parallelized over the length of a sequence.
- **Memory of training grows with length.** BPTT stores every state of the forward pass, and truncating it gives up dependencies longer than the cutoff.
- **A bottleneck state.** All the information about the past has to pass through $H$ numbers, overwritten in full at every step.
- **Bounded counting with tanh.** A finite-precision tanh RNN cannot implement an unbounded counter (Weiss et al.).

## Variants and descendants

- **Jordan networks** feed the outputs of the network, rather than its hidden states, back as inputs at the next step. Lipton et al. describe Jordan's 1986 network as a feedforward network with a single hidden layer whose output values are fed to special units, which feed them to the hidden nodes at the following time step. {#c-jordan}
- **Gated units.** The [[long-short-term-memory|LSTM]] adds a memory cell with an additive update and gates, and the [[gated-recurrent-unit|GRU]] an update gate that interpolates between the old state and a tanh candidate; both turn the multiplicative path of eq-jacobian into one that can carry a gradient unchanged. The GRU contains the simple RNN as the limit in which its update gate is 0 and its reset gate 1.
- **Stacked and bidirectional RNNs.** Several layers are stacked, the states of one being the inputs of the next, and a second layer reads the sequence backwards so that every output sees the whole sequence; torch.nn.RNN provides both through num_layers and bidirectional.
- **Encoder-decoders.** One RNN reads a sequence into its final state and another generates an output sequence from it, the setting in which the GRU was introduced and in which attention was later added.
- **Constrained recurrences.** The IRNN keeps the recurrent matrix near the identity at initialization; unitary and orthogonal RNNs keep it orthogonal throughout training, so that its singular values are all 1.
- **Linear recurrences and state space models.** Dropping the nonlinearity from the recurrence, $\mathbf{h}_t = \mathbf{h}_{t-1} A + \mathbf{x}_t B$, makes the whole sequence of states computable by a parallel scan or a convolution; [[structured-state-space-model|structured state space models]] and Mamba are of this kind, with the nonlinearities placed between the layers instead.

## Notable models

Elman's simple recurrent networks of 1990 were small models of language and of temporal structure in cognitive science.
The RNN language models of Mikolov et al. (2010) brought the Elman network into speech recognition and machine translation, and helped make neural language models standard.
Le et al.'s IRNN (2015) found that a relu RNN with the right initialization can be competitive with an LSTM on long-dependency benchmarks, and the pixel-by-pixel MNIST task they used has since been widely used to test recurrent memory.
In most later sequence models the simple RNN was replaced by gated units, then by the [[transformer]], and it survives mainly as a baseline, as a component of larger models and as the starting point of every analysis of recurrent training.

## Implementations

The NumPy reference `impl/numpy/rnn.py` implements eq-step with tanh and relu, the forward pass over a sequence, backpropagation through time by hand (eq-bptt), the step Jacobian eq-jacobian, the identity initialization of Le et al. and the counts eq-params and eq-flops with those of a GRU and an LSTM of the same width.
The PyTorch version `impl/pytorch/rnn.py` is an `nn.Module` with the same parameters that matches `torch.nn.RNN` to rounding after `to_torch_rnn()` copies its weights, for both nonlinearities; it also implements eq-clip, which `test_clip_by_norm_is_torch_clip_grad_norm` compares with `torch.nn.utils.clip_grad_norm_`, and is tested with `torch.autograd.gradcheck` and by training it to carry a bit across 5 steps.
The JAX version `impl/jax/rnn.py` is a pure function that runs the loop with `jax.lax.scan`, tested under `jax.jit` and `jax.vmap` and with `jax.grad` against the gradients of the NumPy backward pass.
All three load the shared fixture of `fixtures/`: the example config, its states and pre-activations for both nonlinearities, and the gradients of a fixed loss.

### Worked example

Take one unit with input weight $W = 1$, recurrent weight $U = 0.9$, zero biases and $h_0 = 0$, and feed it a single impulse, $x_1 = 1$ and $x_t = 0$ afterwards.
With tanh, $h_1 = \tanh(1) = 0.7616$, $h_2 = \tanh(0.9 \times 0.7616) = 0.5950$, and after ten steps $h_{10} = 0.1892$.
The gradient of $h_{10}$ with respect to $x_1$ is the product of eq-jacobian over the nine steps after the first, times the first step's own derivative: $0.9^9 \prod_{t=1}^{10} (1 - h_t^2) = 0.04195$.
With relu the state is $h_t = 0.9^{t-1}$, so $h_{10} = 0.3874$, and the gradient is $0.9^9 = 0.3874$: the relu unit forgets at exactly the rate of its weight, while the tanh unit also loses a factor $1 - h_t^2$ at every step.
The tests `test_worked_example_tanh` and `test_worked_example_relu` check these numbers in NumPy, the second through the backward pass, and `test_worked_example` checks them in PyTorch and JAX.

### Pitfalls

> [!pitfall]
> **Two biases that are one.** torch.nn.RNN has both `bias_ih` and `bias_hh`, but only their sum enters eq-step, so the split between them is arbitrary and weight decay on each separately is not weight decay on the sum. A model with one bias loads by putting the bias in either vector and zeros in the other. The test `test_two_biases_only_enter_as_a_sum` checks that shifting one bias against the other leaves the outputs unchanged.

> [!pitfall]
> **The transposed weights.** torch.nn.RNN stores `weight_ih_l0` as (hidden_size, input_size) and computes $x W_{ih}^\top$, while this page and many papers write $x W$ with $W$ of shape $(I, H)$, or $W x$ with column vectors. With square layers the wrong orientation runs without error and computes a different function. The test `test_matches_torch_nn_rnn` checks that the transposed copy matches `torch.nn.RNN` and that, on a square layer, the untransposed copy runs and computes something else.

> [!pitfall]
> **Forgetting that the gradient is a sum.** In a hand-written BPTT the gradient of $U$ must be accumulated over every step; keeping only the last term, or overwriting it in the loop, gives a gradient that trains but is wrong. The test `test_shared_weight_gradient_is_the_sum_over_steps` checks that the gradient of the shared weights equals the sum of the gradients of per-step copies.

> [!pitfall]
> **An exploding relu network.** Relu does not squash, so a recurrent matrix with a singular value above 1 makes both the state and the gradient grow exponentially. The test `test_vanishing_and_exploding` checks that with relu, $U = 1.5 I$ and a positive state the gradient over 50 steps is exactly $1.5^{50}$, and that with tanh and an orthogonal $U$ scaled to a spectral norm of 0.5 it is within a factor 2 of $0.5^{50}$ at most; `test_gradient_bound` checks eq-bound on random networks.

> [!pitfall]
> **Detaching the state between chunks.** With truncated BPTT the state must be carried from one chunk to the next but detached from the graph; not carrying it loses the context, and not detaching it makes the graph, and the memory, grow without bound.

## Experiments

This page has no experiment yet.
The PyTorch test `test_learns_to_carry_a_bit_over_5_steps` is a small one: a tanh RNN of 8 units, trained for 200 steps of Adam with the gradient clipped to norm 1, learns to report the sign of the first of 5 inputs, the other 4 being noise, with an accuracy above 95% on 1000 fresh sequences.
The test `test_gradient_vanishes_with_distance` shows the other side: in a tanh RNN of 16 units at the default initialization of torch.nn.RNN, the gradient of the final state with respect to the input 60 steps back is less than $10^{-6}$ of its gradient with respect to the last input.
The checks of the four derivations count parameters and products, compare the bound eq-bound with random and non-normal matrices, run the IRNN without input for many steps, and compare the BPTT recursion with finite differences.

## History

Lipton et al. trace the early recurrent designs to Hopfield's networks of 1982, which have pattern recognition capabilities, and to the network of Jordan (1986), an early architecture for supervised learning on sequences. {#c-hist-early}
They describe the architecture introduced by Elman in 1990 as simpler than the earlier Jordan architecture, and note that Jordan calls his special units state units while Elman calls the corresponding structure context units. {#c-hist-elman}
In 1990 Werbos also published the account of backpropagation through time on which the training of every recurrent network rests.
Lipton et al. note that learning with recurrent networks can be especially challenging due to the difficulty of learning long-range dependencies, as described by Bengio et al. (1994) and expanded upon by Hochreiter et al. (2001). {#c-hist-difficulty}
The same difficulty, which Hochreiter had analysed in 1991, led Hochreiter and Schmidhuber to the LSTM in 1997.
Mikolov et al. presented their recurrent neural network based language model at INTERSPEECH 2010 in Makuhari, Japan. {#c-hist-mikolov}
Pascanu, Mikolov and Bengio's analysis of the vanishing and exploding gradients and their norm clipping (2013), and Le, Jaitly and Hinton's identity initialization (2015), are the main results specific to the simple RNN since then.

## References

- J. L. Elman. Finding Structure in Time. Cognitive Science 14(2), 1990.
- Z. C. Lipton, J. Berkowitz and C. Elkan. A Critical Review of Recurrent Neural Networks for Sequence Learning. arXiv 1506.00019, 2015.
- T. Mikolov, M. Karafiát, L. Burget, J. Černocký and S. Khudanpur. Recurrent neural network based language model. INTERSPEECH 2010.
- R. Pascanu, T. Mikolov and Y. Bengio. On the difficulty of training recurrent neural networks. ICML 2013.
- Q. V. Le, N. Jaitly and G. E. Hinton. A Simple Way to Initialize Recurrent Networks of Rectified Linear Units. arXiv 1504.00941, 2015.
- G. Weiss, Y. Goldberg and E. Yahav. On the Practical Computational Power of Finite Precision RNNs for Language Recognition. ACL 2018.
- S. Hochreiter and J. Schmidhuber. Long Short-Term Memory. Neural Computation 9(8), 1997.
- A. Vaswani et al. Attention Is All You Need. NeurIPS 2017.
- PyTorch contributors. torch.nn.RNN. PyTorch 2.15 documentation.
