---
tag: XAGPFN
kind: architecture
slug: gated-recurrent-unit
name: Gated recurrent unit
aliases: [GRU, gated hidden unit, GRU network, GRU-RNN]
area: deep-learning-architectures/recurrent-networks
level: core
summary: >-
  A recurrent unit whose state is a gated interpolation between its old value and a tanh candidate, with an update gate and a reset gate and no separate memory cell.
uses: [hadamard-product@m2, matrix-multiplication@m2, activation-function@al]
related_to: [long-short-term-memory, transformer, gradient-clipping, structured-state-space-model]
history:
  year: 2014
  people: [Kyunghyun Cho, Bart van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio]
  note: "Cho et al. proposed the unit in arXiv 1406.1078 (June 2014) as the hidden unit of their RNN Encoder-Decoder; Chung, Gulcehre, Cho and Bengio called it the gated recurrent unit and compared it with the LSTM and the tanh unit in arXiv 1412.3555 (December 2014)."
prereqs: [long-short-term-memory, backpropagation-through-time@al, vanishing-gradient-problem@al, matrix-multiplication@m2]
same_as:
  wikipedia: Gated_recurrent_unit
  arxiv: ["1406.1078"]
sources:
  - {paper: "Cho, van Merriënboer, Gulcehre, Bahdanau, Bougares, Schwenk and Bengio (2014)", title: "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation", locator: "arXiv 1406.1078v3, Section 2.3 and Appendix A.1", note: "the original unit"}
  - {paper: "Chung, Gulcehre, Cho and Bengio (2014)", title: "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling", locator: "arXiv 1412.3555v1, Sections 1 to 5, Tables 1 and 2", note: "the name, and the comparison with the LSTM and the tanh unit"}
  - {paper: "Jozefowicz, Zaremba and Sutskever (2015)", title: "An Empirical Exploration of Recurrent Network Architectures", locator: "ICML 2015, PMLR 37, Sections 2.2, 3 and 4, Tables 1 and 3", note: "an architecture search against the GRU and the LSTM, and the forget-gate bias"}
  - {paper: "Weiss, Goldberg and Yahav (2018)", title: "On the Practical Computational Power of Finite Precision RNNs for Language Recognition", locator: "ACL 2018; arXiv 1805.04908v1, Sections 2, 4 and 5", note: "why a GRU cannot count"}
  - {paper: "Greff, Srivastava, Koutník, Steunebrink and Schmidhuber (2017)", title: "LSTM: A Search Space Odyssey", locator: "arXiv 1503.04069v2, pp. 3, 5 and 6", note: "the GRU as an LSTM with coupled gates"}
  - {paper: "Bahdanau, Cho and Bengio (2015)", title: "Neural Machine Translation by Jointly Learning to Align and Translate", locator: "arXiv 1409.0473v7, p. 5 and Appendix A.1.1", note: "the gated hidden unit in attention-based translation"}
  - {paper: "Gu and Dao (2023)", title: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", locator: "arXiv 2312.00752v2, Section 3.5.1", note: "gating as a discretized state space model"}
  - {paper: "Vaswani et al. (2017)", title: "Attention Is All You Need", locator: "Section 1", note: "the sequential computation of recurrent networks"}
  - {url: "https://docs.pytorch.org/docs/2.15/generated/torch.nn.GRU.html", title: "torch.nn.GRU", locator: "the equations, the parameters and the note on the new gate", note: "the reset gate after the matrix, the order of the blocks and the two biases"}
family: recurrent
inputs: "a batch of sequences as a (B, T, I) array, and an optional initial state (B, H)"
outputs: "the state of every step, a (B, T, H) array, and the final state (B, H)"
components: []
inductive_biases: ["sequential order: one step function, shared over time, is applied from left to right", "a gated additive path through time: the update gate interpolates between the old state and the candidate", "a bounded state: every coordinate stays in (-1, 1)", "time scales per unit, set by the update gates"]
params: "3 H I + 3 H^2 + 6 H for one layer with two biases, whatever the length of the sequence"
flops_per_example: "6 T H (I + H) for a sequence of T steps, counting a multiply-add as 2 FLOPs and only the two projections"
---

The gated recurrent unit (GRU) is a recurrent neural network unit with two gates and no memory cell.
At every step it computes a candidate state from the input and a reset version of the previous state, and then moves its state part of the way towards the candidate, by an amount that an update gate chooses for each coordinate.
Cho et al. proposed it in 2014 as a hidden unit with a reset gate and an update gate that adaptively control how much each hidden unit remembers or forgets. {#c-def}
Chung, Gulcehre, Cho and Bengio gave it the name gated recurrent unit later that year, in a comparison with the [[long-short-term-memory|LSTM]]. {#c-name}
It has three blocks of weights where the LSTM has four, and a single state vector where the LSTM has two.

## In plain words

A plain recurrent network rewrites all of its memory at every step: the new memory is a squashed mix of the old memory and the input.
After many steps the trace of an early input, and the gradient that should reach it during training, has been squashed and multiplied so often that it is lost.
A GRU does not overwrite its memory.
At each step it first proposes a new value for every number it keeps, and then, number by number, decides how far to move towards the proposal: not at all, all the way, or somewhere in between.
A number whose update gate stays near "keep" can carry a fact across hundreds of steps unchanged, and the gradient flows back along it the same way.
A second gate, the reset gate, decides how much of the old memory the proposal may look at, so that a unit can start afresh when the input tells it to.

## Overview

A GRU layer reads a sequence $\mathbf{x}_1, \dots, \mathbf{x}_T$ of vectors of size $d_x$, one step at a time, and carries one vector $\mathbf{h}_t$ of size $d_h$ from each step to the next; $\mathbf{h}_t$ is also the output of the step.
At step $t$, two gates and a candidate are computed from $\mathbf{x}_t$ and $\mathbf{h}_{t-1}$, and the new state is an elementwise interpolation between the old state and the candidate.
Layers can be stacked, the states of one layer being the inputs of the next, and a readout on $\mathbf{h}_t$ gives a prediction for each step or, from $\mathbf{h}_T$, for the whole sequence.
Cho et al. describe the unit as motivated by the LSTM unit but much simpler to compute and implement. {#c-motivated}
Chung et al. note that, like the LSTM unit, the GRU has gating units that modulate the flow of information inside the unit, but without a separate memory cell. {#c-no-cell}
The unit was introduced inside an encoder-decoder for machine translation, and the same gated unit is the recurrent unit of the attention-based translation model of Bahdanau, Cho and Bengio (Notable models below).

## Components

The notation is that of torch.nn.GRU, with vectors as rows: $\sigma$ is the logistic sigmoid, $\odot$ is the [[m2:hadamard-product|elementwise product]], $W_r, W_z, W_n$ are the three $d_x \times d_h$ blocks of the input matrix $W$ and $U_r, U_z, U_n$ the three $d_h \times d_h$ blocks of the recurrent matrix $U$.
Each block has two biases, one on the input side and one on the recurrent side.

**The two gates.**
Each is an affine map of the input and the previous state, squashed into $(0, 1)$ by the sigmoid [[al:activation-function|activation function]]:

$$ \mathbf{r}_t = \sigma(\mathbf{x}_t W_r + \mathbf{b}_{ir} + \mathbf{h}_{t-1} U_r + \mathbf{b}_{hr}), \qquad \mathbf{z}_t = \sigma(\mathbf{x}_t W_z + \mathbf{b}_{iz} + \mathbf{h}_{t-1} U_z + \mathbf{b}_{hz}) . $$ {#eq-gates}

**The candidate.**
The candidate, which PyTorch calls the new gate, is a tanh of the input and of the previous state as filtered by the reset gate:

$$ \mathbf{n}_t = \tanh\!\big(\mathbf{x}_t W_n + \mathbf{b}_{in} + \mathbf{r}_t \odot (\mathbf{h}_{t-1} U_n + \mathbf{b}_{hn})\big) . $$ {#eq-candidate}

**The update.**
The new state is a convex combination of the candidate and the old state, coordinate by coordinate:

$$ \mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{n}_t + \mathbf{z}_t \odot \mathbf{h}_{t-1} . $$ {#eq-update}

torch.nn.GRU computes the reset, update and new gates r_t, z_t and n_t with the sigmoid and the tanh as in eq-gates and eq-candidate, and h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}, with an input-hidden and a hidden-hidden bias for each gate. {#c-torch-equations}
It stores the weights of the three blocks stacked in the order reset, update, new, as (W_ir|W_iz|W_in) of shape (3*hidden_size, input_size), and initializes all weights and biases from $\mathcal U(-\sqrt{k}, \sqrt{k})$ with $k = 1/\text{hidden\_size}$. {#c-torch-layout}

**What the gates do.**
Cho et al. explain that when the reset gate is close to 0, the hidden state is forced to ignore the previous hidden state and reset with the current input only. {#c-reset-role}
The update gate controls how much information from the previous hidden state carries over to the current one, which they compare to the memory cell of the LSTM and to an adaptive variant of a leaky-integration unit. {#c-update-role}
Since each unit has its own gates, they expect units that capture short-term dependencies to have reset gates that are frequently active, and units that capture longer-term dependencies to have update gates that are mostly active. {#c-timescales}
Unlike the LSTM, the GRU has no output gate: Chung et al. point out that it does not have any mechanism to control the degree to which its state is exposed, but exposes the whole state each time. {#c-exposes}
Greff et al. describe it as a simplified LSTM that uses neither peephole connections nor output activation functions, and couples the input and the forget gate into an update gate. {#c-greff-gru}

### Where the reset gate goes

There are two versions of the candidate in use, and they are not the same function.
In eq. (8) of Cho et al. the reset gate multiplies the previous state before the recurrent matrix, which gives the candidate below. {#c-cho-eq8}

$$ \mathbf{n}_t = \tanh\!\big(\mathbf{x}_t W_n + \mathbf{b}_{in} + (\mathbf{r}_t \odot \mathbf{h}_{t-1}) U_n + \mathbf{b}_{hn}\big) . $$ {#eq-candidate-cho}

In the decoder of their Appendix A.1.1, the reset gate multiplies the product $U' \mathbf{h}'_{t-1}$ instead, the placement of eq-candidate. {#c-cho-decoder}
Chung et al. use eq-candidate-cho and call the placement of eq-candidate the original one, adding that in their preliminary experiments both formulations performed as well as each other. {#c-chung-footnote}
The PyTorch documentation calls eq-candidate-cho the original implementation and states that torch.nn.GRU differs on purpose for efficiency, applying the reset gate after $W_{hn} h_{(t-1)}$. {#c-torch-note}
Jozefowicz et al. and Weiss et al. write eq-candidate-cho.

The two candidates agree for every input and state if and only if $U_n$ is diagonal and $\mathbf{b}_{hn} = 0$. {#c-reset-agree}
With the reset after the matrix, the recurrent products of the three blocks do not depend on $\mathbf{r}_t$ and form a single $(B, d_h) \times (d_h, 3 d_h)$ product per step, whereas the reset before the matrix needs two products in sequence, with the same number of FLOPs. {#c-reset-cost}
That is the efficiency PyTorch refers to: on a GPU, one large product per step is cheaper to launch than two small dependent ones.
The derivation `reset-placement` proves both statements.

### The other convention for the update gate

Cho et al., Jozefowicz et al., Weiss et al. and PyTorch keep the old state when $z = 1$, as in eq-update.
Chung et al. write the update the other way round, $\mathbf{h}_t = (1 - \mathbf{z}'_t) \mathbf{h}_{t-1} + \mathbf{z}'_t \tilde{\mathbf{h}}_t$, so that $z' = 1$ replaces the state.
Since $1 - \sigma(u) = \sigma(-u)$, negating the weights and the biases of the update gate maps one convention exactly onto the other: the two describe the same family of functions. {#c-flip}

## Shapes and parameters

A layer with input width $I = d_x$ and $H = d_h$ units has

$$ P = 3 H I + 3 H^2 + 6 H $$ {#eq-params}

parameters in the form of torch.nn.GRU, whatever the length of the sequence. {#c-params}
In the shape contract the formula is

{shape:params}

and the example layer, 3 inputs and 4 units, has {shape:params@example} parameters.
Not all of them are free.
With the reset gate after the matrix, the two biases of the reset gate and the two of the update gate only enter through their sums, so the layer computes a family with $3HI + 3H^2 + 4H$ free parameters, {shape:distinct_params@example} in the example; with the reset before the matrix the two biases of the candidate are also a sum, and the count drops to that of a layer with one bias per block, $3HI + 3H^2 + 3H$. {#c-distinct-params}
An [[long-short-term-memory|LSTM]] of the same widths has four blocks, {shape:lstm_params@example} parameters with one bias per block, and both its parameters and its FLOPs are exactly $4/3$ of those of a GRU with one bias per block. {#c-lstm-ratio}

Chung et al. chose the sizes of their models so that each has approximately the same number of parameters.
For the raw speech experiments, Table 1 lists 227 GRUs, 195 LSTM units and 400 tanh units, with about 168.9, 169.1 and 168.4 thousand parameters.
Counting the recurrent layer alone on 20 input samples per step, with one bias per block and the three peephole vectors of their LSTM, gives {shape:one_bias_params@speechgru}, {shape:lstm_peephole_params@speechlstm} and {shape:tanh_params@speechtanh}, which round to those figures. {#c-chung-sizes}
The paper does not say what its counts include, and the same rule does not reproduce its figures for the music models, so this is a consistency check rather than a reading of the paper.
The derivation `parameter-and-flop-count` proves these counts.

## Computation and cost

Each step makes two matrix products, $\mathbf{x}_t W$ of $B \times I$ by $I \times 3H$ and the recurrent product of $B \times H$ by $H \times 3H$, so a forward pass over a batch of $B$ sequences of length $T$ costs

$$ C = 6 B T H I + 6 B T H^2 $$ {#eq-flops}

FLOPs, counting a multiply-add as two and leaving out the gates' nonlinearities and the elementwise products. {#c-flops}
The example layer on a batch of 2 sequences of 5 steps costs {shape:flops_forward@example} FLOPs; a layer of 1024 units on 1024 input features costs {shape:flops_forward@wide} per batch of 2 sequences of 5 steps, against {shape:lstm_flops@wide} for an LSTM of the same width.
The input products of all steps do not depend on the state and can be computed in one product of $(BT) \times I$ by $I \times 3H$ before the loop; only the recurrent product has to wait for the previous step.
The state carried from step to step is $H$ numbers per sequence, {shape:state@example} in the example, against $2H$ for an LSTM.
Training stores the gates and the candidate of every step for the backward pass, $O(BTH)$ numbers.
Like every recurrent network, the GRU generates its states one after another: Vaswani et al. write that recurrent models generate a sequence of hidden states h_t as a function of h_{t-1} and the input for position t, and that this inherently sequential nature precludes parallelization within training examples. {#c-sequential}
The [[transformer]] removed the recurrence for this reason.

## Training

A GRU is trained by [[al:backpropagation-through-time|backpropagation through time]]: the loop over steps is unrolled and differentiated like any network.
The Jacobian of one step with respect to the previous state is

$$ \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}} = \operatorname{diag}(\mathbf{z}_t) + \operatorname{diag}(1 - \mathbf{z}_t) \frac{\partial \mathbf{n}_t}{\partial \mathbf{h}_{t-1}} + \operatorname{diag}(\mathbf{h}_{t-1} - \mathbf{n}_t) \frac{\partial \mathbf{z}_t}{\partial \mathbf{h}_{t-1}} , $$ {#eq-jacobian}

for both placements of the reset gate. {#c-jacobian}
The first term is the direct path of eq-update.
When the gates and the candidate do not depend on the state ($U = 0$), the Jacobian is $\operatorname{diag}(\mathbf{z}_t)$ and the gradient that reaches a state $k$ steps back is the product of the $k$ update gates, unit by unit, with no matrix in between. {#c-copy-path}
A plain tanh network multiplies its gradient by a matrix and a squashing derivative at every step, which is the source of the [[al:vanishing-gradient-problem|vanishing gradient problem]] and of its exploding counterpart.
Chung et al. call the additive update a shortcut path that bypasses multiple temporal steps and lets the error be back-propagated easily without too quickly vanishing, if the gating unit is nearly saturated at 1. {#c-chung-shortcut}
The shortcut does not remove the other terms of eq-jacobian, which can still make gradients grow; Chung et al. train their models with RMSProp and weight noise, rescaling the norm of the gradient to 1 whenever it is larger, to prevent exploding gradients. {#c-chung-training}
That rescaling is [[gradient-clipping]].

The bias of the update gate sets the time scale of a unit at initialization.
With a constant gate $z = \sigma(b)$ and a constant candidate, the state is an exponential moving average whose weights on past candidates decay as $z^k$, with a mean lag of $z/(1 - z) = e^{b}$ steps. {#c-leaky}
A bias of 0 gives $z = 1/2$, and the direct path then halves the gradient at every step.
Jozefowicz et al. point out the same problem for the LSTM, where small random initialization effectively sets the forget gate to 0.5 and introduces a vanishing gradient with a factor of 0.5 per timestep, and they recommend a forget-gate bias of 1 or 2. {#c-joz-init}
For the GRU the analogous choice is a positive bias on the update gate in the convention of eq-update.
The derivation `update-gate-path` proves the Jacobian, the copy path and the moving average.

## Inductive biases

**Sequential processing with shared weights.** One step function is applied at every position, so the layer handles sequences of any length with the same parameters, and the order of the inputs matters.

**A bounded state.** If the state starts in $[-1, 1]^H$, every coordinate stays in the open interval $(-1, 1)$ forever, whatever the inputs and the parameters, because eq-update is a convex combination of the old state and a tanh. {#c-bounded}
The LSTM's cell has no such bound, since it is not squashed and its two gates are not tied.

**Gating as a choice between copying and rewriting.** In the limits $z \to 0$ and $r \to 1$ the GRU is the tanh recurrent network $\mathbf{h}_t = \tanh(\mathbf{x}_t W_n + \mathbf{b}_{in} + \mathbf{h}_{t-1} U_n + \mathbf{b}_{hn})$, in the limit $z \to 1$ it copies its state, and in the limit $r \to 0$ its candidate ignores the state. {#c-special-cases}
Weiss et al. use the first of these limits to argue that the GRU is at least as strong as the simple RNN (SRNN): by setting the gates of the GRU to z_t = 0 and r_t = 1 one obtains the SRNN computation. {#c-weiss-srnn}
The derivation `bounded-state` proves the bound and the limits.

**No counting.** Weiss et al. argue that finite-precision GRUs cannot implement unbounded counting on a given dimension, because the tanh of the candidate combined with the tying of z_t and 1 - z_t restricts the values of h to between -1 and 1. {#c-weiss-cannot}
An LSTM can count in a coordinate of its cell, by holding the forget gate at 1 and adding $\pm 1$ through the input gate.

## Evidence

Cho et al. report that in preliminary experiments they found it crucial to use the new unit with its gates, and could not get meaningful results with a tanh unit without any gating. {#c-cho-tanh}
Chung et al. compare a GRU network, an LSTM network and a tanh network of about equal parameter counts, trained the same way.
On four polyphonic music datasets the three performed closely to each other, and on two raw speech datasets the two gated networks clearly beat the tanh network. {#c-chung-results}
Their test negative log-probabilities (Table 2) are 6.44, 3.59 and 2.70 for the tanh, GRU and LSTM networks on Ubisoft A, and 7.62, 0.88 and 1.26 on Ubisoft B. {#c-chung-table2}
They conclude that their results are not conclusive in comparing the LSTM and the GRU, and that the choice between them may depend heavily on the dataset and the task. {#c-chung-inconclusive}

Jozefowicz et al. evaluated over ten thousand recurrent architectures in an evolutionary search, with the LSTM and the GRU as references. {#c-joz-search}
They report that the GRU beat the LSTM on all their tasks except language modelling, and that the LSTM nearly matched the GRU once its forget-gate bias was initialized to 1. {#c-joz-matched}
In Table 1, the next-step prediction accuracies of the GRU, the LSTM and the LSTM with a forget bias of 1 are 0.89565, 0.89228 and 0.90163 on their arithmetic task, and 0.45963, 0.42470 and 0.44434 on their XML task. {#c-joz-table1}
On Penn Treebank with 20M parameters and dropout, the test perplexities in Table 3 are 91.7 for the GRU, 81.4 for the LSTM and 79.83 for the LSTM with the forget bias. {#c-joz-ptb}

Weiss et al. trained 10-dimensional one-layer LSTM and GRU networks to recognize $a^n b^n$ with $n$ up to 100.
The LSTM generalized well up to n = 256, while the GRU accepted a^n b^(n+1) for n > 38 even within its training domain. {#c-weiss-anbn}
On 1000-sample test sets the LSTM's accuracy was 100% and 98.6% on a^n b^n and a^n b^n c^n, against 87.0% and 86.9% for the GRU. {#c-weiss-acc}
Greff et al., comparing LSTM variants on speech, handwriting and music, find that coupling the input and the forget gate, the change that turns an LSTM's gates into a GRU-like update gate, did not significantly change mean performance on any of their datasets. {#c-greff-cifg}
These are reported results; this page has not reproduced them.

## Limitations

- **Sequential computation.** Each step waits for the previous one, so training cannot be parallelized over the length of a sequence, and long sequences take time proportional to their length.
- **A bounded, single state.** The state is squashed into $(-1, 1)$ and is both the memory and the output, so the unit cannot hold an unbounded count and cannot hide part of its memory from the next layer.
- **Gradients through the gates.** The update gate gives a direct path, but the terms of eq-jacobian through the candidate and the gates can still make gradients explode, and clipping remains necessary.
- **Two incompatible definitions.** The two placements of the reset gate and the two conventions for $z$ coexist in papers and libraries, and weights trained with one placement compute a different function under the other.
- **No clear winner against the LSTM.** The comparisons of the Evidence section disagree from task to task, and depend on details such as the initialization of the gate biases.

## Variants and descendants

- **The minimal gated unit** and similar variants merge the reset gate into the update gate or drop it; Jozefowicz et al.'s searched architecture MUT1 computes its update gate from the input alone, z = sigm(W_xz x_t + b_z). {#c-joz-mut1}
- **Coupled-gate LSTMs (CIFG)** set the forget gate to one minus the input gate, the coupling that the GRU's update gate performs, while keeping a separate cell and an output gate.
- **Bidirectional and stacked GRUs** run one layer forwards and one backwards, or feed the states of one layer to the next; torch.nn.GRU provides both.
- **Selective state space models.** Gu and Dao prove that with a state size of 1, A = -1, B = 1, an input-dependent step size Δ = softplus(Linear(x)), their selective SSM recurrence becomes g_t = σ(Linear(x_t)) and h_t = (1 - g_t) h_{t-1} + g_t x_t, a gated update of the GRU's form. {#c-mamba-thm}
  They adopt the view that discretization of SSMs is the principled foundation of heuristic gating mechanisms. {#c-mamba-view}
  Their gate depends on the input only, which makes the recurrence linear in the state and computable by a parallel scan; see [[structured-state-space-model]].

## Notable models

The RNN Encoder-Decoder of Cho et al., which scored phrase pairs for a statistical translation system, was the first model to use the unit.
In RNNsearch, the attention-based translation model of Bahdanau et al., the recurrent units are the gated hidden unit of Cho et al., with 1000 hidden units in the decoder and in each direction of the bidirectional encoder. {#c-bahdanau}
GRU layers have since been a standard alternative to LSTM layers in speech, language and time-series models, often chosen for their smaller state and parameter count.

## Implementations

The NumPy reference `impl/numpy/gru.py` implements eq-gates, eq-candidate, eq-candidate-cho and eq-update with both placements of the reset gate, the forward pass over a sequence, backpropagation through time by hand, and the counts eq-params and eq-flops together with those of an LSTM and a tanh network.
The PyTorch version `impl/pytorch/gru.py` is an `nn.Module` with the same parameters; with `reset="after"` it matches `torch.nn.GRU` to rounding after `to_torch_gru()` copies its weights, and it is tested with `torch.autograd.gradcheck` and by training it to carry a bit across 30 steps.
The JAX version `impl/jax/gru.py` is a pure function that runs the loop with `jax.lax.scan`, tested under `jax.jit` and `jax.vmap` and with `jax.grad` against the gradients of the NumPy backward pass.
All three load the shared fixture of `fixtures/`: the example config, its states and gates for both placements, and the gradients of a fixed loss.

### Worked example

Take one unit that reads the constant input $x = 1$, with $U = 0$, candidate weight 1, update bias $b_{iz} = 2$ and every other weight and bias 0.
Then $z = \sigma(2) = 0.8808$ and $n = \tanh(1) = 0.7616$ at every step, and from $h_0 = 0$ the state is $h_t = (1 - z^t)\, n$: $h_1 = 0.1192 \times 0.7616 = 0.0908$ and $h_8 = 0.4857$.
The unit is a leaky integrator that approaches $n$ with a mean lag of $e^2 = 7.39$ steps, and the gradient of $h_{10}$ with respect to $h_0$ is $z^{10} = 0.2810$, the product of ten update gates.
The test `test_worked_example_leaky_integrator` checks these numbers in all three implementations, and the NumPy version checks the gradient through its backward pass.

For the two placements of the reset gate, take one unit with $x = h_{t-1} = 1$, input weights 1 for the reset gate and the candidate, $U_n = 1$, $b_{hn} = 1$ and everything else 0.
Then $r = \sigma(1) = 0.7311$ and $z = \sigma(0) = 1/2$.
After the matrix, $n = \tanh(1 + r (1 + 1)) = 0.9856$ and $h_t = 0.9928$; before it, $n = \tanh(1 + r + 1) = 0.9916$ and $h_t = 0.9958$.
The test `test_worked_example_two_resets` checks both.

### Pitfalls

> [!pitfall]
> **Loading weights trained with one reset placement into the other.** torch.nn.GRU and cuDNN use eq-candidate, while many papers and some other libraries use eq-candidate-cho, and the two are different functions of the same weights. The test `test_matches_torch_nn_gru` checks that the PyTorch reference with the reset after the matrix matches `torch.nn.GRU` and that the other placement differs, and `test_the_two_resets_agree_when_U_n_is_diagonal_and_b_hn_is_zero` that they agree only in the case of `reset-placement`.

> [!pitfall]
> **Reading $z$ the wrong way round.** In Chung et al. $z = 1$ replaces the state; in PyTorch it keeps it. A bias meant to make units remember makes them forget if the convention is flipped. The test `test_the_flipped_convention_is_the_same_family` checks that negating the update gate's weights and biases maps one convention onto the other.

> [!pitfall]
> **Treating the two biases as one.** With the reset after the matrix, $\mathbf{b}_{hn}$ is multiplied by the reset gate and $\mathbf{b}_{in}$ is not, so a model with one bias per block cannot be loaded by splitting the candidate bias arbitrarily; only the gate biases may be split. The check of `parameter-and-flop-count` shifts each pair of biases and compares the outputs.

> [!pitfall]
> **An update gate at one half.** With a zero bias, the direct path halves the gradient at every step, and units start with a mean lag of one step. The test `test_jacobian_without_recurrent_weights_is_diag_z` checks that the Jacobian of the direct path is $\operatorname{diag}(\mathbf{z}_t)$, so the gate bias sets how far back gradients reach.

> [!pitfall]
> **An unstable sigmoid.** A naive $1 / (1 + e^{-u})$ overflows for large negative $u$, and saturated gates are common in trained GRUs. The test `test_sigmoid_is_stable` checks the NumPy sigmoid at $\pm 800$, and `test_copy_when_update_is_one` that a saturated update gate copies the state bit for bit over 50 steps.

## Experiments

This page has no experiment yet.
The PyTorch test `test_learns_to_carry_a_bit_over_30_steps` is a small one: a GRU of 8 units trained for 300 steps of Adam learns to report, after 30 steps of noise, the sign of the first input, with an accuracy above 95% on 1000 fresh sequences.
The checks of the four derivations count parameters and products, compare the step Jacobian with finite differences, run 500 steps on inputs scaled to $10^3$ inside the box, and compare the two placements of the reset gate.

## History

Cho et al. introduced the unit in Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation, arXiv 1406.1078, in 2014, as the hidden unit of an encoder-decoder for phrase translation. {#c-hist-cho}
In December 2014, Chung et al. wrote that they refer to the unit as a gated recurrent unit (GRU), and evaluated it against the LSTM unit on polyphonic music and speech signal modeling. {#c-hist-chung}
Jozefowicz, Zaremba and Sutskever published their search over recurrent architectures, with the GRU as one of its references, in the Proceedings of the 32nd International Conference on Machine Learning, Lille, France, 2015. {#c-hist-joz}
The unit was adopted widely in translation and speech in the following years, and Weiss et al. (2018) analysed what its bounded state cannot compute.
Since 2023, selective state space models such as Mamba have revived gated recurrences whose gates depend on the input only.

## References

- K. Cho, B. van Merriënboer, C. Gulcehre, D. Bahdanau, F. Bougares, H. Schwenk and Y. Bengio. Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. arXiv 1406.1078, 2014.
- J. Chung, C. Gulcehre, K. Cho and Y. Bengio. Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling. arXiv 1412.3555, 2014.
- R. Jozefowicz, W. Zaremba and I. Sutskever. An Empirical Exploration of Recurrent Network Architectures. ICML 2015.
- G. Weiss, Y. Goldberg and E. Yahav. On the Practical Computational Power of Finite Precision RNNs for Language Recognition. ACL 2018.
- K. Greff, R. K. Srivastava, J. Koutník, B. R. Steunebrink and J. Schmidhuber. LSTM: A Search Space Odyssey. IEEE Transactions on Neural Networks and Learning Systems, 2017.
- D. Bahdanau, K. Cho and Y. Bengio. Neural Machine Translation by Jointly Learning to Align and Translate. ICLR 2015.
- A. Gu and T. Dao. Mamba: Linear-Time Sequence Modeling with Selective State Spaces. arXiv 2312.00752, 2023.
- A. Vaswani et al. Attention Is All You Need. NeurIPS 2017.
- PyTorch contributors. torch.nn.GRU. PyTorch 2.15 documentation.
