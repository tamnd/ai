---
tag: YVW8D3
kind: architecture
slug: long-short-term-memory
name: Long short-term memory
aliases: [LSTM, LSTM network]
area: deep-learning-architectures/recurrent-networks
level: core
summary: >-
  A recurrent network whose memory cells keep a state that is updated by addition and controlled by input, forget and output gates, so that gradients can pass through many time steps without vanishing.
uses: [hadamard-product@m2, matrix-multiplication@m2, activation-function@al]
related_to: [transformer, dropout, exploding-gradient-problem@al]
history:
  year: 1997
  people: [Sepp Hochreiter, Jürgen Schmidhuber]
  note: "Hochreiter and Schmidhuber (1997), Neural Computation 9(8); first a 1995 technical report. The forget gate is from Gers, Schmidhuber and Cummins (2000)."
prereqs: [backpropagation-through-time@al, vanishing-gradient-problem@al, backpropagation@al, matrix-multiplication@m2]
same_as:
  wikipedia: Long_short-term_memory
sources:
  - {paper: "Hochreiter and Schmidhuber (1997)", title: "Long Short-Term Memory", locator: "Neural Computation 9(8); authors' manuscript, Sections 1 to 6", note: "the original paper"}
  - {paper: "Gers, Schraudolph and Schmidhuber (2002)", title: "Learning Precise Timing with LSTM Recurrent Networks", locator: "JMLR 3, Sections 2 and 3", note: "the forget gate in the cell update, and peephole connections"}
  - {paper: "Greff, Srivastava, Koutník, Steunebrink and Schmidhuber (2017)", title: "LSTM: A Search Space Odyssey", locator: "IEEE TNNLS; arXiv 1503.04069v2, pp. 1 to 3, 6 and 10", note: "the vanilla LSTM, its history and a study of eight variants"}
  - {paper: "Cho, van Merriënboer, Gulcehre, Bahdanau, Bougares, Schwenk and Bengio (2014)", title: "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation", locator: "arXiv 1406.1078v3, Section 2.3", note: "the gated recurrent unit"}
  - {paper: "Pascanu, Mikolov and Bengio (2013)", title: "On the difficulty of training recurrent neural networks", locator: "arXiv 1211.5063v2, Sections 2 and 2.1", note: "vanishing and exploding gradients of plain recurrent networks"}
  - {paper: "Vaswani et al. (2017)", title: "Attention Is All You Need", locator: "Section 1", note: "the sequential computation of recurrent networks"}
  - {url: "https://docs.pytorch.org/docs/2.15/generated/torch.nn.LSTM.html", title: "torch.nn.LSTM", locator: "the equations and the parameters", note: "the order of the gates and the two bias vectors"}
---

Long short-term memory (LSTM) is a recurrent neural network built from memory cells.
Each cell keeps a state that is changed by addition at every step, and three gates decide what is written into it, what is kept and what is read out.
Hochreiter and Schmidhuber introduced it in 1997 to learn dependencies over long time lags, where plain recurrent networks fail because their gradients vanish.
The forget gate was added by Gers, Schmidhuber and Cummins (2000), and the form used today has an input, a forget and an output gate.
The [[transformer]] replaces its recurrence by attention (see Computation and cost).

## In plain words

A plain recurrent network rewrites all of its memory at every step, by passing it through a weight matrix and a squashing function.
After many steps the signal of an early input, and the gradient that should reach it during training, has been multiplied so many times that it is gone or has blown up.
An LSTM keeps a separate memory, the cell, that is not rewritten.
At each step it is scaled down a little by a forget gate and something new is added to it, under the control of an input gate.
An output gate decides how much of the memory the rest of the network sees.
Because the memory is carried forward by addition, the gradient that flows back along it is only scaled by the forget gate, and it survives for as long as the gate stays near 1.

## Overview

An LSTM layer reads a sequence $\mathbf{x}_1, \dots, \mathbf{x}_T$ of vectors of size $d_x$, one step at a time.
It carries two vectors of size $d_h$ from one step to the next: the cell state $\mathbf{c}_t$ and the hidden state $\mathbf{h}_t$, which is also the output of the step.
At step $t$, four affine maps of $\mathbf{x}_t$ and $\mathbf{h}_{t-1}$ give three gates and a candidate for the cell, the cell state is updated, and the hidden state is read from it.
Layers can be stacked, the hidden states of one layer being the inputs of the next, and a readout on $\mathbf{h}_t$ gives the prediction of each step or of the whole sequence.
In a stacked torch.nn.LSTM, the input of each layer after the first is the hidden state of the previous layer multiplied by a Bernoulli mask, which is [[dropout]] between the layers (PyTorch documentation). {#c-torch-dropout}
LSTMs have been used for difficult problems in handwriting recognition and generation, language modeling, translation and acoustic modeling of speech (Greff et al. 2017, p. 1). {#c-applications}

## Components

The notation is that of torch.nn.LSTM with one bias vector per gate: $\sigma$ is the logistic sigmoid and $\odot$ is the [[m2:hadamard-product|elementwise product]].

**Gates and candidate.**
Each of the four is an affine map of the input and the previous hidden state, followed by a nonlinearity:

$$ \begin{aligned} \mathbf{i}_t &= \sigma(\mathbf{W}_i \mathbf{x}_t + \mathbf{U}_i \mathbf{h}_{t-1} + \mathbf{b}_i), & \mathbf{f}_t &= \sigma(\mathbf{W}_f \mathbf{x}_t + \mathbf{U}_f \mathbf{h}_{t-1} + \mathbf{b}_f), \\ \mathbf{g}_t &= \tanh(\mathbf{W}_g \mathbf{x}_t + \mathbf{U}_g \mathbf{h}_{t-1} + \mathbf{b}_g), & \mathbf{o}_t &= \sigma(\mathbf{W}_o \mathbf{x}_t + \mathbf{U}_o \mathbf{h}_{t-1} + \mathbf{b}_o), \end{aligned} $$ {#eq-gates}

with $\mathbf{W}_\ast \in \mathbb{R}^{d_h \times d_x}$, $\mathbf{U}_\ast \in \mathbb{R}^{d_h \times d_h}$ and $\mathbf{b}_\ast \in \mathbb{R}^{d_h}$.
The input gate $\mathbf{i}_t$, the forget gate $\mathbf{f}_t$ and the output gate $\mathbf{o}_t$ are in $(0, 1)$ entry by entry, and the candidate $\mathbf{g}_t$ is in $(-1, 1)$.

**Cell update.**
The new cell state is the old one, scaled by the forget gate, plus the candidate, scaled by the input gate:

$$ \mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \mathbf{g}_t. $$ {#eq-cell}

**Output.**
The hidden state is the squashed cell state, scaled by the output gate:

$$ \mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t). $$ {#eq-hidden}

So every entry of $\mathbf{h}_t$ is in $(-1, 1)$, while the cell state itself is not bounded. {#c-h-bounded}
These are the equations of torch.nn.LSTM, where $\mathbf{i}_t$, $\mathbf{f}_t$, $\mathbf{g}_t$ and $\mathbf{o}_t$ are called the input, forget, cell and output gates and each gate has two biases, $b_{i\ast}$ and $b_{h\ast}$ (PyTorch documentation). {#c-torch-equations}
The vanilla LSTM of Greff et al. (2017, p. 2) also has peephole weights, which add $\mathbf{p}_i \odot \mathbf{c}_{t-1}$ to the input of the input and forget gates and $\mathbf{p}_o \odot \mathbf{c}_t$ to that of the output gate. {#c-vanilla-peepholes}

If $\mathbf{f}_t = \mathbf{1}$ and $\mathbf{i}_t = \mathbf{0}$ at every step, eq-cell gives $\mathbf{c}_t = \mathbf{c}_0$ exactly, whatever the inputs: the cell is a memory that holds its content. {#c-copy}
If $\mathbf{f}_t = \mathbf{0}$, the past is erased and $\mathbf{c}_t = \mathbf{i}_t \odot \mathbf{g}_t$.

**Worked example.**
Take $d_x = d_h = 1$, no recurrent weights ($U_\ast = 0$), no biases, and input weights $W_i = 0$, $W_f = 1$, $W_g = 1$, $W_o = 0$.
With $x_1 = 1$, $h_0 = 0$ and $c_0 = 1$, the gates are $i_1 = \sigma(0) = 0.5$, $f_1 = \sigma(1) = 0.7310585786$, $g_1 = \tanh(1) = 0.7615941560$ and $o_1 = 0.5$.
Then $c_1 = 0.7310585786 \cdot 1 + 0.5 \cdot 0.7615941560 = 1.1118556566$ and $h_1 = 0.5 \tanh(1.1118556566) = 0.4023586789$.
If the same input comes three times, the cell state is the only path from $c_0$ to $c_3$, and $\partial c_3 / \partial c_0 = f_1 f_2 f_3 = 0.7310585786^3 = 0.3907118049$.
We computed these numbers ourselves; the NumPy tests check all of them to $10^{-10}$, and the PyTorch tests check $f_1$, $c_1$ and $h_1$.

## Shapes and parameters

With the four gates stacked, the layer has the input weights $\mathbf{W} \in \mathbb{R}^{4 d_h \times d_x}$, the recurrent weights $\mathbf{U} \in \mathbb{R}^{4 d_h \times d_h}$ and the bias $\mathbf{b} \in \mathbb{R}^{4 d_h}$, so it has

$$ N = 4 d_h (d_x + d_h + 1) $$ {#eq-params}

parameters, which is {shape:params} in the axes of the table. {#c-param-count}
The count is the sum of the sizes of $\mathbf{W}$, $\mathbf{U}$ and $\mathbf{b}$, and the tests check it against the parameters of both implementations.
It does not depend on the length of the sequence, because every step uses the same weights.
For the example config, with $d_x = 3$ and $d_h = 4$, the layer has {shape:params@example} parameters.
torch.nn.LSTM keeps the two biases $b_{i\ast}$ and $b_{h\ast}$, whose sum acts as one, so it has $4 d_h$ more.
The vanilla LSTM of Greff et al. adds the $3 d_h$ peephole weights.
The state carried from one step to the next is {shape:state} numbers per sequence, $\mathbf{h}_t$ and $\mathbf{c}_t$.

The implementations store $\mathbf{W}^\top$ and $\mathbf{U}^\top$, so that a batch of row vectors is multiplied on the right, and they stack the gates in the order $\mathbf{i}, \mathbf{f}, \mathbf{g}, \mathbf{o}$ of torch.nn.LSTM.

## Computation and cost

At each step, the products $\mathbf{W}\mathbf{x}_t$ and $\mathbf{U}\mathbf{h}_{t-1}$ cost $2 \cdot 4 d_h (d_x + d_h) = 8 d_h (d_x + d_h)$ FLOPs per sequence, counting a multiply-add as 2. {#c-flops}
The activations and the elementwise products of eq-cell and eq-hidden add a number of operations proportional to $d_h$, which is left out.
For a batch of $B$ sequences of length $T$, the forward pass costs {shape:flops_forward} FLOPs, which is {shape:flops_forward@example} for the example config.
For backpropagation through time, the implementations keep the gates and the cell states of every step, $O(B T d_h)$ numbers.

The cost is linear in $T$, but the steps depend on each other: $\mathbf{h}_t$ needs $\mathbf{h}_{t-1}$.
The input products $\mathbf{W}\mathbf{x}_t$ of all steps can be computed in one matrix product, but the recurrent products cannot, and Vaswani et al. (2017, Section 1) note that this sequential nature of recurrent models precludes parallelization within training examples. {#c-sequential}
At inference a step costs the same whatever the length of the past, and the memory is the fixed state of {shape:state} numbers.

## Training

An LSTM is trained by [[al:backpropagation-through-time|backpropagation through time]] (BPTT) of a loss on its outputs.
The vanilla LSTM of Greff et al. uses full BPTT, which Graves and Schmidhuber presented for LSTM; with the full gradient, the gradients of an LSTM can be checked with finite differences (Greff et al. 2017, p. 3). {#c-full-gradient}
The original LSTM used a truncated gradient instead (see History).
Greff et al. found that the learning rate was the most crucial hyperparameter, followed by the network size, that momentum was unimportant in their setting of online gradient descent, and that the hyperparameters can be treated as approximately independent, so the learning rate can be tuned first on a fairly small network (p. 10). {#c-greff-hyperparameters}

### The gradient through the cell

Consider first a plain recurrent network $\mathbf{h}_t = \phi(\mathbf{U}\mathbf{h}_{t-1} + \mathbf{W}\mathbf{x}_t + \mathbf{b})$, with $|\phi'| \le \gamma$.

1. By the chain rule, $\partial \mathbf{h}_t / \partial \mathbf{h}_{t-1} = \mathrm{diag}(\phi'(\cdot))\, \mathbf{U}$.
2. So the gradient that reaches step $k$ from step $t$ is multiplied by the product of $t - k$ such matrices, and its norm is at most $(\gamma \lVert \mathbf{U} \rVert)^{t-k}$ times the norm at step $t$, by the submultiplicativity of the [[m2:matrix-norm|matrix norm]].
3. If $\gamma \lVert \mathbf{U} \rVert < 1$, this bound goes to 0 exponentially in $t - k$.

Pascanu et al. (2013, Section 2.1) make this argument, with $\gamma = 1$ for tanh and $\gamma = 1/4$ for the sigmoid, and conclude that long-term contributions go to 0 exponentially fast with $t - k$. {#c-rnn-vanish}
If the norm of the recurrent matrix is large instead, the long-term components can grow exponentially, which is the [[al:exploding-gradient-problem|exploding gradient problem]] that Pascanu et al. attribute to Bengio et al. (1994) (Section 2). {#c-exploding}

Now take the LSTM.
The cell state $\mathbf{c}_t$ depends on $\mathbf{c}_{t-1}$ directly through eq-cell, and indirectly through $\mathbf{h}_{t-1} = \mathbf{o}_{t-1} \odot \tanh(\mathbf{c}_{t-1})$, which enters all four gates through $\mathbf{U}$.

4. Differentiating eq-cell with the gates held fixed gives the direct term $\mathrm{diag}(\mathbf{f}_t)$, because $\mathbf{f}_t \odot \mathbf{c}_{t-1}$ is linear in $\mathbf{c}_{t-1}$ and $\mathbf{i}_t \odot \mathbf{g}_t$ does not contain it.
5. The indirect terms all pass through $\mathbf{U}$, so

$$ \frac{\partial \mathbf{c}_t}{\partial \mathbf{c}_{t-1}} = \mathrm{diag}(\mathbf{f}_t) + (\text{terms through } \mathbf{h}_{t-1} \text{ and } \mathbf{U}), $$ {#eq-cell-jacobian}

and $\partial \mathbf{c}_t / \partial \mathbf{c}_{t-1} = \mathrm{diag}(\mathbf{f}_t)$ exactly when $\mathbf{U} = 0$. {#c-cell-jacobian}
6. Along the path that stays in the cells, the gradient from step $t$ to step $k$ is multiplied by $\prod_{s=k+1}^{t} \mathrm{diag}(\mathbf{f}_s)$: a diagonal matrix whose entries are products of forget gates, with no weight matrix and no derivative of a squashing function.

So the gradient along the cell shrinks only as fast as the forget gates allow, and it is unchanged where they are 1.
With $\mathbf{f}_t = \mathbf{1}$, which is the 1997 LSTM, and with the other paths truncated, as in its training, step 6 is the identity: this is the constant error carrousel of the History section.
The numpy test `test_cell_jacobian_without_recurrence_is_diag_f` builds the Jacobian with $\mathbf{U} = 0$ from the backward pass and checks it against $\mathrm{diag}(\mathbf{f}_t)$, and the worked example above checks the product over three steps.
The other paths still contain products of $\mathbf{U}$ and of derivatives of the sigmoid, and they can vanish or explode as in a plain network.

## Inductive biases

An LSTM assumes that the data is a sequence and that the same computation applies at every step, which is the weight sharing over time of every recurrent network.
It processes the sequence in order, so its output at step $t$ depends only on the past.
Its specific bias is a memory that persists by default when the forget gate is near 1, and is changed only through gates that are functions of the input: information is kept until the network learns to erase it.
Unlike attention, it has to compress all of the past into the fixed state of {shape:state} numbers.

## Evidence

Greff et al. (2017) analysed eight LSTM variants on speech recognition, handwriting recognition and polyphonic music modeling, with the hyperparameters of each variant optimized separately by random search, in 5400 experimental runs of about 15 years of CPU time (p. 1). {#c-greff-setup}
None of the variants improved on the standard LSTM architecture significantly, and the forget gate and the output activation function were its most critical components (p. 1). {#c-greff-none-better}
Removing the output activation function or the forget gate significantly hurt performance on all three datasets (p. 6). {#c-greff-noaf-nfg}
Coupling the input and forget gates (CIFG) or removing the peephole connections (NP) simplified the LSTM without significantly decreasing performance, and both reduce the number of parameters and the computational cost (p. 10). {#c-greff-cifg-np}

## Limitations

The steps of a sequence must be computed one after the other, which limits the parallelism of training on long sequences (see Computation and cost).
The cell path of eq-cell-jacobian keeps the gradient only as far as the forget gates stay near 1, and the other paths behave like those of a plain recurrent network.
Hochreiter and Schmidhuber (1997, Section 6) note that all gradient-based approaches, LSTM among them, are practically unable to count discrete time steps precisely, so that telling whether a signal occurred 99 or 100 steps ago seems to need an additional counting mechanism. {#c-limit-counting}
Greff et al. suggest that without the output activation the cell state, which has no bound, would pass into the rest of the network and destabilize learning (p. 10). {#c-greff-unbounded}

## Variants and descendants

**Peephole connections.**
In the traditional LSTM a gate sees the cell only through its output, which is close to zero while the output gate is closed; Gers, Schraudolph and Schmidhuber (2002, Section 3) added weighted peephole connections from the cell state to the gates of the same memory block, which let all gates inspect the current cell state even when the output gate is closed. {#c-peephole}
With them, an LSTM learned to tell sequences of spikes spaced 50 time steps apart from sequences spaced 49 time steps apart (the abstract). {#c-peephole-timing}

**Gated recurrent unit.**
Cho et al. (2014, Section 2.3) proposed a hidden unit that was motivated by the LSTM unit but is much simpler to compute and implement, with a reset gate $\mathbf{r}_t$ and an update gate $\mathbf{z}_t$, and the state $\mathbf{h}_t = \mathbf{z}_t \odot \mathbf{h}_{t-1} + (1 - \mathbf{z}_t) \odot \tilde{\mathbf{h}}_t$, where $\tilde{\mathbf{h}}_t$ is computed from the input and the reset-gated previous state. {#c-gru}
Greff et al. describe it as an LSTM variant with neither peephole connections nor output activation functions, in which the input and the forget gate are coupled into an update gate (p. 3). {#c-gru-coupled}
They report that Chung et al. performed an initial comparison between the GRU and the vanilla LSTM and reported mixed results (p. 3). {#c-chung-mixed}

Other variants couple the input and forget gates, add a linear projection of the output, or let the gates receive recurrent inputs from each other; Greff et al. (2017, p. 3) review them.

## History

Hochreiter's 1991 analysis, which the 1997 paper reviews, found that error signals going backwards in time with BPTT or RTRL tend to either blow up or vanish, because the temporal evolution of the backpropagated error depends exponentially on the size of the weights (Hochreiter and Schmidhuber 1997, Section 1). {#c-blow-up-or-vanish}
With logistic sigmoid units, whose derivative is at most 0.25, the error tends to vanish whenever all the weights are smaller than 4.0 in absolute value (Section 3.1). {#c-sigmoid-decay}
The paper adds that Bengio et al. (1994) later gave a very similar analysis (Section 3.1). {#c-bengio-1994}

To keep the error constant through a self-connected unit $j$, Hochreiter and Schmidhuber require $f_j'(\mathrm{net}_j(t))\, w_{jj} = 1.0$, which they ensure with the identity activation and $w_{jj} = 1.0$, and they call this the constant error carrousel (CEC) (Section 3.2). {#c-cec}
The LSTM extends the CEC with a multiplicative input gate unit, which protects the memory contents from perturbation by irrelevant inputs, and a multiplicative output gate unit, which protects other units from perturbation by memory contents that are irrelevant at the time (Section 4). {#c-gates-1997}
The internal state of a memory cell is updated as $s_c(t) = s_c(t-1) + y^{\mathrm{in}}(t)\, g(\mathrm{net}_c(t))$ with $s_c(0) = 0$, so the old state is always kept with weight 1 (Section 4). {#c-state-1997}
Once an error signal is inside the CEC of a memory cell it can go back without ever being scaled, and it is truncated when it leaves the cell through the input gate (Section 4). {#c-truncated}
The update complexity per time step is $O(W)$ for $W$ weights, the same as BPTT for fully recurrent networks, and unlike full BPTT the LSTM is local in space and time (Section 4). {#c-complexity}
If the inputs of a cell are mostly positive or mostly negative, its internal state drifts, $h'$ becomes very small and the gradient vanishes, and the authors remedy this at the start of learning by biasing the input gate towards zero (Section 4). {#c-drift}
They also give each output gate a negative initial bias, against the abuse of memory cells as bias cells early in learning (Section 4). {#c-output-bias}
Hochreiter and Schmidhuber report that LSTM, with its gradient truncated where this does no harm, learns minimal time lags of more than 1000 discrete steps (the abstract). {#c-1000-steps}
The 1997 paper had first appeared as a 1995 technical report of the Technische Universität München, FKI-207-95 (Greff et al. 2017, p. 10). {#c-tech-report-1995}

Greff et al. (2017, p. 3) summarize the later changes.
The initial version had cells and input and output gates, but no forget gate and no peephole connections, and only the gradient of the cell was propagated back through time. {#c-history-1997}
The first modification introduced the forget gate (Gers, Schmidhuber and Cummins 2000), which enabled the LSTM to reset its own state and to learn continual tasks such as the embedded Reber grammar. {#c-forget-gate}
In the form of Gers et al. (2002, Section 2.1) the cell update becomes $s_c(t) = y^{\varphi}(t)\, s_c(t-1) + y^{\mathrm{in}}(t)\, g(z_c(t))$: the forget gate learns for how long to retain the information and, once it is outdated, erases it by resetting the cell state to zero, which prevents the cell state from growing to infinity. {#c-forget-reset}
Peephole connections followed, and Graves and Schmidhuber then gave the full BPTT gradient, which completed the vanilla LSTM.
