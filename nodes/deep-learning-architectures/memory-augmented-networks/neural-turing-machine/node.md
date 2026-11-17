---
tag: NEW-neural-turing-machine
kind: architecture
slug: neural-turing-machine
name: Neural Turing machine
aliases: [NTM, neural Turing machines, memory-augmented neural network, differentiable external memory]
area: deep-learning-architectures/memory-augmented-networks
level: core
summary: >-
  A neural network controller with a memory matrix that it reads and writes through soft weightings over the rows, found by content and by location, so the whole machine trains by gradient descent.
uses: [backpropagation@al, cross-entropy@m2, matrix-multiplication@m2]
related_to: [long-short-term-memory, multi-head-attention, transformer, multilayer-perceptron]
history:
  year: 2014
  people: [Alex Graves, Greg Wayne, Ivo Danihelka]
  note: "Graves, Wayne and Danihelka at Google DeepMind posted Neural Turing Machines as arXiv 1410.5401 in October 2014, the same month as the Memory Networks of Weston, Chopra and Bordes. The memory-augmented networks that followed include end-to-end memory networks (2015), the one-shot learner of Santoro et al. (2016) and the differentiable neural computer (2016)."
prereqs: [long-short-term-memory, multilayer-perceptron, backpropagation@al]
same_as:
  arxiv: ["1410.5401"]
sources:
  - {paper: "Graves, Wayne and Danihelka (2014)", title: "Neural Turing Machines", locator: "arXiv 1410.5401v2, the abstract, Sections 1 to 4 and Tables 1 to 3", note: "the architecture, the addressing, the five tasks and the sizes of the networks"}
  - {paper: "Collier and Beel (2018)", title: "Implementing Neural Turing Machines", locator: "arXiv 1807.08518v3, Sections 1 to 5", note: "what the original leaves unspecified, the initialization of the memory, and a stable replication"}
  - {paper: "Weston, Chopra and Bordes (2015)", title: "Memory Networks", locator: "arXiv 1410.3916v11, Section 2", note: "the parallel line of work with a memory of discrete slots"}
  - {paper: "Sukhbaatar, Szlam, Weston and Fergus (2015)", title: "End-To-End Memory Networks", locator: "arXiv 1503.08895v5, Sections 2 to 4 and Table 1", note: "a soft-attention memory trained end to end, compared with the NTM"}
  - {paper: "Santoro, Bartunov, Botvinick, Wierstra and Lillicrap (2016)", title: "One-shot Learning with Memory-Augmented Neural Networks", locator: "arXiv 1605.06065v1, Sections 3 and 4, Tables 1 and 2", note: "a content-only writer and the NTM as a meta-learner"}
family: recurrent
inputs: "a batch of sequences as a (B, T, X) array"
outputs: "logits as a (B, T, Y) array, one output vector per step"
components: []
inductive_biases: ["a separation of computation and storage: the controller's parameters do not grow with the memory, which is a fixed-size matrix of N rows of width M", "sparse, local interaction with the memory: each head reads or writes a few rows, so stored items interfere little", "two ways of finding a row: by its content, through a cosine similarity to a key, and by its position relative to the last row used, through a circular shift"]
params: "H (X + R M) + H + P_h (H + 1) + Y (H + R M) + Y for a one-layer feedforward controller, with P_h = R (M + 3 + K) + W (3 M + 3 + K) head outputs and K = 2 S + 1 shifts; independent of the number N of memory rows"
flops_per_example: "2 T [H (X + R M) + P_h H + Y (H + R M) + (R + W) N (M + K) + R N M], counting a multiply-add as 2 FLOPs and leaving out the elementwise work"
---

A neural Turing machine, or NTM, is a recurrent network whose controller, a feedforward network or an LSTM, reads from and writes to an external memory matrix of $N$ rows of width $M$.
The reads and writes are blurry: each head holds a weighting, a probability distribution over the rows, and reads a weighted average of them or erases and adds to all of them in proportion to their weights.
The weighting is found by comparing a key with the content of every row, then optionally replaced by the previous weighting, rotated by a few positions and sharpened, so a head can both look an item up and walk along the memory one row at a time.
Every step is differentiable, so the controller learns to use the memory by gradient descent on input and output examples alone.
Graves, Wayne and Danihelka introduced the model in 2014 and trained it to copy, repeat, recall and sort sequences, generalizing to sequences longer than those seen in training where an LSTM did not.

## In plain words

An ordinary recurrent network keeps everything it knows about the past in one vector of numbers, which it rewrites at every step.
That vector has to hold the data and the program at the same time, and it is easy to overwrite something that is needed later.
A neural Turing machine gives the network a separate notebook: a table of rows, each a short list of numbers.
At each step the network decides where to look and what to write, and the notebook keeps what it is told until it is changed.
A real computer finds a row by its address; the network cannot, because a jump to one address has no gradient.
Instead it spreads its attention over all the rows, mostly on one or two, and reads a weighted average.
It can find a row by asking which row looks most like a given pattern, or by moving its attention one row forward or back from where it was, and it can combine the two.
With this, a network learns to copy a sequence by writing each item to the next row and then reading the rows back in order, and because the procedure does not depend on the length, it keeps working on sequences longer than any it was trained on.

## Overview

Graves et al. report preliminary results in which such a network infers simple algorithms such as copying, sorting, and associative recall from input and output examples. {#c-graves-infer-algorithms}
They chose the name by analogy to Turing's enrichment of finite-state machines by an infinite memory tape. {#c-graves-turing-analogy}
Unlike the tape, the memory of the model is finite, and unlike a Turing machine, the whole machine is differentiable.
The memory is a matrix $\mathbf M_t$ of shape $(N, M)$, and the heads interact with it through what Graves et al. call blurry read and write operations, which interact to a greater or lesser degree with all the elements in memory. {#c-blurry-ops}

A head at step $t$ emits a weighting $w_t$ over the rows, with $w_t(i) \ge 0$ and $\sum_i w_t(i) = 1$.
A read head returns the convex combination of the rows

$$ \mathbf r_t = \sum_{i=1}^{N} w_t(i)\, \mathbf M_t(i) , $$ {#eq-read}

which is differentiable in the weighting and in the memory. {#c-read-convex}
A write head emits an erase vector $\mathbf e_t$ with entries in $(0, 1)$ and an add vector $\mathbf a_t$, and changes every row in proportion to its weight, first erasing,

$$ \tilde{\mathbf M}_t(i) = \mathbf M_{t-1}(i) \odot \big[\mathbf 1 - w_t(i)\, \mathbf e_t\big] , $$ {#eq-erase}

then adding,

$$ \mathbf M_t(i) = \tilde{\mathbf M}_t(i) + w_t(i)\, \mathbf a_t . $$ {#eq-add}

Graves et al. took the split into an erase and an add from the input and forget gates of the LSTM. {#c-write-from-lstm}
An entry is reset to zero only if both the weight of its row and the matching entry of the erase vector are one, and then the add replaces the row. {#c-reset-needs-both}
With several write heads, all the erases act first and then all the adds, and the result does not depend on the order of the heads, since the erases multiply and the adds sum. {#c-write-commutes}
A weighting spread over several rows erases each of them only in part: with weight $p$ and a full erase, a row keeps the fraction $1 - p$ of its old value. {#c-blurry-erase}

The rest of the machine is the addressing, the four steps that turn the controller's outputs into a weighting, and the controller itself.
The page derives that each step keeps a weighting a weighting, describes what the steps can and cannot express, counts the parameters and the cost, and states what the original and the later experiments measured.

## Components

### Content addressing

A head emits a key $\mathbf k_t$ of width $M$ and a key strength $\beta_t \ge 0$.
It compares the key with every row by the cosine similarity

$$ K[\mathbf u, \mathbf v] = \frac{\mathbf u \cdot \mathbf v}{\lVert \mathbf u \rVert\, \lVert \mathbf v \rVert} , $$ {#eq-cosine}

and turns the similarities into a weighting with a softmax at inverse temperature $\beta_t$:

$$ w^c_t(i) = \frac{\exp\big(\beta_t K[\mathbf k_t, \mathbf M_t(i)]\big)}{\sum_j \exp\big(\beta_t K[\mathbf k_t, \mathbf M_t(j)]\big)} . $$ {#eq-content}

With $\beta_t = 0$ the content weighting is uniform over the rows, and as $\beta_t$ grows it puts all its weight, in equal parts, on the rows most similar to the key. {#c-content-beta-limits}
Graves et al. call content addressing strictly more general than location addressing, since the content of a row could include location information inside it, but found that location addressing proved essential for some forms of generalisation, so the NTM uses both. {#c-content-more-general}

### Interpolation

A scalar gate $g_t \in (0, 1)$ blends the content weighting with the head's weighting of the previous step:

$$ w^g_t = g_t\, w^c_t + (1 - g_t)\, w_{t-1} . $$ {#eq-interpolate}

With $g_t = 1$ the head ignores where it was and uses the content weighting; with $g_t = 0$ it ignores the content and keeps its previous weighting, which the shift then moves. {#c-gate-extremes}

### Convolutional shift

A shift weighting $s_t$ over the integer shifts $-S, \dots, S$, a softmax output of the controller, rotates the weighting by a circular convolution, the indices taken modulo $N$:

$$ \tilde w_t(i) = \sum_{j=0}^{N-1} w^g_t(j)\, s_t(i - j) . $$ {#eq-shift}

When $s_t$ puts all its weight on one shift $d$, the shift moves the weighting $d$ rows forward, wrapping from the last row to the first; a mixture of shifts is the same mixture of the rotated weightings. {#c-shift-is-roll}
Graves et al. give the example of the shifts $-1$, $0$ and $1$, a softmax of three elements, and also experimented with a single scalar read as the lower bound of a uniform distribution of width one, so that 6.7 puts weight 0.3 on the shift 6 and 0.7 on the shift 7. {#c-graves-shift-scalar}
They warn that the convolution can cause leakage or dispersion of weightings over time if the shift weighting is not sharp. {#c-graves-dispersion}
The rate is simple to state: without the next step, a shift weighting with variance $\sigma^2$ spreads a weighting that starts on one row to a variance of $k \sigma^2$ after $k$ steps, $0.2\,k$ for the weights $0.1$, $0.8$ and $0.1$ on the shifts $-1$, $0$ and $1$. {#c-blur-spreads}

### Sharpening

A last scalar $\gamma_t \ge 1$ sharpens the weighting:

$$ w_t(i) = \frac{\tilde w_t(i)^{\gamma_t}}{\sum_j \tilde w_t(j)^{\gamma_t}} . $$ {#eq-sharpen}

At $\gamma_t = 1$ the weighting is unchanged; a larger $\gamma_t$ keeps the order of the weights and moves weight towards the largest, and as $\gamma_t \to \infty$ the weighting becomes uniform on the rows of the largest weight. {#c-sharpen-limits}
Each of the four steps maps weightings to weightings, so whatever the controller emits within the ranges of its nonlinearities, the head's weighting is nonnegative and sums to one. {#c-weighting-stays-normalized}
The derivation `addressing-keeps-a-weighting` proves this and the limits of $\beta$, and `shift-and-sharpen` proves the statements about the gate, the shift, the dispersion and the sharpening; their checks run them on the NumPy reference.

Graves et al. describe three modes that the combined mechanism supports: a content weighting used as it is; a content weighting chosen and then shifted, which finds an item and then the one after it; and a previous weighting rotated without any content, which steps through the memory. {#c-three-modes}

### The controller

At each step the controller receives the input $\mathbf x_t$ and the read vectors of the previous step and emits, for each head, a key, $\beta$, $g$, $s$ and $\gamma$, and for each write head also $\mathbf e$ and $\mathbf a$, together with the output of the step.
Collier and Beel point out that the similarity measure, the initial weightings, the initial memory, the nonlinearities applied to the head outputs and the initial read vector are all undefined in the specification of the NTM. {#c-collier-unspecified}
The references on this page follow the choices of Collier and Beel: the cosine similarity, a controller fed the input together with the previously read vectors, a tanh for the key and the add vector, a sigmoid for the erase vector and the gate, a softplus for $\beta$, a softmax for $s$, and a softplus plus 1 for $\gamma$, each of which keeps its output in its range. {#c-collier-choices}

The controller may be a feedforward network, such as a [[multilayer-perceptron|multilayer perceptron]], or a recurrent one, such as an [[long-short-term-memory|LSTM]]; unlike a plain multilayer perceptron, the NTM keeps a state from step to step in its memory even when its controller has none.
Graves et al. note that a feedforward controller can mimic a recurrent network by reading and writing at the same location in memory at every step, and that it makes the network's use of the memory more transparent. {#c-ff-mimics-rnn}
Its weakness is the number of heads: with a single read head it can perform only a unary transform on a single memory vector at each step, with two only a binary one, while a recurrent controller can store read vectors internally. {#c-ff-heads-limit}
The references use a feedforward controller with one tanh hidden layer of width $H$ and a linear output read from the hidden layer and the new read vectors.

### One step

The order of the operations within a step is not fixed by the paper; the references take the one below, as do most implementations.

1. The controller computes its hidden layer from $\mathbf x_t$ and $\mathbf r_{t-1}$ and the head outputs from the hidden layer.
2. Each write head addresses $\mathbf M_{t-1}$ and the heads write, all erases then all adds, giving $\mathbf M_t$.
3. Each read head addresses $\mathbf M_t$ and reads $\mathbf r_t$.
4. The output layer maps the hidden layer and $\mathbf r_t$ to the logits of the step.

Because the writes come before the reads, a read in a step sees what was written in the same step.

## Shapes and parameters

Write $X$ and $Y$ for the input and output widths, $H$ for the controller's width, $N$ and $M$ for the rows and the width of the memory, $R$ and $W$ for the numbers of read and write heads, and $K = 2S + 1$ for the number of shifts.
A read head takes $M + 3 + K$ outputs of the controller, the key, $\beta$, $g$, $\gamma$ and $s$, and a write head $3M + 3 + K$ with the erase and add vectors.
With $P_h = R(M + 3 + K) + W(3M + 3 + K)$ head outputs, the model has

$$ P = H(X + RM) + H + P_h (H + 1) + Y(H + RM) + Y $$ {#eq-params}

parameters: {shape:params}. {#c-params}
The example config, 4 inputs, 3 outputs, 8 hidden units, 6 rows of width 4, one head of each kind and $S = 1$, has {shape:params@example} parameters; the config `wide`, 9 rows of width 6 and two read heads, has {shape:params@wide}.

None of the parameters depends on $N$: a larger memory costs compute and storage, $BNM$ numbers per batch, {shape:memory_floats@example} for the example, but no parameters. {#c-params-independent-of-n}
Graves et al. point out this property of the architecture. {#c-graves-params-independent}
It holds only while the initial memory is fixed: learning the initial memory, read vectors and weightings, as some implementations do, adds $NM + RM + (R + W) N$ parameters, {shape:learned_init_params@paper} at the sizes of the copy task. {#c-learned-init-grows}

For those sizes, 9 inputs, 8 outputs, 100 hidden units and 128 rows of width 20, the config `paper`, eq-params gives {shape:params@paper}.
Graves et al. report 17,162 for their feedforward copy network, with one head, a controller of 100 units and a memory of $128 \times 20$, without saying how the count is made. {#c-graves-table1}
The difference comes from choices the paper leaves open, such as which vectors feed the output layer and how the head outputs are parametrized, so the two counts should not be compared closely.

## Computation and cost

Counting a multiply-add as 2 FLOPs and leaving out the elementwise work, a forward pass over $T$ steps of a batch of $B$ costs

$$ C_{\text{fwd}} = 2BT\,\big[\,H(X + RM) + P_h H + Y(H + RM) + (R + W)\, N (M + K) + R N M\,\big] $$ {#eq-flops}

FLOPs: the controller, the output layer, the cosine similarities and the shift for each head, and the reads. {#c-flops}
The derivation `parameter-and-flop-count` counts the arrays and the multiply-adds of the NumPy reference.
For the example config and a batch of 2 sequences of 7 steps that is {shape:flops_forward@example} FLOPs, and at the sizes of the copy task {shape:flops_forward@paper}.
The shift costs $NK$ per head, not the $N^2$ of a general circular convolution, because only $K$ shifts have weight.
The memory term grows linearly in $N$ and the controller term does not depend on it, so at $N = 128$ the memory accounts for about two fifths of the forward FLOPs of the copy network.

The machine is a recurrence: step $t$ needs the memory of step $t - 1$, so the steps of a sequence cannot run in parallel as the positions of a [[transformer]] can.
Training stores the memory of every step for the backward pass, $BTNM$ numbers, which for long sequences and large memories dominates the activations.

## Training

The loss is the cross-entropy of the targets under logistic outputs, summed over the steps that have targets and reported in bits per sequence.
Graves et al. trained every network with RMSProp with a momentum of 0.9, and clipped the gradient componentwise to the range $(-10, 10)$. {#c-graves-training}
Collier and Beel found that the open-source implementations they examined were unstable during training or failed to replicate the reported performance. {#c-collier-unstable}
Several of those implementations report that their gradients sometimes become NaN during training, causing training to fail. {#c-collier-nan}
Their stable implementation trains with Adam at a learning rate of 0.001, clips the norm of the gradient at 50 and clips the controller's outputs to $(-20, 20)$. {#c-collier-training}

The initial state matters more than its size suggests.
Collier and Beel compared three ways to set the initial memory: a small constant, $10^{-6}$ in every entry; learned values; and random values from a truncated normal. {#c-collier-init-schemes}
They report that the constant converged on average 2 times faster than the next best scheme, and that on the copy task it was 3.5 times faster than learning the memory, while the random memory failed to train. {#c-collier-constant-init}
The references start the memory at that constant, every weighting on the first row and the read vectors at zero, which costs no parameters.

## Inductive biases

The architecture separates the storage from the program.
The controller's parameters encode what to do and the memory holds the data, so the same weights apply to any number of items, as long as the memory has room for them.
This is what makes length generalization possible: a copy network that writes each item to the next row and reads the rows back does the same thing at every step, whatever the length.

Graves et al. argue that because the interaction with the memory is highly sparse, the NTM is biased towards storing data without interference. {#c-graves-sparse-bias}
The softmax of the content addressing and the sharpening push each weighting towards a few rows, so a write changes those rows and leaves the others nearly as they were.

The addressing builds in two notions of where an item is, its content and its position relative to the last one used, and the circular shift builds in the order of the rows.
The order is the reason the NTM can learn procedures that walk through a sequence; it is also a constraint, since the memory is a ring and a head that moves past the last row comes back to the first.

## Evidence

Graves et al. compared three architectures on each task: an NTM with a feedforward controller, an NTM with an LSTM controller, and an LSTM network without memory, with three stacked layers. {#c-graves-architectures}

- **Copy.** On sequences of 8-bit vectors of lengths 1 to 20, the NTM learned much faster than the LSTM and converged to a lower cost, and it went on copying sequences longer than those of training, while the LSTM degraded beyond length 20. {#c-copy-result}
  The weightings the trained network used show the procedure: content addressing jumps to the start of the stored sequence, and location addressing moves along it. {#c-copy-procedure}
- **Repeat copy.** Asked to copy a sequence a given number of times, the NTM generalized to longer sequences and to more repetitions, but did not predict the end marker correctly beyond the training range. {#c-repeat-copy-result}
- **Associative recall.** Given items of three six-bit vectors and a query item, the network must return the item after the query. {#c-assoc-task}
  The NTM reached near zero cost within about 30,000 episodes, where the LSTM did not reach zero cost after a million, and the feedforward controller learned faster than the LSTM controller. {#c-assoc-result}
  The learned solution writes a compressed form of each item at its delimiter and finds the answer by a content lookup shifted by one row, the second mode of the addressing. {#c-assoc-procedure}
- **Dynamic N-grams.** On binary sequences from random 6-gram distributions, where the optimal estimator counts the contexts, the NTM had a small but significant advantage over the LSTM but never reached the optimal cost. {#c-ngram-result}
- **Priority sort.** Asked to sort 20 vectors by priorities and return the 16 of highest priority, the NTM wrote each vector at a location that is nearly a linear function of its priority and then read the locations in order, and the feedforward controller needed eight parallel read and write heads. {#c-sort-result}

Collier and Beel trained their implementation, with one read and one write head, on the copy, repeat copy and associative recall tasks, with ten runs of each configuration. {#c-collier-replication}
In their comparison the NTM learned the copy task 1.2 times slower than the differentiable neural computer, and both memory-augmented networks learned it 4 to 5 times faster than an LSTM. {#c-collier-vs-dnc}

Sukhbaatar et al. describe their memory as somewhat simpler than the NTM, in part because it always writes each item to memory sequentially, so it does not need operations like sharpening. {#c-memn2n-simpler}
On the bAbI question-answering tasks with 1,000 training examples per task, their best end-to-end model had a mean error of 12.6%, against 6.7% for memory networks trained with supervision of which facts support each answer. {#c-memn2n-babi}

Santoro et al. used the memory for one-shot learning. They replaced the NTM's writer with a purely content-based writer that writes to the least or the most recently used row, since location-based access is not optimal for tasks that emphasize a conjunctive coding of information independent of sequence. {#c-lrua}
With 15 Omniglot classes per episode, their writer classified 62.6% of second instances of a class and 95.3% of tenth instances correctly, against 35.4% and 88.4% for the NTM writer and 2.2% and 12.7% for an LSTM. {#c-santoro-omniglot}

## Limitations

- **Hard to train.** Several open implementations are unstable or fail to reach the published results, and the sizes, nonlinearities and initial state the paper leaves open change the speed and the success of training.
- **Cost grows with the memory.** Every head compares its key with every row and every write touches every row, so the cost per step is linear in $N$, and backpropagation stores the memory at every step.
- **Sequential.** The recurrence over the memory prevents parallel training across the steps of a sequence.
- **A ring of fixed size.** Location addressing moves on a circle of $N$ rows: a sequence longer than the memory wraps and overwrites its start, and there is no notion of freeing a row once used.
- **Limited generalization.** Even on the synthetic tasks, the learned procedures drift: the figures of Graves et al. include a copy of length 120 with a single vector duplicated, and the repeat copy network miscounting beyond the training range. {#c-graves-failures}
- **Synthetic tasks.** The original experiments are algorithmic tasks on short binary vectors; the memory-augmented networks that came after were tested mostly on question answering and few-shot classification.

## Variants and descendants

- **Memory networks.** Weston et al. proposed, at the same time, a memory of discrete slots with four learned components, an input feature map, a generalization step that updates the memory, an output map and a response; in the simplest form the generalization step stores each input in a new slot. {#c-memnn-components}
- **End-to-end memory networks.** Sukhbaatar et al. made the memory network differentiable with a soft attention over the slots, trained end to end and hence with significantly less supervision, and stacked several reads, or hops. {#c-memn2n-end-to-end}
- **Memory-augmented networks for meta-learning.** Santoro et al. kept the NTM's reading and replaced its writing with least-recently-used access.
- **The differentiable neural computer.** Graves et al. extended the NTM in 2016 with a usage-based allocation of free rows and a matrix that records the order of writes, replacing the circular shift.
- **Attention.** The content addressing of eq-content is a soft attention with a cosine score and a learned temperature; the [[multi-head-attention]] of the transformer uses the same softmax over scores, with a scaled dot product and without a recurrent memory.

## Notable models

The NTM is a research architecture rather than a family of deployed models.
Its sizes in the original are small: 17,162 parameters for the feedforward copy network and 508,305 for the feedforward priority sort network with eight heads, {#c-graves-sizes} against 1,352,969 for the copy LSTM of three layers of 256 cells. {#c-graves-lstm-size}

## Implementations

The NumPy reference `impl/numpy/ntm.py` implements the cosine of eq-cosine with a floor and an epsilon, the four steps of the addressing (eq-content, eq-interpolate, eq-shift, eq-sharpen), the read of eq-read, the write of eq-erase and eq-add for any number of heads, the split of the controller's outputs into head parameters with their nonlinearities, the step and the forward pass over a sequence, the loss in bits per sequence, and the counts eq-params and eq-flops.
The PyTorch version `impl/pytorch/ntm.py` follows it step for step on a dict of tensors, and its tests check the gradients with `torch.autograd.gradcheck` and train it on a small copy task.
The JAX version `impl/jax/ntm.py` runs the steps in `jax.lax.scan` with the memory, the weightings and the read vectors as the carry, and is tested under `jax.jit` and `jax.vmap` and against finite differences.
All three are tested against the shared fixture of `fixtures/`: the example config, a batch of 2 copy sequences of 3 items, the logits, the weightings of every head at every step, the last memory and the loss.

### Worked example

Take the example config: 4 inputs, 3 outputs, 8 hidden units, a memory of 6 rows of width 4, one read head, one write head and the shifts $-1$, $0$ and $1$.
The controller reads $4 + 4 = 8$ numbers, the input and the previous read vector, and has $8 \cdot 8 + 8 = 72$ parameters.
A read head takes $4 + 3 + 3 = 10$ outputs: a key of 4, $\beta$, $g$, $\gamma$ and 3 shift weights; the write head takes 8 more for its erase and add vectors, 18 in all, so the head layer has $28 \cdot 9 = 252$ parameters.
The output layer reads the 8 hidden units and the 4 numbers of the new read vector and has $3 \cdot 12 + 3 = 39$, so the network has $72 + 252 + 39 = 363$, as eq-params says, and none of them would change with 600 rows instead of 6.

Now follow a head that copies.
Start the write head on row 4 of 5 rows with $g = 0$, all of $s$ on the shift $+1$, $\gamma = 1$ and a full erase.
At the first step eq-shift moves its weighting to row 0, eq-erase clears row 0 and eq-add writes the first item there; at the next step the weighting moves to row 1, and so on.
A read head started on the same row and moved the same way returns the items in order.
With 7 items and 5 rows the sixth item lands on row 0 again and overwrites the first, so the read returns items 6, 7, 3, 4 and 5: the ring of the limitations.
The test `test_copy_by_location` runs this, and `test_content_then_shift_finds_the_next_item` runs the second mode: a content lookup of row 2 with $g = 1$ and a large $\beta$, shifted by one, reads row 3.

### Pitfalls

> [!pitfall]
> **A blank row and the cosine.** The cosine of eq-cosine is $0/0$ for a row of zeros, and a memory initialized at zero has only such rows. An epsilon in the denominator makes the value 0 and the content weighting uniform, which the test `test_zero_memory_needs_the_epsilon` checks. The gradient needs more: the derivative of $\sqrt{u}$ at $u = 0$ is infinite, and times the zero of the inner derivative it is NaN, which then spreads to every parameter. The references take the square root of $\max(\lVert \mathbf v \rVert^2, 10^{-30})$; `test_cosine_gradient_needs_the_floor` shows the naive form giving NaN and the floored one finite.

> [!pitfall]
> **A blurry shift without sharpening.** A shift weighting of 0.1, 0.8 and 0.1 looks nearly one-hot, but applied at every step it spreads the weighting by a variance of 0.2 per step, and after 20 steps the weighting has a standard deviation of 2 rows. The test `test_blurry_shift_disperses_and_sharpening_refocuses` shows the spread and the sharpening that undoes it; $\gamma$ must be allowed to exceed 1, which $1 + \operatorname{softplus}$ ensures.

> [!pitfall]
> **The sign of $\beta$.** With a negative key strength the softmax of eq-content puts its weight on the row least like the key, the opposite of a lookup. The test `test_a_negative_beta_attends_to_the_least_similar_row` checks this; a softplus or an exponential keeps $\beta \ge 0$.

> [!pitfall]
> **The direction of the shift.** In eq-shift the weight $s(d)$ moves the weighting from row $j$ to row $j + d$, the direction of `np.roll(w, d)`. Implementations that index the convolution the other way move the heads backwards, which a trained model absorbs but which breaks a comparison of weightings between implementations; `test_shift_direction_and_wrap` pins the direction and the wrap.

> [!pitfall]
> **The order of writes and reads.** Whether a read in a step sees that step's write changes the function the network computes. The references write first; `test_write_then_read_in_one_step` checks that the read vector of a step is the read of the memory after the write. Mixing the two orders between implementations gives different results from the same weights.

## Experiments

This page has no experiment yet. The checks of its four derivations play the part of small experiments: they test that each step of the addressing keeps a weighting normalized on random inputs, run the gate, the shift and the sharpening at their extremes, measure the spread of a blurry shift over 20 steps against $k\sigma^2$, check that several write heads commute, and count the parameters and multiply-adds of the reference. The PyTorch tests train a small NTM on a copy task of 3 items of 3 bits and check that the loss on held-out sequences falls below half its starting value.

## History

Graves, Wayne and Danihelka posted Neural Turing Machines in October 2014. They drew on the study of working memory in psychology and neuroscience and on the need to apply rules to rapidly-created variables, and noted that recurrent networks are known to be Turing-complete. {#c-graves-motivation}
They also recalled the argument of Fodor and Pylyshyn that connectionist theories were incapable of variable-binding, the assignment of a datum to a slot in a data structure. {#c-graves-fodor}
In the same month Weston, Chopra and Bordes posted Memory Networks, which store each input in a slot of a memory and learn to retrieve the relevant slots, at first with supervision of which slots support each answer.
Sukhbaatar et al. removed that supervision in 2015 with end-to-end memory networks, and Santoro et al. used a memory-augmented network as a one-shot learner in 2016.
In 2016 Graves et al. published the differentiable neural computer, which added the allocation of free memory and a record of the order of writes.
Collier and Beel revisited the original in 2018, listed the details it leaves unspecified, and published an implementation that trains reliably.
The content-based reading of these models is a soft attention over a set of stored vectors, the operation that the [[transformer]] made the core of its architecture in 2017.

## References

- A. Graves, G. Wayne and I. Danihelka. Neural Turing Machines. arXiv 1410.5401, 2014.
- J. Weston, S. Chopra and A. Bordes. Memory Networks. ICLR 2015.
- S. Sukhbaatar, A. Szlam, J. Weston and R. Fergus. End-To-End Memory Networks. arXiv 1503.08895, 2015.
- A. Santoro, S. Bartunov, M. Botvinick, D. Wierstra and T. Lillicrap. One-shot Learning with Memory-Augmented Neural Networks. arXiv 1605.06065, 2016.
- M. Collier and J. Beel. Implementing Neural Turing Machines. arXiv 1807.08518, 2018.
