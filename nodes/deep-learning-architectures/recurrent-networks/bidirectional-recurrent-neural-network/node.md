---
tag: Z7Z0J3
kind: architecture
slug: bidirectional-recurrent-neural-network
name: Bidirectional recurrent neural network
aliases: [BRNN, BiRNN, bidirectional RNN, BLSTM, BiLSTM, bidirectional LSTM, deep bidirectional LSTM]
area: deep-learning-architectures/recurrent-networks
level: core
summary: >-
  A recurrent network with two sets of states per layer, one run forwards and one backwards over the input, whose concatenation lets the output at every step see the whole sequence.
uses: [matrix-multiplication@m2, cross-entropy@m2, softmax-regression@al]
related_to: [recurrent-neural-network, long-short-term-memory, gated-recurrent-unit, sequence-to-sequence-model, transformer, bert, conditional-random-field@al, hidden-markov-model@al]
history:
  year: 1997
  people: [Mike Schuster, Kuldip K. Paliwal]
  note: "Schuster and Paliwal introduced the BRNN in IEEE Transactions on Signal Processing in November 1997; Graves and Schmidhuber combined it with the LSTM in 2005, and Graves, Mohamed and Hinton stacked bidirectional LSTM layers for speech recognition in 2013."
prereqs: [recurrent-neural-network, long-short-term-memory, backpropagation-through-time@al]
same_as:
  wikipedia: Bidirectional_recurrent_neural_networks
  doi: ["10.1109/78.650093"]
sources:
  - {paper: "Schuster and Paliwal (1997)", title: "Bidirectional Recurrent Neural Networks", locator: "IEEE Transactions on Signal Processing 45(11), the abstract and Sections II and IV", note: "the architecture, its training, and the comparison with delayed and merged networks on TIMIT"}
  - {paper: "Graves, Mohamed and Hinton (2013)", title: "Speech Recognition with Deep Recurrent Neural Networks", locator: "arXiv 1303.5778v1, Sections 2 and 4, Table 1", note: "the deep bidirectional LSTM and its comparison with a unidirectional one of the same size"}
  - {paper: "Huang, Xu and Yu (2015)", title: "Bidirectional LSTM-CRF Models for Sequence Tagging", locator: "arXiv 1508.01991v1, Section 2.2 and Table 2", note: "bidirectional LSTM taggers for part-of-speech tagging, chunking and named entity recognition"}
  - {paper: "Lipton, Berkowitz and Elkan (2015)", title: "A Critical Review of Recurrent Neural Networks for Sequence Learning", locator: "arXiv 1506.00019, Sections 4.2 and 5.5", note: "the three equations, and why the network cannot run online"}
  - {paper: "Bahdanau, Cho and Bengio (2015)", title: "Neural Machine Translation by Jointly Learning to Align and Translate", locator: "arXiv 1409.0473v7, Section 3.2", note: "the bidirectional encoder of attention-based translation"}
  - {paper: "Devlin, Chang, Lee and Toutanova (2019)", title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", locator: "the abstract and p. 2", note: "bidirectional context without recurrence"}
  - {url: "https://docs.pytorch.org/docs/2.15/generated/torch.nn.LSTM.html", title: "torch.nn.LSTM", locator: "the parameters, the outputs and the notes", note: "bidirectional=True, the _reverse weights, the layout of the output and h_n"}
family: recurrent
inputs: "a batch of sequences as a (B, T, I) array with the length of each sequence, all of which is known before the first output"
outputs: "a (B, T, K) array of class scores, one for every step, each a function of the whole input sequence"
components: [long-short-term-memory]
inductive_biases: ["sequential order in both directions: one step function per direction, shared over time", "every output sees the whole sequence, the past through the forward states and the future through the backward states", "a recency bias in each direction: a state represents its recent inputs best, so the concatenation at step t is focused on the inputs around t", "no interaction between the two directions inside a layer: they meet only in the layer above or in the output"]
params: "D [4 d_h (d_x + d_h) + 4 d_h] + (L - 1) D [4 d_h (D d_h + d_h) + 4 d_h] + D d_h K + K for L layers of D = 2 directions of d_h LSTM cells and K classes"
flops_per_example: "8 D T d_h (d_x + d_h) + 8 (L - 1) D T d_h (D d_h + d_h) + 2 T D d_h K for a sequence of T steps, counting a multiply-add as 2 FLOPs and only the products"
---

A bidirectional recurrent neural network (BRNN) is a [[recurrent-neural-network|recurrent network]] in which each layer has two sets of states: one reads the input from the first step to the last, the other from the last step to the first.
The output at a step is computed from both, so it depends on everything before the step and everything after it.
Schuster and Paliwal introduced the BRNN in 1997 as a network that can be trained without the limitation of using input information just up to a preset future frame, by training it simultaneously in the positive and negative time directions. {#c-def}
Built from [[long-short-term-memory|LSTM]] cells and stacked several layers deep, it became the bidirectional LSTM (BLSTM or BiLSTM), the standard recurrent model for speech recognition, sequence tagging and the encoders of translation models in the decade before the [[transformer]].
The price of seeing the future is that no output can be produced before the whole input has been read.

## In plain words

A forward recurrent network reads a sentence the way a person speaks it, one word at a time, and has to decide what each word is before it has heard the rest.
For many tasks that is a needless handicap.
Whether "lead" is a verb or a metal, or whether a sound is the start of "cat" or "cap", often depends on what comes next, and when the whole recording or sentence is already available, there is no reason to ignore it.
A bidirectional network runs two readers over the same input, one left to right and one right to left, each keeping its own running summary.
At every position it puts the two summaries side by side: the left reader's summary of everything up to here, and the right reader's summary of everything from here on.
Whatever is predicted at that position is computed from both, so it can use the context on either side.

## Overview

Schuster and Paliwal describe the idea as splitting the state neurons of a regular RNN into a part responsible for the positive time direction (forward states) and a part for the negative time direction (backward states), with no connections from the outputs of the forward states to the inputs of the backward states, or the other way round. {#c-split}
Graves, Mohamed and Hinton put it the same way: a bidirectional RNN processes the data in both directions with two separate hidden layers, which are then fed forwards to the same output layer, the backward layer being iterated from t = T to 1 and the forward layer from t = 1 to T. {#c-graves-def}
In a deep BRNN the layers are stacked, and every hidden layer receives input from both the forward and the backward layer at the level below; with LSTM cells this is the deep bidirectional LSTM. {#c-deep}

The network on this page is that deep bidirectional LSTM with an output at every step, for sequence labelling: $L$ layers, each with $D = 2$ directions of $d_h$ cells, and an affine layer to $K$ class scores.
With $D = 1$ the same formulas describe the deep unidirectional network that the BRNN is compared with.
The two directions are two independent recurrent networks inside a layer, and all the communication between past and future context happens in the layers above and in the output.
Lipton, Berkowitz and Elkan point out that the LSTM and the BRNN are compatible ideas: the LSTM is a basic unit from which to compose a hidden layer, while the BRNN concerns the wiring of the hidden layers, regardless of what nodes they contain. {#c-lipton-compatible}
Any recurrent cell can therefore fill the two directions: a tanh unit, an LSTM or a [[gated-recurrent-unit|GRU]].

## Components

The notation uses row vectors, as torch.nn.LSTM does.
The input of layer $l$ at step $t$ is $\mathbf{u}^{(l)}_t$, with $\mathbf{u}^{(1)}_t = \mathbf{x}_t$, and a sequence has $n \le T$ real steps.
A recurrent cell maps an input, a previous state and its parameters to a new state; for the LSTM, with $\mathbf{z} = \mathbf{u} W + \mathbf{h} U + \mathbf{b}$ split into four blocks $\mathbf{z}_i, \mathbf{z}_f, \mathbf{z}_g, \mathbf{z}_o$,

$$ \mathbf{c}' = \sigma(\mathbf{z}_f) \odot \mathbf{c} + \sigma(\mathbf{z}_i) \odot \tanh(\mathbf{z}_g), \qquad \mathbf{h}' = \sigma(\mathbf{z}_o) \odot \tanh(\mathbf{c}') , $$ {#eq-cell}

and for the tanh unit $\mathbf{h}' = \tanh(\mathbf{u} W + \mathbf{h} U + \mathbf{b})$.

**The forward direction.**
It runs the cell from the first step to the last, from a zero state:

$$ \overrightarrow{\mathbf{h}}^{(l)}_t = \operatorname{cell}\big(\mathbf{u}^{(l)}_t, \overrightarrow{\mathbf{h}}^{(l)}_{t-1}; \overrightarrow{W}^{(l)}, \overrightarrow{U}^{(l)}, \overrightarrow{\mathbf{b}}^{(l)}\big), \quad t = 1, \dots, n, \qquad \overrightarrow{\mathbf{h}}^{(l)}_0 = \mathbf{0} . $$ {#eq-forward}

**The backward direction.**
It has its own weights and runs the same cell from the last real step to the first:

$$ \overleftarrow{\mathbf{h}}^{(l)}_t = \operatorname{cell}\big(\mathbf{u}^{(l)}_t, \overleftarrow{\mathbf{h}}^{(l)}_{t+1}; \overleftarrow{W}^{(l)}, \overleftarrow{U}^{(l)}, \overleftarrow{\mathbf{b}}^{(l)}\big), \quad t = n, \dots, 1, \qquad \overleftarrow{\mathbf{h}}^{(l)}_{n+1} = \mathbf{0} . $$ {#eq-backward}

For an LSTM the cell state $\mathbf{c}$ of each direction is carried the same way and also starts at zero.

**The concatenation.**
The output of a layer at step $t$ is the pair of states, which is the input of the next layer at the same step:

$$ \mathbf{u}^{(l+1)}_t = \big[\overrightarrow{\mathbf{h}}^{(l)}_t, \overleftarrow{\mathbf{h}}^{(l)}_t\big] \in \mathbb{R}^{2 d_h} . $$ {#eq-concat}

The forward half summarizes $\mathbf{u}_1, \dots, \mathbf{u}_t$ and the backward half $\mathbf{u}_t, \dots, \mathbf{u}_n$.
In torch.nn.LSTM with bidirectional=True, the output contains the concatenation of the forward and reverse hidden states at each time step, of size D * H_out with D = 2. {#c-torch-concat}

**The output.**
An affine layer on the top layer gives $K$ class scores at every step, and their [[al:softmax-regression|softmax]] is the predicted distribution of the label:

$$ \mathbf{s}_t = \mathbf{u}^{(L+1)}_t W_o + \mathbf{b}_o, \qquad p(y_t = k \mid \mathbf{x}_1, \dots, \mathbf{x}_n) = \operatorname{softmax}(\mathbf{s}_t)_k . $$ {#eq-output}

Lipton et al. write the one-layer network as three equations, a forward recurrence on $\mathbf{h}^{(t-1)}$, a backward recurrence on $\mathbf{z}^{(t+1)}$, and an output $\hat{\mathbf{y}}^{(t)} = \operatorname{softmax}(W^{yh} \mathbf{h}^{(t)} + W^{yz} \mathbf{z}^{(t)} + \mathbf{b}_y)$, which is eq-output with $W_o$ split into its two halves. {#c-lipton-equations}

**The boundaries.**
The state before the first step of the forward direction and after the last step of the backward direction are not given by the data.
This page, PyTorch and most implementations set them to zero; Schuster and Paliwal set these boundary state inputs to a fixed value of 0.5, and set the local state derivatives at the boundaries to zero. {#c-boundary}

### Each direction alone

Schuster and Paliwal note that without the backward states the structure is a regular unidirectional forward RNN, and that without the forward states it is a regular RNN with a reversed time axis. {#c-special-cases}
The test `test_each_direction_alone_is_a_unidirectional_rnn` checks both reductions: with one direction's weights set to zero its state stays exactly zero, and the network computes what the other direction computes alone, on the sequence or on the reversed sequence.

### The layout in PyTorch

In torch.nn.LSTM, the input weights weight_ih_l[k] of a layer k > 0 have shape (4*hidden_size, num_directions * hidden_size), the parameters of the backward direction carry the suffix _reverse, and for bidirectional LSTMs forward and backward are directions 0 and 1 of h_n. {#c-torch-layout}
`impl/pytorch/birnn.py` stores $W$ as the transpose of weight_ih and $U$ as the transpose of weight_hh, and `to_torch()` copies them into a torch.nn.LSTM or torch.nn.RNN with bidirectional=True.

## Shapes and parameters

A direction of layer $l$ has input weights $W$ of shape $d_{\text{in}} \times 4 d_h$, recurrent weights $U$ of shape $d_h \times 4 d_h$ and a bias of size $4 d_h$, where $d_{\text{in}} = d_x$ in the first layer and $D d_h$ above it.
With the output layer, the network has

$$ N = D\big[4 d_h (d_x + d_h) + 4 d_h\big] + (L - 1) D \big[4 d_h (D d_h + d_h) + 4 d_h\big] + D d_h K + K $$ {#eq-params}

parameters, with every 4 replaced by 1 for tanh cells. {#c-params}
In the shape contract this is

{shape:params}

and the example network, 3 inputs, 2 layers of 4 cells per direction and 5 classes, has {shape:params@example} parameters, {shape:weights@example} of them weights and {shape:recurrent_weights@example} recurrent weights.
A bidirectional layer doubles the parameters of a unidirectional one of the same width, and an upper layer reads $2 d_h$ inputs instead of $d_h$, so at a fixed parameter budget a bidirectional network has fewer cells per direction.

Graves et al. compare networks with about the same number of weights in Table 1, among them CTC-3l-250h, a three-layer bidirectional LSTM with 250 cells in each direction, and CTC-3l-421h-uni, a three-layer unidirectional one with 421 cells, both listed with 3.8M weights.
On their inputs of size 123 and 62 outputs, the 61 phonemes and the blank of connectionist temporal classification, eq-params gives {shape:params@ctc3l250h} and {shape:params@ctc3l421huni}: both round to 3.8M, and they differ by 106. {#c-graves-weights}
Their LSTM cells also have peephole connections, with diagonal weight matrices from the cell to the gates. {#c-graves-peepholes}
Counted as three vectors of $d_h$ per direction and layer, the peepholes add 4,500 and 3,789 parameters and leave both networks at 3.79M.
torch.nn.LSTM keeps a second bias for every gate, so its recurrent layers have $4 L D d_h$ more parameters than the ones counted here, which only enter the function through their sum with the first. {#c-torch-biases}
The derivation `parameter-count` proves these counts and reproduces the two figures.

## Computation and cost

Each step of each direction makes the two products of its cell, so a forward pass over a batch of $B$ sequences of $T$ steps costs

$$ C = 8 D B T d_h (d_x + d_h) + 8 (L - 1) D B T d_h (D d_h + d_h) + 2 B T D d_h K $$ {#eq-flops}

FLOPs, counting a multiply-add as two and leaving out the gates' nonlinearities and the softmax. {#c-flops}
The example network on 3 sequences of 6 steps costs {shape:flops_forward@example} FLOPs.
The state carried from step to step is a hidden and a cell vector per direction and layer, {shape:state@example} numbers per sequence in the example.

**What an output depends on.**
In a unidirectional network the output at step $t$ is a function of $\mathbf{x}_1, \dots, \mathbf{x}_t$ only; in a bidirectional network of one or more layers the output at every step is a function of the whole input $\mathbf{x}_1, \dots, \mathbf{x}_T$. {#c-context}
The test `test_context_of_each_output` perturbs each input in turn and checks which outputs move.

**Latency.**
It follows that no output of a bidirectional network can be computed before the last input is known: the first output waits $T - 1$ steps longer than in a unidirectional network. {#c-latency}
Lipton et al. note that a BRNN cannot run continuously, since it requires a fixed endpoint in both the future and the past, and that it is not an appropriate algorithm for the online setting, as it is implausible to receive information from the future. {#c-lipton-online}

**Parallelism.**
The two directions of a layer are independent and can run at the same time, but each is sequential, and in a deep network the backward direction of layer $l + 1$ cannot start until the forward direction of layer $l$ has reached the last step.
The longest chain of cell steps that must run one after the other has length $T + L - 1$ in a unidirectional network of $L$ layers, where the layers can run as a wavefront one step apart, and $L T$ in a bidirectional one. {#c-depth}
The derivation `context-and-depth` proves the three statements, and its check computes the dependencies and the longest chain for every $T, L \le 12$.
Like every recurrent network, the BRNN computes its states one after another: Vaswani et al. write that recurrent models generate a sequence of hidden states h_t as a function of h_{t-1} and the input for position t, and that this inherently sequential nature precludes parallelization within training examples. {#c-sequential}

## Training

Schuster and Paliwal note that the BRNN can be trained with the same algorithms as a regular unidirectional RNN, because there are no interactions between the two types of state neurons and the network can therefore be unfolded into a general feedforward network. {#c-train-same}
Training is [[al:backpropagation-through-time|backpropagation through time]] on the unrolled graph: the gradient of the loss flows backwards through the forward direction from step $T$ to 1 and through the backward direction from step 1 to $T$, and into each layer's input from both.
For sequence labelling the loss is the [[m2:cross-entropy|cross-entropy]] of the labels, summed over the real steps of each sequence and averaged over the batch:

$$ J = -\frac{1}{B} \sum_{b=1}^{B} \sum_{t=1}^{n_b} \log p(y_{bt} \mid \mathbf{x}_{b,1}, \dots, \mathbf{x}_{b,n_b}) . $$ {#eq-loss}

Graves et al. replace it by the CTC loss, which sums over all alignments of an output sequence shorter than the input, and train all their networks with stochastic gradient descent, with learning rate 10^-4, momentum 0.9 and random initial weights drawn uniformly from [-0.1, 0.1]. {#c-graves-sgd}

**Batches of different lengths.**
A batch holds sequences of different lengths $n_b$, padded at the end to the longest, $T$.
The forward direction does not care, as long as the outputs at padded steps are ignored.
The backward direction does: eq-backward starts at step $n_b$, not at $T$.
Huang, Xu and Yu run the forward and backward passes over whole sentences and reset the hidden states to 0 at the beginning of each sentence. {#c-huang-bptt}
Reversing the first $n_b$ steps of each sequence, running the backward direction left to right and reversing its states again gives, at the real steps, the backward states of each sequence alone; so does a reverse scan over all $T$ steps that sets the state to zero after every padded step. {#c-padding}
Running the backward direction from step $T$ without either, on the raw padded batch, gives different states at every real step of a shorter sequence, because they have read the padding. {#c-padding-naive}
The derivation `padding` proves the three statements; the NumPy reference uses the reversal, the JAX version the masked scan, and the PyTorch tests compare both with torch.nn.LSTM on a packed batch.

## Inductive biases

**Context on both sides.** The network assumes that the label of a step may depend on inputs before and after it, and gives it both, without fixing how far: the forward states summarize the whole past and the backward states the whole future.
This replaces the delay that a unidirectional network needs to see the future.
Schuster and Paliwal report that delaying the output by some frames improves results, but that the optimal delay is task dependent and has to be found by trial and error on a validation set. {#c-delay}

**Recency in each direction.** A recurrent state represents its recent inputs best, so the concatenation at step $t$ is centred on $t$.
Bahdanau, Cho and Bengio use a bidirectional RNN to make the annotation of each word summarize both the preceding and the following words, and note that, due to the tendency of RNNs to better represent recent inputs, the annotation is focused on the words around it. {#c-bahdanau-recent}

**Independent directions.** The two directions do not see each other inside a layer.
In one layer, eq-output adds a score computed from the past to a score computed from the future, $\mathbf{s}_t = \overrightarrow{\mathbf{h}}_t W_o^{\rightarrow} + \overleftarrow{\mathbf{h}}_t W_o^{\leftarrow} + \mathbf{b}_o$, so a label that depends on an interaction between the two sides, such as the sum of the two neighbours of a step modulo 3, needs a second layer or a nonlinear output.
No pair of additive scores gets all nine combinations of the two neighbours right: for a fixed past, the future runs through the three values and the correct label through the three classes, so the margins of the correct label over the next class sum to zero over the nine combinations, while each of them would have to be positive.
The Experiments section has an example.
This is also why a single BRNN is different from two separately trained networks whose outputs are merged.
Schuster and Paliwal discuss merging the outputs of a forward and a backward network by linear or logarithmic opinion pooling, and argue that it is generally not clear how to merge network outputs in an optimal way, since networks trained on the same data can no longer be regarded as independent. {#c-merge}

## Evidence

**Schuster and Paliwal (1997).** On TIMIT phoneme classification, with every structure adjusted to about 13,000 free parameters, the recognition rates on the test data are 59.67% for an MLP on one segment, 65.69% on three segments and 64.32% on five, 63.2% for the forward RNN and 61.91% for the backward RNN, 65.83% for the forward RNN with a one-segment delay and 63.27% with a two-segment delay, 65.28% for the merged outputs of a forward and a backward RNN, and 68.53% for the BRNN, the best result. {#c-schuster-timit}
They conclude that there is no need to merge the outputs of two separate networks or to search for an optimal delay, and that the training time of the BRNN is about the same as for the other RNNs, the number of free parameters having been kept about equal. {#c-schuster-conclusion}

**Graves, Mohamed and Hinton (2013).** On TIMIT phoneme recognition with CTC, four networks with approximately the same number of weights reach phoneme error rates of 37.6% for a three-layer network of tanh units (CTC-3l-500h-tanh, 3.7M weights), 23.0% for a one-layer bidirectional LSTM (CTC-1l-622h, 3.8M), 19.6% for the three-layer unidirectional LSTM CTC-3l-421h-uni (3.8M) and 18.6% for the three-layer bidirectional LSTM CTC-3l-250h (3.8M). {#c-graves-table}
They conclude from these four that LSTM works much better than tanh for this task, that bidirectional LSTM has a slight advantage over unidirectional LSTM, and that depth is more important than layer size. {#c-graves-conclusion}
They add that all experiments were run only once, so the variance due to random weight initialisation and weight noise is unknown. {#c-graves-once}
The 1.0-point gap between the unidirectional and the bidirectional network therefore comes from one run of each.

**Huang, Xu and Yu (2015).** On part-of-speech tagging (accuracy), CoNLL2000 chunking (F1) and CoNLL2003 named entity recognition (F1), with random word embeddings, the LSTM tagger scores 97.10, 92.88 and 79.82 and the bidirectional LSTM 97.30, 93.64 and 81.11; with Senna embeddings, the LSTM scores 97.29, 92.99 and 83.74 and the bidirectional LSTM 97.40, 93.92 and 85.17. {#c-huang-table}
Adding a CRF output layer on top of the bidirectional LSTM gives 97.43, 94.13 and 84.26 with random embeddings and 97.55, 94.46 and 88.83 with Senna embeddings. {#c-huang-crf-table}
Huang et al. write that their results confirm the superiority of BI-LSTM compared to LSTM. {#c-huang-superiority}

**Handwriting.** Lipton et al. report that in Graves et al. (2009) a bidirectional LSTM reached 81.5% word-level accuracy on handwriting collected from an interactive whiteboard, against 70.1% for an HMM. {#c-lipton-handwriting}

These are reported results; this page has not reproduced them.
In each of these comparisons the bidirectional network wins, and the size of the win varies: about 3 points of accuracy for Schuster and Paliwal's networks, one point of error rate for Graves et al.'s deep LSTMs, and from 0.1 to 1.4 points for Huang et al.'s taggers.

## Limitations

- **No online use.** The first output needs the last input (Computation and cost), so a BRNN cannot transcribe speech as it is spoken, predict the next token of a text, or generate a sequence; it is an encoder, not a decoder or a language model.
- **Twice the state, and a longer chain.** Each layer keeps two states, and a deep BRNN is sequential over $L T$ steps rather than $T + L - 1$ (Computation and cost), which makes deep bidirectional networks slower than unidirectional ones of the same parameter count.
- **Padding has to be handled.** On a padded batch, a backward direction that starts at the end of the array reads the padding first and computes wrong states at every step of a shorter sequence (Training).
- **The two directions meet late.** Inside a layer neither direction knows what the other has seen; a single bidirectional layer with a linear output can only add a score of the past to a score of the future.
- **Independent outputs.** With eq-output the labels of different steps are predicted independently given the input; structured outputs need a CRF or CTC layer on top, or the modified network of Schuster and Paliwal's second part.
- **Sequential computation.** Like every RNN, each direction waits for its previous step, which is what attention-only models removed.

## Variants and descendants

- **Bidirectional LSTM and GRU.** The cells of eq-forward and eq-backward can be any recurrent unit; with LSTM cells and several layers this is the deep bidirectional LSTM of Graves et al., the main architecture of their paper.
  Graves et al. write that, as far as they are aware, theirs is the first time deep LSTM has been applied to speech recognition, and that it yields a dramatic improvement over single-layer LSTM. {#c-graves-deep}
- **Dependent outputs.** In the second part of their paper, Schuster and Paliwal modify the bidirectional structure to estimate the conditional posterior probability of complete symbol sequences without making any explicit assumption about the shape of the distribution. {#c-schuster-dependent}
- **BiLSTM-CRF.** Huang et al. combine a bidirectional LSTM with a conditional random field output layer and describe their work as the first to apply a bidirectional LSTM CRF (BI-LSTM-CRF) model to NLP benchmark sequence tagging data sets. {#c-huang-first}
  The [[al:conditional-random-field|CRF]] scores whole label sequences with a transition matrix and decodes with the Viterbi algorithm.
- **Bidirectional language models.** ELMo word vectors are learned functions of the internal states of a deep bidirectional language model (biLM), which is pre-trained on a large text corpus. {#c-elmo}
  Its two directions are trained as separate left-to-right and right-to-left language models, since a bidirectional network cannot predict a word it has already seen from the other side.
- **Bidirectional Transformers.** Devlin et al. contrast [[bert|BERT]] with Peters et al. (2018a), which uses a shallow concatenation of independently trained left-to-right and right-to-left LMs. {#c-contrast}
  BERT pre-trains deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. {#c-bert}
  Its self-attention gives every position the whole sequence in one layer, the role that the two directions play in a BRNN, without the sequential chain.

## Notable models

The bidirectional LSTMs of Graves et al. for phoneme recognition, CTC-3l-250h among them, are the reference deep BRNNs.
RNNsearch, the attention-based translation model of Bahdanau et al., encodes the source sentence with a bidirectional RNN whose forward RNN reads the sentence from x_1 to x_Tx and whose backward RNN reads it from x_Tx to x_1, and annotates each word with the concatenation of the two hidden states. {#c-bahdanau}
Its decoder attends over these annotations; see [[sequence-to-sequence-model]].
BiLSTM-CRF taggers were a common design for named entity recognition and part-of-speech tagging until pre-trained Transformers replaced them, and ELMo's biLM supplied contextual word vectors to many of them.

## Implementations

The NumPy reference `impl/numpy/birnn.py` implements eq-cell for the LSTM and the tanh unit, eq-forward, eq-backward and eq-concat over a padded batch with lengths, eq-output, eq-loss, backpropagation through time by hand through both directions and all layers, and the counts eq-params and eq-flops.
The PyTorch version `impl/pytorch/birnn.py` is an `nn.Module` with the same parameters; `to_torch()` returns the torch.nn.LSTM or torch.nn.RNN with bidirectional=True and the same weights, and the tests check that the two agree on a packed batch, run `torch.autograd.gradcheck`, and train the network on a task that needs the future.
The JAX version `impl/jax/birnn.py` is a pure function that runs each direction with `jax.lax.scan`, the backward one with `reverse=True` and a mask that resets its state inside the padding, tested under `jax.jit` and `jax.vmap` and with `jax.grad` against the gradients of the NumPy backward pass.
All three load the shared fixture of `fixtures/`: a two-layer bidirectional LSTM on three sequences of lengths 6, 4 and 1 padded to 6 steps with random numbers, its scores, loss and gradients.

### Worked example

Take one tanh unit per direction, with input weight 1, recurrent weight 1/2 and no bias in both, and the input $x = (1, 0, -1)$.
The forward states are $\overrightarrow{h}_1 = \tanh 1 = 0.7616$, $\overrightarrow{h}_2 = \tanh(0.5 \times 0.7616) = 0.3634$ and $\overrightarrow{h}_3 = \tanh(-1 + 0.5 \times 0.3634) = -0.6741$.
The backward states start at the other end: $\overleftarrow{h}_3 = \tanh(-1) = -0.7616$, $\overleftarrow{h}_2 = -0.3634$ and $\overleftarrow{h}_1 = \tanh(1 - 0.5 \times 0.3634) = 0.6741$.
With an output that adds the two halves of eq-concat, the scores are $1.4357$, $0$ and $-1.4357$; a forward network would give $0.7616$ at the first step, not knowing that the sequence ends in $-1$.
Now pad the sequence with a fourth step holding 5.
With the length 3, nothing changes.
Read as data, the padding enters the backward direction first, $\overleftarrow{h}_4 = \tanh 5 = 0.9999$, and changes its states at every real step, to $-0.4622$, $-0.2270$ and $0.7096$ at steps 3, 2 and 1.
The test `test_worked_example` checks these numbers.

### Pitfalls

> [!pitfall]
> **Running the backward direction over padding.** torch.nn.LSTM with bidirectional=True on a padded tensor starts the reverse direction at the last array position, so every state of a shorter sequence has read the padding. Pack the batch with `torch.nn.utils.rnn.pack_padded_sequence` or reverse each sequence within its length. The test `test_matches_torch_on_packed_sequences` checks that the reference matches torch.nn.LSTM on a packed batch and differs from it on the unpacked one, and the NumPy `test_padding` that padding filled with 1000 changes nothing.

> [!pitfall]
> **Taking the last output as the summary of the sequence.** For bidirectional LSTMs, h_n is not equivalent to the last element of output: h_n contains the final forward and reverse hidden states, while the last element of output contains the final forward hidden state and the initial reverse hidden state. {#c-torch-hn}
> The backward half of `output[:, -1]` has read only the last step; a summary of the whole sequence is the forward state at the last real step together with the backward state at the first.

> [!pitfall]
> **Using a bidirectional network where the future is not available.** A BRNN language model or streaming recognizer looks at the token it has to predict through its backward states, so its training loss is meaningless and it cannot run at inference time. The test `test_context_of_each_output` checks that a bidirectional output depends on every input, and `context-and-depth` proves it.

> [!pitfall]
> **Loading the weights of a library with two biases.** torch.nn.LSTM adds bias_ih and bias_hh, and only their sum matters. Copying one of them and leaving the other at its random initialization gives a different network. `to_torch()` sets bias_hh to zero, and `test_param_count` checks that the library's recurrent layers have exactly $4 L D d_h$ more parameters than this page's.

> [!pitfall]
> **Expecting one bidirectional layer to combine past and future.** With a linear output the two directions only add their scores. In the PyTorch `test_the_future_is_needed`, a one-layer bidirectional LSTM fails a task that a two-layer one learns.

## Experiments

This page has no experiment yet.
The PyTorch test `test_the_future_is_needed` is a small one.
The input is a sequence of 8 random tokens from {0, 1, 2}, one-hot, and the label of step $t$ is $(x_{t-1} + x_{t+1}) \bmod 3$, which depends on the next token.
Each network has 16 LSTM cells per direction and is trained for 300 steps of Adam on batches of 64; the accuracy is measured on 1000 fresh sequences, leaving out the last step, whose label a forward network can know.
A two-layer bidirectional LSTM reaches an accuracy above 95%, a two-layer unidirectional one stays below 40%, near the chance level of 1/3, and the same unidirectional network with its output delayed by one step, so that it reads $x_{t+1}$ before it labels step $t$, also passes 95%.
A one-layer bidirectional LSTM stays below 50% in this run: its linear output can only add a function of the past to a function of the future, and the sum modulo 3 is not of that form for a linear scorer.
The checks of the three derivations count parameters and FLOPs for the networks of Graves et al., compute the dependencies and the longest chain of steps, and compare the reversed, masked and naive backward scans on random padded batches.

## History

Schuster and Paliwal published Bidirectional Recurrent Neural Networks in IEEE Transactions on Signal Processing, vol. 45, no. 11, in November 1997. {#c-hist-schuster}
Lipton et al. describe the BRNN, first described by Schuster and Paliwal, as having two layers of hidden nodes, both connected to the input and the output, the first with recurrent connections from the past time steps and the second with the direction of the recurrent connections flipped. {#c-lipton-first}
The combination with the LSTM, the BLSTM, was used for phoneme classification and handwriting recognition by Graves and Schmidhuber (2005) and Graves et al. (2009). {#c-hist-blstm}
In 2013, Graves, Mohamed and Hinton stacked bidirectional LSTM layers and trained them end to end on TIMIT.
From 2014 the bidirectional RNN became the usual encoder of attention-based translation; Bahdanau et al. note that it had been successfully used recently in speech recognition. {#c-hist-bahdanau}
Huang et al. brought the BiLSTM-CRF to sequence tagging in 2015, and ELMo (2018) used a deep biLM for contextual word vectors.
After BERT (2019), bidirectional Transformers took over most of the tasks for which BRNNs had been the standard, and bidirectional recurrent layers remain in use where sequences are long and models small.

## References

- M. Schuster and K. K. Paliwal. Bidirectional Recurrent Neural Networks. IEEE Transactions on Signal Processing 45(11), pp. 2673–2681, 1997.
- A. Graves, A. Mohamed and G. Hinton. Speech Recognition with Deep Recurrent Neural Networks. arXiv 1303.5778, 2013.
- Z. Huang, W. Xu and K. Yu. Bidirectional LSTM-CRF Models for Sequence Tagging. arXiv 1508.01991, 2015.
- Z. C. Lipton, J. Berkowitz and C. Elkan. A Critical Review of Recurrent Neural Networks for Sequence Learning. arXiv 1506.00019, 2015.
- D. Bahdanau, K. Cho and Y. Bengio. Neural Machine Translation by Jointly Learning to Align and Translate. ICLR 2015.
- M. E. Peters et al. Deep Contextualized Word Representations. NAACL 2018.
- J. Devlin, M.-W. Chang, K. Lee and K. Toutanova. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL 2019.
- A. Vaswani et al. Attention Is All You Need. NeurIPS 2017.
- S. Hochreiter and J. Schmidhuber. Long Short-Term Memory. Neural Computation 9(8), 1997.
- PyTorch contributors. torch.nn.LSTM. PyTorch 2.15 documentation.
