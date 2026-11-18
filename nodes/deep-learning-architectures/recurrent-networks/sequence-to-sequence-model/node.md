---
tag: N149H2
kind: architecture
slug: sequence-to-sequence-model
name: Sequence-to-sequence model
aliases: [seq2seq, encoder-decoder, RNN encoder-decoder, sequence to sequence learning, neural encoder-decoder]
area: deep-learning-architectures/recurrent-networks
level: core
summary: >-
  Two recurrent networks: an encoder reads the input sequence into a fixed-size vector, and a decoder started from it generates the output one token at a time.
uses: [matrix-multiplication@m2, cross-entropy@m2, softmax-regression@al]
related_to: [recurrent-neural-network, long-short-term-memory, gated-recurrent-unit, transformer, multi-head-attention, bleu]
history:
  year: 2014
  people: [Ilya Sutskever, Oriol Vinyals, Quoc V. Le, Kyunghyun Cho, Yoshua Bengio]
  note: "Sutskever, Vinyals and Le's Sequence to Sequence Learning with Neural Networks (NeurIPS 2014) and Cho et al.'s RNN Encoder-Decoder (EMNLP 2014) appeared within months of each other, after Kalchbrenner and Blunsom's recurrent continuous translation models (2013); Bahdanau, Cho and Bengio added attention the same year."
prereqs: [long-short-term-memory, backpropagation-through-time@al, beam-search@al, cross-entropy-loss@al]
same_as:
  wikipedia: Seq2seq
sources:
  - {paper: "Sutskever, Vinyals and Le (2014)", title: "Sequence to Sequence Learning with Neural Networks", locator: "arXiv 1409.3215v3, the abstract, Sections 1 to 5, Tables 1 and 2 and Figure 3", note: "the model, the reversed source, the training details, the beam search and the results on WMT'14"}
  - {paper: "Cho, van Merriënboer, Gulcehre, Bahdanau, Bougares, Schwenk and Bengio (2014)", title: "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation", locator: "arXiv 1406.1078v3, the abstract and Section 2.2", note: "the RNN Encoder-Decoder and its two uses"}
  - {paper: "Bahdanau, Cho and Bengio (2015)", title: "Neural Machine Translation by Jointly Learning to Align and Translate", locator: "ICLR 2015; arXiv 1409.0473v7, the abstract, Section 1 and Table 1", note: "the fixed-length bottleneck"}
  - {paper: "Bengio, Vinyals, Jaitly and Shazeer (2015)", title: "Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks", locator: "arXiv 1506.03099v3, the abstract, Sections 2.4 and 4.1 and Table 1", note: "the gap between training and inference, and scheduled sampling"}
  - {paper: "Ranzato, Chopra, Auli and Zaremba (2016)", title: "Sequence Level Training with Recurrent Neural Networks", locator: "ICLR 2016; arXiv 1511.06732v7, Section 1", note: "exposure bias and MIXER"}
  - {paper: "Hochreiter and Schmidhuber (1997)", title: "Long Short-Term Memory", locator: "Section 1", note: "the minimal time lag"}
  - {url: "https://docs.pytorch.org/docs/2.15/generated/torch.nn.LSTM.html", title: "torch.nn.LSTM", locator: "the equations and the parameters", note: "the order of the gates and the two biases"}
family: recurrent
inputs: "a batch of source token sequences (B, S) and, in training, of target token sequences (B, T), each ending in <EOS>"
outputs: "the logits of p(y_t | v, y_1..y_{t-1}) at every target step, a (B, T, V) array; at inference, a generated token sequence"
components: [long-short-term-memory]
inductive_biases: ["the whole input passes through one fixed-size vector, the final state of the encoder", "the output is generated left to right, each token conditioned on the vector and on all the tokens before it", "the encoder and the decoder have separate weights, so the two sequences may be of different kinds and lengths", "the order of the input matters, and reversing it changes what is easy to learn"]
params: "(V_x + V_y) d_e + 2 [4 d_h (d_e + d_h) + 4 d_h + (L - 1)(8 d_h^2 + 4 d_h)] + d_h V_y + V_y, for L LSTM layers in each stack"
flops_per_example: "(S + T) [8 d_h (d_e + d_h) + 16 (L - 1) d_h^2] + 2 T d_h V_y for a pair of S and T tokens, counting a multiply-add as 2 FLOPs and only the products"
---

A sequence-to-sequence model maps a sequence to another sequence of a different length, such as a sentence to its translation, with two recurrent networks: an encoder reads the input and a decoder writes the output.
In the words of Sutskever, Vinyals and Le, the method uses a multilayered Long Short-Term Memory (LSTM) to map the input sequence to a vector of a fixed dimensionality, and then another deep LSTM to decode the target sequence from the vector. {#c-def}
Cho et al. proposed the same structure the same year under the name RNN Encoder-Decoder, in which one RNN encodes a sequence of symbols into a fixed-length vector representation and the other decodes the representation into another sequence of symbols, the two being jointly trained to maximize the conditional probability of a target sequence given a source sequence. {#c-cho-def}
The decoder emits one token at a time, each conditioned on the vector and on the tokens it has already emitted, and stops when it emits an end-of-sequence token.
This page describes the model of Sutskever et al., with [[long-short-term-memory|LSTM]] layers, a reversed source and a beam search, and covers the training and decoding questions every encoder-decoder shares.

## In plain words

To translate a sentence, the model first reads it to the end, word by word, and keeps nothing but a list of numbers, its state, which by then has to hold everything about the sentence that matters.
A second network starts from that state and writes the translation word by word: at each step it says how likely every word of the target language is to come next, one word is chosen, and it is fed back in for the next step.
A special end-of-sentence word lets the second network decide when the translation is done, so the output can be shorter or longer than the input.
During training the second network is always fed the correct previous word; at test time it has to live with its own choices, and picking the best word at each step does not always give the best sentence, so a few candidate sentences are kept in parallel.
The surprising trick of Sutskever et al. was to feed the source sentence backwards: the first words of the source are then the last ones read, close to the first words of the translation, and the network learns much more easily.

## Overview

A training pair is a source sequence $x_1, \dots, x_S$ and a target sequence $y_1, \dots, y_T$ of tokens from two vocabularies of sizes $V_x$ and $V_y$, both containing a token `<EOS>` that ends every sequence.
The model gives a probability to every target given the source, factored into one conditional per target token, Eq. 1 of Sutskever et al.:

$$ p(y_1, \dots, y_T \mid x_1, \dots, x_S) = \prod_{t=1}^{T} p(y_t \mid v, y_1, \dots, y_{t-1}) . $$ {#eq-factor}

Sutskever et al. obtain the fixed-dimensional representation v of the input from the last hidden state of the encoder LSTM and compute the probability of the output with an LSTM language model whose initial hidden state is set to v, each conditional p(y_t | v, y_1, ..., y_{t-1}) being a softmax over all the words in the vocabulary. {#c-factor-def}
They require that each sentence end with the special end-of-sentence symbol `<EOS>`, which enables the model to define a distribution over sequences of all possible lengths. {#c-eos}
The second LSTM, they note, is essentially a recurrent neural network language model, except that it is conditioned on the input sequence. {#c-lm}

Cho et al. point out the two ways a trained encoder-decoder can be used: to generate a target sequence given an input sequence, or to score a given pair of input and output sequences, the score being the probability p(y | x). {#c-cho-uses}
Scoring is cheap, one teacher-forced pass; generating requires a search over the outputs, below.

Sutskever et al. list three ways in which their model differs from the plain description.
Their first change is to use two different LSTMs, one for the input sequence and another for the output sequence, which increases the number of model parameters at negligible computational cost and makes it natural to train on multiple language pairs simultaneously. {#c-two-lstms}
Their second change is depth: they found that deep LSTMs significantly outperformed shallow LSTMs, so they chose an LSTM with four layers. {#c-four-layers}
Their third change is to reverse the order of the words of the input: instead of mapping the sentence a, b, c to α, β, γ, the LSTM is asked to map c, b, a to α, β, γ. {#c-reverse-def}

## Components

The notation is that of the [[long-short-term-memory|LSTM]] page, with vectors as rows: an LSTM layer maps an input $\mathbf{x}$, a hidden state $\mathbf{h}$ and a cell $\mathbf{c}$ to a new $(\mathbf{h}, \mathbf{c})$ through $\mathbf{z} = \mathbf{x} W + \mathbf{h} U + \mathbf{b}$, split into the gates $i, f, g, o$ in the order of torch.nn.LSTM.

**The embeddings.**
Two tables $E_x$ of $V_x \times d_e$ and $E_y$ of $V_y \times d_e$ turn tokens into vectors; the source and the target do not share them, as their vocabularies differ.

**The encoder.**
A stack of $L$ LSTM layers reads the source embeddings, in the reversed order $x_S, \dots, x_1$ when the source is reversed, from a zero state.
Writing $\mathbf{h}^{l}_s, \mathbf{c}^{l}_s$ for the state of layer $l$ after $s$ steps, $\pi(s) = S + 1 - s$ for a reversed source and $\pi(s) = s$ otherwise:

$$ \mathbf{h}^{0}_s = E_x[x_{\pi(s)}], \qquad (\mathbf{h}^{l}_s, \mathbf{c}^{l}_s) = \mathrm{LSTM}^{l}_{\text{enc}}\big(\mathbf{h}^{l-1}_s, \mathbf{h}^{l}_{s-1}, \mathbf{c}^{l}_{s-1}\big), \qquad v = \big(\mathbf{h}^{l}_S, \mathbf{c}^{l}_S\big)_{l=1}^{L} . $$ {#eq-encoder}

The vector $v$ is the final hidden state and cell of every layer, the "8000 real numbers" of Sutskever et al. for four layers of 1000 cells.
Only $v$ is passed on: the states of the earlier steps are dropped, which is the bottleneck that attention later removed.

**The decoder.**
A second stack of $L$ LSTM layers, with its own weights, starts from $v$ and reads the target shifted right by one, `<EOS>` first:

$$ y_0 = \langle \mathrm{EOS} \rangle, \quad \mathbf{s}^{0}_t = E_y[y_{t-1}], \quad (\mathbf{s}^{l}_t, \mathbf{m}^{l}_t) = \mathrm{LSTM}^{l}_{\text{dec}}\big(\mathbf{s}^{l-1}_t, \mathbf{s}^{l}_{t-1}, \mathbf{m}^{l}_{t-1}\big), \quad (\mathbf{s}^{l}_0, \mathbf{m}^{l}_0) = (\mathbf{h}^{l}_S, \mathbf{c}^{l}_S) . $$ {#eq-decoder}

Using `<EOS>` as the start symbol, as Sutskever et al.'s figure does, saves a token; many implementations use a separate start token instead.

**The softmax layer.**
An affine map of the top state of the decoder gives the logits, and their softmax, as in [[al:softmax-regression|softmax regression]], the conditional of eq-factor:

$$ p(y_t \mid v, y_1, \dots, y_{t-1}) = \operatorname{softmax}\big(\mathbf{s}^{L}_t W_o + \mathbf{b}_o\big)_{y_t} . $$ {#eq-step}

Sutskever et al. used a naive softmax over 80,000 words at each output, which, as the count below finds, costs more per target token than a whole step of the decoder stack.

## Shapes and parameters

With $L$ layers in each stack, embeddings of width $d_e$ and $d_h$ cells per layer, the model has

{shape:params}

parameters, {shape:params@example} for the example config of 7 and 6 tokens, embeddings of 3, 4 cells and 2 layers. {#c-params}
The encoder alone has {shape:encoder.params} and the decoder as many; their weights do not depend on the lengths of the sequences.

Sutskever et al. describe deep LSTMs with 4 layers, with 1000 cells at each layer and 1000 dimensional word embeddings, an input vocabulary of 160,000 and an output vocabulary of 80,000, so that the deep LSTM uses 8000 real numbers to represent a sentence; the resulting LSTM has 384M parameters, of which 64M are pure recurrent connections, 32M for the encoder and 32M for the decoder. {#c-sutskever-config}
The count reproduces these numbers exactly if the biases are left out: the model then has {shape:weights@sutskever} weights. {#c-count-sutskever}
The LSTM layers hold {shape:recurrent_weights@sutskever} of them, half in each stack, which is the 64M "pure recurrent connections" if every weight of an LSTM layer is counted as recurrent, those on the input included. {#c-count-recurrent}
The rest is in the vocabulary: {shape:embedding_weights@sutskever} in the two embeddings and {shape:softmax_weights@sutskever} in the softmax layer, so more than five sixths of the model is tables of words.
With the biases the model has {shape:params@sutskever} parameters.
The vector $v$ has $2 L d_h$ numbers, {shape:state@sutskever} with these sizes and {shape:state@example} in the example. {#c-state-8000}

## Computation and cost

A teacher-forced forward pass over $B$ pairs of $S$ source and $T$ target tokens costs, counting a multiply-add as 2 FLOPs and only the products,

{shape:flops_forward}

FLOPs, {shape:flops_forward@example} for the example config and 2 pairs of 5 and 4 tokens. {#c-flops}
The two stacks cost the same per token, and the softmax adds $2 d_h V_y$ per target token.
With the sizes of Sutskever et al. the softmax is the largest term: each target token costs 160,000,000 FLOPs in the softmax layer, 2.5 times the 64,000,000 of one step of a stack. {#c-softmax-cost}
For 2 pairs of 5 and 4 tokens the whole pass is {shape:flops_forward@sutskever} FLOPs.
This is why Sutskever et al. gave the softmax as many GPUs as the LSTM layers.
Their C++ implementation of the deep LSTM processed approximately 1,700 words per second on a single GPU; on an 8-GPU machine, with each of the 4 LSTM layers on its own GPU and the remaining 4 GPUs parallelizing the softmax, each multiplying by a 1000 × 20000 matrix, it reached 6,300 words per second with a minibatch size of 128, and training took about ten days. {#c-parallel}

Both stacks are sequential: the encoder cannot start step $s$ before step $s - 1$, and the decoder at inference cannot start step $t$ before it has chosen $y_{t-1}$.
In training, with the target known, the decoder's inputs are all available at once, but the recurrence still runs step by step.
Generating with a beam of $k$ hypotheses runs the decoder on up to $k$ states per step.

### Decoding by search

Generating the most likely output, $\hat{y} = \arg\max_y p(y \mid x)$, is a search over all finite sequences that end in `<EOS>`, and the number of candidates grows exponentially with the length.
By eq-factor, the probabilities of the sequences that end in `<EOS>` within $n$ steps and of the $n$-token prefixes without `<EOS>` add up to 1, so the probability of every output is well defined and those of all outputs add up to at most 1. {#c-normalized}
Each extra token multiplies the probability by a factor of at most 1, so the probability that a target starts with a given prefix never increases as the prefix grows. {#c-prefix-bound}
This is the source of the bias of the search toward short outputs.

The greedy decoder takes the most likely token at each step and feeds it back.
Sutskever et al. search for the most likely translation with a simple left-to-right beam search decoder which maintains a small number B of partial hypotheses: at each timestep each partial hypothesis is extended with every possible word in the vocabulary, all but the B most likely hypotheses are discarded according to the model's log probability, and as soon as `<EOS>` is appended to a hypothesis it is removed from the beam and added to the set of complete hypotheses. {#c-beam-def}
Writing $\mathcal{H}_t$ for the beam after $t$ steps, $k$ for its size and $\ell(y_{1..t}) = \sum_{s \le t} \log p(y_s \mid v, y_{<s})$:

$$ \mathcal{H}_t = \operatorname{top}_k \Big\{ y_{1..t-1} y \;:\; y_{1..t-1} \in \mathcal{H}_{t-1},\ y \in \{1, \dots, V_y\} \Big\} \text{ by } \ell , \qquad \text{with extensions by } \langle \mathrm{EOS} \rangle \text{ moved to the complete set.} $$ {#eq-beam}

A [[al:beam-search|beam search]] of size 1 is the greedy decoder. {#c-beam-greedy}
A beam large enough never to discard an extension, $V_y (V_y - 1)^{n - 1}$ hypotheses for outputs of at most $n$ tokens, finds the most likely output exactly. {#c-beam-exhaustive}
The search may stop as soon as the best complete hypothesis is at least as likely as every partial one, since extending a partial hypothesis can only lower its probability. {#c-beam-stop}
Over $n$ steps a beam of $k$ runs the decoder on at most $n k$ states and scores at most $n k V_y$ extensions, against $\sum_{t=1}^{n} V_y (V_y - 1)^{t-1}$ for an exhaustive search. {#c-beam-cost}
Neither greedy decoding nor a small beam is exact; the worked example below has a model on which greedy decoding and a beam of 2 both miss the most likely output. {#c-greedy-miss}
Sutskever et al. found that their system performs well even with a beam size of 1, and that a beam of size 2 provides most of the benefits of beam search. {#c-beam-small}

### Rescoring

The model can also score the candidates of another system.
Sutskever et al. rescored the 1000-best lists of the baseline by computing the log probability of every hypothesis with the LSTM and taking an even average of the baseline's score and the LSTM's score. {#c-rescore-def}

## Training

### Maximum likelihood with teacher forcing

The model is trained by [[m2:maximum-likelihood-estimation|maximum likelihood]]: it maximizes the log-probability of the target given the source, summed over the training pairs.
By eq-factor this is a sum over target steps of the log-probability of the correct token, so the loss of a batch is a sum of [[al:cross-entropy-loss|cross-entropies]], one per step:

$$ \mathcal{L} = -\frac{1}{B} \sum_{b=1}^{B} \sum_{t=1}^{T_b} \log p\big(y^{b}_t \mid v^{b}, y^{b}_1, \dots, y^{b}_{t-1}\big) . $$ {#eq-loss}

Sutskever et al.'s objective is the average over the training set of the log-probability of the correct translation given the source. {#c-objective}
The decoder computes each conditional from the true previous tokens, which is called teacher forcing.
Summed over the steps this is exactly the log-likelihood of the target, so the decoder run step by step on its own outputs and the decoder fed the whole target at once give the same numbers on the same tokens. {#c-loss-steps}
The log-likelihood of a pair is the sum of the log-probabilities of its tokens, and its gradient with respect to the logits of step $t$ is the difference between the softmax and the one-hot target, divided by $B$. {#c-factor-sum}
The gradient reaches the encoder only through $v$, by [[al:backpropagation-through-time|backpropagation through time]] across both stacks.

### The recipe of Sutskever et al.

Sutskever et al. initialized all the LSTM's parameters with the uniform distribution between -0.08 and 0.08, and used stochastic gradient descent without momentum at a fixed learning rate of 0.7, halving the learning rate every half epoch after 5 epochs, for a total of 7.5 epochs. {#c-recipe}
They used batches of 128 sequences and divided the gradient by the size of the batch, 128. {#c-batch}
Because LSTMs can still have exploding gradients, they computed for each batch s = ‖g‖₂, where g is the gradient divided by 128, and set g = 5g/s when s > 5. {#c-clip}
Since most sentences are short and some are long, they made sure that all sentences in a minibatch were roughly of the same length, yielding a 2x speedup. {#c-bucket}

### Reversing the source

Sutskever et al. found that the LSTM learns much better when the source sentences are reversed, the target sentences being left in their order: the test perplexity dropped from 5.8 to 4.7, and the test BLEU score of the decoded translations increased from 25.9 to 30.6. {#c-reverse-gain}
They give their explanation as a belief, not a complete account: reversing introduces many short term dependencies, because the average distance between corresponding words in the source and target is unchanged but the first few words of the source are now very close to the first few words of the target, so the problem's minimal time lag is greatly reduced. {#c-reverse-why}
The distances can be counted.
With a source of $n$ words read at steps 1 to $n$ and target word $i$ predicted at step $n + i$, every source word is $n$ steps from the target word it corresponds to when read in order. {#c-lag-forward}
With the source reversed, source word $i$ is read at step $n + 1 - i$ and is $2i - 1$ steps from target word $i$: the average is still $n$, but the first word is one step from its translation. {#c-lag-reversed}
They had expected the reversal to help only the early part of the target, but found that LSTMs trained on reversed source sentences did much better on long sentences than LSTMs trained on the raw source sentences, which suggests that reversing the input results in LSTMs with better memory utilization. {#c-reverse-long}
The reversal only makes sense when the two sequences are roughly aligned in order, as in translation between English and French.

### Exposure bias and scheduled sampling

Teacher forcing trains the decoder on prefixes from the data, but at inference it is fed its own outputs.
Bengio, Vinyals, Jaitly and Shazeer argue that this discrepancy between training and inference can yield errors that accumulate quickly along the generated sequence. {#c-ss-problem}
Ranzato et al. named it: they refer to this discrepancy as exposure bias, which occurs when a model is only exposed to the training data distribution, instead of its own predictions. {#c-exposure}

Scheduled sampling is a curriculum between the two regimes.
For every token to predict in the $i$-th mini-batch, Bengio et al. flip a coin and use the true previous token with probability ε_i, or an estimate coming from the model itself with probability 1 − ε_i, the estimate being a token sampled from the model's distribution or its most likely token. {#c-ss-def}
At ε_i = 1 the model is trained exactly as with teacher forcing, and at ε_i = 0 in the same setting as inference. {#c-ss-limits}
They decrease ε_i with the number $i$ of mini-batches seen, with a linear decay ε_i = max(ε, k − c i), an exponential decay ε_i = k^i with k < 1, or an inverse sigmoid decay ε_i = k / (k + exp(i/k)) with k ≥ 1. {#c-ss-schedules}
In their experiments they flipped the coin for every token; flipping it once per sequence gave results as poor as always sampling from the model. {#c-ss-per-sequence}
The gradient is not taken through the sampled tokens, so the loss is still a sum of cross-entropies, each at a state the model may have reached by its own errors.

Ranzato et al. go further and train on a sequence-level reward with REINFORCE: their Mixed Incremental Cross-Entropy Reinforce (MIXER) starts from the policy given by cross-entropy training and slowly deviates from it to make use of its own predictions, as is done at test time. {#c-mixer}

## Inductive biases

**One vector for the whole input.** Everything the decoder knows about the source has to pass through $v$, $2 L d_h$ numbers whatever the length of the source; this makes the model a clean test of what a recurrent state can hold, and is its main weakness on long inputs.

**Autoregressive output.** Each token is predicted from the vector and all the previous tokens, so the model can represent any distribution over output sequences, and the lengths of input and output are independent; the price is a sequential decoder and a search at inference.

**Separate roles.** The encoder and the decoder have their own weights and embeddings, so the two sequences can be in different languages or of different kinds, and the same encoder can feed several decoders.

**Order.** The recurrent encoder is sensitive to the order of the input, and the order chosen changes how far the gradient has to travel between related tokens, which is why reversing the source helps.

## Evidence

**Translation.** Sutskever et al. report that on the WMT'14 English to French task the translations produced by the LSTM achieve a BLEU score of 34.8 on the entire test set, the LSTM being penalized on out-of-vocabulary words, against 33.3 for a phrase-based SMT system on the same dataset, and that reranking the 1000 hypotheses of that system with the LSTM raises its BLEU score to 36.5. {#c-main}
In their Table 1, a single forward LSTM with a beam of 12 scores 26.17, a single reversed LSTM 30.59, and an ensemble of 5 reversed LSTMs 33.00 with a beam of 1, 34.50 with a beam of 2 and 34.81 with a beam of 12, against 33.30 for the baseline system. {#c-table1}
In their Table 2, rescoring the 1000-best lists of the baseline gives 35.61 with a single forward LSTM, 35.85 with a single reversed LSTM and 36.5 with an ensemble of 5 reversed LSTMs, against 37.0 for the best WMT'14 result. {#c-table2}
They trained on a subset of 12M sentences consisting of 348M French words and 304M English words, with the 160,000 most frequent words of the source language and the 80,000 most frequent of the target. {#c-data}

**Long sentences.** Sutskever et al. report that the LSTM did not have difficulty on long sentences. {#c-long}
Their Figure 3 has no degradation on sentences with less than 35 words, and only a minor degradation on the longest sentences. {#c-fig3}

**Depth.** Each additional layer reduced the perplexity by nearly 10%, possibly due to the much larger hidden state of a deeper model. {#c-depth}

**Representations.** The LSTM learned phrase and sentence representations that are sensitive to word order and relatively invariant to the active and the passive voice. {#c-repr}

**Scheduled sampling.** On the MSCOCO image captioning development set, Bengio et al. report a BLEU-4 of 28.8 for the baseline, 30.6 with scheduled sampling and 11.2 for a model always fed its own samples, with METEOR 24.2, 24.3 and 15.7 and CIDEr 89.5, 92.1 and 49.7. {#c-ss-coco}

**The bottleneck.** Bahdanau, Cho and Bengio conjecture that the use of a fixed-length vector is a bottleneck in improving the performance of the basic encoder-decoder architecture, and propose to extend it by allowing the model to automatically (soft-)search for parts of a source sentence that are relevant to predicting a target word. {#c-bottleneck}
In their Table 1, trained on sentences of up to 50 words, the encoder-decoder RNNencdec-50 scores 17.82 BLEU on the full test set and the attention model RNNsearch-50 26.75; RNNsearch-30, trained on sentences of up to 30 words, scores 21.50 and already outperforms RNNencdec-50. {#c-bahdanau-table}
These models are smaller than those of Sutskever et al. and their source is not reversed, so the two sets of numbers on long sentences do not contradict each other directly.
These are reported results; this page has not reproduced them.

## Limitations

- **A fixed-size bottleneck.** The whole source must fit in $v$; Bahdanau et al. trace the loss of quality on long sentences to it, and attention replaced it by a weighted sum of all the encoder states.
- **Sequential computation.** Both stacks run step by step, and the decoder at inference cannot start a token before it has chosen the previous one.
- **Exposure bias.** Trained on true prefixes and tested on its own, the decoder can drift into states it never saw in training.
- **Approximate search.** Greedy and beam search can miss the most likely output, and the most likely output is biased toward short sequences, so practical systems add length normalization or penalties.
- **A closed vocabulary.** With word-level vocabularies, rare words become an unknown token; Sutskever et al.'s scores are penalized for them, and subword vocabularies later removed most of the problem.
- **A mismatch of objectives.** The model is trained on the likelihood of each token and evaluated on sequence-level metrics such as [[bleu|BLEU]], which is what sequence-level training addresses.

## Variants and descendants

- **The RNN Encoder-Decoder** of Cho et al. passes the summary vector $c$ to every step of the decoder and to its output layer, not only as the initial state, and introduced the gated recurrent unit; it was first used to score phrase pairs inside a statistical system.
- **Attention.** Bahdanau et al. keep every encoder state, of a bidirectional encoder, and let the decoder compute at each step a weighted average of them; Luong et al. simplified and compared the forms of the score. Attention removed the fixed-size bottleneck and became the core of the [[transformer]].
- **The Transformer** keeps the encoder-decoder structure, the factorization eq-factor, teacher forcing and beam search, and replaces both recurrent stacks with [[multi-head-attention|multi-head attention]], so that training is parallel over positions.
- **Convolutional encoders and decoders**, from Kalchbrenner and Blunsom's first model to the convolutional sequence-to-sequence models of 2017.
- **Other inputs and outputs.** The same structure with a convolutional image encoder gives image captioning, with an acoustic encoder speech recognition, and with a single shared stack a decoder-only language model, which treats the source as a prefix of the output.
- **Training at the sequence level.** Scheduled sampling, MIXER and minimum risk training change the objective so that the model sees its own outputs or is trained on the final metric.

## Notable models

Sutskever et al.'s deep LSTM on WMT'14 English to French was the first neural model to translate directly, without a phrase-based system, at a BLEU score above that system's.
Cho et al.'s RNN Encoder-Decoder, used as a feature of a phrase-based system, introduced the gated recurrent unit.
Bahdanau et al.'s RNNsearch added attention to it, and Google's neural machine translation system of 2016, an 8-layer LSTM encoder-decoder with attention, brought the design into production before the Transformer of 2017 replaced the recurrent layers.
Show, Attend and Tell (2015) applied the same decoder with attention to images.

## Implementations

The NumPy reference `impl/numpy/seq2seq.py` implements eq-encoder, eq-decoder and eq-step with LSTM layers, the loss eq-loss and its gradient by hand through both stacks, incremental decoding, the greedy decoder and the beam search eq-beam with the complete set and the early stop of Sutskever et al., the schedules of scheduled sampling, and the counts of parameters and FLOPs.
The PyTorch version `impl/pytorch/seq2seq.py` is an `nn.Module` with the same parameters whose two stacks match `torch.nn.LSTM` to rounding after `to_torch_lstm()` copies their weights; it also implements scheduled sampling in `forward_scheduled()`, and is tested with `torch.autograd.gradcheck` and by training it to reverse a sequence.
The JAX version `impl/jax/seq2seq.py` is a set of pure functions that run both stacks with `jax.lax.scan`, tested under `jax.jit` and `jax.vmap` and with `jax.value_and_grad` against the gradients of the NumPy backward pass.
All three load the shared fixture of `fixtures/`: the example config with a reversed source, its logits, its vector $v$, the loss and its gradients, the greedy and beam outputs, and the logits of the same parameters reading the source in order.

### Worked example

Take a decoder with a target vocabulary of three tokens, `<EOS>`, 1 and 2, the model of `test_greedy_can_miss_the_most_likely_output` (seed 0, one layer, a source of 4 tokens).
At the first step it gives `<EOS>` a probability of 0.2207, token 1 0.4918 and token 2 0.2875.
The greedy decoder takes token 1; after it the model gives `<EOS>` 0.3646, token 1 0.3707 and token 2 0.2646, so greedy takes token 1 again, then `<EOS>` at 0.4755, and returns [1, 1, `<EOS>`] with probability $0.4918 \times 0.3707 \times 0.4755 = 0.0867$.
A beam of 2 keeps tokens 1 and 2 at the first step and drops `<EOS>`; at the second step [1, `<EOS>`], of probability $0.4918 \times 0.3646 = 0.1793$, is the most likely extension, and it is complete, so the search stops and returns it.
A beam of 3 moves `<EOS>` alone, of probability 0.2207, to the complete set at the first step and keeps searching until every partial hypothesis has fallen below it; it is the most likely output, and the search returns it.
An exhaustive search of the outputs of at most 4 tokens scores 45 extensions and agrees with the beam of 3.
The most likely output here is the empty one, an instance of the bias toward short outputs that makes practical systems normalize scores by length.
The tests `test_greedy_can_miss_the_most_likely_output`, `test_beam_of_one_is_greedy` and `test_a_beam_that_never_prunes_finds_the_most_likely_output` check these results, and the derivation beam-search recomputes the probabilities.

### Pitfalls

> [!pitfall]
> **The decoder inputs off by one.** In teacher forcing the decoder at step $t$ reads $y_{t-1}$ and predicts $y_t$; feeding it $y_t$ lets it copy the answer, and the training loss falls to near zero while generation fails. The test `test_decoder_inputs_are_the_target_shifted_by_one` checks the shift, and `test_loss_is_the_sum_of_the_steps` checks that incremental decoding, which can only see the past, gives the same loss as teacher forcing.

> [!pitfall]
> **Reversing the target, or the padding.** Only the source is reversed; reversing the target as well makes the model generate backwards. With padded batches, the padding must stay out of the reversal and out of $v$: reversing a padded row puts the padding first, which changes the final state. The test `test_reversal_is_reading_the_reversed_source` checks that the reversed model is exactly the forward model reading the reversed source, and that the reversal changes the output.

> [!pitfall]
> **Losing the source.** If $v$ is not passed to the decoder, or only the hidden state is passed and not the cell, the decoder is a language model that ignores its input and still trains to a reasonable loss. The test `test_the_source_reaches_the_decoder_only_through_v` checks that two sources with different vectors give different outputs, and that decoding step by step from $v$ alone reproduces the logits of the whole model.

> [!pitfall]
> **Scoring by the sum of log-probabilities.** Comparing hypotheses of different lengths by their log-probability favors short ones, by eq-factor; a beam large enough may return `<EOS>` alone. The test `test_longer_outputs_are_never_more_likely_than_their_prefixes` checks the bound, and the worked example is a model on which the empty output is the most likely.

> [!pitfall]
> **Averaging the loss over tokens or over sequences.** eq-loss sums over the tokens of a sequence and averages over the batch, as Sutskever et al. divide the gradient by 128; averaging over all the tokens of the batch instead weights short and long targets differently and changes the effective learning rate and clipping threshold. The fixtures and the gradient tests use the sum per sequence.

## Experiments

This page has no experiment yet.
The PyTorch test `test_learns_to_reverse_a_sequence` is a small one: a model with one layer of 64 cells, trained for 400 steps of Adam with the gradient clipped to norm 5, learns to write 5 tokens back in reverse order with a sequence accuracy above 90% on 500 fresh sequences, everything passing through $v$.
The checks of the four derivations count parameters and products with an instrumented matrix product, verify the factorization and the normalization over all outputs, compare the beam search with an exhaustive one, and count the time lags with and without reversal.

## History

Bahdanau et al. trace neural machine translation to Kalchbrenner and Blunsom (2013), Sutskever et al. (2014) and Cho et al. (2014b), and place most of the models proposed by 2014 in a family of encoder-decoders. {#c-hist-family}
Sutskever et al. describe Kalchbrenner and Blunsom as the first to map the input sentence into a vector and then back to a sentence, although they map sentences to vectors with convolutional neural networks, which lose the ordering of the words. {#c-hist-kb}
They describe Cho et al. as using an LSTM-like RNN architecture to map sentences into vectors and back, with their primary focus on integrating the network into an SMT system. {#c-hist-cho}
In their conclusion they report that they were unable to train a standard RNN on the non-reversed translation problem, and that they believe one should be easily trainable on reversed sources, which they did not verify experimentally. {#c-hist-rnn}
Bahdanau, Cho and Bengio's attention, the same year, and Luong et al.'s study of its forms in 2015, turned the encoder-decoder into the attentional model that dominated machine translation until 2017.
Bengio et al. used scheduled sampling in their winning entry to the MSCOCO image captioning challenge of 2015. {#c-hist-coco}
The Transformer of Vaswani et al. (2017) kept the encoder-decoder framing of sequence transduction and its training and decoding, and dropped the recurrence.

## References

- I. Sutskever, O. Vinyals and Q. V. Le. Sequence to Sequence Learning with Neural Networks. NeurIPS 2014; arXiv 1409.3215.
- K. Cho, B. van Merriënboer, C. Gulcehre, D. Bahdanau, F. Bougares, H. Schwenk and Y. Bengio. Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. EMNLP 2014; arXiv 1406.1078.
- D. Bahdanau, K. Cho and Y. Bengio. Neural Machine Translation by Jointly Learning to Align and Translate. ICLR 2015; arXiv 1409.0473.
- M.-T. Luong, H. Pham and C. D. Manning. Effective Approaches to Attention-based Neural Machine Translation. EMNLP 2015.
- S. Bengio, O. Vinyals, N. Jaitly and N. Shazeer. Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks. NeurIPS 2015; arXiv 1506.03099.
- M. Ranzato, S. Chopra, M. Auli and W. Zaremba. Sequence Level Training with Recurrent Neural Networks. ICLR 2016; arXiv 1511.06732.
- N. Kalchbrenner and P. Blunsom. Recurrent Continuous Translation Models. EMNLP 2013.
- S. Hochreiter and J. Schmidhuber. Long Short-Term Memory. Neural Computation 9(8), 1997.
- A. Vaswani et al. Attention Is All You Need. NeurIPS 2017.
- PyTorch contributors. torch.nn.LSTM. PyTorch 2.15 documentation.
