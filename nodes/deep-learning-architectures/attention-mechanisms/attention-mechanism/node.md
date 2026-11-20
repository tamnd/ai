---
tag: B95BYJ
kind: method
slug: attention-mechanism
name: Attention mechanism
aliases: [attention, neural attention, soft attention, additive attention, Bahdanau attention, Luong attention, content-based attention, soft alignment]
area: deep-learning-architectures/attention-mechanisms
level: core
summary: >-
  At each output step, score every input position against the current state, turn the scores into weights by a softmax, and read the weighted mean of the inputs as that step's context.
purpose: let a decoder read a different part of its input at every output step instead of one fixed-size summary of all of it
uses: [matrix-multiplication@m2, softmax-regression@al, backpropagation@al]
related_to: [sequence-to-sequence-model, multi-head-attention, transformer, recurrent-neural-network, gated-recurrent-unit, bidirectional-recurrent-neural-network, long-short-term-memory, beam-search@al]
hyperparameters:
  - {name: score function, meaning: "how a query is compared with a source vector", typical: "additive (concat), v_a . tanh(W_a s + U_a h), in Bahdanau et al.; dot or general in Luong et al.; the scaled dot product in the Transformer"}
  - {name: n', meaning: "the hidden width of the additive alignment model", typical: "1000 in Bahdanau et al., equal to the decoder width"}
  - {name: query, meaning: "the state compared with the source", typical: "the previous decoder state s_{i-1} in Bahdanau et al.; the current top-layer state h_t in Luong et al."}
  - {name: D, meaning: "the half-width of the window of local attention", typical: "10 in Luong et al.; global attention has no window"}
  - {name: mask, meaning: "which source positions may receive weight", typical: "the real words of each sentence of a padded batch"}
cost:
  train: "4 T_x n n' + 2 T_y n n' + 2 T_x T_y (n' + 2n) FLOPs per sentence pair for the additive model with U_a h_j computed once, about 7% of the forward pass of RNNsearch at 30 words; the T_x T_y weights are stored for the backward pass"
  inference: "per generated word, one product W_a s_{i-1} and T_x scores and weighted sums; the projected annotations U_a h_j are computed once per source sentence and kept"
history:
  year: 2014
  people: [Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio]
  note: "Bahdanau, Cho and Bengio posted Neural Machine Translation by Jointly Learning to Align and Translate to arXiv (1409.0473) in September 2014, published at ICLR 2015. Graves (2013) had used a monotonic Gaussian window in handwriting synthesis; Luong, Pham and Manning (EMNLP 2015) compared global and local attention and the dot, general and concat scores, and Vaswani et al. (2017) built the Transformer from attention alone."
prereqs: [sequence-to-sequence-model, gated-recurrent-unit, bidirectional-recurrent-neural-network, softmax-regression@al, backpropagation@al]
same_as:
  wikidata: null
  wikipedia: "Attention (machine learning)"
  arxiv: ["1409.0473", "1508.04025"]
sources:
  - {paper: "Bahdanau, Cho and Bengio (2015)", title: "Neural Machine Translation by Jointly Learning to Align and Translate", locator: "arXiv 1409.0473, ICLR 2015, Sections 1 to 6, Table 1, Figures 2 and 3, Appendices A.1.2, A.2 and B.1", note: "the alignment model, the encoder, the decoder, the sizes, the initialization and the experiments"}
  - {paper: "Luong, Pham and Manning (2015)", title: "Effective Approaches to Attention-based Neural Machine Translation", locator: "EMNLP 2015, Sections 3.1 to 3.3, 4.2 and 5.2 to 5.4, Tables 1, 4 and 6", note: "global and local attention, the score functions, input feeding and their comparison"}
---

An attention mechanism lets a network that writes an output sequence look back at its input at every step and decide which parts of it to use.
At each output step it scores every input position against the current state of the network, turns the scores into weights that are positive and sum to 1, and takes the weighted mean of the input vectors as a context vector for that step.
Bahdanau, Cho and Bengio introduced it in 2014 for neural machine translation, where it replaced the single fixed-size vector through which a [[sequence-to-sequence-model|sequence-to-sequence model]] had to pass its whole source sentence.
Bahdanau et al. conjecture that the use of a fixed-length vector is a bottleneck in improving the performance of the basic encoder-decoder architecture, and propose to extend it by allowing a model to automatically (soft-)search for parts of a source sentence that are relevant to predicting a target word, without having to form these parts as a hard segment explicitly. {#c-bottleneck}
Luong, Pham and Manning then compared several forms of it, and the scaled dot-product form became the building block of [[multi-head-attention|multi-head attention]] and the [[transformer|Transformer]].

## In plain words

A translator who reads a long sentence once and then writes the translation from memory loses details, and the longer the sentence, the more.
A translator who can glance back at the source while writing does better: before each word, they look over the source, find the words that matter for the next word, and read those closely.
Attention gives a network that second habit.
The encoder writes one vector per source word, a note on that word and its neighbours.
Before each output word the decoder compares its current state with every note, gives each note a score, and turns the scores into shares of one unit of attention.
The context for the step is the mix of the notes in those shares.
The shares are a soft alignment: they say which source words the output word comes from, and since they change smoothly with the weights of the network, gradient descent learns them from translated sentences alone, with no aligned words to copy.

## Setting

The task is to map a source sequence $x = (x_1, \dots, x_{T_x})$ to a target sequence $y = (y_1, \dots, y_{T_y})$, as in machine translation.
A plain encoder-decoder reads $x$ into one vector $c$ and generates $y$ from it.
A potential issue with the encoder-decoder approach is that a neural network needs to be able to compress all the necessary information of a source sentence into a fixed-length vector, which may make it difficult to cope with long sentences, especially those that are longer than the sentences in the training corpus. {#c-long-issue}
The model of Bahdanau et al. does not attempt to encode a whole input sentence into a single fixed-length vector; instead, it encodes the input sentence into a sequence of vectors and chooses a subset of these vectors adaptively while decoding the translation. {#c-sequence-of-vectors}
They call the model RNNsearch and the plain model it extends RNNencdec, the RNN Encoder-Decoder of Cho et al. (2014).

## The method

**Annotations.**
A [[bidirectional-recurrent-neural-network|bidirectional RNN]] with [[gated-recurrent-unit|gated recurrent units]] reads the source forward into states $\overrightarrow{h}_1, \dots, \overrightarrow{h}_{T_x}$ and backward into $\overleftarrow{h}_1, \dots, \overleftarrow{h}_{T_x}$, and the annotation of position $j$ is

$$ h_j = \big[\overrightarrow{h}_j ; \overleftarrow{h}_j\big] \in \mathbb{R}^{2n} . $$ {#eq-annot}

The annotation h_j of each word concatenates the forward and the backward hidden states of a bidirectional RNN, so that it contains the summaries of both the preceding words and the following words; due to the tendency of RNNs to better represent recent inputs, it will be focused on the words around x_j. {#c-birnn}

**Scores, weights and context.**
At target step $i$, with $s_{i-1}$ the previous state of the decoder, every annotation gets an energy

$$ e_{ij} = a(s_{i-1}, h_j) = v_a^\top \tanh\big(W_a s_{i-1} + U_a h_j\big) , $$ {#eq-score}

the energies of the step become weights by a softmax over the source positions,

$$ \alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})} , $$ {#eq-weights}

and the context of the step is the weighted sum of the annotations,

$$ c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j . $$ {#eq-context}

The energy e_ij = a(s_{i-1}, h_j) is given by an alignment model which scores how well the inputs around position j and the output at position i match, based on the RNN hidden state s_{i-1} and the j-th annotation h_j. {#c-alignment-model}
Since the alignment model needs to be evaluated Tx × Ty times for each sentence pair, Bahdanau et al. use a single-layer multilayer perceptron, a(s_{i-1}, h_j) = v_a^T tanh(W_a s_{i-1} + U_a h_j). {#c-additive}
Here $W_a$ maps the $n$-dimensional decoder state and $U_a$ the $2n$-dimensional annotation to $n'$ hidden units, and $v_a \in \mathbb{R}^{n'}$; the score is called additive, after the sum inside the $\tanh$, or concat, after the equivalent form of Luong et al.
Since U_a h_j does not depend on i, it can be pre-computed in advance to minimize the computational cost. {#c-precompute-attr}

**Decoder.**
Unlike the existing encoder-decoder approach, the probability of each target word is conditioned on a distinct context vector c_i, computed as a weighted sum of the annotations to which an encoder maps the input sentence. {#c-distinct-context}
The decoder is a GRU whose three gates also read the context,

$$ s_i = \mathrm{GRU}\big(s_{i-1}, E y_{i-1}, c_i\big), \qquad s_0 = \tanh\big(W_s \overleftarrow{h}_1\big) , $$ {#eq-decoder}

with $C$, $C_z$ and $C_r$ the matrices that bring $c_i$ into the candidate state and the update and reset gates.
The initial hidden state of the decoder is s_0 = tanh(W_s h_1), computed from the first backward hidden state of the encoder. {#c-s0}
The output is a deep output with one maxout layer: with $\tilde t_i = U_o s_{i-1} + V_o E y_{i-1} + C_o c_i \in \mathbb{R}^{2l}$ and $t_i$ the maximum of each pair of its entries,

$$ p(y_i \mid y_{<i}, x) \propto \exp\big(y_i^\top W_o t_i\big) . $$ {#eq-output}

The probability of a target word is computed from the decoder state s_{i-1}, the context c_i and the last generated word y_{i-1} through a deep output with a single maxout hidden layer. {#c-deep-output}
Training minimizes the negative log-likelihood of the target words,

$$ \mathcal{L} = -\frac{1}{N} \sum_{(x, y)} \sum_{i=1}^{T_y} \log p(y_i \mid y_{<i}, x) , $$ {#eq-loss}

with $N$ the number of target words of the batch, and gradient descent trains the alignment model with everything else.
Unlike in traditional machine translation, the alignment is not considered to be a latent variable: the alignment model directly computes a soft alignment, which allows the gradient of the cost function to be backpropagated through, so that the alignment model and the whole translation model are trained jointly. {#c-not-latent}

**Reading the weights.**
Taking a weighted sum of all the annotations can be understood as computing an expected annotation, where the expectation is over possible alignments, alpha_ij being a probability that the target word y_i is aligned to, or translated from, a source word x_j. {#c-expected}
Bahdanau et al. describe this as a mechanism of attention in the decoder: by letting the decoder decide parts of the source sentence to pay attention to, they relieve the encoder from the burden of having to encode all information in the source sentence into a fixed-length vector, and the information can be spread throughout the sequence of annotations, which can be selectively retrieved by the decoder. {#c-mechanism}
The model becomes the RNN Encoder-Decoder of Cho et al. if c_i is fixed to the last forward hidden state of the encoder. {#c-encdec-fixed}

**The forms of Luong et al.**
Luong et al. examine two classes of attentional mechanism: a global approach which always attends to all source words and a local one that only looks at a subset of source words at a time. {#c-luong-classes}
Their decoder is a stack of LSTM layers, and attention comes after the top layer:
Given the target hidden state h_t at the top layer of a stacking LSTM and the source-side context vector c_t, a concatenation layer produces the attentional hidden state h̃_t = tanh(W_c [c_t; h_t]), which is fed through the softmax layer. {#c-luong-htilde}

$$ \tilde h_t = \tanh\big(W_c [c_t ; h_t]\big), \qquad p(y_t \mid y_{<t}, x) = \mathrm{softmax}(W_s \tilde h_t) . $$ {#eq-luong-out}

The content-based score of global attention is one of three alternatives: dot, h_t^T h̄_s; general, h_t^T W_a h̄_s; and concat, v_a^T tanh(W_a [h_t; h̄_s]); a location-based function computes a_t = softmax(W_a h_t) from solely the target hidden state. {#c-luong-scores}

$$ \mathrm{score}(h_t, \bar h_s) = \begin{cases} h_t^\top \bar h_s & \text{dot} \\ h_t^\top W_a \bar h_s & \text{general} \\ v_a^\top \tanh\big(W_a [h_t ; \bar h_s]\big) & \text{concat} \end{cases} \qquad a_t = \mathrm{softmax}(W_a h_t) \ \text{(location)} . $$ {#eq-luong-scores}

Local attention derives the context vector as a weighted average over the set of source hidden states within the window [p_t - D, p_t + D], D being empirically selected; local-m simply sets p_t = t, assuming that source and target sequences are roughly monotonically aligned. {#c-local-def}
Local-p predicts the aligned position p_t = S sigmoid(v_p^T tanh(W_p h_t)), S being the source sentence length, and multiplies the alignment weights by a Gaussian centered around p_t whose standard deviation is set as D/2, which lets backpropagation compute gradients for W_p and v_p. {#c-local-p-def}

$$ p_t = S \cdot \sigma\big(v_p^\top \tanh(W_p h_t)\big), \qquad a_t(s) = \mathrm{align}(h_t, \bar h_s) \exp\Big(-\frac{(s - p_t)^2}{2\sigma^2}\Big), \quad \sigma = \frac{D}{2} , $$ {#eq-local-p}

with align the softmax of the scores over the window and $s$ an integer position in it.
In the input-feeding approach the attentional vectors h̃_t are concatenated with inputs at the next time steps, so that alignment decisions take past alignment information into account. {#c-input-feeding}
Luong et al. use hidden states at the top LSTM layers in both the encoder and decoder and go from h_t to a_t to c_t to h̃_t, whereas Bahdanau et al. build from the previous hidden state h_{t-1} to a_t to c_t to h_t, and only experimented with one alignment function, the concat product. {#c-luong-path}

## Derivation

**The context is an expectation.**
Each context vector c_i is the expected annotation under the distribution alpha_i over the source positions, so it lies in the convex hull of the annotations of the real positions. {#c-convex}
Each coordinate of $c_i$ therefore lies between the smallest and the largest value of that coordinate over the source, and attention can only interpolate the annotations, never extrapolate from them.
With v_a = 0 every energy is 0, and every context vector is the mean of the annotations of the real positions. {#c-uniform}
If the context vector is the same at every step, the decoder is that of an RNN Encoder-Decoder conditioned on the fixed vector, and the alignment model has no effect on the output. {#c-fixed}
The derivation `expected-annotation` gives the proofs and checks them on random models.

**The gradient at the energies.**
Write $g_i = \partial \mathcal{L} / \partial c_i$ and $\gamma_{ij} = g_i^\top h_j$, how much the loss would change per unit of weight moved onto $h_j$.
Through the softmax,

$$ \frac{\partial \mathcal{L}}{\partial e_{ij}} = \alpha_{ij}\Big(\gamma_{ij} - \sum_k \alpha_{ik} \gamma_{ik}\Big) , $$

so an energy rises when its annotation helps more than the current context does, and the more weight it already has, the faster.
With gamma_ij the inner product of the gradient of the loss at c_i with h_j, the gradient of the loss at the energy e_ij is alpha_ij (gamma_ij minus the alpha_i-weighted mean of gamma_i); it is zero at a masked position and at every position when the real annotations are all equal, and the gradients of the energies of one step sum to zero. {#c-grad-energy}
The annotation $h_j$ receives $\alpha_{ij} g_i$ from step $i$, so the encoder learns through the alignment as well as through $s_0$.
W_a and U_a are initialized by sampling each element from the Gaussian distribution of mean 0 and variance 0.001 squared, and all the elements of v_a and all the bias vectors are initialized to zero. {#c-init}
With v_a = 0, as Bahdanau et al. initialize it, the weights are uniform and the gradients at W_a and U_a are exactly zero, while the gradient at v_a is not, so the first update moves v_a alone. {#c-zero-init}
Training therefore starts from the mean of the annotations and learns to sharpen it.
The derivation `alignment-gradient` gives the steps and checks them against finite differences.

**Parameters and cost.**
For all the models of Bahdanau et al., the size of a hidden layer n is 1000, the word embedding dimensionality m is 620, the size of the maxout hidden layer in the deep output l is 500, and the number of hidden units in the alignment model n' is 1000. {#c-sizes}
At the sizes of Bahdanau et al. and with shortlists of 30,000 words, RNNsearch without biases has 80,401,000 weights, of which the alignment model has 3,001,000, or 3.7%. {#c-params}
For a pair of sentences of 30 words, the alignment model and the contexts take 185,400,000 FLOPs with U_a h_j computed once per sentence and 3,665,400,000 if it is recomputed at every target step, a factor of about 20. {#c-precompute}
With the precomputation, attention is 7.3% of the 2,539,400,000 FLOPs of the forward pass of RNNsearch on a pair of sentences of 30 words, and its cost grows as the product of the two lengths. {#c-share}
The derivation `parameter-and-cost` counts the products of the NumPy reference on random small models and evaluates the formulas at the paper's sizes.

**How the scores relate.**
The general score with W_a the identity is the dot score, and every general score is the dot score of a linear map of the query. {#c-dot-general}
The concat score of Luong et al., with W_a split by columns into two blocks, is the additive score of Bahdanau et al. with the query h_t in place of s_{i-1}. {#c-concat-additive}
So the scores fall in two families, the multiplicative dot and general, which cost one product per pair of positions, and the additive concat, which runs a small network on every pair.
Local-m attention is global attention when its window covers the whole sentence. {#c-local-m-global}
The weights of local-p attention sum to at most 1, and to less than 1 whenever p_t is not an integer, so its context is a weighted sum of the source states but not an average of them. {#c-local-sum}
The location score softmax(W_a h_t) does not depend on the source states. {#c-location-blind}
The derivation `luong-scores` proves the five relations and checks them numerically.

## Worked example

Take a decoder state $s = (0.5, -0.5)$ and three annotations $h_1 = (1, 0)$, $h_2 = (0, 1)$ and $h_3 = (1, 1)$, with $W_a = U_a = I$ and $v_a = (1, 1)$, so that $n' = 2$.
The pre-activations $s + h_j$ are $(1.5, -0.5)$, $(0.5, 0.5)$ and $(1.5, 0.5)$, and their $\tanh$ is $(0.9051, -0.4621)$, $(0.4621, 0.4621)$ and $(0.9051, 0.4621)$.
Summing the coordinates, as $v_a = (1, 1)$ does, gives the energies $e = (0.4430, 0.9242, 1.3673)$.
The softmax gives the weights $\alpha = (0.1946, 0.3149, 0.4905)$, and the context is
$$ c = 0.1946\,(1, 0) + 0.3149\,(0, 1) + 0.4905\,(1, 1) = (0.6851, 0.8054) . $$
The dot score of Luong et al. on the same vectors gives $s^\top h_j = (0.5, -0.5, 0)$ and the weights $(0.5065, 0.1863, 0.3072)$: it puts the most weight on $h_1$, the annotation that points the way $s$ does, while the additive score, whose $\tanh$ saturates the large first coordinate, puts it on $h_3$.
The two scores are not two scalings of one ranking, and which one a model has changes what it can learn to look at.
The test `test_worked_example` of the NumPy reference recomputes these numbers.

## Hyperparameters

The **score function** is the main choice.
The additive score has $n'(n + 2n) + n'$ weights and a hidden layer of its own; the dot score has none and needs the query and the source vectors to have the same width; the general score adds one $n \times n$ matrix.
The **width $n'$** of the additive model sets its capacity; Bahdanau et al. take it equal to the decoder width.
The **query** is the state that is compared with the source: Bahdanau et al. use the state before the step, Luong et al. the state after it, and the two orders lead to different decoders, described under Variants.
For local attention the **half-width $D$** trades cost against reach; Luong et al. set it to 10 in their experiments.
A **mask** sets the weight of every padding position to exactly 0, as eq-weights sums only over the real positions of each sentence.

## Cost

The alignment model is evaluated for every pair of a source and a target position, $T_x T_y$ pairs per sentence pair.
With $U_a h_j$ computed once per sentence, the additive model costs
$$ 4 T_x n n' + 2 T_y n n' + 2 T_x T_y (n' + 2n) $$
FLOPs per sentence pair, the last term being the scores and the weighted sums.
At the sizes of Bahdanau et al. this is under a tenth of the arithmetic and under a twentieth of the weights of RNNsearch, as the derivation computes.
The approach requires computing the annotation weight of every word in the source sentence for each word in the translation; this drawback is not severe in translation, in which most of input and output sentences are only 15–40 words, but it may limit the applicability of the scheme to other tasks. {#c-quadratic}
Training stores the $T_x T_y$ weights of each sentence pair for the backward pass, and decoding keeps the projected annotations of the source, $T_x n'$ numbers, for as long as the sentence is being written.

## Evidence

**Translation quality.**
Bahdanau et al. train each model with minibatch stochastic gradient descent and Adadelta on minibatches of 80 sentences for approximately 5 days, and find translations with a beam search. {#c-training}
Bahdanau et al. use a shortlist of 30,000 most frequent words in each language, map any other word to a special token [UNK], and evaluate on news-test-2014 from WMT '14, which consists of 3003 sentences not present in the training data. {#c-shortlist}
On all the sentences of the test set, RNNencdec-30, RNNsearch-30, RNNencdec-50, RNNsearch-50 and RNNsearch-50* obtain BLEU scores of 13.93, 21.50, 17.82, 26.75 and 28.45, against 33.30 for Moses; on the sentences without any unknown word they obtain 24.19, 31.44, 26.71, 34.16 and 36.15, against 35.63. {#c-table1}
The suffix is the length of the training sentences, up to 30 or 50 words, and RNNsearch-50* was trained until the development performance stopped improving.
In all the cases RNNsearch outperforms the conventional RNNencdec, and its performance is as high as that of the phrase-based system Moses when only the sentences consisting of known words are considered, although Moses uses a separate monolingual corpus of 418M words. {#c-outperforms}

**Long sentences.**
The performance of RNNencdec dramatically drops as the length of the sentences increases, while RNNsearch-30 and RNNsearch-50 are more robust to the length of the sentences, RNNsearch-50 having no performance deterioration even with sentences of length 50 or more. {#c-length}
RNNsearch-30 even outperforms RNNencdec-50. {#c-30-beats-50}
The attentional models are more effective than the non-attentional model in handling long sentences: the translation quality does not degrade as sentences become longer. {#c-luong-long}
Luong et al. report this for English to German, and it is the effect the fixed-vector bottleneck predicts.

**Alignments.**
The alignment of words between English and French found by RNNsearch-50 is largely monotonic, with strong weights along the diagonal, and there are a number of non-trivial, non-monotonic alignments: the model aligns [zone] with [Area], jumping over [European] and [Economic], and then looks one word back at a time to complete the phrase. {#c-monotonic}
The phrase is [European Economic Area], translated as [zone économique européenne].
A hard alignment would map [the] to [l'] and [man] to [homme], which is not helpful because the word following [the] determines whether it should be translated into [le], [la], [les] or [l']; the soft alignment lets the model look at both [the] and [man], and it deals with source and target phrases of different lengths without mapping some words to or from nowhere ([NULL]). {#c-the-man}
The one-to-one alignments extracted from the attentional models achieve AER scores comparable to the one-to-many alignments of the Berkeley aligner, and those of the local attention models achieve lower AERs than those of the global one. {#c-luong-aer}

**The forms of attention.**
On top of a model with a reversed source and dropout, global attention gives a boost of +2.8 BLEU, input feeding another +1.3 BLEU, and local attention with predictive alignments a further +0.9 BLEU, in total 5.0 BLEU over the non-attentional baseline. {#c-luong-gains}
Dot works well for the global attention and general is better for the local attention; among the models of Table 4, local attention with predictive alignments is best both in terms of perplexities and BLEU. {#c-luong-dot-general}
The location-based function does not learn good alignments, and the implementation of concat by Luong et al. does not yield good performances, with perplexities of 6.7 (global), 7.1 (local-m), and 7.1 (local-p). {#c-luong-location-concat}
Luong et al. did not train every combination of model and score, and they leave the reason for the result of concat open, so the comparison is between their implementations at one setting and not a ranking of the scores in general.
An ensemble of different attention approaches obtains 23.0 BLEU on newstest2014, outperforming the existing best system (Jean et al., 2015) by +1.4 BLEU, and 25.9 BLEU on the WMT'15 English to German task, an improvement of 1.0 BLEU points over the existing best system backed by NMT and an n-gram reranker. {#c-luong-ensemble}

## Limitations and failure modes

**Quadratic cost.** Every target position scores every source position, so time and the memory of the weights grow as $T_x T_y$; this is small for sentences and large for documents, and it is the reason for local attention and, later, for sparse and linear forms of attention.

**Attention is not a proof of use.** The weights say where the context was read from, not what the decoder did with it; the decoder state and the output layer also carry information, and a weight near 0 on a word does not prove that the word had no effect through the annotations of its neighbours, which a bidirectional encoder spreads.

**Interpolation only.** The context is a convex combination of the annotations, so with a smooth softmax it blurs several positions together; a model that must copy one source word exactly needs sharp weights, which it reaches only with large energies.

**No memory of past reading.** In the form of Bahdanau et al. and in global attention without input feeding, nothing in the score says which source words were already translated, so a model can attend twice to the same word and skip another, which shows up as repeated or dropped words; input feeding and coverage models address it.

**A one-way window in early forms.** Graves (2013) used a mixture of Gaussian kernels to compute the weights of the annotations in handwriting synthesis, with the location restricted to increase monotonically, which Bahdanau et al. call a severe limitation in machine translation, as long-distance reordering is often needed. {#c-graves}
Local-m attention has the same weakness, and local-p trades it for a position that has to be predicted.

## Variants

**Additive or concat attention** is the form of Bahdanau et al., eq-score, with the previous decoder state as the query.
**Global attention** of Luong et al. scores all source positions with the current top-layer state, through dot, general or concat, eq-luong-scores, and feeds the context into the attentional state eq-luong-out.
**Location-based attention** computes the weights from the query alone.
**Local attention** restricts the softmax to a window around a position that is $t$ in local-m and predicted in local-p, eq-local-p.
**Input feeding** passes the attentional state $\tilde h_t$ to the next step, so the decoder knows where it looked.
**Hard attention** samples one position instead of averaging, and is trained with variance reduction or reinforcement learning because sampling is not differentiable.
**Scaled dot-product attention** divides the dot score by $\sqrt{d_k}$ and separates keys from values; run several times in parallel it is [[multi-head-attention|multi-head attention]], and applied by a sequence to itself, self-attention, it is the layer of the [[transformer|Transformer]], which drops the recurrent network altogether.

## When to use it

Use attention whenever a model writes one sequence from another and the input is too long, or too detailed, to squeeze into one vector: translation, summarization, speech recognition and image captioning are the classic cases.
In a recurrent encoder-decoder the additive form with a mask is a sound default, and the dot or general form is cheaper when the widths match.
For new models the multi-head scaled dot-product form of the Transformer has largely replaced the recurrent forms on this page; the recurrent forms remain the clearest setting in which to see what attention adds, as the reversal experiment of this page does.
If the inputs are very long and the alignment is close to monotonic, as in speech, a local or monotonic form saves the quadratic cost.

## Pitfalls

> **Masking the padding.** The softmax of eq-weights runs over the real words of each sentence. Without a mask, the pad positions of a batch get weight and the context of a short sentence depends on the batch it is in; set their energies to minus infinity, so that their weights are exactly 0. The test `test_padding_changes_nothing` checks that a padded sentence gets the logits it gets alone.

> **Which decoder state the output reads.** Eq. 4 of Bahdanau et al. writes the output as $g(y_{i-1}, s_i, c_i)$, while Appendix A.2.2 computes $\tilde t_i$ from $s_{i-1}$. The implementations of this page follow the appendix: at step $i$ the output and the alignment both read $s_{i-1}$, and $s_i$ is computed after. Mixing the two puts the logits one state out of step.

> **Initializing $v_a$ at zero.** With $v_a = 0$ the gradient at $W_a$ and $U_a$ is exactly 0 at the first step, by c-zero-init; this is harmless with the small Gaussian of Bahdanau et al. for $W_a$ and $U_a$, but if $W_a$ and $U_a$ are also initialized at 0 the alignment model can never leave uniform weights.

> **The fixed context of the baseline.** Bahdanau et al. fix $c_i$ to $\overrightarrow{h}_{T_x}$ to recover RNNencdec. Our `attention=False` fixes it to $[\overrightarrow{h}_{T_x}; \overleftarrow{h}_1]$, the two halves that have read the whole sentence, so that the shapes of $C$ and $C_o$ stay those of RNNsearch; it is a choice of this page, not of the paper.

> **The first input word.** The decoder needs $E y_0$ before any word is written. The implementations here use a zero vector; a start token with its own embedding is the common alternative, and the two give different numbers.

> **Local-p does not renormalize.** Eq. 11 multiplies the softmax weights by the Gaussian and does not divide by the new sum, so the weights sum to less than 1 by c-local-sum and the context is shrunk toward 0. An implementation that renormalizes is a different model.

> **Recomputing $U_a h_j$.** Computing $U_a h_j$ inside the loop over target steps gives the same numbers at about 20 times the cost of the alignment model, by c-precompute.

## Implementations

The NumPy reference `impl/numpy/attention.py` implements RNNsearch: the bidirectional GRU encoder eq-annot with masks, the alignment model eq-score, eq-weights and eq-context, the decoder eq-decoder with the deep maxout output eq-output, the loss eq-loss and its gradient by hand through the alignment and both GRUs, and a greedy decoder; with `attention=False` it is the same network with a fixed context.
It also implements the layer of Luong et al.: the four scores eq-luong-scores, global attention, local-p attention eq-local-p and the attentional state eq-luong-out.
The PyTorch version `impl/pytorch/attention.py` is an `nn.Module` with the same parameters, tested against the fixture and the gradients of the NumPy backward pass, and a `LuongAttention` layer tested against the NumPy functions for every score with and without a window.
The JAX version `impl/jax/attention.py` is a set of pure functions that run the GRUs with `jax.lax.scan`, tested under `jax.jit` and against the same gradients through `jax.value_and_grad`.
All three load the shared fixture of `fixtures/`: a small RNNsearch with a padded second sentence, its logits, weights, annotations, loss and gradients, and the logits of the same parameters with the context fixed.

## Experiments

The PyTorch test `test_attention_removes_the_bottleneck` trains RNNsearch and the same network with a fixed context, both of width 16, to write a sequence of 12 random words from a vocabulary of 9 in reverse order, for 300 steps of Adam on batches of 32.
Over the seeds 0 to 3, RNNsearch got 0.996 to 1.0 of the output words of 256 fresh sequences right by greedy decoding, and the network with a fixed context 0.29 to 0.32, chance being 1/9.
The fixed-context network gets 0.63 to 0.80 of the first output word, the last word it read, and 0.17 to 0.31 of each word from the fifth on: a vector of 16 numbers holds the recent words and loses the early ones.
With attention, the largest weight of step $i$ sits on source position $13 - i$, the anti-diagonal of the reversal, for 0.89 to 0.99 of the output words, so the alignment is learned from the outputs alone.
The checks of the four derivations verify the gradient at the energies against finite differences, the expectation and the fixed-context limit, the counts of weights and products on random small models, and the relations between the scores.

## History

The idea of reading an input through learned, input-dependent weights is older than its name in neural translation: Graves (2013) generated handwriting with a soft window that moved along the text, and neural memory models of 2014 read their memories through a softmax of similarities.
Bahdanau, Cho and Bengio posted their paper in September 2014, the same month as the sequence-to-sequence model of Sutskever, Vinyals and Le appeared at NeurIPS, and presented it at ICLR 2015, where it changed neural translation from compressing a sentence to aligning with it.
Xu et al. (2015) carried soft and hard attention to image captioning, Chorowski et al. (2015) to speech recognition, and Luong, Pham and Manning (2015) compared global and local attention and the scores, and introduced input feeding.
Vaswani et al. (2017) kept the attention and dropped the recurrence, with the scaled dot product of [[multi-head-attention|multi-head attention]], and the [[transformer|Transformer]] made attention the main operation of deep learning for language and much else.

## References

- D. Bahdanau, K. Cho and Y. Bengio. Neural Machine Translation by Jointly Learning to Align and Translate. ICLR 2015; arXiv 1409.0473.
- M.-T. Luong, H. Pham and C. D. Manning. Effective Approaches to Attention-based Neural Machine Translation. EMNLP 2015; arXiv 1508.04025.
- K. Cho, B. van Merriënboer, C. Gulcehre, D. Bahdanau, F. Bougares, H. Schwenk and Y. Bengio. Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. EMNLP 2014; arXiv 1406.1078.
- A. Graves. Generating Sequences With Recurrent Neural Networks. arXiv 1308.0850, 2013.
- I. Sutskever, O. Vinyals and Q. V. Le. Sequence to Sequence Learning with Neural Networks. NeurIPS 2014; arXiv 1409.3215.
- K. Xu, J. Ba, R. Kiros, K. Cho, A. Courville, R. Salakhutdinov, R. Zemel and Y. Bengio. Show, Attend and Tell: Neural Image Caption Generation with Visual Attention. ICML 2015; arXiv 1502.03044.
- J. Chorowski, D. Bahdanau, D. Serdyuk, K. Cho and Y. Bengio. Attention-Based Models for Speech Recognition. NeurIPS 2015; arXiv 1506.07503.
- A. Vaswani et al. Attention Is All You Need. NeurIPS 2017; arXiv 1706.03762.
