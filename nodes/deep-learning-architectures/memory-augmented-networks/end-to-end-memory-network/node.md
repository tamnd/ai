---
tag: VDRYRT
kind: architecture
slug: end-to-end-memory-network
name: End-to-end memory network
aliases: [MemN2N, end-to-end memory networks, weakly supervised memory network, multi-hop memory network]
area: deep-learning-architectures/memory-augmented-networks
level: core
summary: >-
  A network that stores each sentence of a story as a vector, reads them by a softmax attention driven by the question, and repeats the read over several hops, trained end to end from answers alone.
uses: [backpropagation@al, cross-entropy@m2, matrix-multiplication@m2]
related_to: [neural-turing-machine, multi-head-attention, transformer, long-short-term-memory, gradient-clipping]
history:
  year: 2015
  people: [Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus]
  note: "Sukhbaatar, Szlam, Weston and Fergus at New York University and Facebook AI Research posted End-To-End Memory Networks as arXiv 1503.08895 in March 2015. It made the memory networks of Weston, Chopra and Bordes (2014) differentiable, and was tested on the bAbI tasks that Weston et al. had released a month before."
prereqs: [multi-head-attention, backpropagation@al, cross-entropy@m2]
same_as:
  arxiv: ["1503.08895"]
sources:
  - {paper: "Sukhbaatar, Szlam, Weston and Fergus (2015)", title: "End-To-End Memory Networks", locator: "arXiv 1503.08895v5, Sections 2 to 6, Tables 1 and 2, Figure 2", note: "the architecture, the tying, the encodings, the training details and both sets of experiments"}
  - {paper: "Weston, Chopra and Bordes (2015)", title: "Memory Networks", locator: "arXiv 1410.3916v11, Sections 2 and 3", note: "the memory network with hard lookups and supervision of the supporting facts"}
  - {paper: "Weston, Bordes, Chopra, Rush, van Merriënboer, Joulin and Mikolov (2015)", title: "Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks", locator: "arXiv 1502.05698v10, Sections 3, 4 and 6", note: "the 20 bAbI tasks and the criterion for passing one"}
  - {paper: "Miller, Fisch, Dodge, Karimi, Bordes and Weston (2016)", title: "Key-Value Memory Networks for Directly Reading Documents", locator: "arXiv 1606.03126v2, Section 3", note: "a descendant that separates the keys used to address from the values that are read"}
  - {paper: "Graves, Wayne and Danihelka (2014)", title: "Neural Turing Machines", locator: "arXiv 1410.5401v2, Section 3", note: "the contemporary memory with content and location addressing"}
  - {paper: "Vaswani et al. (2017)", title: "Attention Is All You Need", locator: "Section 2", note: "the transformer paper's account of end-to-end memory networks"}
family: attention
inputs: "a story as a (B, N, J) array of word ids, N sentences of J words with the most recent first, and a question as a (B, J) array"
outputs: "logits over the V words of the vocabulary, a (B, V) array, one answer per question"
components: []
inductive_biases: ["a memory of one vector per sentence, read by content: the question picks sentences by their inner product with it, and the order of the sentences enters only through learned temporal rows", "several reads in sequence, each conditioned on what the earlier ones returned, which lets the model chain two or three facts", "soft attention everywhere, so the model needs only the answers and never the supporting facts"]
params: "(K + 1)(V d + N d) with adjacent tying of K + 1 embedding matrices and K + 1 temporal matrices; 4 V d + d² + 2 N d with layer-wise tying; independent of the sentence length J"
flops_per_example: "2 J d ((K + 1) N + 1) + 4 K N d + 2 V d with the position encoding and adjacent tying, counting a multiply-add as 2 FLOPs and leaving out the lookups and the softmaxes"
---

An end-to-end memory network, or MemN2N, answers a question about a story by attending over the sentences of the story.
Each sentence is embedded as a memory vector and an output vector; the question is embedded as a state $u$; a hop computes a softmax over the inner products of $u$ with the memory vectors, adds the weighted sum of the output vectors to $u$, and the next hop starts from the new state.
After $K$ hops, usually three, a linear map and a softmax over the vocabulary give the answer.
Every operation is differentiable, so the network is trained from question and answer pairs alone, with no label of which sentences matter.
Sukhbaatar, Szlam, Weston and Fergus introduced it in 2015, as a version of the memory networks of Weston et al. that needs much less supervision, and applied it to the bAbI question answering tasks and to language modeling.

## In plain words

Read a short story: John went to the bedroom, John travelled to the bathroom, Mary went to the office.
Now answer: where is John?
You look back over the sentences for the ones about John, prefer the most recent, and read off the place.
An end-to-end memory network does the same with numbers.
It turns each sentence into a list of numbers that says roughly which words it contains, and turns the question into another such list.
It scores every sentence by how well its list lines up with the question's, turns the scores into weights that add up to one, and blends the sentences by those weights.
The blend is added to the question, and the result is the network's new idea of what it is looking for.
For some questions one look is enough. For "where is the milk?" the network must first find who took the milk and then where that person went, so it looks again with the updated question, and a third time if needed.
Nothing tells the network which sentences hold the answer. It learns where to look only because looking in the right place makes its answers right.

## Overview

Sukhbaatar et al. describe the model as a recurrent attention model over a possibly large external memory which, unlike the memory networks before it, is trained end-to-end and hence requires significantly less supervision during training. {#c-memn2n-end-to-end}
The input is a set of sentences $x_1, \dots, x_N$ and a question $q$, all sequences of words from a vocabulary of $V$ words.

**One hop.** An embedding matrix $A$ maps each sentence to a memory vector $m_i$ of width $d$, and a matrix $B$ maps the question to the internal state $u$. The match between $u$ and each memory is the softmax of their inner products, {#c-match}

$$ p_i = \operatorname{softmax}\big(u^\top m_i\big) = \frac{\exp(u^\top m_i)}{\sum_{i'} \exp(u^\top m_{i'})} . $$ {#eq-match}

A second matrix $C$ maps each sentence to an output vector $c_i$, and the response of the memory is their sum weighted by the match, {#c-out}

$$ o = \sum_{i=1}^{N} p_i\, c_i . $$ {#eq-out}

In the single-layer model the answer is predicted from the sum of the response and the state, by a matrix $W$ of shape $V \times d$, {#c-answer}

$$ \hat a = \operatorname{softmax}\big(W (o + u)\big) . $$ {#eq-answer}

The matrices $A$, $B$, $C$ and $W$ are learned jointly by minimizing a standard cross-entropy loss between $\hat a$ and the true answer, with stochastic gradient descent. {#c-joint-training}
Because the function from input to output is smooth, the gradients and the backpropagation are easy to compute. {#c-smooth}

**Several hops.** The layers are stacked by adding the response to the state, {#c-stack}

$$ u^{k+1} = u^k + o^k , $$ {#eq-stack}

and each layer $k$ has its own $A^k$ and $C^k$, constrained to ease training and reduce the number of parameters; the answer uses the state after the last hop, $\hat a = \operatorname{softmax}(W u^{K+1})$. {#c-hops-matrices}
The paper explores two ways to tie them. {#c-two-tyings}
In **adjacent** tying the output embedding of one layer is the input embedding of the next, $A^{k+1} = C^k$; the answer matrix is the last output embedding, $W^\top = C^K$; and the question embedding is the first input embedding, $B = A^1$. {#c-adjacent}
In **layer-wise** tying, like an RNN, all the hops share one $A$ and one $C$, and a linear map $H$, learned with the rest, is added to the update of the state, {#c-layerwise}

$$ u^{k+1} = H u^k + o^k . $$ {#eq-layerwise}

Sukhbaatar et al. note that the whole model is similar to the Memory Network of Weston et al., except that the hard max operations within each layer have been replaced with a continuous weighting from the softmax. {#c-like-memnn}
With layer-wise tying it can be cast as a traditional RNN whose outputs are divided into internal and external ones: emitting an internal output corresponds to considering a memory, and emitting an external output to predicting a label. {#c-as-rnn}

The rest of the page describes how a sentence becomes a vector, counts the parameters and the cost, follows the training recipe, and reports what the experiments measured.

## Components

### Sentence representation

The simplest representation is the bag of words: $m_i = \sum_j A x_{ij}$, the sum of the embeddings of the words of the sentence, with $c_i$ and $u$ formed the same way from $C$ and $B$. {#c-bow}
It cannot capture the order of the words in a sentence, which is important for some tasks. {#c-bow-order}
The position encoding, or PE, weighs the $j$-th word of a sentence of $J$ words elementwise by a vector $l_j$, $m_i = \sum_j l_j \circ A x_{ij}$, with

$$ l_{kj} = \Big(1 - \frac{j}{J}\Big) - \frac{k}{d}\Big(1 - \frac{2j}{J}\Big) $$ {#eq-pe}

for the dimension $k$ of $d$, with 1-based indices; the same representation is used for questions, memory inputs and memory outputs. {#c-pe}
The derivation `position-encoding-and-order` rewrites the weight as a bilinear interpolation, $l_{kj} = (1 - j/J)(1 - k/d) + (j/J)(k/d)$: every weight lies in $[0, 1]$, the low dimensions weigh the first words of a sentence most and the high dimensions the last, and the word at $j = J/2$ weighs $1/2$ everywhere. {#c-pe-ramp}
Swapping two words changes the sentence vector in dimension $k$ in proportion to $2k/d - 1$, so every dimension but the middle one sees the order, where the bag of words sees none of it. {#c-pe-swap}

### Temporal encoding

Many of the tasks need some notion of time: Sam is in the bedroom after he is in the kitchen.
The temporal encoding adds a learned row to each memory vector, $m_i = \sum_j A x_{ij} + T_A(i)$, and the output vectors are augmented the same way with a matrix $T_C$; both are learned and subject to the same tying as $A$ and $C$. {#c-temporal}
In full, with the position encoding,

$$ m_i = \sum_{j=1}^{J} l_j \circ A x_{ij} + T_A(i), \qquad c_i = \sum_{j=1}^{J} l_j \circ C x_{ij} + T_C(i) . $$ {#eq-temporal}

The sentences are indexed in reverse order, by their distance from the question, so $x_1$ is the last sentence of the story. {#c-reverse-order}
The memory has no other notion of order: the softmax of eq-match is the same whatever the order of the slots, and only the temporal rows tie a slot to a position.
Since the softmax ignores a number added to all its inputs, adding the same vector to every row of $T_A$ changes no attention weight, so only the differences of the rows matter. {#c-temporal-common-shift}

### Empty memories and random noise

Stories have different numbers of sentences and sentences different numbers of words, so a null symbol pads them all to a fixed size, and the embedding of the null symbol is constrained to be zero. {#c-null}
An empty slot then has $m_i = T_A(i)$ and $c_i = T_C(i)$: it still takes a share of the attention and adds its temporal output row to the response, as the derivation `temporal-rows-and-empty-memories` works out. {#c-empty-memory}
Sukhbaatar et al. found it helpful to add dummy memories to regularize $T_A$, adding 10% of empty memories to the stories at random at training time; they call this random noise, or RN. {#c-rn}
An inserted empty slot moves every older sentence one row down, so the model cannot tie a fact to one fixed row.

### Linear start

Training can start with the softmax of every memory layer removed, which makes the model entirely linear except for the final softmax; when the validation loss stops decreasing, the softmaxes are put back and training continues. {#c-ls}
This is linear start, or LS, and it uses an initial learning rate of 0.005. {#c-ls-rate}
The derivation `linear-start-is-linear` makes the first claim exact: with a fixed story, each linear hop maps $u$ to $\big(I + \sum_i c_i m_i^\top\big) u$, so the state after $K$ hops is a product of $K$ matrices times the question vector. {#c-linear-start-linear}

### One forward pass

With adjacent tying there are $K + 1$ embedding matrices $E_0, \dots, E_K$ and as many temporal matrices.
The question is embedded by $E_0$; hop $k$ reads memory vectors made by $E_{k-1}$ and output vectors made by $E_k$; the answer map is $E_K^\top$.
So the story is embedded $K + 1$ times, not $2K$, and the output vectors of one hop are the memory vectors of the next.
A forward pass embeds the question, embeds the story once per matrix, runs eq-match, eq-out and eq-stack $K$ times, and applies eq-answer.

## Shapes and parameters

Write $V$ for the vocabulary, $d$ for the width of the embeddings and of the state, $K$ for the hops, $N$ for the sentences kept in memory and $J$ for the words of a padded sentence.
With adjacent tying and the temporal encoding the model has

$$ P_{\text{adj}} = (K + 1)(V d + N d) $$ {#eq-params}

parameters: {shape:params}. {#c-params-adjacent}
$J$ does not enter: the position encoding has no parameters.
The example config, 12 words, $d = 6$, 3 hops and 5 sentences, has {shape:params@example}.
At the sizes of the bAbI experiments of Sukhbaatar et al., $V = 177$, $d = 20$, $K = 3$ and the 50 most recent sentences, the config `babi`, it has {shape:params@babi}.
Without any tying, an $A$, a $C$, a $T_A$ and a $T_C$ for every hop and a separate $B$ and $W$, it would have {shape:untied_params@babi}: adjacent tying roughly halves the count. {#c-params-tying-halves}
Layer-wise tying has $4Vd + d^2 + 2Nd$, {shape:layerwise_params@babi} at the same sizes.

The language model of Section 5 uses layer-wise tying with no question embedding, $3Vd + d^2 + 2Nd$ parameters.
At the Penn Treebank sizes, 10,000 words, 150 hidden units and 100 memories, the config `ptb`, that is {shape:lm_params@ptb}, against {shape:rnn_params@ptb} for a simple recurrent network with 150 hidden units and the same vocabulary, a ratio of 1.5. {#c-params-lm-ratio}
Sukhbaatar et al. note that MemN2N has about 1.5 times more parameters than RNNs with the same number of hidden units, and the LSTM about 4 times more. {#c-lm-params-paper}
The derivation `parameter-and-flop-count` counts the arrays of the NumPy reference for both tyings and checks these numbers.

## Computation and cost

Counting a multiply-add as 2 FLOPs and only the products, a forward pass of a batch of $B$ questions with the position encoding and adjacent tying costs

$$ C_{\text{fwd}} = B\,\big[\,2 J d\,\big((K + 1) N + 1\big) + 4 K N d + 2 V d\,\big] $$ {#eq-flops}

FLOPs: the position-weighted sums of the words of the story, once per embedding matrix, and of the question; the $N$ inner products and the weighted sum of every hop; and the answer map. {#c-flops}
For the example config and a batch of 2 that is {shape:flops_forward@example} FLOPs, and at the `babi` sizes with $J = 7$ it is {shape:flops_forward@babi} for a batch of 2.
There the embedding of the story is the largest term, about three quarters of the total, because it is repeated $K + 1$ times over $N J$ words; it exceeds the answer map whenever $J(K + 1)N > V$. {#c-flops-embedding}
The bag of words only adds, so with it the embedding costs no multiplications at all, but the same number of additions.

The cost is linear in the size of the memory: every hop scores every slot.
Sukhbaatar et al. name this as a limit, writing that smooth lookups may not scale well to the case where a larger memory is required. {#c-smooth-scale}
The hops are sequential, but unlike a recurrent network over the text, the story is embedded in parallel, and all the slots of a hop are scored at once.
The memory vectors of a story do not depend on the question, so several questions about one story can share the embedded story.

## Training

For the bAbI tasks, 10% of the training set was held out to choose the architecture and the hyperparameters. {#c-validation}
The models were trained with a learning rate of 0.01, halved every 25 epochs until 100 epochs, with no momentum or weight decay; the weights were initialized from a Gaussian with mean zero and $\sigma = 0.1$. {#c-train-schedule}
The batch size was 32, the cost was not averaged over a batch, and gradients with an $\ell_2$ norm larger than 40 were scaled down to norm 40, as in [[gradient-clipping]]. {#c-train-batch-clip}
In the question answering tasks the gradient of each weight matrix is measured separately. {#c-clip-per-matrix}
The memory was restricted to the most recent 50 sentences. {#c-memory-50}
On some tasks there was a large variance in the performance depending on the initialization, so each training was repeated 10 times with different random initializations and the one with the lowest training error picked. {#c-restarts}
Joint training on all 20 tasks used $d = 50$, against $d = 20$ for a model of one task. {#c-dims}

The language models were trained the same way except for a few changes. {#c-lm-training-same}
The norm of the whole gradient of all parameters was clipped at 50, which Sukhbaatar et al. found crucial for good performance. {#c-lm-clip}
The learning rate was divided by 1.5 whenever the validation cost had not decreased after an epoch, and training stopped when it fell below $10^{-5}$, after about 50 epochs. {#c-lm-anneal}
The weights were initialized from $N(0, 0.05)$ and the batch size was 128. {#c-lm-init}

## Inductive biases

The memory holds one vector per sentence and is addressed only by content.
A sentence is found because its embedding lines up with the state, and the order of the sentences enters only through the temporal rows.
Sukhbaatar et al. contrast this with the [[neural-turing-machine]], whose memory uses both content and address-based access, unlike theirs, which only explicitly allows the former, although the temporal features allow a kind of address-based access. {#c-vs-ntm-access}
Because each item is always written to memory sequentially, their model is somewhat simpler and needs no operations like sharpening. {#c-memn2n-simpler}

The hops build in a short chain of reasoning: each read is conditioned on the earlier ones through the state.
The number of hops is fixed in advance, so the depth of the chain is a hyperparameter, not something the model decides for each question.
The softmax keeps every read soft, which is what makes training from answers alone possible, and it also means the model never commits to one fact; it can spread its attention when it is unsure.

## Evidence

The bAbI tasks are 20 synthetic question answering tasks, each a set of statements followed by a question whose answer is usually a single word. {#c-babi-tasks}
For each problem there are at most 320 sentences, the vocabulary has $V = 177$ words, and two versions of the data have 1,000 and 10,000 training problems per task. {#c-babi-sizes}
Weston et al. count a task as passed if a model reaches at least 95% accuracy, and call the choice of 95% and of 1,000 training examples arbitrary. {#c-babi-pass}
Unless otherwise stated, the experiments used $K = 3$ hops and adjacent tying. {#c-default-k3}

The baselines were the strongly supervised memory network, MemNN, which uses a max operation at each layer and is trained directly with the supporting facts; a weakly supervised heuristic version, MemNN-WSH; and an LSTM trained on question and answer pairs. {#c-baselines}
Table 1 of Sukhbaatar et al. reports these mean test errors over the 20 tasks with 1,000 training examples:

| Model | Mean error (1k) | Failed tasks (1k) | Mean error (10k) |
|---|---|---|---|
| MemNN, strongly supervised | 6.7% | 4 | 3.2% |
| LSTM | 51.3% | 20 | 36.4% |
| MemNN-WSH | 40.2% | 18 | 39.2% |
| MemN2N, BoW | 25.1% | 15 | 15.4% |
| MemN2N, PE | 20.3% | 13 | 9.4% |
| MemN2N, PE LS | 16.3% | 12 | 7.2% |
| MemN2N, PE LS RN | 13.9% | 11 | 6.6% |
| MemN2N, PE LS RN, joint | 12.4% | 11 | 7.5% |

A failed task has an error above 5%. {#c-table1-mean}
The text gives a mean error of 12.6% for the model with position encoding, linear start and random noise, jointly trained, against 6.7% for the strongly supervised memory network. {#c-best-text}

- **Weak beats weak.** All the variants of MemN2N comfortably beat the weakly supervised baselines. {#c-beats-weak}
- **Still short of strong supervision.** Sukhbaatar et al. describe their strongest variants as reasonably close to the supervised models, which are still superior. {#c-close-supervised}
- **Word order.** The position encoding improves on the bag of words with clear improvements on tasks 4, 5, 15 and 18, where word ordering is particularly important. {#c-pe-helps}
- **Linear start.** Linear start seems to help avoid local minima: on task 16 the PE model has an error of 52.1% in Table 1, and 1.6% with linear start. {#c-ls-task16}
  The text quotes the first number as 53.6%. {#c-ls-task16-text}
- **Random noise and joint training.** Jittering the time index with random empty memories gives a small but consistent boost, especially with 1k training examples, and joint training on all tasks helps. {#c-rn-joint}
- **Hops.** More hops give improved performance. {#c-hops-help} With position encoding, linear start and joint training on 1k examples, the mean error is 25.8% with one hop, 15.6% with two and 13.3% with three. {#c-hops-table}

The attention of the trained model can be read off.
On a story of the second task, with the question "where is the milk?", the first hop put a weight of 0.88 on "John took the milk there" and the third put 1.00 on "John moved to the hallway", two supporting facts the model was never told about. {#c-fig2}
Sukhbaatar et al. report that MemN2N learns to focus on the correct supporting sentences without using the labeled supporting facts during training. {#c-fig2-caption}

**Language modeling.** Here each memory slot holds one word of the previous words of the text, so no bag of words is needed; the question is fixed to a constant vector of 0.1, the model uses layer-wise tying, and ReLU is applied to half of the units in each layer. {#c-lm-setup}
In Table 2, MemN2N reached a test perplexity of 111 on Penn Treebank, against 115 for the LSTM and the SCRN and 129 for the RNN, and 147 on Text8, against 154 for the LSTM. {#c-lm-results}
The text of Section 5.2 gives the Penn Treebank comparison as 111 against 115 for RNN/SCRN, which does not match the RNN row of the table. {#c-lm-results-text}
In Table 2, on Penn Treebank with 150 hidden units and a memory of 100 words, the test perplexity falls from 121 with 2 hops to 114 with 7. {#c-lm-hops}
Averaged over the test set, the attention of the trained models puts some hops only on recent words and others broadly over all memory locations, and the two types of hops tend to alternate. {#c-lm-attention}
Unlike in a traditional RNN, the cache does not decay exponentially, and Sukhbaatar et al. suggest this may be the source of the improvement. {#c-lm-cache}

## Limitations

- **Supervision still helps.** The model is still unable to exactly match the memory networks trained with strong supervision, and both fail on several of the 1k tasks. {#c-limits-supervised}
  On the path finding task 19 every MemN2N variant of Table 1 has an error between 82.8% and 90.7%, and the strongly supervised memory network 64.0%. {#c-task19}
- **Smooth lookups.** Every hop scores every slot, so the cost and the spread of the softmax grow with the memory; Sukhbaatar et al. plan multiscale attention or hashing for larger memories. {#c-limits-hashing}
- **High variance.** Training on some tasks fails or succeeds with the initialization, which the recipe handles by ten restarts.
- **A fixed number of hops.** The depth of reasoning is set before training. In the 1k results of Table 1 every MemN2N variant stays above 20% error on task 3, three supporting facts, and above 40% on task 17, positional reasoning. {#c-task3-17}
  With 10k examples, Sukhbaatar et al. found that adding more non-linearity solves tasks 17 and 19. {#c-nonlinearity}
- **Write by appending.** The memory is written by embedding each sentence once; there is no learned write, no update of a stored item, and no way to free a slot.
- **Synthetic data.** The question answering results are on synthetic tasks with a vocabulary of 177 words. Weston et al. write that even if a method works well on the 20 tasks, it should be shown to be useful on real data as well. {#c-babi-real}

## Variants and descendants

- **Memory networks.** Weston, Chopra and Bordes proposed networks with a long-term memory and four learned components, an input feature map, a generalization step that updates the memory, an output map and a response. {#c-memnn-components} Their output step picks the best memory by a hard max, which needs the supporting facts as labels.
- **Key-value memory networks.** Miller et al. based their model on the end-to-end memory network and stored pairs of vectors, a key that is compared with the question and a value that is read, with a different matrix $R_j$ updating the query at each hop. {#c-kv} A first step, key hashing, uses an inverted index to preselect the memories whose key shares a word with the question. {#c-kv-hashing}
- **Dynamic memory networks and other bAbI models.** Weston et al. list the weakly supervised MemN2N, the dynamic memory networks of Kumar et al. and the Neural Reasoner as algorithms that the bAbI tasks influenced. {#c-babi-impact}
- **Attention-only models.** The [[transformer]] paper describes end-to-end memory networks as based on a recurrent attention mechanism instead of sequence-aligned recurrence, performing well on simple-language question answering and language modeling. {#c-vaswani} A hop is a single-head attention whose query is the state, whose keys are the memory vectors and whose values are the output vectors; [[multi-head-attention]] adds a scaling by $1/\sqrt{d}$, several heads, and learned projections in place of separate embedding matrices.

## Notable models

The models of the paper are small.
A model of one bAbI task has {shape:params@babi} parameters with adjacent tying and the temporal encoding. The Penn Treebank language model with 150 hidden units and a memory of 100 words has {shape:lm_params@ptb} by the count of the derivation `parameter-and-flop-count`, nearly all of them in its three embedding matrices of $10{,}000 \times 150$.
Sukhbaatar et al. released their source code. {#c-code}

## Implementations

The NumPy reference `impl/numpy/memn2n.py` implements the position encoding of eq-pe, the bag of words, the temporal rows of eq-temporal, one hop as eq-match and eq-out with an option for the linear start, the stacking of eq-stack and of eq-layerwise, the answer of eq-answer, both tyings, the summed cross-entropy, and the counts eq-params and eq-flops.
It caches the embedded story by matrix, so adjacent tying embeds it $K + 1$ times.
The PyTorch version `impl/pytorch/memn2n.py` follows it on a dict of tensors and adds the clipping of each matrix's gradient at 40; its tests check the gradients of both tyings with `torch.autograd.gradcheck` and train a one-hop model on a toy task.
The JAX version `impl/jax/memn2n.py` is a pure function of the parameters, tested under `jax.jit` and `jax.vmap` and against finite differences.
All three are tested against the shared fixture of `fixtures/`: the example config, a batch of 2 toy stories, one with an empty slot, the logits, the attention of every hop and the loss.

### Worked example

Take one hop, the bag of words and adjacent tying, with $d = 9$: dimensions 0 to 2 stand for three people, 3 to 7 for five places, and 8 for recency.
$E_0$, which is both $A^1$ and $B$, maps each person to 10 in that person's dimension and the word "where" to 1 in dimension 8. $T_0$, the $T_A$ of the hop, gives slot $i$ the value $-5i$ in dimension 8. $E_1$, both $C^1$ and $W^\top$, maps each place to 1 in its own dimension. Every other entry is zero.
The story, most recent first, is "John went kitchen", "Mary went garden", "John went bedroom", and two sentences about a third person; the question is "where is John".

The state is $u = 10\, e_{\text{John}} + e_8$.
The first slot scores $10 \cdot 10 + 0 = 100$, the third $100 - 10 = 90$, Mary's slot $-5$, and the last two $-15$ and $-20$.
The softmax of eq-match gives the first slot a weight of $1/(1 + e^{-10} + \dots) \approx 0.99995$.
The response of eq-out is almost exactly $e_{\text{kitchen}}$, and eq-answer gives the logit 0.99995 to "kitchen", $4.5 \times 10^{-5}$ to "bedroom" and 0 to the other words: the network answers kitchen.
The recency term breaks the tie between the two sentences about John; without $T_A$ both would score 100 and the response would be half kitchen, half bedroom.
The test `test_a_hand_wired_hop_answers_the_toy_task` runs this weighting on 200 random stories, and `test_without_the_temporal_rows_the_hop_cannot_tell_old_from_new` the tie.
The model has $2 \cdot (12 \cdot 9 + 5 \cdot 9) = 306$ parameters, as eq-params says for $K = 1$.

### Pitfalls

> [!pitfall]
> **The null word must stay zero.** If the embedding of the padding symbol is an ordinary row, it is learned like any other, and a sentence of two words padded to six gets four copies of it added. The constraint of Section 4.2 holds the row at zero; in a framework with autograd the row still receives a gradient, so it must be reset after every update. `test_null_word_adds_nothing` and `test_null_rows_get_a_gradient_unless_reset` check both.

> [!pitfall]
> **The order of the slots.** The temporal rows are indexed from the most recent sentence, $x_1$ the last sentence of the story. Indexed from the first sentence, the row of the latest fact would depend on the length of the story, and a model trained on short stories would read long ones with the wrong rows. `test_reversed_order_keeps_the_latest_sentence_at_row_zero` checks that older sentences do not move the recent ones.

> [!pitfall]
> **Which $J$.** Section 4.1 defines $J$ in eq-pe as the number of words in the sentence. Sentences are padded to a fixed width, and with the padded width the weights of a short sentence's words differ from those with its own length. The references use the padded width for all sentences, so the encoding is the same for every sentence position; an implementation that uses each sentence's own length computes a different function from the same weights.

> [!pitfall]
> **Two tyings, two parameter sets.** Under adjacent tying $C^k$ and $A^{k+1}$ are one matrix, and so are $W^\top$ and $C^K$. Creating $2K + 2$ separate matrices trains a different, untied model with nearly twice the parameters. `test_adjacent_tying_shares_the_matrices_between_hops` checks that changing $E_1$ moves hop 2's attention and not hop 1's.

> [!pitfall]
> **Empty slots take attention.** A padded slot has the memory vector $T_A(i)$ and is not masked out of the softmax, so it takes part of the weight and contributes $T_C(i)$. This is the model of the paper, and random noise relies on it, but masking the padded slots, as many attention codes do by default, gives a different model.

## Experiments

This page has no experiment yet. The checks of its four derivations play the part of small ones: they count the parameters and the multiply-adds of the reference against eq-params and eq-flops for both tyings, rebuild the linear start as a product of matrices, test the bilinear form of the position encoding and the effect of a swap on a grid of sizes, and shift the temporal rows. The PyTorch tests train a one-hop model with Adam on a toy task in the form of the first bAbI task and check that it answers more than 95% of held-out questions and puts most of its attention on the supporting sentence.

## History

Weston, Chopra and Bordes posted Memory Networks in October 2014, the same month as the neural Turing machine of Graves et al.
Their model stores each input in a slot and finds the relevant slots with hard lookups, trained with labels of which slots support each answer.
In February 2015 Weston et al. released the bAbI tasks, 20 kinds of synthetic questions meant as prerequisites for any system that aims to be capable of conversing with a human. {#c-babi-aim}
A month later Sukhbaatar, Szlam, Weston and Fergus replaced the hard lookups with a softmax, stacked several hops, and trained the result from answers alone.
They related the model to the attention of Bahdanau et al. for translation, and noted that their memory spans many sentences rather than one and is read in several hops. {#c-vs-bahdanau}
Key-value memory networks applied the idea to reading Wikipedia in 2016, and in 2017 the [[transformer]] made attention over a set of vectors the whole of the architecture, with no recurrence and no separate memory.
The [[long-short-term-memory]] networks that MemN2N was compared with keep their memory in the state of the network, which Sukhbaatar et al. describe as latent and inherently unstable over long timescales. {#c-vs-lstm}

## References

- S. Sukhbaatar, A. Szlam, J. Weston and R. Fergus. End-To-End Memory Networks. arXiv 1503.08895, 2015.
- J. Weston, S. Chopra and A. Bordes. Memory Networks. ICLR 2015.
- J. Weston, A. Bordes, S. Chopra, A. M. Rush, B. van Merriënboer, A. Joulin and T. Mikolov. Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks. arXiv 1502.05698, 2015.
- A. H. Miller, A. Fisch, J. Dodge, A.-H. Karimi, A. Bordes and J. Weston. Key-Value Memory Networks for Directly Reading Documents. arXiv 1606.03126, 2016.
- A. Graves, G. Wayne and I. Danihelka. Neural Turing Machines. arXiv 1410.5401, 2014.
- A. Vaswani et al. Attention Is All You Need. 2017.
