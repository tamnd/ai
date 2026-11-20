---
tag: NEW-neural-probabilistic-language-model
kind: method
slug: neural-probabilistic-language-model
name: Neural probabilistic language model
aliases: [NPLM, neural network language model, NNLM, feedforward neural language model, Bengio language model, neural n-gram model]
area: language-models/neural-language-models
level: core
summary: >-
  A language model that maps each of the previous n − 1 words to a learned vector and predicts the next word with a tanh layer and a softmax, learning the vectors and the network together.
purpose: predicting the next word from a fixed window of previous words so that similar words share statistical strength through learned word vectors, which lets the model generalise to word sequences never seen in training
uses: [cross-entropy@m2, stochastic-gradient-descent@m2, maximum-likelihood-estimation@m2, conditional-probability@m2, backpropagation@al, softmax-regression@al]
related_to: [n-gram-language-model, multilayer-perceptron, recurrent-neural-network, long-short-term-memory, transformer, weight-decay, byte-pair-encoding, curse-of-dimensionality@al, huffman-coding@m2, mixture-model@m2]
hyperparameters:
  - {name: context n − 1, meaning: "the number of previous words the network sees; the model is an n-gram model in the sense of what it conditions on", typical: "2 to 4 previous words in Bengio et al. (2003) (n = 3 to 5, n = 6 for AP News); 2 in Mnih and Teh (2012); 10 in Mikolov et al. (2013)"}
  - {name: feature size m, meaning: "the length of each word feature vector, the row of C", typical: "30, 60 or 100 against a vocabulary of about 17,000 (Bengio et al., 2003); 100 (Mnih and Teh, 2012)"}
  - {name: hidden units h, meaning: "the width of the tanh hidden layer; 0 removes it and leaves only the direct connections", typical: "0, 50 or 100 on Brown and 60 on AP News (Bengio et al., 2003); 500 to 1000 (Mikolov et al., 2013)"}
  - {name: direct connections, meaning: "whether the word features also feed the output through the matrix W", typical: "with or without; without took twice as many epochs on Brown and reached a slightly lower perplexity"}
  - {name: learning rate, meaning: "the step size ε_t = ε_0 / (1 + r t) of stochastic gradient ascent after t updates", typical: "ε_0 = 10^−3 and r = 10^−8 (Bengio et al., 2003)"}
  - {name: weight decay, meaning: "the penalty on the weights and on C, not on the biases", typical: "10^−4 on Brown and 10^−5 on AP News"}
  - {name: mixture weight, meaning: "the weight of the network in a mixture with an n-gram model", typical: "0.5, or weights learned per context-frequency bucket"}
  - {name: output layer, meaning: "the full softmax, a tree of binary decisions, or a sampled objective such as noise-contrastive estimation", typical: "full softmax up to about 20,000 words; a tree or 25 noise samples beyond"}
cost:
  train: "per example, about |V|(1 + (n − 1)m + h) multiply-adds at the output and h(1 + (n − 1)m) at the hidden layer, with the same again for the gradient; the output share is 99.7% for the AP News network, so time grows linearly with the vocabulary unless the softmax is replaced"
  inference: "one forward pass per predicted word, dominated by the |V| × h output product; log2 |V| sigmoids per word with a tree softmax, but a full softmax for a whole distribution"
history:
  year: 2003
  people: [Yoshua Bengio, Réjean Ducharme, Pascal Vincent, Christian Jauvin, Frédéric Morin, Andriy Mnih, Yee Whye Teh, Tomas Mikolov]
  note: "Bengio, Ducharme and Vincent presented the model at NIPS 2000 and in JMLR in 2003; Bengio and Senécal (2003) trained it by importance sampling; Morin and Bengio (2005) replaced the softmax by a tree; Mikolov et al. (2010) replaced the window with a recurrent network; Mnih and Teh (2012) trained it with noise-contrastive estimation; Mikolov et al. (2013) dropped the hidden layer to learn word vectors."
prereqs: [multilayer-perceptron, n-gram-language-model, cross-entropy@m2, backpropagation@al]
same_as:
  arxiv: ["1206.6426", "1301.3781", "1608.05859"]
sources:
  - {paper: "Bengio, Ducharme, Vincent and Jauvin (2003)", title: "A Neural Probabilistic Language Model", locator: "JMLR 3, pp. 1137 to 1155; Sections 1 to 5, Tables 1 and 2", note: "the model, its training, the parallel implementation and the Brown and AP News results"}
  - {paper: "Morin and Bengio (2005)", title: "Hierarchical Probabilistic Neural Network Language Model", locator: "AISTATS 2005, Sections 3 to 6 and Tables 1 to 3", note: "the tree softmax built from WordNet and its speed-up"}
  - {paper: "Mnih and Teh (2012)", title: "A Fast and Simple Algorithm for Training Neural Probabilistic Language Models", locator: "ICML 2012; arXiv 1206.6426v1, Sections 3 and 4", note: "noise-contrastive estimation for the output layer"}
  - {paper: "Mikolov, Karafiát, Burget, Černocký and Khudanpur (2010)", title: "Recurrent Neural Network Based Language Model", locator: "INTERSPEECH 2010, Sections 1 to 3", note: "the fixed window as the weakness and the recurrent replacement"}
  - {paper: "Mikolov, Chen, Corrado and Dean (2013)", title: "Efficient Estimation of Word Representations in Vector Space", locator: "arXiv 1301.3781v3, Section 2.1", note: "the cost of the feedforward model and the Huffman-tree softmax"}
  - {paper: "Press and Wolf (2017)", title: "Using the Output Embedding to Improve Language Models", locator: "EACL 2017; arXiv 1608.05859v3, Sections 3 and 4.2, Table 5", note: "tying the input and output word vectors"}
---

A neural probabilistic language model maps each of the previous n − 1 words to a learned feature vector, passes the concatenated vectors through a tanh hidden layer, and gives the probability of every next word with a softmax.
It conditions on the same window as an [[n-gram-language-model|n-gram model]], but similar words get similar vectors, so what is learned about one word sequence carries over to sequences never seen in training.
Bengio, Ducharme and Vincent presented it at NIPS 2000 and in JMLR in 2003.
Later work made the output layer cheaper, with importance sampling, a tree of binary decisions and noise-contrastive estimation, replaced the window by a [[recurrent-neural-network|recurrent network]], and dropped the hidden layer to learn word vectors alone.

## In plain words

An [[n-gram-language-model|n-gram model]] treats every word as an unrelated symbol, so seeing "the cat sat" says nothing about "the dog sat".
A neural probabilistic language model gives every word a short vector of real numbers and predicts the next word from the vectors of the previous few words with a small neural network.
The vectors are learned along with the network, and words used in similar contexts end up with similar vectors.
Because the network is a smooth function of the vectors, evidence about one sentence raises the probability of the many sentences that differ from it only by similar words.
The idea, published by Bengio et al. in 2003, is the root of word embeddings and of every neural language model since.

## Setting

A language model gives the probability of a text as the product of the probability of each word given the words before it, and is judged by the perplexity of held-out text, as on the [[n-gram-language-model|n-gram]] page.
Bengio et al. note that learning the joint probability function of sequences of words is intrinsically difficult because of the curse of dimensionality: a word sequence on which the model will be tested is likely to be different from all the word sequences seen during training. {#c-bengio-curse}
Modelling the joint distribution of 10 consecutive words in a natural language with a vocabulary of size 100,000 has potentially $10^{50} - 1$ free parameters. {#c-bengio-free-parameters}
This is the [[al:curse-of-dimensionality|curse of dimensionality]] in its discrete form: a table over word sequences has no notion of nearby entries.

Bengio et al. summarise the approach in three steps: associate with each word in the vocabulary a distributed word feature vector in $\mathbb{R}^m$, express the joint probability function of word sequences in terms of the feature vectors of these words, and learn simultaneously the word feature vectors and the parameters of that probability function. {#c-bengio-idea}
"Similar" words are expected to have a similar feature vector, and because the probability function is a smooth function of these feature values, the presence of one sentence in the training data increases the probability of its combinatorial number of neighbours in sentence space. {#c-bengio-generalise}
The model keeps the n-gram assumption that only the last $n - 1$ words matter; what changes is how the conditional probability is computed from them.

## The method

**The network.** Write $V$ for the vocabulary, $C$ for the $\lvert V \rvert \times m$ matrix of word features, $h$ for the number of hidden units and $w_{t-n+1} \dots w_{t-1}$ for the context.
The input is the concatenation of the context feature vectors, newest first, $x = (C(w_{t-1}), C(w_{t-2}), \dots, C(w_{t-n+1}))$, a vector of length $(n-1)m$.
The network computes the unnormalised log-probabilities $y = b + Wx + U \tanh(d + Hx)$, with the hyperbolic tangent applied element by element and $W$ optionally zero (no direct connections), and a softmax output layer guarantees positive probabilities summing to 1. {#c-bengio-model}

$$y = b + W x + U \tanh(d + H x), \qquad \hat P(w_t = i \mid w_{t-1}, \dots, w_{t-n+1}) = \frac{e^{y_i}}{\sum_{j \in V} e^{y_j}}.$$ {#eq-model}

$H$ is $h \times (n-1)m$, $U$ is $\lvert V \rvert \times h$, $W$ is $\lvert V \rvert \times (n-1)m$, and $b$, $d$ are the output and hidden biases.
The same matrix $C$ maps every context position, so a word has one vector wherever it appears.
Bengio et al. note that there are really two hidden layers: the shared word features layer $C$, which has no non-linearity, and the ordinary hyperbolic tangent hidden layer. {#c-bengio-two-layers}
Without $C$ and with one-hot inputs the network would be a [[multilayer-perceptron|multilayer perceptron]] over $(n-1)\lvert V \rvert$ inputs; $C$ is a learned linear compression of each one-hot word to $m$ numbers.

**Training.** Training looks for the parameters that maximise the training corpus penalised log-likelihood $L = \frac{1}{T}\sum_t \log f(w_t, w_{t-1}, \dots, w_{t-n+1}; \theta) + R(\theta)$, where in the experiments $R$ is a weight decay penalty applied only to the weights of the network and to $C$, not to the biases. {#c-bengio-objective}

$$L(\theta) = \frac{1}{T} \sum_{t} \log \hat P(w_t \mid w_{t-1}, \dots, w_{t-n+1}) - \lambda \big(\lVert C \rVert^2 + \lVert H \rVert^2 + \lVert U \rVert^2 + \lVert W \rVert^2\big).$$ {#eq-objective}

The negative of the first term is the [[m2:cross-entropy|cross-entropy]] of the training text, so maximising $L$ is [[m2:maximum-likelihood-estimation|maximum likelihood]] with [[weight-decay|weight decay]].
The gradient comes from [[al:backpropagation|backpropagation]] through the softmax, the tanh layer and the lookup into $C$; only the rows of $C$ for the $n - 1$ context words receive a gradient from one example.
Bengio et al. train by stochastic gradient ascent with the learning rate $\varepsilon_t = \varepsilon_0 / (1 + rt)$ after $t$ updates, with $\varepsilon_0 = 10^{-3}$ and $r = 10^{-8}$. {#c-bengio-schedule}

**The output gradient.** For the logit $y_j$, the gradient of $-\log \hat P(w_t)$ is $\hat P(j) - 1\{j = w_t\}$.
Bengio et al. set $\partial L / \partial y_j \leftarrow 1_{j = w_t} - p_j$, update $b_j \leftarrow b_j + \varepsilon\, \partial L / \partial y_j$, and back-propagate through the hidden layer with $\partial L / \partial o_k \leftarrow (1 - a_k^2)\, \partial L / \partial a_k$. {#c-bengio-output-gradient}

$$\frac{\partial(-\log \hat P(w_t))}{\partial y_j} = \hat P(j) - 1\{j = w_t\}, \qquad \frac{\partial(-\log \hat P(w_t))}{\partial o_k} = (1 - a_k^2) \sum_j U_{jk} \frac{\partial(-\log \hat P(w_t))}{\partial y_j},$$ {#eq-gradient}

with $o = d + Hx$ and $a = \tanh(o)$.
The gradient of the negative log-likelihood with respect to a logit is the model probability minus the indicator of the target, so it sums to zero over the vocabulary and is non-zero for every word, and the factor at a tanh unit is $1 - a^2$. {#c-softmax-gradient}
Every row of $U$ and $b$ moves on every example, which is why the output layer dominates the cost.

**Mixing with an n-gram model.** Bengio et al. also report the network mixed with an interpolated trigram, with the output probabilities averaged.
The interpolated n-gram was used to form a mixture with the networks since they appear to make "errors" in very different ways. {#c-bengio-mixture-reason}
A fixed-weight [[m2:mixture-model|mixture]] cannot be much worse than its parts, as the bound in the derivations makes precise.

## Derivation

Five derivations are written out with checks in `derivations/`.

`parameter-count.md` counts the parameters and the cost of one example.
The network of Eq. 1 has $\lvert V\rvert(1 + nm + h) + h(1 + (n-1)m)$ free parameters with direct connections and $\lvert V\rvert(1 + m + h) + h(1 + (n-1)m)$ without them; the AP News network with $\lvert V\rvert = 17{,}964$, $h = 60$, $n = 6$ and $m = 100$ has 11,904,264 parameters, 95.2 MB as 8-byte floats. {#c-param-count}
Bengio et al. give the same number of free parameters and call $\lvert V \rvert(nm + h)$ the dominating factor. {#c-bengio-param-formula}
In that network the weighted sums of the output units take 99.7% of the multiply-adds of one example, and the output terms $\lvert V\rvert(nm + h)$ are 99.6% of the parameters. {#c-output-share}
The number of parameters grows linearly in $\lvert V \rvert$ and in $n$, against $\lvert V \rvert^{n-1}(\lvert V \rvert - 1)$ for an n-gram table.

`softmax-gradient.md` derives Eq. 4 and its consequence that an exact gradient touches every output row.

`tree-softmax.md` concerns the hierarchical softmax.
With each word a leaf of a binary tree and the probability of the word the product of sigmoid branch probabilities along its path, the probabilities sum to one over the vocabulary for every hidden vector with no sum over the vocabulary, and one word costs one sigmoid per node on its path. {#c-tree-normalised}
A balanced tree has leaf depths $\lfloor \log_2 \lvert V \rvert \rfloor$ or $\lceil \log_2 \lvert V \rvert \rceil$, and a [[m2:huffman-coding|Huffman]] tree built from the word frequencies $q$ has mean depth between the unigram entropy $H(q)$ in bits and $H(q) + 1$; for $\lvert V \rvert = 10{,}000$ a balanced tree needs 13 or 14 sigmoids per word against 10,000 softmax terms. {#c-huffman-depth}

`nce-gradient.md` concerns noise-contrastive estimation.
With $k$ noise samples from $P_n$, the noise-contrastive gradient with respect to the score of word $w$ is $k P_n(w) / (P(w) + k P_n(w))$ times $P_d(w) - P(w)$; the factor tends to 1 as $k$ grows, which leaves the maximum-likelihood gradient for a normalised model, the gap falls like $1/k$, and $P = P_d$ is a stationary point for every $k$. {#c-nce-limit}

`mixture-bound.md` bounds the perplexity of a mixture.
By concavity of the logarithm, a mixture with fixed weight $w$ has perplexity at most $\mathrm{PP}_p^{\,w}\,\mathrm{PP}_q^{\,1-w}$, so a half-half mixture is no worse than the geometric mean of the two perplexities; for Bengio et al.'s MLP9 at 276 and the trigram at 336 that bound is 304.5, and the reported mixture MLP10 reaches 252. {#c-mixture-bound}

## Worked example

Take $V = \{a, b, c\}$, $m = 2$, one hidden unit, a one-word context ($n = 2$) and no direct connections.
Let $C(a) = (1, 0)$, $C(b) = (0, 1)$, $C(c) = (1, 1)$, $H = (\ln 2, 0)$, $d = 0$, $U = (0, \tfrac53 \ln 4, \tfrac53 \ln 3)^\top$ and $b = 0$.
The model has $3(1 + 2 + 1) + 1 \cdot (1 + 2) = 15$ parameters.

After the word $a$, $x = (1, 0)$, $o = \ln 2$ and $a_1 = \tanh(\ln 2) = (4 - 1)/(4 + 1) = 0.6$.
The logits are $0.6\,U = (0, \ln 4, \ln 3)$, so the probabilities are $(1, 4, 3)/8$:

| Next word | Logit | Probability | Gradient of $-\log \hat P(b)$ |
|---|---|---|---|
| a | 0 | 1/8 | 1/8 |
| b | ln 4 | 1/2 | −1/2 |
| c | ln 3 | 3/8 | 3/8 |

If the next word is $b$, the loss is $\ln 2 \approx 0.693$ nats.
The gradient at the hidden unit is $\sum_j U_j g_j = -\tfrac12 \cdot \tfrac53 \ln 4 + \tfrac38 \cdot \tfrac53 \ln 3 \approx -0.469$, and at its input $1 - 0.6^2 = 0.64$ times that, $\approx -0.300$.
A gradient step therefore raises $H_{11}$, the weight from the first feature, and the first feature of $C(a)$, which pushes $a_1$ up and with it the logits of $b$ and $c$, $b$ by more.
The $c$ row of $U$ moves down although $c$ was not the target, because every word not observed gets the positive gradient $\hat P(j)$.
The tests check these numbers and the gradients of all parameters against finite differences.

The tests also train the model on a synthetic language in which each word belongs to one of 6 classes of 10 words and the next class depends on the previous two classes.
On 20,000 training and 4,000 test tokens, after 6 epochs with $m = 8$, $h = 16$ and $n = 3$:

| Model | Test perplexity |
|---|---|
| The source that generated the text | 27.1 |
| Neural model | 32.25 |
| Half-half mixture of the two models | 34.72 |
| Interpolated absolute-discount trigram | 50.75 |

The trigram has to see each word pair to use it, and most of the 3,600 word pairs are rare in 20,000 tokens; the network only needs the classes, and learns them in $C$.
The test requires the nearest feature vector of at least 90% of the words to belong to a word of the same class, with no class labels given.
Here the mixture is worse than the network alone, because the trigram is much worse; it stays under the bound $\sqrt{32.25 \cdot 50.75} = 40.5$ of the mixture derivation.

## Hyperparameters

The context length is the n of the n-gram model, but the cost grows only linearly with it.
On Brown, going from 2 words of context to 4 words brought improvements to the neural network, not to the n-grams. {#c-bengio-more-context}
The feature size $m$ is far smaller than the vocabulary; Bengio et al. use $m$ = 30, 60 or 100 against a vocabulary of about 17,000. {#c-bengio-m}
Bengio et al. find the hidden units useful, comparing MLP3 with MLP1 and MLP4 with MLP2 on Brown. {#c-bengio-hidden}
The direct connections from $x$ to $y$ are optional.
Without direct connections the Brown network took twice as much time to converge (20 epochs instead of 10), albeit to a slightly lower perplexity. {#c-bengio-direct}
A weight decay penalty of $10^{-4}$ was used in the Brown experiments and $10^{-5}$ in the AP News experiments, selected by a few trials on validation perplexity. {#c-bengio-decay}

## Cost

The output layer dominates.
Bengio et al. identify the computation of the activations of the output layer as the computational bottleneck of the neural implementation. {#c-bengio-bottleneck}
Mikolov et al. write the cost of one example of the feedforward model as $Q = N \times D + N \times D \times H + H \times V$, with $N$ context words, $D$-dimensional features, $H$ hidden units and $V$ words, and call $H \times V$ the dominating term; with binary tree representations of the vocabulary the number of output units to evaluate falls to around $\log_2 V$, and most of the complexity is then caused by the term $N \times D \times H$. {#c-mikolov13-cost}
With a Huffman tree only about $\log_2(\text{unigram perplexity}(V))$ outputs need evaluation, which for a vocabulary of one million words gives about two times speedup in evaluation over a balanced tree. {#c-mikolov13-huffman}

For the AP News runs Bengio et al. parallelised over the output units on a cluster.
Training on AP News ran only 5 epochs, over 3 weeks using 40 CPUs, without signs of overfitting on the validation set. {#c-bengio-ap-time}
The communication overhead of the parameter-parallel implementation was only 1/15th of the total time for one training epoch. {#c-bengio-overhead}
Mikolov et al. report about 6 hours to train their basic recurrent model on the Brown corpus (800K words, 100 hidden units), while Bengio reports 113 days for a basic implementation and 26 hours with importance sampling. {#c-mikolov10-time}

Three ways around the softmax were tried soon after.
Importance sampling for the output layer gave a 100-fold speed-up (Bengio and Senécal, 2003). {#c-bengio-importance}
A tree of binary decisions gives $O(\log \lvert V \rvert)$ cost per word.
Morin and Bengio report a speed-up of about 200 during both training and recognition from a hierarchical decomposition of the conditional probabilities, with a binary hierarchical clustering constrained by WordNet. {#c-morin-speedup}
Noise-contrastive estimation replaces the normalised softmax by a classification of the data word against $k$ noise words.
Mnih and Teh compute that for a model with a 2-word context, 100-dimensional feature vectors and a 10K vocabulary, an NCE update using 25 noise samples should be about 45 times faster than a maximum-likelihood update. {#c-mnih-predicted-speed}

## Evidence

**Bengio et al. (2003).** Brown has 1,181,041 words, split into 800,000 training, 200,000 validation and 181,041 test words; its 47,578 distinct words were reduced to $\lvert V \rvert$ = 16,383 by merging rare words. {#c-bengio-brown-data}
The AP News training set has 13,994,528 words, the validation and test sets 963,138 and 963,071, and the 148,721 distinct words were reduced to $\lvert V \rvert$ = 17,964. {#c-bengio-ap-data}

| Model on Brown | Test perplexity |
|---|---|
| MLP9 (no direct connections, no mixture) | 276 |
| MLP10 (no direct connections, mixed with the trigram) | 252 |
| Deleted-interpolation trigram | 336 |
| Kneser-Ney back-off 3-, 4- and 5-gram | 323, 321, 321 |
| Class-based back-off, 500 classes | 312 |

On Brown, the network with the lowest validation perplexity, mixed with the trigram, has test perplexity 252, against 336 for the deleted interpolation trigram, 323 for the modified Kneser-Ney back-off trigram and 312 for a class-based back-off model with 500 classes. {#c-bengio-table1}
The deleted interpolation trigram has a test perplexity 33% above that of the neural network with the lowest validation perplexity, and the difference is 24% for the class-based n-gram with 500 word classes. {#c-bengio-33}
On the AP News corpus, MLP10, with n = 6, h = 60, m = 100, direct connections and the mixture, scores 104 valid. and 109 test, against 126 and 132 for Del. Int. and 112 and 117 for back-off KN with n = 5. {#c-bengio-table2}
These are validation and test perplexities, in the column labels of Table 2.
Bengio et al. report a test perplexity difference of about 24% on Brown and about 8% on AP News between the network and the n-gram chosen on the validation set. {#c-bengio-result}
Mixing the output probabilities of the neural network with the interpolated trigram always helped to reduce perplexity. {#c-bengio-mixing-helps}

**Faster output layers.** On a Brown corpus with $\lvert V \rvert$ = 10,000, Morin and Bengio measure 462.6 ms per training example for the original network, 6.73 ms with importance sampling and 1.79 ms with the hierarchical model, a 258-fold speed-up. {#c-morin-train-time}
Test perplexity is 268.7 for an interpolated trigram, 249.1 for a class-based model, 195.3 for the original neural network, 192.6 for importance sampling and 220.7 for the hierarchical model. {#c-morin-perplexity}
Morin and Bengio conclude that a speed-up of around 200-fold can be achieved with only a little degradation in generalisation performance. {#c-morin-conclusion}
Mnih and Teh report that noise-contrastive estimation reduces training times by more than an order of magnitude without affecting the quality of the resulting models. {#c-mnih-abstract}
On the Penn Treebank with 930K training words and a 10K vocabulary, more noise samples gave better models, 25 samples were sufficient to match the maximum-likelihood model, and NCE training was 14 times faster than maximum-likelihood training. {#c-mnih-ptb}
Fixing the normalising constants to 1 instead of learning them did not affect the performance of the resulting models. {#c-mnih-fixed-z}
The unigram noise distribution led to much better test perplexity than the uniform one for 1, 5, 25 and 100 noise samples, with the gap shrinking from almost 100 for one sample to under 15 for 100. {#c-mnih-noise}

**Tied embeddings.** Press and Wolf find that the output matrix $U$ of a neural language model is a valid word embedding, recommend tying it to the input embedding, and report a significant reduction in perplexity on a variety of neural network language models. {#c-press-abstract}
On the Penn Treebank, tying shrinks the large LSTM model of Zaremba et al. from 66M to 51M parameters and lowers its test perplexity from 78.4 to 74.3, while the training perplexity rises from 37.8 to 48.5. {#c-press-table5}

## Limitations and failure modes

The context is a fixed window.
Mikolov et al. call it a major deficiency that a feedforward network has to use a fixed-length context specified ad hoc before training, which usually means that it sees only five to ten preceding words. {#c-mikolov10-window}
Long-range agreement, topic and discourse are outside the window, as for an n-gram model; the [[recurrent-neural-network|recurrent]] and [[transformer]] language models remove this limit.

The softmax over the vocabulary makes training slow and makes every output row move on every example.
The full distribution is needed to report a perplexity, so the sampled objectives speed up training but not evaluation, and a model trained with NCE and $Z = 1$ is only approximately normalised.
A tree softmax depends on the tree: Morin and Bengio's WordNet tree cost about 13% in perplexity against the flat network (220.7 against 195.3).

Bengio et al. note that in theory, with weight decay on $W$ and $H$ but not on $C$, $W$ and $H$ could converge towards zero while $C$ would blow up, though they did not observe this with stochastic gradient ascent. {#c-bengio-blowup}
The model decays $C$ for this reason.
Rare words get few updates to their row of $C$, so their vectors stay close to their initialisation; words never seen in training have no vector at all.

## Variants

**Energy-based model.** In the network above, the distributed word features are used only for the input words; Bengio et al. describe a variant in which the output word is also represented by its feature vector and the network outputs an energy that can be read as an unnormalised log-probability of the whole sub-sequence. {#c-bengio-energy}
Tying input and output vectors is the modern form of this idea.

**Tied embeddings.** Press and Wolf note that in the untied model the only row of the input embedding updated at a step is that of the current input word, while every row of the output embedding is updated; with tying the two updates are summed into one matrix. {#c-press-updates}
Many [[transformer]] language models tie the two matrices, and the tokens are often [[byte-pair-encoding|subwords]] rather than words.

**Hierarchical softmax** replaces the output layer by a binary tree, built from WordNet by Morin and Bengio and from word frequencies by Mikolov et al.
**Noise-contrastive estimation** and **importance sampling** replace it by an objective over a few sampled words.
**Log-bilinear models** of Mnih and Hinton drop the tanh layer and predict the target vector as a linear function of the context vectors.
**Recurrent language models.** Mikolov et al. replace the window with a [[recurrent-neural-network|recurrent network]] whose hidden state summarises the whole history.
They obtain around 50% reduction of perplexity by using a mixture of several recurrent language models, compared to a backoff language model. {#c-mikolov10-result}
[[long-short-term-memory|LSTM]] networks later made such models easier to train.
**Word2vec.** Mikolov et al. (2013) remove the hidden layer and train only for the word vectors, which turns the feature matrix $C$ into a product in its own right.

## When to use it

Use the feedforward model to teach or to test the ideas of neural language modelling: learned word vectors, a softmax over the vocabulary, and the cost of the output layer.
Use it as a fast scorer over a short window where a recurrent or [[transformer]] model is too slow, as in early speech and translation systems, and mix it with an n-gram model.
For any task that needs context beyond a few words, use a [[long-short-term-memory|recurrent]] or [[transformer]] language model; they keep the embedding matrix and softmax of this model and replace the window.

## Pitfalls

> [!pitfall]
> Subtract the maximum logit before exponentiating in the softmax; Bengio et al. do this, and without it large logits overflow.

> [!pitfall]
> Order the context consistently: Eq. 1 puts the newest word first, and an implementation that concatenates oldest first trains fine but cannot load weights or fixtures from one that does not.

> [!pitfall]
> Compare perplexities only on the same vocabulary and test text; Bengio et al. count end-of-sentence and punctuation tokens like words, and Morin and Bengio use a different Brown split from Bengio et al.

> [!pitfall]
> Do not apply weight decay to the biases, and do not drop it from $C$; the parameter-count fixture and the tests follow the paper on both.

> [!pitfall]
> With noise-contrastive estimation, draw noise from the unigram distribution and evaluate with the explicitly normalised softmax; Mnih and Teh found uniform noise much worse.

## History

Bengio, Ducharme and Vincent presented the model at NIPS 2000, and the journal version with Jauvin appeared in JMLR in 2003.
Earlier work had used neural networks to model joint distributions of discrete variables and to learn distributed representations of symbols; the new step was a language model trained on millions of words with a learned word feature matrix.
Training was slow, and Bengio and Senécal (2003) introduced importance sampling for the output layer, followed by the tree of Morin and Bengio (2005).
Schwenk and Gauvain put the model into large-vocabulary speech recognition.
Mikolov et al. (2010) replaced the fixed window with a recurrent network, and Mnih and Teh (2012) trained the output layer by noise-contrastive estimation.
Mikolov et al. (2013) dropped the hidden layer altogether to learn word vectors, word2vec, which made learned embeddings a standard input to other models.
Press and Wolf (2017) and Inan et al. (2017) tied the input and output vectors, and the [[transformer]] language models kept the embedding, the tied softmax and the training objective of this model, replacing only the network in between.
