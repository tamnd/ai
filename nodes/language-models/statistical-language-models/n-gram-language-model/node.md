---
tag: NEW-n-gram-language-model
kind: method
slug: n-gram-language-model
name: N-gram language model
aliases: [n-gram model, n-gram LM, Markov language model, Kneser-Ney language model, Stupid Backoff, infini-gram, infinity-gram]
area: language-models/statistical-language-models
level: core
summary: >-
  A language model that predicts each word from the previous n − 1 words only, estimated from counts in a corpus and smoothed, as by Kneser-Ney, so that unseen word sequences still get probability.
purpose: assigning a probability to each next word given the previous words, from corpus counts, for speech recognition, machine translation and as a baseline for neural language models
uses: [conditional-probability@m2, markov-chain@m2, maximum-likelihood-estimation@m2, cross-entropy@m2, shannon-entropy@m2]
related_to: [byte-pair-encoding, long-short-term-memory, transformer, neural-scaling-laws, hidden-markov-model@al]
hyperparameters:
  - {name: order N, meaning: "the number of words in each n-gram: the model conditions on the previous N − 1 words", typical: "3 for the speech systems of Brown et al. (1992); 5 for machine translation and the One Billion Word baselines; gains stop by 5 to 7 in Goodman (2001); unbounded for the infinity-gram"}
  - {name: discounts D_n(k), meaning: "the amount subtracted from an adjusted count k of an n-gram of length n, which funds the backoff weight", typical: "one discount per order (interpolated Kneser-Ney) or three per order for counts 1, 2 and 3+ (modified Kneser-Ney), both from the closed-form estimates of the count-of-counts"}
  - {name: alpha, meaning: "the fixed backoff factor of Stupid Backoff", typical: "0.4 (Brants et al., 2007)"}
  - {name: vocabulary, meaning: "the set of words with their own probability; every other word is mapped to an unknown word", typical: "all training words, or the words above a count threshold; 793 471 words in the One Billion Word benchmark"}
  - {name: pruning threshold, meaning: "the entropy or count threshold below which n-grams are dropped to shrink the model", typical: "none for the unpruned baselines; 15M n-grams for the pruned One Billion Word models"}
cost:
  train: "one pass to count the n-grams of every order up to N and a few sorts of the counts; time and memory linear in the number of distinct n-grams, which grows with both the corpus and N"
  inference: "at most N table lookups per token; one sum over the vocabulary to normalise only when a full distribution is needed; an infinity-gram query is a few binary searches in a suffix array"
history:
  year: 1948
  people: [Claude Shannon, Frederick Jelinek, Slava Katz, Reinhard Kneser, Hermann Ney, Stanley Chen, Joshua Goodman]
  note: "Shannon (1948) generated text from n-gram approximations of English; Jelinek and Mercer (1980) interpolated estimates of several orders; Katz (1987) introduced backoff; Kneser and Ney (1995) introduced continuation counts; Chen and Goodman (1996, 1998) compared the smoothing methods and proposed modified Kneser-Ney."
prereqs: [conditional-probability@m2, markov-chain@m2, maximum-likelihood-estimation@m2, cross-entropy@m2]
same_as:
  arxiv: ["cmp-lg/9606011", "cs/0108005", "1312.3005", "2401.17377"]
sources:
  - {paper: "Brown, deSouza, Mercer, Della Pietra and Lai (1992)", title: "Class-Based n-gram Models of Natural Language", locator: "Computational Linguistics 18(4), pp. 467 to 479; Sections 1 and 2", note: "language models, n-gram models, perplexity, sparsity and interpolated estimation"}
  - {paper: "Chen and Goodman (1996)", title: "An Empirical Study of Smoothing Techniques for Language Modeling", locator: "ACL 1996; arXiv cmp-lg/9606011v1", note: "the comparison of smoothing methods"}
  - {paper: "Goodman (2001)", title: "A Bit of Progress in Language Modeling", locator: "MSR-TR-2001-72; arXiv cs/0108005v1, Sections 1, 2 and the appendix", note: "Kneser-Ney, its modified form, higher orders and the marginal-preservation argument"}
  - {paper: "Heafield, Pouzyrevsky, Clark and Koehn (2013)", title: "Scalable Modified Kneser-Ney Language Model Estimation", locator: "ACL 2013, Section 3", note: "adjusted counts, closed-form discounts, normalisation and interpolation"}
  - {paper: "Heafield (2011)", title: "KenLM: Faster and Smaller Language Model Queries", locator: "WMT 2011", note: "probing hash tables and tries for queries"}
  - {paper: "Brants, Popat, Xu, Och and Dean (2007)", title: "Large Language Models in Machine Translation", locator: "EMNLP-CoNLL 2007, Sections 3 and 7", note: "Stupid Backoff and training on up to 2 trillion tokens"}
  - {paper: "Teh (2006)", title: "A Hierarchical Bayesian Language Model based on Pitman-Yor Processes", locator: "COLING-ACL 2006, Sections 3 and 4", note: "interpolated Kneser-Ney as an approximate hierarchical Pitman-Yor model"}
  - {paper: "Chelba et al. (2013)", title: "One Billion Word Benchmark for Measuring Progress in Statistical Language Modeling", locator: "arXiv 1312.3005v3, Table 1", note: "n-gram and neural baselines"}
  - {paper: "Jozefowicz et al. (2016)", title: "Exploring the Limits of Language Modeling", locator: "arXiv 1602.02410v2, the abstract", note: "LSTM perplexities on the same benchmark"}
  - {paper: "Bengio, Ducharme, Vincent and Jauvin (2003)", title: "A Neural Probabilistic Language Model", locator: "JMLR 3, Sections 1 and 4", note: "the weaknesses of n-gram models and the first neural comparison"}
  - {paper: "Liu, Min, Zettlemoyer, Choi and Hajishirzi (2024)", title: "Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens", locator: "arXiv 2401.17377v3, Sections 2 and 4", note: "the infinity-gram and its suffix-array engine"}
---

## In plain words

To guess the next word, look only at the last few words, find every place in a large text where those words occur, and see what came next.
If "the cat" was followed by "sat" in two of its three occurrences, guess "sat" with probability about two thirds.
That raw guess fails on anything the text never contained, so some probability is taken from the seen continuations and given to a guess from a shorter context, and finally to every word.
Kneser-Ney smoothing does this well by asking, for the shorter guess, how many different words a word follows instead of how often it occurs.
The model is a table of counts, cheap to build and query, and was the standard language model until neural networks replaced it.

## Setting

Brown et al. write the probability of a string as the product $\Pr(w_1)\Pr(w_2 \mid w_1)\cdots\Pr(w_k \mid w_1 \dots w_{k-1})$, call $w_1 \dots w_{k-1}$ the history and $w_k$ the prediction, and call a computational mechanism for obtaining these conditional probabilities a language model. {#c-brown-lm}
The product is exact by the chain rule; the modelling is in the conditional probabilities.
In an n-gram language model, two histories are treated as equivalent if they end in the same $n - 1$ words. {#c-brown-ngram}
So the model is a Markov chain of order $n - 1$ over words:

$$P(w_1 \dots w_k) = \prod_{i=1}^{k} P(w_i \mid w_{i-n+1} \dots w_{i-1}).$$ {#eq-markov}

Brown et al. count $V^n - 1$ independent parameters for an n-gram model over a vocabulary of $V$ words, and note that the IBM Tangora speech recognition system, with a vocabulary of about 20,000 words, employs a 3-gram language model with over eight trillion parameters. {#c-brown-parameters}
Almost all of those n-grams never occur in any corpus, so the method is mostly about estimating probabilities from counts that are zero or small.
Sentences are padded: words near the beginning of a sentence form n-grams padded by the marker `<s>`, possibly repeated, and the end marker `</s>` is appended to each sentence and acts like a normal token. {#c-heafield-padding}
Predicting `</s>` makes the probabilities of all finite sentences sum to one.

The model is judged by the probability it gives to held-out text.
The perplexity of a language model with respect to a sample of text is the reciprocal of the geometric average of the probabilities of the predictions in the sample. {#c-brown-perplexity}
For $T$ predicted tokens,

$$\mathrm{PP} = \Big(\prod_{i=1}^{T} P(w_i \mid h_i)\Big)^{-1/T} = 2^{H}, \qquad H = -\frac{1}{T}\sum_{i=1}^{T} \log_2 P(w_i \mid h_i),$$ {#eq-perplexity}

where $H$ is the [[m2:cross-entropy|cross-entropy]] of the text under the model in bits per token.

## The method

**Maximum likelihood.** Write $c(\cdot)$ for counts in the padded training text.
Brown et al. call estimating each order-n parameter by the relative frequency of the n-gram among the n-grams with the same history sequential maximum likelihood estimation. {#c-brown-sequential-ml}

$$P_{\mathrm{ML}}(w \mid h) = \frac{c(h\,w)}{c(h\,\cdot)},$$ {#eq-mle}

where $c(h\,\cdot)$ counts the occurrences of $h$ followed by any token.
In 365,893,263 words of running text, of the $1.773 \times 10^{16}$ possible 3-grams only 75,349,888 occurred and 53,737,350 of these occurred only once; from Turing's formula, Brown et al. expect maximum likelihood estimates to be 0 for 14.7 percent of the 3-grams and for 2.2 percent of the 2-grams in a new sample of English text. {#c-brown-sparsity}
A single zero makes the probability of a sentence zero and its perplexity infinite.
Chen and Goodman note that the maximum likelihood estimate is a poor one when the amount of training data is small compared to the size of the model, which is generally the case in language modeling, and that Gale and Church argued additive smoothing generally performs poorly. {#c-chen-ml-poor}
Additive smoothing adds a constant to every count; it gives unseen words probability, but takes far too much from the seen ones.

**Interpolation and backoff.** The remedy is to fall back on shorter histories, whose counts are larger.
Brown et al. combine the estimates of 1-, 2- and 3-gram models by the interpolated estimation of Jelinek and Mercer (1980), choosing the weights with the help of the EM algorithm to maximize the probability of held-out data; they divide the weights into 1,782 sets according to the 2-gram counts and estimate them from a held-out sample of 4,630,934 words. {#c-brown-interpolated}
Backoff methods instead use the shorter history only when the longer count is zero.
Goodman reports that backoff smoothing methods, which use lower order distributions only when the higher order count is missing, do not do well on low counts such as one counts and two counts, while interpolated models always combine both the higher-order and the lower order distribution and typically work better. {#c-goodman-interpolated}

**Kneser-Ney: the continuation count.** Kneser-Ney smoothing uses a modified backoff distribution based on the number of contexts each word occurs in, rather than the number of occurrences of the word; the word Francisco is common but occurs in exceedingly few contexts, so its probability of occurring in a new one is very low. {#c-goodman-continuation}
The lower-order distribution is only consulted when the longer context has not explained the word, and Francisco is nearly always explained by San.
Heafield et al. replace the counts $c$ with adjusted counts $a$: $a(w_1 \dots w_n)$ is $c(w_1 \dots w_n)$ if $n = N$ or $w_1 = $ `<s>`, and otherwise the number of words $v$ with $c(v\,w_1 \dots w_n) > 0$. {#c-heafield-adjusted}

$$a(w_1 \dots w_n) = \begin{cases} c(w_1 \dots w_n) & n = N \text{ or } w_1 = \texttt{<s>}, \\ \lvert\{v : c(v\,w_1 \dots w_n) > 0\}\rvert & \text{otherwise.} \end{cases}$$ {#eq-adjusted}

**Discounts.** For each length $n$, Heafield et al. collect the number $t_{n,k}$ of n-grams with adjusted count $k$ in $[1, 4]$ and compute closed-form estimates of the discounts $D_n(k) = k - (k + 1)\, t_{n,1} t_{n,k+1} / ((t_{n,1} + 2 t_{n,2})\, t_{n,k})$ for $k$ in $[1, 3]$, with $D_n(0) = 0$ and $D_n(k) = D_n(3)$ for $k \ge 3$. {#c-heafield-discounts}

$$D_n(k) = k - (k+1)\,\frac{t_{n,1}}{t_{n,1} + 2t_{n,2}}\,\frac{t_{n,k+1}}{t_{n,k}}, \qquad k = 1, 2, 3.$$ {#eq-discounts}

Modified Kneser-Ney smoothing, proposed by Chen and Goodman, uses multiple discounts, one for one counts, another for two counts, and another for three or more counts, and typically works slightly better than Interpolated Kneser-Ney. {#c-goodman-modified}
Interpolated Kneser-Ney uses the single discount $D_n = t_{n,1} / (t_{n,1} + 2t_{n,2})$ for every count.

**Normalisation and interpolation.** Write $h = w_1 \dots w_{n-1}$ and $h' = w_2 \dots w_{n-1}$ for the context with its first word dropped.
Normalization computes the pseudo probability $u$, the adjusted count minus its discount divided by the sum of the adjusted counts of the context, and the backoff $b$, the sum of the discounts of the context's entries divided by the same denominator. {#c-heafield-normalisation}

$$u(w \mid h) = \frac{a(h\,w) - D_n(a(h\,w))}{\sum_{x} a(h\,x)}, \qquad b(h) = \frac{\sum_{x} D_n(a(h\,x))}{\sum_{x} a(h\,x)}.$$ {#eq-normalise}

The interpolation step computes the final probability $p(w_n \mid w_1 \dots w_{n-1}) = u(w_n \mid w_1 \dots w_{n-1}) + b(w_1 \dots w_{n-1})\, p(w_n \mid w_2 \dots w_{n-1})$; the recursion terminates when unigrams are interpolated with the uniform distribution, and the unknown word counts as part of the vocabulary with count zero, so its probability is $b(\varepsilon)/\lvert\text{vocabulary}\rvert$. {#c-heafield-interpolation}

$$p(w \mid h) = u(w \mid h) + b(h)\, p(w \mid h'), \qquad p(w \mid \varepsilon) = u(w \mid \varepsilon) + \frac{b(\varepsilon)}{\lvert V \rvert}.$$ {#eq-interpolate}

To avoid zero probabilities, Goodman always smooths the unigram distribution of a Kneser-Ney model with the uniform distribution over the vocabulary. {#c-goodman-uniform}
A context never seen in training has no entries; it takes $u = 0$ and $b = 1$, so its distribution is that of its suffix.
`<s>` is never predicted and is left out of the vocabulary that the probabilities sum over.

**Stupid Backoff.** For very large corpora Brants et al. drop normalisation altogether.
Stupid Backoff does not generate normalized probabilities: it applies no discounting and directly uses the relative frequency $f(h\,w)/f(h)$ when the n-gram was seen, and otherwise $\alpha$ times the score of the shorter context, with a single $\alpha$ heuristically set to 0.4; the recursion ends at unigrams with $f(w)/N$, $N$ being the size of the training corpus. {#c-brants-stupid}

$$S(w \mid h) = \begin{cases} \dfrac{c(h\,w)}{c(h)} & c(h\,w) > 0, \\ \alpha\, S(w \mid h') & \text{otherwise}, \end{cases} \qquad S(w) = \frac{c(w)}{N}.$$ {#eq-stupid}

**Infinity-gram.** Liu et al. drop the fixed order instead.
The infinity-gram backs off only when the denominator is zero, so the effective n is equal to one plus the length of the prompt's longest suffix that appears in the training data; unlike Katz backoff, it is a valid distribution by construction and does not require discounting. {#c-infini-definition}
Counts of arbitrary strings come from a suffix array of the tokenized corpus, and a count is two binary searches.

## Derivation

Six derivations are written out with checks in `derivations/`.

`kn-normalisation.md` proves that the recursion of Eq. 6, Eq. 7 is a distribution.
For a seen context, $\sum_w u(w \mid h) = 1 - b(h)$ because the discounts subtracted in $u$ are exactly those added in $b$; by induction on the order, $\sum_w p(w \mid h) = (1 - b(h)) + b(h) \cdot 1 = 1$.
With discounts $0 \le D(k) \le k$, interpolated Kneser-Ney gives non-negative probabilities that sum to exactly one over the vocabulary for every context, seen or not, and all are positive when every discount is. {#c-kn-normalised}
The check builds 60 models with random discounts and sums 2259 context distributions in exact fractions.

`kn-continuation.md` explains the continuation count.
For an interpolated bigram with absolute discount $0 < D \le 1$, the only lower-order distribution that keeps the observed marginal of the predicted word is the share of distinct bigram types that end in each word. {#c-kn-continuation}
Goodman proves that any modeling technique that does not preserve known marginal distributions can be improved by preserving those distributions, and cites Chen and Goodman for the result that the smoothing technique that uses absolute discounting, interpolation, and preserves marginals is Interpolated Kneser-Ney smoothing. {#c-goodman-marginals}
In the check's San Francisco corpus, $p(\text{francisco} \mid \text{on})$ is 0.0114 with continuation counts and 0.0532 with relative frequencies.

`good-turing-leftover.md` gives Turing's estimate of the unseen mass.
The share $n_1/N$ of tokens whose n-gram occurred once equals the share of tokens whose n-gram would be unseen if that token were held out, and for independent draws its expectation exceeds the expected unseen probability mass by at most $1/N$. {#c-gt-leave-one-out}
Dividing the counts of n-grams seen once by the 365,893,263 words of Brown et al.'s Table 1 gives 14.69% for 3-grams and 2.20% for 2-grams. {#c-gt-brown-numbers}

`perplexity-entropy.md` relates the two scales of Eq. 2.
Perplexity equals 2 to the cross-entropy in bits per token, so cutting perplexity by a fraction $r$ removes $\log_2(1/(1 - r))$ bits whatever the start: 100 to 90 is 6.64 to 6.49 bits, and 67.6 to 43.8 removes 10.3% of the cross-entropy. {#c-perplexity-bits}

`infinigram-normalisation.md` concerns the backoff rules without discounts.
Choosing the longest suffix that occurs followed by a token, before looking at the next word, and dividing by the number of such occurrences gives a distribution; dividing by all occurrences loses mass whenever the suffix also ends a document. {#c-infinigram-valid}
Backing off separately for each next word without discounting, as naive backoff and Stupid Backoff do, can give scores that sum to more than one. {#c-backoff-not-normalised}
In the bigram Stupid Backoff model of the sentences "a b", "a c" and "b b", the scores after a sum to 11/9.

`ikn-as-hpy.md` connects Kneser-Ney to a Bayesian model.
With strengths zero, one discount in $(0, 1]$ per order and at most one table per word in each restaurant, the predictive probability of the hierarchical Pitman-Yor language model equals interpolated Kneser-Ney with continuation counts at every lower order; keeping raw counts for n-grams that start with `<s>` falls outside this correspondence. {#c-ikn-hpy}
The check compares 7410 probabilities in exact fractions.

## Worked example

Train a bigram interpolated Kneser-Ney model on three sentences: "the cat sat", "the dog sat", "the cat ran".
After padding, the bigrams are `<s>` the 3, the cat 2, the dog 1, cat sat 1, cat ran 1, dog sat 1, sat `</s>` 2 and ran `</s>` 1.
Their count-of-counts are $t_{2,1} = 5$ and $t_{2,2} = 2$, so $D_2 = 5/(5 + 4) = 5/9$.

The unigram level uses continuation counts: the follows only `<s>`, so $a(\text{the}) = 1$, and likewise cat 1, dog 1, ran 1, while sat follows cat and dog, so $a(\text{sat}) = 2$, and $a(\texttt{</s>}) = 2$.
They sum to 8, with $t_{1,1} = 4$ and $t_{1,2} = 2$, so $D_1 = 4/8 = 1/2$ and $b(\varepsilon) = 6 \cdot \tfrac12 / 8 = 3/8$.
The vocabulary is the, cat, dog, sat, ran, `</s>` and the unknown word, $\lvert V \rvert = 7$.
So $p(\text{the}) = (1 - \tfrac12)/8 + \tfrac38 \cdot \tfrac17 = 13/112$, the same for cat, dog and ran; $p(\text{sat}) = p(\texttt{</s>}) = 27/112$; and the unknown word gets $3/56$.

After the, the entries are cat 2 and dog 1, out of 3.
Then $u(\text{cat} \mid \text{the}) = (2 - \tfrac59)/3 = 13/27$, $b(\text{the}) = 2 \cdot \tfrac59 / 3 = 10/27$, and

$$p(\text{cat} \mid \text{the}) = \frac{13}{27} + \frac{10}{27} \cdot \frac{13}{112} = \frac{793}{1512} \approx 0.5245.$$

Maximum likelihood gives 2/3 and add-one smoothing gives $3/10$.
Likewise $p(\text{dog} \mid \text{the}) = 289/1512$, and the five words never seen after the share $10/27$ in proportion to their unigram probabilities.

Score the held-out sentence "the dog ran", which has the unseen bigram dog ran.

| Token | Context | Probability |
|---|---|---|
| the | `<s>` | 281/336 ≈ 0.8363 |
| dog | the | 289/1512 ≈ 0.1911 |
| ran | dog | 65/1008 ≈ 0.0645 |
| `</s>` | ran | 583/1008 ≈ 0.5784 |

The log probability is −5.1224 nats over 4 tokens, so the perplexity is $e^{5.1224/4} \approx 3.599$.
Under maximum likelihood $p(\text{ran} \mid \text{dog}) = 0$ and the perplexity is infinite.
Stupid Backoff with $\alpha = 0.4$ scores cat after the as 2/3 and sat after the as $0.4 \cdot 2/12$; its scores after the sum to $1 + 0.4 \cdot 9/12 = 1.3$.

The shared fixture of the tests is a larger case with a known answer.
It draws 6000 training and 300 test sentences from a second-order Markov source over 2000 words, so the true cross-entropy of the 3043 test tokens is known: perplexity 24.2.

| Order | Interpolated KN | Modified KN |
|---|---|---|
| 1 | 121.1 | 119.7 |
| 2 | 54.6 | 53.2 |
| 3 | 46.5 | 44.9 |
| 4 | 49.1 | 47.1 |

Modified Kneser-Ney is lower at every order.
The trigram matches the true order of the source; the 4-gram is worse because its extra counts are sparse and add noise, and none comes near the source's 24.2 on this little data.

## Hyperparameters

The order $N$ trades context against sparsity.
Goodman examines n-gram orders up to 20-grams and finds that even for very large training sets, performance has plateaued by 5 to 7 grams. {#c-goodman-plateau}
Smaller corpora favour smaller orders, as in the fixture table above.

The discounts are usually not tuned but computed from the count-of-counts by Eq. 5, one per order for interpolated Kneser-Ney and three per order for modified Kneser-Ney.
They need $t_{n,1}, \dots, t_{n,4} > 0$ and must come out in $[0, k]$; the implementations raise an error otherwise.
Stupid Backoff has the single factor $\alpha = 0.4$.
The vocabulary decides which words have their own probability; perplexities computed with different vocabularies or different handling of unknown words are not comparable.

## Cost

Training counts every n-gram of every order up to $N$ and then sorts the counts once per order, so time and memory are linear in the number of distinct n-grams.
Liu et al. note that n-gram count tables grow almost exponentially with $n$, and that the 5-gram count table for a 1.4-trillion-token corpus would consume 28 TB of disk space. {#c-infini-table}
Using one machine with 140 GB RAM for 2.8 days, Heafield et al. built an unpruned model on 126 billion tokens, which improved machine translation by 0.8 BLEU point over constrained systems; on 302 million tokens their estimation used 7.7% of the RAM and 14.0% of the wall time of SRILM. {#c-heafield-scale}
Heafield et al. report that the MapReduce estimation of Kneser-Ney models by Brants et al. took 400 machines for two days on 31 billion words. {#c-heafield-mapreduce}

A query needs at most $N$ lookups, one per order, each a hash or trie lookup.
Compared with the widely-used SRILM, the PROBING data structure of KenLM, which uses linear probing hash tables, is 2.4 times as fast while using 57% of the memory, and the TRIE data structure is a trie with bit-level packing and sorted records. {#c-kenlm-probing}
A suffix array instead stores one integer per token and answers a count of any string with binary searches.
Liu et al. train an n-gram language model on 5 trillion tokens, serve infinity-gram probabilities with millisecond-level latency from suffix arrays, and find that the infinity-gram has a next-token prediction accuracy of 47% on human-written text. {#c-infini-scale}

## Evidence

The Brown corpus, which contains 1,014,312 words and is not included in the training or held-out data, has a perplexity of 244 with respect to the interpolated 3-gram model of Brown et al. {#c-brown-244}
In an extensive empirical comparison of smoothing techniques measured by the cross-entropy of test data, additive smoothing performs poorly and the methods katz and interp-held-out consistently perform well, while performance varies widely with respect to training set size and n-gram order. {#c-chen-comparison}
Kneser-Ney variants came out ahead in the longer study of Chen and Goodman (1998), which this page does not cite directly.
Combining language modeling techniques gives up to a 50% perplexity reduction over a Katz smoothed trigram on 100,000 words of training data, which is one bit of entropy; the improvement declines to 41% on 284,000,000 words, and on a similar large set without punctuation the reduction is 38% with an 8.9% word error rate reduction. {#c-goodman-combination}
Goodman notes that a 10% perplexity reduction, from 100 to 90, corresponds to a reduction from 6.64 bits to 6.49 bits, which is only a 2% entropy reduction. {#c-goodman-entropy}

On the One Billion Word Benchmark, an interpolated Kneser-Ney 5-gram with 1.1B n-grams and 1.76 billion parameters has perplexity 67.6, a Katz 5-gram of the same size 79.9, and a recurrent neural network with 1024 hidden units combined with a MaxEnt 9-gram 51.3. {#c-chelba-baselines}
The baseline unpruned Kneser-Ney 5-gram model achieves perplexity 67.6, and a combination of techniques leads to 35% reduction in perplexity, or 10% reduction in cross-entropy (bits), over that baseline. {#c-chelba-combination}
Large [[long-short-term-memory|LSTM]] language models improve the single-model perplexity on the One Billion Word Benchmark from 51.3 down to 30.0, while an ensemble of models improves perplexity from 41.0 down to 23.7. {#c-jozefowicz}
A neural network language model obtains a test perplexity difference of about 24% on Brown and about 8% on AP News against the best of the smoothed n-gram models. {#c-bengio-result}

Brants et al. train language models on up to 2 trillion tokens, resulting in language models having up to 300 billion n-grams, and propose Stupid Backoff, which is inexpensive to train on large data sets and approaches the quality of Kneser-Ney smoothing as the amount of training data increases. {#c-brants-scale}
The fraction of 5-grams in the test data that are known to the language model grows from 0.06 for 13 million tokens to 0.56 for 2 trillion tokens; perplexities cannot be calculated for language models with Stupid Backoff because their scores are not normalized probabilities. {#c-brants-coverage}
With Kneser-Ney smoothing the BLEU score improves from 0.3559 for 13 million tokens to 0.3832 for 237 million tokens, where Stupid Backoff with $\alpha = 0.4$ is around 1 BP worse; with more data the gap narrows from 0.85 BP to a not significant 0.24 BP, and a Stupid Backoff model on web data surpasses every Kneser-Ney model, each trained on less data, by at least 0.82 BP. {#c-brants-bleu}

On human-written text, over 90% of tokens have an effective n of at least 5, and the median effective n is 7 (mean 9.1). {#c-infini-effective-n}
Heuristically interpolating between the estimates made by the infinity-gram and neural language models can reduce perplexity by up to 73% compared to the neural models alone, even when the neural model is as large as 70B. {#c-infini-interpolation}

## Limitations and failure modes

Bengio et al. note the curse of dimensionality in statistical language modeling, and that n-gram approaches obtain generalization by concatenating very short overlapping sequences seen in the training set. {#c-bengio-curse}
Bengio et al. name two weaknesses of n-gram models: they do not take into account contexts farther than 1 or 2 words, and they do not take into account the similarity between words. {#c-bengio-limits}
Seeing "the cat sat" says nothing about "the dog sat" unless that exact bigram was counted; a neural model shares the evidence through word vectors.
Agreement across a long clause, topic and anything else beyond the window are invisible to the model.

Pruning interacts badly with Kneser-Ney.
Pruned to 15M n-grams, the interpolated Kneser-Ney 5-gram has perplexity 243.2 against 127.5 for the Katz 5-gram pruned to 15M n-grams. {#c-chelba-pruning}
Chelba et al. note that under pruning Kneser-Ney smoothing degrades much more rapidly than Katz, calling for a discerning choice in a given application. {#c-chelba-pruning-rate}
The lower orders of Kneser-Ney are continuation distributions, poor on their own, and pruning sends more queries to them.

Stupid Backoff and the infinity-gram are not smoothed in the usual sense.
Stupid Backoff scores are not probabilities, so they cannot give a perplexity or be mixed with other models as probabilities.
The infinity-gram gives probability zero to every word not seen after the longest matching suffix, so on its own its perplexity is infinite; Liu et al. use it interpolated with a neural model.
A long matching suffix may also be a verbatim copy of the training text, which is useful for finding memorised data and harmful for generating new text.

## Variants

**Katz backoff** uses the discounted higher-order estimate when its count is positive and otherwise backs off with a weight that renormalises.
**Jelinek-Mercer** interpolation mixes the maximum-likelihood estimates of all orders with weights fit on held-out data.
**Interpolated and modified Kneser-Ney** are the variants of the method section, the default in SRILM and KenLM.
**Hierarchical Pitman-Yor.** According to Teh, an approximation to the hierarchical Pitman-Yor language model recovers the exact formulation of interpolated Kneser-Ney, and reports cross entropy results superior to interpolated Kneser-Ney and comparable to modified Kneser-Ney. {#c-teh-hpy}
Since the discounts in a hierarchical Pitman-Yor language model are limited to between 0 and 1, Teh concludes that modified Kneser-Ney is not an approximation of the hierarchical Pitman-Yor language model. {#c-teh-modified}
**Class-based models** of Brown et al. predict a word class and then the word, which shares counts across words of a class.
**Stupid Backoff** and the **infinity-gram** are the unnormalised and the unbounded variants above.
A [[al:hidden-markov-model|hidden Markov model]] puts the Markov chain on hidden states instead of words.

## When to use it

Use a Kneser-Ney model as the first baseline for any language modelling task, and to measure how much a neural model gains.
Use one where compute or latency is tight: first-pass decoding in speech recognition, on-device prediction, or scoring billions of sentences to filter data.
Use the infinity-gram over a large corpus to find where a text or a model output copies the training data, or to interpolate with a neural model.
Use a neural model, such as a [[transformer]], when the task needs context beyond a few words or generalisation across similar words; the trend of [[neural-scaling-laws|neural scaling laws]] has no counterpart for n-grams, whose gains with more data come only from more counted strings.

## Pitfalls

> [!pitfall]
> Compare perplexities only on the same test text, vocabulary and tokenization: changing the [[byte-pair-encoding|tokenizer]] changes the number of tokens $T$ in Eq. 2, and mapping more words to the unknown word lowers perplexity without a better model.

> [!pitfall]
> Count `</s>` as a predicted token and never predict `<s>`; leaving out the end marker makes sentence probabilities sum to more than one over all lengths.

> [!pitfall]
> Do not prune a Kneser-Ney model with the thresholds tuned for Katz, and re-estimate after pruning rather than deleting entries; Chelba et al. report how fast Kneser-Ney degrades under pruning.

> [!pitfall]
> Check the closed-form discounts: with few singletons or doubletons the estimates can leave $[0, k]$, and the counts must come from the adjusted counts at each lower order, not the raw counts.

> [!pitfall]
> Do not report a perplexity for Stupid Backoff scores; they do not sum to one.

## History

Shannon (1948) generated random text from letter and word n-gram statistics of English as successive approximations to the language.
Speech recognition at IBM made n-gram models the practical language model; Jelinek and Mercer (1980) interpolated estimates of several orders, and Katz (1987) introduced backoff with Good-Turing discounts.
Kneser and Ney (1995) replaced the lower-order counts with continuation counts.
Chen and Goodman (1996, 1998) compared the smoothing methods systematically and proposed modified Kneser-Ney, which became the default.
Bengio et al. (2003) proposed a neural language model that beat smoothed n-grams on perplexity, and Teh (2006) explained Kneser-Ney as approximate inference in a hierarchical Pitman-Yor process.
Brants et al. (2007) trained on up to 2 trillion tokens with Stupid Backoff, and Heafield et al. (2013) built a Kneser-Ney model on 126 billion tokens with one machine.
On the One Billion Word benchmark of Chelba et al. (2013), recurrent and [[long-short-term-memory|LSTM]] models pulled clearly ahead, and neural language models took over.
Liu et al. (2024) returned to counts at the scale of trillions of tokens with the unbounded infinity-gram.
