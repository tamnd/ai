---
tag: NEW-perplexity
kind: metric
slug: perplexity
name: Perplexity
aliases: [PPL, word perplexity, token perplexity, character perplexity, bits per character, bits per byte, BPB, BPC, test-set perplexity]
area: language-models/language-model-evaluation
level: core
summary: >-
  The exponential of a language model's average negative log-likelihood per token on held-out text: the geometric mean of the inverse probabilities it gave the tokens that occurred.
related_to: [n-gram-language-model, byte-pair-encoding, decoder-only-transformer, nucleus-sampling, neural-scaling-laws, emergent-abilities]
properties:
  measures: "how well a probabilistic language model predicts held-out text: the exponential of its cross-entropy per unit of text"
  range: "1 to infinity; exactly V for a model that is uniform over V tokens"
  better: lower
  formula: "exp(-(1/N) sum_i ln p(x_i | x_<i)), with the sum over the scored tokens of the test text and N its number of units"
  level: "per corpus: the negative log-likelihood of all sequences is pooled and divided by their total number of units"
  reference_impl: "no single package; exp of the mean token cross-entropy, as in impl/numpy/perplexity.py, which follows the definition of Brown et al. (1992, J92-4003, p. 3)"
history:
  year: 1977
  people: [L. Bahl, J. Baker, F. Jelinek, R. Mercer]
  note: "Brown et al. (1992, J92-1002) cite Bahl, Baker, Jelinek and Mercer (1977), 'Perplexity: a measure of the difficulty of speech recognition tasks', for word perplexity; Brown et al. (1992, J92-4003) define the perplexity of a language model on a sample of text; Gao et al. (2020) report bits per UTF-8 byte instead, and Fang et al. (ICLR 2025) restrict perplexity to key tokens for long contexts."
prereqs: [cross-entropy@m2, shannon-entropy@m2, cross-entropy-loss@al]
same_as:
  wikidata: null
  wikipedia: Perplexity
  arxiv: ["2101.00027", "1906.04726", "2106.00085", "2410.23771", "1609.07843"]
sources:
  - {paper: "Brown, deSouza, Mercer, Della Pietra and Lai (1992)", title: "Class-Based n-gram Models of Natural Language", locator: "Computational Linguistics 18(4), J92-4003, pp. 3 and 4", note: "the definition of perplexity on a sample of text and the perplexity of 244 on the Brown corpus"}
  - {paper: "Brown, Della Pietra, Mercer, Della Pietra and Lai (1992)", title: "An Estimate of an Upper Bound for the Entropy of English", locator: "Computational Linguistics 18(1), J92-1002, pp. 1, 2, 8 and 9", note: "cross-entropy as an upper bound on entropy, the compression view, character perplexity and 1.75 bits per character"}
  - {paper: "Goodman (2001)", title: "A Bit of Progress in Language Modeling", locator: "arXiv cs/0108005v1, pp. 2, 4 and 5", note: "perplexity as the geometric average of inverse probabilities, entropy as its logarithm, and perplexity against word error rate"}
  - {paper: "Gao, Biderman, Black and others (2020)", title: "The Pile: An 800GB Dataset of Diverse Text for Language Modeling", locator: "arXiv 2101.00027v1, p. 6", note: "bits per UTF-8 byte, the conversion from a token loss, and evaluating each document on its own"}
  - {paper: "Mielke, Cotterell, Gorman, Roark and Eisner (2019)", title: "What Kind of Language Is Hard to Language-Model?", locator: "ACL 2019, P19-1491, PDF p. 3", note: "surprisal as bits, open-vocabulary models for fair comparison, and the canonical segmentation"}
  - {paper: "Radford, Wu, Child, Luan, Amodei and Sutskever (2019)", title: "Language Models are Unsupervised Multitask Learners", locator: "OpenAI report, pp. 4 and 5", note: "evaluation per canonical unit and invertible de-tokenizers"}
  - {paper: "Holtzman, Buys, Du, Forbes and Choi (2020)", title: "The Curious Case of Neural Text Degeneration", locator: "ICLR 2020, arXiv 1904.09751v2, pp. 6 and 7", note: "the perplexity of generated text against human text"}
  - {paper: "Meister and Cotterell (2021)", title: "Language Model Evaluation Beyond Perplexity", locator: "ACL-IJCNLP 2021, 2021.acl-long.414, PDF pp. 1 and 2", note: "what perplexity does not show about the statistical tendencies of language"}
  - {paper: "Fang, Wang, Liu and others (2025)", title: "What is Wrong with Perplexity for Long-context Language Modeling?", locator: "ICLR 2025, arXiv 2410.23771v5, pp. 1, 2 and 4", note: "why perplexity misses long-context ability, the long-short difference and LongPPL"}
  - {paper: "Merity, Xiong, Bradbury and Socher (2016)", title: "Pointer Sentinel Mixture Models", locator: "arXiv 1609.07843v1, pp. 1 and 5", note: "WikiText-2 and WikiText-103, their vocabularies and out-of-vocabulary rates"}
---

Perplexity scores a language model by how much probability it gives to text it has not seen.
Run the model over a held-out text, take the probability it gave to each token that actually came next, and average the negative logarithms: that is the cross-entropy of the text under the model.
Perplexity is the exponential of that average.
It is the geometric mean of the inverse probabilities, so a perplexity of 20 means the model was, on average, as uncertain as if it had to pick among 20 equally likely tokens.
Lower is better, the minimum is 1, and one token given probability 0 makes it infinite.
The number depends on what is counted as a unit, so a perplexity needs its tokenizer, its unit and its test text next to it.

## In plain words

A language model reads text one piece at a time and, before each piece, spreads its bets over every piece that could come next.
When the real next piece arrives, look at the share of the bet that the model put on it.
A good model puts a large share on what actually happens.
Multiply those shares over a long text and take the root that turns the product into a per-piece average: this is the typical share the model bet on the right answer.
Perplexity is one over that typical share.
If the model typically put a quarter of its bet on the right piece, its perplexity is 4: it was as unsure as someone choosing among four equally likely options.
A perfect guesser scores 1, and a model that never bets anything on something that then happens scores infinity.

## Definition

A model $p$ assigns each token $x_i$ of a test text a conditional probability $p(x_i \mid x_{<i})$ given the tokens before it.
Over $T$ scored tokens, the per-token cross-entropy and the perplexity are

$$ H = -\frac{1}{T}\sum_{i=1}^{T} \ln p(x_i \mid x_{<i}), \qquad \mathrm{PPL} = e^{H} = \Big(\prod_{i=1}^{T} \frac{1}{p(x_i \mid x_{<i})}\Big)^{1/T}. $$ {#eq-perplexity}

Brown et al. (1992) define it in these words: the perplexity of a language model with respect to a sample of text is the reciprocal of the geometric average of the probabilities of the predictions in the sample. {#c-brown-def}
Goodman (2001) puts it the same way, with the example that a language model that assigned equal probability to 100 words would have perplexity 100. {#c-goodman-def}
He calls the base-2 logarithm of perplexity entropy, the average number of bits per word an optimal coder would need for the test data, and notes that what is measured is really the cross entropy of the test data given the model. {#c-goodman-entropy}

A test corpus has many sequences: sentences, documents or windows.
If sequence $j$ has total negative log-likelihood $S_j = -\sum_i \ln p(x_{ji} \mid x_{j,<i})$ and is normalised by $n_j$ units, the corpus perplexity pools them:

$$ \mathrm{PPL} = \exp\Big(\frac{\sum_j S_j}{\sum_j n_j}\Big), \qquad \mathrm{PPL}_j = e^{S_j / n_j}. $$ {#eq-corpus}

The unit is usually the model's token, but it can be a word, a character or a byte; the sum of log-probabilities is the same, and only the divisor changes.
Reported in bits, the per-unit cross-entropy is called bits per character or bits per byte.
The Pile turns a per-token loss $\ell$ in nats into bits per UTF-8 byte with $\mathrm{BPB} = (L_T/L_B)\,\ell/\ln 2$, where $L_T$ and $L_B$ are the lengths of the dataset in tokens and in bytes, and measures $L_T/L_B = 0.29335$ GPT-2 tokens per byte across the Pile. {#c-pile-bpb}

$$ \mathrm{BPB} = \frac{L_T}{L_B}\cdot\frac{\ell}{\ln 2}, \qquad \mathrm{PPL}_{\text{byte}} = 2^{\mathrm{BPB}}. $$ {#eq-bpb}

Cross-entropy bounds entropy.
For a stationary process $P$ and a model $M$, Brown et al. define the cross-entropy $H(P, M) = -E_P \log M(X_0 \mid X_{-1}, X_{-2}, \dots)$ and note that it is an upper bound on the entropy: $H(P) \le H(P, M)$ for any model $M$. {#c-brown-upper-bound}
So a test-set perplexity is an estimate of $e^{H(P, M)}$, and it can only approach the true entropy of the text from above.

## Properties

- **Geometric mean.** Over $T$ tokens, eq-perplexity is the geometric mean of the inverse probabilities $1/p(x_i \mid x_{<i})$. {#c-geometric-mean}
- **Independent of the base.** $\mathrm{PPL} = b^{H_b}$ for the cross-entropy $H_b$ in any base $b$, so bits and nats give the same perplexity. {#c-base-invariant}
- **Uniform model.** A model that gives each of $V$ tokens probability $1/V$ has perplexity exactly $V$ on any text. {#c-uniform} GPT-2 has a vocabulary of 50,257 tokens, so a uniform model over it scores 50,257. {#c-gpt2-vocab}
- **Bounds.** $1 \le 1/\max_i p_i \le \mathrm{PPL} \le 1/\min_i p_i$, and $1/\bar{p} \le \mathrm{PPL} \le \overline{1/p}$: it lies between one over the mean probability and the mean inverse probability. {#c-bounds} One token with probability 0 makes it infinite.
- **Corpus weighting.** $\ln \mathrm{PPL}$ of eq-corpus is the mean of the $\ln \mathrm{PPL}_j$ weighted by the unit counts, so the corpus value lies between the smallest and largest per-sequence values; with equal unit counts it is their geometric mean and at most their arithmetic mean. {#c-corpus-weighting}
- **Proper.** If the next symbol has distribution $P$ and the model predicts $M$, the expected log-loss is $H(P) + D_{\mathrm{KL}}(P \,\|\, M)$, which is smallest exactly when $M = P$. {#c-expected-loss} Goodman notes that the true model for any data source will have the lowest possible perplexity for that source. {#c-goodman-true-model} This holds in expectation; on a finite test set, another model can score lower by chance.
- **Units are powers of each other.** Per word, $\mathrm{PPL}_W = \mathrm{PPL}_T^{\,N_T/N_W}$ for $N_T$ tokens and $N_W$ words, and in general the log-perplexities scale with the ratio of the counts. {#c-unit-conversion} Eq-bpb is the case of bytes and base 2. {#c-bpb-formula}
- **Relative cuts are fixed amounts of cross-entropy.** Cutting perplexity by a fraction $r$ removes $-\ln(1 - r)$ nats per unit whatever the starting value. {#c-reduction-scale} Goodman's example: a 10% perplexity reduction, from 100 to 90, is a reduction from 6.64 bits to 6.49 bits, only a 2% entropy reduction. {#c-goodman-ten-percent}

The derivations `perplexity-identities.md` and `units-and-context.md` prove these statements, and their checks test them on random draws.

## Worked example

Take a corpus of two sequences, "the reindeer" and "the lamp", under a model whose tokenizer splits them as `the| rein|deer` and `the| lamp`.
The model gives the scored tokens these probabilities:

| sequence | token probabilities | NLL (bits) | tokens | words | UTF-8 bytes | PPL per token | per word | per byte |
|---|---|---|---|---|---|---|---|---|
| the reindeer | 1/2, 1/4, 1/8 | 6 | 3 | 2 | 12 | $2^{6/3} = 4$ | $2^{6/2} = 8$ | $2^{0.5} \approx 1.414$ |
| the lamp | 1/2, 1/2 | 2 | 2 | 2 | 8 | $2^{2/2} = 2$ | $2^{2/2} = 2$ | $2^{0.25} \approx 1.189$ |
| corpus (eq-corpus) | | 8 | 5 | 4 | 20 | $2^{1.6} \approx 3.031$ | $2^{2} = 4$ | $2^{0.4} \approx 1.320$ |

The same 8 bits give three perplexities: 3.031 per token, 4 per word and 1.320 per byte, and indeed $3.031^{5/4} = 4$.
The cross-entropy is 1.6 bits or $1.6 \ln 2 \approx 1.109$ nats per token, 2 bits per word and 0.4 bits per byte.
Averaging the per-sequence perplexities instead of pooling gives $(4 + 2)/2 = 3$ per token and $(8 + 2)/2 = 5$ per word.
Per word, both sequences have 2 units, so the pooled value 4 is the geometric mean of 8 and 2 and the arithmetic mean 5 is larger.
These are the rows `worked-example`, `worked-example-bits`, `worked-example-bytes`, `worked-example-words` and the two `mean-of-sequences` rows of `vectors/edge.jsonl`, and `test_worked_example` checks every number.

## Reference implementation

There is no reference package for perplexity: it is the exponential of the token cross-entropy that every language-model training loop already computes.
`impl/numpy/perplexity.py` takes, as `contract.yaml` defines them, the natural-log probabilities of the scored tokens of each sequence as `predictions` and each sequence's unit count as `references`.
It sums the log-probabilities with `math.fsum`, divides the pooled negative log-likelihood by the pooled units and exponentiates once, so no product of probabilities is formed.
It returns the corpus perplexity, the cross-entropy in nats or bits, and the perplexity of each sequence.
The PyTorch version does the same in `torch.float64`, and its `token_logprobs` turns logits and targets into the inputs with a log-softmax and a gather; the test `test_token_logprobs_matches_cross_entropy` checks that its mean equals `torch.nn.functional.cross_entropy`.
Both reproduce the 14 rows of `vectors/edge.jsonl` within a relative $10^{-9}$.

A text longer than the model's context has to be cut.
`windows(n, W, s)` returns strided windows of $W$ tokens that start every $s$ tokens and score only the tokens no earlier window scored.
Every token is then scored exactly once, there are $1 + \lceil (n - W)/s \rceil$ windows when $n > W$, and after the first window each scored token sees at least $W - s$ tokens of its window before it. {#c-sliding-window}
With $s = W$ the windows do not overlap and the first token of each window is predicted with no context at all, which raises the perplexity; with $s = 1$ every token gets $W - 1$ tokens of context, at $n - W + 1$ forward passes.

> [!pitfall]
> **Multiplying probabilities.** The product in eq-perplexity underflows in floating point for any realistic text. The row `no-underflow` has 1000 tokens of probability 0.01, whose product $10^{-2000}$ is 0 in float64, while the log-domain value is exactly 100.

> [!pitfall]
> **Averaging perplexities.** The mean of per-sequence perplexities is not the corpus perplexity: it gives a one-token sequence the same weight as a long document. The row `unequal-lengths-mean-of-sequences` gives 8.5 where the corpus value is 2.

> [!pitfall]
> **Counting the wrong tokens.** Padding, the tokens of a prompt that is not part of the test text, and the first token of a sequence that has no beginning-of-sequence token are not predictions and must not enter the sum or the count. The divisor must match the tokens or units that the log-probabilities cover.

> [!pitfall]
> **Comparing across tokenizers.** Per-token perplexities of models with different tokenizers have different units. The row `units-not-tokens` shows 6 subword tokens of 1 bit each: perplexity 2 per token but 4 per word.

## Interpretation

Perplexity reads as a branching factor: a model with perplexity $k$ was, on average, as uncertain as a uniform choice among $k$ tokens.
It also reads as a code length.
Mielke et al. interpret the surprisal $-\log_2 p(s)$ of a sentence as the number of bits needed to represent it under a compression scheme derived from the language model. {#c-mielke-surprisal}
Brown et al. (J92-1002) make the comparison with real compressors on the Brown Corpus: the ASCII code uses 8 bits per character, a simple Huffman code reaches 4.46 bits per character, the UNIX command compress 4.43, and an adaptive Lempel-Ziv scheme of Miller and Wegman 4.20. {#c-brown-compression}
Their word trigram model, built from 583 million words of training text, measures the cross-entropy of the 5.96 million character Brown Corpus at 1.75 bits per character, an upper bound for the entropy of characters in printed English. {#c-brown-175}

Lower perplexity tends to help the systems that use the model.
Brown et al. note that for tasks such as speech recognition, machine translation, handwriting recognition, stenotype transcription and spelling correction, language models with lower cross-entropy lead directly to better performance. {#c-brown-downstream}
But the link is loose.
Goodman reports perplexity reductions of up to 50% over a Katz smoothed trigram on 100,000 words of training data, which is one bit of entropy, falling to 38% on a large set without punctuation, where the word error rate drops by 8.9%. {#c-goodman-wer}
He adds that measures that correlate better with speech recognition performance typically have free variables to tune for a particular recognizer, or are much harder to compute. {#c-goodman-alternatives}

A perplexity is comparable only with the same test text, the same unit and the same preprocessing.
The Brown corpus, 1,014,312 words not used in training, has a perplexity of 244 under the interpolated 3-gram model of Brown et al. (J92-4003). {#c-brown-244}
On the One Billion Word Benchmark, large LSTM models improve the single-model perplexity from 51.3 to 30.0, and an ensemble improves it from 41.0 to 23.7. {#c-jozefowicz}
The pointer sentinel-LSTM of Merity et al. reaches 70.9 on the Penn Treebank. {#c-merity-ptb}
These three numbers are per word on three different corpora with three different vocabularies, so they cannot be ranked against each other.
Within one benchmark, the neural scaling laws page shows the test loss, the logarithm of the perplexity, falling as a power law in model size, data and compute, and the emergent abilities page contrasts that smooth curve with the sharp jumps of some task metrics.
The n-gram language model page works through perplexity for smoothed count models, and the decoder-only transformer page describes the models that are trained to minimise exactly this token cross-entropy.

## Limitations and criticism

**Tokenization changes the number.**
Per-token perplexity depends on how many tokens the tokenizer makes, so it does not compare models with different vocabularies; the byte-pair encoding page shows how the number of merges sets that count.
Gao et al. prefer bits per UTF-8 byte over bits per character or perplexity because of its invariance to different tokenization schemes and the ambiguity of measuring characters in Unicode. {#c-pile-bpb-pref}
Mielke et al. vary the number of BPE merges to move smoothly between a word-level model and a kind of character-level model. {#c-mielke-bpe}
They stress that for fairness the models must be open-vocabulary, predicting the entire string, and cannot cheat by predicting only UNK for some words. {#c-mielke-open-vocab}
A closed-vocabulary benchmark replaces rare words by one unknown token, whose probability is the sum of all the words it stands for: in WikiText-2, WikiText-103 and the Penn Treebank, 2.6%, 0.4% and 4.8% of tokens are replaced by such a token. {#c-wikitext-unk}
A subword model also has many segmentations of one string, and Mielke et al. evaluate only the probability of the canonical segmentation, not the total over all segmentations. {#c-mielke-canonical}
Since the probability of a string is the sum over its segmentations, scoring one segmentation can only overstate the string's perplexity. {#c-canonical-bound}

**Preprocessing changes the number.**
Radford et al. note that results on standard benchmarks are reported as a scaled or exponentiated version of the average negative log probability per canonical prediction unit, usually a character, a byte or a word, and compute the same quantity as the log-probability of a dataset divided by the number of canonical units. {#c-radford-units}
On text with tokenization artifacts, they used invertible de-tokenizers and observed gains of 2.5 to 5 perplexity for GPT-2. {#c-radford-detok}
So a perplexity depends on the exact bytes scored, not only on the model.

**The context changes the number.**
Gao et al. evaluate each document independently, as opposed to concatenating all documents as is common practice for computing perplexity on large corpora. {#c-pile-independent}
A model that sees the previous document, or a longer window, gets more context and a lower perplexity on the same tokens, so the window and stride belong next to the number.

**It does not measure generated text.**
A low perplexity on human text does not make the model's own samples good.
Holtzman et al. compute the perplexity of generated text under the model that produced it: human text scores 12.38, greedy decoding 1.50, beam search with $b = 16$ 1.48, pure sampling 22.73, and nucleus sampling with $p = 0.95$ 13.13. {#c-holtzman-table}
They argue that a good decoding strategy should produce text with a perplexity close to that of the gold text, because text of much lower perplexity tends to have low diversity and get stuck in repetition loops. {#c-holtzman-gold}
The nucleus sampling page uses this comparison to set its threshold.
Meister and Cotterell note that perplexity on held-out data is a monotonic function of the average per-token log-likelihood and, while low perplexity reflects some fit to natural language, it does not give a fine-grained view of which linguistic attributes a model has learned. {#c-meister-view}
They test whether generated text follows the statistical tendencies of human text, and find that neural language models learn only a subset of the tendencies considered and that the fit depends strongly on the architecture and the generation strategy. {#c-meister-findings}

**It averages away the tokens that matter for long contexts.**
Fang et al. find that perplexity overlooks key tokens, which are essential for long-context understanding, by averaging across all tokens. {#c-fang-key-tokens}
In their analysis, perplexity computed only on the answer tokens of long-context tasks strongly correlates with LongEval accuracy, whereas perplexity on non-answer tokens shows little to no correlation. {#c-fang-answer}
To find key tokens in natural text, they compute the long-short difference $\mathrm{LSD}_\theta(x_i) = \log P_\theta(x_i \mid l_i) - \log P_\theta(x_i \mid s_i)$, where $l_i$ is the full long context and $s_i$ a truncated short context of length $K$, e.g. 64. {#c-fang-lsd}
LongPPL, the perplexity on these key tokens, reaches a Pearson correlation of $-0.96$ with LongBench scores on GovReport, while standard perplexity shows no correlation. {#c-fang-corr}

## Variants

The options of `contract.yaml` change the number, and `vectors/variants.yaml` lists the rows that pin each value.

| option | values | effect | used by |
|---|---|---|---|
| `aggregate` | `corpus` (default), `mean-of-sequences` | pool the negative log-likelihood (eq-corpus), or average the per-sequence perplexities | `corpus` in Brown et al. (1992), Goodman (2001) and Gao et al. (2020); the mean in evaluation scripts that report per-document values |
| `base` | `e` (default), `2` | the cross-entropy in nats or bits per unit; the perplexity does not change | bits per character in Brown et al. (J92-1002); bits per byte in Gao et al. (2020) |

The unit is not an option: it is the count passed as `references`.
Word perplexity comes from speech recognition: Brown et al. (J92-1002) describe it as a commonly used measure of the difficulty of a speech recognition task, and they cite Bahl et al. (1977) for it. {#c-brown-word-perplexity}
Character perplexity follows the same recipe per character, and Brown et al. note that their cross-entropy is just the base-two logarithm of the character perplexity of a sample of text with respect to a language model. {#c-brown-char-perplexity}
Bits per byte, eq-bpb, removes the dependence on the tokenizer and on what counts as a character.
Several variants change which tokens are scored rather than how they are counted.
LongPPL scores only the key tokens that a long context helps to predict.
A conditional perplexity scores only a continuation given a prompt, and per-language comparisons score the same content: Cotterell et al. (2018), as Mielke et al. describe, evaluated the model for each language by its total surprisal on a complete multitext corpus, the same sentences in every language. {#c-mielke-multitext}

## Implementations

## History

The measure came from speech recognition, where the language model narrows down which word sequences the acoustic model has to choose between.
Brown et al. (J92-4003) used perplexity to compare their n-gram and class-based models.
In the same year, Brown et al. (J92-1002) suggested the Brown Corpus of printed English as a standard against which to measure progress in language modeling. {#c-brown-standard}
They framed their 1.75 bits per character as a bound: previous estimates of the entropy of English since Shannon's 1951 paper had used human subjects and samples of at most a few hundred letters. {#c-brown-previous}
By 2001, Goodman called perplexity the most commonly used method for measuring language model performance. {#c-goodman-common}
Merity et al. (2016) introduced WikiText-2 and WikiText-103 so that models could be tested on longer contexts and more realistic vocabularies. {#c-wikitext}
WikiText-103 has 103,227,021 training tokens and a vocabulary of 267,735, where the Penn Treebank has 929,590 and 10,000. {#c-wikitext-size}
With open-vocabulary byte-level models, Radford et al. (2019) could score one model on many benchmarks zero-shot, and Gao et al. (2020) moved the comparison to bits per byte.
Fang et al. (2025) then found that the plain average misses long-context ability, and proposed LongPPL and the matching LongCE loss.

## References
