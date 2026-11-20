---
tag: NEW-masked-language-modelling
kind: method
slug: masked-language-modelling
name: Masked language modelling
aliases: [masked language model, masked LM, MLM, Cloze objective, BERT-style denoising objective, masked language modeling]
area: language-models/pretraining-objectives
level: core
summary: >-
  A pre-training objective that hides a random subset of the tokens of a text and trains a network to predict them from the tokens on both sides, which gives bidirectional representations.
purpose: learning contextual representations of every token from unlabelled text, conditioned on the words to the left and to the right, for fine-tuning on downstream tasks
uses: [cross-entropy@m2, stochastic-gradient-descent@m2, conditional-probability@m2, backpropagation@al, softmax-regression@al, logistic-regression@al]
related_to: [bert, transformer, neural-probabilistic-language-model, n-gram-language-model, autoregressive-model, autoencoder, discrete-diffusion-model, generative-adversarial-network, byte-pair-encoding, dropout, mutual-information@m2, geometric-distribution@m2, gibbs-inequality@m2, conditional-entropy@m2]
hyperparameters:
  - {name: masking rate, meaning: "the share of the tokens chosen for prediction", typical: "15% in BERT, RoBERTa, SpanBERT and T5; 25% for ELECTRA-Large; 40% for BERT-large size models in Wettig et al. (2022)"}
  - {name: replacement split, meaning: "what a chosen token becomes in the input: [MASK], a random token, or itself", typical: "80/10/10 (Devlin et al., 2019); [MASK] only (Wettig et al., 2022); all [MASK] for the ELECTRA generator"}
  - {name: masking unit, meaning: "what is chosen at once: a subword, a whole word, or a span", typical: "subword tokens in BERT; spans of complete words with geometric lengths, p = 0.2, at most 10, mean 3.8, in SpanBERT; spans of mean length 3 in T5"}
  - {name: static or dynamic, meaning: "whether a sequence keeps the corruption drawn once in preprocessing or gets a fresh one each time it is fed", typical: "dynamic since RoBERTa"}
  - {name: target, meaning: "what the network predicts: the original token at each chosen position, the whole input, or the dropped-out spans after sentinel tokens", typical: "the chosen positions in BERT; the spans after sentinels in T5"}
  - {name: discriminator weight λ, meaning: "in ELECTRA, the weight of the replaced-token-detection loss against the generator's masked LM loss", typical: "50, with a generator 1/4 to 1/2 the size of the discriminator"}
cost:
  train: "one forward and backward pass of the encoder over all n positions per sequence, as for a left-to-right model, but only about 0.15n loss terms and output softmaxes; ELECTRA's discriminator gets n loss terms from the same pass"
  inference: "one encoder pass gives a contextual vector for every token; scoring a sentence with the model's own conditionals needs one pass per masked position"
history:
  year: 2018
  people: [Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova, Wilson L. Taylor]
  note: "Taylor (1953) introduced the cloze test of filling in deleted words; Devlin et al. made masked prediction the pre-training objective of BERT in 2018 (NAACL 2019); RoBERTa (2019) made the masking dynamic; SpanBERT (2019) masked spans; XLNet (2019) replaced it by a permutation objective; T5 (2019, JMLR 2020) used span corruption with sentinels; ELECTRA (ICLR 2020) replaced masking by replaced token detection; Wettig et al. (2022) revisited the 15% rate."
prereqs: [transformer, neural-probabilistic-language-model, cross-entropy@m2, softmax-regression@al]
same_as:
  arxiv: ["1810.04805", "1907.11692", "1907.10529", "1906.08237", "1910.10683", "2003.10555", "2202.08005"]
sources:
  - {paper: "Devlin, Chang, Lee and Toutanova (2019)", title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", locator: "NAACL 2019; arXiv 1810.04805v2, Section 3.1, Table 5, Appendices A.1, C.1 and C.2 with Table 8", note: "the objective, the 80/10/10 rule and its ablations"}
  - {paper: "Liu et al. (2019)", title: "RoBERTa: A Robustly Optimized BERT Pretraining Approach", locator: "arXiv 1907.11692v1, Section 4.1 and Table 1", note: "static against dynamic masking"}
  - {paper: "Joshi et al. (2019)", title: "SpanBERT: Improving Pre-training by Representing and Predicting Spans", locator: "arXiv 1907.10529v3, Sections 3.1, 3.2 and 6.1, Table 6", note: "geometric span masking and the span boundary objective"}
  - {paper: "Yang et al. (2019)", title: "XLNet: Generalized Autoregressive Pretraining for Language Understanding", locator: "NeurIPS 2019; arXiv 1906.08237v2, Sections 2.1 and 2.6", note: "the independence assumption and the pretrain-finetune discrepancy"}
  - {paper: "Raffel et al. (2020)", title: "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer", locator: "JMLR 21; arXiv 1910.10683v4, Sections 3.1.4 and 3.3, Tables 4 to 7", note: "span corruption with sentinels, corruption rates and span lengths"}
  - {paper: "Clark, Luong, Le and Manning (2020)", title: "ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators", locator: "ICLR 2020; arXiv 2003.10555v1, Sections 1, 2 and 3.3, Table 5, Appendix A", note: "replaced token detection and where its gains come from"}
  - {paper: "Wettig, Gao, Zhong and Chen (2022)", title: "Should You Mask 15% in Masked Language Modeling?", locator: "arXiv 2202.08005v3, Sections 1, 6 and 7, Tables 1, 3 and 4", note: "masking rates, corruption against prediction, and the 80/10/10 rule"}
---

Masked language modelling is a pre-training objective: hide a random subset of the tokens of a text, and train a network to predict the hidden tokens from the tokens on both sides.
Unlike a left-to-right language model, the network sees the whole context, so the vector it builds for each token reflects the words after it as well as before.
Devlin et al. made it the pre-training objective of [[bert|BERT]] in 2018, choosing 15% of the tokens and replacing most of them by a [MASK] token.
Later models changed what is hidden and how: RoBERTa drew a fresh mask each time a sequence was fed, SpanBERT and [[t5|T5]] hid spans, and ELECTRA trained a discriminator to spot replaced tokens.

## In plain words

A [[neural-probabilistic-language-model|left-to-right language model]] predicts each word from the words before it, so the vector it builds for a word knows nothing about what follows.
Masked language modelling hides some of the words of a sentence and asks the network to fill them in from everything that is left, on both sides, the way a reader fills in the blanks of a cloze test.
To do that well the network has to build, for every word, a vector that reflects the whole sentence.
After this pre-training on plain text, those vectors are the starting point for tasks such as question answering and classification.
The objective was introduced with [[bert|BERT]] in 2018, and most encoder models since use it or a close relative.

## Setting

Devlin et al. note that standard conditional language models can only be trained left-to-right or right-to-left, since bidirectional conditioning would allow each word to indirectly "see itself", and the model could trivially predict the target word in a multi-layered context. {#c-bert-see-itself}
The way out is to remove the word from the input.
In order to train a deep bidirectional representation, they simply mask some percentage of the input tokens at random, and then predict those masked tokens; they refer to this procedure as a "masked LM" (MLM), although it is often referred to as a Cloze task in the literature (Taylor, 1953). {#c-bert-mlm}
The output is not a probability of the text: a masked model gives conditionals of hidden tokens given visible ones, which do not combine into a joint distribution the way the conditionals of an [[autoregressive-model|autoregressive model]] do.
The network is usually a [[transformer|Transformer]] encoder; the objective does not depend on it.

## The method

**Choosing and corrupting.** Write $x = (x_1, \dots, x_n)$ for a sequence of tokens and $M \subset \{1, \dots, n\}$ for the chosen positions.
Devlin et al. mask 15% of all WordPiece tokens in each sequence at random, and, in contrast to denoising auto-encoders, they only predict the masked words rather than reconstructing the entire input. {#c-bert-rate}
A downside is a mismatch between pre-training and fine-tuning, since the [MASK] token does not appear during fine-tuning. {#c-bert-mismatch}
To lessen it, the training data generator chooses 15% of the token positions at random for prediction, and replaces the chosen token with the [MASK] token 80% of the time, a random token 10% of the time and the unchanged token 10% of the time; then $T_i$ is used to predict the original token with cross entropy loss. {#c-bert-801010}
Call the corrupted sequence $\tilde x$.

**The loss.** The encoder maps $\tilde x$ to vectors $h_1, \dots, h_n$, and a softmax layer over the vocabulary gives $q_\theta(\cdot \mid \tilde x, i)$ at each position. The loss is the [[m2:cross-entropy|cross-entropy]] of the original tokens at the chosen positions only:

$$\mathcal{L}_{\text{MLM}}(\theta) = -\,\mathbb{E}_{x}\, \mathbb{E}_{M, \tilde x} \frac{1}{\lvert M \rvert} \sum_{i \in M} \log q_\theta(x_i \mid \tilde x, i).$$ {#eq-mlm}

The positions outside $M$ contribute nothing, although the network reads them.
The gradient with respect to the logits at a chosen position is $q_\theta(\cdot) - e_{x_i}$, divided by $\lvert M \rvert$, the gradient of [[al:softmax-regression|softmax regression]]; it is zero at every other position, and [[al:backpropagation|backpropagation]] carries it into the encoder.

**Why the random and unchanged tokens.** Devlin et al. explain the 10% of unchanged tokens as a way to bias the representation towards the actual observed word, and the mix as a way to keep the encoder from knowing which words it will be asked to predict or which have been replaced by random words, so it is forced to keep a distributional contextual representation of every input token. {#c-bert-purpose}
They add that random replacement only occurs for 1.5% of all tokens (10% of 15%), which does not seem to harm the model's language understanding capability. {#c-bert-random-share}

**Static or dynamic.** The original BERT implementation performed masking once during data preprocessing, resulting in a single static mask; the training data was duplicated 10 times so that each sequence is masked in 10 different ways over the 40 epochs of training, and each training sequence was seen with the same mask four times. {#c-roberta-static}
Liu et al. compare this with dynamic masking, where the masking pattern is generated every time a sequence is fed to the model. {#c-roberta-dynamic-def}

**Spans.** Joshi et al. choose the subset by iteratively sampling spans of text until the masking budget (e.g. 15% of the tokens) has been spent: at each iteration they sample a span length (number of words) from a geometric distribution, randomly (uniformly) select the starting point, always sample a sequence of complete words, and set p = 0.2 and clip the length at 10, which yields a mean span length of 3.8. {#c-spanbert-masking}
As in BERT they mask 15% of the tokens in total, replacing 80% of the masked tokens with [MASK], 10% with random tokens and 10% with the original tokens, but they perform this replacement at the span level and not for each token individually. {#c-spanbert-replace}
Their span boundary objective predicts each token of a masked span using only the representations of the observed tokens at the boundaries. {#c-spanbert-sbo}

**Sentinels instead of one mask per token.** For the encoder-decoder T5, Raffel et al. randomly sample and drop out 15% of tokens in the input sequence; all consecutive spans of dropped-out tokens are replaced by a single sentinel token, and the target is the dropped-out spans delimited by the same sentinel tokens, plus a final sentinel token to mark the end of the target sequence. {#c-t5-objective}
They chose to mask consecutive spans of tokens and only predict dropped-out tokens to reduce the computational cost of pre-training. {#c-t5-cost}

## Derivation

Four derivations are written out with checks in `derivations/`.

`corruption-fractions.md` counts what the 80/10/10 rule does to the input.
With masking rate $r$ and $V'$ ordinary tokens, a share $0.8r$ of the input tokens are [MASK], $0.1r(1 - 1/V')$ are other words and $0.1r(1 + 1/V')$ are chosen but show their own word; at $r = 0.15$ just under 13.5% of the input differs from the text, and a sequence of $n$ tokens gives $0.15n$ masked LM loss terms against $n$ discriminator terms for ELECTRA, 6.67 times as many (76.8 against 512 at $n = 512$). {#c-fractions}

`span-length.md` checks SpanBERT's mean.
A [[m2:geometric-distribution|geometric distribution]] with $p = 0.2$ truncated at 10 and renormalised has mean $\bigl(1 - 11 q^{10} + 10 q^{11}\bigr) / \bigl(p(1 - q^{10})\bigr) = 3.797$ with $q = 0.8$, which matches the 3.8 of Joshi et al.; capping the lengths at 10 instead would give $(1 - q^{10})/p = 4.463$, and the untruncated mean is 5. {#c-span-mean}

`span-corruption-lengths.md` counts T5's sequence lengths.
With $k$ noise tokens in $s$ runs, span corruption gives an input of $n - k + s$ tokens and a target of $k + s + 1$; for 500 tokens with 15% noise in 25 spans that is 450 and 101, while i.i.d. noise at 15% has $r + (n-1)r(1-r) = 63.77$ runs in expectation, a mean run of 1.18 tokens, an input of 488.77 tokens and a target of 139.77. {#c-t5-lengths}

`independence-gap.md` measures what the masked objective leaves out.
When two positions $a$ and $b$ are masked in the same example, the lowest expected masked LM loss for the pair is $H(A \mid C) + H(B \mid C)$, which exceeds the joint entropy $H(A, B \mid C)$ by the [[m2:mutual-information|mutual information]] $I(A; B \mid C) \ge 0$; in the "New York" example below the gap is $\log 2$ nats. {#c-independence-gap}
The proof is [[m2:gibbs-inequality|Gibbs' inequality]] for each position and the chain rule for [[m2:conditional-entropy|conditional entropy]].

## Worked example

Devlin et al. illustrate the masking procedure on the unlabeled sentence "my dog is hairy" when the random masking procedure chose the 4-th token, hairy: 80% of the time the word is replaced with the [MASK] token, my dog is hairy → my dog is [MASK]; 10% of the time with a random word, my dog is hairy → my dog is apple; and 10% of the time the word is kept unchanged, my dog is hairy → my dog is hairy. {#c-bert-example}
In every case the target at position 4 is "hairy", and positions 1 to 3 give no loss.

Raffel et al. illustrate span corruption on "Thank you for inviting me to your party last week." with "for", "inviting" and "last" chosen: since "for" and "inviting" occur consecutively they are replaced by a single sentinel, the input is "Thank you <X> me to your party <Y> week." and the target is "<X> for inviting <Y> last <Z>". {#c-t5-example}
The input has 9 tokens instead of 11 and the target has 6, matching $n - k + s = 11 - 3 + 2$ and $k + s + 1 = 3 + 2 + 1$.
The tests of both implementations reproduce both examples.

The tests also train a small bidirectional model on a synthetic language.
The text has 48 words in 6 classes of 8; the class follows a first-order Markov chain and the word is uniform within its class, so the source's own probability of a word given both its neighbours, and given only the words before it, is known exactly.
The model is the [[neural-probabilistic-language-model|neural probabilistic language model]] turned around: the embeddings of the five tokens at offsets −2 to 2 of the corrupted input, the position itself included, go through one tanh layer of 32 units to a softmax.
With offsets −2 and −1 only it is a left-to-right model.
Each run trains on 400 sequences of 32 words for 40 epochs and is scored on 200 test sequences by the cross-entropy of each word with only that word masked:

| Run | Test cross-entropy (nats) |
|---|---|
| Source, given both neighbours | 2.523 |
| 80/10/10 at 15%, dynamic masking | 2.628 |
| [MASK] only at 15%, dynamic | 2.602 |
| 80/10/10 at 40%, dynamic | 2.647 |
| 80/10/10 at 15%, one static mask per sequence | 3.227 |
| Source, given only the words before | 2.966 |
| Left-to-right model | 3.007 |
| Uniform over the 48 words | 3.871 |

Every masked model with dynamic masking does better than the source itself can do from the left context alone, because the right neighbour carries information about the class.
The static mask lets the model fit the 1,920 masked positions of each epoch again and again: its training loss falls to 1.95 while its test loss rises to 3.23.
On this small model and corpus, [MASK]-only training scores a little better than 80/10/10 on a test that also uses [MASK]; the ablations under Evidence measure the same choice on real models and downstream tasks.

## Hyperparameters

The masking rate trades corruption against signal.
Raffel et al. compare corruption rates of 10%, 15%, 25% and 50% and find that the corruption rate had a limited effect on the model's performance, except that 50% results in a significant degradation of performance on GLUE and SQuAD. {#c-t5-rate}
Wettig et al. find that masking 40% outperforms 15% for BERT-large size models on GLUE and SQuAD, and that an extremely high masking rate of 80% can still preserve 95% fine-tuning performance. {#c-wettig-40}
They also find that uniform masking requires a higher masking rate compared to sophisticated masking strategies such as span or PMI masking. {#c-wettig-strategy}
For ELECTRA-Large, Clark et al. used a higher mask percent (25 instead of 15) because the generator was achieving high accuracy with 15% masking, resulting in very few replaced tokens. {#c-electra-large-mask}

The span length matters little.
Raffel et al. compare average span lengths of 2, 3, 5 and 10 at a corruption rate of 15% and find a limited difference, though the version with an average span length of 10 slightly underperforms the other values in some cases, and an average span length of 3 slightly (but significantly) outperforms the i.i.d. objective on most non-translation benchmarks. {#c-t5-span}

ELECTRA has two more settings.
Clark et al. set λ, the weight for the discriminator objective in the loss, to 50. {#c-electra-lambda}
They find that models work best with generators 1/4-1/2 the size of the discriminator. {#c-electra-gen-size}

## Cost

The encoder reads every position, so a training step costs about as much as a step of a left-to-right model of the same size.
What differs is the signal per step: the loss has about $0.15n$ terms, and the output softmax is needed only at those positions.
Clark et al. argue that masked language modelling incurs a substantial compute cost because the network only learns from 15% of the tokens per example. {#c-electra-cost}
BERT's own appendix addresses the same worry: the MLM model does converge slightly slower than the LTR model, but in terms of absolute accuracy the MLM model begins to outperform the LTR model almost immediately. {#c-bert-converge}

At inference there is no sampling loop.
One pass over a sentence gives a vector for every token, which is what fine-tuning uses.
Scoring a sentence with a masked model's own conditionals takes one pass per position masked in turn; the window model in the implementation does it in one pass only because each prediction sees a fixed window.

## Evidence

**Bidirectional context.** Devlin et al. ablate the objective with BERT-Base on the dev sets of MNLI-m, QNLI, MRPC, SST-2 and SQuAD: BERT-Base scores 84.4, 88.4, 86.7, 92.7 and 88.5; without next sentence prediction 83.9, 84.9, 86.5, 92.6 and 87.9; trained as a left-to-right model without next sentence prediction ("LTR & No NSP") 82.1, 84.3, 77.5, 92.1 and 77.8; and with a BiLSTM added on top of that 82.1, 84.1, 75.7, 91.6 and 84.9. {#c-bert-table5}

**The replacement split.** On the MNLI dev set and for NER fine-tuned and feature-based, Devlin et al. report 84.2, 95.4 and 94.9 for 80% [MASK], 10% same and 10% random; 84.3, 94.9 and 94.0 for [MASK] only; 84.1, 95.2 and 94.6 for 80% [MASK] and 20% random; 84.4, 95.2 and 94.7 for 80% [MASK] and 20% same; 83.7, 94.8 and 94.6 for 20% same and 80% random; and 83.6, 94.9 and 94.6 for random only. {#c-bert-table8}
They read the table as saying that fine-tuning is surprisingly robust to different masking strategies, that using only the [MASK] strategy was problematic for the feature-based approach to NER, and that using only the random strategy performs much worse. {#c-bert-table8-reading}
With 40% masking, Wettig et al. find that same token predictions and random token corruptions deteriorate performance on most downstream tasks and that the 80-10-10 rule performs worse than simply using all [MASK], with the exception of SST-2; they suggest to use only [MASK] for MLM pre-training. {#c-wettig-801010}

**Dynamic masking.** On SQuAD 2.0, MNLI-m and SST-2, Liu et al. report 76.3, 84.3 and 92.8 for the published BERT-Base, 78.3, 84.3 and 92.5 for their reimplementation with static masking, and 78.7, 84.0 and 92.9 with dynamic masking; they find dynamic masking comparable or slightly better than static masking. {#c-roberta-table1}

**Masking rate.** With BERT-large models and an efficient pre-training recipe, Wettig et al. report a validation perplexity of 17.7 and dev scores of 84.2 (MNLI), 90.9 (QNLI) and 88.0 (SQuAD) at a 15% masking rate, 69.4, 84.5, 91.6 and 89.8 at 40%, and 1141.4, 80.8, 87.9 and 86.2 at 80%. {#c-wettig-table1}
They separate the corruption rate, how much of the input is changed, from the prediction rate, how many tokens are predicted, and find that more prediction helps and more corruption hurts: with 40% predicted, MNLI rises from 84.5 at 40% corruption to 85.7 at 20%, 86.3 at 10% and 86.9 at 5%. {#c-wettig-table3}

**What to mask.** With bi-sequence training and next sentence prediction, Joshi et al. report dev scores on SQuAD 2.0, NewsQA, TriviaQA, coreference, MNLI-m, QNLI and the GLUE average of 83.8, 72.0, 76.3, 77.7, 86.7, 92.5 and 83.2 for subword tokens as in BERT; 84.3, 72.8, 77.1, 76.6, 86.3, 92.8 and 82.9 for whole words; and 85.4, 73.0, 78.8, 76.4, 87.0, 93.3 and 83.4 for geometric spans. {#c-spanbert-table6}
With the exception of coreference resolution, masking random spans is preferable to the other strategies they try, which include named entities and noun phrases. {#c-spanbert-reading}

**Objectives compared in one encoder-decoder.** On GLUE, CNNDM, SQuAD and SuperGLUE, Raffel et al. report 80.69, 18.94, 77.99 and 65.27 for prefix language modeling, 82.96, 19.17, 80.65 and 69.85 for the BERT-style objective and 73.17, 18.59, 67.61 and 58.47 for deshuffling, and find that the BERT-style objective performs best, though the prefix language modeling objective attains similar performance on the translation tasks. {#c-t5-table4}
Among variants of the BERT-style objective the MASS-style objective scores 82.32, 19.16, 80.10 and 69.28, replacing corrupted spans 83.28, 19.24, 80.88 and 71.36, and dropping corrupted tokens 84.44, 19.31, 80.52 and 68.67; in their setting all of these variants perform similarly. {#c-t5-table5}

**Learning from all tokens.** Clark et al. report GLUE scores of 85.0 for ELECTRA, 84.3 for All-Tokens MLM, 82.4 for Replace MLM, 82.4 for ELECTRA 15% and 82.2 for BERT in a compute-efficiency comparison. {#c-electra-table5}
They conclude that a large amount of ELECTRA's improvement can be attributed to learning from all tokens and a smaller amount can be attributed to alleviating the pre-train fine-tune mismatch. {#c-electra-reading}

## Limitations and failure modes

**The [MASK] token.** The input during pre-training contains a symbol that never occurs later; the 80/10/10 rule, dynamic masking and replaced token detection are all answers to that, and the evidence above disagrees on how much it matters.
Yang et al. note that replacing [MASK] with original tokens does not solve the problem because original tokens can be only used with a small probability, otherwise the objective would be trivial to optimize. {#c-xlnet-original}

**Independence of the masked tokens.** Yang et al. note that BERT factorizes the joint conditional probability of the masked tokens based on an independence assumption that all masked tokens are separately reconstructed. {#c-xlnet-independence}
Their example: with the targets [New, York] in "New York is a city", BERT's objective is log p(New | is a city) + log p(York | is a city), and the dependency between the pair (New, York) is omitted. {#c-xlnet-newyork}
The independence-gap derivation puts a size on it: if the city could equally be Los Angeles, a masked model with the lowest possible loss still loses $\log 2$ nats on the pair and gives probability 1/4 to "New Angeles".

**Little signal per example.** Only the chosen positions are trained, which is the cost argument above.

**No text generation.** The model gives conditionals of hidden tokens given the rest, not a distribution over whole texts, so it cannot sample a passage left to right; masked [[discrete-diffusion-model|discrete diffusion]] models are one way to generate with a masked objective.

**Static masks overfit.** A corpus masked once is a fixed set of fill-in-the-blank questions; in the worked example that alone raises the test loss from 2.63 to 3.23 nats.

## Variants

- **Whole-word and span masking.** SpanBERT's geometric spans, with or without the span boundary objective; T5's span corruption with sentinels, which also shortens the sequences the network reads and writes.
- **Replaced token detection.** In ELECTRA a small masked LM, the generator, fills in the chosen positions with samples, and a discriminator labels every token as original or replaced, a [[al:logistic-regression|logistic regression]] per position on top of the encoder. If the generator happens to generate the correct token, that token is considered "real" instead of "fake", and the generator is trained with maximum likelihood rather than adversarially, unlike a [[generative-adversarial-network|GAN]]. {#c-electra-real}
- **Permutation language modelling.** XLNet trains an autoregressive model over random factorization orders, which naturally avoids the independence assumption and the pretrain-finetune discrepancy. {#c-xlnet-avoids}
- **Higher masking rates and [MASK] only**, as in Wettig et al.
- **Masked diffusion.** Choosing the masking rate at random for each sequence and weighting the loss turns the objective into a [[discrete-diffusion-model|discrete diffusion model]] over text.

The objective is close to a denoising [[autoencoder|autoencoder]] whose noise is masking, with the loss taken at the corrupted positions only, and to word [[dropout|dropout]] on the input.

## When to use it

Use it to pre-train an encoder whose token vectors will be fine-tuned for classification, tagging, retrieval or extractive question answering, where the whole input is available at once.
For text generation, a left-to-right objective or an encoder-decoder with span corruption fits better; for the same compute, ELECTRA-style replaced token detection gave better GLUE scores than masking in Clark et al.'s comparison.
With little data, mask dynamically.

## Pitfalls

- Selecting from [PAD], [CLS] or [SEP] positions, or counting them in the 15%; only ordinary tokens are chosen.
- Taking the loss over all positions instead of the chosen ones: the network then learns to copy its input at the 85% unchanged positions.
- Drawing the random replacement tokens from the special tokens, or never letting them equal the original; the derivation counts the 1/V′ chance that they do.
- Masking subword pieces independently when the tokenizer is [[byte-pair-encoding|byte-pair encoding]] or WordPiece: the other pieces of the word give the answer away, which whole-word and span masking avoid.
- Reading a masked model's per-token losses as a likelihood or perplexity comparable with an autoregressive model's; they are conditionals given both sides, as the table in the worked example shows, and an [[n-gram-language-model|n-gram]] or left-to-right perplexity is a different quantity.

## History

The cloze procedure of Taylor (1953), filling in deleted words, is the idea behind the objective, and Devlin et al. made it the pre-training task of BERT in October 2018; the paper appeared at NAACL 2019.
In 2019 RoBERTa replaced the static masks with dynamic ones, SpanBERT masked spans, XLNet proposed a permutation objective to avoid the independence assumption, and T5 compared many corruption schemes and settled on span corruption with sentinels.
ELECTRA (ICLR 2020) trained a discriminator to spot replaced tokens, and Wettig et al. (2022) questioned the 15% rate and the 80/10/10 rule.
