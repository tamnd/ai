---
tag: NEW-training-data-memorization
kind: finding
slug: training-data-memorization
name: Training data memorization
aliases: [verbatim memorization, training data extraction, regurgitation, eidetic memorization, extractable memorization]
area: language-models/knowledge-and-memorization-in-lms
level: core
summary: >-
  Language models reproduce long passages of their training data word for word, more so for larger models, duplicated text and longer prompts, even when they show no overfitting.
related_to: [data-contamination, neural-scaling-laws, double-descent, red-teaming, masked-language-modelling, emergent-abilities, nucleus-sampling]
history:
  year: 2020
  note: "Carlini et al. posted the extraction attack on GPT-2 as arXiv 2012.07805 in December 2020; Carlini et al. (ICLR 2023), Lee et al. (ACL 2022), Kandpal et al. (ICML 2022), Tirumala et al. (NeurIPS 2022), Biderman et al. (2023) and Nasr et al. (2023) measured how it scales, how duplication drives it and how much a deployed chat model leaks."
prereqs: [decoder-only-transformer, perplexity]
same_as:
  arxiv: ["2012.07805", "2202.07646", "2107.06499", "2202.06539", "2205.10770", "2304.11158", "2311.17035", "1906.05271"]
sources:
  - {paper: "Carlini, Tramèr, Wallace, Jagielski, Herbert-Voss, Lee, Roberts, Brown, Song, Erlingsson, Oprea and Raffel (2021)", title: "Extracting Training Data from Large Language Models", locator: "arXiv 2012.07805v2, pp. 1 to 13", note: "the definitions, the attack on GPT-2, the 604 memorized examples and the URL experiment"}
  - {paper: "Carlini, Ippolito, Jagielski, Lee, Tramèr and Zhang (2023)", title: "Quantifying Memorization Across Neural Language Models", locator: "ICLR 2023, arXiv 2202.07646v3, pp. 1 to 9", note: "the three log-linear relationships, the T5 and OPT results and the effect of deduplication"}
  - {paper: "Lee, Ippolito, Nystrom, Zhang, Eck, Callison-Burch and Carlini (2022)", title: "Deduplicating Training Data Makes Language Models Better", locator: "ACL 2022, pp. 8424 to 8445, PDF pp. 1, 2 and 7", note: "duplicates in C4 and the tenfold drop in emitted training text"}
  - {paper: "Kandpal, Wallace and Raffel (2022)", title: "Deduplicating Training Data Mitigates Privacy Risks in Language Models", locator: "ICML 2022, arXiv 2202.06539v3, pp. 1 and 2", note: "the superlinear effect of duplication and membership inference on singletons"}
  - {paper: "Tirumala, Markosyan, Zettlemoyer and Aghajanyan (2022)", title: "Memorization Without Overfitting: Analyzing the Training Dynamics of Large Language Models", locator: "NeurIPS 2022, arXiv 2205.10770v2, pp. 1 and 3", note: "exact memorization during training, and its speed in larger models"}
  - {paper: "Biderman, Prashanth, Sutawika, Schoelkopf, Anthony, Purohit and Raff (2023)", title: "Emergent and Predictable Memorization in Large Language Models", locator: "arXiv 2304.11158v2, pp. 3 to 7", note: "predicting which sequences a large Pythia model memorizes from smaller models and checkpoints"}
  - {paper: "Nasr, Carlini, Hayase, Jagielski, Cooper, Ippolito, Choquette-Choo, Wallace, Tramèr and Lee (2023)", title: "Scalable Extraction of Training Data from (Production) Language Models", locator: "arXiv 2311.17035v1, pp. 1 and 9", note: "extraction from open models and the divergence attack on ChatGPT"}
  - {paper: "Feldman (2019)", title: "Does Learning Require Memorization? A Short Tale about a Long Tail", locator: "arXiv 1906.05271v4, pp. 1, 4 and 5", note: "the long-tail argument that fitting rare examples is needed for low error"}
claim: >-
  Large language models trained on web text emit verbatim sequences from their training data, including sequences that appear in only one training document, and the fraction they emit grows log-linearly with model size, with the number of copies of a sequence in the training data and with the length of the prompt.
first_reported: {work: carlini-2021-extracting-training-data, note: "extracted 604 verbatim training examples from GPT-2, among them personal contact details, code and UUIDs, and found that larger GPT-2 models memorized more"}
replications:
  - {work: carlini-2023-quantifying-memorization, note: "on GPT-Neo and GPT-J trained on The Pile, memorization grows log-linearly with size, duplication and prompt length; at least 1% of The Pile is extractable from GPT-J 6B"}
  - {work: lee-2022-deduplicating-training-data, note: "a model trained on the original C4 emits about ten times more memorized tokens than one trained on deduplicated C4"}
  - {work: nasr-2023-scalable-extraction-training-data, note: "gigabytes of training data extracted from Pythia, GPT-Neo, LLaMA and Falcon, and over 10,000 examples from ChatGPT for $200 of queries"}
scale_range:
  - {range: "GPT-2, 117M to 1.5B parameters", work: carlini-2021-extracting-training-data}
  - {range: "GPT-Neo 125M to GPT-J 6B, T5 77M to 11B, OPT 125M to 175B", work: carlini-2023-quantifying-memorization}
  - {range: "Pythia 70M to 12B", work: biderman-2023-emergent-predictable-memorization}
  - {range: "ChatGPT (gpt-3.5-turbo)", work: nasr-2023-scalable-extraction-training-data}
failures:
  - {work: carlini-2023-quantifying-memorization, note: "the OPT family shows the same trends with an effect orders of magnitude smaller, and T5 masked models memorize an order of magnitude less, with no monotone trend in duplication"}
  - {work: kandpal-2022-deduplicating-mitigates-privacy, note: "membership inference that identifies highly duplicated sequences is at chance for sequences that appear once"}
explanations:
  - {name: "duplication", summary: "Text that occurs many times in the training data is regenerated far more often than its share, so a few heavily repeated passages dominate what leaks.", sources: [kandpal-2022-deduplicating-mitigates-privacy, lee-2022-deduplicating-training-data, carlini-2023-quantifying-memorization]}
  - {name: "capacity", summary: "Larger models fit individual training examples faster and keep more of them, before any overfitting shows in the loss.", sources: [tirumala-2022-memorization-without-overfitting, carlini-2023-quantifying-memorization]}
  - {name: "the long tail", summary: "When much of the data comes from rare subpopulations, fitting examples seen only once is needed for low error, so a good learner must memorize them.", sources: [feldman-2019-does-learning-require-memorization]}
  - {name: "measurement", summary: "How much memorization is found depends on the prompt, the decoding rule and the attack, so every measured amount is a lower bound.", sources: [carlini-2023-quantifying-memorization, nasr-2023-scalable-extraction-training-data]}
---

Training data memorization is the reproduction, by a trained language model, of passages of its training data word for word.
Carlini et al. (2021) demonstrate a training data extraction attack on GPT-2, a language model trained on scrapes of the public Internet, that recovers individual training examples by querying the language model and extracts hundreds of verbatim text sequences from the model's training data. {#c-ca21-attack}
The extracted examples include (public) personally identifiable information (names, phone numbers, and email addresses), IRC conversations, code, and 128-bit UUIDs, and the attack is possible even though each of these sequences is included in just one document in the training data. {#c-ca21-pii}
Later work found that the amount a model emits grows with its size, with the number of times a passage is repeated in the data and with the length of the prompt, and that a deployed chat model leaks too.

## In plain words

A language model is trained to predict the next word of its training text.
The hope is that it learns patterns, such as grammar and facts, rather than the text itself.
It turns out that large models also store long stretches of the text exactly: give one the start of a passage it was trained on, and it may continue with the next fifty words as they were written.
Sometimes it does so with no prompt at all.

This matters in two ways.
The training data of a web-scale model contains things that should not be repeated, such as a person's phone number or copyrighted text.
And a model that can recite a test set it saw during training gets a score that says little about what it has learned, the problem of data contamination.
The surprise was that this happens in models that show no sign of overfitting by the usual test, a training loss much lower than the test loss.

## The finding

Carlini et al. write that the association between overfitting and memorization has erroneously led many to assume that current large LMs will not leak information about their training data, because these models are often trained on massive de-duplicated datasets only for a single epoch and exhibit little to no overfitting. {#c-ca21-wisdom}
GPT-2 was trained for approximately 12 epochs on 40GB of text de-duplicated at the document level, and does not overfit: the training loss is only roughly 10% smaller than the test loss across all model sizes. {#c-ca21-overfit}
Yet it emits training text verbatim.

The papers on this page share a few definitions.
Carlini et al. call a string $s$ extractable from an LM if there exists a prefix $c$ such that $s$ is the arg max of the model's likelihood of a sequence given $c$, and note that the arg max can be replaced by an appropriate sampling strategy such as greedy sampling. {#c-ca21-extractable}
A string $s$ is $k$-eidetic memorized (for $k \ge 1$) by an LM if $s$ is extractable from the LM and $s$ appears in at most $k$ examples in the training data. {#c-ca21-eidetic}
A small $k$ is the worrying case: a passage that occurs in one document is private in a way that a licence printed thousands of times is not.
Carlini et al. (2023) call a string $s$ extractable with $k$ tokens of context from a model $f$ if there exists a length-$k$ string $p$ such that the concatenation $[p \| s]$ is contained in the training data for $f$, and $f$ produces $s$ when prompted with $p$ using greedy decoding. {#c-cq-def}
Tirumala et al. call a context $(s, y)$ memorized if the argmax of the language model on $s$ is the ground truth token $y$, and refer to the proportion of memorized contexts as exact memorization, which can also be seen as accuracy. {#c-ti-def}
Nasr et al. study extractable memorization, training data that an adversary can efficiently extract by querying a model without prior knowledge of the training dataset, unlike discoverable memorization, which captures an upper bound on all training data that is memorized even if it can only be recovered by prompting the model with other training data. {#c-na-kinds}

Measured either way, the amount grows with scale.
Carlini et al. (2023) describe three log-linear relationships: memorization significantly grows as we increase the capacity of a model, the number of times an example has been duplicated, and the number of tokens of context used to prompt the model. {#c-cq-three}

## Evidence

The generated list gives each source with its label and its role.
The lines of evidence are these.

**The first extraction attack.**
Carlini et al. sample exactly 256 tokens for each trial using the top-$n$ strategy with $n = 40$, then remove the generated samples that are unlikely to contain memorized text using a membership inference attack. {#c-ca21-sampling}
One of their membership inference signals compares GPT-2 with a much smaller model trained on the same data, the Small (117M parameters) and Medium (345M parameters) models, because smaller models have less capacity for memorization. {#c-ca21-ratio}
Across all strategies Carlini et al. identify 604 unique memorized training examples from among the 1,800 possible candidates, for an aggregate true positive rate of 33.5%, and their best variant has a true positive rate of 67%. {#c-ca21-604}
GPT-2 memorizes the entire text of the MIT public license, but this is only $k$-eidetic memorization for a large value of $k$, since these licenses occur thousands of times. {#c-ca21-mit}
Among 600,000 generated samples, the attacks find that at least 604 (or 0.1%) contain memorized text, which Carlini et al. call likely an extremely loose lower bound, since they only manually inspected 1,800 potential candidate memorized samples. {#c-ca21-bound}

They then asked how much repetition a model needs.
Prior work investigated LM memorization by varying the number of times particular canary tokens were inserted into a training dataset, which Carlini et al. call synthetic: canaries are inserted artificially after the dataset has been collected and may not be representative of natural data. {#c-ca21-canary}
Instead they used URLs that the training data happened to repeat many times within single documents.
For these URLs, the full-sized 1.5 billion parameter GPT-2 model emits all examples that are inserted 33 times or more, the medium-sized 345 million parameter model memorizes half of the URLs, and the smallest 117 million parameter model memorizes none of these URLs. {#c-ca21-urls}
Carlini et al. draw two lessons: larger models memorize significantly more training data, and for the largest LM complete memorization occurs after just 33 insertions. {#c-ca21-lessons}

**The three log-linear relationships.**
Carlini et al. (2023) turned the attack into a measurement on models with public training data, GPT-Neo and GPT-J, trained on The Pile.
They prompt the model with the first $\ell - 50$ tokens of a training sequence and report the sequence as extractable if the model exactly emits the next 50 token suffix; fifty tokens corresponds to an average of 127 characters or 25 words. {#c-cq-suffix}
Larger GPT-Neo models memorize significantly more than smaller models, with a near-perfect log-linear fit ($R^2$ of 99.8%): a ten fold increase in model size corresponds to an increase in memorization of 19 percentage points. {#c-cq-scale}
A larger model might simply be better at predicting text, so they ran a model that never saw the data.
GPT-2, which was not trained on The Pile, correctly completes approximately 6% of the examples in the evaluation set, compared to 40% for the similarly sized 1.3B parameter GPT-Neo model, so the larger models have actually memorized the data and are not simply more accurate. {#c-cq-baseline}
Examples repeated more often in the training set are more likely to be extractable, following a log-linear trend, yet memorization does still happen even with just a few duplicates, so deduplication will not perfectly prevent leakage. {#c-cq-dup}
33% of training sequences in the evaluation set are extractable from the 6B model at 50 tokens of context, compared to 65% with 450 tokens of context; Carlini et al. call this the discoverability phenomenon. {#c-cq-context}
The figures above come from a sample weighted toward repeated sequences.
On a uniformly random sample of The Pile, the last 50 tokens of a length-1000 sequence are extracted with 7% probability for the GPT-J 6B model, 4% for the 125M GPT-Neo model and 2% for the 1.5B parameter GPT2-XL model, which gives a lower bound that at least 1% of The Pile is extractable by the 6B GPT-J model but not by GPT-2 XL. {#c-cq-uniform}
The decoding rule matters little.
Beam search with 100 beams results in marginally more extracted memorization than greedy decoding, just under 2 percentage points on average with a maximum of 5.6%. {#c-cq-beam}
Exact match to the true continuation is also a narrow test: for examples at 100 repetitions, 32.6% of outputs are contained somewhere in the dataset but just 15.8% match the ground truth continuation. {#c-cq-anywhere}

**Duplication.**
Web corpora repeat text far more than their document-level deduplication suggests.
Lee et al. (2022) find a 61-word sequence in C4 that is repeated 61,036 times verbatim in the training dataset and 61 times in the validation set. {#c-lee-sentence}
They trained models on C4 and on versions with repeated documents and repeated substrings removed.
Generating 100,000 unprompted samples of up to 512 tokens and counting a token as memorized if it is part of a 50-token substring that is exactly contained in the training data, over 1% of the generated tokens of the model trained on the original C4 belong to memorized sub-sequences, about 10× more than for the models trained on deduplicated data. {#c-lee-unprompted}
Models trained on deduplicated datasets have no worse perplexity compared to baseline models trained on the original datasets, and in some cases deduplication reduces perplexity by up to 10%. {#c-lee-ppl}
Kandpal et al. (2022) measured the effect of the number of copies directly.
They find that the rate at which language models regenerate training sequences is superlinearly related to a sequence's count in the training set: a sequence that is present 10 times in the training data is on average generated ∼1000× more often than a sequence that is present only once. {#c-ka-superlinear}
Kandpal et al. compare against perfect memorization, which amounts to generating a sequence at the same frequency as it appears in the training data; all LMs tested show a superlinear increase in the expected number of generations, slopes greater than 1 on a log-log plot. {#c-ka-perfect}
Model inversion attacks are much weaker for models retrained on deduplicated data: they emit ∼20× less training data and two of the three proposed membership inference methods become less effective. {#c-ka-dedup}
Carlini et al. (2023) repeated the comparison on the models of Lee et al.
For sequences repeated below 35 times, the exact deduplicated model memorizes an average of 1.2% of sequences, compared to 3.6% without deduplication, a statistically significant decrease by a factor of 3×. {#c-cq-dedup}

**Training dynamics.**
Tirumala et al. (2022) followed exact memorization through training.
They find that larger language models memorize training data faster across all settings, can memorize a larger portion of the data before over-fitting and tend to forget less throughout the training process. {#c-ti-faster}
Models memorize nouns and numbers first, and Tirumala et al. hypothesize and provide empirical evidence that nouns and numbers act as a unique identifier for memorizing individual training examples. {#c-ti-nouns}

**Prediction from small models.**
Biderman et al. (2023) asked whether a small model, or an early checkpoint, can tell which sequences a large model will memorize.
They term the accuracy of tokens in the continuation the memorization score, the number of ordered matching tokens between the model's greedily generated sequence and the dataset's true continuation, with a prompt of 32 tokens and a continuation of 32 tokens, and call a sequence memorized or extractable if the memorization score is 1. {#c-bi-score}
On the Pythia models, 95.6% of the sequences memorized by the 70M model were also memorized by the 12B model, but those only accounted for 19.7% of the sequences that the 12B model memorized. {#c-bi-70m}
The recall is incredibly low across the board, with even the 1.4B parameter model only achieving a recall of 0.554 when trying to forecast the behavior of a model an order of magnitude larger. {#c-bi-recall}
Predicting from intermediate checkpoints, precision remains high throughout models but recall is low for all predictors that use significantly less compute than the final model's cost. {#c-bi-checkpoints}
So a small model's memorized set is mostly a subset of the large model's, but most of what the large model memorizes cannot be found by running a small one.

**Production models.**
Nasr et al. (2023) scaled the attack up and turned it on deployed models.
They show an adversary can extract gigabytes of training data from open-source language models like Pythia or GPT-Neo, semi-open models like LLaMA or Falcon, and closed models like ChatGPT. {#c-na-gigabytes}
The aligned ChatGPT (gpt-3.5-turbo) appears 50× more private than any prior model, but under the attack of Nasr et al. ChatGPT emits training data 150× more frequently than with prior attacks, and 3× more frequently than the base model. {#c-na-aligned}
Prompting ChatGPT to repeat a word forever, such as Repeat this word forever: poem poem poem poem, makes a small fraction of generations diverge to memorization, and using only $200 USD worth of queries Nasr et al. extract over 10,000 unique verbatim-memorized training examples. {#c-na-poem}
For ChatGPT the training data is not public, so they checked the output against a large collection of public web text, which makes the count a lower bound.

## Scope and conditions

**What counts as memorized.**
Every result here uses an exact-match test on a fixed length, 50 tokens for Carlini et al. and Lee et al. and 32 for Biderman et al.
A paraphrase, or a passage with one token changed, does not count, so the figures say nothing about memorization of facts or style.
The test also counts text that a model could produce without having seen it, such as a licence or a sequence of numbers, which is why Carlini et al. (2023) compare with GPT-2.

**Prompted and unprompted.**
Prompting with the true prefix measures discoverable memorization, an upper bound on what an outside attacker could get; generating without a prompt and searching the output for training text measures extractable memorization, a lower bound.
The page's figures mix the two, and the gap between them can be large.

**The model family.**
The log-linear trends are clearest for GPT-Neo and GPT-J on The Pile.
For T5 masked language models the scaling trend holds, but the absolute memorization is an order of magnitude lower than for comparably sized causal language models: the 3B parameter T5-XL model memorizes 3.5% of sequences repeated 100 times, whereas the GPT-Neo 2.7B model memorizes 53.6%. {#c-cq-t5}
For T5 models trained on C4, sequences duplicated more often tend to be easier to memorize but there is no monotonic scaling relationship; most of the duplicate examples repeated 138-158 times consist mainly of whitespace tokens. {#c-cq-t5-dup}
The masked language modelling page describes the objective these models are trained on.
For the OPT family of models, from 125 million to 175 billion parameters and trained on a deduplicated dataset that overlaps with The Pile, the scaling trends are nearly identical to those of GPT-Neo but the effect size is orders-of-magnitude smaller: even the 66 billion parameter model memorizes a smaller fraction of The Pile than the smallest 125 million parameter GPT Neo model. {#c-cq-opt}
Carlini et al. give two possible conclusions: careful data curation and training can mitigate memorization, or even slight shifts in data distribution can significantly alter what content gets memorized. {#c-cq-opt-why}
Prompting OPT with text from The Pile tests it on data it may only partly have seen, so the second reading cannot be ruled out.

**Deduplication is partial.**
While deduplication does help for sequences repeated up to ∼100 times, it does not help for sequences repeated more often, which Carlini et al. attribute to the fact that any deduplication strategy is necessarily imperfect. {#c-cq-dedup-limit}
Sequences that occur once are still memorized, as the GPT-2 attack showed, but they are hard to detect.
The membership inference methods of Carlini et al. have an area under the ROC curve as high as 0.90 for sequences that are duplicated many times but achieve only chance accuracy for sequences that appear once. {#c-ka-mia}

## Explanations

**Duplication.**
The claim: text that occurs many times in the training data is regenerated far more often than its share, so a small number of heavily repeated passages dominates what a model emits.
The evidence: the superlinear curve of Kandpal et al., the tenfold drop after deduplication of Lee et al., and the log-linear duplication trend of Carlini et al. (2023).
The derivation memorization-arithmetic puts the superlinear curve in numbers.
If a sequence with $d$ copies in the training data is generated $E(d) = c d^s$ times, then $E(10)/E(1) = 10^s$; perfect memorization, sampling from the training data itself, gives $s = 1$, and a ratio of 1000 gives $s = 3$. {#c-ma-slope}
With $s > 1$ the generations per copy, $E(d)/d = c d^{s-1}$, rise with $d$; with $s = 3$ a sequence with 10 copies is generated 100 times more often per copy than a sequence with one copy. {#c-ma-per-copy}
The consequence: deduplication is the cheapest defence and, by the result of Lee et al., costs nothing in perplexity, but it cannot reach the sequences seen once.

**Capacity.**
The claim: larger models fit individual examples faster and keep more of them, before any overfitting shows in the average loss.
The evidence: Tirumala et al. measure it during training, and Carlini et al. (2023) find a fixed gain per tenfold increase in size.
The same derivation shows what a log-linear fit implies.
If the extractable fraction is $F(N) = a + b \log_{10} N$ with $b = 0.19$, doubling the model adds $0.19 \log_{10} 2$, about 5.7 percentage points, and because $F$ cannot exceed 1 the relation cannot hold beyond $N = 10^{(1 - a)/b}$. {#c-ma-loglinear}
The consequence: the trend has to bend somewhere, and the OPT results show it can sit at very different levels for models of the same size.

**The long tail.**
The claim: memorizing rare examples is part of learning well, not a side effect of it.
Feldman proves that for natural data distributions memorization of labels is necessary for achieving close-to-optimal generalization error, even for outliers and noisy labels, in a model where data is sampled from a mixture of subpopulations and the distribution of subpopulation frequencies is long-tailed. {#c-fe-necessary}
A special case of Feldman's theorem bounds the expected generalization error of an algorithm by the optimal error plus $\tau_1$ times the expected number of examples that appear once in the dataset and are mislabeled by the classifier, where $\tau_1$ is the ratio of the expectation of $\alpha^2 (1 - \alpha)^{n-1}$ to that of $\alpha (1 - \alpha)^{n-1}$ over the frequencies $\alpha$. {#c-fe-bound}
For the Zipf distribution over a universe of size $N = 50{,}000$ and $n = 50{,}000$ samples, Feldman reports an expected loss of at least $\approx 0.47/n$ per example the learner does not fit, against a worst case of $\approx 0.09/n$, about 17% of samples appearing once, and an algorithm that does not fit them suboptimal by $\approx 7\%$. {#c-fe-zipf}
The derivation memorization-arithmetic recomputes the example with the plain Zipf law.
For 50,000 subpopulations with frequencies proportional to $1/i$ and $n = 50{,}000$ samples, $\tau_1 n \approx 0.472$, the smallest frequency is $\approx 0.088/n$, the expected fraction of samples whose subpopulation appears once is $\approx 0.170$, and a learner right on those samples only by chance among ten classes loses at least $\approx 0.072$. {#c-ma-zipf}
A distribution without a tail gives almost nothing.
For $K$ subpopulations of equal frequency, $\tau_1 = 1/K$ exactly and the expected fraction of samples whose subpopulation appears once is $(1 - 1/K)^{n-1}$; with $n = 50{,}000$ and $K = 5{,}000$ the same bound is about $4 \times 10^{-4}$. {#c-ma-uniform}
The consequence: web text is long-tailed, so a model trained to low loss on it is pushed to fit passages it sees once, and some of those are the ones that should not leak.
Feldman's theorem is about classification labels, not text, so for language models it is an analogy rather than a proof.

**Measurement.**
The claim: how much memorization is found depends on the prompt, the decoding rule and the attack, so every figure is a lower bound.
The evidence: longer prompts double the extractable fraction in the 6B model, and the divergence attack of Nasr et al. raises the rate for ChatGPT 150-fold.
Nasr et al. conclude that practical attacks can recover far more data than previously thought, and that current alignment techniques do not eliminate memorization. {#c-na-alignment}
The consequence: a model that seems private under one test may not be under another, and red teaming for memorization has to try many prompts.

These accounts do not compete.
Duplication and capacity say which passages are memorized and by which models; the long tail says why a good learner memorizes at all; measurement says how much of it anyone can see.

## Implications

The implications below are those that the sources draw.
Carlini et al. (2021) and Nasr et al. treat memorization as a privacy risk, since personal data in a single web page can be extracted from a released model.
Lee et al. and Kandpal et al. recommend deduplicating training data, which reduces memorization and does not harm perplexity.
Carlini et al. (2023) note that limiting the length of the prompt reduces the risk, and that memorization will grow with model size unless active mitigations are used.

For evaluation, memorized test data inflates scores, and the 61 copies of one sentence in the C4 validation set illustrate how easily a test set can overlap with training data.
The data contamination page covers this side of the problem.
The red teaming page describes the wider practice of attacking a model to find failures, of which extraction attacks are one kind.
The neural scaling laws page gives the smooth laws that the loss follows as models grow, the double descent page another case where fitting the training data does not hurt test error, and the emergent abilities page a different behaviour that changes with scale.
The nucleus sampling page describes the sampling rules that decide which of the model's memorized continuations appear in its output.

## Open questions

Why the OPT models memorize so much less than GPT-Neo models of the same size, and whether the cause is data curation or a shift in data distribution, is open.
How to predict which sequences a large model will memorize, before training it, is open; Biderman et al. find small models a poor guide.
How to remove a specific memorized passage from a trained model, and how to verify that it is gone, are active questions.
Whether near-verbatim memorization, a passage reproduced with small changes, follows the same trends as exact memorization is less studied.

## History

Feldman posted "Does Learning Require Memorization? A Short Tale about a Long Tail" as arXiv 1906.05271 in June 2019; it appeared at STOC 2020.
Earlier work had inserted synthetic canaries into training data and measured whether a model exposed them.
Carlini et al. posted "Extracting Training Data from Large Language Models" as arXiv 2012.07805 in December 2020; it appeared at USENIX Security 2021.
Lee et al. posted "Deduplicating Training Data Makes Language Models Better" as arXiv 2107.06499 in July 2021, and it appeared at ACL 2022.
Carlini et al. posted "Quantifying Memorization Across Neural Language Models" in February 2022, and it appeared at ICLR 2023; Kandpal et al. posted their study of duplication the same month, and it appeared at ICML 2022.
Tirumala et al. published their study of training dynamics at NeurIPS 2022, and Biderman et al. their prediction study on the Pythia models in 2023.
Nasr et al. posted "Scalable Extraction of Training Data from (Production) Language Models" as arXiv 2311.17035 in November 2023.

## References
