---
tag: V3JG9E
kind: metric
slug: bleu
name: BLEU
aliases: [BLEU score, bilingual evaluation understudy, corpus BLEU, sacreBLEU]
area: evaluation/text-generation-metrics
level: core
summary: >-
  BLEU scores machine output against human references: the geometric mean of the clipped 1- to 4-gram precisions over the whole test set, times a penalty for output that is too short.
properties:
  measures: "n-gram overlap between system outputs and one or more human references, over a whole test set"
  range: "0 to 100 in sacreBLEU; 0 to 1 in Papineni et al. (2002)"
  better: higher
  formula: "BP * exp((1/K) * sum_{n=1}^{K} log p_n), with BP = min(1, exp(1 - r/c)) and K = 4"
  level: "corpus; sentence BLEU needs smoothing and is a different quantity"
  reference_impl: "sacreBLEU 2.6.0 (https://github.com/mjpost/sacrebleu), signature nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0"
history:
  year: 2002
  people: [Kishore Papineni, Salim Roukos, Todd Ward, Wei-Jing Zhu]
  note: "IBM Research Report RC22176 (2001); published at ACL 2002, Philadelphia, pages 311 to 318."
prereqs: [precision-and-recall@al, tokens-and-lexing@pg, count-the-items-of-each-kind@pg]
same_as:
  wikidata: Q3500773
  wikipedia: BLEU
sources:
  - {paper: "Papineni, Roukos, Ward and Zhu (2002)", title: "BLEU: a Method for Automatic Evaluation of Machine Translation", locator: "ACL 2002, pages 311 to 318, Sections 2.1 to 2.3 and 3", note: "the definition: modified precision, the geometric mean, the brevity penalty"}
  - {paper: "Post (2018)", title: "A Call for Clarity in Reporting BLEU Scores", locator: "WMT 2018, Volume 1, pages 186 to 191, Sections 2 and 3, Table 1", note: "why scores differ across papers, and sacreBLEU"}
  - {url: "https://github.com/mjpost/sacrebleu", title: "sacreBLEU", locator: "version 2.6.0, sacrebleu/metrics/bleu.py and sacrebleu/tokenizers/", note: "the official implementation; every vector of this page comes from it"}
  - {paper: "Chen and Cherry (2014)", title: "A Systematic Comparison of Smoothing Techniques for Sentence-Level BLEU", locator: "WMT 2014, pages 362 to 367, Section 2.2", note: "the smoothing methods 1 to 7; floor is method 1, add-k is method 2, exp is method 3"}
  - {paper: "Lin and Och (2004)", title: "ORANGE: a Method for Evaluating Automatic Evaluation Metrics for Machine Translation", locator: "COLING 2004, Section 4, the definition of BLEUS", note: "add one to the matches and the totals of the orders above 1"}
  - {paper: "Callison-Burch, Osborne and Koehn (2006)", title: "Re-evaluating the Role of BLEU in Machine Translation Research", locator: "EACL 2006, pages 249 to 256, Sections 3 and 4"}
  - {paper: "Reiter (2018)", title: "A Structured Review of the Validity of BLEU", locator: "Computational Linguistics 44(3), 393 to 401, the abstract and Section 5"}
  - {paper: "Koehn (2004)", title: "Statistical Significance Tests for Machine Translation Evaluation", locator: "EMNLP 2004, Sections 5 and 6", note: "bootstrap and paired bootstrap resampling of BLEU"}
  - {paper: "Mathur, Baldwin and Cohn (2020)", title: "Tangled up in BLEU: Reevaluating the Evaluation of Automatic Machine Translation Evaluation Metrics", locator: "ACL 2020, pages 4984 to 4997, Sections 4.2, 5 and 6"}
  - {paper: "Kocmi, Federmann, Grundkiewicz, Junczys-Dowmunt, Matsushita and Menezes (2021)", title: "To Ship or Not to Ship: An Extensive Evaluation of Automatic Metrics for Machine Translation", locator: "WMT 2021, pages 478 to 494, Table 2 and Section 5"}
  - {paper: "Freitag et al. (2022)", title: "Results of WMT22 Metrics Shared Task: Stop Using BLEU – Neural Metrics Are Better and More Robust", locator: "WMT 2022, pages 46 to 68, the title and the conclusions"}
  - {paper: "Popović (2015)", title: "chrF: character n-gram F-score for automatic MT evaluation", locator: "WMT 2015, pages 392 to 395", note: "the character n-gram metric that sacreBLEU also computes"}
---

BLEU is the standard automatic metric of machine translation.
It compares the outputs of a system with human translations of the same sentences, the references.
It counts how many of the short word sequences of the outputs also occur in the references.
Two corrections turn this precision into BLEU.
A word matches at most as often as a reference uses it, and a penalty lowers the score of output that is too short.
BLEU is defined on a whole test set, not on one sentence.
The tokenizer, the casing, the number of references and the smoothing all change the number, so a BLEU score needs the settings that produced it.

## In plain words

Take each sentence the system produced, and look at its single words, its pairs of words, its runs of three and its runs of four.
For each size, count the fraction that also appear in a human translation of that sentence.
Never count a word more often than the human translation uses it.
Pool these counts over the whole test set and average the four fractions, so that one fraction near zero pulls the result down hard.
Then lower the result if the output is shorter overall than the human translations.
A score of 100 means an exact match with the references.
Two scores can be compared only when they were computed the same way on the same references.

## Definition

A test set has segments $s = 1, \dots, S$.
Segment $s$ has one hypothesis $h_s$, the output of the system, and a set $R_s$ of references.
Every segment has the same number of references.
All texts are tokenized first, by default with the `13a` tokenizer of WMT (see Variants).
For a token sequence $x$ and an n-gram $g$, $\#_x(g)$ is the number of times $g$ occurs in $x$.
$G_n(x)$ is the set of n-grams of length $n$ in $x$.

The **modified n-gram precision** of order $n$ clips each hypothesis count at the largest count of the same n-gram in any one reference.
It pools the counts over the whole test set before it divides:

$$ p_n = \frac{\sum_{s=1}^{S} \sum_{g \in G_n(h_s)} \min\big(\#_{h_s}(g),\ \max_{\rho \in R_s} \#_{\rho}(g)\big)}{\sum_{s=1}^{S} \sum_{g \in G_n(h_s)} \#_{h_s}(g)} $$ {#eq-modified-precision}

Let $c$ be the number of hypothesis tokens in the test set.
For each segment, take the length of the reference that is closest in length to the hypothesis, and the shorter one on a tie.
The sum of these lengths is the **effective reference length** $r$.
The **brevity penalty** is

$$ \mathrm{BP} = \begin{cases} 1 & \text{if } c \ge r, \\ e^{\,1 - r/c} & \text{if } 0 < c < r, \\ 0 & \text{if } c = 0. \end{cases} $$ {#eq-brevity-penalty}

With the largest order $K = 4$ and the weight $1/K$ for each order, BLEU is

$$ \mathrm{BLEU} = \mathrm{BP} \cdot \exp\Big( \frac{1}{K} \sum_{n=1}^{K} \log p_n \Big), \qquad \log \mathrm{BLEU} = \min\Big(1 - \frac{r}{c},\ 0\Big) + \frac{1}{K} \sum_{n=1}^{K} \log p_n $$ {#eq-bleu}

and it is 0 when some $p_n$ is 0.
Papineni et al. (2002) report BLEU from 0 to 1; sacreBLEU and most papers multiply it by 100.

> [!note]
> Papineni et al. (2002, Section 2.3) write $N$ for the largest order and $w_n$ for the weights. This page writes $K$, because $N$ is the number of examples on this site. It fixes $w_n = 1/K$, the only weights in use. The paper's condition for the penalty is $c > r$ against $c \le r$. At $c = r$ both cases give 1, so the two formulas agree.

**Smoothing.** A precision of 0 makes the score 0.
This happens often for one sentence and seldom for a test set.
The default of sacreBLEU, `exp`, is method 3 of Chen and Cherry (2014), from the NIST script `mteval-v13a`.
The orders with no match get the counts $1/2$, $1/4$, $1/8$ and so on in place of 0, from the lowest such order up:

$$ p_n = \frac{1}{2^{\,z_n} \, T_n} \quad \text{when the matches of order } n \text{ are } 0 \text{ and } T_n > 0, $$ {#eq-smooth-exp}

where $T_n$ is the denominator of eq-modified-precision and $z_n$ is the number of orders up to $n$ with no match.
When every order has a match, smoothing changes nothing.

## Properties

- **Range.** $0 \le \mathrm{BLEU} \le 1$, and it is 1 only when every clipped count equals its total and $c \ge r$. {#c-score-range}
- **The penalty.** For $c > 0$, $\mathrm{BP}$ is in $(0, 1]$ and is 1 exactly when $c \ge r$. It is continuous at $c = r$ and falls to 0 as $c \to 0$. {#c-bp-range} The file `derivations/brevity-penalty.md` derives both results step by step.
- **A zero precision.** Without smoothing, one $p_n = 0$ makes BLEU 0, whatever the other orders are. {#c-zero-precision}
- **Corpus level.** The counts and the lengths are summed over the segments before any division. So BLEU is a function of $2K + 2$ sums, it does not depend on the order of the segments, and it is not the mean of the sentence scores.
- **No recall.** A hypothesis gets no reward for covering more of a reference. The brevity penalty stands in for recall, because output that leaves content out is usually short (Papineni et al., 2002, Section 2.2.1).
- **More references, higher scores.** A second reference can only raise a clipped count. Papineni et al. (2002, Section 3) and Post (2018, Section 2.1) both report higher BLEU with more references. {#c-more-references} This is not a theorem, because a nearer but longer reference can lower BP. Scores on different numbers of references cannot be compared.
- **Copies.** Output that copies one reference of each segment scores 100 if some segment has at least $K$ tokens. If no segment has a 4-gram, $p_4$ is 0 and so is the score, unless effective order is on.

## Worked example

One segment, one reference and the default options: tokenizer `13a`, case-sensitive, `exp` smoothing, $K = 4$.

- hypothesis: `the cat sat on the mat`, so $c = 6$
- reference: `the cat sat on the red mat`, so $r = 7$

| $n$ | n-grams of the hypothesis | not in the reference | clipped matches | total | $p_n$ |
|---|---|---|---|---|---|
| 1 | the, cat, sat, on, the, mat | none | 6 | 6 | 1 |
| 2 | the cat, cat sat, sat on, on the, the mat | the mat | 4 | 5 | 0.8 |
| 3 | the cat sat, cat sat on, sat on the, on the mat | on the mat | 3 | 4 | 0.75 |
| 4 | the cat sat on, cat sat on the, sat on the mat | sat on the mat | 2 | 3 | 0.6667 |

Both copies of `the` match, because the reference also has `the` twice.

| step | value |
|---|---|
| $p_1 p_2 p_3 p_4 = 1 \cdot 0.8 \cdot 0.75 \cdot \tfrac{2}{3}$ | 0.4 |
| geometric mean $0.4^{1/4}$ | 0.79527 |
| $\mathrm{BP} = e^{1 - 7/6} = e^{-1/6}$ | 0.84648 |
| $\mathrm{BLEU} = 0.84648 \cdot 0.79527$ | 0.67318 |

sacreBLEU 2.6.0 prints `BLEU = 67.32 100.0/80.0/75.0/66.7 (BP = 0.846 ratio = 0.857 hyp_len = 6 ref_len = 7)`.
This is the row `worked-example` of `vectors/official.jsonl`.
The test `test_worked_example` checks each number of the tables.

**Clipping.** Papineni et al. (2002, Section 2.1) give the hypothesis `the the the the the the the` with two references.
The references are `The cat is on the mat.` and `There is a cat on the mat.`.
Plain unigram precision is 7/7, but `the` occurs at most twice in one reference, so the modified unigram precision is 2/7. {#c-clipping-example}
With lowercasing, sacreBLEU gives the precisions 28.6, 8.3, 5.0 and 3.1 and a BLEU of 7.81 (row `papineni-clipping`).
The smoothing keeps that score above 0.

**Corpus against sentences.** The row `corpus-4seg-1ref` has four segments, and its BLEU is 57.31.
Its four sentence scores, from `sacrebleu.sentence_bleu`, are 39.67, 37.82, 76.12 and 100, with the mean 63.40.
The long first segment has the most n-grams, so it weighs more in the corpus score than in the mean.

## Reference implementation

`impl/numpy/bleu.py` follows sacreBLEU 2.6.0 in two functions.
`statistics` tokenizes the texts and counts the n-grams of each segment with `collections.Counter`.
It takes the union of the reference counts for the clip, and returns $c$, $r$ and the clipped and total counts of each order.
`score` turns these sums into the precisions, the smoothing, the penalty and the geometric mean, as float64 arrays.
The PyTorch and JAX versions share the counting code and do the arithmetic in 64-bit floats of their framework.
All three reproduce the 28 rows of `vectors/` within $10^{-9}$.
A differential run of 4,000 random cases against sacreBLEU 2.6.0 found no difference, over every tokenizer and smoothing and the orders 1 to 5.

Two differences from sacreBLEU are on purpose.
First, the code raises `ValueError` when the number of hypotheses and of reference sets differ, or when the segments have different numbers of references.
sacreBLEU 2.6.0 drops the extra segments without a message.
Second, the `intl` tokenizer reads the Unicode categories from Python's `unicodedata`, and sacreBLEU reads them from the `regex` package.
The two can differ on characters that only the newer of the two Unicode tables knows.

> [!pitfall]
> **A different tokenizer.** BLEU on text that the user tokenized, or that `13a` tokenizes a second time, differs from the BLEU of other papers. Post (2018, Table 1) measured changes of up to 1.8 points from the tokenization of the references alone. {#c-tokenization-changes} The test `test_tokenizers` checks the exact output of `13a`, `intl`, `char` and `none`. Its inputs have HTML entities, dashes between digits, decimal numbers and Unicode punctuation.

> [!pitfall]
> **The wrong reference length.** With several references, $r$ uses the reference closest in length to each hypothesis, and the shorter one on a tie. It is not the shortest, the longest or the mean. The test `test_closest_reference_length` checks the tie and the nearer reference.

> [!pitfall]
> **The penalty per sentence.** A penalty for each segment, or a mean of sentence scores, is not BLEU. The test `test_brevity_penalty_is_corpus_level` has a short and a long segment whose corpus penalty is 1.

> [!pitfall]
> **No match, no score.** A short segment with no 4-gram match scores 0 without smoothing. A segment with no 4-gram at all scores 0 even with `exp` smoothing, unless effective order is on. The tests `test_smoothing` and `test_effective_order` check both cases.

> [!pitfall]
> **No clipping.** Code that counts every hypothesis n-gram found in a reference, without the clip, rewards repetition. The test `test_clipping` checks the 2/7 of Papineni et al.

## Interpretation

BLEU ranks systems on one test set; its absolute value says little.
Two scores are comparable only with the same test set, the same references and the same sacreBLEU signature (Post, 2018, Section 3).
The signature records the number of references, the casing, the tokenizer, the smoothing and the version.
The references alone move the score: the WMT17 English to Finnish system online-B scores 22.04 with one reference and 25.25 with two (Post, 2018, Section 2.1).
A difference of a point or two is weak evidence.
Mathur et al. (2020, Sections 5 and 6) found that BLEU differences of 1 to 2 points agree with human judgements only about half of the time. {#c-small-differences}
Koehn (2004, Sections 5 and 6) gives bootstrap resampling over segments for a confidence interval, and paired bootstrap resampling for a significance test.
BLEU is a function of sums, so each resample needs only the statistics of its segments.

## Limitations and criticism

BLEU sees only surface n-grams.
A synonym, a paraphrase or another word order counts as an error.
A fluent but wrong translation can share many n-grams with the reference.
Callison-Burch et al. (2006, Section 3.1) count more than $10^{73}$ reorderings of some hypotheses that get a similar BLEU. {#c-permutations}
That is too many for all of them to be equally good translations.
In the 2005 NIST evaluation of Arabic to English, the entry that human judges ranked first was sixth by BLEU (Callison-Burch et al., 2006, Section 4). {#c-nist-2005-rank}
In their own French to English test, BLEU scored the rule-based system Systran far lower than human judges did.
They conclude that a higher BLEU is neither necessary nor sufficient for a better translation.

Reiter (2018) reviewed 284 correlations from 34 papers. {#c-reiter-review}
The review supports BLEU for the diagnostic evaluation of machine translation systems.
It does not support BLEU for single texts, for tasks other than machine translation, or for testing scientific hypotheses.
Kocmi et al. (2021, Table 2) measured how often a metric orders a pair of systems as human judges do.
Over 3,344 pairs, BLEU did so 74.6 percent of the time, chrF 75.6 percent and COMET 83.4 percent. {#c-pairwise-accuracy}
The WMT22 metrics task put its advice in its title: "Stop Using BLEU – Neural Metrics Are Better and More Robust" (Freitag et al., 2022). {#c-wmt22-stop}
BLEU stays in use because it is cheap and deterministic, it needs no model, and its signature makes a score reproducible.

## Variants

The options of `contract.yaml` change the number, and `vectors/variants.yaml` lists the rows that pin each value.

| option | values | effect | used by |
|---|---|---|---|
| `tokenize` | `13a` (default), `intl`, `char`, `none` | how text becomes tokens; `13a` is `mteval-v13a.pl`, `intl` is the Unicode tokenizer of `mteval-v14.pl` | `13a` in WMT and sacreBLEU; `char` and other tokenizers for Chinese, Japanese and Korean |
| `lowercase` | false (default), true | case-insensitive matching | Papineni et al. (2002, Section 2.3) fold case |
| `smooth_method` | `exp` (default), `floor`, `add-k`, `none` | the value of an order with no match | `exp` is Chen and Cherry method 3, `floor` is method 1, `add-k` with $k = 1$ is method 2 from Lin and Och (2004) |
| `max_ngram_order` | 4 (default), any $K \ge 1$ | BLEU-$K$ | BLEU-4 is the baseline of Papineni et al. (2002, Section 2.3) |
| `effective_order` | false (default), true | average only over the orders that the hypotheses have | sentence BLEU; `sacrebleu.sentence_bleu` turns it on |

Chen and Cherry (2014, Section 2.2) compare seven smoothing methods for sentence BLEU. {#c-smoothing-methods}
Older papers often ran `multi-bleu.perl` of Moses on text that the authors had tokenized themselves.
Those numbers differ from sacreBLEU on the same output (Post, 2018).
chrF (Popović, 2015) replaces word n-grams with character n-grams and adds recall, and sacreBLEU computes it from the same files.

## Implementations

## History

Kishore Papineni, Salim Roukos, Todd Ward and Wei-Jing Zhu of IBM described BLEU in IBM Research Report RC22176 in 2001, and published it at ACL 2002. {#c-introduced-2002}
They presented it as an "understudy" to human judges, for when a system needs quick or frequent evaluations.
The `mteval` scripts of NIST became the reference implementation of the evaluation campaigns, and the WMT shared tasks reported BLEU for years.
Matt Post released sacreBLEU in 2018, with a signature that records every option, so that the scores of different papers can be compared. {#c-sacrebleu-2018}
Over the same years, a line of studies found BLEU weaker than its wide use suggested.
They include Callison-Burch et al. (2006), Reiter (2018), Mathur et al. (2020), Kocmi et al. (2021) and the WMT22 metrics task.

## References
