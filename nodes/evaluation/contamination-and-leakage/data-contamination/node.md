---
tag: EVFGC9
kind: risk
slug: data-contamination
name: Data contamination
aliases: [benchmark contamination, test set contamination, test data leakage, benchmark leakage]
area: evaluation/contamination-and-leakage
level: core
summary: >-
  The test data of a benchmark, or text close to it, is in the training data of the model being evaluated, so its score can overstate what it does on new problems.
history:
  year: 2020
  note: "Brown et al. (2020), arXiv 2005.14165, measured the 13-gram overlap of every benchmark of GPT-3 with its training data and reported scores on clean subsets; later model reports followed the practice."
sources:
  - {paper: "Brown et al. (2020)", title: "Language Models are Few-Shot Learners", locator: "Section 4 and pp. 43 to 44", note: "the 13-gram filter, its bug, the clean subsets and the per-benchmark results"}
  - {paper: "Chowdhery et al. (2022)", title: "PaLM: Scaling Language Modeling with Pathways", locator: "Section 8", note: "four categories of contamination and the 8-gram clean subsets"}
  - {paper: "Dodge et al. (2021)", title: "Documenting Large Webtext Corpora: A Case Study on the Colossal Clean Crawled Corpus", locator: "Section 4.2", note: "benchmark data found in C4"}
  - {paper: "Magar and Schwartz (2022)", title: "Data Contamination: From Memorization to Exploitation", locator: "Section 2 and the abstract", note: "a controlled measure of memorization and exploitation"}
  - {paper: "Srivastava et al. (2022)", title: "Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models", locator: "Section 2.4", note: "the BIG-bench canary string"}
  - {paper: "Golchin and Surdeanu (2023)", title: "Time Travel in LLMs: Tracing Data Contamination in Large Language Models", locator: "p. 1", note: "guided instruction and partition-level detection"}
  - {paper: "Shi et al. (2023)", title: "Detecting Pretraining Data from Large Language Models", locator: "Sections 2.2 and 3", note: "WikiMIA and Min-K% Prob"}
  - {paper: "Sainz et al. (2023)", title: "NLP Evaluation in trouble: On the Need to Measure LLM Data Contamination for each Benchmark", locator: "Section 3", note: "a definition and three types of contamination"}
  - {paper: "Yang, Chiang, Zheng, Gonzalez and Stoica (2023)", title: "Rethinking Benchmark and Contamination for Language Models with Rephrased Samples", locator: "the abstract", note: "rephrased samples that pass n-gram decontamination"}
  - {paper: "Zhang et al. (2024)", title: "A Careful Examination of Large Language Model Performance on Grade School Arithmetic", locator: "the abstract and Sections 3 and 5.4", note: "GSM1k against GSM8K"}
  - {paper: "Jain et al. (2024)", title: "LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code", locator: "Section 5.1", note: "scores by release month of the problems"}
  - {paper: "White et al. (2024)", title: "LiveBench: A Challenging, Contamination-Limited LLM Benchmark", locator: "p. 1", note: "a benchmark with monthly new questions and objective scoring"}
---

Data contamination is the presence of a benchmark's test data in the training data of the model that is evaluated on it.
A contaminated model can score well because it has seen the questions, or their answers, and not because it can solve new ones.
The risk grew with large language models, which are trained on web text that also holds many benchmarks, and whose training data is rarely published.
It is detected by comparing the training text with the test text, or, when the training data is closed, by probing the model itself.
Its main mitigations are clean subsets, private or held-out test sets, canary strings, and test sets of problems written after the model's training cutoff.

## In plain words

A benchmark is a fixed exam for AI models.
It works only if the model has not seen the exam before.
Large language models learn from a large share of the public web, and many exams, with their answers, are on the web.
So a model may have read the exam during training, word for word or in a reworded copy.
Its score then mixes two things: what it can do, and what it remembers.
Researchers check for this by searching the training text for pieces of the exam, by testing whether the model finishes exam questions it should not know, and by writing new exams that cannot have been seen.
When a model does much worse on a new exam of the same kind, the old score was probably too high.

## Definition and scope

Shahriar Golchin and Mihai Surdeanu define data contamination as the presence of test data from downstream tasks in the training data of large language models (LLMs). {#c-def-golchin}
Oscar Sainz and colleagues give a wider definition: any breach in the strict control of datasets that the experimental protocol requires, of which the case where an LLM processed the evaluation benchmark during its pre-training is one. {#c-def-sainz}
They describe three types: guideline contamination, raw text contamination and annotation contamination. {#c-sainz-types}
In their account, annotation contamination of the evaluation split completely invalidates the experiment, while contamination of the train or development splits invalidates claims of zero-shot or few-shot performance. {#c-sainz-annotation}
Jesse Dodge and colleagues separate input-and-label contamination, where the labels of a task are in the pretraining corpus, from input contamination, where only the evaluation inputs are. {#c-dodge-kinds}

The forms differ in how close the leaked text is to the test item.

**Exact and n-gram overlap.**
The test item, or a long run of its words, appears in the training text.
The GPT-3 report counts an example as dirty when it has any N-gram overlap with any training document, and clean when it has no collision. {#c-gpt3-dirty}
There N is the 5th percentile of the example length in words, with a minimum of 8 on non-synthetic tasks and a maximum of 13 for all tasks. {#c-gpt3-n}
Dodge et al. measure exact matches instead, normalized for capitalization and punctuation. {#c-dodge-exact}

**Semantic or paraphrased contamination.**
The training data holds a reworded, translated or reformatted version of the test item, which string matching does not find.
Shuo Yang and colleagues report that simple variations of test data, such as paraphrasing and translation, bypass decontamination by n-gram overlap. {#c-yang-bypass}
According to the GPT-3 report, 132 Winograd schemas in its training data were present in a different format from the one used to present the task to the model. {#c-gpt3-winograd-format}

**Context only.**
Many benchmarks give the model a passage at test time and ask a new question about it.
The PaLM authors argue that when the context is given at evaluation time, having trained on it for language modeling gives no unfair advantage. {#c-palm-context}
They sort benchmarks into four categories: wholesale contamination, constructed from web, context on web, and no significant overlap, and they count only the first two as contaminated. {#c-palm-categories}

**Memorization against exploitation.**
Inbal Magar and Roy Schwartz split the effect in two.
Their mem is the difference in masked language modeling accuracy on the gold label between test examples seen and unseen in pretraining, and their expl is the difference in task performance between seen and unseen examples after fine-tuning. {#c-magar-measures}

**Outside the risk.**
A model that does well on a test set after training on other data of the same distribution is not contaminated; that is the purpose of a train and test split.
Overfitting to a benchmark through many rounds of model selection on its test set is a neighbouring problem of the same area, and it can lower scores on a new test set with no leaked text at all.
Membership inference, the question of whether a given text was in the training data, is the tool that many detection methods use, and it is also a privacy risk of its own.

## How it arises

The training data of LLMs is large, scraped from the web, and mostly not published.
Weijia Shi and colleagues write that, at the scale of up to trillions of tokens, the pretraining data almost certainly includes test data for widely reported benchmarks, but no one can know which data or in what proportions. {#c-shi-scale}
Dodge et al. describe two ways in which a dataset ends up in a snapshot of Common Crawl: it was built from text on the web, or it was uploaded after its creation, for example to a GitHub repository. {#c-dodge-two-ways}

Filtering is the first defence, and it can fail.
To clean the GPT-3 training data, the GPT-3 report says, Tom Brown and colleagues searched for 13-gram overlaps between all the test and development sets of the work and the training data, and removed each colliding 13-gram with a 200-character window around it. {#c-gpt3-filter}
The report says that a bug left part of the detected overlaps in the training data, and that the cost of training made it infeasible to retrain the model. {#c-gpt3-bug}

Contamination can also enter after pretraining.
Sainz et al. note that it can occur in supervised fine-tuning, though less often, since research practice asks that this data be documented. {#c-sainz-sft}
Yang et al. also find contamination in synthetic data generated by GPT-3.5 and GPT-4, which suggests a risk of unintentional contamination. {#c-yang-synthetic}

The mechanism is simple in a controlled setting.
Magar and Schwartz pretrained BERT models on Wikipedia joined with labeled downstream data, then fine-tuned them on the same task. {#c-magar-setup}
They find exploitation in some cases, and in others models that memorize the contaminated data without exploiting it; the number of duplications of the data and the model size affect the two measures. {#c-magar-result}

## Examples

**Benchmarks in a web corpus.**
Dodge et al. find that 1.87 to 24.88% of the target texts of the test sets of their generation tasks appear in C4.EN, the English Colossal Clean Crawled Corpus. {#c-dodge-targets}
For the test inputs of the GLUE benchmark, the share found in C4.EN ranges from less than 2% to over 50%. {#c-dodge-glue}
They also find 4.6% and 5.7% of the examples of the LAMA T-REx and Google-RE sets verbatim in C4.EN. {#c-dodge-lama}

**GPT-3.**
The GPT-3 report flagged more than 90% of the examples of QuAC, SQuAD2 and DROP as potentially contaminated. {#c-gpt3-reading-flag}
According to the report, on inspection the source texts were in the training data but the question and answer pairs were not, so the model gains only background information. {#c-gpt3-reading-context}
The report's overlap analysis flagged 29% of the examples of PIQA, and the clean subset scored 3 percentage points lower, a 4% relative decrease; the paper marks its PIQA results with an asterisk. {#c-gpt3-piqa}
The report flagged 25% of the WMT16 German-English test set, with a total effect of 1 to 2 BLEU. {#c-gpt3-wmt}

**GPT-4.**
Golchin and Surdeanu report that GPT-4 is contaminated with the AG News, WNLI and XSum datasets. {#c-golchin-gpt4}

**Code problems by date.**
Naman Jain and colleagues score models on LeetCode problems by the month they were released.
The DeepSeek-Instruct 33B model drops sharply on problems released after August 2023, right before its release date, and GPT-4o drops on problems released since November, its official cutoff date. {#c-lcb-drop}
The drop occurs mainly for LeetCode; performance is relatively smooth across months on the other platforms. {#c-lcb-leetcode}

**Grade school mathematics.**
Hugh Zhang and colleagues wrote GSM1k to mirror the style and difficulty of [[gsm8k|GSM8K]], and kept it unpublished. {#c-gsm1k-mirror}
On GSM1k, the open and closed models they tested lose up to 8% of accuracy against GSM8K, and several model families show systematic overfitting across almost all sizes. {#c-gsm1k-drops}

**Overfitting on purpose.**
Yang et al. report that a 13B model trained on rephrased test samples reaches very high scores, on par with GPT-4, on benchmarks such as [[mmlu|MMLU]], GSM8K and HumanEval. {#c-yang-13b}

## Evidence

The examples document contamination in many benchmarks and corpora; how much it raises scores depends on the benchmark, the model and the form of the leak.

**Clean subsets.**
The GPT-3 report finds that a quarter of its benchmarks score over 50% potential contamination, yet in most cases performance changes only negligibly, with no evidence that contamination level and performance difference are correlated. {#c-gpt3-quarter}
In the same report, LAMBADA had substantial genuine contamination, but the clean subset scored within 0.5% of the full set. {#c-gpt3-lambada}
The PaLM report puts 10 of its 29 benchmarks in the two contaminated categories, and split each into a clean and a contaminated subset by whether at least 70% of the 8-grams of the question, prompt or target were seen at least once in training. {#c-palm-split}
They find an equal number of sets with positive and negative accuracy deltas on the clean subset, and conclude that contamination does not cause meaningful inflation of their reported results. {#c-palm-deltas}
These two studies are by the developers of the models, and a clean subset defined by n-grams misses paraphrased copies.

**A fresh test set.**
Zhang et al. find that many models show minimal signs of overfitting on GSM1k, and that all models broadly generalize to the new problems. {#c-gsm1k-generalize}
Across models, they find a Spearman rank correlation of 0.36 (p = 0.03) between the per-character log-likelihood of generating the GSM8K test set and the gap between GSM8K and GSM1k. {#c-gsm1k-corr}
They conclude that data contamination is likely not the full explanation for the overfitting. {#c-gsm1k-not-full}

**Hidden copies.**
With an LLM-based decontaminator, Yang et al. find that 8 to 18% of the HumanEval benchmark overlaps with pre-training sets such as RedPajama-Data-1T and StarCoder-Data. {#c-yang-humaneval}

## Mitigations

A mitigation either finds contamination or keeps it from happening; none removes it.

### Detecting contamination

Detection methods differ in what they need: the training data, the model's probabilities, or only its outputs.

**N-gram overlap with the training data.**
This needs the training corpus, and is the method of the GPT-3 and PaLM reports described above.
Brown et al. note that accurately detecting test contamination from internet-scale datasets is a new area of research without established best practices. {#c-gpt3-new-area}

**Min-K% Prob.**
Shi et al. build WikiMIA from Wikipedia event pages: pages created after 1 January 2023 are non-members, and pages created before 2017 are members of the training data of models such as LLaMA, GPT-NeoX and OPT. {#c-wikimia}
Their Min-K% Prob takes the k% of tokens of a text with the lowest probability under the model and averages their log-likelihood; a text is detected as seen when this value passes a threshold. {#c-mink}
The idea is that an unseen text is likely to contain a few outlier words with low probability, while a seen text is not. {#c-mink-hypothesis}
It needs only the token probabilities of the model, and they report a 7.4% improvement on WikiMIA over earlier methods. {#c-mink-gain}

**Guided prompting.**
Golchin and Surdeanu use a guided instruction: a prompt with the dataset name, the partition type and a random-length initial segment of a reference instance, which asks the model to complete it. {#c-guided}
An instance is flagged when the output matches the rest of the reference exactly or nearly. {#c-guided-flag}
A partition is flagged when the completions under the guided instruction overlap with the references significantly more than those under a general instruction without the dataset and partition name, or when a GPT-4 classifier marks several completions as exact or near-exact matches. {#c-guided-partition}
Their best method agrees with human experts in 92% to 100% of cases, across seven datasets with train and test or validation partitions. {#c-guided-accuracy}
It needs only the outputs of the model.

**Canary strings.**
A benchmark can carry a marker that tells corpus builders to leave it out.
All BIG-bench task files include a canary GUID string, meant to help researchers filter BIG-bench data out of web-scraped training corpora, so that models do not train on the test set. {#c-canary}
The task training_on_test_set uses the canary for post-hoc diagnosis of whether BIG-bench data was used in training. {#c-canary-diagnosis}
The authors ask that all publications with excerpts of BIG-bench tasks include the canary string. {#c-canary-publications}
A canary works only if corpus builders filter for it; [[gpqa|GPQA]] carries one as well.

### Preventing and limiting it

Each of these measures targets one route of contamination.

**Decontaminate the training data.**
N-gram filtering of the training data was the first measure, and it misses paraphrases and can fail in practice; the GPT-3 bug is a case of the second.
Yang et al. propose a stronger, LLM-based decontamination method and urge the community to adopt stronger decontamination approaches when using public benchmarks. {#c-yang-urge}

**Report clean subsets.**
Model reports can split each benchmark by overlap and report the score on the clean part, as GPT-3 and PaLM did; the result is only as good as the overlap test.

**Keep the test set private.**
Zhang et al. did not release GSM1k, to prevent the same contamination, and committed to release it when three open-source models with different pretrained lineages reach 95% accuracy on it, or in June 2025, whichever comes first. {#c-gsm1k-private}
Since proprietary models had to receive the data through an API, they also held out a few data points that passed all quality checks, in case GSM1k leaked that way. {#c-gsm1k-holdout}
Gating the data, as GPQA does, has a similar aim.

**Use problems that postdate the model.**
LiveCodeBench collects new problems over time from contests on LeetCode, AtCoder and CodeForces. {#c-lcb-design}
Its authors measure performance on problems released after a model's cutoff date, to estimate its performance on unseen problems. {#c-lcb-cutoff}
LiveBench adds and updates questions every month, from recent math competitions, arXiv papers, news articles and datasets, and scores answers automatically against objective ground-truth values. {#c-livebench}
Yang et al. call on the community to develop fresh one-time exams to evaluate models accurately. {#c-yang-fresh}
A dated cutoff helps only when the training cutoff of the model is known and true.

**Mark the data.**
Canary strings let corpus builders remove a benchmark, if they look for them.

**Rules for the experiments of this site.**
The experiments of this site follow rule M07 of its specification: no experiment trains on a split marked as test.
The datasets that experiments read are pinned in [registers/data-pins.yaml](../../../../registers/data-pins.yaml); the specification has that register mark the test splits, and `aik exp check` reject an experiment that trains on one.
At the time of writing, the one pin of that register, for MNIST, marks no split yet.

## Taxonomies

The generated list maps this page to the outside risk taxonomies when their crosswalks are built.

## Debates

**The question.**
Do contaminated training data inflate the benchmark scores that are reported for large language models by enough to change the conclusions drawn from them?
Inflation here means a higher score on the published test set than on an equivalent set the model cannot have seen.

The positions are listed in the order of the date of their first source.

**Position 1: the measured effect is small.**
The GPT-3 report (Brown et al., 2020) concludes from its clean subsets that GPT-3 is relatively insensitive to contamination. {#c-gpt3-insensitive}
The PaLM report (2022) reaches the same conclusion for its results, and Zhang et al. (2024) find little overfitting in many of the models they test.
The main argument is that most flagged overlaps are context or common phrases, not question and answer pairs, and that removing them changes scores little.
The evidence is the clean-subset results and the GSM1k results of the models that did not overfit, in the Evidence section.
The main objection is that a clean subset defined by n-gram overlap keeps paraphrased copies, which Yang et al. report can raise scores.

**Position 2: it can inflate scores and invalidate comparisons.**
Magar and Schwartz (2022) conclude that analysing web-scale datasets is needed to verify that progress in NLP comes from better language understanding and not better data exploitation. {#c-magar-conclusion}
Sainz et al. (2023) argue that contamination causes an overestimation of the performance of a contaminated model, which can lead to wrong scientific conclusions being published. {#c-sainz-overestimation}
The main argument is that leaks are hard to find from outside, and that a model which has seen test items can score above its ability.
The evidence is the drops on GSM1k for some model families, the drops of LiveCodeBench scores after training cutoffs, and the paraphrase results of Yang et al.
The main objection is that a drop on a new test set can also come from a slightly different distribution.
Zhang et al. ask that their results be read with the understanding that GSM8K and GSM1k are highly similar but not identically distributed. {#c-gsm1k-distribution}

**What would settle it.**
Paired scores on published and unseen test sets of the same distribution, for many models, with the training data of each known, would settle it.

## References
