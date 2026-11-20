---
tag: NEW-hallucination
kind: finding
slug: hallucination
name: Hallucination
aliases: [confabulation, LLM hallucination, unfaithful generation, factual errors in generated text, imitative falsehoods]
area: language-models/hallucination
level: core
summary: >-
  Neural text generators state things that their input or the world does not support, fluently and often; part of the cause is statistical, part is how models are trained and graded.
related_to: [training-data-memorization, expected-calibration-error, reinforcement-learning-from-human-feedback, neural-scaling-laws, data-contamination, nucleus-sampling, perplexity]
history:
  year: 2020
  note: "Maynez et al. posted their human study of hallucinations in abstractive summarization as arXiv 2005.00661 in May 2020; Lin et al. (ACL 2022), Kadavath et al. (2022), Kuhn et al. (ICLR 2023), Manakul et al. and Min et al. (EMNLP 2023), Kalai and Vempala (2023), Gekhman et al. (EMNLP 2024) and Kalai et al. (2025) measured it in language models, detected it and explained it."
prereqs: [decoder-only-transformer, perplexity]
same_as:
  arxiv: ["2005.00661", "2109.07958", "2207.05221", "2303.08896", "2305.14251", "2405.05904", "2311.14648", "2509.04664", "2302.09664"]
sources:
  - {paper: "Maynez, Narayan, Bohnet and McDonald (2020)", title: "On Faithfulness and Factuality in Abstractive Summarization", locator: "ACL 2020, pp. 1906 to 1919, PDF pp. 1 to 8", note: "the definitions of intrinsic, extrinsic and factual hallucination and the human study on XSum"}
  - {paper: "Lin, Hilton and Evans (2022)", title: "TruthfulQA: Measuring How Models Mimic Human Falsehoods", locator: "ACL 2022, pp. 3214 to 3252, PDF pp. 1, 3 and 4 and Sections 4.1 to 4.4", note: "imitative falsehoods, the benchmark and the inverse scaling of truthfulness"}
  - {paper: "Min, Krishna, Lyu, Lewis, Yih, Koh, Iyyer, Zettlemoyer and Hajishirzi (2023)", title: "FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation", locator: "EMNLP 2023, pp. 12076 to 12100, PDF pp. 1, 4 and 5", note: "factual precision of biographies and its fall with entity rarity"}
  - {paper: "Manakul, Liusie and Gales (2023)", title: "SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models", locator: "EMNLP 2023, pp. 9004 to 9017, PDF pp. 1 and 5 to 7", note: "detection from the agreement of sampled responses"}
  - {paper: "Kuhn, Gal and Farquhar (2023)", title: "Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation", locator: "ICLR 2023, arXiv 2302.09664v3, pp. 1 and 5", note: "semantic entropy"}
  - {paper: "Kadavath, Conerly, Askell, Henighan, Drain, Perez and others (2022)", title: "Language Models (Mostly) Know What They Know", locator: "arXiv 2207.05221v4, pp. 1, 3 and 4", note: "calibration, P(True) and P(IK)"}
  - {paper: "Kalai and Vempala (2024)", title: "Calibrated Language Models Must Hallucinate", locator: "arXiv 2311.14648v3, pp. 1, 4 and 5", note: "the lower bound for arbitrary facts and the monofact estimator"}
  - {paper: "Kalai, Nachum, Vempala and Zhang (2025)", title: "Why Language Models Hallucinate", locator: "arXiv 2509.04664v1, pp. 1, 3, 4 and 13", note: "the reduction from binary classification and the argument about binary grading"}
  - {paper: "Gekhman, Yona, Aharoni, Eyal, Feder, Reichart and Herzig (2024)", title: "Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?", locator: "EMNLP 2024, pp. 7765 to 7784, PDF pp. 1, 2 and 4", note: "fine-tuning on facts the model does not know"}
claim: >-
  Neural text generators, from summarizers to large language models, produce content that is not supported by their input or is false, in a large share of their outputs; the rate depends on how often the facts involved occur in the training data, and is not removed by scaling alone.
first_reported: {work: maynez-2020-faithfulness-factuality-summarization, note: "a human study of five summarizers on XSum found hallucinated content in more than 70% of single-sentence summaries, most of it erroneous; the word had been used for unsupported output of neural generators before, as their citations show"}
replications:
  - {work: lin-2022-truthfulqa, note: "GPT-3 and other models answer TruthfulQA truthfully on at most 58% of questions against 94% for a person, often repeating popular misconceptions"}
  - {work: min-2023-factscore, note: "ChatGPT biographies have 58.3% of their atomic facts supported by Wikipedia, and every model tested is less precise on rarer people"}
  - {work: manakul-2023-selfcheckgpt, note: "1392 of 1908 sentences of GPT-3 biographies were labelled major- or minor-inaccurate"}
scale_range:
  - {range: "summarizers with and without BERT pretraining on XSum", work: maynez-2020-faithfulness-factuality-summarization}
  - {range: "GPT-Neo/J 125M to 6B, GPT-2, GPT-3 up to 175B, UnifiedQA", work: lin-2022-truthfulqa}
  - {range: "Anthropic models up to 52B parameters", work: kadavath-2022-language-models-know}
  - {range: "OPT 2.7B to 30B", work: kuhn-2023-semantic-uncertainty}
  - {range: "InstructGPT, ChatGPT, PerplexityAI and 13 later models", work: min-2023-factscore}
failures:
  - {work: lin-2022-truthfulqa, note: "on matched control questions truthfulness improves with model size, so the inverse scaling belongs to questions that invite imitative falsehoods, not to questions in general"}
  - {work: kadavath-2022-language-models-know, note: "large models are calibrated on multiple-choice questions and can be trained to predict whether they know an answer, so a model's own signals catch part of what it gets wrong"}
  - {work: kalai-2024-calibrated-must-hallucinate, note: "the statistical lower bound covers only arbitrary facts; for facts seen more than once and for systematic facts it gives no reason to hallucinate"}
explanations:
  - {name: "imitative falsehoods", summary: "A model trained to imitate web text reproduces the false beliefs common in that text, and a model that fits its data better reproduces more of them.", sources: [lin-2022-truthfulqa]}
  - {name: "the missing mass", summary: "A calibrated model must put probability on facts it never saw, about as much as the share of facts seen once, and it cannot tell which of them are true.", sources: [kalai-2024-calibrated-must-hallucinate, kalai-2025-why-language-models-hallucinate]}
  - {name: "grading that rewards guessing", summary: "Most benchmarks give no credit for abstaining, so a model tuned to score well guesses when unsure instead of saying it does not know.", sources: [kalai-2025-why-language-models-hallucinate]}
  - {name: "fine-tuning on unknown facts", summary: "Fine-tuning on facts the model did not learn in pretraining teaches it to answer without knowledge, which raises hallucination on other questions.", sources: [gekhman-2024-fine-tuning-new-knowledge]}
  - {name: "noisy training pairs", summary: "When training targets contain information absent from the inputs, a conditional model learns to add unsupported content.", sources: [maynez-2020-faithfulness-factuality-summarization]}
---

A hallucination is content generated by a model that is not supported by its input or is false.
Maynez et al. analyzed limitations of neural text generation models for abstractive document summarization and found that these models are highly prone to hallucinate content that is unfaithful to the input document. {#c-my-prone}
In language models answering questions the input is only a prompt, so the test becomes whether the answer is true.
Lin et al. call false answers that a model's training objective incentivizes imitative falsehoods: for GPT-3 a false answer is an imitative falsehood if it has high likelihood on GPT-3's training distribution. {#c-tq-imitative}
Kalai and Vempala prove that there is an inherent statistical lower bound on the rate that pretrained language models hallucinate certain types of facts, having nothing to do with the transformer LM architecture or data quality. {#c-kv-bound}

## In plain words

A language model writes the most plausible continuation of its text.
Plausible and true usually coincide, but not always: a model asked about a little-known person will produce a birth year, a university and a list of works, in the same confident tone as for a famous one, and some of them will be made up.
The word hallucination, borrowed loosely from psychology, names this.
Some researchers prefer confabulation, since the model is not perceiving anything.

Two kinds of failure share the name.
A summary can add a detail that is not in the document it summarizes; that is a failure of faithfulness, whether or not the detail happens to be true.
An answer to a question can be false; that is a failure of factuality.
The research on this page measures both, finds them common in every model tested, and offers several reasons, none of which is "the model is broken": some hallucination follows from statistics alone, and some from what training and grading reward.

## The finding

The first large study defined the terms that later work uses.
Maynez et al. define a summary as being hallucinated if it has a span of words that is not supported by the input document, and do not categorize paraphrases or any information that can be inferred from the document as hallucinations. {#c-my-def}
Intrinsic hallucinations use terms or concepts from the document but misrepresent information from the document, while extrinsic hallucinations are model generations that ignore the source material altogether. {#c-my-kinds}
An extrinsic addition can still be true.
A summary $S$ of a document $D$ contains a factual hallucination if it contains information not found in $D$ that is factually correct. {#c-my-factual}
So faithfulness is judged against the input and factuality against the world, and a summary can fail one test and pass the other.

## Evidence

The generated list gives each source with its label and its role.
The lines of evidence are these.

**Summarization.**
Maynez et al. had people read model summaries of BBC articles from the XSum dataset, where each reference summary is a single sentence.
Maynez et al. randomly sampled 500 articles from the test set and elicited judgments from three different annotators for each of 2500 (500x5) document-summary pairs. {#c-my-setup}
The five summaries per article came from four systems and the reference.
Intrinsic and extrinsic hallucinations happen frequently, in more than 70% of single-sentence summaries, and over 90% of extrinsic hallucinations were erroneous. {#c-my-70}
The references themselves were not clean.
The BBC gold summaries often have extrinsic hallucinations due to the dataset artifact that gold summaries are introductory sentences prefacing each article. {#c-my-gold}
The models differed from the references in the other kind.
The percentage of system summaries with intrinsic hallucination was much higher than in gold summaries (7.4% vs others), which revealed the models' tendency to misrepresent information in the document due to the lack of document-level understanding and inference. {#c-my-intrinsic}
Pretraining helped.
In total, 34.7% of the BertS2S abstracts were faithful (26.9%) and/or factual (+7.8%), 7.4% absolute better than the next-best model (PtGen), and even so over 90% of BertS2S hallucinations are erroneous. {#c-my-bert}
For measuring it without people, Maynez et al. compared automatic metrics with the human labels.
Textual entailment scores are best correlated with both faithful (moderate, $0.40 \le |r_s| \le 0.59$) and factual (weak, $0.20 \le |r_s| \le 0.39$) human scores. {#c-my-entail}

**Questions that invite false answers.**
Lin et al. (2022) built a benchmark to test truthfulness rather than knowledge.
The TruthfulQA benchmark comprises 817 questions that span 38 categories, including health, law, finance and politics, crafted so that some humans would answer falsely due to a false belief or misconception. {#c-tq-bench}
Lin et al. define an answer to a question as truthful iff it avoids asserting a false statement, so a model is perfectly truthful if it answers 'No comment' for every question, and in practice they want answers that are both truthful and informative. {#c-tq-truthful}
The human participant produced 94% true answers, 87% of them both true and informative, while the model with the most true answers (GPT-3-175B with helpful prompt) produced 58% true answers and 21% true and informative answers, and gave false and informative answers 42% of the time (compared to 6% for the human participant). {#c-tq-best}
The models got worse as they grew.
Larger models generally do worse than smaller models in the same family: the GPT-Neo/J model with the most parameters is 17% less truthful than a model 60x smaller. {#c-tq-inverse}
In the multiple-choice task GPT-Neo/J 6B was 12% less truthful than GPT-Neo/J 125M, and no models significantly outperformed random guessing. {#c-tq-mc}
While larger models were less truthful, they were more informative. {#c-tq-informative}
Scoring thousands of answers by hand is slow, so Lin et al. trained a judge.
The fine-tuned GPT-judge model is able to predict human evaluations of truthfulness with 90-96% validation accuracy. {#c-tq-judge}

**Long answers.**
A paragraph can mix true and false statements, so a single label for it says little.
FActScore breaks a generation into a series of atomic facts, short statements that each contain one piece of information, and computes the percentage of atomic facts supported by a reliable knowledge source. {#c-fs-def}
Min et al. (2023) asked models for biographies and checked each atomic fact against Wikipedia by hand.
On people biographies InstructGPT and ChatGPT achieve FActScores of 42.5% and 58.3%, and PerplexityAI, which uses a commercial search engine, attains a FActScore of 71.5%. {#c-fs-scores}
ChatGPT and PerplexityAI often abstain from answering, which presumably improves their factual precision. {#c-fs-abstain}
An automated model that estimates FActScore using retrieval and a strong language model has less than a 2% error rate, and Min et al. use it to evaluate 6,500 generations from 13 recent LMs that would have cost \$26K if evaluated by humans. {#c-fs-auto}
Precision depended on whom the biography was about.
There is a notable decrease in FActScore as the rarity of entities increases, consistently across all the LMs evaluated. {#c-fs-rare}
Retrieval did not remove the effect.
The FActScore of PerplexityAI still significantly drops as entities are rarer, with a relative drop of 50% and 64% at the atomic level and sentence level. {#c-fs-perplexity}
Across all LMs, the later part of the generation has significantly worse precision. {#c-fs-later}

**Detection from the model's own samples.**
A model that knows a fact tends to state it the same way each time it is asked; a model that is making something up tends to make up something different each time.
SelfCheckGPT leverages the idea that if an LLM has knowledge of a given concept, sampled responses are likely to be similar and contain consistent facts, whereas for hallucinated facts stochastically sampled responses are likely to diverge and contradict one another. {#c-sc-idea}
SelfCheckGPT is a sampling-based approach that can be used to fact-check the responses of black-box models in a zero-resource fashion, without an external database. {#c-sc-zero}
Manakul et al. (2023) had GPT-3 write passages about people from the WikiBio dataset and labelled every sentence by hand.
Of the 1908 annotated sentences, 761 (39.9%) were labelled major-inaccurate, 631 (33.1%) minor-inaccurate, and 516 (27.0%) accurate. {#c-sc-labels}
The model's token probabilities already carry a signal.
Probability measures from the LLM generating the texts are strong baselines for assessing factuality: factual sentences can be identified with an AUC-PR of 53.97, significantly better than the random baseline of 27.04, with the AUC-PR for hallucination detection also increasing from 72.96 to 83.21. {#c-sc-prob}
Comparing samples did better.
SelfCheckGPT-Prompt considerably outperforms the grey-box approaches, including GPT-3's output probabilities, as well as other black-box approaches. {#c-sc-prompt}
Agreement between samples has to be judged by meaning, not wording.
Kuhn et al. introduce semantic entropy, an entropy which incorporates linguistic invariances created by shared meanings, because different sentences can mean the same thing. {#c-ku-def}
A sequence means the same thing as a second sequence if and only if they entail each other, and the classifier returns equivalent if and only if both directions were entailment. {#c-ku-entail}
Semantic entropy is the entropy of the model's probability over these clusters of answers rather than over strings.
Semantic entropy is more predictive of model accuracy on question answering data sets than comparable baselines. {#c-ku-pred}

**What models know about their own knowledge.**
Kadavath et al. (2022) asked whether a model can tell when it is likely to be wrong.
Larger models are well-calibrated on diverse multiple choice and true/false questions when they are provided in the right format. {#c-kd-calib}
Asking models to first propose answers and then to evaluate the probability P(True) that their answers are correct gives encouraging performance, calibration, and scaling on a diverse array of tasks. {#c-kd-ptrue}
As model size and capabilities increase, models improve at self-evaluation, which suggests that verification improves faster than generation quality. {#c-kd-verify}
They also trained models to predict, from the question alone, whether they would answer it correctly.
Models perform well at predicting P(IK), the probability that 'I know' the answer to a question, and partially generalize across tasks, though they struggle with calibration of P(IK) on new tasks. {#c-kd-pik}
The predicted P(IK) probabilities increase appropriately in the presence of relevant source materials in the context, and in the presence of hints towards the solution of mathematical word problems. {#c-kd-source}
The calibration is fragile.
Replacing an option with 'none of the above' reduces accuracy and calibration significantly. {#c-kd-nota}
RLHF policies naively seem miscalibrated, but with a simple temperature adjustment they become fairly well-calibrated on several evaluations. {#c-kd-rlhf}
So a model often has the information needed to flag a hallucination; whether it says so depends on how it is asked and trained.

## Scope and conditions

**Faithful and factual.**
The summarization results count unsupported content, true or not; the question-answering results count false content, supported or not.
The two rates are not comparable, and a figure for one says nothing about the other.

**The reference.**
In the XSum study the reference summaries hallucinate too, because the dataset pairs each article with a sentence written to introduce it rather than summarize it.
A model trained on such pairs learns to add content, so part of what was measured is the dataset.
FActScore uses Wikipedia as its knowledge source, so a true fact missing from Wikipedia counts as unsupported.

**Abstention.**
Under the TruthfulQA definition "No comment" is truthful, and in FActScore an abstention is not scored.
A model can raise either number by declining to answer more often, which is why Lin et al. also report informativeness.

**Automatic judges.**
GPT-judge, the FActScore estimator and entailment classifiers are models with their own error rates.
Entailment is a referenceless measure, so it can easily be gamed, since the first sentence of any source document is always entailed by the whole document, and entailment-based measures for evaluation need to be coupled with reference-based measures like ROUGE. {#c-my-gamed}

**Proxies.**
Gekhman et al. measure test performance as a proxy for hallucinations, since in a closed-book QA setup with disjoint train/test splits the model has to use its pre-existing knowledge to answer test questions. {#c-gk-proxy}
A wrong answer to a closed-book question is a hallucination only in this sense: the model answers instead of declining.

## Explanations

**Noisy training pairs.**
The claim: a conditional model trained on targets that contain information absent from the inputs learns to add such information.
Models are usually agnostic to the noises or artifacts of the training data, such as reference divergence, making them vulnerable to hallucinations. {#c-my-noise}
The evidence: the extrinsic hallucinations of the XSum references, matched by those of every model trained on them.
The consequence: cleaning the training pairs, or training on faithful ones, should reduce unfaithful output in conditional generation; it does not address factual errors of an unconditioned model.

**Imitative falsehoods.**
The claim: a model trained to imitate human text imitates human errors, and fits them better as it grows.
One explanation of the inverse scaling on TruthfulQA is that larger models produce more imitative falsehoods because they are better at learning the training distribution; another is that the questions adversarially exploit weaknesses in larger models not arising from imitation of the training distribution. {#c-tq-two}
The evidence favours the first.
On matched control questions, made by editing 1-3 words of a TruthfulQA question to turn it into a straightforward trivia or common-sense question, truthfulness improves with model size for all model families. {#c-tq-controls}
The consequence: more data of the same kind will not fix it.
Lin et al. suggest that scaling up models alone is less promising for improving truthfulness than fine-tuning using training objectives other than imitation of text from the web. {#c-tq-scaling}

**The missing mass.**
The claim: a model that predicts well must sometimes produce facts it has never seen, and for facts with no pattern it cannot tell the true ones from the false.
If the maximum probability of any fact is bounded, the probability of generating a hallucination is close to the fraction of facts that occur exactly once in the training data, a Good-Turing estimate, even assuming ideal training data without errors. {#c-kv-goodturing}
The Good-Turing estimate of the missing mass is the fraction of samples that appear exactly once in the training data, which Kalai and Vempala call the MonoFacts estimator. {#c-kv-mf}
Under a regularity assumption on the factoid distribution, the simplest bound of Kalai and Vempala implies that for any algorithm, with probability at least 99% over training sets, the hallucination rate is at least the monofact estimate minus a miscalibration rate, minus a term proportional to the number of facts over the number of possible hallucinations, minus a term proportional to one over the square root of $n$. {#c-kv-corollary}
Their example is gossip about who ate what, when, where and why.
If half of the 5W (Who-Ate-What-When-Where-Why) posts occur exactly once, a calibrated-factoid model would hallucinate on about half of its generations on 5W factoids. {#c-kv-5w}
The derivation hallucination-arithmetic works through the estimator and a toy generator.
For $n$ independent draws from a distribution $p$ over facts, the expected fraction of draws whose fact appears exactly once exceeds the expected missing mass by $\sum_i p_i^2 (1 - p_i)^{n-1}$, which lies between 0 and $1/n$. {#c-ha-gap}
A generator that reproduces the observed facts at their observed frequencies and spreads a mass equal to the monofact fraction $\mathrm{MF}$ evenly over $U$ unseen true facts and $M$ false factoids hallucinates at the rate $\mathrm{MF} \cdot M / (M + U)$, which tends to $\mathrm{MF}$ when $M$ is much larger than $U$. {#c-ha-toy}
With $N$ equally likely facts and $n = N$ draws, the expected missing mass is $(1 - 1/N)^N$, about $1/e \approx 0.368$ for large $N$; a generator that reproduces the facts it has seen and spreads the missing mass evenly over the unseen true facts and a hundred false factoids for every true fact hallucinates on about 36.7% of its generations. {#c-ha-uniform}
The bound has limits that its authors state.
There is no statistical reason that pretraining will lead to hallucination on facts that tend to appear more than once in the training data, like references to publications such as articles and books, or on systematic facts like arithmetic calculations. {#c-kv-once}
Kalai et al. (2025) put the same result in terms of classification.
Kalai et al. argue that hallucinations originate as errors in binary classification: if incorrect statements cannot be distinguished from facts, then hallucinations in pretrained language models will arise through natural statistical pressures. {#c-kz-binary}
Treating the language model as a classifier for the Is-It-Valid binary classification problem gives the relationship (generative error rate) $\gtrsim 2 \cdot$ (IIV misclassification rate). {#c-kz-iiv}
The reduction recovers the bound that the hallucination rate after pretraining should be at least the fraction of training facts that appear once: if 20% of birthday facts appear exactly once in the pretraining data, one expects base models to hallucinate on at least 20% of birthday facts. {#c-kz-singleton}
The consequence: pretraining alone leaves a floor.
Models pretrained to be sufficiently good predictors, that is calibrated, may require post-training to mitigate hallucinations on the type of arbitrary facts that tend to appear once in the training set. {#c-kv-post}
The FActScore result, lower precision for rarer people, fits this account, since facts about rare people are the ones seen once or never.

**Grading that rewards guessing.**
The claim: post-training could teach a model to say "I don't know", but the benchmarks it is tuned for give no credit for that.
Kalai et al. argue that language models hallucinate because the training and evaluation procedures reward guessing over acknowledging uncertainty. {#c-kz-guess}
If Model A correctly signals uncertainty and never hallucinates, and Model B is similar but never indicates uncertainty and always guesses when unsure, Model B will outperform A under 0-1 scoring, the basis of most current benchmarks. {#c-kz-ab}
For any distribution over binary graders, the optimal responses to a prompt are not abstentions. {#c-kz-obs}
A short meta-evaluation of popular evaluations finds that the vast majority have binary grading. {#c-kz-table}
The same derivation puts numbers on it.
Under binary grading, a model that guesses on the questions where it is unsure, with probability $q_j$ of being right on question $j$, scores $\sum_j q_j$ more than a model that abstains on them, and makes $\sum_j (1 - q_j)$ more errors. {#c-ha-ab}
Their remedy is to state the cost of an error in the instructions.
Kalai et al. propose that evaluations state confidence targets in their instructions: answer only if you are $> t$ confident, since mistakes are penalized $t/(1 - t)$ points, while correct answers receive 1 point, and an answer of 'I don't know' receives 0 points; a threshold of $t = 0$ corresponds to binary grading. {#c-kz-target}
Under a confidence target $t$, where a correct answer earns 1 point, an error costs $t/(1 - t)$ points and an abstention earns 0, answering with probability $q$ of being right has expected score $(q - t)/(1 - t)$, which is positive exactly when $q > t$; with $t = 0$ answering is never worse than abstaining. {#c-ha-score}
The penalty $t/(1 - t)$ is 1 for $t = 0.5$, 3 for $t = 0.75$ and 9 for $t = 0.9$; a penalty of 2 corresponds to $t = 2/3$. {#c-ha-penalty}
Kalai et al. list 2 as the penalty for $t = 0.75$, which does not match their own formula.
The consequence: the rule pays off only for a model that knows its own $q$, which is what the calibration results of Kadavath et al. measure.

**Fine-tuning on unknown facts.**
The claim: fine-tuning a model on facts it did not learn in pretraining teaches it to answer without knowledge.
Gekhman et al. (2024) fine-tuned PaLM 2-S on closed-book questions, varying the share of questions whose answers the model did not know.
Fine-tuning examples that introduce new knowledge are learned significantly slower than those consistent with the model's knowledge, and as the examples with new knowledge are eventually learned, they linearly increase the model's tendency to hallucinate. {#c-gk-slow}
Early-stopping can minimize the risk of the hallucinations caused by fitting the Unknown examples, since they primarily emerge in later training stages as a form of overfitting. {#c-gk-early}
Gekhman et al. take their results to support the view that large language models mostly acquire factual knowledge through pre-training, whereas fine-tuning teaches them to use it more efficiently. {#c-gk-view}
The consequence: fine-tuning data should be filtered against what the model already knows.

These accounts fit together.
The missing mass explains why a pretrained model has a floor of errors on rare arbitrary facts; imitative falsehoods explain errors on facts that are common but commonly wrong; grading and fine-tuning explain why post-training does not teach the model to abstain where it should.

## Implications

The implications below are those that the sources draw.
Maynez et al. propose entailment as a measure of faithfulness, coupled with reference-based metrics.
Lin et al. recommend training objectives other than imitation of web text, and Kalai et al. recommend changing the scoring of the benchmarks that dominate leaderboards rather than adding more hallucination tests.
Manakul et al. and Kuhn et al. offer detectors that need only samples from the model, and Kadavath et al. find that a model can be trained to report whether it knows an answer.
Gekhman et al. recommend early stopping and filtering fine-tuning data for facts the model does not know.

The training data memorization page covers the opposite behaviour, a model reproducing text it saw; the same frequency of a fact in the training data decides both.
The expected calibration error page defines the calibration measure behind the results of Kadavath et al. and the assumption of Kalai and Vempala.
The reinforcement learning from human feedback page describes the post-training on which Kadavath et al. report calibration after a temperature adjustment, and which Kalai et al. argue inherits the incentives of binary grading.
The neural scaling laws page gives the smooth fall of loss with scale, which TruthfulQA shows need not mean more truthful answers.
The data contamination page treats the reverse risk in evaluation, a benchmark answered from memory.
The nucleus sampling page describes the sampling rule whose randomness SelfCheckGPT exploits, and the perplexity page the likelihood measure that a fluent hallucination can score well on.

## Open questions

How far the lower bound of Kalai and Vempala is from the hallucination rate of deployed models, which are post-trained and not calibrated in their sense, is not measured.
Whether models can be trained to abstain exactly when they would be wrong, without losing informative answers, is open; TruthfulQA shows the trade-off and P(IK) shows the signal exists.
Whether retrieval removes the dependence on entity frequency is open; the FActScore results for PerplexityAI suggest it does not fully.
How to detect hallucinations in long reasoning, where one false step leads to a wrong conclusion, is less studied than detection in short answers.

## History

Maynez et al. posted "On Faithfulness and Factuality in Abstractive Summarization" as arXiv 2005.00661 in May 2020, and it appeared at ACL 2020.
Earlier work on data-to-text generation and summarization had already reported unsupported content, as their citations show.
Lin et al. posted TruthfulQA as arXiv 2109.07958 in September 2021, and it appeared at ACL 2022.
Kadavath et al. posted "Language Models (Mostly) Know What They Know" as arXiv 2207.05221 in July 2022.
Kuhn et al. posted semantic entropy as arXiv 2302.09664 in February 2023, and it appeared at ICLR 2023.
Manakul et al. posted SelfCheckGPT as arXiv 2303.08896 in March 2023 and Min et al. posted FActScore as arXiv 2305.14251 in May 2023; both appeared at EMNLP 2023.
Kalai and Vempala posted "Calibrated Language Models Must Hallucinate" as arXiv 2311.14648 in November 2023.
Gekhman et al. posted their study of fine-tuning as arXiv 2405.05904 in May 2024, and it appeared at EMNLP 2024.
Kalai et al. posted "Why Language Models Hallucinate" as arXiv 2509.04664 in September 2025.

## References
