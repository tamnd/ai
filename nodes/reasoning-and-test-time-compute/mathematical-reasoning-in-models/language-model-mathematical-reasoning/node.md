---
tag: NEW-language-model-mathematical-reasoning
kind: finding
slug: language-model-mathematical-reasoning
name: Mathematical reasoning ability of language models
aliases: [LLM math reasoning, math word problems, GSM8K results, MATH benchmark results, GSM-Symbolic, mathematical reasoning in LLMs, can LLMs do math]
area: reasoning-and-test-time-compute/mathematical-reasoning-in-models
level: core
summary: >-
  Language models went from solving few grade-school and competition math problems to most of them, yet small changes to names, numbers or irrelevant clauses still cut their accuracy.
related_to: [chain-of-thought-prompting, self-consistency, process-reward-model, data-contamination, language-model-planning, group-relative-policy-optimization]
history:
  year: 2021
  note: "Hendrycks et al. released MATH (March 2021) and Cobbe et al. GSM8K (October 2021); Wei et al. (January 2022) reported chain-of-thought prompting on GSM8K; Lewkowycz et al. (June 2022) trained Minerva on mathematical text; Shi et al. (ICML 2023) added irrelevant context; Dziri et al. (May 2023) studied multiplication; Lightman et al. (May 2023) and Shao et al. (February 2024) reported process supervision and DeepSeekMath; Zhang et al. (May 2024) built GSM1k and Mirzadeh et al. (October 2024, ICLR 2025) built GSM-Symbolic."
prereqs: [chain-of-thought-prompting]
same_as:
  arxiv: ["2110.14168", "2103.03874", "2206.14858", "2410.05229", "2302.00093", "2405.00332", "2305.18654"]
sources:
  - {paper: "Cobbe et al. (2021)", title: "Training Verifiers to Solve Math Word Problems", locator: "arXiv 2110.14168v2, pp. 1 to 3, 6 and 7", note: "GSM8K, the scaling extrapolation, the value of intermediate steps and verifiers"}
  - {paper: "Hendrycks et al. (2021)", title: "Measuring Mathematical Problem Solving With the MATH Dataset", locator: "NeurIPS 2021 Datasets and Benchmarks, arXiv 2103.03874v2, pp. 1 to 4", note: "the MATH dataset, early accuracies, human baselines and the scaling extrapolation"}
  - {paper: "Lewkowycz et al. (2022)", title: "Solving Quantitative Reasoning Problems with Language Models", locator: "arXiv 2206.14858v2, pp. 5, 8, 10 and 11", note: "Minerva, its training data, results with majority voting, false positives and the memorization check"}
  - {paper: "Mirzadeh et al. (2025)", title: "GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models", locator: "ICLR 2025, arXiv 2410.05229v2, pp. 1 and 5 to 9 and 18", note: "templated variants of GSM8K, variance across sets, difficulty levels and GSM-NoOp"}
  - {paper: "Shi et al. (2023)", title: "Large Language Models Can Be Easily Distracted by Irrelevant Context", locator: "ICML 2023, arXiv 2302.00093v3, pp. 2 and 4", note: "GSM-IC and the effect of prompting fixes"}
  - {paper: "Zhang et al. (2024)", title: "A Careful Examination of Large Language Model Performance on Grade School Arithmetic", locator: "arXiv 2405.00332v4, pp. 1 to 3", note: "GSM1k, overfitting and its relation to memorization"}
  - {paper: "Dziri et al. (2023)", title: "Faith and Fate: Limits of Transformers on Compositionality", locator: "arXiv 2305.18654v3, pp. 1 and 5 to 7 and 9", note: "multi-digit multiplication, in- and out-of-domain accuracy and the error compounding argument"}
  - {paper: "Wei et al. (2022)", title: "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models", locator: "arXiv 2201.11903v6, p. 20", note: "PaLM 540B on GSM8K with and without chain of thought"}
  - {paper: "Lightman et al. (2023)", title: "Let's Verify Step by Step", locator: "arXiv 2305.20050v1, p. 2", note: "the process-supervised reward model on MATH"}
  - {paper: "Shao et al. (2024)", title: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", locator: "arXiv 2402.03300v3, pp. 1 and 2", note: "continued pretraining on mathematical text and GRPO"}
claim: >-
  Language models have gone from solving few multi-step mathematics problems to solving most grade-school problems and many competition problems, through scale, mathematical pretraining data, written intermediate steps, sampling with voting or verification, and reinforcement learning; at the same time their accuracy falls when the names or numbers of a problem are changed, falls faster than a constant per-step error rate predicts as steps are added, and falls sharply when a clause that does not matter is inserted, which suggests that part of the measured skill is matching problems to familiar patterns.
first_reported: {work: hendrycks-2021-measuring-mathematical, note: "MATH, released in March 2021, found large language models of the time solving 3.0% to 6.9% of competition problems, and extrapolated an impractical model size for 40% accuracy"}
replications:
  - {work: cobbe-2021-training-verifiers, note: "grade-school word problems in GSM8K, which even the largest GPT-3 models of 2021 mostly failed"}
  - {work: mirzadeh-2025-gsm-symbolic, note: "25 open and closed models on templated variants of GSM8K with changed names and numbers, added clauses and irrelevant clauses"}
  - {work: shi-2023-distracted-irrelevant-context, note: "GPT-3-family models with several prompting techniques on GSM8K problems with one irrelevant sentence added"}
  - {work: zhang-2024-careful-examination-large, note: "open and closed models on 1205 new problems written to match GSM8K"}
  - {work: dziri-2023-faith-and-fate, note: "GPT-3, ChatGPT and GPT-4 on multi-digit multiplication, zero-shot, few-shot and fine-tuned"}
scale_range:
  - {range: "GPT-3 models from 6B to 175B parameters on GSM8K", work: cobbe-2021-training-verifiers}
  - {range: "PaLM and Minerva from 8B to 540B parameters on MATH, GSM8K and university STEM problems", work: lewkowycz-2022-solving-quantitative-reasoning}
  - {range: "25 models from 2B parameters to GPT-4o and o1 on 50 sets of 100 templated problems each", work: mirzadeh-2025-gsm-symbolic}
failures:
  - {work: lewkowycz-2022-solving-quantitative-reasoning, note: "changing the framing or numbers of problems Minerva had solved left its accuracy correlated with the original, which the authors read as minimal memorization"}
  - {work: zhang-2024-careful-examination-large, note: "even the models with the largest gaps on GSM1k still solve most new problems, so overfitting lowers measured skill without erasing it"}
  - {work: mirzadeh-2025-gsm-symbolic, note: "o1-preview keeps its accuracy across all difficulty levels of GSM-Symbolic, though it still drops on GSM-NoOp"}
explanations:
  - {name: "pattern matching to training problems", summary: "Models match a problem to solution patterns seen in training rather than following the logic of the problem, so changed numbers, extra clauses and irrelevant statements break the match.", sources: [mirzadeh-2025-gsm-symbolic, dziri-2023-faith-and-fate]}
  - {name: "compounding step errors", summary: "Each step of a long solution is a chance to err and errors are not repaired, so accuracy falls with the number of steps.", sources: [dziri-2023-faith-and-fate]}
  - {name: "contamination of public benchmarks", summary: "Benchmark problems or close copies in training data raise scores on the public set above scores on equivalent new problems.", sources: [zhang-2024-careful-examination-large, mirzadeh-2025-gsm-symbolic]}
---

The mathematical reasoning ability of language models is how well a model solves mathematics problems stated in words, from grade-school arithmetic stories to competition problems.
Measured on public benchmarks, it grew quickly: from a minority of grade-school problems in 2021 to most of them a few years later, and from a few percent of competition problems to around half.
Studies that change the problems tell a more mixed story.
Accuracy falls when only names or numbers change, falls faster than step counts alone explain as clauses are added, and falls sharply when an irrelevant sentence is inserted.
Both results hold at once: the models solve many problems they have not seen, and part of their measured skill rests on resemblance to problems they have seen.

## In plain words

A math word problem asks a model to read a short story, pick out the quantities, decide which operations to do and carry them out in order.
A typical grade-school problem takes two to eight such steps.

Early large models got most of these wrong.
Three things helped most: letting the model write out its steps before the answer, training on much more mathematical text, and sampling many solutions and taking the most common answer.
With these, scores on the standard test sets rose from a fifth to most problems.

The doubts come from tests that keep the problem but change its surface.
If a model has learned the arithmetic, a problem about Sophie with 31 apples should be as easy as the same problem about Ana with 27 pears.
For many models it is not; scores move by several points from one such variant set to another, and the original public problems tend to score above the variants.
Adding a sentence like "five of them are a bit smaller than average", which changes nothing, leads many models to subtract five.

## The finding

The benchmarks were built because models did badly on simple problems.
Language models can match human performance on many tasks, but they still struggle to robustly perform multi-step mathematical reasoning, and even very large transformer models fail to achieve high test performance on GSM8K despite the conceptual simplicity of the problem distribution. {#c-gsm-weak}
On MATH, large language models achieved accuracies ranging from 3.0% to 6.9%, though they achieve up to 15% accuracy on the easiest difficulty level and generate step-by-step solutions that are coherent and on-topic even when incorrect. {#c-math-low}

Within three years, models trained for mathematics did much better.
DeepSeekMath 7B, which continues pretraining with 120B math-related tokens sourced from Common Crawl, achieves 51.7% on the competition-level MATH benchmark without relying on external toolkits and voting techniques, and self-consistency over 64 samples achieves 60.9%. {#c-dsm-result}

Studies that perturb the problems found a different side of the same skill.
The performance of models declines when only the numerical values in the question are altered in the GSM-Symbolic benchmark, and their performance significantly deteriorates as the number of clauses in a question increases. {#c-gs-claim}
When a single clause that appears relevant to the question is added, performance drops significantly (up to 65%) across all models tested, even though the added clause does not contribute to the reasoning chain needed to reach the final answer. {#c-gs-noop-abs}
A fresh set of problems written to match GSM8K gave a similar warning and a limit to it.
On GSM1k, there are accuracy drops of up to 8% against GSM8k, with several families of models displaying evidence of systematic overfitting across almost all model sizes. {#c-1k-abs}
All models, including the most overfit ones, are still capable of successfully generalizing to new mathematical grade school problems, albeit occasionally at lower rates than their benchmark numbers would suggest. {#c-1k-gen}

## Evidence

**The benchmarks.**
GSM8K consists of 8.5K high quality grade school math problems created by human problem writers, segmented into 7.5K training problems and 1K test problems, which take between 2 and 8 steps to solve using basic arithmetic operations. {#c-gsm-data}
The MATH dataset consists of 12,500 problems from high school math competitions, each with a full step-by-step solution, and its answers are unique after normalization, allowing MATH to be scored with exact match. {#c-math-data}
MATH has 12,500 problems, 7,500 training and 5,000 test. {#c-math-split}
Both sets give a full written solution for every problem, so they can be used to train models on solutions as well as to test them.
A computer science PhD student who does not especially like mathematics attained approximately 40% on MATH, while a three-time IMO gold medalist attained 90%. {#c-math-human}

**Scale alone looked insufficient.**
Assuming a log-linear trend, a finetuned model with 10^16 parameters would be required to reach an 80% solve rate on GSM8K when using the full training set. {#c-gsm-extrap}
Assuming a log-linear scaling trend, models would need around 10^35 parameters to achieve 40% accuracy on MATH, which is impractical. {#c-math-scale}
These extrapolations were made from the models of 2021 and did not hold: the gains that followed came from other changes, listed below, rather than from parameters alone.

**Written steps.**
If a 6B model is finetuned to directly output the final answer without any intermediate steps, performance drops drastically from 20.6% to 5.2%. {#c-gsm-direct}
On GSM8K, PaLM 540B reaches a solve rate of 17.9 with standard prompting and 56.9 with chain-of-thought prompting. {#c-cot-gsm}
Training on solutions and prompting for steps are different interventions, and on MATH the early result went the other way for prompting.
Having models generate step-by-step solutions before producing an answer actually decreased accuracy relative to immediately outputting a final answer, while having models train on solutions increases relative accuracy by 10%. {#c-math-steps}

**Mathematical training data.**
Minerva models were trained on a dataset of 38.5B tokens from webpages filtered for mathematical content and from papers submitted to the arXiv preprint server. {#c-min-data}
PaLM 540B scores 8.8% on MATH, Minerva 540B 33.6% and Minerva 540B with majority voting 50.3%, and on GSM8k 56.5%, 58.8% and 78.5%. {#c-min-table}
PaLM and Minerva at the same size differ mainly in this additional training, so the MATH gain from 8.8% to 33.6% is largely due to the data.

**Sampling, voting and verification.**
Majority voting significantly improves performance over greedy decoding, saturates faster than pass@k, with 97% of the large k accuracy achieved at k = 64 for MATH, and performs significantly better than log-likelihood reranking. {#c-min-vote}
Compared to a finetuning baseline, the use of verifiers results in approximately the same performance boost as a 30x model size increase, and verifiers scale significantly better with increased data. {#c-gsm-verif}
A process-supervised reward model solves 78.2% of problems from a representative subset of the MATH test set. {#c-light}
While the pass@256 accuracy is 84.5% for the 62B model, false positives account for part of it; after removing false positives, the pass@256 accuracy is estimated to be bigger than 68%. {#c-min-fp}
The derivation compounding-errors-and-benchmark-noise works out why voting and pass@k behave as they do.
With samples right with probability 0.4, majority voting over 5, 15 and 41 samples is right with probability 0.7149, 0.9951 and 1.0000 when wrong answers never repeat, but 0.4039, 0.3411 and 0.2493 when one fixed wrong answer has probability 0.5. {#c-d-vote}
Voting helps when wrong answers are scattered and hurts when one wrong answer is more likely than the right one, so its gains on MATH mean that the models' errors are mostly scattered.
The unbiased pass@k estimate 1 − C(n − c, k)/C(n, k) with n = 256 is 0.0391 at k = 1 and 0.9470 at k = 64 for c = 10, and 0.25 at k = 64 for a single right sample. {#c-d-passk}
A high pass@k says that a right answer is somewhere among the samples, not that the model can find it; the false positives Minerva's authors found among right final answers are a reminder that a right answer is not a right solution.

**Reinforcement learning.**
Reinforcement learning with GRPO improves DeepSeekMath-Instruct both in-domain (GSM8K: 82.9% → 88.2%, MATH: 46.8% → 51.7%) and out-of-domain (CMATH: 84.6% → 88.8%). {#c-dsm-grpo}

**Changing names and numbers.**
GSM-Symbolic uses 100 templates and 50 samples per template, giving 50 datasets of 100 examples each, evaluated with Chain-of-Thought prompting with 8-shots and greedy decoding. {#c-gs-setup}
All models exhibit a non-negligible variance across different sets; for the Gemma2-9B the gap between the worst performance and the best performance is more than 12%. {#c-gs-gap}
The performance on the original GSM8K questions used as templates is often more than one standard deviation away from the center of the GSM-Symbolic distribution, frequently on the right side, which holds for 21 out of 25 models; one explanation could be data contamination. {#c-gs-right}
The variance is lower when changing names compared to numbers, and the original GSM8K accuracy of models is much closer to the center of the changed proper names distribution, in contrast to changed numbers or both. {#c-gs-names}
The derivation puts a scale on these spreads.
If template i is solved with probability p_i over its instances, accuracy on a set of one instance per template has standard deviation sqrt(Σ p_i (1 − p_i)) / 100, which is 0 when correctness depends only on the template and at most 4.07 points at a mean of 0.791; a spread of 3.0 points means Σ p_i (1 − p_i) = 9.0. {#c-d-spread}
A spread of a few points is what one expects if each template is solved with a probability between 0 and 1 and the model is not merely solving or failing each template; the more telling result is the direction of the gap between the original problems and the variants.

**Adding clauses.**
As the difficulty increases from GSM-M1 to GSM-Symb, GSM-P1 and GSM-P2, the performance decreases and the variance increases, and the rate of accuracy drop also increases as the difficulty increases. {#c-gs-levels}
If each step is right with probability p independently and wrong steps cannot be undone, a T-step problem is solved with probability p^T: at p = 0.95 this is 0.9025, 0.8145, 0.6634 and 0.4401 for T = 2, 4, 8 and 16. {#c-d-chain}
Under a constant per-step accuracy successive accuracies keep a constant ratio, but the GSM-Symbolic levels of Gemma2-9b-it (84.4, 79.1, 68.1, 41.8) give ratios 0.937, 0.861 and 0.614 and those of Phi-3.5-mini-instruct 0.937, 0.789 and 0.691, while GPT-4o-mini gives 0.991, 0.884 and 0.893. {#c-d-shape}
For two of the three models the drop per added level grows, which a fixed per-step error rate does not produce; the harder variants are new kinds of problems as well as longer ones.
The o1-preview model demonstrates robust performance across all levels of difficulty, but both o1-preview and o1-mini experience a significant performance drop on GSM-NoOp. {#c-gs-o1}

**Irrelevant information.**
GSM-IC adds to each base problem of an easy subset of GSM8K one irrelevant sentence that does not affect the solution; the full benchmark consists of 58,052 examples. {#c-ic-data}
Among the original problems that can be solved by baseline prompts with greedy decoding, no more than 18% can be consistently solved for all types of irrelevant information. {#c-ic-18}
Self-consistency improves the performance of all prompting techniques on GSM-IC, and the recall rate of the correct answer is as high as 99.7% with 20 samples per problem. {#c-ic-sc}
Adding irrelevant information to the exemplars shown in the prompt consistently boosts the performance, and the same holds for adding an instruction to ignore irrelevant context. {#c-ic-fix}
On GSM-NoOp there is a catastrophic performance decline across all tested models, with the Phi-3-mini model experiencing over a 65% drop and even stronger models such as o1-preview showing significant declines. {#c-gs-noop}
Models tend to convert statements to operations without truly understanding their meaning, for instance interpreting statements about discount as multiplication regardless of the context. {#c-gs-convert}

**New problems.**
GSM1k is a newly constructed collection of 1205 grade school level math problems designed to mirror GSM8k, created solely with human annotators, without assistance from any LLM or other synthetic data source. {#c-1k-data}
There is a positive relationship (Spearman's r2 = 0.36) between a model's probability of generating an example from GSM8k and its performance gap between GSM8k and GSM1k, suggesting that some models may have partially memorized GSM8k. {#c-1k-r2}
At an accuracy of 0.8 the standard error of the difference between accuracies on 1205 and 1319 independent problems is 1.59 points, so an 8-point drop is 5.02 standard errors and drops under about 3.2 points are within sampling noise. {#c-d-gap}
Drops of a few points on a new set are therefore not, on their own, evidence of overfitting; drops near 8 points are.

**Arithmetic itself.**
Off-the-shelf ChatGPT and GPT4 achieve only 55% and 59% accuracies on 3-digit by 3-digit multiplication. {#c-dz-mul}
GPT3 finetuned on about 1.8M multiplication pairs, or with question-scratchpad pairs, achieves near-perfect performance in-distribution but fails in generalizing to out-of-domain cases with wider or deeper computation graphs. {#c-dz-ft}
Relative information gain indicates that the first digits of the output of multiplication highly correlate with the first digits of each input number, and the last digits of the output rely solely on the last digits of each input number. {#c-dz-ig}
82.3% of the final correct answers for 4-digit by 2-digit multiplications, a setting unseen during training, had at least one error in the computation graph but still produced correct answers. {#c-dz-mem}
The j lowest digits of a product depend only on the j lowest digits of each factor, since a · b mod 10^j = (a mod 10^j)(b mod 10^j) mod 10^j; 200,000 random draws cover all 10^4 combinations behind the two lowest digits but only 0.1813 of the 10^6 pairs of three-digit numbers. {#c-d-digits}
The small PyTorch implementation trains a network on three-digit multiplication with this coverage: it learns the lowest and the highest digits, which follow from parts of the inputs it has seen in full or in approximation, and fails on the middle digits, which depend on the whole of both inputs.

## Scope and conditions

The results concern problems with a single numerical answer, checked by exact match after normalization.
Most perturbation studies use GSM8K and its variants, problems of two to eight arithmetic steps; competition mathematics and proofs have been perturbed far less.
Most were run with few-shot chain-of-thought prompting and greedy decoding, so they measure a single answer, not the best of many samples.

The model range matters.
The irrelevant-context study used GPT-3-family models of 2022 and 2023; GSM-Symbolic included models up to GPT-4o and o1-preview, and the reasoning models were the most robust to added clauses while still losing accuracy on irrelevant ones.
The new-problems study found that the gap between public and new problems varies widely across model families.

Exact-match scoring can also mislead in the other direction.
For twenty problems the 62B model answered correctly, modifying the framing or the numbers left accuracy before and after the modification correlated, with no clear bias in favor of the original formulation, suggestive of minimal memorization. {#c-min-mod}
Minerva's check used twenty problems, a small sample beside GSM-Symbolic's thousands of variants.

## Explanations

**Pattern matching to training problems.**
The GSM-Symbolic authors read their results as evidence that models match a problem to familiar reasoning steps rather than follow the problem's logic.
Higher variance for changed numbers than for changed names, original scores to the right of the variant distribution, and subtracting irrelevant quantities all fit that reading.
Under reasonable assumptions, the probability of incorrect predictions converges exponentially to about 1 for abstract compositional tasks, and the proofs apply to autoregressive LMs in general. {#c-dz-theory}
The multiplication results point the same way: a model fine-tuned on many products learns which parts of the inputs predict which parts of the output, and solves new sizes only by such shortcuts.

**Compounding step errors.**
Longer problems give more chances to err, and the derivation gives the size of that effect under independent errors.
It explains part of the drop with added clauses, not all of it, since the observed drops grow faster.

**Contamination of public benchmarks.**
Data contamination, the presence of test problems in training data, raises scores on the public set.
The correlation between a model's likelihood of producing GSM8K problems and its GSM1k gap supports this for some models, and the position of the original GSM8K scores in the GSM-Symbolic distributions fits it.
It cannot explain the NoOp results, which use new problems.

## Implications

For evaluation, a single score on a public benchmark is weak evidence of mathematical skill.
Templated variants, fresh problems written to the same specification, and problems with irrelevant clauses each test a different way the score can mislead, and differences between sets should be read against their sampling noise.

For training, the results favour methods that check the steps and not only the answer.
Process reward model training, which scores each step of a solution, and reinforcement learning with checked answers, as in group relative policy optimization, both target the gap between reaching the right number and solving the problem.

Self-consistency, the method of taking the majority answer over sampled solutions, and chain-of-thought prompting are the inference methods behind most of the measured gains; both are standard parts of how these benchmarks are now run.
The planning ability of language models raises a similar question in another domain, where disguised names separate retrieval from reasoning in the same way changed numbers do here.

## Open questions

How much of the gap closes with reasoning models trained by reinforcement learning is open: o1-preview resisted added clauses but not irrelevant ones.
Whether a model that fails on irrelevant clauses can be said to reason, or only to reason without knowing what to ignore, is partly a question of definition.
Perturbation studies of competition mathematics and of proofs, where answers cannot be checked by exact match, are few.

## History

Hendrycks and coauthors released MATH in March 2021, and Cobbe and coauthors at OpenAI released GSM8K in October 2021 with a study of trained verifiers.
Chain-of-thought prompting (January 2022) and Minerva (June 2022) raised scores on both.
Shi and coauthors added irrelevant context in February 2023 and Dziri and coauthors studied multiplication and compositional tasks in May 2023.
Process supervision (May 2023), DeepSeekMath and GRPO (February 2024) and later reasoning models raised scores further, while GSM1k (May 2024) and GSM-Symbolic (October 2024) tested whether the scores reflected the skill.

## References
