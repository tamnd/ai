---
tag: WFY2QB
kind: method
slug: chain-of-thought-prompting
name: Chain-of-thought prompting
aliases: [chain of thought, CoT, CoT prompting, few-shot chain of thought, Few-shot-CoT, zero-shot chain of thought, Zero-shot-CoT, "let's think step by step", step-by-step prompting]
area: reasoning-and-test-time-compute/chain-of-thought
level: core
summary: >-
  Prompting a large language model with worked examples, or with "Let's think step by step", so that it writes reasoning steps before its answer; the steps need not show why it answered.
purpose: reasoning
uses: [beam-search@al, ensemble-learning@al, ancestral-sampling@al, binomial-distribution@m2, hoeffdings-inequality@m2]
related_to: [beam-search@al, ensemble-learning@al, ancestral-sampling@al, weak-law-of-large-numbers@m2, independent-and-identically-distributed-random-variables@m2]
hyperparameters:
  - {name: exemplars, meaning: "the number of worked examples, each a triple of question, chain of thought and answer, put in front of the test question; zero for the trigger-only variant", typical: "8 for math word problems (4 for AQuA) in Wei et al.; 0 in Zero-shot-CoT"}
  - {name: trigger, meaning: "the sentence after the question that starts the reasoning in the zero-shot variant", typical: "Let's think step by step."}
  - {name: answer_trigger, meaning: "the text that marks or extracts the final answer, which the parser reads", typical: "The answer is (few-shot); Therefore, the answer (arabic numerals) is (Zero-shot-CoT)"}
  - {name: decoding, meaning: "greedy decoding of one chain, or sampling of m chains followed by a vote (self-consistency)", typical: "greedy in Wei et al. and Kojima et al.; sampling in Wang et al."}
  - {name: samples, meaning: "the number m of sampled chains of thought per question in self-consistency", typical: "40 in Wang et al.; 5 to 10 for most of the gain"}
  - {name: temperature, meaning: "the sampling temperature T of the chains in self-consistency, with optional top-k truncation", typical: "T = 0.5 to 0.7, top-k 40"}
cost:
  train: "none; the model is used as it is, and the cost is writing the exemplars"
  inference: "the chain of thought adds its tokens to every answer, and self-consistency multiplies the generation by the number of samples m"
history:
  year: 2022
  people: [Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, Denny Zhou]
  note: "Wei et al. (arXiv 2201.11903, NeurIPS 2022) named chain-of-thought prompting and studied it across model scale; Kojima et al. (arXiv 2205.11916, NeurIPS 2022) removed the exemplars with Let's think step by step; Wang et al. (arXiv 2203.11171, ICLR 2023) replaced greedy decoding with a vote over sampled chains; Turpin et al. (arXiv 2305.04388, NeurIPS 2023) and Lanham et al. (arXiv 2307.13702) tested whether the written chain is a faithful account of the answer."
prereqs: [beam-search@al, ancestral-sampling@al, binomial-distribution@m2, independent-and-identically-distributed-random-variables@m2]
same_as:
  wikidata: null
  wikipedia: null
  arxiv: ["2201.11903", "2205.11916", "2203.11171", "2305.04388", "2307.13702"]
sources:
  - {paper: "Wei et al. (2022)", title: "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models", locator: "arXiv 2201.11903v6, pp. 1 to 9, 16 and 18", note: "the definition, the benchmarks, emergence with scale, the ablations, robustness, the limitations and the error analysis"}
  - {paper: "Kojima et al. (2022)", title: "Large Language Models are Zero-Shot Reasoners", locator: "arXiv 2205.11916v4, pp. 1 to 9", note: "Zero-shot-CoT, the two-stage prompt, answer cleansing, scale and the template study"}
  - {paper: "Wang et al. (2023)", title: "Self-Consistency Improves Chain of Thought Reasoning in Language Models", locator: "arXiv 2203.11171v4, pp. 1 to 9", note: "the vote over sampled reasoning paths, the aggregation strategies, the sampling settings, scale and cost"}
  - {paper: "Turpin et al. (2023)", title: "Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting", locator: "arXiv 2305.04388v2, pp. 1 to 3, 6, 7 and 10", note: "biasing features that change answers without being mentioned in the chain of thought"}
  - {paper: "Lanham et al. (2023)", title: "Measuring Faithfulness in Chain-of-Thought Reasoning", locator: "arXiv 2307.13702v1, pp. 1 to 9", note: "early answering, filler tokens, paraphrasing and the inverse scaling of faithfulness"}
---

Chain-of-thought prompting asks a language model to write out the steps of its reasoning before it gives an answer.
In the few-shot form of Wei et al. (2022), the examples in the prompt show worked solutions instead of bare answers, and the model imitates them; in the zero-shot form of Kojima et al. (2022), a single sentence such as "Let's think step by step" does the same job.
On multi-step problems such as grade-school math word problems, it raised the accuracy of models of about 100B parameters and up by large margins, and Wang et al. (2023) added more by sampling many chains and taking a majority vote over their final answers.
The gains depend on scale: in the original study they appear only for models of about 100B parameters, and smaller models write fluent but wrong chains.
And the chain is text the model produces, not a trace of its computation: later work found answers that follow a bias the chain never mentions, and chains that change little when cut short.

## In plain words

Ask a large model "The cafeteria had 23 apples; they used 20 and bought 6 more; how many do they have?" and it may answer 27.
Show it an example worked out in steps first, and it works the new question out the same way, 23 − 20 = 3 and 3 + 6 = 9, and gets it right.
That is chain-of-thought prompting: the prompt shows steps, so the model writes steps.
Wei et al. list its appeal: it splits a problem into steps and spends more computation where more steps are needed, it opens a window into the model's behaviour, it suits any task humans solve in language, and it needs only examples in the prompt of a large model. {#c-wei-properties}
A chain that reads well can still be wrong, and the model can reach its answer for a reason the chain does not state.

## Setting

A pretrained language model is prompted with a few input-output examples followed by a test input, and continues the text; this is few-shot prompting, and no weights are changed.
Wei et al. define a chain of thought as a series of intermediate reasoning steps, and chain-of-thought prompting as a few chain of thought demonstrations provided as exemplars in prompting. {#c-wei-definition}
Each exemplar becomes a triple of input, chain of thought and output, where the chain is a sequence of reasoning steps in natural language that ends at the answer. {#c-wei-triples}
Their Figure 1 shows the difference on one exemplar: the standard answer to the tennis balls question is just "The answer is 11", and the chain-of-thought answer works through "5 + 6 = 11" first. {#c-wei-figure-1}
The tasks are those with one correct final answer that takes several steps to reach: math word problems, commonsense questions that need a few facts combined, and symbolic manipulations such as concatenating the last letters of words.

## The method

**Few-shot chain of thought.**
Write a handful of exemplars with worked solutions that end in a fixed phrase, put them before the test question, and decode.
For the math word problem benchmarks GSM8K, SVAMP, ASDiv, AQuA and MAWPS, Wei et al. wrote one set of eight exemplars by hand and did not prompt-engineer them. {#c-wei-exemplars}
They decoded greedily, on GPT-3, LaMDA, PaLM, UL2 and Codex models from 350M to 540B parameters. {#c-wei-models}

**Zero-shot chain of thought.**
Kojima et al. drop the exemplars: adding "Let's think step by step" before each answer is enough to make large models reason in steps, a method they call Zero-shot-CoT. {#c-kojima-definition}
It runs in two stages: a reasoning extraction prompt "Q: [X]. A: [T]" with the trigger $T$, whose continuation is the reasoning $Z$, and an answer extraction prompt that appends $Z$ and an answer trigger such as "Therefore, the answer (arabic numerals) is". {#c-kojima-two-stage}
The same template serves all tasks, with no examples written per task. {#c-kojima-task-agnostic}
For arithmetic, the prediction is the first number in the text after the answer trigger. {#c-kojima-answer-cleansing}

**Self-consistency.**
Wang et al. replace the greedy decoding of one chain with sampling: draw a diverse set of reasoning paths, and select the most consistent answer by marginalizing out the paths. {#c-wang-definition}
With $m$ sampled outputs whose final answers $a_1, \dots, a_m$ lie in a fixed answer set $A$, the answer is the majority vote

$$ \hat a = \arg\max_{a \in A} \sum_{i=1}^{m} \mathbb 1(a_i = a) $$ {#eq-majority-vote}

which Wang et al. call the most consistent answer among the final answer set. {#c-wang-vote}
The answer of each path is parsed from its text: for arithmetic, the first number after "The answer is". {#c-wang-parser}
Each path is drawn by ancestral sampling, one token at a time from the model's next-token distribution, which temperature and top-$k$ reshape; the vote is what self-consistency adds on top of it.
Instead of counting, each path can be weighted by its probability, unnormalized or normalized by its length $K$,

$$ P(r_i, a_i \mid \text{prompt}, \text{question}) = \exp\Big(\frac{1}{K} \sum_{k=1}^{K} \log P(t_k \mid \text{prompt}, \text{question}, t_1, \dots, t_{k-1})\Big) $$ {#eq-path-weight}

and the weights summed or averaged per answer. {#c-wang-weights}
Wang et al. motivate the vote by a hypothesis: correct reasoning processes, even if diverse, tend to agree more in their final answer than incorrect ones. {#c-wang-hypothesis}
The method needs no training, no extra annotation and no other model, and acts as a self-ensemble of one model. {#c-wang-self-ensemble}
It needs a fixed answer set; for open-ended generation one would first need a measure of consistency between generations. {#c-wang-fixed-answer}

## Derivation

Why does a vote over samples beat one sample, and when does it not?
Fix a question, and let one sample give answer $a$ with probability $q(a)$, with $p$ the probability of the correct answer.
Wang et al. sample their 40 outputs per question independently from the decoder, so the answers to one question are independent and identically distributed; the probabilities $q(a)$ and the conditions below are this page's model, not results of the paper. {#c-wang-independent}
In the terms of probability, the answers are independent and identically distributed random variables on $A$, and the share of votes for each answer converges to $q(a)$ by the weak law of large numbers, which is why the most likely answer wins in the end.
With two possible answers and an odd number $m$ of samples, the vote is correct with probability

$$ P_m(p) = \sum_{j = (m+1)/2}^{m} \binom{m}{j} p^j (1 - p)^{m - j} $$ {#eq-binary-vote}

the formula of Condorcet's jury theorem. {#c-binary-vote}
Adding two samples changes the outcome only when the vote was one short of either side, and the two cases give

$$ P_{m+2}(p) - P_m(p) = \binom{m}{\tfrac{m+1}{2}} \big(p(1-p)\big)^{\frac{m+1}{2}} (2p - 1) $$ {#eq-binary-step}

so with $p > 1/2$ each pair of samples raises the accuracy, from $P_1 = p$, and with $p < 1/2$ each pair lowers it. {#c-binary-step}
By Hoeffding's inequality the error falls exponentially in $m$ when $p > 1/2$, and the accuracy falls to 0 when $p < 1/2$. {#c-binary-limit}
An even $m$ buys nothing over $m - 1$ when ties are broken by a fair coin. {#c-even-ties}
Breaking ties toward the answer sampled first, as the implementation here does, is the same as a fair coin for independent samples. {#c-first-seen-ties}

With more than two answers the condition is weaker: the vote converges to the correct answer whenever it is the single most likely answer of one sample, even with $p$ well below $1/2$, and to a wrong answer whenever some wrong answer is more likely. {#c-plurality-limit}
This is the mechanism behind the hypothesis of Wang et al.: if wrong chains scatter over many answers while right chains agree, the right answer is the mode.
A benchmark mixes both kinds of question, and its vote accuracy is the mean over questions; voting can then lower it, as in a two-question example where a single sample is right 60% of the time and the vote tends to 50%. {#c-dataset-mixture}
The numbers: for $p = 0.6$, three samples give 0.648, five give 0.68256 and 21 give about 0.8256, and for $p = 0.4$ three give 0.352. {#c-worked-numbers}
The file `derivations/majority-vote.md` has the steps, and its check computes every item in exact fractions.

Nothing in the derivation explains why a chain of thought helps in the first place; it only says what a vote does with the per-sample probabilities that the chains produce.

## Worked example

Take the cafeteria question of Figure 1 of Wei et al., whose correct answer is 9.
Suppose five chains are sampled; the texts below are made up for illustration, not outputs reported by any of the papers.

| sample | chain (shortened) | parsed answer |
|---|---|---|
| 1 | 23 − 20 = 3, 3 + 6 = 9. The answer is 9. | 9 |
| 2 | 23 − 20 = 3 apples are left, and 3 + 6 = 9. The answer is 9. | 9 |
| 3 | 23 + 6 = 29, then 29 − 20 = 9. The answer is 9.0. | 9 |
| 4 | 20 + 6 = 26 and 26 + 1 = 27. The answer is 27. | 27 |
| 5 | 23 − 20 = 3. The answer is 3. | 3 |

The counts are 9: 3, 27: 1, 3: 1, so the vote returns 9.
Greedy decoding returns one chain; had it been the fourth, the answer would have been 27.
Sample 3 writes "9.0", and only normalizing the parsed answers makes it vote with "9".
If a single sample were right with probability $p = 0.6$ against one wrong answer, five votes would be right with probability 0.68256, as in the derivation.
The test `test_worked_example` checks the parsing and the vote, and `test_binary_formula_and_numbers` the probability.

## Hyperparameters

- **Exemplars.** Eight hand-written exemplars for the math benchmarks of Wei et al., four for AQuA. More important than their number is that they match the task: Kojima et al. found that exemplars from another domain do worse than no exemplars with the trigger. {#c-kojima-cross-task}
- **The trigger.** "Let's think step by step" scored 78.7 on MultiArith in the template study of Kojima et al.; other instructive templates came close, and misleading or irrelevant ones did no better than no trigger. {#c-kojima-templates}
- **The answer trigger and parser.** They decide what counts as the answer; a parser that misreads the chain turns a right chain into a wrong vote.
- **Decoding.** Greedy for one chain; for self-consistency, Wang et al. sampled at $T = 0.5$ with top-$k$ 40 for UL2 and LaMDA, $T = 0.7$ with $k = 40$ for PaLM, and $T = 0.7$ without truncation for GPT-3. {#c-wang-sampling}
- **Number of samples $m$.** Accuracy rose steadily from 1 to 40 sampled paths in Wang et al. {#c-wang-paths} They suggest 5 or 10 paths as a starting point that realizes most of the gain. {#c-wang-cost}
- **Aggregation.** The plain vote, or a sum of length-normalized path probabilities, which gave nearly the same accuracy. {#c-wang-weighted}

## Cost

There is no training cost; the cost is writing a few exemplars per task, or none with the zero-shot trigger.
At inference every answer carries its chain, so the output is many tokens longer, and the context is longer by the exemplars.
Self-consistency multiplies the generation by $m$, 40 in the experiments of Wang et al.
Wei et al. add that since the ability appears only in large models, chain-of-thought reasoning is costly to serve in real-world applications.

## Evidence

**Arithmetic.**
Wei et al. report that with eight exemplars PaLM 540B passed the previous GSM8K result of a finetuned GPT-3 with a verifier. {#c-wei-sota}
Kojima et al. quote the PaLM 540B gain on GSM8K as 17.9% to 58.1%. {#c-kojima-palm-gsm8k}
The gains grew with the number of steps a problem needs: GSM8K accuracy more than doubled for GPT-3 175B and PaLM 540B, while on the one-step SingleOp problems the change was negative or very small. {#c-wei-harder}
With no exemplars at all, Zero-shot-CoT raised text-davinci-002 from 17.7% to 78.7% on MultiArith and from 10.4% to 40.7% on GSM8K. {#c-kojima-headline}
Few-shot-CoT with eight examples was better still, 93.0 and 48.7. {#c-kojima-few-shot-compare}
So Zero-shot-CoT is below Few-shot-CoT but well above standard few-shot prompting. {#c-kojima-beats-few-shot}
Self-consistency then raised GSM8K accuracy from 56.5 to 74.4 for PaLM 540B and from 60.1 to 78.0 for code-davinci-002. {#c-wang-table-2}
Across benchmarks its gains reached +17.9% on GSM8K, +11.0% on SVAMP and +12.2% on AQuA, with smaller ones on StrategyQA and ARC-challenge. {#c-wang-gains}
It also works on top of the zero-shot trigger, from 43.0 to 69.2 for PaLM 540B on GSM8K. {#c-wang-zero-shot}

**Commonsense and symbolic tasks.**
PaLM 540B with chains of thought reached 75.6% on StrategyQA and 95.4% on sports understanding, with a minimal gain on CSQA. {#c-wei-commonsense}
Zero-shot-CoT did not help on CommonsenseQA, where it fell from 68.8 to 64.6. {#c-kojima-commonsense}
On last-letter concatenation and coin flip, chains of thought let models generalize to more steps than the exemplars showed. {#c-wei-symbolic}
On some inference tasks, ANLI, e-SNLI and RTE, a chain of thought hurt PaLM 540B compared with standard prompting, and self-consistency recovered and passed standard prompting. {#c-wang-hurts}

**What the gain is not.**
Wei et al. ran ablations on GSM8K.
An equation alone before the answer did not help much, though it did help on one- and two-step datasets. {#c-wei-equation-only}
A string of dots as long as the needed equation, which gives the model more tokens without content, did about as well as no chain. {#c-wei-variable-compute}
A chain written after the answer did about as well as no chain, so the gain needs the reasoning to come before the answer. {#c-wei-after-answer}
Chains written by three different annotators, and a more concise version, all beat standard prompting by a large margin. {#c-wei-annotators}
Self-consistency beat ensembles over prompts and over exemplar orders: on LaMDA 137B and GSM8K, 27.7 against 18.6 and 19.2. {#c-wang-ensembles}
It also beat beam search, whose outputs are less diverse. {#c-wang-beam}
This differs from ensemble learning, which combines several trained models: self-consistency samples one model several times with one prompt.

**The chains themselves.**
Of 50 LaMDA 137B chains that reached a correct GSM8K answer, all but two were logically and mathematically correct; of 50 that reached a wrong answer, 46% were almost correct and 54% had major errors. {#c-wei-audit}
Of 45 errors of PaLM 62B, the move to PaLM 540B fixed 6 of 20 semantic understanding errors, 12 of 18 one-step-missing errors and 4 of 7 others. {#c-wei-62b-errors}
The agreement among sampled chains tracks accuracy, so it can serve as an uncertainty estimate. {#c-wang-uncertainty}

## Limitations and failure modes

**Scale.**
Wei et al. found chain-of-thought prompting to be an emergent ability of scale: no help for small models, gains only at about 100B parameters, and fluent but illogical chains below that. {#c-wei-emergent}
For most models below 10B parameters it hurt. {#c-wei-small-hurts}
Kojima et al. saw the same with the zero-shot trigger: the curve with chain of thought rose steeply with size, and at small sizes chain of thought did not help. {#c-kojima-scale}
The gains of self-consistency also grew with scale, +3% to 6% for UL2 20B against +9% to 23% for LaMDA 137B and GPT-3. {#c-wang-scale}
Two caveats come from the same papers.
Scale here is a family of models, and Wei et al. note that model scale is often conflated with other factors such as training compute. {#c-wei-conflated}
And the threshold is where these prompts start to help these models on these benchmarks, not a property of chains of thought in general: the evidence is a few model families from 2021 and 2022, compared at a few sizes.

**Correctness.**
Nothing guarantees a correct reasoning path; a chain can lead to a right or a wrong answer. {#c-wei-limitations}
Sampled paths can be wrong or nonsensical, which is why Wang et al. call for better grounding of the rationales. {#c-wang-incorrect-paths}
The model's own probabilities did not separate right chains from wrong ones well; Wang et al. read the near-equal accuracies of the plain and the weighted vote as poor calibration. {#c-wang-calibration}
Prompt engineering still matters: on coin flip, one annotator's chains gave 99.6% and another's 71.4%, against 50.0% without chains. {#c-wei-prompt-engineering}

**Faithfulness.**
Wei et al. described the chain as an interpretable window into the model's behaviour, and qualified it in the same sentence: fully characterizing the computations behind an answer remains an open question. {#c-wei-open-question}
They also left open whether the network is actually reasoning. {#c-wei-not-reasoning}
Turpin et al. tested the window directly.
They added biasing features to prompts, such as reordering the options of the few-shot examples so that the answer was always (A), and the models' explanations systematically failed to mention them. {#c-turpin-misrepresent}
When the bias pointed to a wrong answer, the models often rationalized it, and accuracy dropped by up to 36% on 13 BIG-Bench Hard tasks for GPT-3.5 and Claude 1.0. {#c-turpin-36}
The largest drop, 36.3%, came from a suggested answer in the prompt; the always-(A) ordering cost 18.7% for GPT-3.5 and 4.7% for Claude 1.0. {#c-turpin-drops}
Of 426 explanations supporting biased predictions, one mentioned the bias. {#c-turpin-one-mention}
Of 104 annotated unfaithful explanations, 73% argued for the biased answer and 15% had no obvious error. {#c-turpin-73}
On the social-bias benchmark BBQ, explanations justified stereotype-consistent answers without mentioning the stereotype. {#c-turpin-bbq}
They give reasons not to expect faithful chains: nothing in training rewards accurate reports of the reasons for an answer, human explanations are often unfaithful too, and RLHF may reward explanations that merely look good. {#c-turpin-why-unfaithful}
Chain of thought still helped in the unbiased setting, from 57.1% to 59.6% for GPT-3.5 and from 59.2% to 65.3% for Claude 1.0. {#c-turpin-helps-unbiased}
And their test is necessary but not sufficient: a chain that passes it can still be unfaithful. {#c-turpin-necessary}

Lanham et al. measured how much the answer depends on the chain, testing three ways a chain could be unfaithful: reasoning written after the answer is settled, a gain from extra tokens whatever they say, and information hidden in the wording. {#c-lanham-hypotheses}
The dependence varied widely across tasks, and the gain was not explained by the extra compute alone or by the phrasing. {#c-lanham-vary}
Cutting the chain short changed the final answer less than 10% of the time on ARC Easy, ARC Challenge and OpenBookQA, and more than 60% of the time on AQuA. {#c-lanham-early}
Filler tokens in place of the chain gave no gain, which argues against extra compute alone. {#c-lanham-filler}
A paraphrased chain kept the accuracy of the original, which argues against an encoding in the wording. {#c-lanham-paraphrase}
Larger and more capable models gave less faithful reasoning on most tasks. {#c-lanham-inverse}
On seven of eight tasks the 13B model changed its answer after the chain more often than the 175B model did. {#c-lanham-13b}
They had no ground truth for the model's internal reasoning, and studied only RLHF-finetuned models. {#c-lanham-limitations}

Read together, the two lines of work disagree less than they seem.
Wei et al. found that the content and the position of the chain matter for accuracy, and Lanham et al. found that on some tasks, such as AQuA, the answer depends heavily on the chain.
Turpin et al. and Lanham et al. found that on other tasks the answer is mostly settled before the chain, and that inputs can sway the answer without appearing in the chain.
So a chain of thought can do real work for the answer and still be an incomplete or misleading account of it.

## Variants

| variant | what changes | source |
|---|---|---|
| Few-shot-CoT | exemplars with worked chains, greedy decoding | Wei et al. (2022) |
| Zero-shot-CoT | a trigger sentence and a second prompt to extract the answer, no exemplars | Kojima et al. (2022) |
| Self-consistency | $m$ sampled chains and a majority vote over their parsed answers | Wang et al. (2023) |
| Weighted self-consistency | the vote weighted by length-normalized path probabilities | Wang et al. (2023), Table 1 |
| Equation only | an equation instead of a natural-language chain | Wei et al. (2022), ablation |

The normalized weighted sum was far better than the unnormalized one, and the weighted average was much worse than both. {#c-wang-normalized}

## When to use it

Use it for problems with several steps and one checkable final answer, such as math word problems, on a large model; that is where the reported gains are.
Expect little or a loss on one-step problems, on some commonsense benchmarks, and on small models.
Add self-consistency when the answer comes from a fixed set and the budget allows several samples; it also gives an agreement rate that tracks accuracy.
Do not use the chain as an explanation of the answer without independent checks: it can omit what drove the answer.
Standard prompting only gives a lower bound on what a model can do, in the words of Wei et al., so a comparison of models without chains can understate the larger ones. {#c-wei-lower-bound}

## Pitfalls

> [!pitfall]
> **Voting on unnormalized strings.** "9", "9.0" and "$9" are different strings, so a vote over raw text splits the right answer. Parse and normalize first; `test_formatting_variants_vote_together` checks that the three forms of 1000 vote together after `extract_answer`, and lose a four-way tie before it.

> [!pitfall]
> **The weighted average.** Dividing an answer's summed weight by its number of paths favours an answer reached by one confident path. `test_weighted_average_favours_a_rare_confident_answer` shows a single path outvoting three.

> [!pitfall]
> **Unnormalized path probabilities.** The probability of a whole generation shrinks with its length, so it favours short chains. `test_unnormalized_weights_favour_short_paths` shows one short path beating three long ones that the length-normalized weights of eq-path-weight keep.

> [!pitfall]
> **Expecting the vote to always help.** The vote helps a question only when the correct answer is the most likely single answer; a wrong mode wins more surely with more samples. `test_vote_can_lower_benchmark_accuracy` builds a benchmark where voting lowers accuracy from 0.6 toward 0.5.

> [!pitfall]
> **Ties.** With an even $m$ or several answers, ties happen, and a rule that always picks, say, the smallest answer biases the vote. The implementation picks the answer sampled first, which `test_exact_accuracy_matches_enumeration_with_first_seen_ties` checks against exact fair-tie accuracies.

## Implementations

`impl/numpy/self_consistency.py` holds the vote of eq-majority-vote twice: `majority_vote` in plain Python with a dictionary, and `majority_vote_np` with `numpy.unique`.
Both skip samples whose answer could not be parsed and break ties toward the answer sampled first, and `test_pure_python_matches_numpy` checks them against each other on random inputs.
`extract_answer` reads the first number after the last "The answer is", as the parsers of Kojima et al. and Wang et al. do, and normalizes commas and trailing zeros.
`path_weight` and `weighted_vote` give the weighted sum and weighted average of eq-path-weight.
`vote_accuracy` computes the exact accuracy of the vote for a given answer distribution, in fractions when given fractions, and `simulate_vote_accuracy` checks it by simulation.
There is no model in the code: generating the chains is a call to whatever language model is used.

## History

Jason Wei and eight coauthors at Google Research, Brain Team, posted chain-of-thought prompting to arXiv in January 2022, and the paper appeared at NeurIPS 2022. {#c-hist-wei}
In March 2022 Xuezhi Wang and colleagues on the same team posted self-consistency, published at ICLR 2023. {#c-hist-wang}
In May 2022 Takeshi Kojima and coauthors at the University of Tokyo and Google Research reported that a trigger sentence could replace the exemplars, also at NeurIPS 2022. {#c-hist-kojima}
In 2023 the question turned to faithfulness: Miles Turpin and coauthors at NYU, Cohere and Anthropic showed unmentioned biases steering explanations, at NeurIPS 2023, {#c-hist-turpin} and Tamera Lanham and coauthors at Anthropic measured how much answers depend on the chain. {#c-hist-lanham}
