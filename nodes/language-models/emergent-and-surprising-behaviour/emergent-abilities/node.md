---
tag: NEW-emergent-abilities
kind: finding
slug: emergent-abilities
name: Emergent abilities
aliases: [emergent abilities of large language models, emergence in language models, abrupt capability scaling, phase transitions in language models, sharp left turn in benchmarks]
area: language-models/emergent-and-surprising-behaviour
level: core
summary: >-
  Some tasks stay near chance for language models over orders of magnitude of scale, then rise sharply past a threshold; how much of the jump is in the model and how much in the metric is disputed.
related_to: [grokking, chain-of-thought-prompting, mmlu, pass-at-k, expected-calibration-error, double-descent]
history:
  year: 2022
  note: "Ganguli et al. (arXiv February 2022, FAccT 2022) described abrupt specific capability scaling; Wei et al. (arXiv 2206.07682, TMLR 2022) defined emergent abilities and surveyed them; Schaeffer et al. (arXiv 2304.15004, 2023) argued that the metric creates them; Hu et al. (ICLR 2024), Lu et al. (ACL 2024) and Du et al. (NeurIPS 2024) refined the question."
prereqs: [neural-scaling-laws, cross-entropy-loss@al, few-shot-in-context-learning]
same_as:
  wikipedia: Large_language_model
  arxiv: ["2206.07682", "2304.15004", "2202.07785", "2403.15796", "2310.03262", "2309.01809"]
sources:
  - {paper: "Wei, Tay, Bommasani, Raffel, Zoph, Borgeaud, Yogatama, Bosma, Zhou, Metzler, Chi, Hashimoto, Vinyals, Liang, Dean and Fedus (2022)", title: "Emergent Abilities of Large Language Models", locator: "TMLR 2022, arXiv 2206.07682v2, pp. 1 to 9", note: "the definition, the few-shot and augmented-prompting examples, and the discussion of explanations"}
  - {paper: "Ganguli, Hernandez, Lovitt and others (2022)", title: "Predictability and Surprise in Large Generative Models", locator: "FAccT 2022, arXiv 2202.07785v2, pp. 4 to 6", note: "smooth general scaling beside abrupt specific capabilities"}
  - {paper: "Schaeffer, Miranda and Koyejo (2023)", title: "Are Emergent Abilities of Large Language Models a Mirage?", locator: "arXiv 2304.15004v2, pp. 1 to 9", note: "the metric account, the mathematical model, the BIG-Bench meta-analysis and the induced emergence in vision"}
  - {paper: "Hu, Liu, Han and others (2024)", title: "Predicting Emergent Abilities with Infinite Resolution Evaluation", locator: "ICLR 2024, arXiv 2310.03262v3, pp. 1 to 9", note: "PassUntil, the task scaling law and accelerated emergence"}
  - {paper: "Lu, Bigoulaeva, Sachdeva, Tayyar Madabushi and Gurevych (2024)", title: "Are Emergent Abilities in Large Language Models just In-Context Learning?", locator: "ACL 2024, pp. 5098 to 5139, PDF pp. 1 to 5", note: "the in-context learning confound"}
  - {paper: "Du, Zeng, Dong and Tang (2024)", title: "Understanding Emergent Abilities of Language Models from the Loss Perspective", locator: "NeurIPS 2024, arXiv 2403.15796v3, pp. 1 to 8", note: "emergence as a threshold in pre-training loss, under continuous metrics too"}
claim: >-
  On some tasks, language models of increasing scale score near chance across several orders of magnitude of training compute and then score well above chance past a threshold scale, so that the larger models' performance could not have been predicted by extrapolating the smaller models' scores.
first_reported: {work: wei-2022-emergent-abilities, note: "named and defined emergent abilities and collected examples from GPT-3, LaMDA, Gopher, Chinchilla and PaLM; Ganguli et al. had described the same abrupt jumps months earlier"}
replications:
  - {work: ganguli-2022-predictability-surprise, note: "three-digit addition in GPT-3, MMLU in Gopher and program synthesis, beside smooth scaling of the loss"}
  - {work: du-2024-emergent-abilities-loss-perspective, note: "more than 30 models pre-trained from scratch; MMLU, C-Eval, GSM8K and GSM8K-Chinese stay at chance until the pre-training loss falls below about 2.2, under continuous metrics too"}
  - {work: hu-2024-predicting-emergent-abilities-passuntil, note: "with sampling until success, small models show steady gains of order 10^-5, and some tasks still speed up with scale faster than a scaling law allows"}
scale_range:
  - {range: "GPT-3, LaMDA, Gopher, Chinchilla and PaLM, from about 10^20 to 2.5 × 10^24 training FLOPs", work: wei-2022-emergent-abilities}
  - {range: "models of 300M to 32B parameters trained from scratch on varied numbers of tokens, and LLaMA 7B to 65B", work: du-2024-emergent-abilities-loss-perspective}
failures:
  - {work: schaeffer-2023-emergent-abilities-mirage, note: "the same model outputs give smooth curves under token edit distance or the Brier score, and more test data shows above-chance accuracy at small scale"}
  - {work: lu-2024-emergent-abilities-in-context-learning, note: "with in-context learning controlled, only two of the previously emergent tasks remain emergent in davinci, and neither needs reasoning"}
explanations:
  - {name: "the metric", summary: "A smooth gain in per-token accuracy becomes a sharp curve under all-or-nothing scoring of a long answer, and too little test data hides the small values.", sources: [schaeffer-2023-emergent-abilities-mirage, wei-2022-emergent-abilities, hu-2024-predicting-emergent-abilities-passuntil]}
  - {name: "a tiny slice of the distribution", summary: "A specific task is a small part of what the model predicts, so it can change fast while the average loss changes smoothly.", sources: [ganguli-2022-predictability-surprise]}
  - {name: "a loss threshold", summary: "Some abilities appear only once the pre-training loss is low enough, whatever the mix of parameters and data that reaches it.", sources: [du-2024-emergent-abilities-loss-perspective]}
  - {name: "in-context learning, memory and linguistic knowledge", summary: "The apparently new abilities are a combination of skills the models already show, unlocked by examples in the prompt.", sources: [lu-2024-emergent-abilities-in-context-learning]}
---

Emergent abilities are tasks on which language models of growing size score near chance for a long stretch of scale and then, past some threshold, well above it.
Wei et al. (2022) consider an ability to be emergent if it is not present in smaller models but is present in larger models, so that emergent abilities cannot be predicted simply by extrapolating the performance of smaller models. {#c-wei-def}
The pattern contrasts with the smooth power laws that the training loss follows, and it raised the question of what else further scaling might bring.
From 2023, work traced much of the sharpness to how the tasks are scored, and the debate turned to which jumps survive a smooth metric, more test data, and controls for in-context learning.

## In plain words

Train a series of language models, each larger than the last, and give them a test, such as adding two three-digit numbers.
The training loss improves steadily along the series.
The test score often does not: the small models score close to zero, then a model somewhere in the series suddenly gets many answers right.
Plotted against model size, the score is flat and then shoots up like a hockey stick, and the size of the jump could not have been read off the flat part.

That is the observation.
What it means is less clear.
A sum of several digits is marked right only if every digit is right, so a model that is slowly getting better at each digit can still score zero on whole answers for a long time.
Score the same answers by how many digits are wrong, and the flat part becomes a steady slope.
On this reading the jump is in the ruler, not in the model.
Other researchers find that some jumps stay when the ruler is smooth, and that they line up with how low the training loss has gone.

## The finding

Wei et al. adopt a general definition: emergence is when quantitative changes in a system result in qualitative changes in behavior, a definition rooted in the 1972 essay "More Is Different" by the physicist Philip Anderson. {#c-anderson}
On a scaling curve, with model scale on the x-axis and performance on the y-axis, emergent abilities show a clear pattern: performance is near-random until a certain critical threshold of scale is reached, after which performance increases to substantially above random, a qualitative change also known as a phase transition. {#c-phase}
They measure scale chiefly by training compute in FLOPs, and also by the number of parameters.

Their clearest examples come from few-shot prompting.
On an arithmetic benchmark of 3-digit addition and subtraction and 2-digit multiplication, GPT-3 and LaMDA have close-to-zero performance for several orders of magnitude of training compute, before performance jumps to sharply above random at 2 × 10^22 training FLOPs (13B parameters) for GPT-3, and 10^23 training FLOPs (68B parameters) for LaMDA. {#c-arithmetic}
On the Word in Context benchmark, GPT-3 and Chinchilla fail to achieve one-shot performance better than random even at their largest size of about 5 × 10^23 FLOPs, and above-random performance emerged when PaLM was scaled to 2.5 × 10^24 FLOPs (540B parameters). {#c-wic}

Wei et al. also count a prompting or fine-tuning method as emergent when it does not help, or hurts, until a certain scale.
Chain-of-thought prompting only surpasses standard prompting without intermediate steps when scaled to 10^23 training FLOPs (about 100B parameters). {#c-cot}
On 8-digit addition, fine-tuning on a scratchpad of intermediate outputs only helps for models of about 9 × 10^19 training FLOPs (40M parameters) or larger. {#c-scratchpad}
The thresholds thus range over four orders of magnitude of compute, from a 40M-parameter model to PaLM 540B.

The threshold is not fixed by the task.
Wei et al. note that the scale at which an ability is first observed to emerge depends on a number of factors and is not an immutable property of the ability. {#c-not-immutable}
There are 14 BIG-Bench tasks for which LaMDA 137B and GPT-3 175B perform at near-random, but PaLM 62B achieves above-random performance, despite having fewer model parameters and training FLOPs. {#c-palm-62b}

## Evidence

The generated list gives each source with its label and its role.
The lines of evidence are these.

**An earlier description.**
Ganguli et al. (2022) contrast smooth general scaling with what they call abrupt specific capability scaling: though performance is predictable at a general level, performance on a specific task can sometimes emerge quite unpredictably and abruptly at scale. {#c-ganguli-abrupt}
Their example from GPT-3: three-digit addition is performed accurately less than 1% of the time on any model with less than 6B parameters, but this jumps to 8% accuracy on a 13B parameter model and 80% accuracy on a 175B parameter model. {#c-ganguli-addition}

**The metric account.**
Schaeffer, Miranda and Koyejo (2023) name the two properties that make emergent abilities intriguing: their sharpness, transitioning seemingly instantaneously from not present to present, and their unpredictability, appearing at seemingly unforeseeable model scales. {#c-two-properties}
Their alternative explanation: for a particular task and model family, when analyzing fixed model outputs, nonlinear or discontinuous metrics produce apparent emergent abilities, whereas linear or continuous metrics produce smooth, continuous, predictable changes in model performance. {#c-mirage}
They test it in three ways.

First, they collected outputs of the InstructGPT/GPT-3 family on 2-shot multiplication of two 2-digit integers and 2-shot addition of two 4-digit integers; under accuracy the family's performance appears sharp and unpredictable on longer targets, and under token edit distance the family exhibits smooth, predictable performance improvements. {#c-gpt3-arith}
Generating additional test data increases the resolution and reveals that even on accuracy, the InstructGPT/GPT-3 family's performance is above chance and improves in a smooth, continuous, predictable manner. {#c-more-data}

Second, a meta-analysis of BIG-Bench.
Of the 39 preferred metrics in BIG-Bench, at most 5 display emergence by the emergence score of Srivastava et al., and in the hand-annotated task, metric and model family triplets, 2 metrics account for more than 92% of claimed emergent abilities: Multiple Choice Grade and Exact String Match. {#c-bigbench-metrics}
Multiple Choice Grade is discontinuous, and Exact String Match is nonlinear.
For the LaMDA model family, emergent abilities under Multiple Choice Grade disappear when the same outputs are measured under the continuous Brier Score. {#c-lamda-brier}

Third, they produced emergence on purpose.
Shallow nonlinear autoencoders trained on CIFAR100 display smoothly decreasing mean squared reconstruction error as the bottleneck grows, and a newly defined metric that counts the test images with squared reconstruction error below a threshold $c$ induces a sharp and seemingly unpredictable change. {#c-induced-vision}

**Infinite-resolution evaluation.**
Hu et al. (2024) attack the resolution problem directly.
Their PassUntil evaluation samples answers until $r$ of them pass and records the number $K$ of samples drawn; the estimate of the success probability is $\mathrm{PU} = r/K$. {#c-passuntil}
With it, smaller models show subtle but non-negligible performance on the order of 10^-5 that improves steadily as the model scales up, and the relation between $\log(-\log \mathrm{PU})$ and $\log N$ is almost exactly linear. {#c-task-law}
Using this task scaling law they predicted the performance of their 2.4B model on code generation with a deviation of 0.05% before training started. {#c-predict-2p4b}

**The loss threshold.**
Du et al. (2024) pre-trained more than 30 language models of varied model and data sizes from scratch, with a fixed data corpus, tokenization and architecture, and evaluated them on 12 datasets. {#c-du-setup}
Transformer models with the same pre-training loss, but different model and data sizes, generate the same performance on various downstream tasks. {#c-du-same-loss}
On MMLU, C-Eval, GSM8K and GSM8K-Chinese, performance stays flat while the loss is above a threshold, and the thresholds are all around 2.2, despite differences in languages, tasks, prompting types and answer forms. {#c-du-threshold}
They report that the emergent abilities persist across both discontinuous and continuous metrics, measuring MMLU and C-Eval by the probability of the correct choice and by the Brier score as well as by accuracy. {#c-du-continuous}

**The in-context learning confound.**
Lu et al. (2024) ran over 1000 experiments on 18 models with and without in-context examples, and conclude that purported emergent abilities are not truly emergent but result from a combination of in-context learning, model memory, and linguistic knowledge. {#c-lu-combination}
With in-context learning controlled, just two of the previously emergent tasks remain emergent, Nonsense words grammar and Hindu knowledge, and they find no functional linguistic abilities emergent in davinci, the 175B GPT model without instruction tuning. {#c-lu-two-tasks}

## Scope and conditions

**The metric and the length of the answer.**
Wei et al. themselves warn that using exact string match as the evaluation metric for long-sequence targets may disguise compounding incremental improvements as emergence. {#c-disguise}
They also measured the cross-entropy loss on six emergent BIG-Bench tasks and found that it improves even for small model scales where the downstream metrics are close to random and do not improve. {#c-xent-improves}
But they argue that metrics without partial credit are at best an incomplete explanation, because emergent abilities are still observed on many classification tasks. {#c-classification}
Du et al. hold the same against Schaeffer et al. for multiple-choice tasks: the jump at their loss threshold remains under continuous scores. {#c-du-against}

**The test set.**
Under all-or-nothing scoring, a model whose chance of a fully correct answer is below one in the size of the test set will usually score zero.
Schaeffer et al. list insufficient resolution to estimate performance in the smaller parameter regime, with resolution set by one over the test set size, as one of the three factors behind sharp curves. {#c-resolution}
PassUntil removes this factor by sampling until a fixed number of successes.

**The x-axis.**
Wei et al. view emergence as a function of many correlated variables, and also plot it against WikiText103 perplexity, which closely correlates with training compute for Gopher and Chinchilla. {#c-perplexity-axis}
Du et al. argue for the pre-training loss as the x-axis, since different mixes of parameters and data reach different losses at the same compute.
They define an ability as emergent if it is not present in models with higher pre-training loss but is present in models with lower pre-training loss: normalized performance is $f(L)$ for $L < \eta$ and 0 otherwise, with $f$ decreasing and $\eta$ the threshold. {#c-du-def}

**The prompt.**
An ability counted as emergent under few-shot prompting may be in-context learning, which Lu et al. separate by also testing without examples.
The few-shot in-context learning page describes how examples in the prompt act as a learning signal.

**Multiple comparisons.**
Schaeffer et al. note that BIG-Bench alone has at least 220 tasks, about 40 metrics per task and about 10 model families, for about 10^6 task, metric and model family triplets, so some apparent emergence is expected by chance. {#c-multiple-comparisons}

**What is not claimed.**
Schaeffer et al. state that nothing in their paper should be interpreted as claiming that large language models cannot display emergent abilities; their message is that previously claimed emergent abilities might be a mirage induced by the researchers' analyses. {#c-not-claimed}

## Explanations

**The metric.**
The claim: a smooth improvement in each token becomes a sharp curve when a long answer is scored all or nothing.
The evidence: Schaeffer et al. model the per-token cross-entropy as a power law in parameters, so the per-token probability of the correct token is $\exp(-(N/c)^{\alpha})$; with $L$ independent tokens, accuracy is about this probability to the power $L$, while token edit distance is about $L$ times one minus it. {#c-metric-model}
Write the per-token loss as $\ell(N) = (c/N)^{\alpha}$ with $\alpha > 0$, so that a token is right with probability $e^{-\ell}$, a whole answer of $L$ tokens with probability $A_L(N) = e^{-L\ell(N)}$, and the expected number of wrong tokens is $E_L(N) = L(1 - e^{-\ell(N)})$.
The derivation metric-model works this model out.
$A_L$ is a smooth, strictly increasing function of $\log N$, with no jump at any scale. {#c-mm-smooth}
It passes $1/e$ at $N^\ast = c\,L^{1/\alpha}$, so a longer answer moves the rise to larger models by the factor $L^{1/\alpha}$. {#c-mm-midpoint}
Its steepest slope against $\ln N$ is $\alpha/e$, and its rise from 0.1 to 0.9 spans $\ln(\ln 10/\ln(10/9))/\alpha \approx 3.08/\alpha$ in $\ln N$, whatever $L$ is: on a log axis the rise is no sharper for long answers, only later and preceded by a longer flat stretch. {#c-mm-width}
For $\alpha = 0.5$ and $c = 10^8$, the check of the derivation finds the rise from 0.1 to 0.9 between $1.9 \times 10^7$ and $9.0 \times 10^9$ parameters for one-token answers and between $4.8 \times 10^9$ and $2.3 \times 10^{12}$ for sixteen-token answers; a family that stops at $10^{10}$ parameters shows the first as a steady rise and the second as a flat line with an upturn at its end.
On the other side, $\ln(-\ln A_L) = \ln L + \alpha \ln c - \alpha \ln N$ is a straight line in $\ln N$, the form of the task scaling law of Hu et al., and $E_L$ lies below $L\ell$ with ratio tending to 1 as $\ell \to 0$, a power law in $N$. {#c-mm-linear}
With $n$ test questions, the measured accuracy is exactly zero with probability $(1 - A_L)^n \ge 1 - nA_L$, so every model with $A_L \le 1/(10n)$ scores zero at least nine times in ten. {#c-mm-resolution}
Hu et al. derive the same form for PassUntil, $\mathrm{PU} \sim \exp(-c N^{-\alpha})$, from the loss scaling law of each token of the answer. {#c-hu-same-form}
The consequence: the flat part of the curve carries information that a smooth metric or enough samples can recover, and a prediction from small models is possible.

**A tiny slice of the distribution.**
The claim: Ganguli et al. suggest the abrupt jumps are possible because any specific task is a tiny slice of a model's output probability distribution, and so can change rapidly even as the full distribution remains smooth. {#c-tiny-slice}
They liken it to daily weather, which is less predictable than seasonal averages. {#c-weather}
The consequence: smooth scaling of the average loss does not imply smooth scaling of every task.

**Depth and memory.**
The claim: Wei et al. offer intuitions, not proofs: if a multi-step reasoning task requires $l$ steps of sequential computation, it might require a model with a depth of at least $O(l)$ layers, and more parameters and training may give the memorization needed for world knowledge. {#c-depth}
The consequence: on this view some thresholds are real, set by what the task needs.

**Accelerated emergence.**
The claim: Hu et al. find tasks whose scaling curve cannot be fitted by the standard scaling law function and whose speed increases with scale, which they call accelerated emergence, and they suggest the multiple circuits hypothesis might be responsible. {#c-accelerated}
The consequence: even with infinite resolution, some curves bend upward faster than a single power law, which the metric account alone does not predict.

**A loss threshold.**
The claim: Du et al. relate the threshold to grokking, the rise of performance from chance to perfect generalization long after the training loss is low, though pre-trained models are usually underfitting rather than overfitting. {#c-grokking}
The consequence: on this view, emergence is real but is a property of the loss reached, not of size as such.

**In-context learning.**
The claim: Lu et al. explain the reported abilities as in-context learning plus memory and linguistic knowledge, all of which grow smoothly. {#c-lu-theory}
The consequence: abilities that need the prompt's examples are not evidence of new latent skills.

These accounts overlap.
The metric account explains why curves look sharp; the loss and circuit accounts address the jumps that remain when the metric is smooth; the in-context account changes which abilities are counted at all.

## Implications

The implications below are those that the sources draw.
Wei et al. note that some abilities may never emerge, for instance on tasks far out of the distribution of even a very large training set, and that an ability could emerge and then plateau, so there is no guarantee that scaling enables an ability to reach the desired level. {#c-plateau}
Schaeffer et al. draw three lessons: a task and a metric are distinct choices when building a benchmark; whoever uses accuracy should have enough data to measure it accurately; and claims about the capabilities of large models need proper controls. {#c-sc-lessons}
Ganguli et al. argue that the high-level predictability and appearance of useful capabilities drives rapid development of such models, while the unpredictable qualities make it difficult to anticipate the consequences of model deployment. {#c-policy}
Lu et al. note that an assumption of emergent functional abilities can lead users to provide insufficiently detailed instructions, potentially resulting in hallucinations and errors. {#c-lu-safety}

For evaluation practice the dispute leaves a simple rule: report a smooth score, such as the per-token loss on the answer, pass@k with many samples or the Brier score, beside any all-or-nothing score.
The MMLU and pass@k pages give the scores most often used; the expected calibration error page covers the related question of whether a model knows when it is right.
The neural scaling laws page gives the smooth laws against which emergence is defined, and the double descent page another case where a curve against size is not monotone.
The chain-of-thought prompting page covers the most cited emergent prompting method.

## Open questions

Whether the jumps that survive smooth metrics, such as those at the loss threshold of Du et al. or the accelerated curves of Hu et al., have a mechanistic cause, such as a circuit that forms at a certain loss, is open.
How to forecast a large model's ability on a new task from small models, before training, is the practical form of the question.

## History

Philip Anderson's essay "More Is Different" appeared in Science in 1972; Wei et al. take their general definition of emergence from it by way of Steinhardt (2022).
The abrupt arithmetic results go back to GPT-3 in 2020, whose few-shot accuracy on multi-digit arithmetic rose steeply with size.
Ganguli et al. posted "Predictability and Surprise in Large Generative Models" as arXiv 2202.07785 in February 2022, and it appeared at FAccT in June 2022.
Wei et al. posted "Emergent Abilities of Large Language Models" as arXiv 2206.07682 in June 2022, the month the BIG-Bench collaboration posted its benchmark, and published it in Transactions on Machine Learning Research in August 2022.
Schaeffer, Miranda and Koyejo posted "Are Emergent Abilities of Large Language Models a Mirage?" as arXiv 2304.15004 in April 2023.
In 2024, Hu et al. (ICLR), Lu et al. (ACL) and Du et al. (NeurIPS) recast the question in terms of resolution, prompting and pre-training loss.

## References
