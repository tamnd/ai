---
tag: NEW-process-reward-model
kind: method
slug: process-reward-model
name: Process reward model
aliases: [PRM, process-supervised reward model, process supervision, step-level reward model, step-level verifier, step-aware verifier, outcome reward model, ORM, outcome supervision, PRM800K, Math-Shepherd, OmegaPRM, implicit PRM]
area: reasoning-and-test-time-compute/process-and-outcome-reward-models
level: core
summary: >-
  A reward model that scores every step of a chain of thought rather than only its final answer, trained on step labels from humans, rollouts or a judge, and used to rerank, search or train.
purpose: reasoning
uses: [reward-model, chain-of-thought-prompting, cross-entropy-loss@al, bernoulli-distribution@m2, conditional-expectation@m2, beam-search@al]
related_to: [reward-model, self-consistency, reinforcement-learning-from-human-feedback, reward-hacking, hallucination, beam-search@al]
hyperparameters:
  - {name: step segmentation, meaning: "how a solution is cut into steps whose ends get a score", typical: "one step per line or per paragraph (double line break) for math"}
  - {name: label source, meaning: "where the step labels come from", typical: "human annotators (PRM800K); Monte Carlo rollouts (Math-Shepherd, OmegaPRM); an LLM judge; consensus of rollouts and judge"}
  - {name: rollouts per step, meaning: "the number N of completions sampled from a prefix to estimate a Monte Carlo label", typical: "8"}
  - {name: label type, meaning: "how rollouts become a label", typical: "soft (the share of completions that reach the right answer) or hard (1 if any does)"}
  - {name: aggregation, meaning: "how step scores become one solution score", typical: "product (Lightman et al.), minimum (Math-Shepherd), last step (Snell et al.)"}
  - {name: selection, meaning: "how the scores are used at inference", typical: "best-of-N; reward-weighted voting over answers; beam search over steps"}
cost:
  train: "a reward model the size of the policy or smaller, fine-tuned with cross-entropy on step labels; the labels cost human time (800K labels in PRM800K) or N rollouts per labelled step, reduced to about log M checks per solution by binary search"
  inference: "one forward pass over each candidate solution gives all its step scores; best-of-N costs N generations plus N PRM passes, and step-level search calls the PRM after every step"
history:
  year: 2022
  people: [Jonathan Uesato, Hunter Lightman, Karl Cobbe, Vineet Kosaraju, Yura Burda, Peiyi Wang, Liangchen Luo, Lifan Yuan, Chujie Zheng, Zhenru Zhang]
  note: "Cobbe et al. (2021) trained outcome verifiers for GSM8K; Uesato et al. (2022) compared outcome and process supervision on GSM8K; Lightman et al. (2023) released PRM800K and found process supervision ahead on MATH; Math-Shepherd (2023) and OmegaPRM (2024) replaced human labels with rollouts; implicit PRMs (2024) read step rewards off an outcome model; ProcessBench and Zhang et al. (2024 to 2025) measured the limits, and DeepSeek-R1 (2025) trained without a neural reward model."
prereqs: [reward-model, chain-of-thought-prompting, cross-entropy-loss@al]
same_as:
  wikidata: null
  wikipedia: null
  arxiv: ["2305.20050", "2211.14275", "2312.08935", "2406.06592", "2412.01981", "2412.06559", "2501.07301", "2408.03314", "2501.12948", "2110.14168"]
sources:
  - {paper: "Lightman, Kosaraju, Burda, Edwards and others (2023)", title: "Let's Verify Step by Step", locator: "arXiv 2305.20050v1, pp. 1 to 8, 10, 11 and 20", note: "outcome and process supervision on MATH, PRM800K, active learning, scoring and the results"}
  - {paper: "Uesato, Kushman, Kumar, Song and others (2022)", title: "Solving math word problems with process- and outcome-based feedback", locator: "arXiv 2211.14275v1, pp. 1, 2, 4 and 6", note: "the first comparison of process and outcome supervision, trace error and final-answer error"}
  - {paper: "Wang, Li, Shao, Xu and others (2023)", title: "Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations", locator: "arXiv 2312.08935v3, pp. 2, 4 and 5", note: "automatic step labels by completion, hard and soft estimation, step-by-step PPO"}
  - {paper: "Luo, Liu, Liu, Phatale and others (2024)", title: "Improve Mathematical Reasoning in Language Models by Automated Process Supervision", locator: "arXiv 2406.06592v2, pp. 1 and 5", note: "OmegaPRM, binary search for the first error"}
  - {paper: "Yuan, Li, Chen, Cui and others (2024)", title: "Free Process Rewards without Process Labels", locator: "arXiv 2412.01981v1, pp. 1, 3 and 4", note: "implicit PRMs from a log-likelihood-ratio outcome reward"}
  - {paper: "Zheng, Zhang, Zhang, Lin and others (2024)", title: "ProcessBench: Identifying Process Errors in Mathematical Reasoning", locator: "arXiv 2412.06559v4, pp. 1 and 6", note: "a benchmark for locating the first error, and how PRMs and critics do on it"}
  - {paper: "Zhang, Zheng, Wu, Zhang and others (2025)", title: "The Lessons of Developing Process Reward Models in Mathematical Reasoning", locator: "arXiv 2501.07301v2, pp. 1 to 3", note: "Monte Carlo labels against judges and humans, biases of best-of-N evaluation, consensus filtering"}
  - {paper: "Snell, Lee, Xu and Kumar (2024)", title: "Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters", locator: "arXiv 2408.03314v1, Section 5", note: "last-step aggregation and beam search against a PRM"}
  - {paper: "DeepSeek-AI (2025)", title: "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning", locator: "arXiv 2501.12948v2, pp. 62 to 63", note: "why PRMs were not used"}
  - {paper: "Cobbe, Kosaraju, Bavarian, Chen and others (2021)", title: "Training Verifiers to Solve Math Word Problems", locator: "arXiv 2110.14168v2, p. 10", note: "search against a verifier and adversarial solutions"}
---

A process reward model (PRM) reads a step-by-step solution and gives each step a score, the estimated probability that the reasoning so far is right.
An outcome reward model (ORM) gives one score to the whole solution, trained only on whether its final answer was right.
Scoring steps tells a search or a training loop where a solution went wrong, and in mathematics it picks correct solutions out of many samples more reliably than an ORM or a vote.
This page covers how PRMs are trained, where their step labels come from (humans, rollouts, judges, or nowhere), how their scores are used, what the evidence shows, and why some recent reasoning models do without them.

## In plain words

A teacher marking a maths exam can look only at the final answer, or can read the working and mark the first line where it goes wrong.
The second kind of marking takes longer but says more: a right answer reached by a wrong argument loses marks, and a wrong answer shows exactly which step to fix.
A PRM is a model trained to mark the working.
It reads the solution once and, at the end of every step, outputs a number between 0 and 1 for "everything up to here is still right".

Those numbers have three uses.
Sample many solutions and keep the one whose steps score best.
Build a solution step by step, generating several candidate next steps and keeping the ones the PRM likes.
Or use the scores as rewards in reinforcement learning, so the model is rewarded for each good step and not only for a lucky final answer.

The hard part is getting step labels to train on.
Final answers can be checked automatically; steps cannot, so early PRMs used human annotators, and later ones guess a step's quality from how often the model, continuing from that step, still reaches the right answer.

## Setting

Outcome-based approaches supervise the final result, and process-based approaches supervise each step of the reasoning process, including the last step outputting the final result. {#c-ue-two}
Outcome-supervised reward models (ORMs) are trained using only the final result of the model's chain-of-thought, while process-supervised reward models (PRMs) receive feedback for each step in the chain-of-thought. {#c-lm-def}
Process supervision provides more precise feedback, since it specifies the exact location of any errors that occur, and it is easier for humans to interpret. {#c-lm-why}
A PRM is a reward model in the sense of RLHF, a language model with a scoring head, but its training signal and its outputs are per step.

Write a solution as steps s_1, ..., s_M.
For the process-supervised reward model, the binary label after each step indicates whether the steps so far are correct. {#c-ue-label}
Later work has used different targets for the same position.
Given an instruction and an n-step response y, a PRM receives the instruction and the first t − 1 steps and assigns a reward to the t-th step; Lightman et al. define the process reward as the correctness of each step, while Wang et al. directly consider Q values as process rewards. {#c-im-q}
Whatever the target, the score at step t is computed from the problem and the first t steps only, so a PRM can be queried in the middle of generation.

The usual test of a reward model for reasoning is selection from samples.
Lightman et al. evaluate a reward model by its ability to perform best-of-N search over uniformly sampled solutions from the generator: for each test problem they select the solution ranked highest by the reward model and grade it by its final answer. {#c-lm-bon}

## The method

### Training on step labels

Lightman et al. train PRMs to predict the correctness of each step after the last token in each step; the prediction takes the form of a single token, and they maximize the log-likelihood of these target tokens. {#c-lm-tok}
To determine the step-level predictions at test time, it suffices to perform a single PRM forward pass over the whole solution. {#c-lm-oneshot}
A PRM is usually trained with the cross-entropy loss summed over the K reasoning steps, Σ_{i=1}^K y_{s_i} log r_{s_i} + (1 − y_{s_i}) log(1 − r_{s_i}), where r_{s_i} is the sigmoid score of the i-th step. {#c-sh-loss}
Lightman et al. also treat PRM training as a three-class classification of each step as good, neutral or bad, but Wang et al. found not much difference between the binary and the three classifications. {#c-sh-three}

Labels stop at the first mistake.
Lightman et al. deliberately supervise only up to the first incorrect step: for correct solutions outcome and process supervision provide the same information, and for incorrect solutions process supervision additionally reveals the precise location of the mistake. {#c-lm-first}
Steps after the first error are left unlabelled, because once a solution is wrong "the steps so far are correct" is false for every later step, whatever the later steps say.

### Labels from humans

Human data-labelers assign each step a label of positive, negative, or neutral, and the PRM800K training set contains 800K step-level labels across 75K solutions to 12K problems. {#c-lm-labels}
Labelling uniformly sampled solutions wastes effort on obvious mistakes.
Lightman et al. surface convincing wrong-answer solutions, rated highly by the current best PRM but reaching an incorrect final answer, to the labelers, since the PRM is mistaken about at least one step in each such solution. {#c-lm-active}
Active learning leads to a 2.6× improvement in the data efficiency of process supervision. {#c-lm-26}

### Labels from rollouts

Human labels are expensive, so most later PRMs estimate step quality by continuing from the step.
Wang et al. define the quality of a reasoning step as its potential to deduce the correct answer, and estimate it by having a completer finalize N reasoning processes from the step. {#c-sh-def}
Hard estimation supposes that a reasoning step is good as long as it can reach the correct answer, and soft estimation takes the quality of a step as the frequency with which it reaches the correct answer. {#c-sh-he}
For hard labels, Zhang et al. treat a step as correct if any one of the 8 completions yields the correct final answer; for soft labels the value is the proportion of completions leading to the correct final answers. {#c-ls-hard}
The two labels train different things.
Under cross-entropy, the expected loss for a label y with mean μ, −(μ log r + (1 − μ) log(1 − r)), is minimized at r = μ, so a PRM trained on soft labels estimates the probability q that a completion from the step reaches the correct answer. {#c-pr-ce}
With N completions the hard label is 1 with probability 1 − (1 − q)^N, so a PRM trained on hard labels learns that probability rather than q: with N = 8, a step from which one completion in ten succeeds is labelled correct with probability 1 − 0.9^8 ≈ 0.570. {#c-pr-hard}
The soft label is an average of N Bernoulli(q) draws, unbiased for q with variance q(1 − q)/N, at most 1/(4N), so 8 completions give a standard deviation of up to about 0.18. {#c-pr-var}

Rolling out from every step of every solution is costly.
OmegaPRM identifies the first error in the chain of thought with binary search over Monte Carlo rollouts and balances the positive and negative examples. {#c-om-binary}
When the remaining part of a solution is short enough to be considered as a single step, binary search locates the first error with a time complexity of O(k log M) rather than O(kM) in the brute-force setting, where M is the total number of steps. {#c-om-logm}
If every prefix before the first error has a correct rollout and none after it does, a binary search over prefixes finds the first wrong step of an M-step solution with ⌈log2(M + 1)⌉ prefix checks, against up to M checks step by step: 5 checks instead of up to 20 for M = 20. {#c-pr-bsearch}
OmegaPRM collects over 1.5 million process supervision annotations, and with weighted self-consistency raises the instruction-tuned Gemini Pro from 51% to 69.4% on MATH500. {#c-om-scale}

Rollout labels confuse two things: whether a step is right and whether the completer can recover from it.
Monte Carlo estimation-based data synthesis for PRMs typically yields inferior performance and generalization compared to LLM-as-a-judge and human annotation methods. {#c-ls-mc}
MC estimation relies on completion models to evaluate current-step correctness, which can generate correct answers from incorrect steps or incorrect answers from correct steps. {#c-ls-why}
Under consensus filtering, instances are only retained when both LLM-as-a-judge and MC estimation show consensus on the error reasoning step locations in the solution. {#c-ls-consensus}

### Process rewards without step labels

An implicit PRM can be obtained at no additional cost by training an ORM on response-level labels, with the only assumption that the outcome reward is parameterized as the log-likelihood ratio of the policy and reference models, r_θ(y) = β log π_θ(y)/π_ref(y). {#c-im-abs}
Proposition 3.1: if the reward is parameterized by the log-likelihood ratio of two causal LMs, r_θ(y) := β log π_θ(y)/π_ref(y), and q_θ^t is the partial log-ratio over the first t steps, then q_θ^t = β log E_{π_ref(y | y≤t)} e^{r_θ(y)/β}; q_θ is the exponential average of r_θ at step t. {#c-im-prop}
If the process reward of step t is the difference q^t − q^(t−1) of these partial sums, the process rewards of a response add up to its outcome reward r(y), because the sum telescopes and q^0 = 0. {#c-pr-tele}
The implicit PRM outperforms a strong MCTS-based baseline in the style of Math-Shepherd using less than 1/38 of the training data. {#c-im-38}
The step rewards come from the same forward passes that score the whole response, so this PRM costs nothing beyond the outcome model.

### Turning step scores into decisions

A solution needs one score to be compared with others.
Lightman et al. define the PRM score for a solution to be the probability that every step is correct under the PRM, implemented as the product of the correctness probabilities for each step. {#c-lm-prod}
Of the scoring strategies Lightman et al. tried, taking the product of step-level scores and treating neutral steps as positive did best, but the difference in performance between all strategies is minor. {#c-lm-appf}
Following Lightman et al., Wang et al. use the minimum score across all steps to represent the final score of a solution assigned by the PRM. {#c-sh-min}
Snell et al. use the PRM's prediction at the last step as the full-answer score rather than the product or minimum, and attribute the difference from earlier work to their verifier being trained with soft Monte Carlo return labels. {#c-sn-last}
For step scores in (0, 1], the product is at most the minimum, which is at most the last score; the product falls with the number of steps, so ten steps at 0.95 score about 0.599 by product while one step at 0.7 scores 0.7. {#c-pr-agg}

Scores can also be combined with a vote over final answers, as in self-consistency.
Solutions can be grouped by final answer and scored per group, selecting arg max_a Σ_{i=1}^N 1(a_i = a) · RM(p, S_i), which combines self-consistency with the reward model. {#c-sh-vote}

Inside generation, the PRM guides a search over steps.
Used during generation, the PRM selects among K candidate steps: at each step K candidates are generated, the candidate with the highest PRM score is kept, and generation continues from it until the final answer, with K = 96. {#c-ue-select}
Beam search against a PRM is more effective on harder questions and at lower compute budgets, whereas best-of-N is more effective on easier questions and at higher budgets. {#c-sn-beam}
In reinforcement learning, it supplies dense rewards.
Step-by-step PPO offers rewards at the end of each reasoning step, unlike PPO with an ORM, which only offers a reward at the end of the response. {#c-sh-ppo}

## Derivation

The full derivation is in `derivations/process-reward-arithmetic.md`, and `process-reward-arithmetic.check.py` checks every step numerically.

**Cross-entropy targets.** The expected cross-entropy −(μ log r + (1 − μ) log(1 − r)) has derivative −μ/r + (1 − μ)/(1 − r), zero only at r = μ, so a PRM learns the mean of its labels.

**Soft and hard labels.** With N independent rollouts that succeed with probability q, the soft label C/N has mean q and variance q(1 − q)/N, and the hard label 1(C ≥ 1) has mean 1 − (1 − q)^N, which exceeds q.

**Binary search.** "The prefix of length t is still right" is true for t below the first error and false from it on, so the first error is found by halving, with ⌈log2(M + 1)⌉ checks.

**Implicit rewards.** Both models factor over steps, so the partial log-ratios q^t telescope to the outcome reward, and the expectation of the remaining ratio under the reference model is 1, which gives q^t = β log E_ref e^{r/β}.

**Aggregation.** For scores in (0, 1] the product is at most the minimum, which is at most the last score, and each added step can only lower the product.

**F1.** The harmonic mean 2xy/(x + y) of the accuracies on erroneous and correct solutions is 0 if either is 0 and lies below their average otherwise.

## Worked example

A model writes 16 solutions to a problem with 6 steps each.
Each step is right with probability 0.9 until the first mistake, and a solution with a mistake still lands on the right answer 15% of the time.
A single sample is fully right with probability 0.9^6 ≈ 0.53 and has the right answer with probability 0.53 + 0.47 × 0.15 ≈ 0.60.
An ORM trained on final answers learns to prefer the 60%, including the lucky ones; a PRM trained on first-error labels learns to prefer the 53% with clean working, and with the minimum step score as the solution score it rarely picks a solution with a low-scoring step.

To label one of those solutions by rollouts, sample 8 completions from each prefix.
A prefix whose true success rate is 0.1 gets a soft label near 0.1, give or take up to about 0.18, but a hard label of 1 with probability 0.57.
Finding the first error by binary search over 6 steps needs ⌈log2 7⌉ = 3 prefix checks instead of up to 6.

The NumPy tests run this world: they check the label means, train a logistic PRM by cross-entropy, and confirm that best-of-16 under the PRM picks clean solutions more often than picking any solution with the right answer.
The PyTorch tests train a small recurrent PRM that reads a whole solution in one pass and scores every step, check that best-of-8 under it finds clean solutions far more often than a single sample, and verify the implicit-reward identity by enumerating every sequence of a tiny model.

## Hyperparameters

The step boundary matters more than it looks: steps must be units a labeller or a rollout can judge, which in mathematics means lines or paragraphs, and outside mathematics there may be no natural unit.
For rollout labels, 8 completions per step is common; more rollouts cut the variance of soft labels, while hard labels become more lenient as N grows.
Soft labels estimate the completer's success rate and suit last-step aggregation; hard or human labels mark correctness and suit the product or minimum.
The aggregation should match how the PRM was trained, and the gains between product, minimum and last are small next to the gains from better labels.
For search, beam width and the number of candidates per step trade compute against the risk of searching the PRM's errors.

## Cost

Training costs a fine-tuning run of a reward model, plus the labels.
Human step labels are the costly part: Lightman et al. collected 800K of them and still needed active learning to make them count.
Rollout labels replace people with compute: N completions per labelled step, or about N log M per solution with binary search.
At inference one PRM forward pass scores every step of a solution, so best-of-N with a PRM costs about the same as with an ORM; step-level search calls the PRM after every step and costs more.

## Evidence

Process supervision significantly outperforms outcome supervision for training models to solve problems from the MATH dataset, and the process-supervised model solves 78% of problems from a representative subset of the MATH test set. {#c-lm-main}
Best-of-1860 on the MATH subset, the PRM solves 78.2% of problems, the ORM 72.4% and majority voting 69.6%, and the gap between the PRM and the others widens as N increases. {#c-lm-fig3}
RM-weighted voting, which combines the PRM with majority voting, did not noticeably improve performance in Lightman et al.'s experiments. {#c-lm-rmvote}
On held-out STEM questions from recent AP Physics, AP Calculus, AP Chemistry, AMC10 and AMC12 exams, the best-of-100 rates were 72.9% for the PRM, 63.8% for the ORM and 61.3% for majority voting. {#c-lm-ood}

The earlier comparison on GSM8K was closer.
Outcome-based and process-based approaches lead to similar final-answer error rates on GSM8K, both without reward models (23.5% vs. 22.3%) and with reward models (16.6% vs. 14.8%). {#c-ue-similar}
Even reward models trained with outcome-based labels give predictions that agree more closely with the process-based labels than with the outcome-based labels themselves. {#c-ue-emulate}
All models using reinforcement learning directly against final-answer correctness had high trace error, at best 12.4% against 3.8% for the best process-based method, and reinforcement learning against a reward model reduced it to 5.5%. {#c-ue-trace}

With labels from rollouts instead of people, PRMs also help as rewards in reinforcement learning.
Step-by-step PPO with the automatically trained PRM raises Mistral-7B from 77.9% to 84.1% on GSM8K and from 28.6% to 33.0% on MATH, and verification lifts these to 89.1% and 43.5%. {#c-sh-res}
Lightman et al. argue that process supervision in fact incurs a negative alignment tax, since it improves performance while rewarding a human-endorsed chain of thought. {#c-lm-tax}

## Limitations and failure modes

Selecting the right final answer is not the same as finding the first error, and benchmarks built for the second task found PRMs weaker than best-of-N suggested.
ProcessBench consists of 3,400 test cases of competition- and Olympiad-level math problems with error locations annotated by human experts, and models must identify the earliest step that contains an error or conclude that all steps are correct. {#c-pb-task}
ProcessBench computes the accuracies on erroneous and correct samples and their harmonic mean as the F1 score, which balances being overly critical against being incapable of identifying errors. {#c-pb-f1}
A model that flags an error in every solution scores 0 on the correct samples, and a model that never flags one scores 0 on the erroneous samples, so either way its harmonic mean is 0; accuracies of 0.9 and 0.5 give 2 · 0.9 · 0.5/1.4 ≈ 0.643. {#c-pr-f1}
Existing PRMs typically fail to generalize to more challenging math problems beyond GSM8K and MATH, and underperform both critic models, which are prompted general language models, and a PRM straightforwardly fine-tuned on PRM800K. {#c-pb-fail}
Best-of-N evaluation of PRMs is biased: policy models generate responses with correct answers but flawed processes, PRMs that tolerate them get inflated scores, and many PRMs place their minimum scores on the final answer steps, a shift from process to outcome-based assessment. {#c-ls-bon}

The labels are noisy in both directions.
Grading by final answer is not perfectly reliable: false positives, solutions that reach the correct answer with incorrect reasoning, will be misgraded. {#c-lm-fp}
Rollout labels add the completer's errors, and human labels are expensive and need experts for hard problems.

A PRM is a learned model, so optimizing against it invites reward hacking.
Against a verifier, search beyond a certain number of samples can lower accuracy, as the risk of finding adversarial solutions that fool the verifier outweighs the benefits of search. {#c-cob-adv}
DeepSeek-R1's authors list three limitations of PRMs: it is challenging to define a fine-grain step in general reasoning, determining whether an intermediate step is correct is challenging, and a model-based PRM leads to reward hacking and needs extra training resources. {#c-r1-three}
A PRM also inherits the hallucination problem it is meant to catch: it can score a fluent but false step as correct, and its errors are correlated with those of the generator when both are trained on similar data.

## Variants

**Outcome reward model (ORM).** One score per solution trained on final-answer correctness; cheap to label, and often a baseline for PRMs.

**Human-labelled PRM.** Step labels from annotators, as in PRM800K, usually only up to the first error.

**Monte Carlo PRM.** Step labels from the success rate of completions, hard or soft (Math-Shepherd), found efficiently by binary search and tree search (OmegaPRM).

**Judge-labelled and consensus PRMs.** Step labels from a prompted language model, or only where the judge and the rollouts agree.

**Implicit PRM.** An outcome model parameterized as a log-likelihood ratio, whose per-token ratios give step rewards with no step labels.

**Generative verifiers and critics.** A language model prompted or trained to write a critique of each step and a verdict, instead of a scalar head.

**PRM-guided search.** Best-of-N, reward-weighted voting, step-level beam search and lookahead search with the PRM as value estimate.

## When to use it

Use a PRM when solutions come as steps that can be judged one at a time, final answers can be checked to make labels, and you sample many candidates or search over steps.
Prefer rule-based outcome rewards (an answer checker or unit tests) for reinforcement learning at scale when they exist, since they cannot be hacked the way a learned model can.
Prefer reward-weighted voting or plain self-consistency when the PRM has not been tested on the target distribution, and evaluate a PRM on locating errors as well as on best-of-N.

## Pitfalls

* Training with hard rollout labels and reading the scores as probabilities of correctness: they overstate them.
* Evaluating only best-of-N accuracy, which rewards a PRM that ignores the working and checks the final answer.
* Using the product of step scores for solutions of very different lengths without checking for a bias towards short ones.
* Searching or optimizing hard against the PRM, which finds solutions that fool it.
* Labelling steps after the first error, which gives contradictory targets.
* Assuming a PRM trained on grade-school or MATH problems transfers to olympiad problems or to other domains.

## History

Cobbe et al. (2021) trained verifiers that scored whole GSM8K solutions, outcome reward models in later terms, and used them to pick among samples.
Uesato et al. (2022) compared outcome and process supervision on GSM8K and found similar final-answer error but much lower reasoning error with process-based feedback.
Lightman et al. (2023) repeated the comparison on MATH with GPT-4-based models, found process supervision clearly ahead, and released PRM800K, which made step-level reward models a common component of reasoning systems.
Math-Shepherd (Wang et al., 2023) and OmegaPRM (Luo et al., 2024) replaced annotators with rollouts, and Yuan et al. (2024) derived step rewards from an outcome model with no step labels at all.
ProcessBench and Zhang et al. (2024 to 2025) then measured what PRMs actually detect and found that best-of-N scores had flattered them.
At the same time, DeepSeek-R1 (2025) and other reasoning models trained by reinforcement learning with rule-based outcome rewards, citing the cost of step labels and the risk of reward hacking, so PRMs are now used mostly for search and selection at inference.
