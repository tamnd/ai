---
tag: NEW-self-consistency
kind: method
slug: self-consistency
name: Self-consistency
aliases: [self-consistency decoding, majority voting, maj@k, sample-and-vote, Universal Self-Consistency, USC, Adaptive-Consistency, early-stopping self-consistency, ESC, Agent Forest]
area: reasoning-and-test-time-compute/self-consistency-and-voting
level: core
summary: >-
  Sampling several chains of thought for one question and returning the final answer that most of them reach, instead of the answer of one greedy chain.
purpose: reasoning
uses: [chain-of-thought-prompting, nucleus-sampling, ancestral-sampling@al, binomial-distribution@m2, hoeffdings-inequality@m2, weak-law-of-large-numbers@m2]
related_to: [chain-of-thought-prompting, reward-model, neural-scaling-laws, hallucination, beam-search@al, ensemble-learning@al]
hyperparameters:
  - {name: samples, meaning: "the number m of reasoning paths sampled per question", typical: "40 in Wang et al.; 5 to 10 for most of the gain; hundreds to thousands in repeated-sampling studies"}
  - {name: temperature and truncation, meaning: "the sampling temperature T and the top-k or nucleus cut-off that make the paths differ", typical: "T = 0.5 to 0.7 with top-k 40 in Wang et al."}
  - {name: answer parser, meaning: "the rule that reads the final answer off a path so that answers can be compared", typical: "the first number after The answer is for arithmetic; an option letter for multiple choice; program outputs on test inputs for code"}
  - {name: weighting, meaning: "how much each sampled path counts in the vote", typical: "one vote each; length-normalized sequence probability; a verifier's probability that the path is correct"}
  - {name: stopping rule, meaning: "when to stop sampling for one question", typical: "a fixed m; Beta confidence above 0.95 (Adaptive-Consistency); a unanimous window of w samples (ESC)"}
cost:
  train: "none; the method works on a pretrained or instruction-tuned model as it is"
  inference: "m times the generation of one chain of thought per question, less with a stopping rule; the vote itself is a count"
history:
  year: 2022
  people: [Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, Denny Zhou]
  note: "Cobbe et al. (2021) voted among verifier-ranked samples on GSM8K; Wang et al. (arXiv 2203.11171, ICLR 2023) named self-consistency and voted over plain samples; 2023 and 2024 work added free-form answers (Universal Self-Consistency), stopping rules (Adaptive-Consistency, ESC) and scaling studies (More Agents, Large Language Monkeys, Chen et al.)."
prereqs: [chain-of-thought-prompting, ancestral-sampling@al, binomial-distribution@m2]
same_as:
  wikidata: null
  wikipedia: null
  arxiv: ["2203.11171", "2110.14168", "2311.17311", "2305.11860", "2401.10480", "2402.05120", "2403.02419", "2407.21787", "2206.02336"]
sources:
  - {paper: "Wang, Wei, Schuurmans, Le and others (2023)", title: "Self-Consistency Improves Chain of Thought Reasoning in Language Models", locator: "arXiv 2203.11171v4, pp. 1 to 9", note: "the method, the aggregation strategies, the sampling settings, the results, the ablations and the limitations"}
  - {paper: "Chen, Aksitov, Alon, Ren and others (2023)", title: "Universal Self-Consistency for Large Language Model Generation", locator: "arXiv 2311.17311v1, pp. 1, 2, 6 and 7", note: "letting the model choose the most consistent free-form response"}
  - {paper: "Aggarwal, Madaan, Yang and Mausam (2023)", title: "Let's Sample Step by Step: Adaptive-Consistency for Efficient Reasoning and Coding with LLMs", locator: "arXiv 2305.11860v2, pp. 1 and 3 to 5", note: "the Beta stopping criterion and the sample savings"}
  - {paper: "Li, Yuan, Feng, Pan and others (2024)", title: "Escape Sky-high Cost: Early-stopping Self-Consistency for Multi-step Reasoning", locator: "arXiv 2401.10480v1, pp. 1 to 3", note: "window-based early stopping and its savings"}
  - {paper: "Li, Zhang, Yu, Fu and others (2024)", title: "More Agents Is All You Need", locator: "arXiv 2402.05120v2, pp. 1, 6 and 10", note: "sampling and voting at larger ensemble sizes, small models against large, difficulty"}
  - {paper: "Chen, Davis, Hanin, Bailis and others (2024)", title: "Are More LM Calls All You Need? Towards the Scaling Properties of Compound AI Systems", locator: "arXiv 2403.02419v2, pp. 1 and 5", note: "non-monotone voting accuracy and query difficulty"}
  - {paper: "Brown, Juravsky, Ehrlich, Clark and others (2024)", title: "Large Language Monkeys: Scaling Inference Compute with Repeated Sampling", locator: "arXiv 2407.21787v3, pp. 1 and 3", note: "coverage against majority voting at large sample counts"}
  - {paper: "Li, Lin, Zhang, Fu and others (2023)", title: "Making Large Language Models Better Reasoners with Step-Aware Verifier", locator: "arXiv 2206.02336v3, pp. 1 and 2", note: "DIVERSE, verifier-weighted voting over diverse prompts"}
  - {paper: "Cobbe, Kosaraju, Bavarian, Chen and others (2021)", title: "Training Verifiers to Solve Math Word Problems", locator: "arXiv 2110.14168v2, pp. 10 and 11", note: "voting among the top verifier-ranked samples and the limit of search against a verifier"}
---

Self-consistency asks a language model the same question several times, lets it write a different chain of thought each time, and returns the final answer that most of the chains reach.
It replaces the single greedy chain of chain-of-thought prompting with a vote, needs no training, and was one of the first ways of turning extra inference compute into accuracy.
The vote converges to the most probable answer of the model, with the reasoning summed out, which helps when the model is more often right than wrong in any one way and hurts when it is not.
This page covers the vote and its weightings, rules that stop sampling early, extensions to answers that cannot be compared exactly, what the evidence shows, and where voting stops scaling.

## In plain words

Ask a student a hard arithmetic question once and the answer may be off because of one slip.
Ask five students, or the same student five times on different days, and if three of them get 18 while the others get 26 and 14, 18 is a good bet: slips tend to scatter over different wrong answers, while correct work tends to land on the same answer.
Self-consistency does this with one model.
It samples, rather than takes the most probable word at each step, so each run writes a different line of reasoning; it reads the final answer off each run; and it returns the answer that came up most often.

Two conditions make it work.
The answers must be comparable, a number or an option letter rather than an essay, so that "the same answer" means something.
And the model must put more probability on the right answer than on any single wrong one; voting then makes the right answer more and more likely as samples are added, and if a wrong answer is more probable, voting makes that wrong answer more and more likely instead.

## Setting

Self-consistency is a decoding strategy that replaces the naive greedy decoding used in chain-of-thought prompting: it first samples a diverse set of reasoning paths, and then selects the most consistent answer by marginalizing out the sampled reasoning paths. {#c-wang-define}
Self-consistency leverages the intuition that a complex reasoning problem typically admits multiple different ways of thinking leading to its unique correct answer. {#c-wang-intuition}
Wang et al. hypothesize that correct reasoning processes, even if they are diverse, tend to have greater agreement in their final answer than incorrect processes. {#c-wang-hypothesis}

Formally, a prompt with few-shot exemplars and a question define a distribution over outputs.
Given a prompt and a question, self-consistency introduces a latent variable r_i, a sequence of tokens representing the reasoning path in the i-th output, and couples the generation of (r_i, a_i) where r_i → a_i. {#c-wang-latent}
After sampling multiple (r_i, a_i), self-consistency applies a marginalization over r_i by taking a majority vote over a_i, that is arg max_a Σ_{i=1}^m 1(a_i = a). {#c-wang-argmax}
The parser is task dependent: for arithmetic reasoning, Wang et al. parse the first numerical part as the final answer after the model generates "The answer is ". {#c-wang-parser}
Self-consistency can be applied only to problems where the final answer is from a fixed answer set, but in principle the approach can be extended to open-text generation if a good metric of consistency can be defined between multiple generations. {#c-wang-fixed}

The m samples come from one model, which makes the method different from ensemble learning.
Self-consistency differs from a typical ensemble approach where multiple models are trained and their outputs aggregated; it acts more like a self-ensemble that works on top of a single language model. {#c-wang-selfens}
Self-consistency is entirely unsupervised, works off-the-shelf with pre-trained language models, requires no additional human annotation, and avoids any additional training, auxiliary models or fine-tuning. {#c-wang-unsup}

What the vote estimates follows from the law of large numbers.
As the number of samples m grows, the share of samples with answer a converges to the model's marginal probability P(a), so the majority vote converges to arg max_a P(a), the mode of the answer distribution with the reasoning paths summed out. {#c-sc-lln}
Li et al. re-read self-consistency as mitigating the noise of individual sampling through multiple samplings, with the objective that the prediction with the highest probability, arg max_i P(i), is chosen as the final answer. {#c-esc-lln}
This is not the same as the answer of the most probable single path, which is what greedy or beam search decoding approximates: many paths, each less likely than the best one, can share an answer and together outweigh it.

## The method

### Sampling and voting

The procedure has three steps: prompt with chain-of-thought exemplars, sample m outputs with a temperature above zero, and vote over the parsed answers.
For UL2-20B and LaMDA-137B Wang et al. applied temperature sampling with T = 0.5 and truncated at the top-k (k = 40) tokens, for PaLM-540B they applied T = 0.7, k = 40, and for GPT-3 they use T = 0.7 without top-k truncation. {#c-wang-sampling}
Wang et al. report the results of self-consistency averaged over 10 runs, where they sampled 40 outputs independently from the decoder in each run. {#c-wang-40}
The paths are independent, so all m can be generated in one batch.

### Weighting the vote

The vote can count every path once or weight it.
Each (r_i, a_i) can instead be weighted by P(r_i, a_i | prompt, question), either unnormalized or normalized by the output length as exp of (1/K) Σ_k log P(t_k | prompt, question, t_1, ..., t_{k−1}). {#c-wang-norm}
Taking the unweighted sum, a majority vote directly over a_i, yields a very similar accuracy to aggregating with the normalized weighted sum. {#c-wang-unweighted}
The normalized conditional probabilities P(r_i, a_i | prompt, question) are quite close to each other, so the language model regards those generations as similarly likely. {#c-wang-similar}
This also means that the language model is not well calibrated and cannot distinguish well between correct and wrong solutions, which explains why additional re-rankers were trained to judge solutions in previous work. {#c-wang-calib}
The normalized weighted sum yields a much higher accuracy than its unnormalized counterpart, and a weighted average, each answer's weighted sum divided by its count, gives a much worse performance. {#c-wang-unnorm}
A weighted vote picks arg max_a Σ_i w_i 1(a_i = a); with equal weights it is the majority vote, and with the unnormalized sequence probability as weight a path's weight shrinks geometrically with its length, so the vote favours short paths. {#c-sc-weighted}
The plain vote is therefore the default, and weights that do help come from a separately trained judge, such as a verifier or reward model, rather than from the generating model's own probabilities.

### Sampling settings and other decoders

Self-consistency is robust to sampling strategies and parameters, varying T in temperature sampling, k in top-k sampling and p in nucleus sampling over PaLM-540B. {#c-wang-robust}
Diversity is what the method needs, and decoders that remove it lose the gain.
Self-consistency can also adopt beam search to decode each reasoning path, but its performance is worse than self-consistency with sampling, because beam search yields a lower diversity in the outputs. {#c-wang-beam}
Sample-and-rank, which ranks sampled sequences by their log probability and takes the answer of the top-ranked one, improves the accuracy with more samples, but the gain is much smaller than that of self-consistency. {#c-wang-rank}

### Stopping early

A fixed m spends as much on an easy question, where every sample agrees, as on a hard one.
Existing self-consistency methods generate a constant number of samples per question, where a better approach is to distribute the budget non-uniformly based on the agreement in the samples generated so far. {#c-ada-fixed}
After generating each sample, Adaptive-Consistency employs a lightweight stopping criterion to decide whether to generate an additional sample or to cease sampling and report the current majority answer. {#c-ada-loop}
As an approximation, Adaptive-Consistency checks only the majority of p1 over the next largest probability p2, which simplifies to a Beta distribution with parameters (v1 + 1, v2 + 1) under a non-informative Beta(1, 1) prior. {#c-ada-beta}
The only hyperparameters in Adaptive-Consistency are those of the stopping criterion, and Aggarwal et al. use a high threshold C_thresh = 0.95. {#c-ada-thresh}
Under the Beta stopping rule with threshold 0.95, counts of 9 against 1 give a confidence of 1 − 12/2048 ≈ 0.994 and stop, a unanimous run stops after 4 samples (1 − 1/32 ≈ 0.969) but not after 3 (0.9375), and counts of 6 against 2 give 0.910 and go on. {#c-sc-betastop}
Over 17 reasoning and code generation datasets and three LLMs, Adaptive-Consistency reduces the sample budget by up to 7.9 times with an average accuracy drop of less than 0.1%. {#c-ada-result}

Early-stopping self-consistency (ESC) checks agreement in blocks instead of after every sample.
Early-stopping self-consistency divides the large sample size into several sequential small windows and stops sampling when the answers within a window are all the same, that is when the entropy of the predicted answer distribution is zero. {#c-esc-window}
A window of w samples is unanimous on an answer other than the mode with probability at most 2^−(w−1): every such answer has P(a) ≤ 1/2, so Σ_a P(a)^w ≤ 2^−(w−1) Σ_a P(a) ≤ 2^−(w−1), which is 1/128 for w = 8. {#c-sc-escbound}
A window is unanimous with probability q = Σ_a P(a)^w, so with no cap on the number of windows the expected number of samples is w/q; for answer probabilities 0.9, 0.08 and 0.02 and w = 5, q ≈ 0.590 and the expected cost is about 8.5 samples. {#c-sc-escstop}
ESC reduces the average number of samples of chain-of-thought reasoning on MATH (−33.8%), GSM8K (−80.1%), StrategyQA (−76.8%), Coin Flip (−84.2%) and Last Letters (−67.4%), while attaining comparable performance. {#c-esc-results}
Testing the entire MATH dataset with self-consistency at a sampling size of 64 costs about 2000$ through the GPT-4 API. {#c-esc-cost}

### Answers that cannot be compared exactly

Self-consistency relies on an answer extraction process to aggregate multiple solutions, which is not applicable to free-form answers. {#c-usc-limit}
Universal Self-Consistency (USC) hands the comparison to the model.
Universal Self-Consistency concatenates all sampled responses and constructs a prompt with an instruction asking the language model to select the most consistent response. {#c-usc-method}
On GSM8K and MATH, USC generally matches the performance of the standard self-consistency, and on text-to-SQL and Python code generation it matches execution-based consistency without requiring execution results. {#c-usc-math}
USC also improves the performance for open-ended question answering and long-context summarization, where the standard self-consistency is not applicable. {#c-usc-open}
USC does not further improve the performance on SummScreen after 5 samples, and the accuracy on GSM8K decreases with 16 samples, which can be due to weakness in long-context understanding and the imperfect counting ability of LLMs. {#c-usc-16}
Asking the LLM to choose the most detailed response rather than the most consistent one results in gains of about 2 ROUGE-1 and ROUGE-Lsum points on summarization. {#c-usc-detailed}
For programs, agreement can be measured by running them.
For code generation, execution-based consistency executes all predicted programs on the given inputs, clusters programs with the same execution outputs, and selects a program from the largest cluster. {#c-code-exec}

### Votes weighted by a verifier

A trained verifier can supply the weights that the model's own probabilities cannot.
DIVERSE's voting verifier selects ŷ = arg max_y Σ_{i=1}^M 1(y_i = y) · f(x_i, z_i, y_i), where f is the probability produced by the verifier that the reasoning path leads to the correct answer. {#c-div-vote}
Majority voting may fail when the majority of the reasoning paths are misled while the minority of the reasoning paths are reasonable. {#c-div-why}
With the language model code-davinci-002, DIVERSE achieves new results on GSM8K (74.4% → 83.2%). {#c-div-gsm}
Cobbe et al. take a majority vote among the top verifier-ranked solutions; with 100 samples it is optimal to let only the top 3 to 5 samples vote, and with 3200 samples about the top 30. {#c-cob-topk}

## Derivation

The full derivation is in `derivations/self-consistency-voting.md`, and `self-consistency-voting.check.py` checks every step numerically.

**The limit.** Each indicator 1(a_i = a) has mean P(a), so by the weak law of large numbers every share N_a/m converges to P(a); once all shares are within half the gap between the first and second answer, the vote returns the mode.

**Two answers.** With the correct answer sampled with probability p and odd m, the vote is right with probability Σ_{j>m/2} C(m, j) p^j (1 − p)^(m−j), which tends to 1 if p > 1/2 and to 0 if p < 1/2.

**An exponential bound.** For a competitor a, X_i = 1(a_i = y) − 1(a_i = a) has mean at least δ and range 2, so Hoeffding's inequality bounds P(N_a ≥ N_y) by exp(−mδ²/2), and a union bound covers all competitors.

**The Beta rule.** For integer parameters, P(q > 1/2) under Beta(v1 + 1, v2 + 1) equals 1 − P(Bin(v1 + v2 + 1, 1/2) ≥ v1 + 1), which is exact and cheap to evaluate.

**Windows.** A window is unanimous with probability Σ_a P(a)^w; every non-mode answer has probability at most 1/2, which bounds the wrong-answer part by 2^−(w−1), and the number of windows is geometric.

**Coverage.** Some sample among k is correct with probability 1 − (1 − p)^k, and 1 − C(n − c, k)/C(n, k) is the probability that a random k-subset of n samples with c correct contains a correct one, whose expectation is 1 − (1 − p)^k.

## Worked example

A model answers a word problem with 0.9 probability on 18, 0.08 on 26 and 0.02 on 14 when sampled at the chosen temperature.
Greedy decoding gives one path, which may end in any of these answers, depending on the first few tokens it commits to.
Sampling 40 paths gives about 36 votes for 18, and by the Hoeffding bound with δ = 0.82 the vote is wrong with probability at most 2 exp(−40 × 0.82²/2) ≈ 3e-6.
Adaptive-Consistency stops after the first four samples if they agree, which happens with probability 0.9^4 ≈ 0.66; ESC with w = 5 needs about 8.5 samples on average.

A harder question has 0.4 on the correct 7 and 0.6 on the wrong 9.
A single sample is right 40% of the time, but the vote converges on 9: it is right with probability 0.317 at m = 5 and 0.097 at m = 41.
Coverage behaves the other way: some sample among 10 is correct with probability 1 − 0.6^10 ≈ 0.994, so a verifier that could recognize 7 would find it.

The NumPy tests run both questions through the stopping rules, check the exact vote accuracy against simulation and the Hoeffding bound, and confirm that the pass@k estimator is unbiased.
The PyTorch tests sample from a small recurrent model, enumerate every path to get the exact answer distribution, check that vote frequencies match it, and find seeds where the vote and the greedy answer differ.

## Hyperparameters

The number of samples sets the cost; most of the accuracy comes from the first 5 to 10, and the rest of the curve flattens, so a stopping rule that spends samples only on questions where they disagree is usually better than a larger fixed m.
The temperature has to be high enough for the paths to differ and low enough for them to stay sensible; 0.5 to 0.7 with top-k or nucleus truncation is the common range, and the results are not sensitive within it.
The parser decides what counts as the same answer; normalizing numbers, units and option letters before the vote matters as much as the model.
Unweighted votes are the default; weights from a trained verifier help, and the model's own sequence probabilities, if used, must be length-normalized.
For a stopping rule the threshold trades cost against agreement with the full vote: 0.95 for the Beta rule and windows of 5 to 8 for ESC keep almost all of the accuracy.

## Cost

There is no training cost.
Inference costs m chains of thought per question instead of one; with batched decoding the extra samples share the prompt's key-value cache, so the prefill is paid once, but decoding scales linearly in m.
Stopping rules bring the average down on easy questions, which are usually the majority, and keep the full budget for hard ones.
Universal Self-Consistency adds one more call whose prompt contains all m responses, and execution-based voting adds the cost of running every program.

## Evidence

Self-consistency boosts chain-of-thought prompting on GSM8K (+17.9%), SVAMP (+11.0%), AQuA (+12.2%), StrategyQA (+6.4%) and ARC-challenge (+3.9%). {#c-wang-gains}
The gains become more significant as the language model's scale increases: +3% to 6% absolute accuracy over UL2-20B but +9% to 23% for LaMDA-137B and GPT-3. {#c-wang-scale}
Sampling a higher number of reasoning paths (1, 5, 10, 20, 40) leads to consistently better performance. {#c-wang-paths}
For some tasks such as ANLI-R1, e-SNLI and RTE, adding chain-of-thought hurts performance compared to standard prompting, but self-consistency boosts the performance and outperforms standard prompting. {#c-wang-hurt}
Self-consistency also works for zero-shot chain-of-thought and improves the results significantly (+26.2%). {#c-wang-zeroshot}
With equations as reasoning paths self-consistency still improves accuracy, but the gain is smaller since the equations are much shorter and leave less opportunity for diversity in decoding. {#c-wang-equations}
The consistency, the percentage of decodes agreeing with the final aggregated answer, is highly correlated with accuracy, so low consistency can be used as an indicator that the model has low confidence. {#c-wang-uncert}

Later work pushed the number of samples higher.
Through a sampling-and-voting method, called Agent Forest, the performance of large language models scales with the number of agents instantiated. {#c-mag-scale}
With a larger ensemble, Llama2-13B achieves 59% accuracy on GSM8K, above the 54% of Llama2-70B. {#c-mag-small}
Gains increase then decrease as the inherent difficulty rises, and gains increase with the number of reasoning steps. {#c-mag-difficulty}

## Limitations and failure modes

One limitation of self-consistency is that it incurs more computation cost; in practice people can try a small number of paths, such as 5 or 10, to realize most of the gains, as in most cases the performance saturates quickly. {#c-wang-cost}
Language models can sometimes generate incorrect or nonsensical reasoning paths, and further work is needed to better ground models' rationale generations. {#c-wang-nonsense}
A vote among wrong chains is still wrong: when the model's errors are systematic, as with a misread question or a fact it holds wrongly, self-consistency makes the error more confident rather than correcting it, and agreement is not evidence against hallucination of that kind.

Voting amplifies whatever answer the model prefers.
With two possible answers, the correct one sampled with probability p, and an odd number m of samples, the vote is correct with probability Σ_{j>m/2} C(m, j) p^j (1 − p)^(m−j); for p = 0.6 this is 0.6, 0.683, 0.753 and 0.903 at m = 1, 5, 11 and 41, and for p = 0.4 it falls to 0.4, 0.317, 0.247 and 0.097. {#c-sc-binary}
If the correct answer y is sampled with probability exceeding that of every other answer by at least δ, the vote is wrong with probability at most (|A| − 1) exp(−mδ²/2) over an answer set A, so the error falls exponentially in m: for δ = 0.3 and one competing answer it is below 0.17 at m = 40 and below 0.012 at m = 100. {#c-sc-hoeffding}
On a mixed benchmark, the two effects can cancel and reverse.
Across multiple language tasks, the performance of Vote and Filter-Vote can first increase but then decrease as a function of the number of LM calls. {#c-mlc-nonmono}
More LM calls lead to higher performance on easy queries but lower performance on hard queries, and non-monotone behavior can emerge when a task contains both types of queries. {#c-mlc-why}
For Vote, the query difficulty indicator is d_V(x) = max_{a≠y} Pr[G(x, θ) = a] − Pr[G(x, θ) = y], and d_V(x) > 0 indicates that an incorrect answer is more likely to be generated than the correct one. {#c-mlc-difficulty}
With 60% of queries answered correctly with probability 0.6 and 40% with probability 0.45, two answers per query and odd m, voting accuracy rises from 0.54 at m = 1 to about 0.653 at m = 71 and then falls towards 0.6 as m grows. {#c-sc-mixture}
This is unlike the steady improvement that neural scaling laws describe for training compute: past the peak, more samples cost more and score less.

Voting also discards most of what extra samples could offer.
Across tasks and models, coverage, the fraction of problems solved by any generated sample, scales with the number of samples over four orders of magnitude. {#c-mnk-coverage}
In domains without automatic verifiers, majority voting and reward models plateau beyond several hundred samples and fail to fully scale with the sample budget. {#c-mnk-plateau}
On MATH with Llama-3-8B-Instruct, coverage increases from 82.9% with 100 samples to 98.44% with 10,000 samples, but with majority voting or reward models selecting the answer the increase is only from 40.50% to 41.41%. {#c-mnk-math}
If one sample is correct with probability p, the chance that at least one of k samples is correct is 1 − (1 − p)^k, which tends to 1 for any p > 0, while the vote tends to the mode; for a correct answer sampled with probability 0.2 and a wrong one with 0.3, coverage at k = 10 is 0.893 while the vote converges on the wrong answer. {#c-sc-coverage}
From n samples with c correct, 1 − C(n − c, k)/C(n, k) is an unbiased estimate of 1 − (1 − p)^k for k ≤ n. {#c-sc-passk}
Selecting with a verifier closes some of the gap but has its own limit.
With a 6B verifier, performance improves as the number of completions per test problem increases up to 400, and beyond this point performance starts to decrease, as the benefits of search are outweighed by the risk of finding adversarial solutions that fool the verifier. {#c-cob-400}

## Variants

**Weighted self-consistency** weights each path by its length-normalized probability; it performs about as well as the plain vote.

**Verifier-weighted voting** (DIVERSE, Cobbe et al.) weights or filters paths with a trained verifier or reward model before the vote.

**Universal Self-Consistency** asks the model to pick the most consistent of several free-form responses, for summaries, open-ended answers and code.

**Execution-based voting** clusters programs by their outputs on test inputs and picks one from the largest cluster.

**Adaptive-Consistency and ESC** stop sampling when the samples so far agree enough, by a Beta posterior or by a unanimous window.

**Prompt ensembles** vary the exemplars or their order across samples, as DIVERSE and Wang et al.'s permutation study do, so that diversity comes from the prompt as well as from sampling.

**Agent Forest** (More Agents) runs the same query through many instantiated agents, alone or on top of other prompting methods, and votes over their answers.

## When to use it

Use self-consistency when the task has a short answer that can be compared exactly, the model already gets it right more often than it settles on any one wrong answer, and a few times the inference cost is acceptable.
Add a stopping rule when cost matters, since most questions settle within a few samples.
Prefer a verifier, tests or a reward model when they exist, since they can find correct answers that the majority misses; prefer USC or execution-based clustering when answers are free-form or programs.
Do not expect it to help on questions the model gets systematically wrong, and check accuracy against m on held-out data, since more samples can lower it.

## Pitfalls

* Sampling at a temperature so low that the paths are nearly identical, which spends the budget on copies of the greedy chain.
* Voting over raw strings, so that "18", "18.0" and "$18" count as different answers.
* Weighting by unnormalized sequence probability, which favours the shortest chains.
* Reporting the best m chosen on the test set, since the accuracy curve can peak and fall.
* Treating high agreement as proof of correctness on questions where the model's errors are systematic.
* Comparing a voted model with a single-sample baseline at different inference budgets without saying so.

## History

Cobbe et al. (2021) trained a verifier for GSM8K solutions and found that a majority vote among its top-ranked samples beat taking the single top-ranked one.
Wang et al. (2022) dropped the verifier, voted over plain chain-of-thought samples, and named the result self-consistency; it gave large gains over greedy chain-of-thought prompting on arithmetic and commonsense benchmarks and became a standard baseline, often written maj@k.
Li et al. (DIVERSE, 2022) combined prompt diversity with a step-aware verifier.
In 2023 Aggarwal et al. and, in early 2024, Li et al. (ESC) cut the cost with stopping rules, and Chen et al. (2023) extended the vote to free-form answers by letting the model judge consistency.
Through 2024, studies of inference-time scaling found the limits: Li et al. (More Agents) saw gains grow with the number of samples, Chen et al. (2024) found accuracy that first rises and then falls, and Brown et al. found that coverage keeps growing for thousands of samples while the vote stops improving after a few hundred, which shifted attention to verifiers and learned reward models for selecting among samples.
