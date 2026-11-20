---
tag: NEW-test-time-scaling
kind: finding
slug: test-time-scaling
name: Test-time scaling
aliases: [test-time scaling, inference scaling, inference-time scaling, scaling test-time compute, inference scaling laws, repeated sampling, test-time compute scaling]
area: reasoning-and-test-time-compute/test-time-scaling
level: core
summary: >-
  Spending more computation when a model answers, by sampling many answers, searching or thinking longer, raises accuracy in a predictable way, until the method that picks the answer saturates.
related_to: [process-reward-model, reward-model, group-relative-policy-optimization, reward-hacking, beam-search@al]
history:
  year: 2024
  note: "Jones (2021) measured a trade between train-time and test-time compute for AlphaZero on Hex; Brown et al. (July 2024) and Snell et al. (August 2024) measured it for language models, Wu et al. (August 2024, ICLR 2025) fitted compute-optimal inference, and in 2025 DeepSeek-R1 and s1 scaled the length of a single chain of thought."
prereqs: [chain-of-thought-prompting, self-consistency, neural-scaling-laws]
same_as:
  arxiv: ["2407.21787", "2408.03314", "2408.00724", "2501.19393", "2104.03113", "2502.12215", "2412.21187"]
sources:
  - {paper: "Brown, Juravsky, Ehrlich, Clark, Le, Ré and Mirhoseini (2024)", title: "Large Language Monkeys: Scaling Inference Compute with Repeated Sampling", locator: "arXiv 2407.21787v3, pp. 1 to 11", note: "coverage against the number of samples, the exponentiated power law, the FLOPs comparison and the verification plateau"}
  - {paper: "Snell, Lee, Xu and Kumar (2024)", title: "Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters", locator: "arXiv 2408.03314v1, Sections 3 to 7", note: "difficulty bins, search against a process reward model, revisions and the exchange with pretraining compute"}
  - {paper: "Wu, Sun, Li, Welleck and Yang (2025)", title: "Inference Scaling Laws: An Empirical Analysis of Compute-Optimal Inference for LLM Problem-Solving", locator: "ICLR 2025, arXiv 2408.00724v3, pp. 1 to 6", note: "saturation, the 7B-against-34B comparison and the voting convergence theorems"}
  - {paper: "Muennighoff, Yang, Shi, Li, Fei-Fei, Hajishirzi, Zettlemoyer, Liang, Candès and Hashimoto (2025)", title: "s1: Simple test-time scaling", locator: "arXiv 2501.19393v3, pp. 1 to 9", note: "budget forcing, sequential against parallel scaling and the limits"}
  - {paper: "Jones (2021)", title: "Scaling Scaling Laws with Board Games", locator: "arXiv 2104.03113v2, pp. 1 and 5", note: "the train-time against test-time trade-off for AlphaZero on Hex"}
  - {paper: "Zeng, Cheng, Yin, Zhou and Qiu (2025)", title: "Revisiting the Test-Time Scaling of o1-like Models: Do they Truly Possess Test-Time Scaling Capabilities?", locator: "arXiv 2502.12215v2, pp. 1 and 2", note: "longer chains of thought do not consistently help"}
  - {paper: "Chen, Xu, Liang, He, Pang, Yu, Song, Liu, Zhou, Zhang, Wang, Tu, Mi and Yu (2024)", title: "Do NOT Think That Much for 2+3=? On the Overthinking of o1-Like LLMs", locator: "arXiv 2412.21187v2, p. 2", note: "overthinking on easy problems"}
claim: >-
  For a fixed model, accuracy on reasoning, mathematics and coding problems rises smoothly with the computation spent at inference, through more samples, search against a verifier or longer chains of thought; the share of problems solved by some sample grows roughly log-linearly over orders of magnitude, while what a method can actually return is capped by how well it selects among the samples.
first_reported: {work: brown-2024-large-language-monkeys, note: "coverage against up to 10,000 samples on five tasks and three model families, with an exponentiated power-law fit and a FLOPs-matched comparison of model sizes"}
replications:
  - {work: snell-2024-scaling-test-time-compute, note: "PaLM 2-S* on MATH with best-of-N, beam search against a process reward model and sequential revisions, compared by difficulty bin and against a 14 times larger model"}
  - {work: wu-2024-inference-scaling-laws, note: "Pythia models of several sizes and Llemma 7B and 34B on MATH500 and GSM8K with voting, weighted voting, tree search and REBASE"}
  - {work: muennighoff-2025-s1-simple-test-time-scaling, note: "sequential scaling of one reasoning trace by budget forcing on AIME24, MATH500 and GPQA Diamond"}
  - {work: jones-2021-scaling-scaling-laws-board-games, note: "an earlier analogue in games, AlphaZero on Hex, where tree-search size at test time trades against training compute"}
scale_range:
  - {range: "Pythia, Gemma and Llama-3 models from 70M to 70B parameters, from 1 to 10,000 samples per problem", work: brown-2024-large-language-monkeys}
  - {range: "PaLM 2-S* on MATH with generation budgets up to 256 samples, compared with a 14 times larger model", work: snell-2024-scaling-test-time-compute}
  - {range: "Pythia models of several sizes and Llemma 7B and 34B", work: wu-2024-inference-scaling-laws}
  - {range: "Qwen2.5-32B fine-tuned on 1,000 examples, with average thinking lengths of about 512 to 8,192 tokens", work: muennighoff-2025-s1-simple-test-time-scaling}
failures:
  - {work: brown-2024-large-language-monkeys, note: "majority voting and reward models plateau around 100 samples while coverage keeps rising, and all Pythia models have zero coverage on CodeContests at 10,000 samples"}
  - {work: cobbe-2021-training-verifiers, note: "best-of-n against a 6B verifier on GSM8K improves up to 400 samples and then declines"}
  - {work: snell-2024-scaling-test-time-compute, note: "on the hardest difficulty bin no method makes much progress, and beam search degrades on easy questions as the budget grows"}
  - {work: zeng-2025-revisiting-test-time-scaling-o1, note: "for QwQ, R1 and LIMO, longer chains of thought do not consistently raise accuracy, and some models degrade as revisions accumulate"}
  - {work: chen-2024-overthinking-o1-like-llms, note: "o1-like models spend many times more tokens on easy problems for little gain in accuracy"}
explanations:
  - {name: "a spread of per-problem success rates", summary: "Each extra sample helps every problem with a small but non-zero success rate, and a heavy tail of such rates turns coverage into an exponentiated power law in the number of samples.", sources: [brown-2024-large-language-monkeys, wu-2024-inference-scaling-laws]}
  - {name: "a selection bottleneck", summary: "Voting converges to the most common answer and a verifier can be fooled, so the returned accuracy saturates at a limit set by the answer distribution and the verifier, below coverage.", sources: [wu-2024-inference-scaling-laws, brown-2024-large-language-monkeys, cobbe-2021-training-verifiers]}
  - {name: "sequential refinement", summary: "A longer reasoning trace lets the model check and revise earlier steps, which helps on easier problems and when the model was trained to use the extra length.", sources: [snell-2024-scaling-test-time-compute, muennighoff-2025-s1-simple-test-time-scaling, deepseek-2025-r1]}
---

Test-time scaling is the finding that a language model answers reasoning, mathematics and coding problems more accurately when it is allowed to spend more computation at inference, and that the gain follows a smooth curve in the amount of computation.
The extra computation can take three forms: drawing many independent answers, searching over partial answers with a verifier, or letting one answer think for longer.
The finding complements the neural scaling laws for training: accuracy can be bought either with a larger, longer-trained model or with more work per question, and the two prices can be compared.
It also has a ceiling that the training laws do not, because a method that generates many answers must still pick one, and picking is often the harder part.

## In plain words

Ask a model the same hard question a hundred times at a non-zero temperature and it will give different answers.
If one of the hundred is right and there is a way to recognise it, such as a unit test or a proof checker, the model has solved the problem.
The share of problems solved this way, called coverage, keeps growing as the number of attempts grows, roughly by the same amount each time the number of attempts is multiplied by ten.

Without a checker, the answer must be chosen from the samples.
Taking the most common answer, as self-consistency does, works when the right answer is the most likely one, but stops improving once the vote has settled.
Ranking samples with a learned reward model works until the samples include wrong answers that fool it.

The third route is to let the model think longer before it answers, either by asking it to revise its answer or by training it to write long chains of thought.
This helps most on problems that are within reach of the model, and on very easy problems the extra thinking is mostly wasted.

## The finding

Across multiple tasks and models, coverage, the fraction of problems that are solved by any generated sample, scales with the number of samples over four orders of magnitude, and the relationship is often log-linear and can be modelled with an exponentiated power law. {#c-br-coverage}
Applying repeated sampling to SWE-bench Lite, the fraction of issues solved with DeepSeek-Coder-V2-Instruct increases from 15.9% with one sample to 56% with 250 samples, above the 43% of the best single-sample result reported at the time. {#c-br-swe}
The gain is not limited to strong models or easy tasks: the smallest models gain most in relative terms, starting from very low coverage.

The effectiveness of different approaches to scaling test-time compute critically varies depending on the difficulty of the prompt, and a compute-optimal strategy that allocates test-time compute adaptively per prompt improves the efficiency of test-time compute scaling by more than 4x compared to a best-of-N baseline. {#c-sn-difficulty}
In a FLOPs-matched evaluation, on problems where a smaller base model attains somewhat non-trivial success rates, test-time compute can be used to outperform a 14x larger model. {#c-sn-14x}
Increasing the compute budget leads to higher accuracy until the accuracy reaches saturation; as the budget increases, smaller models initially perform better than larger ones, but once the accuracy of the smaller models saturates, the larger models have favorable performance. {#c-wu-saturate}

Repeated sampling and search are parallel methods; a longer single reasoning trace is sequential.
Budget forcing controls test-time compute by forcefully terminating the model's thinking process or lengthening it by appending "Wait" multiple times to the model's generation when it tries to end, which can lead the model to double-check its answer. {#c-s1-bf}
Scaling s1-32B with budget forcing allows extrapolating beyond its performance without test-time intervention, from 50% to 57% on AIME24. {#c-s1-57}
Models trained with reinforcement learning against verifiable rewards, such as with group relative policy optimization, learn to spend this sequential compute themselves.
DeepSeek-R1-Zero exhibits a steady increase in thinking time throughout reinforcement learning, driven solely by intrinsic adaptation, generating hundreds to thousands of tokens to explore and improve its problem-solving strategies. {#c-r1-length}

The trade between computation at training and at inference was measured before language models, in games.
For AlphaZero on Hex, the trade-off between train-time and test-time compute is linear in log-compute: for each additional 10x of train-time compute, about 15x of test-time compute can be eliminated, down to a floor of a single-node tree search. {#c-jo-trade}
The performance of a specific snapshot is sigmoid in the test-time compute budget, and larger tree searches at test time can substantially improve the performance of an agent. {#c-jo-sigmoid}

## Evidence

**Repeated sampling.**
Brown et al. evaluate GSM8K, MATH, MiniF2F-MATH, CodeContests and SWE-bench Lite, generating 10,000 independent samples per problem for the Llama-3 models and restricting attempts to 250 per issue on SWE-bench Lite. {#c-br-tasks}
Repeated sampling leads to consistent coverage gains across model sizes (70M-70B), families (Llama, Gemma and Pythia) and levels of post-training (base and instruct models). {#c-br-models}
When solving CodeContests problems with Gemma-2B, coverage increases by over 300x, from 0.02% with one sample to 7.1% with 10,000 samples. {#c-br-gemma}
When solving MATH problems with Pythia-160M, coverage increases from 0.27% to 57%. {#c-br-pythia}
Brown et al. model the log of coverage c as a function of the number of samples k with log(c) ≈ ak^b, so c ≈ exp(ak^b); these laws are not as exact as training scaling laws, most strikingly on MiniF2F-MATH. {#c-br-fit}
Within a model family, coverage curves on a logarithmic axis have the same slope with distinct horizontal offsets, so the multiplicative increase in the sample budget needed to improve coverage from c to c′ is approximately constant. {#c-br-shift}
The derivation coverage-and-voting explains this form: if per-sample success rates across problems have a Beta distribution, the failure rate falls as a power of $k$.
If one sample solves a problem with probability p, k independent samples cover it with probability 1 − (1 − p)^k; the k-th extra sample adds p(1 − p)^k, so coverage is increasing and concave in k, and a problem with p = 0 is never covered. {#c-tt-cov}
With N samples of which C are correct, 1 − C(N − C, k)/C(N, k) is an unbiased estimate of 1 − (1 − p)^k for every k ≤ N, while the plug-in 1 − (1 − C/N)^k underestimates it on average for k ≥ 2. {#c-tt-unbiased}
If per-sample success rates across problems follow Beta(α, β), the failure rate 1 − c(k) is B(α, β + k)/B(α, β) ≈ A k^(−α) with A = Γ(α + β)/Γ(β), so c(k) ≈ exp(a k^b) with a = −A and b = −α; for α = 0.5 and β = 2, coverage is 0.20, 0.612, 0.868 and 0.9867 at k = 1, 10, 100 and 10,000. {#c-tt-beta}

**Inference compute against model size.**
With the FLOP budget fixed, Llama-3-8B-Instruct always obtains higher coverage than the 70B model on MiniF2F, GSM8K and MATH, while for CodeContests the 70B model is almost always more cost effective. {#c-br-flops}
Keeping the agent framework constant, sampling five times from the weaker and cheaper DeepSeek model solves more SWE-bench Lite issues than single samples from Claude or GPT while also being over 3x cheaper. {#c-br-cost}
On MATH500 the 7B model outperforms the 34B model until 128 Llemma 7B solutions are sampled. {#c-wu-128}
Llemma-7B can achieve accuracy competitive with Llemma-34B while using 2x less FLOPs on MATH500 or GSM8K, and Llemma-7B with REBASE outperforms Llemma-34B with standard majority voting across all compute budgets. {#c-wu-7b}
With 6ND training FLOPs and 2ND inference FLOPs, a model M times smaller matches the total FLOPs of the larger one by multiplying its inference tokens by M + 3(D_pretrain/D_inference)(M − 1), which for M = 14 is 53 when the two token counts are equal and 17.9 when inference tokens are ten times pretraining tokens. {#c-tt-flops}

**Search and revision against difficulty.**
Snell et al. use the MATH benchmark with the split of 12k train and 500 test questions of Lightman et al., and the PaLM 2-S* base model, which attains non-trivial performance on MATH without saturating it. {#c-sn-setup}
Snell et al. define question difficulty from the base model's pass@1 rate, estimated from 2048 samples and binned into five quantiles, and found these model-specific bins more predictive of the efficacy of test-time compute than the hand-labeled difficulty levels of MATH. {#c-sn-bins}
Their verifier is a process reward model, which scores each step of a solution rather than only the final answer, and beam search keeps the candidates with the highest step scores.
Beam search against a process reward model is more effective on harder questions and at lower compute budgets, whereas best-of-N is more effective on easier questions and at higher budgets; on the easy questions beam search degrades as the generation budget increases, suggesting exploitation of the PRM signal. {#c-sn-search}
There is a tradeoff between sequential (revisions) and parallel (best-of-N) test-time computation: easier questions benefit from purely sequential test-time compute, whereas harder questions often perform best with some ideal ratio of sequential to parallel compute. {#c-sn-revise}

**Longer thinking.**
s1K consists of 1,000 questions with reasoning traces distilled from Gemini Thinking, and supervised fine-tuning of Qwen2.5-32B-Instruct on it took 26 minutes on 16 NVIDIA H100 GPUs. {#c-s1-data}
Scaling test-time compute on the base model Qwen2.5-32B-Instruct via majority voting cannot catch up with the performance of s1-32B, which the authors take to validate that sequential scaling is more effective than parallel. {#c-s1-par}

**Voting and verifiers before the term.**
In self-consistency, sampling a higher number of reasoning paths, up to 40, leads to consistently better performance. {#c-sc-40}
For a 6B verifier on GSM8K, performance improves as the number of completions per test problem increases up to 400, and beyond this point it starts to decrease, suggesting that the benefits of search are outweighed by the risk of finding adversarial solutions that fool the verifier. {#c-co-400}

**Train-time against test-time compute in games.**
A fit with slope −1.2 between the logarithms of test-time and train-time compute at fixed Elo means that 10x more train-time compute divides test-time compute by 10^1.2 ≈ 15.8, whatever the base of the logarithm. {#c-tt-jones}

## Scope and conditions

**Selection saturates below coverage.**
With majority voting, reward-model best-of-N and reward-weighted voting, success rates on GSM8K and MATH increase at first and plateau around 100 samples, while coverage keeps increasing and eventually exceeds 95%. {#c-br-plateau}
On MATH with Llama-3-8B-Instruct, coverage increases from 82.9% with 100 samples to 98.44% with 10,000 samples, while the biggest increase for majority voting or reward models is only from 40.50% to 41.41% over the same range. {#c-br-math}
For some problems correct solutions are sampled at low probabilities, e.g. 1% or lower, and majority voting saturates because rare correct solutions do not affect the most common answer. {#c-br-needle}
Theorems 1 and 2 of Wu et al. state the convergence of the accuracy of majority voting and of weighted majority voting with increasing number of samples, almost surely, to the fraction of problems whose true answer is the arg max of the answer distribution of the language model (weighted by the reward model for weighted voting), with an expected accuracy below this limit by O(c^(−n)) for some constant c > 1. {#c-wu-theorem}
If the correct answer has probability q* and a competitor q_y < q*, the chance that the competitor gets at least as many of n votes is at most exp(−n(q* − q_y)^2/2); with q* = 0.4 against 0.3 the bound is about 0.61 at n = 100 and 0.0067 at n = 1,000. {#c-tt-vote}
As n grows, majority-vote accuracy over a benchmark tends to the fraction of problems whose correct answer is the unique mode of the answer distribution, while coverage tends to the fraction with q* > 0; the problems with 0 < q* < max q_y make up the gap. {#c-tt-gap}
As long as the reward model assigns higher rewards to correct solutions on average, the accuracy limit of weighted majority voting is higher than that of majority voting, and weighted majority voting consistently dominates in experiments. {#c-wu-weighted}

**Verifiers can be fooled.**
A learned reward model is not a checker, and a larger sample contains more wrong answers that it scores highly, a form of reward hacking at inference.
If a fraction ε of samples are wrong answers a verifier ranks above every correct one and a fraction p are correct, best-of-n is right with probability (1 − ε)^n − (1 − ε − p)^n, which peaks at n* = log(log b/log a)/log(a/b) with a = 1 − ε and b = 1 − ε − p; for p = 0.3 and ε = 0.001, n* ≈ 16.5, with accuracy 0.981 at n = 16 and 0.368 at n = 1,000. {#c-tt-bon}
Exact checkers are not perfect either.
11.3% of SWE-bench Lite problems have flaky test suites that do not produce consistent results on the same candidate solution, and 35 of 122 CodeContests test problems have correct solutions that fail the tests. {#c-br-flaky}

**Problems out of reach.**
All Pythia models achieve zero coverage on CodeContests, even with a budget of 10,000 samples. {#c-br-zero}
If a fraction π_0 of problems has per-sample success rate 0, coverage is at most 1 − π_0 for every number of samples. {#c-tt-cap}

**Pretraining and inference are not interchangeable.**
Test-time and pretraining compute are not 1-to-1 exchangeable: on easy and medium questions, or with a small inference requirement, test-time compute can cover for additional pretraining, but on challenging questions or under a higher inference requirement pretraining is likely more effective. {#c-sn-exch}
MCTS consumes substantially more resources than sampling, often requiring dozens of times more generated tokens, because many paths in the search tree are used to estimate and select nodes and do not become part of the final solution. {#c-wu-mcts}

**Longer is not always better.**
Budget forcing improves AIME24 performance but eventually flattens out at six times, and suppressing the end-of-thinking token delimiter too often can lead the model into repetitive loops. {#c-s1-flat}
Budget forcing has two key limitations when scaling further: it eventually flattens out, and the context window of the underlying model constrains it; with step-conditional control at 512 steps, 12 of the 30 questions exceed the context window. {#c-s1-ctx}
Sampling until a generation fits a specific length gives an inverse scaling trend, because longer samples are often ones in which the model made mistakes and backtracks. {#c-s1-rs}
For QwQ, R1 and LIMO, longer chains of thought do not consistently enhance accuracy, and correct solutions are often shorter than incorrect ones for the same questions. {#c-ze-length}
QwQ and R1-Distill-1.5b show performance degradation as the length of reflection increases, while R1-Distill-14b, R1-Distill-32b and LIMO improve during early revisions and then oscillate. {#c-ze-revise}
Comparing sequential and parallel scaling on QwQ, R1 and LIMO, parallel scaling achieves better coverage and scalability. {#c-ze-parallel}
On the question "what is the answer of 2 plus 3?", o1-like models consumed 1,953% more tokens than conventional models on average, and the overthinking patterns contribute minimally to improving accuracy, lack diversity in reasoning strategies and occur more frequently with simple problems. {#c-ov-tokens}

## Explanations

**A spread of per-problem success rates.**
Sampling turns a small success rate into a large one: a problem the model solves once in a hundred tries is covered with probability about 0.63 after a hundred samples.
Across a benchmark the per-problem rates vary widely, and each new sample mostly helps the problems whose rate is small but not zero.
When these rates have a heavy tail near zero, the average failure rate falls as a power of the number of samples, which gives the exponentiated power law that Brown et al. fit; a mass of problems with rate exactly zero puts a ceiling on coverage that no number of samples removes.

**A selection bottleneck.**
Majority voting returns the most common answer, so as the number of samples grows it converges to whatever answer the model gives most often, right or wrong.
Its limit is the share of problems where the right answer is the mode, which can be far below coverage when correct solutions are rare.
A verifier lifts this limit when it ranks correct answers higher on average, but a verifier that can be fooled turns more samples into more chances to be fooled, so best-of-n first rises and then falls.

**Sequential refinement.**
A longer trace gives the model room to check a step, notice an error and try again, which parallel samples cannot do because they do not see each other.
Snell et al. found this most useful on easier problems, where the first attempt is close to right, and found a mix of sequential and parallel compute better on harder ones.
Models trained to reason at length, such as DeepSeek-R1, use the extra length well up to a point; models that write long traces without being able to use them, as Zeng et al. and Chen et al. report, spend tokens without gaining accuracy.

## Implications

**Allocating compute per problem.**
A fixed number of samples per question wastes compute on problems already solved and on problems out of reach.
To maximize expected coverage with a budget of K samples spread over problems, giving each next sample to the problem with the largest marginal gain p(1 − p)^k is optimal; a problem with p = 0.9 gains 0.9, 0.09 and 0.009 from its first three samples, one with p = 0.05 still gains about 0.0315 from its tenth, and one with p = 0 gets none. {#c-tt-alloc}
The compute-optimal strategy of Snell et al. applies the same idea with estimated difficulty in place of the unknown success rates.

**Smaller models with more inference.**
Smaller models combined with advanced inference algorithms offer Pareto-optimal trade-offs in cost and performance, and scaling inference compute with inference strategies can be more computationally efficient than scaling model parameters. {#c-wu-pareto}
The comparison depends on how many queries the model will serve, since pretraining is paid once and inference on every query.

**Classifying methods.**
Test-time scaling methods can be classified as sequential, where later computations depend on earlier ones as in a long reasoning trace, and parallel, where computations run independently as in majority voting. {#c-s1-seq}
Combining the two, as in beam search over steps or several long traces with a vote, is common in practice.

**Shorter traces.**
Training QwQ-32B-Preview on its own streamlined responses reduces token output by 48.6% while maintaining accuracy on MATH500. {#c-ov-48}
Shortest Majority Vote improves majority voting by prioritizing answer clusters that have both more solutions and shorter solution lengths, and it substantially outperforms conventional majority voting for QwQ and R1. {#c-ze-smv}

## Open questions

There is no agreed way to compare the scaling of methods that use compute so differently, since tokens, samples and FLOPs give different rankings and the cost of a verifier is often left out.
The selection bottleneck means that the large gains in coverage reach users only when a reliable checker exists, and it is unclear how much of the gap learned verifiers can close in domains such as open-ended writing, where no checker exists.
How sequential and parallel scaling should be mixed for a given problem, and how a model can estimate in advance how much compute a question deserves, remain open.

## History

Jones measured in April 2021 that, for AlphaZero playing Hex, tree search at test time trades against training compute at a fixed rate in log-compute.
Cobbe et al. (2021) trained verifiers for best-of-n on grade-school mathematics, and Wang et al. (2022) introduced self-consistency, both scaling the number of samples at inference before the term was in use.
In 2024, Brown et al. posted "Large Language Monkeys" in July, Wu et al. their inference scaling laws and Snell et al. their study of compute-optimal test-time compute in August, and OpenAI released o1 in September as a model that thinks longer before answering.
In January 2025, DeepSeek-R1 described a long chain of thought learned with reinforcement learning, and s1 reproduced sequential test-time scaling with 1,000 training examples and budget forcing; Chen et al. and Zeng et al. reported the limits of longer thinking in the same months.

## References
