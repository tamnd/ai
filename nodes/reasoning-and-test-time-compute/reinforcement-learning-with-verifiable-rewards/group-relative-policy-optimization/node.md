---
tag: NEW-group-relative-policy-optimization
kind: method
slug: group-relative-policy-optimization
name: Group relative policy optimization
aliases: [GRPO, group relative policy optimisation, reinforcement learning with verifiable rewards, RLVR, RL from verifiable rewards, R1-Zero training, DAPO, Dr. GRPO, GRPO Done Right, group baseline]
area: reasoning-and-test-time-compute/reinforcement-learning-with-verifiable-rewards
level: core
summary: >-
  A PPO variant without a value model: sample a group of answers per prompt, score them, and use each reward minus the group mean, over the group's standard deviation, as the advantage.
purpose: reasoning
uses: [ppo@al, reinforce@al, policy-gradient-theorem@al, kullback-leibler-divergence@m2, reward-model, chain-of-thought-prompting]
related_to: [reinforcement-learning-from-human-feedback, process-reward-model, direct-preference-optimization, reward-hacking, chain-of-thought-prompting, self-consistency]
hyperparameters:
  - {name: group size G, meaning: "outputs sampled per question; their rewards form the baseline", typical: "8 to 64 (64 in DeepSeekMath, 16 in DeepSeek-R1-Zero)"}
  - {name: clip range, meaning: "how far the probability ratio of a token may move from 1 before its gradient is cut", typical: "ε = 0.2; DAPO uses 0.2 below and 0.28 above"}
  - {name: KL coefficient β, meaning: "weight of the KL penalty to the reference policy", typical: "0.04 (DeepSeekMath), 0.001 (DeepSeek-R1-Zero), 0 (DAPO)"}
  - {name: inner updates μ, meaning: "gradient steps on each batch of samples before sampling again", typical: "1 to 4"}
  - {name: loss reduction, meaning: "how per-token losses are averaged", typical: "per sequence then per group (GRPO), over all tokens (DAPO), or by a constant (Dr. GRPO)"}
  - {name: advantage normalization, meaning: "whether centred rewards are divided by the group standard deviation", typical: "yes in GRPO, no in Dr. GRPO"}
  - {name: reward, meaning: "the score of an output", typical: "a verifier of the final answer or unit tests, plus a format reward; or a learned reward model"}
cost:
  train: "per step, G samples per question from the current policy, one forward pass of the reference model for the KL term, and μ backward passes of the policy; no value model, which in PPO is about the size of the policy"
  inference: "none beyond the trained model; models trained this way tend to write longer chains of thought, which costs tokens at inference"
history:
  year: 2024
  people: [Zhihong Shao, Peiyi Wang, Qihao Zhu, Daya Guo, Nathan Lambert, Qiying Yu, Zichen Liu]
  note: "Introduced in DeepSeekMath (2024) as a cheaper PPO for mathematical reasoning; Tülu 3 (2024) named reinforcement learning with verifiable rewards; DeepSeek-R1 (2025) trained R1-Zero from a base model with GRPO and rule-based rewards; DAPO and Dr. GRPO (2025) corrected its loss and normalization."
prereqs: [reinforcement-learning-from-human-feedback, ppo@al, chain-of-thought-prompting]
same_as:
  wikidata: null
  wikipedia: null
  arxiv: ["2402.03300", "2501.12948", "2411.15124", "2503.14476", "2503.20783", "2504.13837", "2402.14740", "1707.06347"]
sources:
  - {paper: "Shao, Wang, Zhu, Xu and others (2024)", title: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", locator: "arXiv 2402.03300v3, pp. 2, 13 to 15 and 19 to 21", note: "introduces GRPO, outcome and process supervision with it, the setup and results"}
  - {paper: "DeepSeek-AI (2025)", title: "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning", locator: "arXiv 2501.12948v2, pp. 2 to 5, 8, 15 and 16", note: "R1-Zero trained with GRPO and rule-based rewards; the comparison with PPO"}
  - {paper: "Lambert, Morrison, Pyatkin, Huang and others (2024)", title: "Tulu 3: Pushing Frontiers in Open Language Model Post-Training", locator: "arXiv 2411.15124v5, pp. 6 and 10", note: "names reinforcement learning with verifiable rewards"}
  - {paper: "Schulman, Wolski, Dhariwal, Radford and Klimov (2017)", title: "Proximal Policy Optimization Algorithms", locator: "arXiv 1707.06347v2, p. 3", note: "the clipped surrogate objective"}
  - {paper: "Ahmadian, Cremer, Gallé, Fadaee and others (2024)", title: "Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs", locator: "arXiv 2402.14740v2, pp. 4 and 6", note: "REINFORCE leave-one-out against PPO"}
  - {paper: "Yu, Zhang, Zhu, Yuan and others (2025)", title: "DAPO: An Open-Source LLM Reinforcement Learning System at Scale", locator: "arXiv 2503.14476v2, pp. 2 to 5 and 8", note: "clip-higher, dynamic sampling, token-level loss, overlong shaping"}
  - {paper: "Liu, Chen, Li, Qi and others (2025)", title: "Understanding R1-Zero-Like Training: A Critical Perspective", locator: "COLM 2025 (arXiv 2503.20783v2), pp. 1 and 6", note: "the length and difficulty biases of GRPO, and Dr. GRPO"}
  - {paper: "Yue, Chen, Lu, Zhao and others (2025)", title: "Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?", locator: "arXiv 2504.13837v5, p. 1", note: "pass@k of RLVR-trained models against their base models"}
---

Group relative policy optimization (GRPO) is a reinforcement learning algorithm for language models that drops the value model of PPO.
For each prompt it samples a group of outputs, scores each one, and uses how much better than the group average an output scored, divided by the group's standard deviation, as the advantage of every token in it.
Paired with rewards that a program can check, such as whether the final answer matches or the code passes its tests, it became the standard way to train reasoning models by reinforcement learning, starting with DeepSeek-R1.
This page covers the objective, why the group baseline works, the KL term, the settings used in practice, the biases later work found and fixed, and what this kind of training does and does not add to a model.

## In plain words

Reinforcement learning makes the outputs that scored well more likely and the ones that scored badly less likely.
"Well" has to be relative to something: if every answer to an easy question is right, nothing should change much.
PPO, the usual algorithm in reinforcement learning from human feedback, learns a second network, the value model, to predict how well the policy will do from each point, and judges each token against that prediction.

GRPO replaces the value model with a comparison inside a group.
Ask the model the same question 16 times; check the 16 answers; the ones that were right get pushed up and the ones that were wrong get pushed down, in proportion to how unusual they were in the group.
If 1 answer in 16 is right, that one gets a large push; if all 16 are right or all wrong, the question teaches nothing this step.

The reward can come from a learned reward model, but the most influential use pairs GRPO with rewards from a checker: a math answer compared with the reference, a program run against tests, a regular expression for the required format.
This combination is called reinforcement learning with verifiable rewards (RLVR).
Self-consistency also samples a group of answers to one question, but it votes among them at inference, while GRPO trains on how their rewards differ.

## Setting

Tülu 3 applies a standard reinforcement learning paradigm to skills that can be evaluated against a ground-truth outcome, such as math, and calls the method Reinforcement Learning with Verifiable Rewards (RLVR); it obtains a constant reward value if a completion is successful. {#c-tu-rlvr}
RLVR is an RL-based method that only gets a reward if the model's completions are verified to be correct. {#c-tu-only}
For DeepSeek-R1-Zero, rule-based rewards deliver precise feedback for data in mathematical, coding, and logical reasoning domains, and the rule-based reward system mainly consists of two types of rewards: accuracy rewards and format rewards. {#c-r1-rule}
For math problems with deterministic results, the model is required to provide the final answer in a specified format (e.g., within a box), enabling reliable rule-based verification of correctness. {#c-r1-acc}
For code, a compiler evaluates the responses against a suite of predefined test cases, generating objective feedback on correctness. {#c-r1-code}
Format rewards incentivize the model to encapsulate its reasoning process within designated tags, specifically <think> and </think>, and the accuracy reward and format reward are combined with the same weight. {#c-r1-format}
A verifiable reward is binary or nearly so and arrives only at the end of the output, which is the setting the group baseline suits.

### What PPO does

PPO maximizes a clipped surrogate objective, the minimum of the probability ratio times the advantage and the clipped probability ratio times the advantage, which removes the incentive for moving the ratio outside the interval and makes the objective a lower bound on the unclipped objective. {#c-ppo-clip}
In PPO for language models, a value function needs to be trained alongside the policy model, and to mitigate over-optimization of the reward model, a per-token KL penalty from a reference model, usually the initial SFT model, is added in the reward at each token. {#c-dsm-ppo}
The value function in PPO is typically another model of comparable size as the policy model, a substantial memory and computational burden, and because usually only the last token is assigned a reward score, training a value function that is accurate at each token is complicated. {#c-dsm-value}
Predicting the final reward from a partial response is even harder for long chain-of-thought models, which engage in reflection and revision, so content generated early may later be revised or contradicted. {#c-r1-revise}

## The method

### The group baseline

Group Relative Policy Optimization (GRPO) obviates the need for additional value function approximation as in PPO, and instead uses the average reward of multiple sampled outputs, produced in response to the same question, as the baseline. {#c-dsm-grpo}
For each question q, GRPO samples a group of outputs o_1, ..., o_G from the old policy and maximizes, averaged over the G outputs and over the tokens of each output, the PPO clipped term with the advantage Â_{i,t}, minus β times the KL divergence between the policy and the reference policy, where ε and β are hyper-parameters. {#c-dsm-obj}
With outcome supervision, the G rewards are normalized by subtracting the group average and dividing by the group standard deviation, and the advantages of all tokens in output o_i are set to the normalized reward (r_i − mean(r))/std(r). {#c-dsm-os}
With process supervision, which provides a reward at the end of each reasoning step, a process reward model scores each step of the outputs, the step rewards are normalized in the same way, and the advantage of a token is the sum of the normalized rewards of the steps from it onwards. {#c-dsm-ps}
The group relative way of calculating advantages aligns with the comparative nature of reward models, which are typically trained on datasets of comparisons between outputs on the same question. {#c-dsm-comp}

Why a group mean is a valid baseline comes from the policy gradient.
Subtracting a baseline b that does not depend on the sampled output leaves the policy gradient unbiased, because E_{o∼π_θ}[b ∇ log π_θ(o)] = b ∇ Σ_o π_θ(o) = 0. {#c-gr-unbiased}
Since r_i − mean(r) = (G − 1)/G · (r_i − mean of the other G − 1 rewards), the centred group advantage is the leave-one-out advantage scaled by (G − 1)/G, so before division by the standard deviation the GRPO gradient is an unbiased estimate up to that constant. {#c-gr-loo}
REINFORCE Leave-One-Out (RLOO) uses the rewards of the other k − 1 samples for the same prompt as the baseline of each sample, an unbiased estimate of the expected return for the prompt, akin to a parameter-free value function estimated at each training step. {#c-rl-loo}

With binary rewards the advantages have a simple form.
Group advantages sum to zero; if every output of a group gets the same reward all its advantages are 0, and with binary rewards and a fraction p of correct outputs the normalized advantages are √((1 − p)/p) for correct and −√(p/(1 − p)) for incorrect outputs. {#c-gr-zero}
If each output is correct with probability q independently, a group of G gives a nonzero advantage with probability 1 − q^G − (1 − q)^G: 0.815 for q = 0.9 and G = 16, but 0.337 for q = 0.95 and G = 8. {#c-gr-informative}

### The KL term

Instead of adding a KL penalty in the reward, GRPO regularizes by directly adding the KL divergence between the trained policy and the reference policy to the loss, estimated per token with an unbiased estimator that is guaranteed to be positive. {#c-dsm-kl}
With x = π_ref(o_t)/π_θ(o_t), the estimate x − log x − 1 is at least 0 for every token, and its expectation under π_θ equals KL(π_θ ‖ π_ref), because E_{π_θ}[x] = 1. {#c-gr-k3}

### Settings in practice

The RL training data of DeepSeekMath are chain-of-thought-format questions related to GSM8K and MATH from the SFT data, around 144K questions. {#c-dsm-data}
For GRPO in DeepSeekMath the policy learning rate is 1e-6, the KL coefficient is 0.04, 64 outputs are sampled for each question, the max length is 1024, the training batch size is 1024, and the policy has a single update following each exploration stage. {#c-dsm-setup}
In iterative GRPO, new training sets for the reward model are generated from the policy's samples, the reward model is continually trained with a replay mechanism that incorporates 10% of historical data, and the reference model is set to the policy model. {#c-dsm-iter}
To train DeepSeek-R1-Zero, the learning rate is 3e-6, the KL coefficient 0.001 and the sampling temperature 1; for each question 16 outputs are sampled with a maximum length of 32,768 tokens before the 8.2k step and 65,536 tokens afterward. {#c-r1-setup}
Each training step of DeepSeek-R1-Zero consists of 32 unique questions, resulting in a training batch size of 512, and every 400 steps the reference model is replaced with the latest policy model. {#c-r1-batch}
Note that the clip ratio plays a crucial role in training: a lower value can lead to the truncation of gradients for a significant number of tokens, degrading performance, while a higher value may cause instability during training. {#c-r1-clip}

## Derivation

The full derivation is in `derivations/group-relative-advantage.md`, and `group-relative-advantage.check.py` checks every step numerically.

**Baselines.** For any b fixed before the output is drawn, E[b ∇ log π(o)] = b ∇ Σ_o π(o) = 0, so subtracting b leaves the policy gradient unchanged in expectation.

**Group mean and leave-one-out.** r_i − mean(r) = (G − 1)/G · (r_i − m_{−i}), where m_{−i} is the mean of the other rewards, which is independent of o_i; so the centred advantage is an unbiased leave-one-out estimate scaled by (G − 1)/G.

**Binary rewards.** With a fraction p correct, the population standard deviation is √(p(1 − p)), giving advantages √((1 − p)/p) and −√(p/(1 − p)); a group carries signal with probability 1 − q^G − (1 − q)^G.

**Difficulty weighting.** The absolute advantages of a group sum to 2Gp(1 − p) before normalization and 2G√(p(1 − p)) after, a reweighting by 1/√(p(1 − p)).

**Length weighting.** Averaging per sequence gives token i weight A_i/(G|o_i|), so long outputs get a smaller push per token; averaging over all tokens or by a constant does not.

**KL estimate.** x − log x − 1 ≥ 0 with equality only at x = 1, and E_π[x] = 1 when x = π_ref/π, so its mean is KL(π ‖ π_ref).

**pass@k.** With pass@k = 1 − (1 − p)^k, moving probability onto problems already solved can raise pass@1 and lower pass@k.

## Worked example

A policy answers a question 8 times and 2 answers are right, so p = 1/4.
The mean reward is 0.25 and the standard deviation √(0.25 × 0.75) ≈ 0.433, so each right answer gets advantage 0.75/0.433 ≈ 1.73 and each wrong one −0.25/0.433 ≈ −0.58; the eight advantages sum to 0.
Every token of a right answer is pushed up with weight 1.73, every token of a wrong one down with weight 0.58, each subject to the clip.
With ε = 0.2, once a token of a right answer has become 1.2 times as likely as under the policy that sampled it, it stops getting gradient from this batch.

Later in training the policy gets this question right 95% of the time.
With G = 8, all 8 answers are right with probability 0.95^8 ≈ 0.66, so two times in three the group has zero advantage and wastes its samples; this is the case DAPO's dynamic sampling filters out.

The NumPy tests check these numbers, check the analytic gradient of the objective against finite differences, confirm that at the first inner step the gradient is REINFORCE with the group baseline, and train a tabular policy from 1/6 to over 90% accuracy on a checkable task.
The PyTorch tests train a small recurrent policy that writes an answer and an end token, with reward 1 only when the token before the end is right, under both the per-sequence and the per-token loss.

## Hyperparameters

The group size trades samples against the quality of the baseline: larger groups give a better baseline and more informative groups on hard questions, at G generations per question.
The clip range bounds how far one batch can move each token; DeepSeek-R1 found its value crucial, and DAPO raised only the upper bound so that unlikely tokens can grow.
The KL coefficient keeps the policy near the reference: useful when the reward is a learned model that can be exploited, often small or zero for long reasoning training with a checker, and in R1's training the reference itself is replaced by the current policy every 400 steps.
The loss reduction matters once outputs differ greatly in length, and the per-token or constant forms avoid rewarding long wrong answers.
The learning rate is small, 1e-6 for the 7B DeepSeekMath policy and 3e-6 for DeepSeek-R1-Zero.

## Cost

GRPO removes PPO's value model, a network about the size of the policy, with its memory, optimizer state and training.
What remains is generation, which dominates: G outputs per question, each possibly tens of thousands of tokens for a long chain of thought.
The reference model adds one forward pass per sample if β > 0.
Verifiable rewards are cheap to compute, so the only learned model in the loop is the policy itself.

## Evidence

Using only a subset of English instruction tuning data, GRPO obtains a substantial improvement over DeepSeekMath-Instruct, both in-domain (GSM8K: 82.9% → 88.2%, MATH: 46.8% → 51.7%) and out-of-domain (e.g., CMATH: 84.6% → 88.8%). {#c-dsm-gain}
In DeepSeekMath's comparison, Online RFT significantly outperforms RFT, and GRPO, which adjusts its gradient coefficient by the reward value and penalizes incorrect responses, surpasses online RFT. {#c-dsm-online}
GRPO+PS shows superior performance compared to GRPO+OS, indicating the benefits of using fine-grained, step-aware gradient coefficients. {#c-dsm-psbetter}

During the RL training of DeepSeek-R1-Zero, the average score on AIME 2024 increases from an initial 15.6% to 77.9%. {#c-r1-aime}
DeepSeek-R1-Zero naturally learns to solve reasoning tasks with more thinking time, its average response length on the training set growing during the RL process. {#c-r1-length}
An intermediate version of DeepSeek-R1-Zero shows an aha moment in which the model learns to rethink using an anthropomorphic tone. {#c-r1-aha}

On the MATH task, PPO with GAE λ = 0.95, the default value in most open-source PPO implementations, performs considerably worse than GRPO, while with careful tuning (λ = 1.0) PPO's performance improves substantially, nearing that of GRPO. {#c-r1-ppo}
On RLHF of language models, Ahmadian et al. report that REINFORCE-style optimization outperforms PPO, and that RLOO consistently outperforms PPO, DPO and RAFT across datasets and models. {#c-rl-ppo}

## Limitations and failure modes

DeepSeek-R1-Zero faces challenges such as poor readability and language mixing, occasionally combining English and Chinese within a single chain-of-thought response. {#c-r1-mixing}

The loss of the original GRPO has two biases.
GRPO introduces a response-level length bias, from dividing by |o_i|: correct responses get larger gradient updates when shorter, and incorrect responses are penalized less when longer, so the policy prefers lengthier responses among incorrect ones. {#c-dr-bias}
Under the sequence-mean loss each token of output i has weight A_i/(G|o_i|), so for a fixed negative advantage a 1,000-token wrong answer is pushed down per token one tenth as hard as a 100-token one; the token-mean loss gives every token weight A_i/Σ_j|o_j| and the constant normalizer A_i/(G·L_max), neither of which depends on the output's own length. {#c-gr-length}
GRPO also introduces a question-level difficulty bias by dividing by the standard deviation of the group rewards: questions that are too easy or too hard, with rewards almost all 1 or 0, are given higher weights during policy updates. {#c-dr-diff}
With binary rewards and a fraction p correct, the total absolute centred advantage of a group is 2Gp(1 − p), and dividing by the standard deviation √(p(1 − p)) makes it 2G√(p(1 − p)); relative to the unnormalized form, a question with p = 1/16 is weighted 1/√(p(1 − p)) ≈ 4.13 against 2 at p = 1/2. {#c-gr-weight}
Dr. GRPO removes the length and standard deviation normalization terms, which prevents the model from generating progressively longer incorrect responses and improves token efficiency while maintaining reasoning performance. {#c-dr-dr}

Reproductions of R1-Zero found the plain algorithm fragile.
With naive GRPO, a run from the Qwen2.5-32B base model reached only 30 points on AIME, significantly below DeepSeek's RL at 47 points, and analysis found entropy collapse, reward noise and training instability. {#c-da-naive}
DAPO introduces four techniques: Clip-Higher, which promotes diversity and avoids entropy collapse; Dynamic Sampling, which improves training efficiency and stability; Token-Level Policy Gradient Loss, critical in long-CoT RL; and Overlong Reward Shaping, which reduces reward noise. {#c-da-four}
With ε = 0.2, tokens with old probabilities 0.01 and 0.9 can rise at most to 0.012 and 1.08, so the upper clipping threshold restricts the increase of low-probability exploration tokens; DAPO sets ε_low to 0.2 and ε_high to 0.28. {#c-da-clip}
If all outputs of a prompt are correct and receive the same reward, the advantage for the group is zero and so are its policy gradients; the number of such samples keeps increasing during training, so DAPO over-samples and filters out these prompts. {#c-da-dyn}
DAPO removes the KL penalty, because when training a long-CoT reasoning model the distribution can diverge significantly from the initial model, so the restriction is not necessary. {#c-da-kl}
DAPO achieves 50 points on AIME 2024 with the Qwen2.5-32B base model, outperforming DeepSeek-R1-Zero-Qwen-32B (47 points) using 50% of the training steps. {#c-da-50}

What RLVR adds to a model is debated.
DeepSeekMath's analysis suggests that RL enhances overall performance by rendering the output distribution more robust, boosting the correct response from TopK rather than enhancing fundamental capabilities. {#c-dsm-why}
Across model families, RL algorithms and math, coding and visual reasoning benchmarks, RLVR-trained models outperform their base models at small k (e.g., k=1), but base models achieve higher pass@k when k is large. {#c-yu-k}
The reasoning paths of RLVR models are already in the base models' sampling distribution, six popular RLVR algorithms perform similarly, and distillation, unlike RLVR, can introduce new reasoning patterns from the teacher. {#c-yu-dist}
Sharpening can raise pass@1 and lower pass@k: two problems solved with probabilities 0.2 and 0.05 have mean pass@1 0.125 and pass@64 ≈ 0.981, while 0.6 and 0 give pass@1 0.3 and pass@64 ≈ 0.5. {#c-gr-passk}
DeepSeek-V3-Base already exhibits an Aha moment, and Qwen2.5 base models show strong reasoning capabilities even without prompt templates, suggesting potential pretraining biases. {#c-dr-aha}

A learned reward model inside the loop brings back reward hacking, and DeepSeek-R1 used rule-based rewards for reasoning for that reason; checkers have their own gaps, such as tests that accept wrong programs and answers that are right by luck.

## Variants

**GRPO with outcome supervision.** One advantage per output from the group-normalized reward, shared by all its tokens; the common form.

**GRPO with process supervision.** A process reward model scores each step, and a token's advantage is the sum of the normalized step rewards from it onwards.

**Iterative GRPO.** The reward model is retrained on the policy's samples, and the reference is reset to the policy, between rounds.

**RLOO.** REINFORCE with the leave-one-out mean as baseline, without clipping or normalization by the standard deviation.

**DAPO.** Decoupled clip ranges, filtering of groups with all-equal rewards, a per-token loss, shaping or masking of truncated outputs, and no KL term.

**Dr. GRPO.** The same objective without division by the output length or by the group's standard deviation.

**PPO with verifiable rewards.** Tülu 3 trained its RLVR stage with PPO and a value model initialized from a reward model.

## When to use it

Use GRPO when rewards come at the end of an output, can be compared within a group, and the cost of a value model matters; with a reliable checker it is the standard choice for training reasoning in mathematics and code.
Prefer a per-token or constant loss reduction and filtering of all-equal groups when outputs are long.
Prefer preference methods such as direct preference optimization, or reinforcement learning from human feedback with a learned reward model, when quality cannot be checked by a program.
Expect gains in pass@1 and in the reliability of sampling, while the problems the model can solve in many tries may not grow.

## Pitfalls

* Training on questions the policy always or never solves: their groups have zero advantage and only cost generation.
* Reading longer responses as better reasoning: under per-sequence averaging, length can grow on wrong answers.
* A KL term that is too strong for long training, or a reference that is never updated.
* Giving truncated outputs reward 0 without masking them, which punishes sound but long reasoning.
* A format reward that can be satisfied without solving the problem, or a checker that accepts near-misses.
* Evaluating only pass@1 when the claim is about new reasoning ability.

## History

Williams's REINFORCE (1992) introduced the baseline, and Kool et al. (2019) used other samples of the same input as a leave-one-out baseline.
PPO (Schulman et al., 2017) became the algorithm of reinforcement learning from human feedback, with a learned value model and reward model.
Shao et al. (2024) introduced GRPO in DeepSeekMath to cut PPO's memory and used it with learned outcome and process reward models on chain-of-thought math data.
Ahmadian et al. (2024) argued that REINFORCE with a leave-one-out baseline outperforms PPO for language models, and Tülu 3 (2024) named reinforcement learning with verifiable rewards.
DeepSeek-R1 (2025) trained R1-Zero with GRPO and rule-based rewards directly from a base model, and its long, self-correcting chains of thought made the recipe widely copied.
In 2025 DAPO and Dr. GRPO corrected the loss and normalization, and Yue et al. questioned whether this training adds reasoning ability or concentrates what the base model already had.
