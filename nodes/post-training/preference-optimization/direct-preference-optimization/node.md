---
tag: NEW-direct-preference-optimization
kind: method
slug: direct-preference-optimization
name: Direct preference optimization
aliases: [DPO, direct preference optimisation]
area: post-training/preference-optimization
level: core
summary: >-
  Fine-tune a language model on preference pairs with a logistic loss on policy-to-reference log-ratios, optimizing the KL-constrained RLHF objective without a reward model or RL.
purpose: aligning a language model with human preferences from pairwise comparisons, without fitting a separate reward model and without sampling from the policy during training
uses: [reward-function@al, kullback-leibler-divergence@m2, logistic-regression@m2, cross-entropy@m2]
related_to: [ppo@al, reinforce@al]
hyperparameters:
  - {name: beta, meaning: "the strength of the KL penalty toward the reference model, which also scales the implicit reward beta·log(pi/pi_ref); smaller beta lets the policy move further from the reference", typical: "0.1; 0.5 in the paper's summarization runs"}
  - {name: reference model, meaning: "the frozen model the log-ratios are taken against, ideally the model that generated the preference data: the SFT model, or one fine-tuned on the preferred responses when no SFT model exists", typical: "the SFT model, also used to initialize the policy"}
  - {name: learning rate and batch, meaning: "the optimizer settings; the paper uses RMSprop with a linear warm-up", typical: "1e-6 with 150 warm-up steps, batch size 64"}
cost:
  train: "a forward and backward pass of the policy and a forward pass of the frozen reference on both responses of every pair; no sampling, no reward model and no value network"
  inference: "that of the fine-tuned policy alone"
history:
  year: 2023
  people: [Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn]
  note: "Introduced in Direct Preference Optimization: Your Language Model is Secretly a Reward Model (NeurIPS 2023), which solved the KL-constrained reward objective of RLHF in closed form and turned it into a classification loss on the policy; Azar et al. (2023) placed it in a general family of preference objectives and proposed IPO against its overfitting on deterministic preferences, and Xu et al. (2024) compared it with carefully tuned PPO."
prereqs: [reward-function@al, kullback-leibler-divergence@m2, logistic-regression@m2, ppo@al]
same_as:
  wikidata: null
  wikipedia: null
  arxiv: ["2305.18290", "2310.12036", "2404.10719"]
  al: null
sources:
  - {paper: "Rafailov, Sharma, Mitchell, Ermon, Manning and Finn (2023)", title: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", locator: "pages 1 to 10, 15, 16, 19, 20 and 22", note: "the RLHF objective, its closed-form optimum, the reparameterized Bradley-Terry loss, the gradient, the theory of reward classes, the experiments, the implementation and hyperparameters"}
  - {paper: "Azar, Rowland, Piot, Guo, Calandriello, Valko and Munos (2023)", title: "A General Theoretical Paradigm to Understand Learning from Human Preferences", locator: "pages 1, 4, 5 and 7", note: "the two approximations of RLHF, weak regularization and overfitting under deterministic preferences, and the IPO loss"}
  - {paper: "Xu, Fu, Gao, Ye, Liu, Mei, Wang, Yu and Wu (2024)", title: "Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study", locator: "pages 1 to 4", note: "PPO solutions as a proper subset of DPO solutions, out-of-distribution responses, and benchmarks where tuned PPO beats DPO"}
---

Direct preference optimization (DPO) fine-tunes a language model on human comparisons of the form "for this prompt, response $y_w$ is better than response $y_l$".
It starts from the same goal as reinforcement learning from human feedback (RLHF): maximize a learned reward while staying close, in KL divergence, to a reference model.
The optimal policy for that goal has a closed form, so a reward can be written as a scaled log-ratio between the policy and the reference, and substituting this into the Bradley-Terry model of preferences turns reward learning into a loss on the policy itself.
The result is a binary logistic loss on the difference of two log-ratios, trained like ordinary supervised fine-tuning, with no reward model, no sampling and no reinforcement learning.
Rafailov et al. introduced it in 2023 and reported that it matched or beat PPO-based RLHF at up to 6B parameters; later work found that it overfits deterministic preferences and that carefully tuned PPO can beat it.

## In plain words

To teach a chatbot what people like, you can show it pairs of its own answers and say which one a person preferred.
The usual recipe first trains a separate "judge" model to score answers, then uses trial-and-error reinforcement learning to make the chatbot write answers the judge scores highly, while a penalty stops it drifting too far from where it started.
DPO notices that, for this exact goal, the ideal chatbot and the judge are tied together by a simple formula.
So it can skip the judge: it directly nudges the chatbot to make the preferred answer more likely and the other one less likely, each measured against a frozen copy of the starting model.
Pairs the chatbot already gets right are nudged a little; pairs it gets wrong are nudged a lot.
The whole thing is one loss function, trained like ordinary fine-tuning.

## Setting

Rafailov et al. describe the RLHF pipeline as three phases: supervised fine-tuning (SFT), preference sampling and reward learning, and RL optimization. {#c-ra-three-phases}
The SFT model is prompted with $x$ to produce a pair of answers, human labelers express a preference for one of them, the preferred completion $y_w$ over the dispreferred $y_l$, and the preferences are assumed to come from a latent reward model $r^*(x, y)$ to which there is no access. {#c-ra-pairs}
The Bradley-Terry model sets the probability of the preference to

$$ p^*(y_1 \succ y_2 \mid x) = \frac{\exp r^*(x, y_1)}{\exp r^*(x, y_1) + \exp r^*(x, y_2)} = \sigma\big(r^*(x, y_1) - r^*(x, y_2)\big) , $$ {#eq-bradley-terry}

and a reward model $r_\phi$ is fit to a static dataset of comparisons by minimizing the negative log-likelihood $-\mathbb E \log\sigma(r_\phi(x, y_w) - r_\phi(x, y_l))$, a binary classification problem. {#c-ra-reward-nll}
The reward model is typically initialized from the SFT model, adding a linear layer on the final transformer layer that gives a single scalar prediction of the reward. {#c-ra-reward-init}
The RL phase then solves

$$ \max_{\pi_\theta}\; \mathbb E_{x\sim\mathcal D,\, y\sim\pi_\theta(y\mid x)}\big[r_\phi(x, y)\big] - \beta\, \mathbb D_{\mathrm{KL}}\big[\pi_\theta(y\mid x) \,\Vert\, \pi_{\mathrm{ref}}(y\mid x)\big] , $$ {#eq-rlhf-objective}

where $\beta$ is a parameter that controls how far the policy deviates from the base reference policy $\pi_{\mathrm{ref}}$, in practice the initial SFT model. {#c-ra-objective}
They explain the constraint as important because it prevents the model deviating too far from where the reward model is accurate, maintains generation diversity, and prevents mode collapse onto single high-reward answers. {#c-ra-constraint-why}
Sampling text is a discrete operation, so this objective cannot be differentiated through, and the standard approach optimizes it with PPO, using $r_\phi(x, y) - \beta(\log\pi_\theta(y\mid x) - \log\pi_{\mathrm{ref}}(y\mid x))$ as the reward. {#c-ra-ppo-reward}
They call RLHF a complex and often unstable procedure: it first fits a reward model to the human preferences, then fine-tunes the language model with reinforcement learning to maximize that estimated reward while not drifting far from the original model. {#c-ra-rlhf-complex}

## The method

DPO uses the same preference data and the same objective, but never builds $r_\phi$.
Its loss over a dataset of triples $(x, y_w, y_l)$ is

$$ \mathcal L_{\mathrm{DPO}}(\pi_\theta; \pi_{\mathrm{ref}}) = -\,\mathbb E_{(x, y_w, y_l)\sim\mathcal D}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)} - \beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right)\right] . $$ {#eq-dpo-loss}

The authors read this as fitting an implicit reward model, $\hat r_\theta(x, y) = \beta\log(\pi_\theta(y\mid x)/\pi_{\mathrm{ref}}(y\mid x))$, whose optimal policy is $\pi_\theta$ itself. {#c-ra-implicit-reward}
Each log-probability is that of the whole response, the sum of its token log-probabilities given the prompt, so a batch of pairs needs two forward passes of the policy and two of the frozen reference.
The pipeline is to sample completions from $\pi_{\mathrm{ref}}$ for every prompt, label them with human preferences to build the dataset, and optimize $\pi_\theta$ to minimize the loss for the given $\pi_{\mathrm{ref}}$, $\mathcal D$ and $\beta$. {#c-ra-pipeline}
For public preference datasets, which were sampled from some SFT model, the authors set $\pi_{\mathrm{ref}} = \pi^{\mathrm{SFT}}$ when that model is available; when it is not, they train $\pi_{\mathrm{ref}}$ by maximum likelihood on the preferred completions, so that it is closer to the unknown distribution that produced the data. {#c-ra-ref-init}
The PyTorch code for the DPO loss provided in the paper takes the policy and reference model logprobs with the indices of the preferred and dispreferred completions, computes the losses as `-F.logsigmoid(beta * (pi_logratios - ref_logratios))`, and returns them with the rewards `beta * (pi_logps - ref_logps).detach()`. {#c-ra-code}
The implementations here follow it: `dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta)` in both NumPy and PyTorch, with `sequence_logps` to sum token log-probabilities over the response.

## Derivation

**The optimal policy.**
For any reward $r$, the solution of eq-rlhf-objective over all policies is

$$ \pi_r(y\mid x) = \frac{1}{Z(x)}\,\pi_{\mathrm{ref}}(y\mid x)\exp\!\Big(\frac1\beta r(x, y)\Big), \qquad Z(x) = \sum_y \pi_{\mathrm{ref}}(y\mid x)\exp\!\Big(\frac1\beta r(x, y)\Big) . $$ {#eq-optimal-policy}

Rafailov et al. derive it by reorganizing the objective as the minimization of $\mathbb D_{\mathrm{KL}}(\pi\,\Vert\,\pi^*) - \log Z(x)$, where the partition function does not depend on $\pi$, and use Gibbs' inequality: the KL divergence is minimized at 0 if and only if the two distributions are identical. {#c-ra-gibbs}
For a finite set of responses, $\pi^* \propto \pi_{\mathrm{ref}}\exp(r/\beta)$ maximizes $\mathbb E_\pi[r] - \beta\,\mathbb D_{\mathrm{KL}}(\pi\Vert\pi_{\mathrm{ref}})$, the maximum is $\beta\log Z(x)$, and any other policy falls short of it by exactly $\beta\,\mathbb D_{\mathrm{KL}}(\pi\Vert\pi^*)$. {#c-optimal-policy}
Even with the true reward, $Z(x)$ is expensive to estimate, which makes the closed form hard to use directly. {#c-ra-z-expensive}
The derivation `kl-constrained-optimum` gives the steps, and its check verifies the optimum, its value and the gap on 2000 random problems.

**The reward in terms of the policy.**
Taking logarithms of eq-optimal-policy and solving for the reward,

$$ r(x, y) = \beta\log\frac{\pi_r(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)} + \beta\log Z(x) . $$ {#eq-reward-reparam}

Every reward is thus a scaled log-ratio of its optimal policy to the reference plus a term that depends only on the prompt. {#c-reward-reparam}

**The partition function cancels.**
Because a Bradley-Terry probability depends only on the reward difference between the two completions, substituting eq-reward-reparam into eq-bradley-terry makes the partition function cancel, so the human preference probability can be expressed with only the optimal policy $\pi^*$ and the reference policy. {#c-ra-cancels}
Explicitly, $p^*(y_1\succ y_2\mid x) = \sigma\big(\beta\log\frac{\pi^*(y_1\mid x)}{\pi_{\mathrm{ref}}(y_1\mid x)} - \beta\log\frac{\pi^*(y_2\mid x)}{\pi_{\mathrm{ref}}(y_2\mid x)}\big)$, and the maximum-likelihood objective for a parametrized policy $\pi_\theta$ under this model is eq-dpo-loss. {#c-dpo-is-nll}
Rewards that differ by any function of the prompt alone give the same Bradley-Terry preferences and the same optimal policy, so the loss cannot tell them apart and does not need to. {#c-shift-invariance}
The derivation `partition-cancels` gives the steps.
Rafailov et al. call two rewards equivalent when their difference is a function of $x$ alone, and prove two lemmas: equivalent rewards give identical preference distributions under Plackett-Luce, Bradley-Terry included, and identical optimal policies for the KL-constrained problem. {#c-ra-lemmas}
Their Theorem 1 states that, under mild assumptions, every reward class consistent with the Plackett-Luce and Bradley-Terry models can be represented with the reparameterization $r(x, y) = \beta\log(\pi(y\mid x)/\pi_{\mathrm{ref}}(y\mid x))$ for some model $\pi$ and a given reference model. {#c-ra-theorem1}
So the reparameterization loses no generality: whatever reward class the data support, some policy expresses it.
The same substitution works for the Plackett-Luce model over rankings of $K$ responses, where $Z(x)$ again cancels. {#c-ra-plackett-luce}

**The gradient.**
With $\hat r_\theta(x, y) = \beta\log(\pi_\theta(y\mid x)/\pi_{\mathrm{ref}}(y\mid x))$,

$$ \nabla_\theta\mathcal L_{\mathrm{DPO}} = -\beta\,\mathbb E\Big[\sigma\big(\hat r_\theta(x, y_l) - \hat r_\theta(x, y_w)\big)\big(\nabla_\theta\log\pi(y_w\mid x) - \nabla_\theta\log\pi(y_l\mid x)\big)\Big] . $$ {#eq-dpo-gradient}

The gradient increases the likelihood of $y_w$ and decreases that of $y_l$, and weights each example by how much higher the implicit reward rates the dispreferred completion, that is, by how incorrectly it orders the pair, scaled by $\beta$. {#c-ra-gradient}
For one pair, the derivative of the loss in $\log\pi_\theta(y_w\mid x)$ is $-\beta\,\sigma(\hat r_l - \hat r_w)$ and in $\log\pi_\theta(y_l\mid x)$ is $+\beta\,\sigma(\hat r_l - \hat r_w)$; the weight lies between 0 and 1, is $\tfrac12$ when the two implicit rewards are equal and goes to 0 as the margin $\hat r_w - \hat r_l$ grows. {#c-gradient-form}
The authors found the weighting important: without the coefficient, a naive version of the method can make the language model degenerate. {#c-ra-naive-degenerate}
They relate this to actor-critic RLHF, where a normalization term, interpretable as a soft value function of $\pi_{\mathrm{ref}}$, must be estimated with a learned value function or a baseline to keep the policy gradient's variance down, whereas the DPO reward needs no baseline at all. {#c-ra-no-baseline}
The derivation `dpo-gradient` gives the steps, and its check compares them with finite differences, also through a softmax policy.

## Worked example

**One pair.**
Take $\beta = 0.1$ and a pair whose chosen response has log-probability $-12$ under the policy and $-13$ under the reference, and whose rejected response has $-15$ and $-14$.

| quantity | value |
|---|---|
| implicit reward $\hat r_w = 0.1\cdot(-12 + 13)$ | 0.1 |
| implicit reward $\hat r_l = 0.1\cdot(-15 + 14)$ | −0.1 |
| margin $\hat r_w - \hat r_l$ | 0.2 |
| loss $-\log\sigma(0.2)$ | 0.598139 |
| weight $\sigma(\hat r_l - \hat r_w)$ | 0.450166 |
| $\partial\mathcal L/\partial\log\pi_\theta(y_w)$ | −0.045017 |
| $\partial\mathcal L/\partial\log\pi_\theta(y_l)$ | +0.045017 |

With these log-probabilities and $\beta = 0.1$, the DPO loss is $\log(1 + e^{-0.2}) = 0.598139$, and the derivative in the policy log-probability of the chosen response is $-0.1\,\sigma(-0.2) = -0.045017$. {#c-we-pair}
If policy and reference agreed the loss would be $\log 2 = 0.693147$; the policy already prefers $y_w$ a little more than the reference does, so the loss is lower and the weight is below one half.
`dpo_loss` returns 0.598139 in both implementations, and `test_worked_example` checks the rows.

**The optimal policy.**
With three responses, $\pi_{\mathrm{ref}} = (0.5, 0.3, 0.2)$, rewards $r = (1, 0, -1)$ and $\beta = 0.5$, eq-optimal-policy gives $Z = 0.5e^2 + 0.3 + 0.2e^{-2} = 4.021595$ and $\pi^* = (0.918672, 0.074597, 0.006730)$; with $\beta = 10$ it gives $(0.534647, 0.290261, 0.175093)$, close to the reference. {#c-we-optimal}
The Bradley-Terry probability that the first response beats the second is $\sigma(1) = 0.731059$, both from the rewards and from the log-ratios $0.5\log(\pi^*/\pi_{\mathrm{ref}})$, which equal $r - 0.5\log Z$. {#c-we-bt}
`test_dpo_recovers_the_optimal_policy` trains a four-response softmax policy with the DPO loss on Bradley-Terry preferences over all pairs and recovers this closed-form optimum to $10^{-6}$.

## Hyperparameters

By default the authors train with $\beta = 0.1$, 64 pairs per batch and RMSprop at a learning rate of 1e-6, reached by a linear warm-up over the first 150 steps; their TL;DR summarization runs raise $\beta$ to 0.5 and keep the rest. {#c-ra-hparams}
In the sentiment experiments they swept $\beta \in \{0.05, 0.1, 1, 5\}$ and, for PPO, a target KL in $\{3, 6, 9, 12\}$. {#c-ra-sweep}
They note that they did not meaningfully tune $\beta$ in the summarization experiments, so the results there may underestimate DPO's potential. {#c-ra-beta-untuned}
In the optimal policy, $\beta$ acts as a temperature on the reward: as $\beta$ grows $\pi^*$ approaches $\pi_{\mathrm{ref}}$, and as it shrinks $\pi^*$ concentrates on the highest-reward responses, as the worked example shows.

## Cost

Training needs the policy and a frozen copy of the reference in memory, and per pair a forward and backward pass of the policy on two sequences and a forward pass of the reference; the reference log-probabilities can be computed once and stored.
There is no reward model to train, no value network and no generation during training, which is where most of the cost and the engineering of PPO-based RLHF lie.
Rafailov et al. describe DPO as stable, performant and computationally lightweight, since it needs neither sampling from the language model while fine-tuning nor significant hyperparameter tuning. {#c-ra-lightweight}

## Evidence

**Controlled sentiment.**
In the IMDb task the prompt is a prefix of a movie review and the policy must continue it with positive sentiment; preference pairs are labeled with a pre-trained sentiment classifier, and the SFT model is GPT-2-large fine-tuned on IMDb reviews. {#c-ra-imdb-setup}
Across 22 training runs with different conservativeness settings, the authors measured the true reward and the sequence-level KL to the reference, and found that DPO reached more reward than the other methods at each level of KL. {#c-ra-reward-kl}
Its reward-KL trade-off strictly dominated PPO's, even when PPO had access to the ground-truth reward. {#c-ra-dominates-ppo-gt}

**Summarization.**
On Reddit TL;DR, with GPT-4 judging against the human reference summaries, DPO reached a win rate of about 61% at temperature 0, against 57% for PPO at its best temperature, also 0, and was much more robust to the sampling temperature. {#c-ra-tldr}
In a human evaluation, DPO samples at temperature 0.25 were preferred 58% of the time over PPO samples at temperature 0. {#c-ra-tldr-human}

**Dialogue.**
On single-turn dialogue from the Anthropic Helpful and Harmless dataset, with Pythia-2.8B fine-tuned on the preferred completions as the reference, DPO was the only computationally efficient method that improved over the preferred completions in the test set, and performed similarly to or better than the computationally demanding Best of 128 baseline. {#c-ra-hh}
The authors could not find a prompt or sampling temperature with which a publicly available PPO model trained on the same dataset beat the base Pythia-2.8B, and so took Best of 128 as a rough proxy for PPO-level performance. {#c-ra-hh-ppo}

**Distribution shift and evaluation.**
Applied to CNN/DailyMail news articles, the TL;DR policies had GPT-4 win rates against the ground-truth summaries of 0.36 for DPO and 0.26 for PPO at temperature 0. {#c-ra-cnn}
In a human study on TL;DR samples, GPT-4 agreed with humans about as often as humans agreed with each other, which the authors take as support for GPT-4 as a proxy for human evaluation. {#c-ra-gpt4-agree}
They also found that the win rates computed by GPT-4 are affected by the prompt, and that GPT-4 with a simple prompt prefers longer, more repetitive summaries than humans do. {#c-ra-gpt4-length}

**Against tuned PPO.**
Xu et al. benchmarked DPO and PPO on dialogue and code generation and report that PPO surpassed the other alignment methods in all cases, including challenging code competitions. {#c-xu-ppo-wins}
They identify advantage normalization, a large batch size and an exponential moving average update of the reference model as critical factors for PPO's best performance in fine-tuning language models. {#c-xu-ppo-factors}

## Limitations and failure modes

**Deterministic preferences.**
Azar et al. consider two actions $y$ and $y'$ where $y$ is always preferred: the Bradley-Terry model would then require $r(y) - r(y')$ to go to $+\infty$, and the optimal policy would give $\pi^*(y') = 0$ irrespective of the constant $\tau$ used for the KL regularisation, so the regularisation becomes weaker and weaker the more deterministic the preferences. {#c-az-deterministic}
In the finite-data regime this becomes overfitting: even when the true preference probability is 0.8, a few samples can give an empirical estimate of 1, and the empirical optimal policy then sets the probability of the dispreferred response to 0 for any regularization strength. {#c-az-finite-data}
They argue that standard RLHF is more robust here because the reward model is underfit and regularized in practice, and that DPO, in avoiding the training of a reward function, loses the regularization toward the reference that the underfit reward provides. {#c-az-rlhf-robust}
In eq-dpo-gradient the weight $\sigma(\hat r_l - \hat r_w)$ is positive for every finite margin, so on a pair that is never contradicted the loss keeps pushing the margin up; `test_deterministic_preference_dpo_diverges_ipo_does_not` shows the margin still growing after 3000 steps.

**Out-of-distribution responses.**
Xu et al. prove that the policies PPO can reach from a reward model trained on a preference dataset form a proper subset of the policies that minimize the DPO loss on the same dataset. {#c-xu-subset}
Their counter-example has three actions, a reference policy that gives the third no probability and a dataset with a single comparison of the first two; DPO can minimize the loss with a policy that puts 0.9 on the third action, which PPO cannot produce because the reference forces it to give that action zero probability. {#c-xu-counterexample}
They report that DPO's performance is significantly affected by the distribution shift between the model's outputs and the preference dataset. {#c-xu-shift}

**Scope of the original evidence.**
Rafailov et al. evaluated models of up to 6B parameters, and list scaling to much larger models, out-of-distribution generalization and reward over-optimization in the direct setting as open questions. {#c-ra-limits}

## Variants

**IPO.**
Azar et al. derive a general objective, $\Psi$PO, written in terms of pairwise preferences, of which RLHF and DPO are special cases, and take $\Psi$ to be the identity to obtain IPO. {#c-az-psipo}
With $h_\pi(y_w, y_l) = \log\frac{\pi(y_w)\pi_{\mathrm{ref}}(y_l)}{\pi(y_l)\pi_{\mathrm{ref}}(y_w)}$, the sampled IPO loss is

$$ \mathcal L_{\mathrm{IPO}} = \mathbb E_{(y_w, y_l)\sim\mathcal D}\Big[\big(h_\pi(y_w, y_l) - \tfrac{1}{2\tau}\big)^2\Big] , $$ {#eq-ipo-loss}

which regresses the gap between the log-likelihood ratios of policy and reference to $1/(2\tau)$, so it always regularizes toward $\pi_{\mathrm{ref}}$. {#c-az-ipo-loss}
In their two-action example with a deterministic preference, DPO converges to the deterministic policy whatever $\tau$, while for IPO $\tau$ controls how close the solution stays to the reference. {#c-az-two-action}
`ipo_loss` implements eq-ipo-loss; on the worked-example pair, with $h = 2$ and $\tau = 0.1$, it gives $(2 - 5)^2 = 9$.

**Rankings.**
The Plackett-Luce form of the loss trains on rankings of $K$ responses instead of pairs, and reduces to eq-dpo-loss when $K = 2$.

**Unlikelihood.**
The paper's unlikelihood baseline simply maximizes the probability of $y_w$ and minimizes that of $y_l$; the authors found that it fails to generate meaningful responses for complex problems such as summarization and dialogue. {#c-ra-unlikelihood}

## When to use it

Use DPO when you have, or can collect, pairwise preferences over responses to the same prompts, and you want the effect of KL-regularized RLHF with the machinery of supervised fine-tuning.
Start from the SFT model as both the initial policy and the reference, with $\beta = 0.1$.
Prefer data sampled from the reference model itself; when reusing a public dataset, first fine-tune on its preferred responses so that they are in distribution.
Watch the implicit rewards, which `dpo_loss` returns detached, and the preference accuracy; a margin that keeps rising on training pairs is a sign of the overfitting Azar et al. describe, and IPO or early stopping are the remedies they discuss.
When a reward signal can be queried on fresh samples, or the preference data cover the policy's outputs poorly, the results of Xu et al. suggest that well-tuned PPO is worth its cost.

## Pitfalls

> [!pitfall]
> **Writing the loss as `-torch.log(torch.sigmoid(x))`.** For a large negative margin the sigmoid underflows to 0 and the loss becomes infinite. Use `F.logsigmoid` or `-softplus(-x)`. `test_stable_for_large_margins` shows the naive form returning inf where `dpo_loss` returns $10^5$ with gradient $-1$.

> [!pitfall]
> **Letting the reference receive gradient.** The reference is frozen: compute its log-probabilities under `torch.no_grad()` or detach them, and log the implicit rewards detached. `dpo_loss` detaches both, and `test_reference_gets_no_gradient_and_rewards_are_detached` checks it.

> [!pitfall]
> **Misaligned log-probabilities.** In a causal model the logits at position $t$ predict token $t + 1$, so they must be shifted by one before gathering, and the prompt and padding positions must be masked out. A mistake here still trains, on the wrong numbers. `test_sequence_logps_shift_and_mask` checks the shift and the mask.

> [!pitfall]
> **Averaging instead of summing token log-probabilities.** The derivation is about the probability of the whole response, the sum over its tokens. A per-token mean is a different, length-normalized objective; it can be a deliberate choice, but it is not eq-dpo-loss.

> [!pitfall]
> **Mixing up chosen and rejected, or the sign.** Swapping them trains the model toward the dispreferred responses, and the loss still decreases. `test_matches_indexed_reference` compares against a hand-built reference written with `F.logsigmoid` in the indexed form of the paper's appendix, and `test_loss_is_softplus_of_negative_margin` checks that the loss falls as the margin grows.

## History

Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning and Chelsea Finn of Stanford University and CZ Biohub published "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" at the 37th Conference on Neural Information Processing Systems, NeurIPS 2023. {#c-hist-dpo}
Mohammad Gheshlaghi Azar, Mark Rowland, Bilal Piot, Daniel Guo, Daniele Calandriello, Michal Valko and Rémi Munos of Google DeepMind posted "A General Theoretical Paradigm to Understand Learning from Human Preferences" as arXiv 2310.12036, marked as under review. {#c-hist-ipo}
Shusheng Xu, Wei Fu, Jiaxuan Gao, Wenjie Ye, Weilin Liu, Zhiyu Mei, Guangju Wang, Chao Yu and Yi Wu posted "Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study" as arXiv 2404.10719 in 2024. {#c-hist-xu}
