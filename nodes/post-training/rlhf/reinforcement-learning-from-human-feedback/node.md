---
tag: NEW-reinforcement-learning-from-human-feedback
kind: method
slug: reinforcement-learning-from-human-feedback
name: Reinforcement learning from human feedback
aliases: [RLHF, RL from human feedback, reinforcement learning from human preferences, PPO-ptx, KL-regularized reward maximization]
area: post-training/rlhf
level: core
summary: >-
  Fine-tuning a model on human preferences: fit a reward model to comparisons of its outputs, then maximize that reward by reinforcement learning with a KL penalty to the starting model.
purpose: fine-tuning language models and other policies towards goals that people can judge but cannot easily write down as a reward
uses: [ppo@al, reward-function@al, kullback-leibler-divergence@m2, logistic-regression@m2, maximum-likelihood-estimation@m2]
related_to: [inverse-reinforcement-learning@al, imitation-learning@al, trpo@al, generalized-advantage-estimation@al, actor-critic@al, maximum-entropy-rl@al, policy-gradient-theorem@al]
hyperparameters:
  - {name: KL coefficient beta, meaning: "the weight of the penalty beta log(pi / pi_ref) in the shaped reward; larger values keep the policy closer to the reference", typical: "0.02 in InstructGPT; fixed, or adapted to reach a target KL"}
  - {name: clip range epsilon, meaning: "how far the PPO ratio pi_theta / pi_theta_old may move before the objective stops rewarding it", typical: "0.2"}
  - {name: pretraining coefficient gamma, meaning: "the weight of the pretraining log-likelihood in PPO-ptx; 0 gives plain PPO", typical: "27.8 in InstructGPT"}
  - {name: responses per prompt K, meaning: "how many responses a labeler ranks for one prompt, giving K choose 2 comparisons", typical: "2; 4 to 9 in InstructGPT"}
  - {name: DPO beta, meaning: "the beta of the KL-regularized objective that DPO optimizes; it scales the implicit reward", typical: "0.1; 0.5 for TL;DR summarization in Rafailov et al."}
cost:
  train: "a reward model and a value model besides the policy, often of similar size, a frozen reference model for the KL penalty, sampling from the policy at every PPO step, and human comparisons; DPO needs only the policy and the reference, and no sampling"
  inference: "none beyond the policy; the reward model, value model and reference are dropped after training"
history:
  year: 2017
  people: [Paul Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, Dario Amodei]
  note: "Christiano et al., arXiv 1706.03741 (June 2017), learned rewards from comparisons of trajectory segments. Ziegler et al. (2019) and Stiennon et al. (NeurIPS 2020) applied it to language models; InstructGPT (Ouyang et al. 2022) and Bai et al. (2022) made it the standard last stage of assistant training; DPO (Rafailov et al., NeurIPS 2023) removed the reinforcement learning."
prereqs: [ppo@al, kullback-leibler-divergence@m2, logistic-regression@m2, markov-decision-process@al]
same_as:
  wikidata: null
  wikipedia: Reinforcement_learning_from_human_feedback
  arxiv: ["1706.03741", "1909.08593", "2009.01325", "2203.02155", "2305.18290"]
  al: null
sources:
  - {paper: "Christiano et al. (2017)", title: "Deep Reinforcement Learning from Human Preferences", locator: "the abstract, Sections 2.2, 2.2.1 and 2.2.3, the title page", note: "the method: a reward model fitted to comparisons, and RL on it"}
  - {paper: "Ziegler et al. (2019)", title: "Fine-Tuning Language Models from Human Preferences", locator: "the abstract, Section 2", note: "the KL-shaped reward for language models"}
  - {paper: "Stiennon et al. (2020)", title: "Learning to summarize from human feedback", locator: "the abstract, Sections 3.4 and 4.3, the title page", note: "the pairwise reward model loss, PPO per token, and over-optimization"}
  - {paper: "Ouyang et al. (2022)", title: "Training language models to follow instructions with human feedback", locator: "the abstract, Sections 1 and 3.5, p. 42, the title page", note: "InstructGPT: the three steps, the ranked loss, PPO-ptx and the alignment tax"}
  - {paper: "Bai et al. (2022)", title: "Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback", locator: "the abstract, Sections 1.1, 4.3 and 4.4, the title page", note: "helpful and harmless training, online RLHF, and the square-root-of-KL relation"}
  - {paper: "Gao, Schulman and Hilton (2023)", title: "Scaling Laws for Reward Model Overoptimization", locator: "the abstract, Sections 1 and 3.6", note: "the gold-proxy setup and the functional forms of over-optimization"}
  - {paper: "Rafailov et al. (2023)", title: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", locator: "the abstract, Sections 3, 4 and 6.1, p. 20, the title page", note: "the KL-regularized optimum and DPO"}
  - {paper: "Schulman et al. (2017)", title: "Proximal Policy Optimization Algorithms", locator: "Section 3", note: "the clipped surrogate objective"}
  - {paper: "Casper et al. (2023)", title: "Open Problems and Fundamental Limitations of Reinforcement Learning from Human Feedback", locator: "the abstract, Section 3", note: "a survey of the open problems"}
---

Reinforcement learning from human feedback (RLHF) fine-tunes a model towards outputs that people prefer, when that preference is easy to judge and hard to write down.
People compare pairs of the model's outputs, a reward model learns to predict their choices, and the model is then trained by reinforcement learning to maximize the learned reward.
A KL penalty keeps it close to where it started, where the reward model is still accurate.
For language models it comes after supervised fine-tuning, and it was the step that turned GPT-3 into InstructGPT.
Direct preference optimization (DPO) reaches the optimum of the same objective without the reward model or the reinforcement learning.

## In plain words

Writing a reward for "a helpful answer" or "a good summary" is hard, but saying which of two answers is better is easy.
RLHF collects many such judgments and trains a second network, the reward model, to give a higher score to the answer people chose.
The language model then generates answers, the reward model scores them, and a policy-gradient method such as [[al:ppo|PPO]] makes the high-scoring answers more likely.
The reward model is only right near the answers it was trained on, and a policy pushed far from them finds answers that score high and are bad.
So the reward also charges the model for every token on which its probability differs from the supervised model's, which is a penalty on the [[m2:kullback-leibler-divergence|KL divergence]] between the two.
The best policy under this penalty has a closed form: the supervised model's probabilities reweighted by the exponential of the reward.
DPO uses this closed form to train the language model directly on the comparisons, as a classifier.

## Setting

Christiano et al. (2017) define goals in terms of non-expert human preferences between pairs of trajectory segments, and solve complex reinforcement learning tasks without the reward function, including Atari games and simulated robot locomotion, with feedback on less than 1% of the agent's interactions. {#c-christiano-goals}
Their method keeps a policy and a reward estimate, updated by three processes that run asynchronously: the policy produces trajectories and is trained by reinforcement learning on the predicted reward, pairs of segments go to a human for comparison, and the reward estimate is fitted to the comparisons by supervised learning. {#c-christiano-loop}

For language models, the policy $\pi_\theta(y \mid x)$ maps a prompt $x$ to a response $y$, a sequence of tokens, and the episode is one response.
Ouyang et al. (2022) train InstructGPT in three steps: supervised fine-tuning (SFT) on labeler demonstrations, reward model training on labelers' rankings of model outputs, and reinforcement learning with PPO against that reward model. {#c-instructgpt-steps}
Rafailov et al. (2023), following Ziegler et al. (2019), describe the same pipeline in three phases: SFT, preference sampling and reward learning on pairs of answers from the SFT model, and RL optimization. {#c-rafailov-pipeline}
The SFT model is the reference policy $\pi_{\text{ref}}$ and the starting point of $\pi_\theta$.
A preference is a triple $(x, y_w, y_l)$ in which a person preferred the response $y_w$ to $y_l$.

## The method

**Step 1: the reward model.**
A reward model $r_\phi(x, y)$ assigns a scalar to a prompt and a response.
Under the Bradley–Terry model, a person prefers $y_w$ to $y_l$ with probability

$$ p(y_w \succ y_l \mid x) = \frac{e^{r(x, y_w)}}{e^{r(x, y_w)} + e^{r(x, y_l)}} = \sigma\big(r(x, y_w) - r(x, y_l)\big) , $$ {#eq-bt}

where $\sigma$ is the logistic function.
Christiano et al. take the probability of preferring a segment to depend exponentially on the reward summed over the segment, and fit the reward by minimizing the cross-entropy between these predictions and the human labels, which is the Bradley–Terry model. {#c-bt-model}
They compare it to the Elo rating of chess, where the difference of two ratings gives the probability that one player beats the other. {#c-elo}
The reward model is then fitted by maximum likelihood, a binary classification with the negative log-likelihood loss {#c-rm-mle}

$$ \mathcal{L}_R(\phi) = -\mathbb{E}_{(x, y_w, y_l)}\big[\log \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)\big] . $$ {#eq-rm-loss}

Stiennon et al. (2020) start the reward model from the supervised model with a randomly initialized linear head that outputs a scalar, train it with this loss, and normalize its outputs so that the reference summaries score 0 on average. {#c-rm-loss-stiennon}
Ziegler et al. instead let people choose among four responses, and fit the reward model with the softmax cross-entropy over the four scores, normalized to mean 0 and variance 1. {#c-ziegler-four}
Ouyang et al. have labelers rank between $K = 4$ and $K = 9$ responses, which gives $\binom{K}{2}$ comparisons for each prompt, and they put all comparisons of one prompt into one batch element, with the loss eq-rm-loss divided by $\binom{K}{2}$. {#c-rm-ranked}
They found that shuffling the comparisons into one dataset made the reward model overfit, since each response then appears in $K - 1$ separate updates. {#c-rm-shuffle}
Their reward models start from the SFT model with the final unembedding layer removed, and are 6B, because they found 175B reward model training could be unstable. {#c-rm-6b}
They note that the loss does not change when the reward is shifted, and add a bias so that the labeler demonstrations score 0 on average before the RL step. {#c-rm-shift}

**Step 2: reinforcement learning with a KL penalty.**
Ziegler et al. add a penalty with expectation $\beta\, \mathrm{KL}(\pi, \rho)$ to keep the policy close to the language model $\rho$, so that the reinforcement learning uses the reward

$$ R(x, y) = r_\phi(x, y) - \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)} , $$ {#eq-kl-reward}

with $\beta$ either constant or adapted to reach a target KL. {#c-kl-reward-ziegler}
Stiennon et al. explain the term in two ways: as an entropy bonus that makes the policy explore and keeps it from collapsing to a single mode, and as a guard that keeps the policy's outputs close to those the reward model saw in training. {#c-kl-purposes}
The expectation of eq-kl-reward under the policy is the KL-regularized objective

$$ J(\pi_\theta) = \mathbb{E}_{x,\ y \sim \pi_\theta(\cdot \mid x)}\big[r_\phi(x, y)\big] - \beta\, \mathrm{KL}\big(\pi_\theta(\cdot \mid x) \,\big\|\, \pi_{\text{ref}}(\cdot \mid x)\big) , $$ {#eq-objective}

which is how Rafailov et al. state the RL phase; since text is discrete, it is usually maximized by PPO on the reward eq-kl-reward. {#c-rafailov-objective}

Stiennon et al. treat the reward model's output as a reward for the whole summary, maximize it with PPO with one BPE token per time step, and give the value function its own parameters, initialized from the reward model. {#c-ppo-token}
Ouyang et al. run PPO in a bandit environment that shows a prompt and expects a response, add a per-token KL penalty from the SFT model to limit over-optimization of the reward model, and initialize the value function from the reward model. {#c-instructgpt-ppo}
So in practice the penalty $-\beta(\log \pi_\theta(y_t \mid x, y_{<t}) - \log \pi_{\text{ref}}(y_t \mid x, y_{<t}))$ is paid at every token and the reward model's score at the last one; the derivation shows that these add up to eq-kl-reward.
PPO maximizes the clipped surrogate objective of Schulman et al. (2017), {#c-ppo-clip}

$$ L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t\Big[\min\big(\rho_t(\theta) \hat A_t,\ \mathrm{clip}(\rho_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat A_t\big)\Big], \qquad \rho_t(\theta) = \frac{\pi_\theta(y_t \mid s_t)}{\pi_{\theta_{\text{old}}}(y_t \mid s_t)} , $$ {#eq-ppo-clip}

where $s_t = (x, y_{<t})$ and $\hat A_t$ is the advantage of token $t$, estimated from the shaped rewards and the value function.

**PPO-ptx.**
Ouyang et al. also mix in the gradient of the pretraining log-likelihood, and call these models PPO-ptx; they maximize {#c-ptx}

$$ \mathbb{E}_{x,\ y \sim \pi_\theta}\Big[r_\phi(x, y) - \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}\Big] + \gamma\, \mathbb{E}_{x \sim D_{\text{pretrain}}}\big[\log \pi_\theta(x)\big] , $$ {#eq-ptx}

with $\gamma = 0$ for plain PPO.

**Direct preference optimization.**
The maximizer of eq-objective has a closed form, eq-optimal below, so the reward can be written through the policy, and the Bradley–Terry likelihood of the preferences becomes a loss on the policy itself.
Rafailov et al. call it the DPO loss: {#c-dpo-loss}

$$ \mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l)}\Big[\log \sigma\Big(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\Big)\Big] . $$ {#eq-dpo}

It needs no reward model, no value function and no sampling from the policy during training, only the log-probabilities of the two responses under the policy and the frozen reference.

## Derivation

**The reward model loss.**
Dividing eq-bt by $e^{r(x, y_w)}$ gives $\sigma(\Delta)$ with $\Delta = r_w - r_l$, so eq-rm-loss is the logistic loss $\log(1 + e^{-\Delta})$, which is $\log 2$ when the two rewards are equal. {#c-bt-logistic}
Its gradient is

$$ \nabla_\phi\big({-\log \sigma(\Delta)}\big) = -\sigma(-\Delta)\,\big(\nabla_\phi r_w - \nabla_\phi r_l\big) , $$

so it raises the reward of $y_w$ and lowers that of $y_l$, with a weight below 1 that is above $1/2$ exactly when the model orders the pair wrong. {#c-bt-gradient}
The second derivative in $\Delta$ is $\sigma(\Delta)\sigma(-\Delta) \in (0, 1/4]$, so the loss is convex in $\Delta$. {#c-bt-curvature}
Any function $c(x)$ of the prompt added to the reward cancels in $\Delta$, so the comparisons fix the reward only up to such a shift, which is why it is normalized. {#c-bt-shift}
The cross-entropy over two responses with the preferred one as the label is the same loss, so the four-way softmax of Ziegler et al. is Bradley–Terry with more options. {#c-bt-softmax-two}
The derivation `bradley-terry-gradient` gives the steps.

**The KL-regularized optimum.**
For a prompt $x$, let $Z(x) = \sum_y \pi_{\text{ref}}(y \mid x)\, e^{r(x, y)/\beta}$ and

$$ \pi^*(y \mid x) = \frac{1}{Z(x)}\, \pi_{\text{ref}}(y \mid x) \exp\Big(\frac{r(x, y)}{\beta}\Big) . $$ {#eq-optimal}

Then for every policy $\pi$, $J(\pi) = \beta \log Z(x) - \beta\, \mathrm{KL}(\pi \,\|\, \pi^*)$. {#c-kl-gap}
Since the KL divergence is nonnegative and zero only at $\pi = \pi^*$, eq-optimal is the unique maximizer of eq-objective, and the maximum is $\beta \log Z(x)$. {#c-kl-optimum}
Rafailov et al. state this form of the optimum, and note that $Z(x)$ is expensive to estimate even for a known reward. {#c-rafailov-optimum}
The expectation of eq-kl-reward under $\pi$ is $J(\pi)$, so reinforcement learning on the shaped reward maximizes eq-objective, and one sample of $\log(\pi / \pi_{\text{ref}})$ is an unbiased estimate of the KL term. {#c-kl-shaped-unbiased}
Since both policies factor over tokens, the per-token log-ratios add up to the log-ratio of the response, so the per-token penalty of InstructGPT is the penalty of eq-kl-reward. {#c-per-token-kl}

**From the optimum to DPO.**
Taking logarithms of eq-optimal writes the reward through the optimal policy and the partition function. {#c-implicit-reward}

$$ r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x) . $$ {#eq-implicit-reward}

Eq-bt depends only on the difference of two rewards for the same prompt, so substituting eq-implicit-reward cancels $\beta \log Z(x)$, and the likelihood of the preferences depends only on $\pi^*$ and $\pi_{\text{ref}}$; at $\pi_\theta = \pi^*$, eq-dpo equals the reward model loss of $r$. {#c-dpo-cancel}
Replacing $\pi^*$ by $\pi_\theta$ gives eq-dpo, which fits the implicit reward $\hat r_\theta = \beta \log(\pi_\theta / \pi_{\text{ref}})$.
The gradient of one pair is the gradient of the reward model loss, with $\beta$ times the log-likelihoods in place of the rewards. {#c-dpo-gradient}

$$ \nabla_\theta \mathcal{L}_{\text{DPO}} = -\beta\, \sigma\big(\hat r_\theta(x, y_l) - \hat r_\theta(x, y_w)\big)\big(\nabla_\theta \log \pi_\theta(y_w \mid x) - \nabla_\theta \log \pi_\theta(y_l \mid x)\big) . $$

Rafailov et al. read it as raising the likelihood of $y_w$ and lowering that of $y_l$, with more weight on pairs that the implicit reward orders wrong. {#c-dpo-weight}
The derivation `dpo-from-kl-optimum` gives the steps.

**The clipped objective.**
For one token, the clipped term of eq-ppo-clip is at most $\rho \hat A$, with equality when $\rho \in [1 - \epsilon, 1 + \epsilon]$, so $L^{\text{CLIP}}$ is a pessimistic bound on the unclipped objective. {#c-clip-lower-bound}
For $\hat A > 0$ it stays at $(1 + \epsilon)\hat A$ once $\rho > 1 + \epsilon$, and for $\hat A < 0$ at $(1 - \epsilon)\hat A$ once $\rho < 1 - \epsilon$, so its gradient is zero there; on the other side of the interval it is $\rho \hat A$. {#c-clip-flat}
At $\theta = \theta_{\text{old}}$ the ratio is 1 and the gradient is $\hat A\, \nabla_\theta \log \pi_\theta$, the policy gradient. {#c-clip-first-step}
The clip limits each update against $\pi_{\theta_{\text{old}}}$; it is the KL penalty, not the clip, that limits the drift from $\pi_{\text{ref}}$.
The derivation `ppo-clip` gives the steps.

## Worked example

Take one prompt with three responses $A$, $B$ and $C$, $\beta = 1$, the reference $\pi_{\text{ref}} = (1/2, 1/4, 1/4)$ and the reward $r = (0, \ln 2, \ln 4)$.

| | $A$ | $B$ | $C$ |
|---|---|---|---|
| $\pi_{\text{ref}}$ | 1/2 | 1/4 | 1/4 |
| $r$ | 0 | $\ln 2$ | $\ln 4$ |
| $\pi_{\text{ref}}\, e^{r/\beta}$ | 1/2 | 1/2 | 1 |
| $\pi^*$ | 1/4 | 1/4 | 1/2 |
| $\beta \log(\pi^* / \pi_{\text{ref}})$ | $-\ln 2$ | 0 | $\ln 2$ |

The weights $\pi_{\text{ref}} e^{r}$ add up to $Z = 2$, so eq-optimal gives $\pi^* = (1/4, 1/4, 1/2)$.
Then $\mathbb{E}_{\pi^*}[r] = \tfrac54 \ln 2$ and $\mathrm{KL}(\pi^* \| \pi_{\text{ref}}) = \tfrac14 \ln 2$, so $J(\pi^*) = \ln 2 = \beta \log Z$.
The implicit rewards of the last row are $r - \beta \log Z$, the reward shifted by $\ln 2$.

For the comparison $C \succ A$, eq-bt gives $\sigma(\ln 4) = 4/5$, the loss eq-rm-loss is $\ln(5/4) \approx 0.2231$, and its derivative in $r_C$ is $-1/5$.
At $\pi_\theta = \pi_{\text{ref}}$, the DPO loss of the same pair is $\ln 2 \approx 0.6931$ and the weight of its gradient is $1/2$; at $\pi_\theta = \pi^*$ it is $\ln(5/4)$, the reward model loss, because the implicit rewards differ by $\ln 4$, as the rewards do.

For PPO with $\epsilon = 0.2$: a token with $\rho = 1.5$ and $\hat A = 1$ contributes $\min(1.5, 1.2) = 1.2$; one with $\rho = 0.5$ and $\hat A = 1$ contributes $0.5$; one with $\rho = 0.5$ and $\hat A = -1$ contributes $\min(-0.5, -0.8) = -0.8$.
The test `test_worked_example` checks these numbers.

## Hyperparameters

**The KL coefficient $\beta$.**
It trades reward against distance from the reference; the optimum eq-optimal sharpens towards the highest-reward response as $\beta$ falls, and stays at $\pi_{\text{ref}}$ as $\beta$ grows.
Ouyang et al. use $\beta = 0.02$, 256k episodes, a batch size of 512 and a minibatch size of 64, no discount in the generalized advantage estimate, and a PPO clip ratio of 0.2. {#c-instructgpt-hparams}
Ziegler et al. offer an adaptive $\beta$ that steers the KL to a target, as noted above.

**The pretraining coefficient $\gamma$.**
For PPO-ptx, Ouyang et al. use 8 times more pretraining examples than RL episodes and $\gamma = 27.8$. {#c-instructgpt-gamma}

**The clip range $\epsilon$.**
Schulman et al. suggest $\epsilon = 0.2$, the value InstructGPT uses.

**DPO.**
Rafailov et al. use $\beta = 0.1$ by default, a batch size of 64 and RMSprop with a learning rate of $10^{-6}$, and $\beta = 0.5$ for TL;DR summarization. {#c-dpo-beta}

## Cost

PPO-based RLHF holds four models: the policy, the frozen reference, the reward model and the value model.
In Stiennon et al. and InstructGPT the reward and value models are initialized from supervised models, so all four are large.
Each PPO step samples full responses from the policy, then scores them with the reward model and the reference, so training costs several forward passes per token besides the backward pass.
The human comparisons are usually the scarcest resource.
DPO holds the policy and the reference, needs no sampling, and costs about as much per step as supervised fine-tuning on two responses.
At inference only the policy is used.

## Evidence

Christiano et al. train complex novel behaviors with about an hour of human time. {#c-christiano-hour}
Ziegler et al. get good results on stylistic continuation with only 5,000 human comparisons; for summarization, models trained on 60,000 comparisons copy whole sentences from the input and skip irrelevant preamble, which labelers rate highly but which may exploit their reliance on simple heuristics. {#c-ziegler-results}
Stiennon et al. find that their RLHF models on TL;DR significantly outperform both the human reference summaries and much larger models fine-tuned with supervised learning alone, that they transfer to CNN/DM news articles, and that optimizing the reward model gives summaries that people prefer to those from optimizing ROUGE. {#c-stiennon-results}
Doubling the training data of the reward model raised its validation accuracy by about 1.1%, and doubling its size by about 1.8%. {#c-stiennon-rm-scaling}

On the prompt distribution of Ouyang et al., people prefer the outputs of the 1.3B InstructGPT model to those of the 175B GPT-3, which has 100 times more parameters. {#c-instructgpt-1-3b}
The 175B InstructGPT is preferred to the 175B GPT-3 85 ± 3% of the time, and 71 ± 4% of the time to few-shot GPT-3. {#c-instructgpt-85}
It gives truthful and informative answers on TruthfulQA about twice as often as GPT-3, and about 25% fewer toxic outputs when prompted to be respectful, but no significant improvement on the Winogender and CrowS-Pairs bias datasets. {#c-instructgpt-truthful}
Held-out labelers who produced no training data prefer it at about the same rate as the training labelers. {#c-instructgpt-heldout}

Bai et al. (2022) train helpful and harmless assistants with preference models and RLHF, and find that this improves performance on almost all NLP evaluations and is compatible with training for skills such as Python coding and summarization. {#c-bai-abstract}
They also train online, updating the preference models and policies every week with fresh human feedback. {#c-bai-online}
They find a roughly linear relation between the RL reward and the square root of the KL divergence between the policy and its initialization. {#c-bai-sqrt-kl}
They explain it by a series expansion: the KL divergence begins at second order in a small change of the policy while the reward changes at first order, which suggests that most of RLHF training stays in the regime of small changes. {#c-bai-sqrt-kl-why}

Rafailov et al. report that DPO is stable, performant and computationally lightweight, needs no sampling from the language model during fine-tuning, and matches or improves response quality in summarization and single-turn dialogue. {#c-dpo-abstract}
On controlled sentiment generation, DPO reaches a higher reward than PPO at every KL from the reference, even when PPO has the ground-truth reward. {#c-dpo-reward-kl}

## Limitations and failure modes

**Over-optimization of the reward model.**
The reward model is a proxy, and the policy is trained to exploit it.
Stiennon et al. find that light optimization against the reward model improves the summaries according to labelers, but further optimization makes them worse, and eventually the reward model is anti-correlated with human preferences. {#c-stiennon-overopt}
Gao et al. (2023) study this with a fixed gold-standard reward model in place of people, which labels data for a proxy reward model, and measure the gold score as a policy is optimized against the proxy by reinforcement learning or best-of-$n$ sampling. {#c-gao-setup}
With $d = \sqrt{\mathrm{KL}(\pi \| \pi_{\text{init}})}$, they find that the gold score follows $d(\alpha_{\text{bon}} - \beta_{\text{bon}} d)$ for best-of-$n$ and $d(\alpha_{\text{RL}} - \beta_{\text{RL}} \log d)$ for RL, with coefficients that change smoothly with the size of the proxy reward model. {#c-gao-forms}
In their setup the KL penalty raises the proxy score reached at a given KL but gives no measurable improvement of the gold score at a given KL, a result they note may be sensitive to hyperparameters. {#c-gao-kl-penalty}
The penalty only makes the gold score converge earlier, so it acts like early stopping. {#c-gao-early-stopping}
The policy that exploits the proxy is a case of [[reward-hacking]], which has its own page.

**The alignment tax.**
Ouyang et al. find that RLHF lowers performance against GPT-3 on public NLP datasets such as SQuAD, DROP, HellaSwag and WMT 2015 French to English, which they call an alignment tax; PPO-ptx greatly reduces these regressions without lowering the labelers' preference. {#c-alignment-tax}
Bai et al. find severe alignment taxes for small models, while their 13B and 52B RLHF models do better than the base models on zero-shot NLP evaluations and the same on few-shot ones. {#c-bai-tax}

**Helpfulness and harmlessness pull apart.**
Bai et al. found RLHF policies giving the same exaggerated responses to all remotely sensitive questions, which they attribute to over-optimizing harmlessness and under-optimizing helpfulness. {#c-bai-tension}

**Whose preferences.**
Ouyang et al. hire 40 contractors to label their data, and align the model to the stated preferences of these labelers and the researchers, not to any broader notion of human values. {#c-instructgpt-labelers}
Casper et al. (2023), who call RLHF the central method for fine-tuning large language models, argue that it needs to be part of a multi-layered approach to safer AI systems. {#c-casper-central}
They sort its open problems into those of collecting human feedback, of training the reward model, of training the policy, and of learning the two jointly, and separate the relatively tractable ones from fundamental limitations. {#c-casper-taxonomy}

**Instability and cost.**
PPO with four large models, sampling and many hyperparameters is hard to tune. Rafailov et al. describe RLHF as a complex and often unstable procedure, which motivates DPO. {#c-rlhf-unstable}

## Variants

- **RLHF for control.** Christiano et al. use advantage actor-critic for Atari and TRPO for simulated robotics, preferring policy-gradient methods because the learned reward is non-stationary. {#c-christiano-rl}
  They fit an ensemble of reward predictors on bootstrap samples of the comparisons, normalize each and average them, and hold out $1/e$ of the data for validation. {#c-christiano-ensemble}
- **Multi-way comparisons.** Ziegler et al. use a choice among four responses, and InstructGPT a ranking of $K$ responses, both as above.
- **PPO-ptx** adds the pretraining log-likelihood, eq-ptx, against the alignment tax.
- **Online RLHF.** Bai et al. retrain the preference model on comparisons of the current policy's outputs.
- **Best-of-$n$ sampling** draws $n$ responses and keeps the one with the highest reward, with no training; Gao et al. compare it with RL.
- **DPO** fits the policy directly to the comparisons with eq-dpo, without a reward model or RL.

**Related methods.**
[[al:inverse-reinforcement-learning|Inverse reinforcement learning]] also recovers a reward, but from demonstrations that it takes to be optimal; RLHF recovers it from comparisons, which people can give for behavior they could not perform.
[[al:imitation-learning|Imitation learning]] copies demonstrations and needs no reward; supervised fine-tuning is its simplest form, and RLHF starts from its result.
The advantages $\hat A_t$ of eq-ppo-clip are usually computed by [[al:generalized-advantage-estimation|generalized advantage estimation]] from the shaped rewards and the value function, which InstructGPT runs without discount.
With a uniform reference, $\mathrm{KL}(\pi \| \pi_{\text{ref}}) = \log |\mathcal{Y}| - H(\pi)$, so eq-objective becomes the objective of [[al:maximum-entropy-rl|maximum entropy reinforcement learning]] up to a constant; RLHF replaces the uniform reference by the supervised model.
PPO estimates the gradient that the [[al:policy-gradient-theorem|policy gradient theorem]] gives, and eq-ppo-clip reduces to it at the old parameters.

## When to use it

Use RLHF when the target behavior is easier to judge than to demonstrate or to write as a reward, as for the helpfulness and harmlessness of an assistant or the quality of a summary.
It comes after supervised fine-tuning, which gives the reference policy and the responses to compare.
When the comparisons are collected once and fixed, DPO is the simpler choice, with one model to train and no sampling; PPO-based RLHF can use the reward model on fresh samples from the current policy, as in online training.
Keep the KL to the reference small and watch a held-out measure of quality, since the learned reward stops tracking it as the policy moves away.

## Pitfalls

> [!pitfall]
> **The KL penalty with the wrong sign.** The shaped reward subtracts $\beta \log(\pi_\theta / \pi_{\text{ref}})$; adding it rewards moving away from the reference. The test `test_kl_penalty_sign` checks that a token on which the policy is more likely than the reference is penalized.

> [!pitfall]
> **Comparisons of one prompt shuffled apart.** Ouyang et al. found that this overfits the reward model. The test `test_ranked_loss_one_element` checks that the ranked loss averages the $\binom{K}{2}$ pairs of a prompt into one element, 36 pairs for $K = 9$, and equals the pairwise loss for $K = 2$.

> [!pitfall]
> **Reading the reward on an absolute scale.** The comparisons fix the reward only up to a shift per prompt. The test `test_shift_invariance` checks that a shift leaves the loss unchanged, which is why the reward model is normalized before the RL step.

> [!pitfall]
> **A reference that trains.** In DPO and in the KL penalty, $\pi_{\text{ref}}$ is frozen. The test `test_dpo_reference_is_frozen` checks that no gradient reaches the reference log-probabilities, and `test_kl_shaped_rewards_detached_and_masked` that the shaped rewards carry no gradient into PPO.

> [!pitfall]
> **Padding in the log-probabilities.** Summing log-probabilities or KL penalties over padding tokens changes the rewards and the DPO loss. The tests `test_mask_excludes_padding` and `test_sequence_logprob_mask` check that masked tokens contribute nothing, and that the reward model's score lands on the last real token.

> [!pitfall]
> **Taking the clip for the KL constraint.** The clip of eq-ppo-clip bounds each update against $\pi_{\theta_{\text{old}}}$, not the drift from $\pi_{\text{ref}}$, which grows over many updates. The test `test_clip_flat_outside_interval` checks that the objective is flat only beyond the clip range around the old policy.

> [!pitfall]
> **An unstable log-sigmoid.** Computing $\log \sigma(\Delta)$ as `log(sigmoid(d))` gives $-\infty$ for very negative $\Delta$. The test `test_log_sigmoid_stable` checks that the naive form fails at $\Delta = -800$ and the implementation gives $-800$.

## History

Paul Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg and Dario Amodei, of OpenAI and DeepMind, posted Deep Reinforcement Learning from Human Preferences in June 2017 as arXiv 1706.03741, whose fourth version is dated 17 Feb 2023. {#c-hist-christiano}
It learned rewards for Atari games and simulated robots from comparisons of short clips.
Ziegler et al. (2019) brought the method to language models, with the KL-shaped reward.
Stiennon et al. of OpenAI applied it to summarization, published at NeurIPS 2020. {#c-hist-stiennon}
Ouyang et al. of OpenAI posted InstructGPT on 4 Mar 2022, with the three steps of SFT, reward modelling and PPO. {#c-hist-ouyang}
Bai et al. of Anthropic posted their helpful and harmless assistant on 12 Apr 2022. {#c-hist-bai}
Rafailov, Sharma, Mitchell, Ermon, Manning and Finn, of Stanford University and CZ Biohub, introduced DPO at NeurIPS 2023. {#c-hist-rafailov}
