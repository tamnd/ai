---
tag: NEW-reward-model
kind: method
slug: reward-model
name: Reward model
aliases: [reward modelling, reward modeling, preference model, PM, RM, Bradley-Terry reward model]
area: post-training/reward-modelling
level: core
summary: >-
  A language model with a scalar head, trained on human comparisons under the Bradley-Terry model so that the preferred response scores higher; it stands in for human judgment in RLHF and best-of-n.
purpose: turning a finite set of human comparisons into a reward function that can score any prompt and response, so that a policy can be optimized or outputs reranked without asking a person each time
uses: [decoder-only-transformer, logistic-regression@m2, maximum-likelihood-estimation@m2]
related_to: [reinforcement-learning-from-human-feedback, direct-preference-optimization, reward-hacking, constitutional-ai, instruction-tuning, expected-calibration-error]
hyperparameters:
  - {name: initialization and size, meaning: "which checkpoint the reward model starts from and how large it is relative to the policy", typical: "the SFT or chat model with the unembedding replaced by a scalar head; InstructGPT used a 6B reward model for a 175B policy, Llama 2 the same size as the chat model"}
  - {name: epochs and learning rate, meaning: "how long to train; comparisons overfit quickly", typical: "one epoch; 9e-6 with cosine decay (InstructGPT), 5e-6 for 70B and 1e-5 for smaller models (Llama 2)"}
  - {name: responses per prompt, meaning: "how many responses a labeler ranks, all of whose pairs go into one batch element", typical: "K = 2 for pairwise data; 4 to 9 in InstructGPT"}
  - {name: margin, meaning: "a shift m in -log sigma(r_w - r_l - m) that asks for a larger gap on clearly separated pairs", typical: "0; Llama 2's small variant 1, 2/3, 1/3, 0 by rating"}
  - {name: label-noise rate, meaning: "the probability epsilon that a label is a random answer, which bounds the loss of a contradicted label", typical: "0; 0.1 in Christiano et al."}
  - {name: ensemble size and combination, meaning: "how many reward models are trained and how their scores are combined", typical: "a single model; or a few members combined by mean, minimum, or mean minus lambda times variance"}
  - {name: normalization, meaning: "the constant subtracted from the reward, which comparisons do not determine, and any rescaling", typical: "shift so that reference responses or the initial policy score 0 on average; whitening before RL"}
cost:
  train: "one forward and backward pass per response, a few epochs or fewer over tens of thousands to a few million comparisons, with a model at most the size of the policy"
  inference: "one forward pass per response scored: one per sampled response during RL, n per prompt for best-of-n"
history:
  year: 2017
  people: [Paul Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, Dario Amodei, Nisan Stiennon, Long Ouyang, Amanda Askell, Yuntao Bai, Leo Gao, Nathan Lambert]
  note: "Christiano et al. (2017) fitted a reward predictor to comparisons of trajectory segments under the Bradley-Terry model and trained Atari and MuJoCo agents on it. Stiennon et al. (2020) put a scalar head on a language model for summarization, InstructGPT (2022) added ranked comparisons, Askell et al. and Bai et al. (2021, 2022) studied scaling, pre-training and calibration of preference models, Gao et al. (2023) measured overoptimization, and RewardBench (2024) made reward models a thing to benchmark directly."
prereqs: [logistic-regression@m2, decoder-only-transformer, instruction-tuning]
same_as:
  wikidata: null
  wikipedia: null
  arxiv: ["1706.03741", "2009.01325", "2203.02155", "2112.00861", "2204.05862", "2307.09288", "2210.10760", "2310.02743", "2312.09244", "2403.13787", "2209.13085"]
  al: null
sources:
  - {paper: "Christiano, Leike, Brown, Martic, Legg and Amodei (2017)", title: "Deep Reinforcement Learning from Human Preferences", locator: "arXiv 1706.03741v4, pp. 5 and 9", note: "the Bradley-Terry fit, label noise, ensembles and normalization, the offline-training failure"}
  - {paper: "Stiennon, Ouyang, Wu, Ziegler and others (2020)", title: "Learning to summarize from human feedback", locator: "arXiv 2009.01325v3, pp. 6, 8 and 9", note: "the scalar head and loss, normalization, scaling, transfer, length bias, comparison with ROUGE"}
  - {paper: "Ouyang, Wu, Jiang, Almeida and others (2022)", title: "Training language models to follow instructions with human feedback", locator: "arXiv 2203.02155v1, pp. 8, 9, 12, 41 and 42", note: "the 6B reward model, ranked comparisons as one batch element, normalization, held-out labelers, training settings"}
  - {paper: "Askell, Bai, Chen, Drain and others (2021)", title: "A General Language Assistant as a Laboratory for Alignment", locator: "arXiv 2112.00861v3, pp. 4, 9 and 15", note: "the value head, ranked versus binary preferences, preference model pre-training"}
  - {paper: "Bai, Jones, Ndousse, Askell and others (2022)", title: "Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback", locator: "arXiv 2204.05862v1, pp. 6, 7, 13 and 14", note: "scaling, calibration, accuracy at high scores, robustness"}
  - {paper: "Touvron, Martin, Stone and others (2023)", title: "Llama 2: Open Foundation and Fine-Tuned Chat Models", locator: "arXiv 2307.09288v2, pp. 10 to 13, 15 and 52", note: "two reward models, the margin, the safety-helpfulness combination, accuracy by rating, hyperparameters"}
  - {paper: "Gao, Schulman and Hilton (2023)", title: "Scaling Laws for Reward Model Overoptimization", locator: "arXiv 2210.10760v1, pp. 2, 4 to 6", note: "the functional forms of overoptimization, the best-of-n KL, data requirements"}
  - {paper: "Coste, Anwar, Kirk and Krueger (2024)", title: "Reward Model Ensembles Help Mitigate Overoptimization", locator: "arXiv 2310.02743v2, pp. 4 and 5", note: "mean, worst-case and uncertainty-weighted ensembles and their effect"}
  - {paper: "Eisenstein, Nagpal, Agarwal, Beirami and others (2024)", title: "Helping or Herding? Reward Model Ensembles Mitigate but do not Eliminate Reward Hacking", locator: "arXiv 2312.09244v3, pp. 1 and 3", note: "underspecification, pretraining-seed ensembles, shared errors"}
  - {paper: "Lambert, Pyatkin, Morrison, Miranda and others (2024)", title: "RewardBench: Evaluating Reward Models for Language Modeling", locator: "arXiv 2403.13787v2, pp. 2, 4 to 8", note: "the benchmark, its sections and score, DPO models as reward models, score distributions"}
  - {paper: "Skalse, Howe, Krasheninnikov and Krueger (2022)", title: "Defining and Characterizing Reward Hacking", locator: "arXiv 2209.13085v2, pp. 1 to 3", note: "the definition of hackability, the impossibility result, the cleaning-robot example"}
---

A reward model scores a response to a prompt with one number, trained so that the response people prefer gets the higher score.
It is fitted to human comparisons, pairs or rankings of responses to the same prompt, under the Bradley-Terry model, which makes the probability of preferring one response a logistic function of the difference of their scores.
For language models it is usually the policy's own architecture with the next-token head replaced by a scalar head.
Once trained, it stands in for the people who labelled the data: reinforcement learning from human feedback optimizes a policy against it, best-of-n sampling reranks outputs with it, and benchmarks evaluate it directly.
It is also the weakest link of that pipeline, because a policy optimized hard against any learned reward finds where the reward is wrong.

## In plain words

People find it much easier to say which of two answers is better than to give an answer a score out of ten.
A reward model turns many such judgments into a scoring function: it reads a prompt and an answer and outputs a number, and it is trained until, on the judged pairs, the preferred answer usually gets the higher number.
Only the differences between scores mean anything; adding the same constant to every answer to a prompt changes nothing.
Since the judgments are noisy, even a perfect reward model agrees with the labels only part of the time.
And since it is a learned approximation, pushing a policy to maximize it eventually rewards the approximation's mistakes rather than what people wanted.

## Setting

The data is a set of comparisons $(x, y_w, y_l)$: a prompt $x$ and two responses, of which a person preferred $y_w$ to $y_l$; or a prompt with $K$ responses ranked from best to worst, which gives $\binom{K}{2}$ such pairs.
The model is a function $r_\phi(x, y)$, and its architecture follows the language model it will judge.
Stiennon et al. build a reward model by starting from a supervised baseline and adding a randomly initialized linear head that outputs a scalar value, trained to predict which of two summaries a human judged better with the loss −E[log σ(r(x, y_i) − r(x, y_{1−i}))], where y_i is the preferred summary. {#c-sti-head}
InstructGPT's reward model starts from the SFT model with the final unembedding layer removed and is trained to take in a prompt and response and output a scalar reward. {#c-ou-arch}
Askell et al.'s preference models consist of a value head that predicts a single scalar score on top of the final token of a context, with larger scores indicating more desirable samples. {#c-as-head}
Llama 2 initializes its reward models from pretrained chat model checkpoints, so that both the chat model and the reward model benefit from knowledge acquired in pretraining. {#c-ll-init}
Starting the reward model from the chat model prevents an information mismatch between the two that could result in favoring hallucinations; the reward model's architecture is that of the pretrained model with the classification head for next-token prediction replaced by a regression head that outputs a scalar reward. {#c-ll-head}
For language models a reward model is often implemented by appending a linear layer that predicts one logit, or by removing the final decoding layers and replacing them with a linear layer; a win between two completions is r(x, y1) > r(x, y2). {#c-rb-impl}
The starting checkpoint is usually the model after instruction tuning, the supervised stage that comes before the reward model in a chat pipeline.
Reading the score at the last token lets the head see the whole prompt and response through causal attention; with right padding the last token is the last non-padding one, and `TinyRewardModel` in the PyTorch file reads it there.

The reward model need not be as large as the policy it judges.
InstructGPT uses only 6B reward models, which saves a lot of compute, because 175B reward model training could be unstable and thus less suitable for use as the value function during RL. {#c-ou-6b}
Bai et al., Llama 2 and most open recipes use a model of comparable size, and the Evidence section gives what is known about how accuracy scales.

## The method

**The Bradley-Terry loss.**
Training a reward model means training a classifier to predict the human preference probability between two answers, as modeled by a Bradley-Terry model, by optimizing the maximum likelihood loss. {#c-rb-bt}
With $\sigma(t) = 1/(1 + e^{-t})$ and $\Delta = r_\phi(x, y_w) - r_\phi(x, y_l)$, the model is

$$ P(y_w \succ y_l \mid x) = \frac{e^{r_\phi(x, y_w)}}{e^{r_\phi(x, y_w)} + e^{r_\phi(x, y_l)}} = \sigma(\Delta) , $$ {#eq-bt}

and the loss is the negative log-likelihood of the comparisons,

$$ \mathcal L(\phi) = -\mathbb E_{(x, y_w, y_l)}\big[\log \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)\big] . $$ {#eq-loss}

It is logistic regression on the difference of two scores, with no intercept, since the label is the order of the pair.

**Ranked responses.**
InstructGPT labelers rank between K = 4 and K = 9 responses, which produces K choose 2 comparisons for each prompt shown to a labeler. {#c-ou-rank}
Because comparisons within a labeling task are very correlated, shuffling them into one dataset made the reward model overfit in a single pass; InstructGPT instead trains on all comparisons from a prompt as a single batch element, which needs a single forward pass per completion and achieves much improved validation accuracy and log loss. {#c-ou-batch}
The loss for one prompt is

$$ \mathcal L_x(\phi) = -\binom{K}{2}^{-1} \sum_{i < j} \log \sigma\big(r_\phi(x, y_i) - r_\phi(x, y_j)\big) , $$ {#eq-ranked}

with the responses listed best first, and the batch loss averages $\mathcal L_x$ over prompts, so each prompt counts once however many responses it has.
InstructGPT's reward model batch holds 64 prompts with K = 4 to 9 completions each and ties dropped, so a single batch could contain up to 2,304 comparisons. {#c-ou-2304}
`ranked_loss` in the PyTorch file computes eq-ranked for a batch of prompts with different $K$ by masking, and its test compares it with a loop over pairs.

**Label noise.**
Christiano et al. assume a 10% chance that the human responds uniformly at random, because human raters have a constant probability of making an error, which does not decay to 0 as the reward difference becomes extreme. {#c-ch-noise}
With a noise rate $\epsilon$ the modelled probability becomes

$$ P_\epsilon(y_w \succ y_l \mid x) = \frac{\epsilon}{2} + (1 - \epsilon)\,\sigma(\Delta) , $$ {#eq-noise}

and the loss is $-\log P_\epsilon$.
The point is the behaviour on a pair the model confidently gets wrong: with plain eq-loss its loss grows linearly in $-\Delta$ and its gradient approaches 1, so a single mislabelled pair keeps pulling the model; with eq-noise the loss is bounded and the pull fades.

**Margins.**
Llama 2 adds a margin to the binary ranking loss, −log σ(r(x, y_c) − r(x, y_r) − m(r)), where the margin m(r) is a discrete function of the preference rating, large for pairs with distinct responses and smaller for similar ones. {#c-ll-margin}
The small margin variant of Llama 2 uses margins 1, 2/3, 1/3 and 0 and the large variant 3, 2, 1 and 0 for responses rated significantly better, better, slightly better and negligibly better or unsure. {#c-ll-margin-values}
The loss is $-\log \sigma(\Delta - m)$ for the margin $m$ of the pair's rating.

**Preference model pre-training.**
Human comparisons are expensive, so Askell et al. add a stage that learns from cheap, noisier preference signals first.
Askell et al. improve sample efficiency with a preference model pre-training (PMP) stage that first trains on large public datasets that encode human preference information, such as Stack Exchange, Reddit and Wikipedia edits, before finetuning on smaller datasets encoding more specific human preferences. {#c-as-pmp}
Given a context C with a binary label A or B, Askell et al. create a preference modeling pair C:A > C:B, where B denotes the incorrect label. {#c-as-format}
Appending a special end-of-context token to each sequence to unambiguously delineate the end of the passage sometimes improves performance. {#c-as-eoc}

**Normalization.**
Eq-bt depends only on differences of scores, so the training data fixes the reward up to a constant for each prompt, and each pipeline picks the constant.
Christiano et al. normalize the rewards produced by the reward predictor to have zero mean and constant standard deviation, since the position of the rewards is underdetermined by the learning problem. {#c-ch-norm}
At the end of training, Stiennon et al. normalize the reward model outputs so that the reference summaries from their dataset achieve a mean score of 0. {#c-st-norm}
Since the RM loss is invariant to shifts in reward, InstructGPT normalizes the reward model using a bias so that the labeler demonstrations achieve a mean score of 0 before doing RL. {#c-ou-norm}
Because RM scores are translation-invariant, Gao et al. recenter each RM so that the average reward of the initial policy is 0. {#c-ga-recenter}

**Several reward models.**
A single reward model may have to trade off goals that pull apart.
Because helpfulness and safety sometimes trade off, which can make it challenging for a single reward model to perform well on both, Llama 2 trains two separate reward models, a Helpfulness RM and a Safety RM. {#c-ll-two}
The Llama 2 Safety reward model is trained on all Meta Safety and Anthropic Harmless data mixed with helpfulness data in a 90/10 proportion; the 10% of helpfulness data is especially beneficial for accuracy on samples where both responses were deemed safe. {#c-ll-mix}
Llama 2's RL reward uses the Safety RM score when the prompt is tagged as potentially unsafe or that score is below 0.15, and the Helpfulness RM score otherwise; the threshold corresponds to a precision of 0.89 and a recall of 0.55 on the Meta Safety test set. {#c-ll-combine}
Llama 2 whitens the combined score after reversing the sigmoid with the logit function, to increase stability and balance properly with the KL penalty term. {#c-ll-whiten}
`llama2_combined_reward` and `llama2_rl_reward` in the NumPy file implement the rule.

**Ensembles.**
Several reward models trained on the same data disagree off the data, and their disagreement is information.
Christiano et al. fit an ensemble of predictors, each trained on |D| triples sampled from D with replacement, and define the reward estimate by independently normalizing each predictor and then averaging the results. {#c-ch-ens}
Coste et al. build an ensemble from proxy reward models trained with identical data and hyperparameters but different random seeds, which change the initialization of the scalar reward head and the data shuffling order. {#c-co-seed}
Mean optimization over an ensemble is not conservative: if at least one member overestimates the reward, even while the others estimate it accurately, the policy can exploit the faulty reward model. {#c-co-mean}
Worst-case optimization creates a conservative estimate by choosing the lowest reward from the ensemble at every step. {#c-co-wco}
Uncertainty-weighted optimization combines the average reward across the ensemble with the intra-ensemble variance weighted by a coefficient λ, penalizing responses on which the reward models disagree. {#c-co-uwo}
For member rewards $R_1, \dots, R_k$ the three combinations are

$$ R_{\text{mean}} = \bar R, \qquad R_{\text{WCO}} = \min_i R_i, \qquad R_{\text{UWO}} = \bar R - \lambda \cdot \frac{1}{k}\sum_{i=1}^{k} (R_i - \bar R)^2 . $$ {#eq-ens}

**Using the reward model.**
The trained model is used in three ways.
Reinforcement learning from human feedback trains a policy to maximize it, with a KL penalty that keeps the policy near its starting point.
Best-of-$n$ sampling draws $n$ responses and returns the one it scores highest, which needs no training at all.
And direct preference optimization skips the separate model: it trains the policy on the same comparisons with eq-loss written in terms of the policy's log-probability ratios, so the policy is its own reward model.

## Derivation

The derivation `reward-model-arithmetic` states and proves the facts below, and its check verifies each numerically.

**Bradley-Terry and its gradient.**
Under the Bradley-Terry model, P(y_w preferred to y_l | x) = exp(r_w) / (exp(r_w) + exp(r_l)) = σ(r_w − r_l), so the negative log-likelihood of one comparison is log(1 + exp(r_l − r_w)) = −log σ(r_w − r_l). {#c-rm-bt}
The gradient of −log σ(r_w − r_l) is −σ(r_l − r_w) with respect to r_w and +σ(r_l − r_w) with respect to r_l: the two sum to zero, and their size is the probability the model gives to the wrong order. {#c-rm-grad}
So the gradient of eq-loss with respect to the parameters is

$$ \nabla_\phi \mathcal L = -\mathbb E\big[\sigma(-\Delta)\,\big(\nabla_\phi r_\phi(x, y_w) - \nabla_\phi r_\phi(x, y_l)\big)\big] , $$ {#eq-grad}

which weights each pair by how wrong the model is about it, as in logistic regression: pairs already ordered with a large margin contribute almost nothing.

**What comparisons determine.**
Adding any function c(x) of the prompt alone to the reward leaves every pairwise preference probability and the loss unchanged, while multiplying the reward by a constant other than 1 changes them; so comparisons determine reward differences within a prompt but not the level of the reward. {#c-rm-shift}
That is why every pipeline above sets the constant itself, and why the scale of a reward model's outputs means something (it sets how confident the modelled preferences are) while its level does not.

**Ranked comparisons.**
For K ranked responses the per-prompt loss averages −log σ(r_i − r_j) over the K(K − 1)/2 pairs with i ranked above j; scoring the K responses once needs K forward passes, against K(K − 1) when every pair is a separate example, 9 against 72 for K = 9, and 64 prompts with K = 9 give 64 × 36 = 2,304 comparisons. {#c-rm-ranked}
The saving is not only compute: the $\binom{K}{2}$ pairs of one prompt share responses and are strongly correlated, and treating them as one example stops them from being seen many times in one epoch.

**Noise and margins.**
With a probability ε that the label is a uniformly random answer, the modelled preference probability is ε/2 + (1 − ε)σ(Δ), which stays between ε/2 and 1 − ε/2; with ε = 0.1 the loss of one comparison is at most −log 0.05 ≈ 3.00, and its derivative in Δ tends to 0 as Δ → −∞, so a confidently contradicted label stops pulling. {#c-rm-noise}
The margin loss −log σ(Δ − m) is the Bradley-Terry loss with the reward difference shifted by m: it equals log 2 at Δ = m and keeps pushing Δ up until Δ is well above m; at Δ = 1 with m = 1 it is log 2 ≈ 0.693, against −log σ(1) ≈ 0.313 without a margin. {#c-rm-margin}
A margin and the noise term move the loss in opposite directions: the margin asks for larger differences on pairs already ordered correctly, and the noise term asks for less effort on pairs ordered wrongly.

**The ceiling on accuracy.**
If comparisons follow the Bradley-Terry model, a pair with reward difference Δ is labelled in the order of the rewards with probability σ(|Δ|): 0.731 at |Δ| = 1, 0.881 at 2 and 0.953 at 3; no reward model can beat the true reward's accuracy on such labels, so label noise sets a ceiling on accuracy. {#c-rm-calib}
So a reward model's accuracy on held-out comparisons measures two things at once, how close it is to the labellers' preferences and how noisy those are, and accuracies are only comparable on the same data.
A model is calibrated when pairs it predicts at $\sigma(\Delta) = p$ come out in its order a fraction $p$ of the time; `calibration_table` in the NumPy file computes the table, the expected calibration error summarizes such a table in one number, and the test checks that multiplying a calibrated reward by 3 leaves its accuracy unchanged and makes it overconfident.

**Ensembles.**
For member rewards R_1, …, R_k the worst-case reward min R_i and the uncertainty-weighted reward mean − λ·var (population variance, λ ≥ 0) are both at most the mean; for (1.0, 1.2, 3.0) the mean is 1.733, the minimum 1.0, the variance 0.809, and the uncertainty-weighted reward with λ = 0.5 is 1.329. {#c-rm-ens}
Both conservative combinations agree with the mean where the members agree and fall below it where they disagree, which is where a single model is most likely to be exploited.

**The KL of best-of-n.**
If the n samples have distinct rewards almost surely, the best-of-n distribution has KL divergence log n − (n − 1)/n from the base policy: 0.636 nats for n = 4, 1.835 for n = 16 and 3.175 for n = 64. {#c-rm-bonkl}

$$ \mathrm{KL}(\pi_{\text{bon}} \,\|\, \pi) = \log n - \frac{n - 1}{n} . $$ {#eq-bonkl}

The KL grows only as $\log n$: 64 samples move the policy about 3.2 nats.
For a base policy whose responses repeat with positive probability eq-bonkl is an upper bound, and `bon_kl_discrete` computes the exact value.

**Hackability.**
Any reward model differs from the true reward somewhere, and Skalse et al. ask when the difference can be exploited.
Skalse et al. call a proxy reward unhackable if increasing the expected proxy return can never decrease the expected true return. {#c-sk-def}
For the set of all stochastic policies, two reward functions can only be unhackable if one of them is constant. {#c-sk-all}
In Skalse et al.'s cleaning-robot example, with true reward [1, 1, 1] for three rooms the proxy [1, 0, 0] is hackable while [1, 1, 0] is not, and the proxy [1, 1, 1] is unhackable for the true reward [1, 1.5, 2] but hackable for [1, 1.5, 3]. {#c-sk-robot}
Enumerating the eight deterministic cleaning policies confirms the four cases; for proxy [1, 1, 1] and true reward [1, 1.5, 3], cleaning attic and bedroom has proxy return 2 against 1 for the kitchen alone, but true return 2.5 against 3. {#c-rm-robot}
On a small set of policies some proxies are safe to optimize; over all policies, essentially none are.

## Worked example

Take one prompt and three responses ranked $y_1 \succ y_2 \succ y_3$, which the reward model scores $2.0$, $0.5$ and $0.0$.
The three pairs give

| pair | $\Delta$ | $\sigma(\Delta)$ | $-\log \sigma(\Delta)$ | $\sigma(-\Delta)$ |
|---|---|---|---|---|
| $y_1 \succ y_2$ | 1.5 | 0.817574 | 0.201413 | 0.182426 |
| $y_1 \succ y_3$ | 2.0 | 0.880797 | 0.126928 | 0.119203 |
| $y_2 \succ y_3$ | 0.5 | 0.622459 | 0.474077 | 0.377541 |

For three ranked responses with rewards 2.0, 0.5 and 0.0, the pair losses are 0.201413, 0.126928 and 0.474077 and the ranked loss is their mean, 0.267473; its gradients with respect to the three rewards are −0.100543, −0.065038 and 0.165581. {#c-rm-we}
The gradient pushes $r_1$ up and $r_3$ down, and $r_2$ up a little: it is the lower member of one pair and the upper member of another, and the pair $y_2 \succ y_3$, whose margin is smallest, pulls hardest.
The three gradients sum to zero, as eq-grad says they must, so a gradient step does not change the average score.
`test_ranked_loss_worked_example` in the NumPy tests and `test_ranked_loss_matches_loop` in the PyTorch tests reproduce the numbers.

## Hyperparameters

The InstructGPT reward model was trained for a single epoch at a learning rate of 9e-6 with a cosine schedule dropping to 10% of its initial value and a batch size of 64 prompts; changes of up to 50% in the learning rate gave similar performance, but multiple epochs quickly overfit. {#c-ou-hparams}
The final InstructGPT reward model was initialized from a 6B GPT-3 model fine-tuned on public NLP datasets, mostly for historical reasons, with similar results when initializing from the GPT-3 or SFT models. {#c-ou-init}
Llama 2 trains its reward models for one epoch, since training longer can lead to over-fitting, with a maximum learning rate of 5 × 10−6 for the 70B model and 1 × 10−5 for the rest. {#c-ll-hparams}
Bai et al. apply preference model pretraining before finetuning on their human feedback data and typically train preference models for a single epoch. {#c-hh-epoch}
Christiano et al. hold out 1/e of the data as a validation set for each predictor and adjust an ℓ2 regularization coefficient to keep the validation loss between 1.1 and 1.5 times the training loss. {#c-ch-reg}
The common thread is one epoch: comparisons are few and correlated, and a second pass memorizes them.
The margin, the noise rate and the ensemble combination are listed in the frontmatter; each changes the loss only on some pairs and is worth an ablation on held-out comparisons before use.

## Cost

Training costs one forward and backward pass per response, and a ranked batch element with $K$ responses costs $K$ passes rather than $K(K - 1)$.
The datasets are small next to pretraining or even instruction tuning, so a reward model trains in a small fraction of the policy's compute.
The cost that matters is at use: during RL the reward model scores every sampled response, so its forward pass is part of every policy update, and a reward model as large as the policy roughly doubles the cost of generating and scoring a batch, which is one reason InstructGPT kept it at 6B.
Best-of-$n$ costs $n$ generations and $n$ reward-model passes per prompt.
An ensemble of $k$ members multiplies the scoring cost by $k$.

## Evidence

**Scaling and transfer.**
Across 7 reward models from 160M to 13B parameters trained on 8k to 64k comparisons, doubling the training data increases validation accuracy by about 1.1% and doubling the model size by about 1.8%. {#c-st-scale}
Preference model accuracy follows roughly log-linear trends in model and dataset size. {#c-hh-scale}
More data and a larger model generally improve Llama 2 reward model accuracy, and the models appear not to have saturated from learning on the training data. {#c-ll-scale}
For all reward model sizes, with fewer than around 2,000 comparisons there is very little improvement over near-chance loss. {#c-ga-data}
Stiennon et al.'s reward models generalize to CNN/DM summaries, agreeing with labeler preferences 62.4% and 66.5% of the time for the 1.3B and 6.7B models; the 6.7B model nearly matches the inter-labeler agreement of 66.9%. {#c-st-cnn}
The reward models prefer summaries improved by minimal human edits almost as often (79.4% for 1.3B and 82.8% for 6.7B) as a separate set of human evaluators (84.1%), and select the original over a summary with the participants' roles reversed 92.9% and 97.2% of the time. {#c-st-edits}
ROUGE agrees with labelers about 57% of the time on samples from supervised baselines but about 50% on samples from human-feedback models, and log probability drops to at most 50%, while the reward models stay above chance at 62%. {#c-st-rouge}
InstructGPT reward models trained with 5-fold cross-validation over groups of labelers predict the preferences of the held-out group with 69.6 ± 0.9% accuracy, a small decrease from 72.4 ± 0.4% on labelers in their training set. {#c-ou-heldout}

**Rankings, pre-training and calibration.**
Preference modeling improves on and scales more favorably than imitation learning when preferences are part of a ranked hierarchy or continuum, rather than associated with a binary choice. {#c-as-rank}
A PMP stage between language model pretraining and finetuning on small final datasets significantly improves sample efficiency, even when the PMP data are quite different from the final dataset, such as finetuning from Stack Exchange to summarization. {#c-as-pmp-gain}
It is better for the PMP stage to focus on binary discrimination than on ranked preferences, possibly because hierarchies of preferences are difficult to quickly unlearn during finetuning. {#c-as-binary}
Preference models trained only on helpfulness data are very well calibrated against the prediction 1/(1 + e^−Δ) for a score difference Δ, while those trained on a mixture of helpful and harmless data are slightly under-confident. {#c-hh-calib}
When only comparisons in which both samples have a preference model score above a threshold are kept, preference model accuracy drops as the threshold rises. {#c-hh-top}
That drop matters because RL pushes the policy toward exactly the responses the reward model scores highest.

**Accuracy by how clear the preference is.**
The Llama 2 Helpfulness RM is 80.7% accurate on Meta Helpfulness pairs rated significantly better and 54.7% on pairs rated negligibly better or unsure, 63.2% on average; the Safety RM on Meta Safety is 94.3% and 55.3%, 64.5% on average. {#c-ll-table8}
In the margin ablation for the Helpfulness RM, average accuracy is 62.5 without a margin, 63.0 with the small margin and 62.9 with the large one, and on significantly better pairs 79.1, 80.4 and 80.7. {#c-ll-margin-ablation}
Both tables fit the Bradley-Terry ceiling: pairs that labellers mark as nearly equal are close to a coin toss for any model.

**Benchmarks.**
Validation sets from earlier RLHF work, such as Anthropic's Helpful and Harmless data and OpenAI's Learning to Summarize, have ceilings on accuracy between 60 and 70% due to inter-annotator disagreement. {#c-rb-ceiling}
RewardBench was built to separate reward models more clearly, with pairs whose preferred response is verified.
RewardBench has five sections: Chat, Chat Hard (trick questions and subtly different instruction responses), Safety, Reasoning (code and math) and Prior Sets drawn from existing preference test sets. {#c-rb-sections}
RewardBench counts a prompt as a win when the reward model scores the verified chosen completion above the verified rejected one, and reports accuracy as the percentage of wins in each subset. {#c-rb-score}
The final RewardBench score is the weighted average of the section scores, with Prior Sets at 0.5 weight. {#c-rb-weight}
`section_score` in the NumPy file implements the weighting.
To score a DPO model as a reward model, RewardBench compares log π(y|x)/π_ref(y|x) between the two completions, which requires the reference model's probabilities. {#c-rb-dpo-reward}
DPO models, while more plentiful due to the method's simplicity, fail to generalize to popular preference data test sets and present a higher variance in performance than classifier reward models. {#c-rb-dpo}
Reasoning accuracy ranges from 35%, well below random, to 97%; much of the data is code in which only one or two tokens differ between the chosen and rejected samples. {#c-rb-reasoning}
Few reward models give Gaussian scores across RewardBench, fewer are centred around 0, and none of those tested are centred Gaussians. {#c-rb-shape}

**Ensembles.**
Using an ensemble almost always helps over a single reward model; for best-of-n, worst-case and uncertainty-weighted optimization show no overoptimization, while mean optimization does overoptimize with 25% label noise. {#c-co-results}
For best-of-n, the ensemble methods improve final performance by up to about 30% over individual reward models without label noise and up to about 75% with 25% label noise. {#c-co-gain}
For PPO, worst-case and uncertainty-weighted optimization reduce overoptimization, but mitigating it completely requires combining them with a small KL penalty. {#c-co-ppo}
Reward ensembles whose members vary by pretraining seed generalize better than ensembles differing only by fine-tuning seed, and both outperform individual reward models. {#c-ei-pretrain}

## Limitations and failure modes

**Overoptimization.**
A reward model is accurate on the distribution it was trained on; optimizing a policy against it moves the policy off that distribution.
Gao et al. find that the gold reward follows R_bon(d) = d(α_bon − β_bon d) for best-of-n sampling and R_RL(d) = d(α_RL − β_RL log d) for reinforcement learning, where d is the square root of the KL divergence from the initial policy. {#c-ga-forms}
The α and β coefficients vary smoothly with the number of proxy reward model parameters, following approximate logarithmic trends, which allows prediction of the attained gold reward. {#c-ga-coef}
Gao et al. compute the KL distances for best-of-n (BoN) analytically as KL_bon = log n − (n − 1)/n. {#c-ga-bonkl}
Both forms rise and then fall: past some KL, the proxy reward keeps increasing while the gold reward drops.
Gao et al. find weak evidence that two reward models of equal validation loss achieve the same robustness against optimization, whatever their combination of model size and data size. {#c-ga-valloss}
When RLHF policies are trained against one of two preference models fitted to halves of the data, training is robust up to about 150k samples, after which the training preference model assigns a higher mean reward than the held-out one. {#c-hh-robust}
The reward-hacking page covers the general phenomenon.

**Spurious features.**
Stiennon et al.'s 6.7B reward model is biased towards longer summaries, preferring improving edits that make a summary shorter only 62.6% of the time, against 76.4% for humans. {#c-st-length}
Length is the best-documented such feature, but any feature correlated with preference in the training data, such as format, confidence or politeness, can be learned in place of the preference.

**Underspecification and shared errors.**
Reward models are underspecified: models that perform similarly in distribution can give very different rewards when used in alignment, because of distribution shift, and aligning to one does not improve the reward measured by another trained on the same data. {#c-ei-under}
Even ensembles of differently pretrained reward models do not eliminate reward hacking, because all members of the ensemble exhibit similar error patterns. {#c-ei-shared}
Policy optimization amplifies shared errors: summarization models produce outputs that are too short when tuned for factuality and too verbose when tuned for summarization quality, and assistant models overuse formulaic answer formats when tuned for helpfulness. {#c-ei-examples}
Ensembles help against errors that differ between members, not against errors that all members learned from the same data.

**Distribution shift during training.**
Training the reward predictor offline performs poorly: because the occupancy distribution is nonstationary, the predictor captures only part of the true reward, and on Pong the agent learned to avoid losing points without scoring, giving extremely long volleys. {#c-ch-offline}
In language-model RLHF the same problem appears as the need to collect fresh comparisons on the current policy's outputs, as Llama 2 and Bai et al. did in iterated rounds.

**Score distributions.**
A margin in the ranking loss pushes the reward model towards more extreme scores, and a larger margin makes this distribution shift more significant, which suggests investing in reward calibration because PPO can be sensitive to changes in the reward distribution. {#c-ll-shift}
The scale of a reward model's outputs reaches the policy through the KL coefficient, so normalization and margins are not cosmetic.

**What a learned reward can be.**
Skalse et al. conclude that reward functions learned by reward modeling are perhaps best viewed as auxiliaries to policy learning, rather than specifications that should be optimized. {#c-sk-aux}
Constitutional AI replaces part of the human labelling with model-written labels, which changes where the labels come from but not this limit.

## Variants

- **Pairwise Bradley-Terry reward models** (Christiano et al., Stiennon et al., Llama 2): eq-loss on pairs, optionally with a margin or a noise term.
- **Ranked reward models** (InstructGPT): eq-ranked on $K$ responses per prompt, one batch element per prompt.
- **Preference model pre-training** (Askell et al., Bai et al.): an extra stage on public preference-like data before the human comparisons.
- **Several specialized reward models** (Llama 2): separate helpfulness and safety models combined by a rule.
- **Ensembles** (Christiano et al., Coste et al., Eisenstein et al.): several members, combined by mean, minimum or mean minus variance.
- **Implicit reward models** (direct preference optimization): the policy's log-probability ratio to a reference model plays the role of the reward, and RewardBench evaluates such models alongside classifiers.
- **AI-labelled reward models** (Constitutional AI): the comparisons come from a model following written principles rather than from people.

## When to use it

Use a reward model when preferences are easier to collect than demonstrations and you need to score many new outputs: to run RL, to rerank samples with best-of-$n$, or to filter generated data.
Initialize it from the model being tuned or one of similar size, train it for one epoch, and check held-out accuracy against the agreement between labellers rather than against 100%.
Normalize its outputs before RL, keep a KL penalty or a limit on best-of-$n$, and watch a held-out reward model or human evaluation for the point where the proxy keeps rising and quality stops.
Use an ensemble when overoptimization is the main risk and the extra scoring cost is affordable.
If the only goal is to fit a policy to fixed comparisons, direct preference optimization avoids training and serving a separate model, at the cost of the reusable scorer.

## Pitfalls

> [!pitfall]
> **Reading the score at a padding token.** With right padding the last position of a row is often padding, and a head read there scores the padding rather than the response. Read at the last non-padding token. `test_reward_ignores_right_padding` checks that the reward changes when it is read at the wrong position.

> [!pitfall]
> **Shuffling the pairs of a ranking into separate examples.** The $\binom{K}{2}$ pairs of one prompt share responses, and spreading them through an epoch overfits and costs $K(K-1)$ forward passes instead of $K$. Keep a prompt's responses together and divide its loss by its number of pairs, as `ranked_loss` does; `test_ranked_loss_matches_loop` and `test_ranked_gradient_and_counts` pin the weighting.

> [!pitfall]
> **Writing the loss as log(sigmoid(d)).** For a confidently wrong pair $\sigma(\Delta)$ underflows to 0 and the loss becomes infinite. Use log-sigmoid, and for the noisy loss a log-sum-exp form; `test_bradley_terry_identity_and_stability` and `test_noisy_loss_is_stable_and_stops_pulling` check large differences.

> [!pitfall]
> **Comparing raw scores across reward models or prompts.** Comparisons fix scores only up to a constant per prompt, so the level of a score means nothing and two models' scores are on different scales. Normalize against a reference set before using the score as an RL reward or a threshold; `test_shift_invariance_and_normalization` checks that the loss is unchanged by a shift.

> [!pitfall]
> **Reading accuracy as quality without the ceiling.** With noisy labels even the true reward agrees with the labels only a fraction $\sigma(|\Delta|)$ of the time, so 65% may be near the ceiling on one dataset and poor on another. `test_accuracy_ceiling_and_calibration` simulates the ceiling.

## History

Christiano, Leike, Brown, Martic, Legg and Amodei fitted a reward predictor to human comparisons of trajectory segments in 2017, with label noise, ensembles and normalization already in place.
Stiennon and colleagues at OpenAI put a scalar head on a language model and trained it on comparisons of summaries in 2020.
Ouyang and colleagues used ranked comparisons for InstructGPT in 2022.
Askell, Bai and colleagues at Anthropic studied preference models, their pre-training and their calibration in 2021 and 2022, and Touvron and colleagues at Meta trained separate helpfulness and safety reward models for Llama 2 in 2023.
Gao, Schulman and Hilton measured overoptimization in 2022, Skalse and colleagues defined hackability in the same year, Coste and colleagues and Eisenstein and colleagues studied ensembles in 2023, and Lambert and colleagues released RewardBench in 2024.
