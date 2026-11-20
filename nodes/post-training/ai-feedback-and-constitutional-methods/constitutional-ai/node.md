---
tag: 20KVDN
kind: method
slug: constitutional-ai
name: Constitutional AI
aliases: [CAI, RLAIF, RL from AI feedback, reinforcement learning from AI feedback, SL-CAI, RL-CAI, critique and revision]
area: post-training/ai-feedback-and-constitutional-methods
level: core
summary: >-
  Training a harmless assistant from a short list of written principles: the model critiques and revises its own answers, and AI preference labels replace human harmlessness labels.
purpose: training an assistant to be harmless without being evasive, with a written list of principles in place of human labels of harmful outputs
uses: [reward-function@al, ppo@al, knowledge-distillation@al, cross-entropy@m2, logistic-regression@m2]
related_to: [ensemble-learning@al, reinforce@al]
hyperparameters:
  - {name: principles, meaning: "the number of written principles; one is sampled at random for each critique-revision step and for each AI preference label", typical: "16 in Bai et al. (2022)"}
  - {name: revisions per prompt, meaning: "how many critique-revision steps are applied in a row to one response", typical: "4"}
  - {name: label type, meaning: "whether the preference model is trained on the feedback model's normalized probabilities (soft), on 0 or 1 (hard), or on probabilities clamped to an interval", typical: "soft without chain of thought; clamped to 40-60% with it"}
  - {name: SL fine-tuning, meaning: "the supervised step on the revisions and the helpfulness samples", typical: "one epoch, a constant learning rate of 0.5 times the pretraining rate, batches of 1024 sequences"}
  - {name: sampling temperature, meaning: "the temperature of the responses, critiques and revisions", typical: "T = 1"}
cost:
  train: "a helpful RLHF model to write responses, critiques and revisions (4 per prompt for 182,831 red-team prompts in Bai et al.), one scoring pass of a feedback model per comparison (a generation first with chain of thought), then the whole RLHF pipeline of preference model and RL; Lee et al. estimate AI labels at over 10x cheaper than human ones"
  inference: "none beyond the policy; the feedback model and the preference model are used only in training"
history:
  year: 2022
  people: [Yuntao Bai, Jared Kaplan]
  note: "Bai et al., 51 authors at Anthropic, posted Constitutional AI: Harmlessness from AI Feedback as arXiv 2212.08073 on 15 Dec 2022, with RL from AI feedback (RLAIF) as its second stage. Lee et al. of Google DeepMind and Google posted RLAIF vs. RLHF as arXiv 2309.00267 in September 2023 and published it at ICML 2024."
prereqs: [ppo@al, reward-function@al, cross-entropy@m2, logistic-regression@m2]
same_as:
  wikidata: null
  wikipedia: null
  arxiv: ["2212.08073", "2309.00267"]
  al: null
sources:
  - {paper: "Bai et al. (2022b)", title: "Constitutional AI: Harmlessness from AI Feedback", locator: "the abstract, Sections 1 to 6, pp. 9 and 11, the title page", note: "the method: critique and revision, AI preference labels, chain of thought, and the results"}
  - {paper: "Bai et al. (2022a)", title: "Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback", locator: "the abstract, Sections 2.2 and 4.4", note: "the human-feedback baseline and its evasiveness"}
  - {paper: "Lee et al. (2024)", title: "RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback", locator: "the abstract, Sections 1 to 5 and 7, pp. 4, 6, 8 and 9, the title page", note: "a direct comparison of AI and human feedback, position bias, labeler size and direct RLAIF"}
  - {paper: "Ouyang et al. (2022)", title: "Training language models to follow instructions with human feedback", locator: "Section 1", note: "whose preferences a human-feedback model learns"}
  - {paper: "Ganguli et al. (2022)", title: "Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned", locator: "arXiv 2209.07858, pp. 1 and 2", note: "the released red-team attacks, the scaling of red-team difficulty by model type, and harmlessness through evasion"}
---

Constitutional AI (CAI) trains an assistant to be harmless with a short list of principles written in natural language, the constitution, in place of human labels of which outputs are harmful.
It has two stages.
In the supervised stage the model critiques its own answers against a principle and rewrites them, and is fine-tuned on the rewrites.
In the reinforcement learning stage a language model, not a person, judges which of two answers better follows a principle, and these AI preferences train the preference model of an otherwise ordinary RLHF pipeline, which is why this stage is called reinforcement learning from AI feedback (RLAIF).
Bai et al. (2022b) introduced it to get an assistant that is harmless without being evasive, one that explains why it will not help instead of refusing.

## In plain words

RLHF needs people to compare many pairs of answers, and for harmlessness that means reading harmful text and judging which is worse.
Constitutional AI gives this work to the model, guided by a short list of written principles.
First the model answers a provocative prompt, points out what is harmful in its answer, and rewrites it; the rewrites become fine-tuning data.
Then the fine-tuned model writes two answers to each such prompt, and a language model reads both with a principle and says which is better.
Its probability for each answer is the label, and from here the training is RLHF: a preference model learns the labels, and reinforcement learning maximizes its score.
People still write the principles, a few examples, and the helpfulness labels.
Open questions remain: whose values the principles encode, and how far a model's judgment can stand in for a person's.

## Setting

Bai et al. (2022b) train a harmless AI assistant through self-improvement, without any human labels identifying harmful outputs; the only human oversight is a list of rules or principles, hence the name Constitutional AI. {#c-cai-definition}
They chose the word "constitutional" because a short list of principles is enough to train less harmful systems, and also to stress that a general AI system cannot avoid being governed by some set of principles, even hidden or implicit ones. {#c-cai-name}
Their motivations were to study how AI systems can help supervise other AIs, to remove evasive responses and reduce the tension between helpfulness and harmlessness, to make the principles governing AI behavior more transparent, and to shorten iteration by needing no new human labels when the objective changes. {#c-cai-motivations}
They note that RLHF typically uses tens of thousands of human preference labels, which shed little light on the training objectives even when they are shared, since no one can summarize their collective effect. {#c-rlhf-labels}

The baseline is the human-feedback assistant of Bai et al. (2022a).
They apply preference modeling and RLHF to make language models helpful and harmless assistants, and find that this improves performance on almost all NLP evaluations and is compatible with training for skills such as Python coding and summarization. {#c-hh-baseline}
Crowdworkers chose the more helpful response in open-ended conversations for one dataset, and for the red-teaming dataset tried to elicit harmful responses and chose the more harmful one, a difference that the authors believe made it hard to train models both helpful and harmless. {#c-hh-datasets}
That assistant often refused controversial questions and could get stuck giving evasive responses for the rest of a conversation, which Bai et al. (2022b) trace to crowdworkers rewarding evasiveness as a response to harmful inputs. {#c-hh-evasive}
Bai et al. (2022a) themselves found RLHF policies giving the same exaggerated responses to all remotely sensitive questions, which they attribute to over-optimizing harmlessness and under-optimizing helpfulness. {#c-hh-tension}
Bai et al. (2022b) collected their human-written red-team prompts as discussed and shared in the same group's red-teaming study, {#c-cai-prompts-ganguli}
whose public release holds 38,961 attacks. {#c-ganguli-dataset}
Across models from 2.7B to 52B parameters, RLHF models grew harder to red team with scale, while plain, prompted and rejection-sampling models stayed flat. {#c-ganguli-scale}
Rejection sampling was the hardest to red team at every size, but the authors note that its answers were often harmless only because they were evasive, which is the failure constitutional AI sets out to fix. {#c-ganguli-rs-evasive}

Write $x$ for a prompt, $y$ for a response, $\pi$ for the policy and $c_k$ for the $k$-th principle.
The constitution of Bai et al. (2022b) has 16 principles. {#c-sixteen}
The pipeline begins from a helpful-only RLHF model, and ends with an RLHF run whose preference model is trained partly on AI labels.

## The method

**Stage 1: critique, revision and supervised learning (SL-CAI).**
The helpful RLHF model is shown a red-team prompt designed to elicit harmful behavior and samples a response; a critique request is appended and the model samples a critique of its own response, then a revision request is appended and it samples a revision. {#c-sl-loop}
One critique request asks the model to "identify specific ways in which the assistant's last response is harmful, unethical, racist, sexist, toxic, dangerous, or illegal", and the matching revision request asks it to "rewrite the assistant response to remove any and all harmful, unethical, racist, sexist, toxic, dangerous, or illegal content". {#c-critique-request}
The prompt and the revision have the form of the original prompt and response, so the pipeline can be applied again and again; a critique request and a revision request together form a principle, and one principle is sampled at random at each revision step. {#c-repeat}
The model sometimes confused its point of view, writing a critique where it should write a revision, which Bai et al. fixed with few-shot examples of critiques and revisions. {#c-pov}
A pretrained model is then fine-tuned on the revisions from all revision steps, together with responses of the helpful RLHF model to helpfulness prompts, so that it keeps its helpfulness; the result is SL-CAI. {#c-sl-finetune}
The purpose of this stage is to change the distribution of responses easily, so that the RL stage needs less exploration and less training. {#c-sl-purpose}

**Stage 2: AI feedback, a preference model, and RL (RL-CAI).**
SL-CAI writes a pair of responses to a prompt, and a feedback model, typically a pretrained language model, is shown the conversation, a principle and the two responses as a multiple-choice question. {#c-feedback-prompt}
The prompt reads

```
Consider the following conversation between a human and an assistant:
[HUMAN/ASSISTANT CONVERSATION]
[PRINCIPLE FOR MULTIPLE CHOICE EVALUATION]
Options:
(A) [RESPONSE A]
(B) [RESPONSE B]
The answer is:
```

and the log-probabilities of the answers (A) and (B), normalized, are the targets of a preference-model comparison, which Bai et al. expect to be fairly well calibrated because they are multiple-choice answers. {#c-normalized-targets}
With $\ell_A$ and $\ell_B$ the two log-probabilities, the label is

$$ p_A = \frac{e^{\ell_A}}{e^{\ell_A} + e^{\ell_B}} = \sigma(\ell_A - \ell_B) , $$ {#eq-label}

where $\sigma$ is the logistic function.
A principle is sampled at random for each label, and Bai et al. found that ensembling over principles gave a notably more robust preference model than one principle for all labels. {#c-principle-ensemble}
One of the 16 reads: "Which of these assistant responses is less harmful? Choose the response that a wise, ethical, polite and friendly person would more likely say." {#c-principle-example}

The labels for harmlessness come from the feedback model and those for helpfulness from people, and from the labels on the pipeline of preference-model training and RL is exactly that of RLHF. {#c-hybrid}
Bai et al. call the result a hybrid human/AI preference model: as RLHF distills human preferences into a preference model, this stage distills a language model's interpretations of the principles into one. {#c-distill}
SL-CAI both writes the response pairs and is the starting point of RL, which Bai et al. expect to help because the policy's responses then resemble the preference model's training data early in RL. {#c-same-model}
A preference model $r_\phi$ trained on soft labels minimizes the cross-entropy

$$ \mathcal{L}(\phi) = -\mathbb{E}\Big[p_A \log \sigma\big(r_\phi(x, y_A) - r_\phi(x, y_B)\big) + (1 - p_A) \log \sigma\big(r_\phi(x, y_B) - r_\phi(x, y_A)\big)\Big] , $$ {#eq-soft-loss}

which is the Bradley-Terry loss of RLHF with the label $p_A$ in place of 0 or 1; the derivation shows that it fits the feedback model's log-odds.
Lee et al. (2024) train their reward model this way, with a cross-entropy loss on the softmax of its scores, and note that training a reward model on AI labels can be viewed as model distillation. {#c-lee-rm}
The policy is then trained by RL against the preference model, as in RLHF.

**Chain-of-thought feedback.**
For chain-of-thought labels, Bai et al. use the helpful RLHF model as the feedback model, in a conversational format, with the prompt "Let's think step-by-step" and hand-written few-shot examples. {#c-cot}
The chain of thought usually states which option is better, so the probabilities are close to 0 or 1 and poorly calibrated; clamping them to 40-60% gave better and more robust behavior, while without the clamp the RL-CAI models learned more extreme responses. {#c-clamp}

**Labels from a general LLM (Lee et al.).**
Lee et al. ask an off-the-shelf LLM which of two responses is better, with a preamble, optional few-shot exemplars, the sample to annotate and an ending such as "Preferred Response="; the softmax of the log-probabilities of the tokens "1" and "2" is the preference distribution. {#c-lee-labeling}
The order in which the candidates are shown can bias the LLM, especially smaller ones, so they make two inferences per pair with the order reversed and average them. {#c-lee-position}
Their chain of thought takes two steps: the LLM first writes a rationale, which is then appended to the prompt before the scoring step. {#c-lee-cot}
They run RL with REINFORCE with a baseline instead of PPO, with the policy and the value model initialized from the SFT model. {#c-lee-reinforce}

## Derivation

**The label is a logistic function of a log-odds.**
Dividing eq-label by $e^{\ell_A}$ gives $p_A = \sigma(\ell_A - \ell_B)$, and adding the same constant to both log-probabilities leaves $p_A$ unchanged, so the probability that the feedback model puts on other tokens does not matter. {#c-label-logistic}

**The soft-label loss fits the log-odds.**
Write $\Delta = r_\phi(x, y_A) - r_\phi(x, y_B)$ and $p = p_A$.
The loss of one pair in eq-soft-loss is $L(\Delta) = -p \log \sigma(\Delta) - (1 - p) \log \sigma(-\Delta)$, its derivative is $\sigma(\Delta) - p$, and its second derivative is $\sigma(\Delta)\sigma(-\Delta) > 0$. {#c-soft-gradient}
For $0 < p < 1$ the loss has the unique minimizer $\Delta^* = \log\big(p / (1 - p)\big)$, the label's log-odds, where it equals the binary entropy of $p$; so the preference model learns to reproduce the feedback model's log-odds $\ell_A - \ell_B$. {#c-soft-minimizer}
For a hard label, $p = 1$, the loss decreases for every $\Delta$ and has no minimizer, so the score difference grows without bound. {#c-hard-no-minimizer}

**Clamping bounds the target.**
Clamping $p$ to $[a, 1 - a]$ with $0 < a < 1/2$ bounds the target score difference by $|\Delta^*| \le \log\big((1 - a)/a\big)$, which is $\log 1.5 \approx 0.405$ for 40-60% and $\log 4 \approx 1.386$ for 20-80%. {#c-clamp-bound}

**A random principle per label trains on the average label.**
The loss is linear in $p$, so if the principle $c_k$ is drawn at random for each label and gives $p_k$, the expected loss is the loss of the averaged label $\bar p = \mathbb{E}_k[p_k]$, with minimizer $\log\big(\bar p / (1 - \bar p)\big)$. {#c-ensemble-average}

**Averaging the two orders removes a position bias.**
Suppose the feedback model's log-odds for the response in the first position are $d + b$, where $d$ is its preference between the contents and $b$ a bias towards the first position.
Then the average over the two orders is

$$ \bar p = \tfrac12\big(\sigma(d + b) + \sigma(d - b)\big) , $$ {#eq-swap}

which is the same for $b$ and $-b$ and equals $1/2$ at $d = 0$ for every $b$, so a pure position bias gives no preference. {#c-swap-symmetric}
For $d > 0$, $\bar p$ falls from $\sigma(d)$ towards $1/2$ as $|b|$ grows and stays above $1/2$, so the average never reverses the preference and only weakens it. {#c-swap-shrinks}
The derivation `ai-preference-label` gives the steps, and its check verifies them numerically.

## Worked example

A feedback model gives the answer (A) the probability 0.45 and (B) the probability 0.15, and the rest to other tokens.

| quantity | value |
|---|---|
| $p_A$ by eq-label | $0.45 / 0.60 = 0.75$ |
| target $\Delta^* = \log(p / (1 - p))$ | $\ln 3 \approx 1.0986$ |
| soft loss at $\Delta = 0$ | $\ln 2 \approx 0.6931$ |
| its derivative at $\Delta = 0$ | $\sigma(0) - 0.75 = -0.25$ |
| soft loss at $\Delta = \ln 3$ | $H(0.75) \approx 0.5623$ |
| $p$ clamped to 40-60% | $0.6$, target $\ln 1.5 \approx 0.4055$ |

With two principles that give $p_1 = 0.9$ and $p_2 = 0.6$, the averaged label is $0.75$, and the expected loss has the same minimizer $\ln 3$.

For the position bias, suppose the order (A, B) gives $p_A = 0.75$ and the order (B, A) gives $p_A = 0.55$.
In the model of eq-swap, $d + b = \ln 3 \approx 1.0986$ and $d - b = \operatorname{logit}(0.55) \approx 0.2007$, so $d \approx 0.6496$ and $b \approx 0.4490$.
The average is $\bar p = 0.65$, which lies between $1/2$ and $\sigma(d) \approx 0.6569$.
The check `ai-preference-label.check.py` verifies these numbers.

## Hyperparameters

**The constitution.**
Bai et al. write 16 principles, sampled at random for each revision step and for each label, as above.
The number of principles had no significant effect on the harmlessness score of the preference model, though Bai et al. expect more principles to give more diverse behavior, which they did not study quantitatively. {#c-num-principles}

**The supervised stage.**
Bai et al. take 42,496 human-written red-team prompts and 140,335 more generated by few-shot prompting a pretrained model, 182,831 in all, and sample 4 critique-revision pairs for each from a helpful RLHF model; for helpfulness they take 135,296 human-written prompts with 2 responses each, always sampling at $T = 1$. {#c-sl-data}
They fine-tune for one epoch with a constant learning rate of 0.5 relative to the pretraining rate and batches of 1024 sequences. {#c-sl-train}

**The RL stage.**
The RL runs use the same hyperparameters as Bai et al. (2022a); the preference model is trained on 135,296 human helpfulness comparisons and 182,831 AI harmlessness comparisons, one for each SL-CAI prompt, and RL adds 491,142 generated red-team prompts and 474,300 generated helpfulness prompts. {#c-rl-data}

**The labels.**
Without chain of thought, soft labels gave much better results than hard labels of 0 and 1, which Bai et al. attribute to the soft labels being fairly well calibrated; with chain of thought, clamping to 20-80% slightly improved results and 40-60% improved them further, and they used 40-60% for the main results. {#c-soft-vs-hard}

## Cost

The supervised stage samples a response, critiques and revisions for every red-team prompt, four critique-revision pairs per prompt in Bai et al., and fine-tunes a pretrained model on the result.
An AI label costs one scoring pass of the feedback model on the prompt, or a generation and a scoring pass with chain of thought; Lee et al. double this to correct for position bias.
After that the cost is that of RLHF: a preference model, a policy, and RL with sampling.
Lee et al. estimate the cost of LLM labeling at over 10 times less than human annotation. {#c-lee-cost}
Since canonical RLAIF calls the AI labeler only once, to generate the preferences, and not during RL, they note that a larger labeler is not necessarily prohibitively expensive. {#c-lee-labeler-once}
Direct RLAIF, below, calls the LLM for every response during RL, but needs no labeling or reward-model training.
At inference only the policy is used.

## Evidence

**Can a model judge harmlessness?**
Models already score above 90% on an older set of 221 pairwise questions about helpfulness, honesty and harmlessness, so Bai et al. added 217 harder pairs that mostly probe subtle harm; reasoning step by step helps the larger models a lot, and voting over five such reasoning samples adds a little more. {#c-hhh-eval}
Extrapolating accuracy on all 438 pairs, Bai et al. expect models beyond 52B parameters to match preference models fit to human labels. {#c-hhh-trend}
Bai et al. take this as a sign that big language models may be close to crowdworkers at spotting and judging harm. {#c-approaching}

**The supervised stage.**
Each further round of critique and revision raises the harmlessness preference model's score, with the caveat that this score is less trustworthy at its top end. {#c-revisions-harmless}
The scores of a preference model trained only on helpfulness fall with the number of revisions. {#c-revisions-helpful}
Writing a critique before revising helped small models and did not matter for large ones, and when the authors read 52B critiques, many were wrong or exaggerated even if some were sound. {#c-critiques}
Crowdworkers rank SL-CAI below both RLHF models on helpfulness, and on harmlessness between them: safer than helpful-only RLHF, less safe than HH RLHF. {#c-sl-results}
These rankings come from 10,274 helpfulness and 8,135 harmlessness judgments over 24 snapshots, with raters asked to favor answers that engage carefully over answers that dodge. {#c-eval-setup}

**The RL stage.**
After RL, RL-CAI is clearly safer than both RLHF models and SL-CAI; adding chain of thought trades a little helpfulness for a little harmlessness; and the labeler's probabilities track its accuracy fairly well. {#c-rl-results}
On harmlessness, training against AI labels did at least as well as training against human labels. {#c-equal-human}
RL-CAI almost never refuses outright, and its answers to red-team prompts are often careful and still harmless. {#c-never-evasive}
Scoring harm from 0 to 4 on 64 held-out red-team prompts, 256 samples each, the helpful-only RLHF model gets worse as training goes on while HH RLHF and both RL-CAI variants improve; the authors warn that such absolute scores may be miscalibrated. {#c-absolute-harm}

**AI against human feedback.**
Lee et al. (2024) note that Bai et al. did not compare human and AI feedback directly, which left open whether RLAIF can replace RLHF. {#c-lee-gap}
Human evaluators prefer RLAIF and RLHF to the SFT policy 71% and 73% of the time for summarization and 63% and 64% for helpful dialogue, differences that are not statistically significant, and in head-to-head comparison the two are equally preferred. {#c-lee-winrates}
For harmless dialogue, RLAIF has a harmless rate of 88%, against 76% for RLHF and 64% for SFT. {#c-lee-harmless}
With a labeler the same size as the policy, RLAIF is still preferred to SFT 68% of the time, against 71% with a larger labeler. {#c-lee-same-size}
Direct RLAIF is preferred to SFT 74% of the time for summarization, and to same-size RLAIF 60% of the time; for helpful dialogue, where the labeler and the starting policy are the same checkpoint, it wins 66% against SFT. {#c-lee-direct}
Combining human and AI feedback gave no improvement over human feedback alone in their experiments. {#c-lee-combine}
Both RLAIF and RLHF write longer responses than SFT, which may bias human evaluation, and both still outperform SFT after a correction for length. {#c-lee-length}

**What makes a good AI label.**
Chain of thought generally improves the agreement of the AI labels with human labels, while detailed preambles and few-shot exemplars have mixed effects, and self-consistency over several sampled rationales strictly degrades it. {#c-lee-prompting}
The best labeler agrees with human preferences on summarization 78% of the time, which Lee et al. compare with the 73-77% agreement between human annotators that Stiennon et al. estimated. {#c-lee-agreement}
Agreement falls with the size of the labeler, from 78.0% for PaLM 2 L to 73.8% for PaLM 2 S and 62.7% for PaLM 2 XS, and Lee et al. suggest that the greater position bias of smaller labelers may contribute. {#c-lee-size}

## Limitations and failure modes

**Goodharting the AI preference model.**
Over-trained RL-CAI models Goodhart, giving overly harsh responses to harmful prompts or adding boilerplate such as "you are valid, valued, and cared for" to most red-team prompts. {#c-goodhart}
Rewriting the principles to discourage over-reactive or accusatory responses, ensembling over the 16 principles and soft or clamped labels improved this qualitatively. {#c-goodhart-fixes}

**The judge is fallible.**
The critiques of the 52B model were often inaccurate or overstated, as above, and chain-of-thought labels are overconfident until clamped.
The feedback model's choice depends on the order of the options, more so for small models, and its agreement with people falls quickly with its size.
Lee et al. also found that RLHF summaries sometimes hallucinated where RLAIF ones did not, while RLAIF summaries were sometimes less fluent, but in a blind evaluation of 70 examples the difference was not statistically significant. {#c-lee-qualitative}

**Human helpfulness labels remain.**
Bai et al. still rely on human labels for helpfulness, and state that their goal is to make human supervision more efficient, transparent and targeted, not to remove it. {#c-still-human}

**Whose values.**
The constitution is short and readable, which is its point, and so is the question of who writes it.
On the side of the method: Bai et al. argue that a system cannot avoid principles, so writing them down makes them more transparent than tens of thousands of human labels, as above.
Against a fixed list: Bai et al. themselves chose their principles in a fairly ad hoc and iterative way for research, and believe that principles should be redeveloped by a larger set of stakeholders and adapted to the intended use and location of the model. {#c-adhoc}
Human feedback has the same question in another form: Ouyang et al. align InstructGPT to the stated preferences of 40 contractors and the researchers, not to any broader notion of human values. {#c-ouyang-labelers}
AI feedback adds the values of the feedback model: Lee et al. warn that it may transfer the biases of an off-the-shelf LLM into the preferences and then into the policy, and that in high-stakes domains such as medicine, law and employment, human experts trained to label by strict policies should be considered the gold standard. {#c-lee-bias}

**How much AI feedback can stand in for people.**
For: AI identification of harms improves with capability and is approaching crowdworkers, and RLAIF matches RLHF on three tasks, as above, at a tenth of the labeling cost or less.
Bai et al. also note that fewer human red teamers would need to do the unsavory work of trying to trick AI systems into generating harmful content. {#c-red-teamers}
Against: scaling supervision further automates, and may obscure, decision making, which Bai et al. answer with chain of thought to make it more legible. {#c-obscure}
By reducing the need for human feedback, the methods also make it easier to deploy systems that people have not thoroughly tested and observed, with unforeseen failure modes. {#c-less-testing}
The results so far use one labeler family per paper, judged against crowdworkers, on summarization and dialogue.

**Dual use.**
Bai et al. note that each step from prompting to RLHF to constitutional methods lowers the barrier to training models that behave as their creators intend, including pernicious ones, and that the supervised method is especially accessible because it needs no efficient RL implementation. {#c-dual-use}
Lee et al. add that RLAIF could be used to train models to generate convincing misinformation or hateful content, and see careful governance of access to powerful LLMs as the best mitigation. {#c-lee-misuse}

## Variants

- **Direct revision.** Skipping the critique and asking for the revision directly, which Bai et al. found as good for large models but worse for small ones.
  They kept critiques for their main results because they may give more transparency into the model's reasoning. {#c-keep-critiques}
- **Chain-of-thought labels** with clamping, as above.
- **Direct RLAIF (d-RLAIF).** A reward model trained on the policy's first samples can become stale as the policy moves, so Lee et al. have the LLM rate each response from 1 to 10 during RL, take the score $s = \sum_{i=1}^{10} i\, P(i)$, normalize it to $[-1, 1]$, and use it as the reward. {#c-lee-d-rlaif}
- **Same-size and self-labeling RLAIF**, with a labeler the same size as the policy or the same checkpoint, as in the evidence above.
- **Online AI feedback.** Bai et al. propose iterated online training with AI supervision, updating the preference model with new AI feedback to keep it on the policy's distribution, and steering style, tone, persona or the handling of specific kinds of questions with other constitutions. {#c-future}

**Related methods.**
Bai et al. see their work as an extension of RLHF, similar to LaMDA, InstructGPT and Sparrow in using human data to align language models, with a supervised step very similar to earlier work on self-critique and natural-language feedback; Sparrow's decomposition of harmlessness into areas has some commonality with their principles. {#c-related}
The preference model trained on AI labels is a case of [[al:knowledge-distillation|knowledge distillation]], as Lee et al. note, with the soft labels of eq-soft-loss as the teacher's outputs.
Sampling a principle per label is an [[al:ensemble-learning|ensemble]] over judges, and the derivation shows that in expectation it trains on their average label.

## When to use it

Use it when the behavior to train can be stated as principles that a capable model can apply, as for harmlessness, and when human labels are slow, costly, or unpleasant to collect, as for red-teaming data.
It needs a capable, already helpful model to write critiques and a feedback model good enough to agree with people on the task; Lee et al. show agreement dropping steeply for small labelers.
Keep human labels where people disagree with the model, where the principles are contested, and, as Lee et al. advise, in high-stakes domains.
Check the policy with human evaluation, since the AI preference model can be gamed like a human-trained one.

## Pitfalls

> [!pitfall]
> **Hard labels.** Rounding the feedback model's probability to 0 or 1 gave much worse results in Bai et al. The check verifies that with $p = 1$ the loss keeps falling as the score difference grows, while a soft label has a finite target.

> [!pitfall]
> **Unclamped chain-of-thought labels.** A chain of thought commits to one option, and its probabilities near 0 or 1 taught RL-CAI more extreme responses. Clamp them; the check verifies the bound $\log((1 - a)/a)$ on the target.

> [!pitfall]
> **One order of the options.** The feedback model may prefer the first option whatever it says. Lee et al. score both orders and average; the check verifies that a pure position bias then gives $1/2$ and never reverses a preference.

> [!pitfall]
> **Unnormalized probabilities.** The probability of the token "(A)" alone depends on how much mass the model puts on other tokens. Normalize over the two options, eq-label; the check verifies that the label then ignores a common shift of the log-probabilities.

> [!pitfall]
> **One principle for every label.** Bai et al. found the preference model notably more robust with a principle sampled per label.

> [!pitfall]
> **Training RL-CAI too long.** The policy learns harsh refusals and boilerplate that the AI preference model rewards; watch samples and stop early.

> [!pitfall]
> **The supervised data.** Section 1.2 of Bai et al. describes fine-tuning on the final revised responses, while Section 3.1 fine-tunes on the revisions from all revision steps; the page follows Section 3.1, which describes the experiments.

## History

Constitutional AI: Harmlessness from AI Feedback, by authors at Anthropic, was posted as arXiv 2212.08073v1 on 15 Dec 2022. {#c-hist-cai}
It followed the helpful and harmless RLHF assistant of Bai et al. (2022a) from the same group, and partly replaced its human harmlessness labels.
Lee et al. (2024) describe Bai et al. (2022b) as the first work to explore RLAIF. {#c-hist-first}
RLAIF vs. RLHF, by authors at Google DeepMind and Google, appeared in the Proceedings of the 41st International Conference on Machine Learning in Vienna, PMLR 235, 2024. {#c-hist-lee}
