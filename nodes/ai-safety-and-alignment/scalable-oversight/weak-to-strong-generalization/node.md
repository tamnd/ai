---
tag: G0ZGHC
kind: method
slug: weak-to-strong-generalization
name: Weak-to-strong generalization
aliases: [W2S, weak-to-strong learning, weak-to-strong supervision, performance gap recovered, PGR, auxiliary confidence loss]
area: ai-safety-and-alignment/scalable-oversight
level: core
summary: >-
  A setup for testing whether a strong model finetuned on a weaker model's labels beats its supervisor, standing in for humans supervising superhuman models, and the methods tried in it.
purpose: studying how to supervise models that are more capable than their supervisors, by measuring how much of a strong model's ability weak labels can elicit
uses: [cross-entropy@m2, logistic-regression@m2, early-stopping@al, knowledge-distillation@al]
related_to: [knowledge-distillation@al, label-smoothing@al, boosting@al]
hyperparameters:
  - {name: alpha_max, meaning: "the final weight of the student's own hardened predictions in the confidence loss", typical: "0.75 for the largest students and 0.5 otherwise in Burns et al."}
  - {name: warm-up, meaning: "the share of training over which alpha rises linearly from 0 to alpha_max", typical: "the first 20% of training"}
  - {name: threshold t, meaning: "the cut that hardens the student's probabilities", typical: "set per batch so that half of the examples are above it, the class prior of a balanced binary task"}
  - {name: epochs and batch size, meaning: "the length of finetuning on weak labels, which controls how much of the weak errors the student fits", typical: "2 epochs at a batch size of 32 on the NLP tasks"}
  - {name: bootstrapping steps, meaning: "the number of intermediate model sizes between the weak supervisor and the final student", typical: "two intermediate sizes, three rounds of weak-to-strong learning, on chess puzzles"}
cost:
  train: "one finetuning run of the weak supervisor on ground truth, one prediction pass over the student's training set, and one finetuning run of the student on the weak labels; the strong ceiling, needed only to report PGR, is a further finetuning run of the student on ground truth; the confidence loss adds a sort of each batch of probabilities"
  inference: "none beyond the student's own forward pass"
history:
  year: 2023
  note: "Christiano, Shlegeris and Amodei (2018) proposed building a training signal for tasks too hard for a person from easier subproblems; Bowman et al. (2022) ran Cotra's sandwiching design with non-expert humans and a language model; Burns et al. (2023) replaced the humans with small models and the superhuman model with a large one, defined PGR and proposed the auxiliary confidence loss."
prereqs: [cross-entropy@m2, overfitting@al, early-stopping@al]
same_as:
  arxiv: ["2312.09390", "2211.03540", "1810.08575"]
sources:
  - {paper: "Burns, Izmailov, Kirchner, Baker, Gao, Aschenbrenner, Chen, Ecoffet, Joglekar, Leike, Sutskever and Wu (2023)", title: "Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision", locator: "arXiv 2312.09390v1, Sections 1 to 6 and Appendix A", note: "the setup, PGR, the results on NLP, chess and reward modeling, bootstrapping, the auxiliary confidence loss, and the disanalogies"}
  - {paper: "Bowman et al. (2022)", title: "Measuring Progress on Scalable Oversight for Large Language Models", locator: "arXiv 2211.03540v2, Sections 1 to 3", note: "the sandwiching design with human participants that the setup resembles, and its relaxations and results"}
  - {paper: "Christiano, Shlegeris and Amodei (2018)", title: "Supervising strong learners by amplifying weak experts", locator: "arXiv 1810.08575v1, Section 1", note: "iterated amplification, an earlier route to a training signal on tasks a person cannot judge"}
---

Weak-to-strong generalization is the ability of a strong model, finetuned on labels from a weaker supervisor, to perform better than that supervisor.
Burns et al. (2023) made it measurable: a small finetuned model labels data, a large pretrained model is finetuned on those labels, and the performance gap recovered (PGR) says what fraction of the gap between the small model and the large model trained on ground truth the weak labels close.
It stands in for humans supervising models smarter than they are.

## In plain words

Methods such as RLHF need people to judge what a model does.
A model that is much more capable than people may do things they cannot judge reliably, so their labels will be noisy in ways that follow their own blind spots.
The question is whether training on such labels makes the model copy the mistakes or brings out what it already knows.
That cannot be tested on superhuman models yet, so the setup moves the gap down: a small model plays the human and a big model plays the superhuman model.
If the big model trained on the small model's labels beats the small model, weak supervision has got more out of it than the supervisor itself knew.

## Setting

Burns et al. start from a dependence of current alignment methods: RLHF and similar techniques rely on humans being able to supervise model behaviour, and future superhuman models will act in ways too complex for humans to evaluate reliably, so humans will be able to supervise them only weakly. {#c-bu-superhuman}
They study an analogy: can supervision from a weak model elicit the full capabilities of a much stronger model? {#c-bu-question}
They test it by finetuning large pretrained models on labels produced by small ones. {#c-bu-analogy}
Their reason to expect it could work is that the strong model already has good representations, so the weak supervisor need not teach new capabilities, only elicit what the strong model knows. {#c-bu-elicit}

This is one answer to the problem of scalable oversight, which Bowman et al. define as supervising systems that potentially outperform us on most skills relevant to the task. {#c-bw-definition}
Burns et al. separate the two lines of work: scalable oversight tries to make human supervision better, for instance with critiques or task decomposition, while they study how a model generalizes beyond the supervision it gets, and they note that their setup can compare both kinds of method. {#c-bu-vs-oversight}
Their setup resembles the sandwiching proposal, which uses weak and strong humans (Cotra, 2021; Bowman, 2022). {#c-bu-sandwiching}
In sandwiching, researchers pick a task where a model is more capable than a typical person but less than an expert, and non-expert participants try to align the model to it without the experts' help. {#c-bw-sandwiching}

As a learning problem, weak-to-strong learning is a special type of weakly supervised learning, training on unreliable labels. {#c-bu-weakly-supervised}
Its errors are harder than uniform label noise because they are instance-dependent: the supervisor is wrong on particular kinds of input. {#c-bu-instance-dependent}

## The method

### The setup and PGR

Burns et al. run three finetuning experiments for each pair of a weak model and a strong model. {#c-bu-three-steps}
First, a small pretrained model is finetuned on ground-truth labels; its test accuracy is the weak performance, and its predictions on held-out examples are the weak labels. {#c-bu-step-weak}
Second, the strong pretrained model is finetuned on the weak labels; its test accuracy is the weak-to-strong performance. {#c-bu-step-student}
Third, the strong model is finetuned on ground-truth labels; its test accuracy is the strong ceiling performance, taken to represent what the strong model knows. {#c-bu-step-ceiling}
The weak supervisor is trained on one half of the dataset and its predictions on the other half are the weak labels, which are soft labels: probabilities, not classes. {#c-bu-split}

The performance gap recovered is {#c-bu-pgr-def}

$$ \mathrm{PGR} = \frac{\text{weak-to-strong} - \text{weak}}{\text{strong ceiling} - \text{weak}} , $$ {#eq-pgr}

the fraction of the gap between the weak model and the strong ceiling that weak supervision closes.
It is 1 for perfect weak-to-strong generalization and 0 when the student does no better than its supervisor. {#c-bu-pgr-ends}
Burns et al. list three advantages of the setup: it works with any pair of weak and strong models, it can be studied for any task of interest, and it can be useful before superhuman models exist. {#c-bu-advantages}

### The naive baseline

The baseline finetunes the student on the weak labels with the ordinary cross-entropy.
On the NLP tasks the models are finetuned for 2 epochs at a batch size of 32. {#c-bu-training}
Multiple-choice datasets are turned into binary ones, and classes are rebalanced where one class is more than 55% of the data, so random guessing scores 50%. {#c-bu-balanced}

### The auxiliary confidence loss

Training the student to fit the weak labels can also teach it the supervisor's errors; Burns et al. want the student to learn the supervisor's intent but not its mistakes. {#c-bu-conf-motivation}
For a binary task with student probability $f(x)$ and weak label $f_w(x)$, the loss is {#c-bu-conf-def}

$$ L_{\text{conf}}(f) = (1 - \alpha)\, \mathrm{CE}\big(f(x), f_w(x)\big) + \alpha\, \mathrm{CE}\big(f(x), \hat f_t(x)\big) , $$ {#eq-conf-loss}

where the student's predictions are hardened with a threshold $t$, {#c-bu-harden}

$$ \hat f_t(x) = \mathbb{1}\big[f(x) > t\big] \in \{0, 1\} . $$ {#eq-harden}

The threshold is set adaptively, so that $f(x) > t$ for exactly half of the examples in a batch. {#c-bu-threshold}
Half reflects the prior over classes and should be computed from the weak predictions on unbalanced or non-binary tasks. {#c-bu-prior}
They set $\alpha_{\max}$ to 0.75 for the largest students and 0.5 otherwise, and warm $\alpha$ up linearly from 0 over the first 20% of training. {#c-bu-alpha}
The same loss is a cross-entropy to a mixed target, {#c-bu-bootstrap-form}

$$ L_{\text{conf}}(f) = \mathrm{CE}\big(f(x), (1 - \alpha) f_w(x) + \alpha \hat f_t(x)\big) , $$ {#eq-conf-bootstrap}

which Burns et al. relate to the bootstrapping methods of Reed et al. (2014) and Arazo et al. (2019) for label noise, to self-training, and to conditional entropy minimization (Grandvalet and Bengio, 2004). {#c-bu-conf-related}
The second term rewards the student for confident predictions even when they disagree with the weak labels. {#c-bu-conf-disagree}

### Bootstrapping

Bootstrapping, a long-standing idea in alignment, aligns a slightly stronger model first and uses it to supervise the next. {#c-bu-bootstrap-idea}
Burns et al. build a sequence of model sizes $M_1 \to M_2 \to \dots \to M_n$, use weak labels from $M_1$ to finetune $M_2$, use $M_2$ to label data for $M_3$, and so on. {#c-bu-bootstrap-chain}
In each run they bootstrap through two intermediate sizes before finetuning the final student, three iterations of weak-to-strong learning. {#c-bu-bootstrap-iterations}

## Derivation

Two derivations support the formulas; each lists its assumptions and has a numeric check.

**PGR** (`pgr`).
Write $P_w$, $P_{w2s}$ and $P_s$ for the three performances, measured with one metric on one test set.
With $P_s \ne P_w$, PGR is an increasing affine function of $P_{w2s}$ that is 0 exactly at $P_{w2s} = P_w$ and 1 exactly at $P_{w2s} = P_s$. {#c-pgr-endpoints}
When $P_s = P_w$ it is undefined, and with the numerator fixed its magnitude grows without bound as the gap shrinks, so PGRs from pairs with a small gap are noisy. {#c-pgr-undefined}
Replacing every performance $P$ by $aP + b$ with $a \ne 0$ multiplies the numerator and the denominator by $a$, so PGR is the same on accuracy and on error rate. {#c-pgr-affine}
It is negative when the student is worse than its supervisor and above 1 when it beats the ceiling. {#c-pgr-range}
A student that agrees with its supervisor on every test input has the same score on any metric of predictions and labels, so its PGR is 0. {#c-pgr-agreement}

**The confidence loss** (`confidence-loss`).
Take $p = f(x) \in (0, 1)$, $q_w = f_w(x)$, $h = \hat f_t(x)$ treated as a constant, and $0 \le \alpha \le 1$.
Since $\mathrm{CE}(p, q) = -\log(1 - p) - q\,(\log p - \log(1 - p))$ is affine in the target $q$, the convex mix of eq-conf-loss equals the cross-entropy to the mixed target of eq-conf-bootstrap. {#c-conf-mixture}
At $\alpha = 0$ the loss is the cross-entropy to the weak labels, the naive baseline. {#c-conf-alpha-zero}
At $\alpha = 1$ it is the cross-entropy to the student's own hardened predictions, self-training on pseudo-labels, and the weak labels drop out. {#c-conf-alpha-one}
For a fixed target the loss of one example is minimized at $p = (1 - \alpha) q_w + \alpha h$, since $\partial \mathrm{CE}/\partial p = (p - q)/(p(1 - p))$. {#c-conf-minimizer}
With $p = \sigma(z)$ the gradient with respect to the logit is $p - ((1 - \alpha) q_w + \alpha h)$, so the loss pulls each prediction toward the mixed target. {#c-conf-gradient}
For $\alpha > 1/2$ the mixed target lies on the side of $1/2$ of the hardened label whatever the weak label; for a weak label $q_w < 1/2$ and $h = 1$ the target crosses $1/2$ exactly when $\alpha > (1/2 - q_w)/(1 - q_w)$, which is 0.286 for $q_w = 0.3$. {#c-conf-side}
If $t$ is the largest value in the lower half of a batch of $B$ distinct probabilities, exactly $\lfloor B/2 \rfloor$ of them exceed it; ties at $t$ make fewer exceed it. {#c-conf-half}

## Worked example

Take a batch of four examples with student probabilities $f = (0.9, 0.6, 0.4, 0.2)$ and weak labels $f_w = (0.7, 0.3, 0.6, 0.1)$.
The adaptive threshold is $t = 0.4$, the larger value of the lower half, so the hardened predictions are $\hat f_t = (1, 1, 0, 0)$.
The mean cross-entropy to the weak labels is 0.6688 and to the hardened predictions 0.3375.
At $\alpha = 0.5$ the loss is their mean, 0.5032, and the mixed target is $(0.85, 0.65, 0.30, 0.05)$.
The logit gradients, times the batch size, are $f - \text{target} = (0.05, -0.05, 0.10, 0.15)$.
The second example shows the intended effect: the weak label 0.3 says negative, but the student leans positive, the target is 0.65, and the gradient raises the logit, so the student is pushed further from the supervisor.
The third shows the cost: the weak label 0.6 is outvoted by the student's 0.4, so if the supervisor was right the loss reinforces a student error.
For PGR, a weak accuracy of 60%, a student accuracy of 70% and a ceiling of 80% give 0.5; the error rates 40%, 30% and 20% give 0.5 too.
The tests `test_worked_example` of both implementations check these numbers.

## Hyperparameters

$\alpha_{\max}$ sets how much the student trusts itself; Burns et al. use 0.75 for the largest students and 0.5 otherwise, after a linear warm-up over the first 20% of training (see the confidence loss above).
By the derivation, at 0.75 every hardened label wins against the weak label, and at 0.5 it wins unless the weak label is at 0 or 1.
The warm-up lets the student first learn the task from the weak labels, so that its hardened predictions mean something when they start to count.
The threshold encodes the class prior; Burns et al. note that their balancing mechanism is only practically feasible in the low-n classification setting. {#c-bu-balance-limit}

## Cost

The setup needs three finetuning runs per weak-strong pair: the weak supervisor and the student, plus the strong ceiling if PGR is to be reported.
The confidence loss adds a sort per batch to find the threshold and a second cross-entropy, negligible next to the model's forward pass.
Bootstrapping with $n$ model sizes costs $n - 1$ finetuning runs and $n - 2$ labelling passes instead of one of each.
The scale of the original study comes from the models: they span 7 orders of magnitude of pretraining compute in the GPT-4 family. {#c-bu-oom}

## Evidence

Burns et al. find PGR positive in almost every setting: students almost always outperform their supervisors. {#c-bu-positive}
On 22 NLP datasets, even very weak supervisors recover more than 20% of the gap, PGR grows with both supervisor and student size, and for the largest students it is often above 50%. {#c-bu-nlp}
On chess puzzles PGR is near zero for the smallest supervisors and can exceed 40% for small gaps, but for a fixed supervisor it falls as the student grows. {#c-bu-chess}
On ChatGPT reward modeling the naive method usually recovers only about 10% of the gap and almost never more than 20%. {#c-bu-rm}
Bootstrapping improves PGR on chess, especially for larger students. {#c-bu-bootstrap-chess}
With the smallest supervisor and the largest student, the confidence loss raises the median NLP PGR from about 25% to nearly 80%. {#c-bu-conf-result}
In the abstract's example, GPT-4 finetuned with a GPT-2-level supervisor and the confidence loss comes close to GPT-3.5-level performance on NLP tasks. {#c-bu-gpt35}
With human participants instead of small models, Bowman et al. find that people chatting with an unreliable language model outperform both the model alone and their own unaided answers. {#c-bw-assisted}
On their validation sets, model-assisted humans score 75% against the model's 66% on MMLU and 77% against 67% on time-limited QuALITY. {#c-bw-numbers}

## Limitations and failure modes

**Imitating the supervisor.**
With infinite data, naive fitting of the weak labels should give perfect imitation and a PGR of zero. {#c-bu-infinite}
For large gaps, weak-to-strong performance often rises and then falls well before one epoch has passed. {#c-bu-overfit}
Early stopping on ground truth, which cheats, gives about 5 points of PGR on reward modeling, and on NLP 15 points over the end of training and 10 over early stopping on weak labels. {#c-bu-early-stop}
For the naive baseline, student-supervisor agreement is high, often higher than the supervisor's accuracy, so the student is copying some of the supervisor's errors. {#c-bu-agreement}
Larger students agree less with the supervisor's errors than smaller ones, an inverse scaling. {#c-bu-inverse}

**Where the confidence loss does not help.**
It does slightly worse than the naive baseline for small students and does not help when the supervisor-student gap is small or when a dataset shows inverse scaling even with ground truth. {#c-bu-conf-limits}
It also cannot tell a weak label that is right from one that is wrong; the worked example shows it overriding a correct one.

**Disanalogies.**
Superhuman models will probably be very good at predicting what humans think and say, so imitating human errors may be easier for them than imitating a small model's errors is for today's strong students. {#c-bu-imitation-saliency}
Knowledge of the tasks may already be in the strong model's pretraining data, which can make elicitation look easier than it will be for capabilities that are latent. {#c-bu-leakage}
Burns et al. view the setup as removing one main disanalogy of prior work, not as a final, perfectly analogous one. {#c-bu-not-final}

## Variants

**Prompting with weak labels.**
At the top of the size range, 5-shot prompting is competitive with finetuning on many tasks. {#c-bu-fewshot}
Prompting with weak labels does worse than prompting with ground truth, so the problem appears in prompting too. {#c-bu-fewshot-weak}

**Generative finetuning.**
Finetuning the strong model without labels on reward-modeling data first raises PGR by about 10 to 20% against a ceiling that had the same step; with ground-truth early stopping as well, PGR reaches about 30 to 40%. {#c-bu-generative}

**Linear probes.**
Linear probes trained on ground truth reach 72% on average against 82% for finetuning on ground truth, while finetuning on weak labels and then probing with ground truth reaches 78%, closing 60% of that gap. {#c-bu-probes}

**Easy-to-hard generalization.**
A semi-supervised variant gives reliable labels only on easy examples and leaves the hard ones unlabeled. {#c-bu-easy-to-hard}

**Vision.**
As a proof of concept against pretraining leakage, Burns et al. supervise a strong vision student with AlexNet, and it generalizes well beyond AlexNet without having seen labels in pretraining. {#c-bu-alexnet}

**Iterated amplification.**
Christiano, Shlegeris and Amodei propose building a training signal for hard problems from solutions to easier subproblems. {#c-ch-amplification}
A human expert works with several copies of the current agent, and the agent learns from that composite system as it would from the expert alone. {#c-ch-amplify}
As long as several agents solve problems better than one, the composite can outperform the agent and so keep providing a useful training signal. {#c-ch-outperform}
Bootstrapping in Burns et al. is the label-only analogue of the same climb.

## When to use it

Use the setup to study or compare supervision methods on today's models when the real target is supervision of models stronger than the supervisor; report PGR with the three performances, since the ratio alone hides small gaps.
Use the confidence loss when the gap between supervisor and student is large and the student is strong enough that its own confident predictions are better than the weak labels; with a small gap or a small student, plain finetuning with early stopping is the safer default.
Use bootstrapping when PGR falls with the gap, as on chess.
For human participants and a model between non-experts and experts, sandwiching is the closer experiment.

Knowledge distillation also trains a student on another model's outputs, but the teacher is the stronger model and the student is not expected to beat it.
Label smoothing also mixes the target, but with a fixed uniform distribution instead of the student's own hardened predictions.
Boosting builds a strong learner out of many weak ones; here one weak model labels data for one strong model that already has the capability.

## Pitfalls

> [!pitfall]
> **Dividing by a tiny gap.** PGR from a pair whose ceiling barely beats the weak model is noise. The tests `test_pgr_undefined_at_zero_gap` and `test_pgr` check that a zero gap raises an error instead of returning infinity.

> [!pitfall]
> **Gradient through the hardened labels.** The indicator must be treated as a constant. The test `test_gradient_is_p_minus_target_with_a_detached_threshold` checks that the gradient is exactly $(p - \text{target})/B$.

> [!pitfall]
> **A threshold that does not halve the batch.** A fixed 0.5 cut, or ties at the threshold, change the class prior the loss imposes. The tests `test_adaptive_threshold_hardens_exactly_half` and `test_ties_at_the_threshold_and_the_balanced_version` check the count.

> [!pitfall]
> **Alpha outside [0, 1] or without warm-up.** The tests `test_alpha_out_of_range` and `test_alpha_schedule` check the range and the linear warm-up.

## History

Christiano, Shlegeris and Amodei (2018) observed that training from human demonstrations or judgments fails when a task is too complicated for a person to evaluate, and proposed iterated amplification. {#c-ch-history}
Bowman et al. (2022) ran the sandwiching design with people and a language model and reported model-assisted humans outperforming the model by about 10 percentage points. {#c-bw-history}
Burns et al. (2023) replaced both people with models, defined PGR, and proposed bootstrapping and the confidence loss as first methods; they called for more analogous setups, scalable methods and scientific understanding. {#c-bu-agenda}
