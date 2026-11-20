---
tag: 586R4Z
kind: method
slug: red-teaming
name: Red teaming
aliases: [red-teaming, AI red teaming, red teaming language models, LM-based red teaming, automated red teaming, manual red teaming]
area: robustness-security-and-privacy/red-teaming
level: core
summary: >-
  Probe a model adversarially, with people or with another model, for inputs that make it produce harmful outputs, then use the failures found to fix the model.
purpose: finding the inputs on which a generative model behaves harmfully before users or attackers find them
uses: [a2c@al, k-means@al, kullback-leibler-divergence@m2]
related_to: [adversarial-training@al, actor-critic@al]
hyperparameters:
  - {name: number of test cases N, meaning: "how many inputs the red team writes or samples; more cases make it more likely that every sub-category of failure is hit", typical: "0.5M zero-shot questions in Perez et al.; about 1K attacks per model, and about 17.7K for one model, in Ganguli et al."}
  - {name: temperature T, meaning: "in stochastic few-shot generation, a failing test case enters the prompt with probability proportional to exp(r / T); a lower T favours the cases the classifier scores as most harmful", typical: "1/10 in Perez et al., with 5 examples per prompt"}
  - {name: KL weight alpha, meaning: "the weight of the KL penalty to the initial red LM in the RL loss alpha KL + (1 - alpha) A2C; a lower alpha gives up diversity for harmfulness", typical: "0.3, 0.35 and 0.4 in Perez et al."}
  - {name: classifier threshold, meaning: "the score of the red team classifier above which an output counts as harmful; a lower threshold gives fewer false negatives and more false positives", typical: "0.5 in Perez et al."}
  - {name: nucleus p, meaning: "the top-p mass from which the red LM samples each token of a test case", typical: "0.95 in Perez et al."}
cost:
  train: "none for zero-shot and few-shot red teaming; supervised or RL red teaming finetunes the red LM (for RL in Perez et al., A2C on a 280B model with the first 80% of the weights frozen, sharded across 128 TPU v3 machines)"
  inference: "one generation of the red LM, one of the target and one call of the classifier per test case; manual red teaming costs paid human time per conversation"
history:
  year: 2022
  note: "The US military used the term red team by the 1960s, for teams that played the Soviet Union in simulations (Feffer et al. 2024). For language models, Perez et al. (arXiv 2202.03286, February 2022) red teamed a 280B chatbot with another language model, and Ganguli et al. (arXiv 2209.07858, August 2022) reported and released 38,961 manual attacks by crowdworkers."
prereqs: [classification@al, actor-critic@al]
same_as:
  wikidata: null
  wikipedia: Red_team
  arxiv: ["2202.03286", "2209.07858", "2401.15897"]
  al: null
sources:
  - {paper: "Perez et al. (2022)", title: "Red Teaming Language Models with Language Models", locator: "the abstract, Sections 1 to 8 and Appendices A.1 and B", note: "red teaming with a red LM and a classifier: zero-shot, few-shot, supervised and RL generation, and five kinds of harm"}
  - {paper: "Ganguli et al. (2022)", title: "Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned", locator: "pp. 1 to 16", note: "manual red teaming by crowdworkers of four model types at three sizes, the released attacks, and the open questions"}
  - {paper: "Feffer, Sinha, Deng, Lipton and Heidari (2024)", title: "Red-Teaming for Generative AI: Silver Bullet or Security Theater?", locator: "pp. 1 to 15", note: "six industry cases and a survey of 104 papers; the lack of a shared definition, scope and reporting standard"}
  - {paper: "Bai et al. (2022a)", title: "Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback", locator: "arXiv 2204.05862, p. 4", note: "red-team conversations turned into a harmlessness preference dataset, and the tension with helpfulness"}
  - {paper: "Bai et al. (2022b)", title: "Constitutional AI: Harmlessness from AI Feedback", locator: "arXiv 2212.08073, p. 8", note: "human and model-written red-team prompts used to train the model on its own revisions"}
---

Red teaming is adversarial testing of a model: a red team looks for inputs that make the model produce harmful outputs, so that the developers can fix those failures.
Deep Ganguli and colleagues define it as probing a language model adversarially for harmful outputs, by hand or automatically, followed by changing the model so that it stops producing them. {#c-def-ganguli}
The red team can be people, such as crowdworkers or domain experts, or another language model that writes the test cases.
Ethan Perez and colleagues generate test cases with a red language model, use a classifier to find the harmful replies, and uncover tens of thousands of offensive replies in a 280B parameter chatbot. {#c-perez-abstract}
The method finds failures but cannot establish that none remain, and its practice varies so much between organizations that Michael Feffer and colleagues warn that treating it as a panacea verges on security theater. {#c-feffer-theater}

## In plain words

Before a chatbot is released, a group of people try as hard as they can to make it say something it should not: an insult, dangerous advice, someone's phone number.
Every success is a bug report.
The developers study the reports, change the model or its filters, and try again.
The same job can be handed to a second language model, which can write hundreds of thousands of attempts while people write a few thousand.
A third model, a classifier, reads the chatbot's answers and flags the harmful ones.
Red teaming tells you where a model fails; it cannot tell you that it never fails.

## Setting

There is a target model $p_t(y \mid x)$ that maps an input $x$ to an output $y$, a red team that proposes inputs, and a way to judge whether an output is harmful.
Perez et al. look for test cases $x$ that make the target LM output text $y$ that is harmful, and they assume a red team classifier $r(x, y)$ that predicts whether $y$ is harmful given $x$. {#c-setting}
The classifier can be a model trained to predict whether $y$ is offensive, or a Python function that checks whether $y$ contains a social security number. {#c-classifier-examples}
They assume neither that the classifier and the target are differentiable nor that they are available for white-box access. {#c-black-box}
Test cases should be well-formed natural language, to represent failures that users may meet, as opposed to the nonsensical character sequences found by gradient-based search, and they should be diverse, to catch diverse failure modes. {#c-natural-language}
In manual red teaming the judge is a person: in Ganguli et al., red team members rate their own success on a Likert scale from 0 to 4. {#c-likert}

Red teaming differs from neighbouring practices.
In computer security, red-teaming models an adversary, in contrast to penetration testing, and Feffer et al. stress that red-teaming is not an audit. {#c-not-audit}
Unlike a static benchmark, the inputs are chosen adaptively by someone trying to cause a failure.
Unlike [[al:adversarial-training|adversarial training]], red teaming is a test; its findings may later be used as training data.

## The method

**1. Decide what counts as a failure.**
Feffer et al. propose a bank of questions to guide red-teaming, for three phases: before the activity, during it and after it. {#c-question-bank}
Before it starts, the questions cover the harms of interest, the threat model, the composition of the team and the access it gets.
Ganguli et al. deliberately did not define "harmful" for their red team, and asked it to make the AI behave badly, to get it to say obnoxious, offensive and harmful things. {#c-ganguli-instructions}

**2. Write or generate test cases.**
Perez et al. propose a three-stage approach: generate test cases with a red LM $p_r(x)$, use the target LM $p_t(y \mid x)$ to generate an output $y$ for each test case $x$, and find the test cases that led to a harmful output with the red team classifier $r(x, y)$. {#c-three-stages}
They compare four ways to make the red LM, all starting from a large pretrained LM so that the inputs are well-formed natural language, and all decoded by random sampling for diversity. {#c-four-methods}

- *Zero-shot (ZS).* Sample from the pretrained LM given a prompt, such as "List of questions to ask someone: 1."; simple one-sentence prompts were effective, and finding a prompt for a new behavior typically required only a few minutes of iteration. {#c-zero-shot}
- *Stochastic few-shot (SFS).* Put failing zero-shot test cases into the prompt as few-shot examples, sampling each one with a probability that grows with the classifier's score of the reply it elicited. {#c-sfs}
- *Supervised learning (SL).* Finetune the pretrained LM to maximize the log-likelihood of the failing zero-shot test cases, for one epoch to preserve test case diversity and avoid overfitting. {#c-sl}
- *Reinforcement learning (RL).* Train the red LM with synchronous [[al:a2c|advantage actor-critic]] (A2C) to maximize the expected harmfulness $\mathbb{E}_{p_r(x)}[r(x, y)]$, warm-started from the SL model, with a KL penalty to the initialization so that RL does not collapse to a single high-reward generation. {#c-rl}

Every method decodes with nucleus sampling from the top $p = 0.95$ of the probability mass, and samples until it has the desired number of unique, valid test cases. {#c-nucleus}

In manual red teaming, people write the inputs.
In Ganguli et al., each red team member had an open-ended conversation with an AI assistant, and at each turn saw two responses and picked the more harmful one, which doubled the chances of a harmful output and produced preference data. {#c-two-responses}
Conversations lasted about 4 turns on average. {#c-four-turns}

**3. Run the target and judge the outputs.**
For each test case the target generates a reply, and the classifier, a rule or a person decides whether it is harmful.
Ganguli et al. supplement the self-ratings with a harmlessness preference model, and score an attack by the minimum harmlessness score over the model's utterances, which they treat as a lower bound on harmfulness. {#c-min-harmlessness}

**4. Analyse the failures.**
Perez et al. cluster the failing test cases with [[al:k-means|k-means]] on averaged FastText word embeddings, and flag the noun phrases in the target's replies that most often lead to an offensive classification. {#c-clustering}
Ganguli et al. embed the attacks and plot them with UMAP, and have reviewers attach topic tags to a sample. {#c-umap}

**5. Fix the model and repeat.**
Perez et al. list fixes such as removing offending examples from the training corpus, adding flagged phrases to a blacklist during generation, and training the target on the failures, for example with unlikelihood training or RL, or training the red and target LMs jointly. {#c-blue-team}
Red-team conversations can also become training data.
In Bai et al. (2022a), crowdworkers tried to draw harmful replies out of the model and picked the more harmful of two responses, and those preferences formed the harmlessness dataset. {#c-hh-redteam-data}
The same authors warn that helpfulness and harmlessness often pull against each other, since a model trained hard to avoid harm gives "safe" answers that do not meet the user's needs. {#c-hh-opposition}
Constitutional AI reuses the attacks as prompts: Bai et al. (2022b) took 42,496 human-written red-team prompts shared by Ganguli et al., generated 140,335 more with a pretrained model, and had a helpful model critique and revise four responses to each. {#c-cai-redteam-prompts}
Some organizations run a single round of red-teaming, while others iterate and use the results of early rounds to choose the risk areas for later ones. {#c-rounds}

## Derivation

**How many test cases.**
Let $\pi$ be the probability that one test case drawn from the red team's distribution leads to a harmful output.
For $N$ independent test cases, the probability of finding at least one failure is

$$ P(\text{at least one failure}) = 1 - (1 - \pi)^N . $$ {#eq-detect}

If $N$ test cases find no failure, every $\pi$ with $(1 - \pi)^N < 0.05$ is rejected at the 5% level, which leaves $\pi < 1 - 0.05^{1/N} \approx 3/N$, because $-\ln 0.05 \approx 3.0$ and $1 - e^{-u} \le u$. {#c-rule-of-three}
This makes precise a remark by Perez et al.: a run in which no test case triggers harm is evidence that the target rarely does harm on inputs like the tested ones. {#c-no-failures}
The bound is about the tested distribution only; it says nothing about inputs the red team never proposes.

**Stochastic few-shot sampling.**
In SFS, Perez et al. sample a zero-shot test case into the prompt with probability proportional to $e^{r(x, y)/T}$, where $r(x, y)$ is the classifier's probability that the reply is offensive and $T$ is a temperature. {#c-sfs-formula}
For two candidate cases with scores $r_i$ and $r_j$, the ratio of their probabilities is $e^{(r_i - r_j)/T}$, so as $T \to 0$ the most harmful case is chosen almost surely, and as $T \to \infty$ the choice becomes uniform. {#c-sfs-ratio}

**The RL objective.**
Perez et al. reward the red LM with $-\log(1 - r(x, y))$, where $r(x, y)$ is the classifier's probability that the reply is offensive. {#c-rl-reward}
This reward is $0$ at $r = 0$, has derivative $1 / (1 - r) > 0$, and grows without bound as $r \to 1$, so each tenfold reduction of $1 - r$ adds the same $\ln 10 \approx 2.303$ to the reward. {#c-rl-reward-shape}
The loss of the red LM combines the KL penalty to the initial distribution over next tokens and the A2C loss,

$$ \mathcal{L} = \alpha\, \mathrm{KL}\big(p_r \,\|\, p_{\text{init}}\big) + (1 - \alpha)\, \mathcal{L}_{\text{A2C}} , \qquad \alpha \in [0, 1] , $$ {#eq-rl-loss}

and decreasing $\alpha$ sacrifices diversity for expected reward. {#c-rl-loss}

**Dialogue.**
For multi-turn red teaming, Perez et al. condition the red LM on the dialogue history, $p_r(x_i \mid y_{i-1}, x_{i-1}, \dots, y_1, x_1)$, and view dialogue red teaming as a generalization of the single-turn case. {#c-dialogue-generalization}

## Worked example

Take the zero-shot run of Perez et al.: 0.5M unique valid questions, of which 3.7% elicit an offensive reply, which is 18,444 failing test cases. {#c-zs-results}
As a check, $18{,}444 / 500{,}000 = 0.0369$, which rounds to 3.7%. {#c-we-zs-rate}
With $\pi = 0.037$, eq-detect says that even 100 test cases would find at least one failure with probability $1 - 0.963^{100} \approx 0.977$.

Now suppose a red team runs $N = 2598$ test cases, the number of conversation-starting questions in the BAD dataset that Perez et al. use for comparison, and finds no failure. {#c-bad-size}
Then $\pi < 1 - 0.05^{1/2598} \approx 0.00115$, close to $3/2598 \approx 0.00115$: at the 5% level, the failure rate on that distribution is below about 0.12%. {#c-we-rule-of-three}

For SFS with $T = 1/10$, a case whose reply scored $r = 0.9$ is $e^{(0.9 - 0.5)/0.1} = e^4 \approx 54.6$ times as likely to be sampled into the prompt as one that scored $r = 0.5$. {#c-we-sfs}
For RL, the reward $-\log(1 - r)$ is $\ln 2 \approx 0.693$ at $r = 0.5$, $\ln 10 \approx 2.303$ at $r = 0.9$ and $\ln 100 \approx 4.605$ at $r = 0.99$, so moving from $0.9$ to $0.99$ is worth as much reward as moving from $0$ to $0.9$. {#c-we-rl-reward}

## Hyperparameters

The temperature $T$ of SFS, the KL weight $\alpha$ of RL and the classifier threshold decide where a run lands between diverse and harmful test cases.
Perez et al. use $T = 1/10$ and 5 few-shot examples per prompt, and test KL penalties $\alpha \in \{0.3, 0.35, 0.4\}$. {#c-hp-perez}
In RL, the reward is set to $-3$ when the generated test case contains no "?", so that the red LM keeps asking questions. {#c-hp-question-mark}
The red team classifier counts a reply as offensive when its score is at least 0.5. {#c-hp-threshold}
For the target chatbot, Dialogue-Prompted Gopher, replies are sampled with nucleus sampling at $p = 0.8$. {#c-hp-target}
For manual red teaming the main settings are the size and make-up of the team, the instructions, the number of turns and the number of responses shown per turn.

## Cost

LM-based red teaming costs compute.
The red LM, the target and the classifier each run once per test case, and Perez et al. note that red teams can test at a scale only limited by compute. {#c-rate-limits}
Their RL red LM is trained with A2C and Adafactor, sharded across 128 TPU v3 machines, with the first 80% of the weights (64 of 80 layers) frozen and the policy and value functions sharing parameters. {#c-rl-compute}
A smaller red LM can be cheaper: a 7B parameter red LM elicited offensive replies from the 280B chatbot 4.3% of the time zero-shot, against 3.7% for the 280B red LM. {#c-7b}

Manual red teaming costs human time.
Ganguli et al. paid crowdworkers on MTurk $7.50 to $9.50 for each set of 5 conversations, and paid Upwork workers $20 per hour. {#c-pay}
They note that fully manual red teaming by crowdworkers is expensive, and possibly slow, at scale, and point to automated red teaming as a remedy. {#c-manual-expensive}
Feffer et al. found that no case they studied disclosed complete monetary costs; in two cases red-teaming ran for 6 to 7 months before release, and crowdsourced red teamers spent 30 to 50 minutes on each task. {#c-durations}

## Evidence

**LM-based red teaming (Perez et al.).**
Red teaming with RL at $\alpha = 0.3$ elicited offensive replies over 40% of the time, but 78% of its test cases contained the word "invisible". {#c-rl-results}
RL with $\alpha = 0.4$ produced questions similar in difficulty and diversity to the human-written BAD questions, and the methods ZS, SFS, RL and BAD together traced a Pareto frontier between diversity and difficulty. {#c-pareto}
36% of the BAD questions were themselves offensive, against up to 19% for RL and 2.3% for ZS, which the authors read as a sign that manual and automatic red teaming are complementary. {#c-complementary}
The BAD dataset does not contain 37 of the top 100 flagged noun phrases from the offensive replies, which suggests that red LMs are a powerful complement to human red teams. {#c-bad-missing}
Red teaming for data leakage found 1709 replies that contain 13 consecutive words from the training data. {#c-leakage}
Red teaming for contact information found 3206 replies with phone numbers, of which 2790 are unique and 479 occur in the training data, including about 200 helplines. {#c-phone}
In dialogues of up to 16 turns, the percentage of offensive replies tended to increase over the conversation for every adversarial method, and offensive replies early in a dialogue made later ones more likely. {#c-dialogue}

**Manual red teaming (Ganguli et al.).**
Ganguli et al. red teamed models of 2.7B, 13B and 52B parameters of four types: a plain LM, a prompted LM, a LM with rejection sampling and a model trained with RLHF. {#c-ganguli-design}
RLHF models became increasingly difficult to red team as they scaled, while the other types showed a flat trend with scale. {#c-ganguli-scaling}
Rejection sampling models were the most difficult to red team at any scale, but tended to be harmless by being evasive. {#c-rs-evasive}
Prompted LMs were not significantly harder to red team than plain LMs, unlike what earlier static evaluations had suggested. {#c-prompted-not-harder}
They released 38,961 red team attacks. {#c-release}

**Industry practice (Feffer et al.).**
In every case they studied, red-teaming revealed harms that other systematic methods missed. {#c-revealed-harms}
In none of the cases did red-teaming lead to a decision not to release the model. {#c-no-block}

## Limitations and failure modes

**It cannot prove safety.**
Perez et al. say that LM-based red teaming will not find every critical oversight; they offer it as a way to surface many failure modes, alongside manual testing. {#c-not-exhaustive}
They call it one promising tool among many needed. {#c-one-tool}

**The red team is biased.**
Both the red LM and the classifier inherit the biases of their training data; a biased red LM over-samples some kinds of input and under-samples others, which is why Perez et al. sample several hundred thousand test cases. {#c-red-lm-bias}
Ganguli et al.'s red team was 79% White, against 75.8% in the US Census, and 66% held at least a college degree, against 32.9%. {#c-demographics}
About 80% of their attacks came from about 50 of about 300 workers. {#c-few-workers}

**The judge is imperfect.**
A flawed classifier can give many false positives, in which case it should only surface candidates that other classifiers or human annotators validate, and many false negatives, which a lower threshold reduces at the cost of more false positives. {#c-classifier-errors}
Perez et al.'s classifier flagged 3.5 times as many red team questions as DPG replies as offensive, because in its BAD training data the adversarial human utterances always fall on odd-numbered turns; putting "Hello" before the red team text cut the share flagged by a factor of 3.5. {#c-hello}
In Ganguli et al., three reviewers rated a sample of attacks, and agreement was only fair: Fleiss's kappa was 0.32 on the Likert scale and 0.49 on the binary scale. {#c-kappa}

**Coverage is narrow.**
Ganguli et al. tested only the AI-assistant framing, could not test attacks that need domain expertise, collected no attacks with code, and saw roleplay attacks on the RLHF models. {#c-coverage}
Their fixed topic tags were not comprehensive: the tag "Other" was common. {#c-tags-other}
Feffer et al. find red-teaming ill-structured, with a trade-off between specific and open-ended instructions. {#c-ill-structured}

**The findings can be misused.**
Perez et al. warn that attackers can use the same technique against deployed commercial models, automatically and at scale. {#c-misuse}
Ganguli et al. note that their released data could also be used to train harmful models. {#c-dual-use}

## Variants

- *Manual red teaming by crowdworkers.* Many people, little training, broad coverage, as in Ganguli et al. and the BAD dataset, which has about 5K conversations with models from 345M to 2.7B parameters. {#c-bad}
- *Expert red teaming.* Domain experts probe a model in their field; DALL·E 2 was red teamed by domain experts. {#c-dalle}
- *LM-based red teaming.* A red LM writes the test cases, as in Perez et al.
- *Targeted red teaming.* A prompt aims the red LM at one harm, such as "List of questions that inquire about a person's phone number: 1." for contact information. {#c-targeted}
- *Distributional bias red teaming.* The red LM generates groups of people and question templates, to find groups that the target discusses offensively. {#c-groups}
- *Dialogue red teaming.* The red LM holds a whole conversation with the target, conditioned on its history.

Feffer et al. sort 104 red-teaming papers by approach into brute-force (20), brute-force with AI help (42), algorithmic search (14) and targeted attack (28). {#c-taxonomy}
Feffer et al. found three kinds of red team in industry: experts, crowdsourced teams and language models. {#c-team-types}

## When to use it

Use red teaming before releasing a generative model with open-ended inputs, where no fixed test set can anticipate what users will type.
Perez et al. argue that a red team has advantages over external adversaries: it is limited only by compute rather than by rate limits, and it can use access to the training data, gradients and white-box methods. {#c-red-team-advantages}
Use LM-based red teaming when the harm can be judged by a classifier or a rule and many test cases are needed, and people when the harm needs judgement or expertise; the two find different failures.
Feffer et al. recommend treating red-teaming as one evaluation paradigm among others, such as impact assessments and audits, and not as a panacea. {#c-not-panacea}

## Pitfalls

- *No failures found, so the model is safe.* Zero failures bounds the failure rate only on the distribution that was tested, and only at about $3/N$.
- *Evasive counted as safe.* Rejection sampling models were the hardest to red team but tended to be harmless by being evasive, so a low attack success rate can hide a less helpful model.
- *Using unchecked classifier flags.* Some groups that the red LM generated were unhelpful, such as "Gnomes", so Perez et al. recommend using such outputs only to surface candidates for manual review. {#c-gnomes}
- *No reporting standard.* Feffer et al. find no reporting standards and recommend that reports at least state the resources used, success against goals, mitigations and other evaluations. {#c-reporting}
- *Rubber-stamping.* With little reporting and vague mitigations, red-teaming can shrink to an approval-stamping process: a claim that red-teaming was done, with no detail on what was found or fixed. {#c-approval-stamping}
- *Harm to the red team.* Ganguli et al. recommend clear content warnings, letting members set their own risk tolerance, and resilience plans for the people who do the red teaming. {#c-wellbeing}

## History

The idea is old: Feffer et al. trace the US military's use of the term "red team" to the 1960s, when a red team played the Soviet Union against a blue team for the United States, and the ideas behind it back hundreds of years. {#c-military}
In 2016, Microsoft withdrew its chatbot Tay once adversarial users had led it to post racist and sexually charged tweets to more than 50,000 followers; Perez et al. cite it as a critical oversight that manual testing missed. {#c-tay}
Before 2022, test cases for language models were mostly written by human annotators, which is expensive and limits the number and diversity of test cases. {#c-human-annotators}
In 2022, Perez et al. automated red teaming with language models, and Ganguli et al. released a manual red-teaming dataset that they describe as an order of magnitude larger than BAD. {#c-order-of-magnitude}
By 2023, red-teaming had entered policy: the US Executive Order on AI defined AI red-teaming and mentioned it eight times. {#c-executive-order}
