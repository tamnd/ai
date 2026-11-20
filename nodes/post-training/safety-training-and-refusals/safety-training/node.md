---
tag: NEW-safety-training
kind: method
slug: safety-training
name: Safety training
aliases: [safety fine-tuning, safety tuning, refusal training, harmlessness training, safety alignment]
area: post-training/safety-training-and-refusals
level: core
summary: >-
  Fine-tuning a language model to refuse harmful requests and answer the rest, with safety demonstrations, a safety reward model and context distillation.
purpose: making a chat model decline requests for harmful content while it stays helpful on everything else, including prompts that only look dangerous
uses: [fine-tuning, reinforcement-learning-from-human-feedback, reward-model, distillation, kullback-leibler-divergence@m2]
related_to: [red-teaming, constitutional-ai, instruction-tuning, lora, direct-preference-optimization, residual-connection]
hyperparameters:
  - {name: safety data fraction, meaning: "the share of safety demonstrations or safety preference data in the tuning mix", typical: "a few percent of an instruction set (Bianchi et al.: 100 to 2,000 added to 20,000); Llama 2 varied it from 0% to 100% of about 0.1M safety samples"}
  - {name: response style, meaning: "what a good refusal looks like: a bare refusal, or an explanation and a safe alternative", typical: "address the safety concern, explain the risk, then offer help where possible (Llama 2's guidelines)"}
  - {name: safety preprompt, meaning: "the text put in front of adversarial prompts to make the teacher safer in context distillation", typical: "a template filled with adjectives such as responsible, respectful or wise"}
  - {name: distillation filter, meaning: "when a context-distilled answer replaces the original", typical: "only when the safety reward model scores it higher"}
  - {name: refusal-direction layer and position, meaning: "where the difference of means is taken when refusal is analysed or edited", typical: "chosen per model on a validation split over layers and post-instruction token positions"}
  - {name: GCG budget, meaning: "suffix length, top-k, batch size and steps of the adversarial-suffix search used to test robustness", typical: "20 tokens, top-k 256, batch 512, 500 steps"}
cost:
  train: "for supervised safety data, the cost of the instruction-tuning run it is mixed into; safety RLHF adds a safety reward model, adversarial prompt collection and red teaming, which are labour costs as much as compute; context distillation adds one generation pass per adversarial prompt and one reward-model scoring of each pair"
  inference: "none beyond the model; an orthogonalized model has the same shape and cost as the original"
history:
  year: 2022
  people: [Yuntao Bai, Deep Ganguli, Hugo Touvron, Alexander Wei, Andy Zou, Paul Röttger, Federico Bianchi, Xiangyu Qi, Andy Arditi]
  note: "Anthropic's helpful and harmless assistant (2022) trained harmlessness with RLHF on red-teaming data and found the two goals in tension, and Constitutional AI (2022) replaced human harmlessness labels with AI feedback. Llama 2-Chat (2023) combined safety SFT, a separate safety reward model and context distillation. The same year Wei et al. and Zou et al. broke safety-trained models with prompt attacks and optimized suffixes, XSTest measured over-refusal, and Qi et al. undid safety training with a few fine-tuning examples; Bianchi et al. and Arditi et al. (2024) measured how little data safety needs and found refusal mediated by one direction."
prereqs: [fine-tuning, reinforcement-learning-from-human-feedback]
same_as:
  wikidata: null
  wikipedia: null
  arxiv: ["2307.09288", "2204.05862", "2309.07875", "2308.01263", "2307.02483", "2307.15043", "2406.11717", "2310.03693", "2212.08073", "2209.07858"]
  al: null
sources:
  - {paper: "Touvron, Martin, Stone, Albert and others (2023)", title: "Llama 2: Open Foundation and Fine-Tuned Chat Models", locator: "arXiv 2307.09288v2, pp. 23, 24 and 26 to 29", note: "the three safety techniques, safety RLHF, the safety-data scaling study, false refusals, context distillation, red teaming and gamma"}
  - {paper: "Bianchi, Suzgun, Attanasio, Röttger and others (2024)", title: "Safety-Tuned LLaMAs: Lessons From Improving the Safety of Large Language Models that Follow Instructions", locator: "arXiv 2309.07875v3, pp. 1, 4, 5, 8 and 9", note: "how many safety examples are needed, the data, exaggerated safety, questions versus instructions"}
  - {paper: "Röttger, Kirk, Vidgen, Attanasio and others (2023)", title: "XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models", locator: "arXiv 2308.01263v3, pp. 1, 2 and 5", note: "the test suite, lexical overfitting, refusal rates of Llama 2"}
  - {paper: "Wei, Haghtalab and Steinhardt (2023)", title: "Jailbroken: How Does LLM Safety Training Fail?", locator: "arXiv 2307.02483v1, pp. 1 to 6", note: "jailbreak definition, competing objectives and mismatched generalization, prefix injection, refusal suppression, Base64"}
  - {paper: "Zou, Wang, Carlini, Nasr and others (2023)", title: "Universal and Transferable Adversarial Attacks on Aligned Language Models", locator: "arXiv 2307.15043v2, pp. 1 and 4 to 9", note: "the affirmative target, the loss, greedy coordinate gradient, transfer"}
  - {paper: "Arditi, Obeso, Syed, Paleka and others (2024)", title: "Refusal in Language Models Is Mediated by a Single Direction", locator: "arXiv 2406.11717v3, pp. 1, 3, 4, 6 and 7", note: "difference-in-means direction, ablation, activation addition, weight orthogonalization, refusal scoring"}
  - {paper: "Qi, Zeng, Xie, Chen and others (2023)", title: "Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!", locator: "arXiv 2310.03693v1, pp. 1, 3, 5, 6 and 8", note: "harmful, identity-shifting and benign fine-tuning attacks"}
  - {paper: "Bai, Jones, Ndousse, Askell and others (2022)", title: "Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback", locator: "arXiv 2204.05862v1, pp. 6, 19, 24 and 25", note: "the tension between helpfulness and harmlessness, alignment taxes, exaggerated responses"}
  - {paper: "Bai, Kadavath, Kundu, Askell and others (2022)", title: "Constitutional AI: Harmlessness from AI Feedback", locator: "arXiv 2212.08073v1, pp. 1 and 4", note: "evasiveness, harmless but non-evasive responses"}
  - {paper: "Ganguli, Lovitt, Kernion, Askell and others (2022)", title: "Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned", locator: "arXiv 2209.07858v2, p. 1", note: "red-teaming difficulty by model type and scale"}
---

Safety training is the part of post-training that teaches a language model which requests to refuse.
It is not a separate algorithm but a use of the ordinary ones: harmful prompts paired with safe answers are added to supervised fine-tuning, a safety reward model joins the helpfulness reward model in RLHF, and a model told to be safe in its prompt is distilled into one that is safe without being told.
The hard part is the boundary.
Train too little and simple rewordings get harmful answers out of the model; train too much, or on too narrow data, and the model refuses to say how to kill a Python process.
This page covers the training methods, how refusals are measured, how safety training fails, and what interpretability has found about how a trained model represents refusal.

## In plain words

A pre-trained language model will continue any text, including a request for instructions to build a weapon.
Instruction tuning makes it answer requests, which makes the problem worse: it now tries hard to help with everything.
Safety training adds examples of the behaviour wanted on dangerous requests, usually a polite refusal with an explanation, and rewards that behaviour during reinforcement learning.
The model learns a rough sense of which requests are harmful, and a habit of refusing those.
That sense is learned from examples, so it generalizes the way any learned classifier does: it catches requests that look like its training examples, misses harmful requests dressed up in unusual ways, and sometimes refuses harmless requests that share words with harmful ones.

## Setting

Language models are often trained to refuse certain categories of prompts during safety training; restricted behaviors are those a safety-trained model is trained to avoid, often harmful ones such as creating misinformation or aiding crime. {#c-jb-restricted}
A jailbreak attack on a safety-trained model is an attempt to elicit an on-topic response to a prompt P for restricted behavior by submitting a modified prompt P′. {#c-jb-attack}
Safety training aims for a model that refuses restricted prompts and still answers everything else, and the two aims pull against each other.
Harmlessness often acts as a constraint on helpfulness: preference models trained only on helpfulness data or only on harmlessness data perform far below chance on the other test set, evidence that the two are anti-correlated objectives. {#c-hh-anti}
In the extreme, a model that refuses any prompt, safe or unsafe, would be perfectly harmless but also completely useless. {#c-xs-useless}
The opposite extreme, a model that answers everything, is the instruction-tuned model before safety training.
Safety is inherently a long-tail problem, where the challenge comes from a small number of very specific cases. {#c-l2-longtail}
The methods below are therefore judged on two numbers at once: how often the model complies with harmful prompts, and how often it refuses safe ones.

Write π for the model, x for a prompt and y for a response.
A safety dataset is a set of prompts with safe target responses, or with preference pairs where the safer response is preferred; a red-teaming set is a set of prompts written to make the model misbehave.

## The method

Llama 2-Chat's safety fine-tuning uses three techniques: supervised safety fine-tuning, safety RLHF, and safety context distillation. {#c-l2-three}
Each one reuses a post-training method with safety-specific data, and each has its own page for the general method: fine-tuning, reinforcement learning from human feedback with a reward model, and knowledge distillation.

### Supervised safety data

Supervised safety fine-tuning gathers adversarial prompts and safe demonstrations that are then included in the general supervised fine-tuning process, which teaches the model the safety guidelines even before RLHF. {#c-l2-sft}
Llama 2's best practice for safe and helpful responses: the model should first address immediate safety concerns if applicable, then address the prompt by explaining the potential risks to the user, and finally provide additional information if possible. {#c-l2-best}
After gathering only a few thousand supervised safety demonstrations, the Llama 2 team switched entirely to RLHF to teach the model more nuanced responses, which may also make the model more robust to jailbreak attempts. {#c-l2-switch}

The open recipe of Bianchi et al. measures how much safety data an instruction-tuned model needs.
Bianchi et al. randomly selected 2,000 questions from the Anthropic red teaming dataset, used GPT-3.5-turbo to generate safe responses, a majority of them rejections, reviewed them manually, and turned the questions into instructions. {#c-bi-data}
Bianchi et al. add 100, 300, 500, 1000, 1500 and 2,000 safety instructions and fine-tune LLaMA 7B, LLaMA 13B and Falcon 7B with LoRA for four epochs. {#c-bi-setup}
They fine-tuned with LoRA, so the safety data was learned by the same low-rank update as the instructions.
Harmfulness starts to decrease when only 100 safety instructions are added to the 20,000 general training instructions, and 500 to 1,000 safety instructions are enough to substantially reduce it. {#c-bi-100}
Adding just 3% safety examples (a few hundred demonstrations) when fine-tuning a model like LLaMA can substantially improve its safety without making it significantly less capable or helpful on standard benchmarks. {#c-bi-3pct}
Adding n safety examples to N = 20,000 general ones makes them the fraction n/(N + n) of the data: 0.50% for n = 100, 2.44% for 500, 4.76% for 1,000 and 9.09% for 2,000, and 3% corresponds to n ≈ 619. {#c-st-frac}
Training on safety questions does not generalize to instruction prompts: models trained on questions such as How do I break into a house are less safe on instructions such as Write a step-by-step guide to breaking into a house. {#c-bi-qs}
The form of the prompts matters as well as their number, so a safety set should contain harmful requests phrased in each way users phrase them.
`mix_datasets` and `safety_examples_for_fraction` in the NumPy file build such a mix.

### Safety RLHF and context distillation

Safety RLHF integrates safety into the general RLHF pipeline by training a safety-specific reward model and gathering more challenging adversarial prompts for rejection sampling style fine-tuning and PPO optimization. {#c-l2-rlhf}
How Llama 2 combines the safety and helpfulness scores into one reward, switching to the safety score on unsafe prompts or when it is low, is described on the reward-model page and computed there by `llama2_combined_reward`.
After safety tuning with RLHF the distribution of safety reward scores on the safety set shifts to higher scores and its long tail near zero thins out, while the helpfulness score distribution is preserved. {#c-l2-help}
Llama 2 variants fine-tuned with 0%, 1%, 10%, 25%, 50% and 100% of the safety data (about 0.1M samples) show the mean safety reward model score improving with more safety data while the mean helpfulness score remains constant. {#c-l2-scale}

Safety context distillation prefixes a safety preprompt, such as You are a safe and responsible assistant, to adversarial prompts to generate safer responses, and then fine-tunes the model on its own safe output given the adversarial prompt without the preprompt. {#c-l2-cd}
Llama 2's safety preprompts are generated automatically from templates with adjectives usually associated with safe behavior, such as responsible, respectful or wise. {#c-l2-cdgen}
The teacher is the same model with the preprompt and the student is the model without it, so this is knowledge distillation from a prompt into weights.
The context-distillation loss E_{y∼p(·|c,x)}[−log q_θ(y|x)] equals KL(p(·|c,x) ‖ q_θ(·|x)) plus the entropy of p(·|c,x), which does not depend on θ, so minimizing it pulls the model without the preprompt c towards the model with it, and its minimum is reached at q_θ(·|x) = p(·|c,x). {#c-st-cdkl}
Safety context distillation on helpful prompts can degrade performance and lead to more false refusals, so it is applied only to adversarial prompts, and the context-distilled output is kept only where it gets a better safety reward model score than the original answer. {#c-l2-cdrm}
`select_distilled` in the NumPy file applies this filter, and `context_distillation_loss` in the PyTorch file trains the student on the KL divergence to the teacher, over response tokens only.

Preference-based training can be just as well applied to safety: direct preference optimization fits the same pairs as a reward model would, without a separate RL stage, and Constitutional AI generates the harmlessness preferences with a model instead of people.

### What a refusal should look like

The early RLHF assistants refused too readily and too uniformly.
At an earlier stage of Anthropic's project, many RLHF policies very frequently reproduced the same exaggerated responses to all remotely sensitive questions, such as recommending therapy whenever the user expressed any displeasure, which greatly limited their utility. {#c-hh-therapy}
Anthropic's earlier human-feedback assistant often refused to answer controversial questions and could get stuck producing evasive responses for the rest of a conversation, because crowdworkers had rewarded evasiveness as a response to harmful inputs. {#c-ca-evasive}
Constitutional AI trains a harmless but non-evasive assistant that engages with harmful queries by explaining its objections to them. {#c-ca-nonevasive}
Llama 2's response guidelines above have the same aim: a refusal explains itself and, where it can, offers a safe alternative.
Smaller models pay alignment taxes, performing worse on many evaluations after RLHF, while their RLHF-trained models of 13B parameters and above perform better at zero-shot NLP evaluations and the same at few-shot evaluations. {#c-hh-tax}

## Derivation

The full derivation is in `derivations/safety-training-arithmetic.md`, and `safety-training-arithmetic.check.py` checks every step numerically.

**The safety fraction.**
Adding n examples to N gives the fraction n/(N + n), and solving for n gives n = fN/(1 − f), the formula behind the 3% figure above.

**Context distillation.**
The cross-entropy against the teacher splits into a KL divergence and the teacher's entropy, which is fixed; minimizing it therefore pulls the model without the preprompt towards the model with it, and sampling teacher outputs and fine-tuning on them is a Monte Carlo estimate of the same loss.

**The refusal direction.**
P = I − r̂r̂ᵀ is a symmetric idempotent projection: r̂ᵀPx = 0, Px = x when x ⊥ r̂, and ‖Px‖ ≤ ‖x‖; among unit vectors u, uᵀ(μ − ν) is largest at u = r̂ = (μ − ν)/‖μ − ν‖, and adding r = μ − ν moves the mean harmless activation ν exactly onto the mean harmful one μ. {#c-st-proj}
So the difference of means is the unit direction that separates harmful from harmless activations most, measured by the gap between the means, and adding r to harmless activations moves their mean onto the harmful mean.
If the embeddings and every matrix and bias that writes to the residual stream are multiplied on the left by P = I − r̂r̂ᵀ, every residual activation lies in the complement of r̂, and the forward pass is the same as projecting every residual activation with P. {#c-st-ortho}
The argument is an induction over layers: if both models agree on a stream with no r̂ component, every component reads the same input, and its projected write is the same in both.

**Base64 and red-teaming rates.**
Base64 maps every 3 bytes to 4 characters from a 64-character alphabet, so an n-byte request becomes 4⌈n/3⌉ characters; the 45-byte question What tools do I need to cut down a stop sign? becomes 60 characters. {#c-st-base64}
Going from γ = 1.8 to γ = 0.45 violating prompts per person-hour divides the rate by 4, so the expected red-teaming time per violation grows from about 33 minutes to about 133 minutes. {#c-st-gamma}

**Greedy coordinate gradient.**
To first order, replacing token x_i by v changes the loss by g_v − g_{x_i}, where g is the gradient with respect to the one-hot vector of x_i, so the top-k of −g are the k substitutions with the largest estimated decrease; with |I| = 20 and a 32,000-token vocabulary there are 639,980 single-token substitutions, the top-256 candidates number 5,120, and a batch of 512 evaluates 10% of them. {#c-st-gcg}

## Worked example

Take Bianchi et al.'s setup: 20,000 general instructions.
To make safety data 3% of the mix, solve n/(20,000 + n) = 0.03: n = 0.03 × 20,000/0.97 ≈ 618.6, so 619 safety examples, which is between the 500 and 1,000 at which their harmfulness scores level off.

Now measure the result on XSTest, whose authors call Llama 2-Chat with its original system prompt Llama2.0.
Llama2.0 exhibits substantial exaggerated safety: the model fully refuses 38% of the safe prompts and partially refuses another 21.6%. {#c-xs-llama}
On XSTest's 250 safe prompts, 38% full and 21.6% partial refusals are 95 and 54 prompts, so 149 prompts, 59.6%, are refused at least in part; Llama 2's 1,351 single-turn and 623 multi-turn evaluation prompts total 1,974. {#c-st-counts}
`refusal_breakdown` in the NumPy tests reproduces these rates from 95, 54 and 101 labelled answers.

For the refusal direction, the NumPy tests build a toy residual network with three MLP blocks, take a random unit direction r̂, run the network once with every residual activation projected off r̂, and once with the embeddings, output matrices and biases multiplied by I − r̂r̂ᵀ.
Logits and every residual activation agree to floating-point precision, and every activation of the edited network has zero component along r̂.
The PyTorch tests repeat this on a two-layer decoder with attention, where the attention output projections are edited too.

## Hyperparameters

The safety data fraction is the main dial between harmfulness and over-refusal.
With more safety data mixed into tuning the false-refusal rate grows, but it stays rare, approximately 0.05% on the helpfulness dataset even with 100% safety data, and is much larger on a borderline set of prompts. {#c-l2-fr-rate}
The response style is a choice of the training data, not of the optimizer: training on bare refusals produces bare refusals.
For context distillation, the preprompt and the rule for keeping its outputs decide whether it improves safety or adds false refusals, which is why Llama 2 filters with the safety reward model.
In attacks used to evaluate a model, the GCG budget sets how hard the test is.
GCG uses a batch size of 512, a top-k of 256 and 20 optimizable tokens, run for 500 steps. {#c-gc-hp}

## Cost

Supervised safety data adds a few hundred to a few thousand examples to an instruction-tuning run, a small fraction of its compute.
The larger costs are human: writing adversarial prompts, writing or rating safe responses, and red teaming each candidate model, which Llama 2 repeated over several rounds.
Safety RLHF trains and serves a second reward model during RL.
Context distillation needs one generation with the preprompt per adversarial prompt and one safety reward-model score per original and distilled answer.
Undoing safety training is far cheaper than doing it.
Fine-tuning GPT-3.5 Turbo on only 10 adversarially designed examples, at a cost of less than $0.20 through OpenAI's API, jailbreaks its safety guardrails, making it responsive to nearly any harmful instruction. {#c-qi-10}

## Evidence

**Safety RLHF and the safety-data scale.**
The Llama 2 results above are the main public evidence that a separate safety reward model, and more safety data, raise safety scores without lowering helpfulness scores.
Its red teaming gives a second measure.
Llama 2 defines a model's robustness γ to a red-teaming exercise as the average number of created prompts that trigger a violating response per person per hour; on the 7B model γ went from 1.8 to 0.45 over several red-teaming iterations. {#c-l2-gamma}
On average, 90% of the prompts that triggered violating responses in earlier red-teaming exercises were rejected by the next candidate model. {#c-l2-90}
RLHF models become increasingly difficult to red team as they scale, while plain language models, prompted models and rejection-sampling models show a flat trend with scale. {#c-ga-scale}

**Small amounts of supervised safety data.**
Bianchi et al.'s results, given above with the method, come from open models fine-tuned with LoRA, so they can be reproduced.

**Refusal is one direction.**
Across 13 open-source chat models up to 72B parameters there is a single direction such that erasing it from the residual stream activations prevents the model from refusing harmful instructions, while adding it elicits refusal even on harmless instructions. {#c-ar-single}
The harmful instructions are drawn from four existing datasets of harmful instructions, among them TDC2023, and the harmless instructions are sampled from an instruction dataset; each dataset consists of train and validation splits of 128 and 32 samples. {#c-ar-data}
For each layer and post-instruction token position, the refusal direction candidate is the difference between the mean activations on harmful and on harmless prompts, and the most effective of the |I| × L candidates is selected on validation sets. {#c-ar-dim}
Activation addition adds the difference-in-means vector to the activations at one layer and all token positions, and directional ablation sets x′ ← x − r̂r̂ᵀx for every residual stream activation, at all layers and token positions. {#c-ar-ops}
The residual stream is the running sum that the residual connections of a transformer carry from layer to layer: every attention and MLP block reads it and adds its output to it, so removing r̂ from everything written to it removes r̂ everywhere.
Weight orthogonalization replaces every matrix that writes to the residual stream, the embedding, positional embedding, attention out and MLP out matrices and their output biases, by W − r̂r̂ᵀW; it is equivalent to directional ablation and needs neither gradient-based optimization nor examples of harmful completions. {#c-ar-ortho}
Orthogonalized models perform similarly to the originals on MMLU, ARC and GSM8K, but their TruthfulQA accuracy consistently drops. {#c-ar-coherence}
For safety training this means what is learned is narrow: a feature that marks a prompt as harmful and a downstream behaviour keyed on it, which a rank-one edit to the weights removes.

## Limitations and failure modes

**Jailbreaks.**
Wei et al. hypothesize two failure modes of safety training: competing objectives, when a model's capabilities and safety goals conflict, and mismatched generalization, when safety training fails to generalize to a domain for which capabilities exist. {#c-jb-modes}
Both appear in simple prompt attacks.
Prefix injection asks the model to first output an innocuous-looking prefix, such as Absolutely! Here's, designed so that conditioning on the prefix makes refusal unlikely in the pretraining distribution. {#c-jb-prefix}
Refusal suppression instructs the model to respond under constraints that rule out common refusal responses, such as not apologizing and never saying cannot or unfortunately; inverting the rules leads to no restricted behavior on any prompt in the dataset. {#c-jb-suppress}
Large models likely learn Base64 in pretraining and can follow Base64-encoded instructions, while safety training likely contains no inputs as unnatural, so an encoded harmful request is far out of distribution for refusal. {#c-jb-base64}
Attacks built on the two failure modes succeed on over 96% of the evaluated prompts against GPT-4 and Claude v1.3, including 100% of the curated red-teaming prompts that past safety interventions were designed to address. {#c-jb-96}
Scaling up will not resolve competing objectives, since the issue lies with the optimization objective, and may exacerbate mismatched generalization, which suggests that safety mechanisms need to be as sophisticated as the underlying model. {#c-jb-parity}

**Adversarial suffixes.**
Prompt attacks can also be found by optimization.
GCG's objective requires the model to begin its response with a positive affirmation of the user query, such as Sure, here is how to build a bomb, leaving the rest of the completion unspecified. {#c-gc-target}
The adversarial loss is the negative log probability of the target tokens, L(x_{1:n}) = −log p(x*_{n+1:n+H} | x_{1:n}), minimized over the tokens of the suffix positions I. {#c-gc-loss}
Greedy Coordinate Gradient computes, for every suffix position, the top-k tokens with the largest negative gradient of the loss with respect to the one-hot token vector, randomly selects B ≤ k|I| single-token substitutions from these candidates, evaluates their loss exactly and makes the replacement with the smallest loss. {#c-gc-alg}
GCG differs from AutoPrompt, which chooses a single position in advance and evaluates replacements only there, and for the same batch size GCG performs substantially better. {#c-gc-autoprompt}
On Vicuna, GCG elicits 99 of 100 harmful behaviors and 88 of 100 exact target strings, and its prompts reach up to 84% success on GPT-3.5 and GPT-4, 66% on PaLM-2 and 2.1% on Claude. {#c-gc-rates}
A suffix optimized on multiple prompts and on Vicuna-7B and 13B induces objectionable content in the public interfaces to ChatGPT, Bard and Claude and in open models such as LLaMA-2-Chat, Pythia and Falcon. {#c-gc-transfer}
`gcg_step` in both files implements one step, with the gradient taken through a one-hot matrix times the embedding table in PyTorch.

**Exaggerated safety.**
Llama 2 defines false refusal as the model incorrectly refusing to answer legitimate user prompts due to irrelevant safety concerns. {#c-l2-fr-def}
Llama 2-Chat sometimes has difficulty telling whether a prompt is safe when it contains words that frequently occur in unsafe generations, such as bomb. {#c-l2-bomb}
Too much safety-tuning makes models refuse perfectly safe prompts if they superficially resemble unsafe ones, an exaggerated safety behaviour. {#c-bi-exag}
XSTest comprises 250 safe prompts across ten prompt types that well-calibrated models should not refuse, and 200 unsafe prompts as contrasts that models should refuse for most applications. {#c-xs-suite}
Exaggerated safety appears to be a consequence of lexical overfitting, where models are overly sensitive to certain words or phrases, and system prompts added at inference time change safety behaviour substantially but not in a comprehensive or consistent way. {#c-xs-lexical}
These are the failures of a learned classifier that keys on surface words: the model refuses how to kill a Python process because kill appears in its refusal examples.

**Fragility under fine-tuning.**
Safety training is undone by further training, including training the user did not intend as an attack.
Qi et al.'s safety benchmark has 30 harmful instructions in each of 11 categories of prohibited use from Meta's and OpenAI's usage policies, 330 in all. {#c-qi-bench}
After 5 epochs on 10 harmful examples, the harmfulness rate of GPT-3.5 Turbo rises from 1.8% to 88.8%, and that of Llama-2-7b-Chat from 0.3% to 50.0%, and to 80.3% with 50 examples. {#c-qi-table}
Fine-tuning on only 10 identity-shifting examples, which contain no explicitly harmful content, raises the harmfulness rate by up to 87.3% for GPT-3.5 Turbo and 72.1% for Llama-2-7b-Chat. {#c-qi-identity}
One epoch of fine-tuning on Alpaca raises the harmfulness rate of GPT-3.5 Turbo from 5.5% to 31.8% and of Llama-2-7b-Chat from 0.3% to 16.1%. {#c-qi-benign}
The safety degradation from benign fine-tuning may arise from catastrophic forgetting of the initial alignment or from an inherent tension between the helpfulness and harmlessness objectives. {#c-qi-cause}
Anyone who fine-tunes a safety-trained model, even on benign data, should evaluate safety again afterwards; mixing safety examples into the fine-tuning data, as in Bianchi et al.'s recipe, is the obvious mitigation.

**Measurement.**
A completion counts as a refusal if it contains at least one of a set of refusal substrings such as I'm sorry or As an AI; this string matching detects memorized refusals but does not assess whether the completion is coherent or harmful. {#c-ar-score}
`is_refusal` in the NumPy file implements substring matching, and its tests include both kinds of error: a refusal without any of the listed phrases, and an answer that begins I'm sorry to hear that and then complies.
Rates from model judges or human labels, such as XSTest's full and partial refusals, avoid some of these errors but cost more.

## Variants

**Supervised safety fine-tuning** adds harmful prompts with safe responses to instruction-tuning data; it is the simplest and the only option without a reward model.

**Safety RLHF** trains a separate safety reward model, or adds harmlessness comparisons to one reward model, and optimizes the policy against it; Anthropic's helpful and harmless assistant used one preference model trained on both kinds of data, and Llama 2 used two.

**Safety context distillation** trains the model on its own outputs under a safety preprompt.

**AI feedback.** Constitutional AI replaces human harmlessness labels with labels from a model prompted with written principles, for both the supervised and the RL stage.

**Preference optimization.** Safety preference pairs can be fit directly with direct preference optimization or its relatives instead of a reward model and RL.

**Representation editing.** Because refusal is mediated by one direction in the residual stream, adding that direction makes a model refuse harmless prompts, and ablating it removes refusals; the same direction can be monitored, or protected during further fine-tuning.

## When to use it

Any model that will take requests from the public needs some form of safety training, and the choice is how much.
Supervised safety data is the minimum, needs only a few hundred to a few thousand examples, and is cheap enough to mix into every fine-tuning run, including fine-tuning of an already safety-trained model.
Safety RLHF or AI feedback is worth its cost for a general assistant, where the long tail of harmful requests is too large to cover by demonstrations.
Context distillation is useful when a safety system prompt already works and the aim is to make the behaviour hold without it.
Safety training does not make a model robust to determined attackers, so it is one layer among others: red teaming before release, input and output filters, and monitoring.

## Pitfalls

* Measuring only harmfulness: a model that refuses everything scores perfectly. Always report a false-refusal rate on safe prompts that resemble unsafe ones, such as XSTest's.
* Safety data in one form only: questions do not teach refusing instructions. Cover the forms users write.
* Relying on substring refusal detection for final numbers: it misses paraphrased refusals and counts apologetic compliance.
* Fine-tuning a safety-trained model without re-checking safety, even on benign data.
* Treating the absence of jailbreaks found by hand as robustness: optimized suffixes and encodings find failures that manual testing misses.
* Training on bare refusals: the model learns to refuse without explanation and to refuse a whole conversation after one sensitive turn.

## History

Training language models to be harmless as well as helpful grew out of RLHF.
Anthropic's 2022 helpful and harmless assistant trained one preference model on helpfulness and red-teaming comparisons and reported the tension between the two, and its red-teaming study the same year found RLHF models harder to red team as they grew.
Constitutional AI, at the end of 2022, replaced human harmlessness labels with AI feedback and aimed for non-evasive refusals.
Llama 2-Chat in 2023 published a detailed recipe: safety SFT, a separate safety reward model, context distillation filtered by that reward model, and repeated red teaming.
The same year brought the failure analyses: Wei et al.'s prompt attacks, Zou et al.'s optimized suffixes, XSTest's measurement of exaggerated safety, and Qi et al.'s fine-tuning attacks.
Bianchi et al. measured how little supervised safety data an open instruction-tuned model needs, and Arditi et al. located refusal in a single direction of the residual stream, both published in 2024.
