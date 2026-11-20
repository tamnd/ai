---
tag: NEW-instruction-tuning
kind: method
slug: instruction-tuning
name: Instruction tuning
aliases: [instruction finetuning, instruction fine-tuning, multitask prompted training]
area: post-training/instruction-tuning
level: core
summary: >-
  Fine-tune a pretrained language model by maximum likelihood on many tasks phrased as natural-language instructions, so that it follows instructions for tasks it was never trained on.
purpose: turning a pretrained language model, which continues text, into one that carries out a task described in words, zero-shot, on tasks outside its fine-tuning data
uses: [decoder-only-transformer, t5, cross-entropy@m2]
related_to: [reinforcement-learning-from-human-feedback, direct-preference-optimization, few-shot-in-context-learning, chain-of-thought-prompting, emergent-abilities, lora, mmlu]
hyperparameters:
  - {name: task mixture and mixing rates, meaning: "which datasets go in and how often each is sampled; examples-proportional mixing with a cap K keeps the largest datasets from crowding out the rest", typical: "FLAN: 62 datasets, at most 30k examples each, cap 3k; T0: proportional, with datasets over 500,000 examples counted as 500,000 divided by their number of templates"}
  - {name: templates per dataset, meaning: "how many natural-language phrasings of each task are used; more phrasings make the model less sensitive to wording", typical: "10 in FLAN; 11.7 on average in P3"}
  - {name: prompt formats, meaning: "the share of zero-shot, few-shot and chain-of-thought examples, and whether input-inverted tasks are added", typical: "a mixture of all three; a few per cent of few-shot templates already helps zero-shot"}
  - {name: learning rate and steps, meaning: "the optimizer settings; instruction tuning runs far fewer steps than pretraining", typical: "Adafactor 3e-5 for 30k steps (FLAN); 1e-3 with dropout 0.1 (T0, an encoder-decoder); AdamW 1e-5 to 1e-6 for 15 epochs (LIMA)"}
  - {name: loss mask and averaging, meaning: "whether the loss covers the response only or the whole sequence, and whether it is averaged over tokens or over sequences", typical: "response only, token average"}
cost:
  train: "ordinary supervised fine-tuning: one forward and backward pass per token of the mixture, on the order of 0.2% of the pretraining compute for Flan-PaLM 540B"
  inference: "unchanged; the instruction-tuned model has the same architecture and size as the base model"
history:
  year: 2021
  people: [Jason Wei, Maarten Bosma, Vincent Y. Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, Quoc V. Le, Victor Sanh, Albert Webson, Colin Raffel, Stephen H. Bach, Alexander M. Rush, Swaroop Mishra, Daniel Khashabi, Chitta Baral, Hannaneh Hajishirzi]
  note: "Mishra et al. (posted April 2021, ACL 2022) fine-tuned BART on Natural Instructions; Wei et al. (posted September 2021, ICLR 2022) named instruction tuning with FLAN 137B; Sanh et al. (posted October 2021, ICLR 2022) trained T0 on the Public Pool of Prompts. Chung et al. and Longpre et al. scaled the recipe to 1,836 tasks in the Flan 2022 collection, Ouyang et al. made it the first stage of InstructGPT, and Self-Instruct and LIMA asked how much and what kind of instruction data is needed."
prereqs: [decoder-only-transformer, t5, few-shot-in-context-learning]
same_as:
  wikidata: null
  wikipedia: null
  arxiv: ["2109.01652", "2110.08207", "2104.08773", "2210.11416", "2301.13688", "2203.02155", "2212.10560", "2305.11206"]
  al: null
sources:
  - {paper: "Wei, Bosma, Zhao, Guu, Yu, Lester, Du, Dai and Le (2022)", title: "Finetuned Language Models Are Zero-Shot Learners", locator: "arXiv 2109.01652v5, pp. 1, 3, 4, 6 to 8", note: "the definition, the FLAN mixture and templates, mixing, training settings, the comparison with GPT-3, the ablations over clusters, scale and instructions"}
  - {paper: "Sanh, Webson, Raffel, Bach and others (2022)", title: "Multitask Prompted Training Enables Zero-Shot Task Generalization", locator: "arXiv 2110.08207v3, pp. 1, 3, 5 to 7", note: "P3, T0's mixing rule and training settings, the comparison with GPT-3, the effect of more prompts"}
  - {paper: "Mishra, Khashabi, Baral and Hajishirzi (2022)", title: "Cross-Task Generalization via Natural Language Crowdsourcing Instructions", locator: "pp. 1 and 2 of the ACL 2022 PDF", note: "the Natural Instructions dataset and schema, the 19% gain, more seen tasks"}
  - {paper: "Chung, Hou, Longpre, Zoph and others (2022)", title: "Scaling Instruction-Finetuned Language Models", locator: "arXiv 2210.11416v5, pp. 1 to 6", note: "the Flan 2022 mixtures, compute, task and model scaling, chain-of-thought data, Flan-T5"}
  - {paper: "Longpre, Hou, Vu, Webson and others (2023)", title: "The Flan Collection: Designing Data and Methods for Effective Instruction Tuning", locator: "arXiv 2301.13688v2, pp. 1, 2, 6 and 7", note: "mixed prompt settings, input inversion, few-shot templates, Flan-T5 as a starting checkpoint"}
  - {paper: "Ouyang, Wu, Jiang, Almeida and others (2022)", title: "Training language models to follow instructions with human feedback", locator: "arXiv 2203.02155v1, pp. 7, 8, 12 and 13", note: "the SFT stage of InstructGPT and its comparison with FLAN- and T0-tuned GPT-3"}
  - {paper: "Wang, Kordi, Mishra, Liu, Smith, Khashabi and Hajishirzi (2023)", title: "Self-Instruct: Aligning Language Models with Self-Generated Instructions", locator: "pp. 1 to 5 of the ACL 2023 PDF", note: "the bootstrapping pipeline, the filter, the data quality, the results on Super-NaturalInstructions"}
  - {paper: "Zhou, Liu, Xu, Iyer and others (2023)", title: "LIMA: Less Is More for Alignment", locator: "arXiv 2305.11206v1, pp. 1, 2, 4, 6 and 8", note: "the 1,000-example training set, the end-of-turn token, the human study, the data ablations, the Superficial Alignment Hypothesis"}
---

Instruction tuning fine-tunes a pretrained language model on many tasks, each written as a natural-language instruction followed by the desired output.
The training is ordinary maximum likelihood on the output tokens; what is new is the data: dozens to thousands of datasets, each rephrased by templates into instructions, mixed so that no single dataset dominates.
The aim is generalization across tasks: after training on summarization, translation and question answering phrased as instructions, the model should follow an instruction for natural language inference, which it never saw.
Mishra et al., Wei et al. (FLAN) and Sanh et al. (T0) introduced the idea in 2021, Chung et al. scaled it to 1,836 tasks, and it became the first stage of chat-model training, the supervised fine-tuning (SFT) step before reinforcement learning from human feedback.
How much data it needs is still debated: LIMA reached strong human ratings with 1,000 examples.

## In plain words

A pretrained language model is good at continuing text, which is not the same as doing what it is asked.
Ask it "Translate to French: the cat sleeps" and it may continue with another English sentence, because that is what text on the web often does.
Instruction tuning shows the model many examples of an instruction followed by its answer, across many kinds of tasks, and trains it to produce the answer.
After enough variety, the model learns the general habit "read the instruction, then do it", and the habit carries over to new kinds of instructions.
The model is only graded on the answer part; the instruction is there to read, not to imitate.

## Setting

Wei et al. call finetuning language models on a collection of datasets described via instructions instruction tuning, and report that it substantially improves zero-shot performance on unseen tasks. {#c-fl-define}
Their motivation is that a large pretrained model, prompted with a task it was not trained on, does far worse zero-shot than few-shot; the question is whether training on instructions closes that gap.
Sanh et al. ask the question from the other side: whether the zero-shot generalization of large language models, hypothesized to be a consequence of implicit multitask learning in pretraining, can instead be directly induced by explicit multitask learning. {#c-t0-question}

Mishra et al. built the first dataset for the setting.
Natural Instructions is a dataset of 61 distinct tasks, their human-authored instructions and 193k task instances, with the instructions obtained from the crowdsourcing instructions used to create existing NLP datasets and mapped to a unified schema. {#c-ni-dataset}
The unified schema covers the most important elements of task descriptions, such as definition, constraints, positive and negative examples. {#c-ni-schema}

The same setting, with different data, is the first stage of chat-model training.
The supervised fine-tuning stage of InstructGPT fine-tunes GPT-3 on labeler demonstrations with supervised learning, for 16 epochs with a cosine learning rate decay and residual dropout of 0.2, selecting the final model by its reward-model score on the validation set. {#c-ig-sft}
The InstructGPT SFT dataset contains about 13k training prompts, from the API and written by labelers. {#c-ig-13k}
Academic instruction tuning draws its tasks from existing NLP datasets; SFT for assistants draws them from what users ask.
The loss and the code are the same.

Formally, there are $M$ datasets $\mathcal D_1, \dots, \mathcal D_M$ of input-output pairs, a set of templates for each, and a pretrained model $p_\theta$.
A template turns a pair $(u, v)$ into a prompt $x$ in words and a target $y$; the model is trained to maximize $\log p_\theta(y \mid x)$ on examples sampled from the mixture, and is then evaluated zero-shot on datasets, or whole task clusters, held out of the mixture.

## The method

**The task mixture.**
Wei et al. aggregate 62 text datasets that are publicly available on Tensorflow Datasets into a single mixture, each categorized into one of twelve task clusters of the same task type. {#c-fl-62}
For each dataset they manually compose ten unique templates that use natural language instructions to describe the task. {#c-fl-templates}
T0 used crowdsourced prompts instead: the Public Pool of Prompts (P3) contains 2073 prompts for 177 datasets, 11.7 prompts per dataset on average. {#c-t0-p3}
Flan 2022 merged these and more: Chung et al. scale to 1,836 finetuning tasks from 473 datasets and 146 task categories by combining four mixtures: Muffin (80 tasks), T0-SF (193 tasks), NIV2 (1554 tasks) and CoT. {#c-fc-mixtures}
Templates can also run a task backwards.
Input inversion gives a model the answer y of a supervised task and trains it to generate the question x, an easy method to enrich the task variety given a limited set of data sources. {#c-co-inversion-def}

**Mixing.**
Datasets differ in size by orders of magnitude, so sampling uniformly over all examples would mostly train on the largest few.
FLAN limits the number of training examples per dataset to 30k and follows the examples-proportional mixing scheme with a mixing rate maximum of 3k, meaning that a dataset receives no additional sampling weight for examples in excess of 3,000. {#c-fl-mix}
With $e_m$ examples in dataset $m$ and cap $K$,

$$ r_m = \frac{\min(e_m, K)}{\sum_{n=1}^{M} \min(e_n, K)} . $$ {#eq-mixing}

Under examples-proportional mixing with a cap K, dataset m with e_m examples is sampled with rate min(e_m, K) divided by the sum of min(e_n, K) over all datasets, so every dataset with at least K examples gets the same rate. {#c-it-rates}
T0 combines and shuffles all examples from all training datasets, which is equivalent to sampling in proportion to dataset size, but treats any dataset with over 500'000 examples as having 500'000 divided by its number of templates. {#c-t0-mix}

**Generated and curated data.**
The tasks need not come from existing datasets.
Self-Instruct starts from 175 seed tasks, each with one instruction and one instance, and at each step samples 8 task instructions from the pool as in-context examples for generating new ones. {#c-si-pipeline}
Self-Instruct adds a new instruction to the task pool only when its ROUGE-L similarity with every existing instruction is less than 0.7. {#c-si-rouge}
Run on GPT-3, the iterative Self-Instruct process yields about 52k instructions paired with about 82K instance inputs and target outputs. {#c-si-52k}
LIMA went the other way, toward a small hand-curated set.
The LIMA training set has 750 top questions and answers from community forums such as Stack Exchange and wikiHow, sampled for quality and diversity, and 250 hand-written prompts and responses in a uniform assistant style. {#c-li-data}

**Encoding and the loss.**
For a decoder-only model each example becomes one sequence, prompt then response then an end token, and position $t$ predicts token $t + 1$.
The loss mask $m_t$ is 1 where the predicted token belongs to the response or is the end token, and 0 on the prompt and on padding:

$$ \mathcal L(\theta) = \frac{\sum_{b,t} m_{b,t}\,\big(-\log p_\theta(s_{b,t+1} \mid s_{b,\le t})\big)}{\sum_{b,t} m_{b,t}} . $$ {#eq-masked-loss}

The end token matters, because it is what the model learns to emit to stop.
LIMA marks the end of each utterance with a special end-of-turn token, which halts generation like EOS but avoids conflation with any other meaning the pretrained model gave the existing EOS token. {#c-li-eot}
An encoder-decoder model needs no mask.
T0 is an encoder-decoder model that generates the target through standard maximum likelihood training; unlike decoder-only language models such as GPT-3, it is never trained to generate the input. {#c-t0-encdec}

**Packing.**
Short examples waste most of a fixed-length row as padding, so several are packed into one row.
Each then needs its own segment: attention must not cross from one example into another, and positions should restart at 0.
`pack`, `segment_attention_mask` and `segment_positions` in the NumPy file build these, and the PyTorch test `test_packed_rows_equal_separate_examples` checks that a packed row gives the same logits as the examples run one at a time.

## Derivation

**Why the mask gives the conditional likelihood.**
An autoregressive model factorizes $p(x, y) = p(x)\,p(y \mid x)$ token by token:

$$ -\log p(x, y) = \underbrace{-\sum_{t \le P} \log p(x_t \mid x_{<t})}_{-\log p(x)} \; \underbrace{- \sum_{r \le R} \log p(y_r \mid x, y_{<r})}_{-\log p(y \mid x)} . $$ {#eq-chain}

For an autoregressive model, the sum of per-token negative log-likelihoods over the response positions of a prompt-response sequence equals −log p(response | prompt), and the loss over all positions adds −log p(prompt). {#c-it-chain}
So the response-only loss is maximum likelihood for the conditional distribution the model will be used for, and the full-sequence loss adds a language-modelling term on the prompts.
The NumPy test `test_response_loss_is_conditional_log_likelihood` checks the identity by enumerating every response on a bigram model.

**The gradient.**
With logits $z_t$ and $\ell_t = -\log \mathrm{softmax}(z_t)_{y_t}$, the gradient of eq-masked-loss is

$$ \frac{\partial \mathcal L}{\partial z_t} = \frac{m_t\,\big(\mathrm{softmax}(z_t) - \mathbf e_{y_t}\big)}{\sum_s m_s} . $$ {#eq-grad}

The gradient of the masked mean token loss with respect to the logits at position t is m_t (softmax(z_t) − onehot(y_t)) divided by the number of unmasked tokens, which is zero at every prompt position. {#c-it-grad}
The prompt still shapes the gradient: it enters through attention into the hidden states at response positions, and so into every weight.
What the mask removes is only the pressure to predict the prompt.

**Token or sequence averaging.**
Eq-masked-loss divides by the number of response tokens in the whole batch.
Averaging the loss over all response tokens in a batch weights each sequence in proportion to its number of response tokens, while averaging per sequence first weights them equally; for responses of 10 and 90 tokens the weights are 0.1 and 0.9 against 0.5 and 0.5. {#c-it-weights}
Neither is wrong, but they train toward different mixtures, and switching between them silently changes the effective data weighting.

**What the cap does.**
After S sampled examples, every dataset smaller than the cap is passed over S divided by the sum of min(e_n, K) times, the same number for all of them, and a dataset with e_m > K is passed over fewer times, S K / (e_m times that sum). {#c-it-epochs}
So the cap equalizes the number of epochs of all small datasets, and large datasets are subsampled rather than repeated.
T0's rule, which counts a dataset with more than 500,000 examples as 500,000 divided by its number of templates, is not monotone in dataset size: with 10 templates a dataset of 500,000 examples counts as 500,000 and one of 500,001 as 50,000. {#c-it-t0rule}
The derivation `instruction-tuning-arithmetic` gives the steps of all of these, and its check verifies them by simulation, enumeration and finite differences.

## Worked example

**Capped mixing.**
Take three datasets with 1,000, 3,000 and 100,000 examples and cap $K = 3{,}000$.

| | dataset 1 | dataset 2 | dataset 3 |
|---|---|---|---|
| examples $e_m$ | 1,000 | 3,000 | 100,000 |
| $\min(e_m, K)$ | 1,000 | 3,000 | 3,000 |
| rate, no cap | 0.0096 | 0.0288 | 0.9615 |
| rate, cap 3,000 | 1/7 | 3/7 | 3/7 |
| passes after 70,000 samples | 10 | 10 | 0.3 |

For datasets of 1,000, 3,000 and 100,000 examples and K = 3,000, the rates are 1/7, 3/7 and 3/7 instead of about 1.0%, 2.9% and 96.2% without the cap; after 70,000 samples the three datasets are passed over 10, 10 and 0.3 times. {#c-it-example}
Without the cap, the smallest dataset would be seen less than once in 70,000 samples; with it, ten times.
`test_worked_example_rates_and_passes` checks the table.

**The masked loss.**
Take the prompt tokens $(5, 6, 7)$, the response $(8, 9)$ and end token 1.

| position $t$ | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| input | 5 | 6 | 7 | 8 | 9 |
| target | 6 | 7 | 8 | 9 | 1 |
| mask | 0 | 0 | 1 | 1 | 1 |
| $p_\theta(\text{target})$ | 0.01 | 0.02 | 0.5 | 0.25 | 0.8 |
| $-\log p_\theta$ | 4.605 | 3.912 | 0.693 | 1.386 | 0.223 |

The first masked position is the last prompt token, which predicts the first response token.
For the prompt (5, 6, 7), the response (8, 9) and end token 1, the loss mask is (0, 0, 1, 1, 1); if the model assigns the three response targets probabilities 0.5, 0.25 and 0.8, the summed loss is ln 10 ≈ 2.302585 and the mean 0.767528, and adding prompt targets of probability 0.01 and 0.02 raises the full-sequence sum to 10.819778, mean 2.163956. {#c-it-we-loss}
The sum $\ln 10$ is $-\log p_\theta(8, 9, \text{end} \mid 5, 6, 7)$, as eq-chain says.
The full-sequence loss is dominated by the two prompt tokens, which a model fine-tuned on that loss would spend effort learning to predict.
`test_encode_masks_only_the_response` checks the encoding.

## Hyperparameters

FLAN is finetuned for 30k gradient steps with a batch size of 8,192 tokens, the Adafactor optimizer at a learning rate of 3e-5, input and target lengths of 1024 and 256, and packing of several examples into one sequence separated by an EOS token. {#c-fl-hparams}
T0 truncates inputs and targets to 1024 and 256 tokens and uses Adafactor with a learning rate of 1e-3 and a dropout rate of 0.1, following standard practice for fine-tuning T5. {#c-t0-hparams}
LIMA is fine-tuned for 15 epochs with AdamW (beta1 = 0.9, beta2 = 0.95, weight decay 0.1), with no warmup and a learning rate decaying linearly from 1e-5 to 1e-6. {#c-li-hparams}
The learning rates differ by two orders of magnitude because the models differ: T5's pretraining used a large constant rate with Adafactor, while decoder-only models are usually fine-tuned well below their pretraining peak.
The number of templates, the cap and the share of each prompt format are data choices, and the Evidence section gives what is known about each.

## Cost

Instruction tuning costs a small fraction of pretraining.
FLAN's instruction tuning takes around 60 hours on a TPUv3 with 128 cores. {#c-fl-time}
Chung et al. use 0.2% of the pre-training compute to instruction-finetune Flan-PaLM 540B, approximately 512 v4 TPU chips for 37 hours. {#c-fc-compute}
For Flan-PaLM 540B, Table 2 of Chung et al. lists 2.5E+24 pre-training FLOPs and 5.6E+21 finetuning FLOPs. {#c-fc-table}
The PaLM pre-training data consists of 780B tokens, while instruction finetuning uses 1.4B tokens, 0.2% of the pre-training tokens. {#c-fc-tokens}
The 6ND rule gives 6 × 540e9 × 780e9 ≈ 2.53e24 FLOPs for PaLM 540B pre-training, matching Table 2, and 6 × 540e9 × 1.4e9 ≈ 4.5e21 for finetuning on 1.4B tokens, about 0.8 of the 5.6e21 in Table 2; both the token ratio and the FLOP ratio are about 0.2%. {#c-it-flops}
The remaining gap is plausibly padding, which the rule does not count, but the paper does not say how the FLOPs were counted.
The masked loss saves no compute: the forward and backward passes cover the prompt tokens whether or not they are in the loss.
Inference is unchanged.

## Evidence

**FLAN and T0.**
FLAN, a 137B model instruction tuned on over 60 NLP datasets, surpasses zero-shot 175B GPT-3 on 20 of 25 datasets evaluated. {#c-fl-gpt3}
With NLI, closed-book QA and commonsense reasoning held out, average performance on the held-out clusters improves as instruction-tuning clusters are added, except for sentiment analysis, and does not appear to saturate at seven clusters. {#c-fl-clusters}
Finetuning without templates, or with only the dataset name in place of instructions, performs substantially worse than FLAN on held-out clusters, indicating that training with instructions is crucial for zero-shot performance on unseen tasks. {#c-fl-instructions}
Few-shot exemplars at inference improve instruction-tuned FLAN on all task clusters over zero-shot FLAN, and lower the standard deviation across templates. {#c-fl-fewshot}
T0 matches or exceeds GPT-3 on 9 of 11 held-out datasets despite being about 16 times smaller, and improves over a large baseline language model on 13 of 14 BIG-bench tasks. {#c-t0-gpt3}
Multitask prompted training gives significant gains over the T5+LM baseline, the same model without it, on all held-out datasets. {#c-t0-baseline}
Training on more prompts per dataset consistently improves the median and decreases the variability of performance on held-out tasks; prompts from more datasets generally improve the median but do not consistently decrease the variability. {#c-t0-prompts}

**Natural Instructions.**
BART fine-tuned with instructions achieves a 19% gain in cross-task generalization over the same model without instructions. {#c-ni-19}
Models using instructions generalize better to unseen tasks when they observe more tasks in training. {#c-ni-more-tasks}

**Scaling: Flan 2022.**
Flan-PaLM 540B, instruction-finetuned on 1.8K tasks, outperforms PaLM 540B by 9.4% on average and reaches 75.2% on five-shot MMLU. {#c-fc-gain}
Multi-task instruction finetuning improves the normalized average on held-out benchmarks by 9.4% to 15.5% over no finetuning at 8B, 62B and 540B parameters; more tasks help, but most of the improvement comes from using up to 282 tasks. {#c-fc-tasks}
Flan-T5 checkpoints from 80M to 11B parameters outperform T5, with Flan-T5 11B improving on T5 11B by double digits and beating PaLM 62B on some challenging BIG-Bench tasks. {#c-fc-flant5}

**Design choices: the Flan Collection.**
Training with mixed prompt settings, zero-shot, few-shot and chain-of-thought together, yields stronger (2%+) performance in all settings. {#c-co-mixed}
Adding as little as 5% few-shot training templates can dramatically improve zero-shot performance, and adding 10%+ of zero-shot data improves few-shot performance too. {#c-co-fewshot}
Enriching the mixture with input-inverted tasks is not beneficial for held-in performance but strongly beneficial for held-out performance. {#c-co-inversion}
Flan-T5 converges faster and at a higher performance than T5 in single-task finetuning, suggesting instruction-tuned models are a more computationally efficient starting checkpoint. {#c-co-checkpoint}

**Instruction tuning as the first stage of an assistant.**
InstructGPT's SFT models overfit on validation loss after 1 epoch, yet training for more epochs helps both the reward-model score and human preference ratings. {#c-ig-overfit}
Ouyang et al. compare InstructGPT with 175B GPT-3 baselines fine-tuned on the FLAN and T0 datasets, and find that these perform better than GPT-3. {#c-ig-flan-base}
On the API prompt distribution, outputs of the 175B InstructGPT model were preferred over the FLAN-tuned model 78 ±4% of the time and over the T0-tuned model 79 ±4% of the time. {#c-ig-flan}
Academic task mixtures, then, teach instruction following, but not the instructions people send to an assistant; the Limitations section returns to why.

**How much data.**
Fine-tuning the vanilla base model on its own Self-Instruct data gives a 33% absolute improvement on Super-NaturalInstructions, on par with InstructGPT-001, and leaves a 5% absolute gap to InstructGPT-001 in human evaluation on novel instructions. {#c-si-33}
In a controlled human study, responses from LIMA are equivalent or strictly preferred to GPT-4 in 43% of cases, to Bard in 58% and to DaVinci003 in 65%. {#c-li-pref}
In the LIMA ablations, scaling up input diversity and output quality have measurable positive effects, while scaling up quantity alone might not. {#c-li-diversity}
7B models trained on exponentially increasing amounts of quality-filtered Stack Exchange data plateau in performance despite an up to 16-fold increase in data size. {#c-li-quantity}

## Limitations and failure modes

**Scale.**
Instruction tuning substantially improves held-out performance for the two models on the order of 100B parameters, but for 8B and smaller models it hurts performance on held-out tasks. {#c-fl-small}
Wei et al. suggest that for small models, learning the roughly 40 tasks used during instruction tuning fills the entire model capacity. {#c-fl-capacity}
The pattern differs from emergent abilities, which appear only above some model size: here an ability is present at every size and it is the effect of the training method that changes sign with scale.
Later mixtures with more tasks and better templates helped small models too, as the Flan-T5 results show, so the threshold belongs to the FLAN recipe rather than to instruction tuning.

**Tasks that are already language modelling.**
Instruction tuning does not improve performance for many language modeling tasks, such as commonsense reasoning or coreference resolution tasks formulated as sentence completions. {#c-fl-lm-tasks}
When the task is already phrased as continuing a sentence, the instruction adds little.

**Losing abilities that are not in the mixture.**
Instruction finetuning without chain-of-thought data severely degrades performance on chain-of-thought evaluations, and adding just nine chain-of-thought datasets to the mixture improves performance on all evaluations. {#c-fc-cot}
What the mixture leaves out can be unlearned.

**Diminishing returns in tasks.**
One explanation Chung et al. give for the small gain after 282 tasks is that most of the gains come from the model learning to better express knowledge it already knows from pretraining. {#c-fc-express}
Longpre et al. caution against concluding that scaling beyond 1800 tasks would bring increased returns without also paying attention to task diversity and quality. {#c-co-diversity}

**Noisy generated data.**
In a sample of Self-Instruct data, 92% of instructions describe a valid task, 79% of inputs suit their instruction, 58% of outputs are correct and acceptable, and 54% of examples have all fields valid. {#c-si-quality}
Training on such data still helped, but nearly half of it is wrong somewhere, and it inherits the generating model's errors.

**The wrong distribution of tasks.**
Ouyang et al. explain the gap partly by usage: classification and QA are only about 18% of what API customers use the models for, while open-ended generation and brainstorming make up about 57% of their prompt dataset. {#c-ig-reason}
A mixture built from NLP benchmarks is mostly short-answer tasks, and the model learns that style.

**Safety.**
When malicious intent is implicit, LIMA is more likely to provide unsafe responses. {#c-li-unsafe}
Instruction tuning teaches the model to comply; it does not by itself teach when not to, which is one reason it is followed by preference training.

**What it can teach.**
Zhou et al. state the Superficial Alignment Hypothesis: a model's knowledge and capabilities are learnt almost entirely during pretraining, while alignment teaches it which subdistribution of formats should be used when interacting with users. {#c-li-hypothesis}
If the hypothesis holds, instruction tuning cannot add knowledge the base model lacks, and data that pushes it to state facts it does not know may teach it to make them up; the hallucination page discusses this.

## Variants

- **Academic task mixtures** (Natural Instructions, FLAN, P3 and T0, Super-NaturalInstructions, Flan 2022): existing NLP datasets rewritten by templates, evaluated zero-shot on held-out tasks.
- **Supervised fine-tuning on demonstrations** (InstructGPT SFT): human-written responses to real user prompts, the first stage before RLHF or DPO.
- **Model-generated instructions** (Self-Instruct and its descendants): a strong model writes the instructions and responses, and filters remove near-duplicates and invalid examples.
- **Small curated sets** (LIMA): a thousand carefully chosen examples, betting on the Superficial Alignment Hypothesis.
- **Chain-of-thought instruction tuning** (part of Flan 2022): targets that include the reasoning steps, so the model keeps the ability to reason step by step.
- **Parameter-efficient instruction tuning**: the same data and loss with LoRA or another adapter instead of full fine-tuning.

## When to use it

Use instruction tuning whenever a pretrained model has to follow instructions rather than continue text, which is nearly always for a deployed model.
For an assistant, collect or generate prompts that resemble real use and responses in the style wanted; quality and diversity matter more than count, and a few thousand good examples are a reasonable start.
For a general zero-shot model, mix many tasks with several templates each, cap the large datasets, and include few-shot and chain-of-thought formats.
Start from an instruction-tuned checkpoint such as Flan-T5 when fine-tuning on a single task, since it converges faster.
Follow it with preference training (RLHF or DPO) when the model must also learn which of several acceptable answers people prefer, or when to refuse.
Do not expect it to add knowledge.

## Pitfalls

> [!pitfall]
> **An off-by-one loss mask.** Position $t$ predicts token $t + 1$, so the mask starts at the last prompt token, not the first response token, and must cover the end token. Starting one position late never trains the first response token; leaving out the end token gives a model that never stops. `test_encode_masks_only_the_response` and `test_encode_and_collate` pin the mask.

> [!pitfall]
> **Changing the averaging without noticing.** A token-level mean over the batch and a per-sequence mean weight long and short responses differently, by 0.9 against 0.5 in the example above, and gradient accumulation over micro-batches can turn one into the other. `test_token_versus_sequence_weights` and `test_token_versus_sequence` check both forms.

> [!pitfall]
> **Packing without a segment mask.** If packed examples share a causal mask, each one attends to the end of the one before it, and the model learns to use context it will not have at inference. Restart the positions too. `test_packed_rows_equal_separate_examples` shows the logits changing without the mask.

> [!pitfall]
> **Reusing an end token that already means something.** If the pretrained model used EOS as a document separator, teaching it to stop with EOS mixes two meanings; LIMA added a separate end-of-turn token for this reason.

> [!pitfall]
> **Training on the prompt by accident.** Without the mask the loss is dominated by the prompt tokens, as in the worked example, and the model spends capacity learning to predict instructions. `test_response_only_training_does_not_learn_the_prompt` contrasts the two losses on a toy task: both learn the responses, and only the full-sequence model learns to predict the prompt.

## History

Mishra, Khashabi, Baral and Hajishirzi posted Natural Instructions in April 2021, published at ACL 2022.
Wei and colleagues at Google posted FLAN in September 2021 and published it at ICLR 2022, giving the method its name.
Sanh and colleagues of the BigScience workshop posted T0 in October 2021, also published at ICLR 2022.
Ouyang and colleagues at OpenAI used supervised fine-tuning on demonstrations as the first stage of InstructGPT in 2022.
Chung and colleagues at Google scaled the recipe in "Scaling Instruction-Finetuned Language Models" in October 2022, and Longpre and colleagues released the data and the ablations as the Flan Collection in January 2023.
Wang and colleagues published Self-Instruct at ACL 2023, and Zhou and colleagues at Meta posted LIMA in May 2023.
