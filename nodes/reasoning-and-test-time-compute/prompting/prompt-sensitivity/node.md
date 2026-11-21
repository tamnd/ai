---
tag: NEW-prompt-sensitivity
kind: finding
slug: prompt-sensitivity
name: Prompt sensitivity of language models
aliases: [prompt brittleness, prompt format sensitivity, few-shot order sensitivity, demonstration order effects, prompt variance, FormatSpread, contextual calibration]
area: reasoning-and-test-time-compute/prompting
level: core
summary: >-
  The accuracy of a language model on a task can move by tens of points when the prompt format, wording, examples or their order change, while the meaning of the prompt stays the same.
related_to: [few-shot-in-context-learning, chain-of-thought-prompting, system-prompt, language-model-mathematical-reasoning, expected-calibration-error]
history:
  year: 2021
  note: "Zhao et al. (February 2021) reported format, example and order effects in GPT-3 and proposed contextual calibration; Lu et al. (April 2021) studied example order; Webson and Pavlick (September 2021) tested irrelevant and misleading prompts; Zhou et al. (November 2022) and Yang et al. (September 2023) searched for prompts with language models; Sclar et al. (October 2023, ICLR 2024) measured format spreads; Mizrahi et al. (January 2024) measured paraphrased instructions across 20 models."
prereqs: [few-shot-in-context-learning]
same_as:
  arxiv: ["2102.09690", "2104.08786", "2310.11324", "2109.01247", "2401.00595", "2211.01910", "2309.03409"]
sources:
  - {paper: "Zhao et al. (2021)", title: "Calibrate Before Use: Improving Few-Shot Performance of Language Models", locator: "arXiv 2102.09690v2, pp. 1 to 6", note: "format, example and order effects in GPT-3, the three biases and contextual calibration"}
  - {paper: "Lu et al. (2022)", title: "Fantastically Ordered Prompts and Where to Find Them: Overcoming Few-Shot Prompt Order Sensitivity", locator: "arXiv 2104.08786v2, pp. 1 to 5", note: "order sensitivity across model sizes, transfer between models and entropy-based order selection"}
  - {paper: "Sclar et al. (2024)", title: "Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design or: How I learned to start worrying about prompt formatting", locator: "ICLR 2024, arXiv 2310.11324v2, pp. 1 to 5", note: "format spreads, reversed model comparisons and FormatSpread"}
  - {paper: "Webson and Pavlick (2022)", title: "Do Prompt-Based Models Really Understand the Meaning of Their Prompts?", locator: "arXiv 2109.01247v2, pp. 1, 2 and 5", note: "irrelevant and misleading templates and the role of target words"}
  - {paper: "Mizrahi et al. (2024)", title: "State of What Art? A Call for Multi-Prompt LLM Evaluation", locator: "arXiv 2401.00595v3, pp. 1, 6 and 7", note: "paraphrased instruction templates across 20 models and multi-prompt metrics"}
  - {paper: "Zhou et al. (2023)", title: "Large Language Models Are Human-Level Prompt Engineers", locator: "ICLR 2023, arXiv 2211.01910v2, pp. 1 and 7", note: "automatic prompt search and the zero-shot chain-of-thought trigger"}
  - {paper: "Yang et al. (2024)", title: "Large Language Models as Optimizers", locator: "arXiv 2309.03409v3, pp. 1 and 13", note: "prompt optimization with a language model and the effect of similar instructions"}
claim: >-
  The accuracy of a language model on a task depends on choices that leave the meaning of the prompt unchanged, such as the format, the separators and casing, the wording of the instruction, the choice of demonstrations and their order, by up to tens of accuracy points and by several points in a typical task; the effect persists with larger models, more demonstrations and instruction tuning, differs between models so that comparisons can reverse, and is partly explained by biases toward labels that are frequent, recent or common in pretraining text.
first_reported: {work: zhao-2021-calibrate-before-use, note: "GPT-3 models from 2.7B to 175B parameters varied from near chance to 93.4% on sentiment classification with only the order of four examples changed"}
replications:
  - {work: lu-2022-fantastically-ordered-prompts, note: "GPT-2 and GPT-3 models from 0.1B to 175B parameters on eleven text classification tasks, with every order of the demonstrations"}
  - {work: sclar-2024-quantifying-sensitivity-spurious-features, note: "LLaMA-2, Falcon and GPT-3.5 on 53 tasks with formats that differ only in separators, spacing and casing"}
  - {work: mizrahi-2024-state-of-what-art, note: "20 models on 39 tasks with automatically paraphrased and verified instruction templates"}
  - {work: webson-2022-prompt-based-models-understand, note: "T0, GPT-3 and other models on natural language inference with irrelevant and misleading templates"}
scale_range:
  - {range: "GPT-2 1.5B and GPT-3 2.7B to 175B parameters on classification and fact retrieval", work: zhao-2021-calibrate-before-use}
  - {range: "LLaMA-2 7B to 70B, Falcon-7B and GPT-3.5 on 53 tasks", work: sclar-2024-quantifying-sensitivity-spurious-features}
  - {range: "20 models from Flan-T5 to GPT-3.5 on 39 tasks from 3 benchmarks", work: mizrahi-2024-state-of-what-art}
failures:
  - {work: zhao-2021-calibrate-before-use, note: "contextual calibration raised accuracy and reduced its variance across prompts in most settings, so part of the sensitivity can be removed without changing the prompt"}
  - {work: lu-2022-fantastically-ordered-prompts, note: "choosing orders by the entropy of predictions on a probing set recovered part of the gap between good and bad orders"}
  - {work: zhou-2023-large-language-models-human-level, note: "searching over instructions turns the sensitivity into a gain, matching or beating human-written instructions on most tasks"}
  - {work: yang-2024-large-language-models-optimizers, note: "optimized instructions beat human-written ones by up to 8% on GSM8K and 50% on Big-Bench Hard"}
explanations:
  - {name: "label and position biases", summary: "Models favour labels that are frequent in the prompt, near its end, or common in pretraining text, so the format, examples and order shift the decision threshold.", sources: [zhao-2021-calibrate-before-use, lu-2022-fantastically-ordered-prompts]}
  - {name: "surface-form matching", summary: "Models respond to the surface of the prompt, especially the answer words, more than to the meaning of its instruction.", sources: [webson-2022-prompt-based-models-understand]}
  - {name: "model-specific format effects", summary: "Which formats and wordings work well depends on the model, so a prompt tuned for one model is not neutral for another.", sources: [sclar-2024-quantifying-sensitivity-spurious-features, mizrahi-2024-state-of-what-art]}
---

Prompt sensitivity is the dependence of a language model's accuracy on parts of the prompt that should not matter.
Two prompts that a person would read as the same request, differing in a colon, a capital letter, the order of four examples or the wording of an instruction, can give accuracies tens of points apart.
The effect was first measured in GPT-3's few-shot classification and has since been found across model families, sizes, instruction-tuned models and tasks.
It matters most for evaluation: a single prompt gives one point from a wide distribution, and two models compared on the same prompt can come out in either order.

## In plain words

A few-shot prompt has a format (how each example is written), a set of examples and an order.
A zero-shot prompt has an instruction and a format for the answer.
None of these change what is being asked.

Yet a model asked to label the sentiment of a review can be right about half the time with one order of four examples and right nine times out of ten with another.
Changing only the separator between a field name and its text, or the capitals of the field name, can move the accuracy of LLaMA-2 by many points.
A model can learn a reasoning task about as fast from a misleading instruction as from a correct one.

Part of the explanation is simple.
The model tends to answer with the label it saw most in the examples, the label of the last example, or a label word that is common in its training text.
Each choice of format and order tilts these tendencies, and so tilts the answers.
Estimating the tilt from a meaningless input such as "N/A" and dividing it out removes much of the effect.

## The finding

Few-shot learning with GPT-3 can be unstable: the choice of prompt format, training examples, and even the order of the training examples can cause accuracy to vary, and varying the permutation of the training examples can cause accuracy to go from near chance (54.3%) to 93.4%. {#c-zh-abs}
The effect is not confined to one prompt type or model family.
Several widely used open-source LLMs are extremely sensitive to subtle changes in prompt formatting in few-shot settings, with performance differences of up to 76 accuracy points when evaluated using LLaMA-2-13B, and sensitivity remains even when increasing model size, the number of few-shot examples, or performing instruction tuning. {#c-sc-abs}
An analysis of single-prompt evaluations across 6.5M instances, involving 20 different LLMs and 39 tasks from 3 benchmarks, finds that different instruction templates lead to very different performance, both absolute and relative. {#c-mz-abs}

Sensitivity also appears where the instruction should matter most.
With over 30 prompt templates manually written for natural language inference, models learn just as fast with many prompts that are intentionally irrelevant or even pathologically misleading as they do with instructively good prompts, and such patterns hold even for models as large as 175 billion parameters and for instruction-tuned models. {#c-wb-abs}

Zero-shot instructions behave the same way.
With the PaLM 2-L scorer on the GSM8K test set, “Let’s think step by step.” achieves accuracy 71.8, “Let’s solve the problem together.” has accuracy 60.5, while the accuracy of “Let’s work together to solve this problem step by step.” is only 49.4, although it is the semantic combination of the two. {#c-opro-sim}

The derivation calibration-orderings-and-format-spread puts these effects into simple quantities.
With two classes, contextual calibration predicts the positive class exactly when its probability exceeds the content-free probability a, so it replaces the threshold 0.5 by a and subtracts logit(a) from every log-odds; a content-free probability of 0.62 is a shift of 0.4895. {#c-d-thresh}
k demonstrations have k! orders, 24 for k = 4 and 40,320 for k = 8, and four demonstrations of two balanced classes have C(4, 2) = 6 label patterns. {#c-d-orders}
If accuracy under a random format is normal with standard deviation σ, the expected range over m formats is 2σ E[max of m standard normals]: 1.1284σ, 3.0775σ, 3.8953σ and 5.7965σ for m = 2, 10, 24 and 320, so a spread over 10 formats is expected to be 0.531 of the spread over 320. {#c-d-range}
An accuracy of 0.5 on 1,000 items has a standard error of 1.58 points, so ten equally good formats measured on separate samples would differ by 4.87 points on average from noise alone; on shared items two formats that disagree on 10% of items have a difference with standard error 1.0 point. {#c-d-noise}
If a format moves each of two models by a normal amount of standard deviation σ with correlation ρ, the weaker model comes out ahead with probability Φ(−Δ / (σ sqrt(2(1 − ρ)))): 0.3886, 0.2398 and 0.0786 for gaps Δ of 2, 5 and 10 points at σ = 5 and ρ = 0, and 0.3446 for Δ = 2 at ρ = 0.5. {#c-d-flip}
Choosing the highest-scoring of m equally good prompts on a development set of 100 items at true accuracy 0.7 overstates the winner's accuracy by 4.583 E[max of m standard normals] points on average: 7.05 for m = 10 and 8.93 for m = 24. {#c-d-select}

## Evidence

**Examples, order and format in GPT-3.**
Zhao et al. run their experiments on three sizes of GPT-3 (2.7B, 13B, and 175B parameters) as well as GPT-2 (1.5B parameters), accessing GPT-3 using the OpenAI API. {#c-zh-models}
GPT-3's accuracy depends highly on both selection and permutation of training examples: with a fixed prompt format and different random sets of training examples, accuracy is evaluated for all possible permutations. {#c-zh-vary}
Keeping the training examples and permutations fixed while varying the prompt format over an additional 14 manually designed formats for SST-2, some of the formats are better than others on average, but all of the formats still suffer from high variance across different training sets. {#c-zh-format}
The variance arises because language models are biased towards outputting answers that are frequent in the prompt (majority label bias), towards the end of the prompt (recency bias), and common in the pre-training data (common token bias). {#c-zh-biases}
On the validation set for 4-shot LAMA with GPT-3 2.7B, 50.2% of the model predictions are a repeat of one of the four training answers, while the correct repeat rate is 24.7%. {#c-zh-maj}
On DBPedia, a balanced 14-way topic classification dataset, GPT-3 predicts the book class 11 times more often than the artist class, and there is a moderate correlation (r = 0.67) between the frequency of a DBPedia label name and the rate at which GPT-3 predicts its class. {#c-zh-tok}
For one sentiment prompt with 67% accuracy, if the classification threshold could be set optimally (p(Positive) = 0.68), the classifier would be highly accurate (94% on the validation set). {#c-zh-thresh}
That last result is the basis of the fix.
Contextual calibration obtains the label probabilities for a content-free input such as N/A, sets W = diag(p_cf)^-1 and b to the all-zero vector, and makes test predictions by taking the argmax of W p + b. {#c-zh-cal}
Contextual calibration consistently improves GPT-3 and GPT-2's accuracy (up to 30.0% absolute) across different choices of the prompt format and examples, and makes the accuracy more stable across different prompts. {#c-zh-gain}
Contextual calibration reduces the variance considerably in a majority of cases, and it does not increase variance by much in the remaining cases. {#c-zh-var}

**Order across models.**
The order in which few-shot samples are provided matters; the effect is present across model sizes, is not related to a specific subset of samples, and a good permutation for one model is not transferable to another. {#c-lu-abs}
Some permutations have comparable performance (over 85% accuracy) to supervised training for sentiment classification, while others perform close to random (around 50%); increasing the model size somewhat addresses this, but the problem is still present for some text classification tasks for models with billions of parameters. {#c-lu-fig}
A specific permutation's performance may drop from 88.7% to 51.6% by changing the underlying model from GPT2-XL (1.5B) to GPT2-Large (0.8B). {#c-lu-transfer}
Increasing the number of training samples leads to increases in performance, but a high level of variance remains, even with a large number of samples, and can even increase. {#c-lu-more}
The majority of failing prompts suffer from highly unbalanced predicted label distributions, and although calibration leads to much higher performance, the variance remains high. {#c-lu-degen}
Lu et al. rank candidate orderings on an artificial probing set generated by the language model itself, using Global Entropy (GlobalE), the entropy of the predicted labels, and Local Entropy (LocalE), the average prediction entropy per data point. {#c-lu-method}
Selecting orderings by entropy statistics on the probing set yields a 13% relative improvement for GPT-family models across eleven different established text classification tasks. {#c-lu-gain}

**Formats that differ only in separators, spacing and casing.**
Accuracy varies by up to 76 points for LLaMA-2-13B between equivalent formats, and by about 10 accuracy points on average across 50+ tasks and several models. {#c-sc-avg}
Across tasks there is a median spread of 7.5 accuracy points across choices in the model and the number of few-shot examples; 20% of tasks consistently result in a spread of at least 15 accuracy points for all LLaMA-2 settings and at least 9 points for all Falcon settings, and several tasks have a spread over 70 accuracy points. {#c-sc-median}
Because this analysis uses only 10 randomly sampled formats, it represents a lower bound of the true spreads for each task. {#c-sc-lower}
The derivation gives the expected size of that bound: under normal format effects, a range over 10 formats is about half of the range over 320.
Comparison trends between models are often reversed just by choosing different formats: LLaMA-2-13B and -70B reverse trend by at least d = 0.02 with probability 0.141, and LLaMA-2-7B and Falcon-7B with probability 0.140. {#c-sc-reverse}
Format performance only weakly correlates between models, which puts into question the methodological validity of comparing models with an arbitrarily chosen, fixed prompt format. {#c-sc-weak}
FormatSpread explores the space of formats as a bandit problem with Thompson sampling, modelling each format's accuracy as a Beta distribution since each data point evaluation is a Bernoulli trial. {#c-sc-thompson}
FormatSpread does not require access to the model weights, and with GPT3.5 it finds a spread up to 56 accuracy points with a median spread of 6.4 accuracy points across 320 formats and 53 tasks at a cost of under 10 USD on average per task. {#c-sc-gpt}

**Instructions that should not work.**
Webson and Pavlick evaluate language models on NLI in zero-shot and few-shot settings using more than 30 manually written templates and 13 sets of LM target words, for a total of over 390 prompts. {#c-wb-setup}
Models are much more sensitive to the choice of the LM target words as opposed to the meaning of the instruction templates. {#c-wb-target}
Models trained with null templates perform far worse than all other categories of templates. {#c-wb-null}

**Paraphrased instructions across many models.**
Overall, 15 tasks out of 25 have instruction template paraphrases with negative Kendall's τ, indicating mostly disagreeing LLM rankings. {#c-mz-tau}
The performance of Alpaca-13B with the original instruction templates outperformed its average performance over paraphrases by more than one standard deviation in 7 out of 10 LMentry tasks. {#c-mz-alpaca}
Major differences in performance can occur even for very similar paraphrase pairs: the Flan-T5-large model demonstrated an average performance degradation of 28% when changing the word excludes to lacks, while the Flan-T5-XL model showed an average performance improvement of 46% on that same edit. {#c-mz-edit}

**Searching over prompts.**
If wording moves accuracy, searching over wordings can raise it.
Automatically generated instructions achieve better or comparable performance to the instructions generated by human annotators on 24/24 Instruction Induction tasks and 17/21 curated BIG-Bench tasks. {#c-ape-abs}
Zero-Shot-CoT, prepending Let's think step by step to the response, improves the zero-shot performance of InstructGPT on MultiArith from 17.7 to 78.7 and on GSM8K from 10.4 to 40.7. {#c-ape-kojima}
APE produces the prompt Let's work this out in a step by step way to be sure we have the right answer, which further improves performance from 78.7 to 82.0 on MultiArith and from 40.7 to 43.0 on GSM8K. {#c-ape-cot}
With a variety of LLMs, the prompts optimized by OPRO outperform human-designed prompts by up to 8% on GSM8K, and by up to 50% on Big-Bench Hard tasks. {#c-opro-abs}
With PaLM 2-L as the scorer, zero-shot GSM8K test accuracy is 71.8 with Let's think step by step, 58.8 with Let's work this out in a step by step way to be sure we have the right answer, 34.0 with the empty string, and 80.2 with Take a deep breath and work on this problem step-by-step. {#c-opro-table}
The instruction found by Zhou et al. with InstructGPT scores below the plain "Let's think step by step" with PaLM 2-L, a sign that a searched prompt is tuned to the model it was found for.

**Implementations.**
The numpy implementation scores a binary classifier whose log-odds are a fixed semantic signal plus biases for the majority label, the most recent labels and the format, and evaluates every order of a set of demonstrations.
Raw accuracy varies across orders; contextual calibration, which subtracts the log-odds of a content-free input, removes the order spread entirely in this model because the bias is a constant shift, as the derivation predicts.
The PyTorch implementation trains a small transformer to classify a query from eight labelled demonstrations of a random linear rule.
With learned position embeddings it reaches 0.84 accuracy, and 8.2% of its predictions change when the demonstrations are reordered; without position embeddings it is invariant to order by construction, at the same accuracy.
In both versions, making all demonstrations positive moves the query logit by about 2.5 to 2.9, a learned majority-label bias.

## Scope and conditions

The early results concern few-shot classification with label words, scored by the probability of each label or by exact match.
Later studies extend them to instruction-following models, generation tasks and chain-of-thought prompts, though most measurements still use tasks with a short answer.

The size of the effect depends on the model and the task.
Sclar et al. found the spread with instruction-tuned models and with more examples, but its size varies widely between tasks, from a few points to more than seventy.
Larger models reduced order sensitivity on some tasks in Lu et al. without removing it.
Mizrahi et al. paraphrased instructions automatically and filtered them for correctness, so their spreads come from wordings a person would accept as equivalent.

Some spread is noise.
On a test set of a few hundred items, the derivation's noise floor is a few points, and any spread below it says nothing about the prompt.
The median spreads reported above exceed it, and the spreads of tens of points are far outside it.

## Explanations

**Label and position biases.**
Zhao et al. traced the instability to the model favouring frequent, recent and common labels.
Calibration against a content-free input addresses all three at once, which fits the explanation for classification tasks: in two-class problems it is a shift of the decision threshold, and a shift is what a constant bias produces.
Lu et al. found that bad orders often produce predictions concentrated on one label, the same symptom, though calibration did not remove all of their variance.

**Surface-form matching.**
Webson and Pavlick read their results as evidence that models use the surface form of a prompt, especially its answer words, more than its meaning.
Irrelevant and misleading instructions working nearly as well as correct ones fits this; null templates doing worse suggests the model still uses some structure in the prompt.

**Model-specific format effects.**
Weak correlations between models in which formats work well, and paraphrase rankings that disagree between models, mean that no single format is neutral.
The source of these differences is not established; training data with different formatting conventions is one candidate.

## Implications

For evaluation, a score from one prompt is one sample.
Future research would benefit from reporting a performance spread over a sufficient sample of plausible formats, instead of simply reporting the formatting used and its performance. {#c-sc-report}
Mizrahi et al. define the maximum performance (MaxP) of a model on a task as the maximum performance across all instruction templates, and the average performance (AvgP) as the mean over all instruction templates. {#c-mz-metrics}
The two metrics separate the question of what a model can do with a good prompt from what it does with a typical one.
The derivation's flip probability gives a rule of thumb: when the gap between two models is not large beside the spread over prompts, the order of the comparison depends on the prompt.

For use, the results favour calibrating or searching rather than guessing.
Contextual calibration costs one extra query per prompt, and prompt search raises scores at the price of selection optimism: the reported score of the winning prompt on the data used to pick it overstates its accuracy on new data.

In-context learning is where these effects were measured first, and they bound what a few demonstrations reveal about a model.
Chain-of-thought prompting is itself a prompt choice, and its trigger phrases vary in effect by tens of points.
A system prompt is subject to the same sensitivity, since it is part of the prompt.
Reported results for the mathematical reasoning ability of language models depend on the prompt as well as on changes to the problems.
Expected calibration error measures a different quantity, the match between confidence and accuracy, and contextual calibration improves accuracy without targeting it.

## Open questions

Whether sensitivity shrinks with newer instruction-tuned and reasoning models is not settled; the studies above stop at GPT-3.5 and models of similar date.
How to choose the set of plausible prompts over which to report a spread is open, since the spread grows with the size and variety of that set.
Why format effects differ between models, and whether training can make a model invariant to formatting without losing accuracy, are also open.

## History

Zhao and coauthors reported the instability of GPT-3 few-shot learning and proposed contextual calibration in February 2021, and Lu and coauthors studied the order of demonstrations in April 2021.
Webson and Pavlick asked whether models understand their prompts in September 2021.
Automatic prompt search with language models followed, with Zhou and coauthors in November 2022 and Yang and coauthors in September 2023.
Sclar and coauthors measured format spreads in October 2023, and Mizrahi and coauthors called for multi-prompt evaluation in January 2024.

## References
