---
tag: NEW-continued-pretraining
kind: method
slug: continued-pretraining
name: Continued pretraining
aliases: [continual pretraining, continual pre-training, further pretraining, domain-adaptive pretraining, DAPT, task-adaptive pretraining, TAPT, language adaptation, long-context continued pretraining]
area: post-training/continued-pretraining-and-domain-adaptation
level: core
summary: >-
  Training a finished pretrained model further with the same next-token objective on new data, to add a domain, a language or a longer context without starting over.
purpose: giving an existing pretrained model new knowledge, a new domain or language, or a longer context window for a fraction of the cost of pretraining again, while keeping what it already does well
uses: [decoder-only-transformer, cosine-learning-rate-schedule, learning-rate-warmup, rotary-position-embedding, byte-pair-encoding, masked-language-modelling]
related_to: [fine-tuning, lora, instruction-tuning, bert, llama, neural-scaling-laws]
hyperparameters:
  - {name: peak learning rate, meaning: "the maximum of the re-warmed schedule", typical: "the pretraining peak or a fraction of it; lower forgets less, higher adapts more; Code Llama kept Llama 2's, long-context runs use 1e-5 to 2e-5"}
  - {name: schedule, meaning: "how the learning rate moves over the continued run", typical: "linear warmup of up to about 1% of the tokens into a cosine decay to a tenth or less of the peak; or, for a model that may be continued again, an infinite schedule with a constant phase"}
  - {name: replay fraction, meaning: "share of each batch drawn from the original pretraining data or a stand-in for it", typical: "1% to 5% for a mild shift, 25% or more for a new language, 2% to 15% general text in domain mixtures"}
  - {name: token budget, meaning: "how many tokens the continued run sees", typical: "a single pass over a domain corpus for BERT-sized models; 50B to 1T tokens for large models; several passes over a small domain corpus"}
  - {name: data mixture, meaning: "the weights of domain, general, code and instruction-style sources", typical: "mostly domain data with general text and code mixed in; a final phase upweighting high-quality and QA-style data"}
  - {name: vocabulary extension, meaning: "how many tokens to add and how to initialize their rows", typical: "none for domains in the same script; thousands to tens of thousands for a new script, with new rows set to the mean of the old ones"}
  - {name: RoPE base, meaning: "the base of the rotary frequencies when extending the context", typical: "raised from 10,000 to 500,000 or 1,000,000 together with longer training sequences"}
cost:
  train: "proportional to the new tokens only, about 6 × parameters × tokens; with replay a fraction r of that compute goes to old data; a long-context stage costs more per token because attention grows with sequence length"
  inference: "the same as the original model, except that an extended vocabulary changes the size of the embedding and output layers and a longer context costs more memory per request"
history:
  year: 2019
  people: [Jinhyuk Lee, Suchin Gururangan, Yu Gu, Kshitij Gupta, Adam Ibrahim, Baptiste Rozière, Wenhan Xiong, Yiming Cui, Zhangir Azerbayev, Jupinder Parmar, Zitong Yang]
  note: "BioBERT (2019) continued BERT on biomedical text, and Gururangan et al. (2020) named domain-adaptive and task-adaptive pretraining. Gu et al. (2021) argued that a domain with enough text should be pretrained from scratch. In 2023 Code Llama, Llemma, Chinese LLaMA and long-context Llama 2 made continued pretraining the standard way to specialize a large model, and Gupta et al. and Ibrahim et al. (2023 to 2024) worked out re-warming and replay."
prereqs: [decoder-only-transformer, cosine-learning-rate-schedule, fine-tuning]
same_as:
  wikidata: null
  wikipedia: null
  arxiv: ["2004.10964", "2308.04014", "2403.08763", "2407.07263", "2308.12950", "2309.16039", "2310.10631", "2402.03300", "2304.08177", "1901.08746", "2007.15779", "2309.09530", "2409.07431", "1612.00796", "2405.09673"]
  al: null
sources:
  - {paper: "Ibrahim, Thérien, Gupta, Richter and others (2024)", title: "Simple and Scalable Strategies to Continually Pre-train Large Language Models", locator: "arXiv 2403.08763v4, pp. 1, 3, 4, 7, 14 and 20", note: "re-warming, re-decaying and compute-equivalent replay against re-training, rules of thumb, infinite schedules"}
  - {paper: "Gupta, Thérien, Ibrahim, Richter and others (2023)", title: "Continual Pre-Training of Large Language Models: How to (re)warm your model?", locator: "arXiv 2308.04014v2, pp. 1, 3 and 6", note: "re-warming, warmup length and the choice of checkpoint"}
  - {paper: "Gururangan, Marasović, Swayamdipta, Lo and others (2020)", title: "Don't Stop Pretraining: Adapt Language Models to Domains and Tasks", locator: "arXiv 2004.10964v3, pp. 1 and 3 to 6", note: "domain-adaptive and task-adaptive pretraining"}
  - {paper: "Parmar, Satheesh, Patwary, Shoeybi and Catanzaro (2024)", title: "Reuse, Don't Retrain: A Recipe for Continued Pretraining of Language Models", locator: "arXiv 2407.07263v1, pp. 1, 3, 4 and 7", note: "two data distributions, the schedule and the switch point"}
  - {paper: "Rozière, Gehring, Gloeckle, Sootla and others (2023)", title: "Code Llama: Open Foundation Models for Code", locator: "arXiv 2308.12950v3, pp. 3 to 6 and 13", note: "code continued pretraining, long context fine-tuning, rehearsal, comparison with training from scratch"}
  - {paper: "Xiong, Liu, Molybog, Zhang and others (2023)", title: "Effective Long-Context Scaling of Foundation Models", locator: "arXiv 2309.16039v3, pp. 1, 3, 7 and 10", note: "long-context continued pretraining, the RoPE base, data quality, FLOPs saved"}
  - {paper: "Azerbayev, Schoelkopf, Paster, Dos Santos and others (2023)", title: "Llemma: An Open Language Model For Mathematics", locator: "arXiv 2310.10631v3, pp. 2 and 3", note: "Proof-Pile-2, the mixture, schedule and RoPE contraction"}
  - {paper: "Shao, Wang, Zhu, Xu and others (2024)", title: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", locator: "arXiv 2402.03300v3, pp. 1, 2 and 7", note: "math continued pretraining from a code model and its mixture"}
  - {paper: "Cui, Yang and Yao (2023)", title: "Efficient and Effective Text Encoding for Chinese LLaMA and Alpaca", locator: "arXiv 2304.08177v3, pp. 3, 4 and 6", note: "vocabulary extension and two-stage training"}
  - {paper: "Lee, Yoon, Kim, Kim and others (2019)", title: "BioBERT: a pre-trained biomedical language representation model for biomedical text mining", locator: "arXiv 1901.08746v4, pp. 1 to 3", note: "continued pretraining of BERT on biomedical text"}
  - {paper: "Gu, Tinn, Cheng, Lucas and others (2021)", title: "Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing", locator: "arXiv 2007.15779v6, pp. 1, 2 and 5", note: "pretraining from scratch against continual pretraining, vocabulary fragmentation"}
  - {paper: "Cheng, Huang and Wei (2024)", title: "Adapting Large Language Models to Domains via Reading Comprehension", locator: "arXiv 2309.09530v4, pp. 1, 4 and 6", note: "raw domain text hurts prompting; reading comprehension texts"}
  - {paper: "Yang, Band, Li, Candès and Hashimoto (2024)", title: "Synthetic continued pretraining", locator: "arXiv 2409.07431v2, pp. 1, 2, 7 and 8", note: "EntiGraph and continued pretraining on a small corpus"}
  - {paper: "Kirkpatrick, Pascanu, Rabinowitz, Veness and others (2017)", title: "Overcoming catastrophic forgetting in neural networks", locator: "arXiv 1612.00796v2, pp. 2 and 3", note: "elastic weight consolidation"}
  - {paper: "Biderman, Portes, Gonzalez Ortiz, Paul and others (2024)", title: "LoRA Learns Less and Forgets Less", locator: "arXiv 2405.09673v2, p. 1", note: "LoRA against full fine-tuning in continued pretraining"}
---

Continued pretraining takes a language model that has finished pretraining and trains it further, with the same next-token (or masked-token) objective, on data it has not seen: papers from one field, code, mathematics, a new language, longer documents, or newer text.
It is how most specialized large models are made, from BioBERT to Code Llama, and how context windows are extended after the fact.
The model learns the new data, but it also drifts away from the old, and the learning rate it restarts with, the old data mixed back in and the vocabulary it reads with decide how much it gains and how much it loses.
This page covers the schedule, replay and other ways to limit forgetting, vocabulary and context extension, and what the evidence says about when to continue and when to start from scratch.

## In plain words

Pretraining a large model costs millions of GPU hours.
When new text arrives, or a model is wanted for a field its training data covered thinly, starting over would repeat almost all of that cost.
Continued pretraining keeps the trained weights and runs the same training loop on the new text.
Nothing about the objective changes: the model still predicts the next token, there are no labels, and the data is raw text rather than instructions or preferences.

Two things go wrong when this is done naively.
The pretraining run ended with a tiny learning rate, and training on at that rate barely moves the model; raising it again moves the model quickly but also disturbs what it knew.
And everything the model learns from the new text is learned at the expense of the old, so its general abilities can fall: catastrophic forgetting.
The usual answers are to raise the learning rate again and decay it again, to mix a small share of the old data back into every batch, and to keep the new data as close to the old in quality and format as possible.

## Setting

Start from parameters θ_0 trained on a corpus D0 until the schedule ran out.
Continual pre-training continues to pre-train existing models on new data instead of re-training them, and its goal is to minimize the loss on new data while maintaining low loss on previous data. {#c-gup-goal}
Ibrahim et al. study an LLM that is randomly initialized and pre-trained on datasets D0, D1, ..., DN−1 in sequence, where N ≥ 2 and each dataset has at least 100B tokens. {#c-ibr-setting}
The new data D1 may be a domain, a language, a context length or simply later text, and it is usually large and unlabelled; supervised fine-tuning and instruction tuning come after and are separate stages.
Across four domains (biomedical and computer science publications, news, and reviews) and eight classification tasks, a second phase of pretraining in-domain, domain-adaptive pretraining, leads to performance gains under both high- and low-resource settings. {#c-gur-dapt}
The prevailing mixed-domain paradigm initializes domain-specific pretraining with a general-domain language model and inherits its vocabulary. {#c-pmb-paradigm}
The alternative is training a new model on D0 ∪ D1, or on D1 alone if it is large enough.
At a fixed model size training compute is proportional to tokens, so continuing a model already trained on D0 costs tokens(D1) where re-training on the union costs tokens(D0) + tokens(D1): 300B instead of 600B tokens for Ibrahim et al.'s Pile and SlimPajama. {#c-cp-compute}

The quantities to watch are adaptation, the fall of the loss on held-out D1 data, and forgetting, the rise of the loss on held-out D0 data or the fall in general benchmark scores.

## The method

### The learning rate: re-warm and re-decay

A pretrained checkpoint usually ends a cosine schedule at ηmin, a tenth or less of the peak.
If the learning rate was cosine-decayed from a large value ηmax to a small value ηmin during pre-training, re-warming and re-decaying it from O(ηmax) to O(ηmin) improves adaptation to a new dataset compared to continuing from small learning rates O(ηmin). {#c-ibr-rewarm}
Decreasing the schedule's maximum learning rate can help reduce forgetting, whereas increasing it can improve adaptation. {#c-ibr-maxlr}
Re-warming first increases the loss on upstream and downstream data, but in the longer run it improves the downstream performance, outperforming models trained from scratch. {#c-gup-rewarm}
The warmup itself matters less than where it goes.
The length of the warmup phase does not appear to have a significant effect on the Pile and SlimPajama validation losses. {#c-gup-warmlen}
In Gupta et al.'s setup, selecting earlier checkpoints for continued training does not lead to improvement in downstream performance, so they continue from the latest checkpoint. {#c-gup-latest}
Some of the damage comes from the jump in learning rate, not from the new data.
Re-warming hurts even without a shift: continuing on the Pile with ηmax = 3·10−4 gives a peak increase of the Pile validation loss of 0.1, against an increase of 0.35 with the same schedule when continuing on SlimPajama. {#c-ibr-samedata}
Infinite learning rate schedules, which transition into a high constant learning rate across tasks, are promising alternatives to cosine decay schedules. {#c-ibr-infinite}
A model that will be continued again and again can then join each new dataset in the constant phase and anneal only before a release.
The re-decayed cosine schedule η(s) = ηmin + (ηmax − ηmin)(1 + cos πs)/2, with s the fraction of the decay done, starts at ηmax, ends at ηmin, and passes the midpoint (ηmax + ηmin)/2 halfway. {#c-cp-cosine}

Parmar et al. continue a model that is already strong on general benchmarks, to improve its weak areas rather than to change domain.
Parmar et al. start on top of a 15B parameter language model that has seen 8T tokens of pretraining data. {#c-par-base}
Their recipe starts with a distribution similar to the pretraining set that places larger weight on high quality sources, then moves to one that incorporates QA data and upweights areas of model weakness; the learning rate starts from the pretrained model's ηmin and decays with cosine annealing to a hundredth of it, and the switch occurs when it reaches a fifth of its start. {#c-par-recipe}
On a cosine decay from 4.5e-5 to 4.5e-7 the learning rate reaches a fifth of its start, 9e-6, about 71% of the way through, so the second distribution gets roughly the last 29% of the tokens. {#c-cp-switch}
On top of the well-trained 15B parameter model, the recipe gives an improvement of 9% in average model accuracy compared to the baseline of continued training on the pretraining set. {#c-par-gain}
Ablations with the WSD annealing schedule were not competitive with cosine annealing. {#c-par-wsd}

### Replay

Replay mixes old data back into every batch.
A combination of learning rate re-warming, re-decaying and replay of previous data is sufficient to match the performance of fully re-training from scratch on all available data, as measured by the final loss and the average score on several language model evaluation benchmarks. {#c-ibr-match}
Compute-equivalent replay keeps the tokens fixed: with D0 and D1 of 100B tokens each, the model with 5% replay sees 100B unique tokens of D0, then 95B unique tokens of D1 plus 5B replayed tokens from D0, for a total of 200B tokens. {#c-ibr-cer}
The runs that match re-training use 5% replay for SlimPajama and 25% replay for German. {#c-ibr-fractions}
With replay fraction r a run of T tokens sees (1 − r)T new tokens, so reaching the new tokens of a run without replay takes 1/(1 − r) times as many tokens: about 1.05 times for 5% replay, 4/3 for 25% and twice for 50%. {#c-cp-replay-cost}
If each sequence comes from the old data with probability r and from the new data otherwise, the expected loss of a batch is r L_old(θ) + (1 − r) L_new(θ), so replay minimises a weighted sum of the two losses and its expected gradient is the same weighted sum of their gradients. {#c-cp-mixture}
The original data is often unavailable, as with Llama 2, so in practice replay means a public corpus that resembles it.
To help the model retain natural language understanding skills, Code Llama samples a small proportion of its batches from a natural language dataset. {#c-cl-nl}
To prevent regressing on general coding and language understanding, Code Llama - Instruct is also trained with a small proportion of data from the code dataset (6%) and the natural language dataset (2%). {#c-cl-rehearsal}
Llemma's training mixture is 95% Proof-Pile-2, 2% the Pile with ArXiv removed, and 3% the GitHub subset of RedPajama. {#c-lle-mix}

### Regularization towards the old weights

Instead of old data, a penalty can hold the parameters near θ_0.
Elastic weight consolidation implements its constraint as a quadratic penalty that can be imagined as a spring anchoring the parameters to the previous solution, stiffer for the parameters that matter more to the old task. {#c-ewc-spring}
EWC approximates the posterior as a Gaussian with mean given by the old parameters and a diagonal precision given by the diagonal of the Fisher information matrix, which can be computed from first-order derivatives alone. {#c-ewc-fisher}
The function L that EWC minimizes is L_B(θ) + Σ_i (λ/2) F_i (θ_i − θ_A,i)², where L_B(θ) is the loss for task B only, λ sets how important the old task is compared to the new one and i labels each parameter. {#c-ewc-loss}
For a new-task loss that is quadratic in one parameter, (a/2)(θ − θ_B)², the EWC objective is minimized at (a θ_B + λF θ_A)/(a + λF), a precision-weighted average that moves from the new optimum θ_B towards the old value θ_A as λF grows. {#c-cp-ewc}
For billion-parameter models the Fisher information is costly to estimate well and the penalty adds a copy of the parameters to memory, so large continued-pretraining runs use replay instead.
Training only a low-rank update is another restraint.
With ≈20B unstructured tokens of continued pretraining on programming and mathematics, LoRA in the standard low-rank settings substantially underperforms full finetuning, but better maintains the base model's performance on tasks outside the target domain. {#c-lor-cpt}
Full finetuning learns perturbations with a rank that is 10-100× greater than typical LoRA configurations. {#c-lor-rank}

### Extending the vocabulary

A tokenizer trained mostly on English splits other scripts and technical terms into many pieces, which costs context and compute and makes each piece carry less meaning.
The original LLaMA vocabulary covers less than a thousand Chinese characters, and since unknown UTF-8 characters fall back to bytes, each Chinese character splits into 3-4 byte tokens. {#c-cui-chars}
Chinese LLaMA trains a Chinese SentencePiece tokenizer of 20,000 tokens, merges it into the LLaMA tokenizer by taking the union of their vocabularies to get 49,953 tokens, and resizes the word embeddings and language model head from V × H to V′ × H, appending the new rows so that the embeddings of the original tokens remain unaffected. {#c-cui-merge}
The number of tokens generated by the Chinese LLaMA tokenizer is approximately half of those generated by the original LLaMA tokenizer. {#c-cui-half}
Growing the vocabulary from 32,000 to 49,953 adds 17,953 rows to both the embedding and the output head, 2 × 17,953 × 4,096 ≈ 147 million new parameters at a hidden size of 4,096. {#c-cp-vocab}
How the new rows start decides what the model predicts before they are trained.
If the k new output rows are the mean of the V old ones, each new logit is the mean of the old logits, so the new tokens take at most k/(V + k) of the probability and the KL divergence from the old next-token distribution to the new one is at most log(1 + k/V), about 0.45 nats for k = 17,953 and V = 32,000. {#c-cp-meaninit}
For the basic Chinese LLaMA-7B, stage 1 fixes the transformer encoders and trains only the embeddings, and stage 2 adds LoRA adapters and trains the embeddings, LM heads and LoRA parameters. {#c-cui-twostage}
Keeping the old tokenizer is the other choice, and is common when the domain shares the script.
BioBERT kept the original WordPiece vocabulary of BERT-base, for compatibility with BERT and because any new words may still be represented by frequent subwords. {#c-bio-vocab}
BERT divides naloxone into four pieces and shatters acetyltransferase into seven, while both terms appear in the vocabulary of PubMedBERT. {#c-pmb-frag}

### Extending the context

A model pretrained on 4,096-token sequences is continued on longer ones, with the rotary position encoding changed so that distant positions are not rotated far out of the range seen in training.
Code Llama's long context fine-tuning presents sequences of 16,384 tokens, up from 4,096, and increases the RoPE base period from 10,000 to 1,000,000. {#c-cl-lcft}
Xiong et al. continually pretrain Llama 2 checkpoints with increased sequence length while keeping the same number of tokens per batch, for a total of 400B tokens over 100,000 steps. {#c-xio-cpt}
They increase the base frequency b of RoPE from 10,000 to 500,000, which reduces the rotation angles of each dimension. {#c-xio-base}
With RoPE frequencies b^(−2i/d), pair i has wavelength 2π b^(2i/d) positions; raising b from 10,000 to 1,000,000 leaves the fastest pair at 2π and stretches the slowest pair of a 128-dimensional head about 93 times, from about 54,000 to about 5 million positions. {#c-cp-rope}
Often the quality of the data plays a more critical role than the length of texts for long-context continual pretraining. {#c-xio-quality}
The 7B/13B models use learning rate 2e−5 with a cosine schedule and 2000 warm-up steps, and the larger 34B/70B models need a smaller learning rate (1e−5) to get monotonically decreasing validation losses. {#c-xio-lr}
Continual pretraining from short context models can save around 40% FLOPs while imposing almost no loss on performance. {#c-xio-save}
The validation loss can be fit as a power law in the context length plus a constant, with a different set of constants for each model size. {#c-xio-law}
The base is part of the model's interface: a later stage that expects the original base has to change it back.
Before training Llemma 7B, the RoPE base period of the Code Llama 7B initialization was contracted from 1,000,000 to 10,000. {#c-lle-rope}

## Derivation

The full derivation is in `derivations/continued-pretraining-arithmetic.md`, and `continued-pretraining-arithmetic.check.py` checks every step numerically.

**Compute.** At a fixed model size compute is about 6 × parameters × tokens, so continuing pays only for the new tokens, and re-training pays for all earlier data again.

**Schedules.** The cosine η(s) = ηmin + (ηmax − ηmin)(1 + cos πs)/2 equals ηmax at s = 0, ηmin at s = 1 and their mean at s = 1/2; solving η(s) = ηmax/5 for Parmar et al.'s range 4.5e-5 to 4.5e-7 gives s = arccos(2 × 0.19/0.99 − 1)/π ≈ 0.711.

**Replay.** A fraction r of old tokens leaves (1 − r) of each batch for new data; by linearity of expectation the expected batch loss and gradient are the r-weighted sums of the old and new ones.

**EWC.** For a quadratic new-task loss the penalized objective is a sum of two quadratics, minimized at the curvature-weighted average of the two optima.

**Mean-initialized rows.** A new row equal to the mean of the old rows gives a logit equal to the mean of the old logits; Jensen's inequality bounds e^{mean} by the mean of e^{z}, which bounds the new tokens' mass by k/(V + k) and the KL divergence by log(1 + k/V).

**RoPE.** Pair i turns through θ_i = b^(−2i/d) per position, so its wavelength 2π b^(2i/d) grows with b for every i > 0, by a factor (b′/b)^(2i/d).

## Worked example

Take a 7B model pretrained on 2T tokens, continued on 300B tokens of code, with the general data unavailable and a public web corpus used for replay.
Re-training on both would cost 6 × 7e9 × 2.3e12 ≈ 9.7e22 FLOPs; continuing costs 6 × 7e9 × 3e11 ≈ 1.3e22, about 13% of that.
With 5% replay, 15B of the 300B tokens are web text and 285B are code; seeing all 300B tokens of code would need 300/0.95 ≈ 316B tokens.
If pretraining ended at ηmin = 3e-5 from a peak of 3e-4, the continued run warms up to 3e-4 over about 1% of its steps and decays back to 3e-5 with a cosine.
Halfway through, the learning rate is (3e-4 + 3e-5)/2 = 1.65e-4.

If the code needs 2,000 new tokens and the hidden size is 4,096, untied embeddings and head gain 2 × 2,000 × 4,096 ≈ 16.4 million parameters.
With a 32,000-token vocabulary and mean-initialized rows, the new tokens start with at most 2,000/34,000 ≈ 5.9% of the probability, and the next-token distribution moves by at most log(1 + 2,000/32,000) ≈ 0.061 nats.

The NumPy tests run this on a bigram model: trained on one Markov chain and continued on another, it loses more than a nat on the first chain; replaying 25% of the old data cuts that loss by more than half, at some cost on the second chain, and an EWC penalty also reduces it.
The PyTorch tests do the same with a small RoPE decoder, and check that changing the RoPE base leaves the weights untouched and that mean-initialized vocabulary rows stay within the KL bound.

## Hyperparameters

The peak learning rate trades adaptation against forgetting: a lower peak keeps more of the old abilities and a higher one learns the new data faster, and the warmup length matters little once the peak is chosen.
The replay fraction scales with the size of the shift: around 1% to 5% for new English web text, 25% for a new language, and somewhat more for a model whose general behaviour must not change at all.
The token budget follows the size of the new data and the goal: a single pass over a domain corpus for BERT-sized models, hundreds of billions of tokens for a code or mathematics model, and repeated passes when the domain corpus is small.
A two-phase mixture, general data with upweighted quality sources first and the target data last, uses the low-learning-rate end of the run for the data that should be learned most precisely.
Vocabulary extension is worth it when the old tokenizer spends several tokens per character or word of the new language; for a domain in the same script the old vocabulary usually stays.
For context extension the RoPE base and the sequence length change together, with a learning rate an order of magnitude below the pretraining peak.

## Cost

Continued pretraining costs about 6 × parameters × tokens of the continued run, the same per token as pretraining, and nothing for the data already learned.
Replay spends a fraction r of that budget on old data.
Long sequences cost more per token, since attention grows with the sequence length, which is why context extension is a short stage at the end rather than the whole run.
A vocabulary extension adds rows to the embedding and output layers, and with them memory and some compute in the output softmax.
EWC stores a copy of the old parameters and their Fisher information, two extra copies of the model in memory.
At inference the model costs what the original cost, apart from vocabulary and context changes.
Neural scaling laws describe the loss of a model trained from scratch at a given size and token count; a continued run starts from a model already partway along that curve, so its budget is measured against the tokens still to be learned rather than the whole curve.

## Evidence

**Domain adaptation of encoders.**
RoBERTa trained on each domain for 12.5K steps, a single pass on each domain dataset, and its masked LM loss decreased in all domains except News, where it rose marginally. {#c-gur-loss}
DAPT significantly outperforms adapting to an irrelevant domain, and in most settings exposure to more data without considering domain relevance is detrimental to end-task performance. {#c-gur-irrelevant}
Task-adaptive pretraining, a second phase of pretraining on the task's unlabeled training data for 100 epochs with masking probability 0.15, consistently improves the RoBERTa baseline for all tasks across domains. {#c-gur-tapt}
TAPT is much less expensive to run than DAPT, and its performance is often competitive with that of DAPT. {#c-gur-taptcost}
Though TAPT is effective, it is harmful when applied across tasks. {#c-gur-transfer}
Whenever the Wiki + Books corpora were used for pre-training, BioBERT was initialized with the pre-trained BERT model, for computational efficiency. {#c-bio-init}
BioBERT was pretrained on biomedical corpora for 23 days on eight NVIDIA V100 GPUs. {#c-bio-cost}
BioBERT outperforms previous models on biomedical named entity recognition (0.62% F1 improvement), relation extraction (2.80% F1) and question answering (12.24% MRR). {#c-bio-gains}

**Code.**
All Code Llama models are initialized with Llama 2 model weights and trained on 500B tokens from a code-heavy dataset, except Code Llama 70B, which was trained on 1T tokens; the Python models are further specialized on 100B tokens. {#c-cl-init}
Retaining the original learning rate of the Llama 2 base model gave better results for Code Llama 7B than changing it, and the 13B, 34B and 70B models use 3e−4, 1.5e−4 and 1.5e−4. {#c-cl-lr}
Llama 2 was trained on 2T tokens including only 80B tokens of code; a 7B model trained from scratch on Code Llama's data mix ends with the loss Code Llama 7B had at about half of its training, with 240B less training tokens. {#c-cl-scratch}
Code Llama's 500B tokens at 85% code hold about 425B tokens of code, over five times the 80B tokens of code in Llama 2's pretraining. {#c-cp-codetokens}

**Mathematics.**
Proof-Pile-2 is a 55B-token mixture of scientific papers, web data containing mathematics, and mathematical code. {#c-lle-data}
Llemma continues training the Code Llama models on Proof-Pile-2 with a standard autoregressive language modeling objective, the 7B model for 200B tokens and the 34B model for 50B tokens. {#c-lle-tokens}
For Llemma 7B the learning rate is warmed up to 1·10−4 over 500 steps, then set to cosine decay to 1/30th of the maximum learning rate. {#c-lle-lr}
DeepSeekMath 7B continues pretraining DeepSeek-Coder-Base-v1.5 7B with 120B math-related tokens sourced from Common Crawl, together with natural language and code data. {#c-dsm-corpus}
Starting from a code training model is a better choice than a general LLM, and the math training also improves the model on MMLU and BBH. {#c-dsm-init}
The DeepSeekMath-Base training data is 56% from the DeepSeekMath Corpus, 4% from AlgebraicStack, 10% from arXiv, 20% Github code, and the remaining 10% natural language data from Common Crawl in both English and Chinese. {#c-dsm-mix}
Llemma 7B's 200B tokens at 95% Proof-Pile-2 are 190B tokens, about 3.5 passes over the 55B-token corpus, and DeepSeekMath's 500B tokens at 56% are 280B tokens, about 2.3 passes over its 120B-token corpus. {#c-cp-epochs}

**Forgetting and replay at scale.**
Even a replay of 1% significantly reduces forgetting on the Pile compared to the no-replay baselines, with little impact on downstream performance for 1%, 5%, and 10% replay. {#c-ibr-1pct}
Replay significantly reduces forgetting across both shifts, but the stronger shift requires more replay to mitigate forgetting to the same extent. {#c-ibr-strong}
With 50% replay the model adapts significantly worse to D1, yet it attains or surpasses the final average validation performance of the baseline trained on D1 ∪ D0 while seeing 150B fewer SlimPajama tokens. {#c-ibr-50}
Glorioso et al. applied re-warming, re-decaying and 60% replay in a decay phase over 50B tokens of high-quality data, and DeepSeek-AI used 30% replay of pre-training data to continually pre-train DeepSeekV2 on 6T tokens. {#c-ibr-apps}
LoRA mitigates forgetting more than common regularization techniques such as weight decay and dropout. {#c-lor-forget}

## Limitations and failure modes

Continuing is not always better than starting over.
For domains with abundant unlabeled text, such as biomedicine, pretraining language models from scratch results in substantial gains over continual pretraining of general-domain language models. {#c-pmb-scratch}
Biomedicine has over thirty million abstracts in PubMed and adds over a million each year. {#c-pmb-abstracts}
A model continued from a general checkpoint keeps the general tokenizer, its fragments and its context length, and those limits stay unless they are changed explicitly.

Raw domain text can teach facts and still make the model harder to use.
Continued training on domain-specific raw corpora endows the model with domain knowledge but drastically hurts its prompting ability for question answering. {#c-che-hurt}
Cheng et al. transform raw corpora into reading comprehension texts, enriching each raw text with a series of tasks related to its content. {#c-che-method}
The comprehension tasks are mined from the raw text with a handful of mining patterns. {#c-che-mining}
Mixing the reading comprehension texts with general instructions, the optimal ratios for biomedicine, finance, and law are 1 : 1, 1 : 2, and 1 : 1. {#c-che-mix}
The resulting 7B model achieves competitive performance with domain-specific models of much larger scales, such as BloombergGPT-50B. {#c-che-bbg}

A small corpus is not enough to learn from directly.
Knowledge acquisition from pretraining is data-inefficient: to learn a given fact, models must be trained on hundreds to thousands of diverse representations of it. {#c-ent-ineff}
Raw CPT, continued pretraining on the raw source documents, performs even worse than Llama 3 8B Base. {#c-ent-raw}
Rewriting the corpus into many more tokens is one fix.
EntiGraph extracts salient entities from the source documents and then generates diverse text by drawing connections between the sampled entities. {#c-ent-method}
Yang et al. use EntiGraph to generate 455M synthetic tokens from 1.3M real tokens using gpt-4-turbo, then continually pretrain Llama 3 8B on the synthetic tokens. {#c-ent-setup}
EntiGraph CPT continually pretrains Llama 3 8B Base on the 455M token EntiGraph corpus for 2 epochs with replay on RedPajama. {#c-ent-replay}
Continued pretraining on the EntiGraph corpus improves closed-book QA accuracy from 39.49% for Llama 3 8B Base to 56.22%. {#c-ent-acc}
With 455M EntiGraph tokens, synthetic continued pretraining provides 80% of the accuracy improvement of having the source documents available at inference time. {#c-ent-rag}

Forgetting is measured on what the evaluator thinks to measure, and a held-out loss on a replay corpus is only a stand-in for the original data.
Benchmarks of general ability can hide losses in safety behaviour or instruction following, which a base model continued after alignment may lose entirely and which then need aligning again.
And every reported recipe is tuned to one model and one shift: the replay fractions, peaks and switch points above are starting points, not constants.

## Variants

**Domain-adaptive pretraining (DAPT)** continues on a large corpus from the target domain, such as biomedical papers, code or law.

**Task-adaptive pretraining (TAPT)** continues on the unlabelled text of one task for many epochs, cheaply, and can follow DAPT.

**Language adaptation** continues on a new language, often with an extended vocabulary and embedding-only training first.

**Long-context continued pretraining** continues on longer sequences with a larger RoPE base, as Code Llama and long-context Llama 2 do.

**Mid-training or annealing on quality data** continues a general model on a reweighted mixture at a decaying learning rate, as in Parmar et al.'s two distributions.

**Synthetic continued pretraining** turns a small corpus into a large synthetic one before continuing, as EntiGraph does.

**Continual pretraining over many stages** repeats the process as new data arrives, with replay and infinite schedules so that every stage can start without re-warming.

## When to use it

Continue a pretrained model when the new data is large enough to matter but small or cheap compared with the original pretraining, when the target is a domain, language or context length the base model handles poorly, and when the base model's general abilities are worth keeping.
Prefer pretraining from scratch when the domain has hundreds of billions of tokens of its own, a vocabulary far from the general one, and no need for general abilities.
Prefer fine-tuning or LoRA when the goal is a behaviour or task format rather than knowledge, and the data is thousands of examples rather than billions of tokens.
Put instruction tuning and preference training after continued pretraining, since continued pretraining on raw text weakens what they teach.

## Pitfalls

* Continuing at the final learning rate of pretraining: the model barely adapts, and the gain looks like evidence that the data does not help.
* Re-warming without replay on a strong shift: the old abilities drop and do not come back by the end of the run.
* Choosing replay data that differs from the original pretraining data, which then guards against forgetting something else.
* Adding new vocabulary rows with random initialization, which can make the untrained tokens take most of the probability at the start.
* Raising the RoPE base without training on long sequences, or training a later stage with a different base than the earlier one.
* Measuring only the target domain: forgetting shows up on general benchmarks, safety and instruction following.
* Continuing a chat model on raw text and expecting it to stay a chat model.

## History

BioBERT (Lee et al., 2019) initialized BERT from its general checkpoint and continued it on PubMed abstracts and PMC articles, and SciBERT and others followed for other fields.
Gururangan et al. (2020) named domain-adaptive and task-adaptive pretraining and tested them across four domains, finding gains in each.
Gu et al. (2021) answered that biomedicine has enough text to pretrain from scratch with its own vocabulary, and did better that way.
Large decoder models brought the method back at a different scale: Chinese LLaMA (2023) extended the vocabulary for a new language, Code Llama (2023) continued Llama 2 on 500B tokens of code and extended its context, long-context Llama 2 (Xiong et al., 2023) did the same for 32,768-token sequences, and Llemma (2023) and DeepSeekMath (2024) continued code models on mathematics.
Gupta et al. (2023) and Ibrahim et al. (2024) studied the schedule and replay directly, and found that re-warming, re-decaying and a few percent of replay can match re-training on everything.
Elastic weight consolidation (Kirkpatrick et al., 2017) came from continual learning on small networks and is the reference regularizer for forgetting, though large runs rely on replay.
Parmar et al. (2024) and Yang et al. (2024) extended the idea to improving a strong general model and to learning from a corpus too small to train on directly.
