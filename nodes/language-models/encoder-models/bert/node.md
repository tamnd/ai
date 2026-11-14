---
tag: NEW-bert
kind: model
slug: bert
name: BERT
aliases: [Bidirectional Encoder Representations from Transformers, BERT-Base, BERT-Large, Multilingual BERT, M-BERT]
area: language-models/encoder-models
level: core
summary: >-
  A family of Transformer encoders from Google, pre-trained on unlabeled text with masked language modelling and next sentence prediction, then fine-tuned for each task; released in 2018.
history:
  year: 2018
  note: "Google announced the open-source release of BERT on 2 November 2018 (Devlin and Chang, 2018); the paper appeared at NAACL 2019 (Devlin et al., 2019)."
same_as:
  wikipedia: BERT_(language_model)
  arxiv: ["1810.04805"]
sources:
  - {paper: "Devlin et al. (2019)", title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", locator: "the abstract, pp. 1 to 9, Appendix A on pp. 13 and 14", note: "the architecture, the pre-training tasks, the data, the training and the results"}
  - {paper: "Google Research (2018 to 2020)", title: "the README, LICENSE and multilingual.md of the google-research/bert repository", locator: "p. 1 of each", note: "the releases, their dates and sizes, the licence and the multilingual data"}
  - {paper: "Devlin and Chang (2018)", title: "Open Sourcing BERT: State-of-the-Art Pre-training for Natural Language Processing", locator: "p. 1"}
  - {paper: "Nayak (2019)", title: "Understanding searches better than ever before", locator: "p. 1"}
  - {paper: "Rogers et al. (2020)", title: "A Primer in BERTology: What We Know About How BERT Works", locator: "p. 1"}
  - {paper: "Liu et al. (2019); Lan et al. (2020); Sanh et al. (2019); Clark et al. (2020)", title: "RoBERTa; ALBERT; DistilBERT; ELECTRA", locator: "the abstract of each"}
  - {paper: "McCoy et al. (2019); Niven and Kao (2019); Ettinger (2020)", title: "the analyses of what BERT gets wrong", locator: "the abstract of each"}
role: family
---

BERT is a family of language representation models made by Google and released with code and pre-trained weights on {fact:bert/released.bert}.
A BERT model is a Transformer encoder that is pre-trained on unlabeled text and then fine-tuned, as a whole, on each downstream task.
Its two pre-training tasks are masked language modelling, which lets every layer see the words on both sides of a position, and next sentence prediction.
The English models come in two sizes, BERT-Base with {fact:bert/params.total.bert_base} and BERT-Large with {fact:bert/params.total.bert_large}, and the weights are under the {fact:bert/licence} licence.

## Overview

BERT stands for Bidirectional Encoder Representations from Transformers. {#c-name}
Devlin et al. designed it to pre-train deep bidirectional representations from unlabeled text, by conditioning on both the left and the right context in all layers.
They write that the pre-trained model can be fine-tuned with just one additional output layer to give models for a wide range of tasks, such as question answering and language inference, without substantial task-specific changes to the architecture. {#c-finetune}

Using BERT has two steps. {#c-two-steps}
In pre-training, the model learns from unlabeled data on two pre-training tasks.
In fine-tuning, the model starts from the pre-trained parameters, and all of the parameters are trained on labeled data from one downstream task.
Each task gets its own fine-tuned model, even though all start from the same pre-trained parameters.

Google open sourced BERT in November 2018, with source code built on TensorFlow and a number of pre-trained models. {#c-open-sourced}
It wrote that with this release anyone could train their own question answering system in about 30 minutes on a single Cloud TPU, or in a few hours on a single GPU. {#c-30-minutes}
The paper reported gains over the earlier published systems on eleven tasks, listed under Results.

## Architecture

BERT's architecture is a multi-layer bidirectional Transformer encoder, based on the original implementation of Vaswani et al. (2017). {#c-arch}
The Transformer is a network architecture based solely on attention mechanisms, with no recurrence and no convolutions. {#c-transformer}
BERT keeps only its encoder stack, so every position can attend to every other position.

The paper writes L for the number of layers (Transformer blocks), H for the hidden size and A for the number of self-attention heads.

| Model | L | H | A | Feed-forward size | Parameters |
|---|---|---|---|---|---|
| BERT-Base | 12 | 768 | 12 | 3072 | {fact:bert/params.total.bert_base} |
| BERT-Large | 24 | 1024 | 16 | 4096 | {fact:bert/params.total.bert_large} |

BERT-Base has 12 layers, a hidden size of 768 and 12 attention heads, with 110M parameters in total; BERT-Large has 24 layers, a hidden size of 1024 and 16 heads, with 340M parameters. {#c-sizes}
The feed-forward size is 4H in both. {#c-ffn}
The authors chose BERT-Base to have the same size as OpenAI GPT, so that the two can be compared. {#c-gpt-size}
The difference between the two is the attention: BERT's self-attention is bidirectional, and in GPT every token can only attend to the context on its left.

**Input.** BERT splits text into WordPiece tokens from a vocabulary of {fact:bert/vocabulary}. {#c-wordpiece}
Every sequence starts with a special classification token, [CLS], and the final hidden state of this token stands for the whole sequence in classification tasks.
A pair of sentences is packed into one sequence and separated by the special token [SEP], and a learned embedding marks each token as part of sentence A or sentence B. {#c-pairs}
The input representation of a token is the sum of its token embedding, its segment embedding and its position embedding. {#c-input-sum}
WordPiece comes from Google's neural machine translation system, where Wu et al. split words into a limited set of common sub-word units to handle rare words. {#c-gnmt-wordpieces}

## Training data

The English models are pre-trained on the BooksCorpus, of 800M words, and English Wikipedia, of 2,500M words. {#c-data}
From Wikipedia only the text passages are kept, without lists, tables and headers.
The authors call it critical to train on whole documents rather than on shuffled sentences, such as the Billion Word Benchmark, so that the model sees long contiguous sequences. {#c-doc-level}
The BooksCorpus is a corpus that Zhu et al. collected from the web: 11,038 free books by authors who had not been published. {#c-bookcorpus}

The multilingual models are trained on the Wikipedias of the 100 languages with the largest Wikipedias, each taken whole without user and talk pages. {#c-ml-data}
Because these Wikipedias differ greatly in size, Google sampled the data with exponentially smoothed weights, with S=0.7, so that English is under-sampled and a smaller language such as Icelandic is over-sampled. {#c-ml-smoothing}
The multilingual models share one WordPiece vocabulary of 110k tokens and do not mark the language of the input, so that zero-shot transfer can work. {#c-ml-vocab}

## Training

**Masked language modelling.** A standard language model can only be trained left to right or right to left, since with context on both sides each word could see itself through the layers.
BERT instead masks 15% of the WordPiece tokens of each sequence at random and predicts only the masked tokens, a procedure also known as a Cloze task. {#c-mlm}
The [MASK] token never appears in fine-tuning, so this creates a mismatch between pre-training and fine-tuning. {#c-mismatch}
To reduce it, a chosen token becomes [MASK] 80% of the time, a random token 10% of the time, and stays unchanged 10% of the time. {#c-80-10-10}
The model then predicts the original token from its final hidden vector, with a cross-entropy loss.

**Next sentence prediction.** For each pre-training pair, sentence B is the sentence that actually follows A half of the time (IsNext) and a random sentence from the corpus the other half (NotNext). {#c-nsp}
The model predicts the label from the final hidden vector of [CLS], and the final model reaches 97% to 98% accuracy on this task. {#c-nsp-acc}
The authors add that this vector is not a meaningful sentence representation without fine-tuning, because it was trained for next sentence prediction. {#c-cls-meaning}
The training loss is the sum of the mean masked LM likelihood and the mean next sentence prediction likelihood. {#c-loss}

**Pre-training.** BERT was trained with batches of 256 sequences of 512 tokens, given in the paper as {fact:bert/training.tokens_per_batch} per batch, for {fact:bert/training.steps} steps, about 40 epochs over the 3.3 billion words of the corpus. {#c-batch}
The optimizer was Adam, with a learning rate of 1e-4, β1 = 0.9, β2 = 0.999, an L2 weight decay of 0.01, warmup over the first 10,000 steps and then linear decay. {#c-optim}
Dropout was 0.1 on all layers, and the activation was GELU rather than ReLU, after OpenAI GPT.
Attention costs grow with the square of the sequence length, so the model was trained on sequences of 128 tokens for 90% of the steps and of 512 tokens for the last 10%, which also teaches the position embeddings for long inputs. {#c-seq128}
BERT-Base was trained on 4 Cloud TPUs ({fact:bert/training.tpu_chips.bert_base} TPU chips) and BERT-Large on 16 Cloud TPUs ({fact:bert/training.tpu_chips.bert_large} chips), each for 4 days. {#c-tpus}

Strubell et al. estimate that training BERT-base on 64 V100 GPUs for 79 hours emits 1438 lbs of CO2e and costs $3751 to $12,571 in cloud compute. {#c-strubell-cost}
They compare training BERT on GPUs to a trans-American flight. {#c-strubell-flight}

**Fine-tuning.** Fine-tuning is much cheaper: the authors write that every result of the paper can be reproduced in at most 1 hour on a single Cloud TPU, or a few hours on a GPU, from the same pre-trained model. {#c-ft-cheap}
They found that batch sizes of 16 and 32, learning rates of 5e-5, 3e-5 and 2e-5 for Adam, and 2 to 4 epochs work well across all tasks. {#c-ft-hparams}

## Releases and versions

| Release | Date | Models | Licence |
|---|---|---|---|
| BERT (English) | {fact:bert/released.bert} | BERT-Base and BERT-Large, Uncased and Cased | {fact:bert/licence} |
| Multilingual and Chinese | {fact:bert/released.bert_multilingual} | BERT-Base, Multilingual (102 languages); BERT-Base, Chinese | {fact:bert/licence} |
| Multilingual Cased | {fact:bert/released.bert_multilingual_cased} | BERT-Base, Multilingual Cased (104 languages) | {fact:bert/licence} |
| Whole Word Masking | {fact:bert/released.bert_wwm} | BERT-Large, Uncased and Cased | {fact:bert/licence} |
| Smaller BERT models | {fact:bert/released.bert_small} | 24 models, from BERT-Tiny to BERT-Base | {fact:bert/licence} |

The first release had TensorFlow code for the model and pre-trained checkpoints of BERT-Base and BERT-Large from the paper, each in a lowercased and a cased version. {#c-rel-repo}
Uncased models see text that was lowercased, with accent markers removed, before WordPiece tokenization; Cased models keep the case and the accent markers. {#c-rel-cased}

On November 3rd, 2018, Google added a multilingual BERT-Base for 102 languages and a BERT-Base for Chinese. {#c-rel-multi}
On November 5th, 2018, third-party PyTorch and Chainer versions became available, and Google notes that it was not involved in making the PyTorch version. {#c-rel-pytorch}
On November 15th, 2018, Google released the code of its SQuAD 2.0 system, at 83% F1. {#c-rel-squad2}
On November 23rd, 2018, it added a multilingual model for 104 languages, Thai and Mongolian among them, which does not normalize its input in any way. {#c-rel-multi-cased}
On February 7th, 2019, BERT was put on TensorFlow Hub. {#c-rel-tfhub}

On May 31st, 2019, Google released BERT-Large models trained with Whole Word Masking, which masks all of the tokens of a word at once, at the same overall rate. {#c-rel-wwm}
The data and the training were otherwise the same as for the original models.
The README reports 92.8/86.7 SQuAD 1.1 F1/EM and 87.07 MultiNLI accuracy for the uncased Whole Word Masking model, against 91.0/84.3 and 86.05 for the original uncased BERT-Large. {#c-rel-wwm-scores}

On March 11th, 2020, Google released 24 smaller BERT models, English only and uncased, from the paper of Turc et al. {#c-rel-small}
Turc et al. show that pre-training still matters for small architectures. {#c-turc}
Google writes that the small models are meant for environments with restricted computational resources, and work best for knowledge distillation from a larger teacher. {#c-rel-small-use}
On GLUE, the README gives scores of 64.2 for BERT-Tiny, 65.8 for BERT-Mini, 71.2 for BERT-Small and 73.5 for BERT-Medium. {#c-rel-small-glue}

## Results

Devlin et al. report gains over the earlier systems on eleven tasks, among them a GLUE score of 80.5% (7.7 points better than before), MultiNLI accuracy of 86.7% (4.6 points better), SQuAD v1.1 test F1 of 93.2 (1.5 points better) and SQuAD v2.0 test F1 of 83.1 (5.1 points better). {#c-res-abstract}
The rows below are as the paper reports them, each next to the earlier systems that the paper compares with.

| Task | Metric | BERT-Base | BERT-Large | OpenAI GPT | Source |
|---|---|---|---|---|---|
| GLUE, test, average without WNLI | score | 79.6 | 82.1 | 75.1 | Devlin et al. (2019), Table 1 |
| GLUE, official leaderboard | score | | 80.5 | 72.8 | Devlin et al. (2019), p. 6 |
| SQuAD v1.1, test (ensemble with TriviaQA) | EM / F1 | | 87.4 / 93.2 | | Devlin et al. (2019), Table 2 |
| SQuAD v2.0, test (single model) | EM / F1 | | 80.0 / 83.1 | | Devlin et al. (2019), Table 3 |
| SWAG, test | accuracy | | 86.3 | 78.0 | Devlin et al. (2019), Table 4 |
| CoNLL-2003 NER, test | F1 | 92.4 | 92.8 | | Devlin et al. (2019), Table 7 |

On GLUE, BERT-Base averages 79.6 and BERT-Large 82.1 on the test sets, against 75.1 for OpenAI GPT. {#c-res-glue}
On the official leaderboard, BERT-Large scored 80.5 and OpenAI GPT 72.8. {#c-res-board}
BERT-Large beats BERT-Base on every task, most of all on tasks with very little training data. {#c-res-large}
On SQuAD 1.1, an ensemble of BERT-Large models, also fine-tuned on TriviaQA, reaches 87.4 EM and 93.2 F1 on the test set. {#c-res-squad1}
On SQuAD 2.0, a single BERT-Large model reaches 80.0 EM and 83.1 F1, a gain of 5.1 F1 over the earlier systems in the paper's comparison. {#c-res-squad2}
On SWAG, BERT-Large reaches 86.3% accuracy, 27.1 points above the ESIM+ELMo baseline and 8.3 points above OpenAI GPT. {#c-res-swag}
On CoNLL-2003 named entity recognition, fine-tuned BERT-Large reaches 92.8 test F1. {#c-res-ner}
Used without fine-tuning, as fixed features, the concatenation of its top four hidden layers is only 0.3 F1 behind fine-tuning the whole model.

**Ablations.** The ablations use BERT-Base and report accuracy on the development sets.
Removing next sentence prediction hurts QNLI, MNLI and SQuAD 1.1 significantly. {#c-abl-nsp}
A left-to-right model without next sentence prediction, which is like OpenAI GPT, is worse than the masked model on every task: MRPC falls from 86.5 to 77.5 and SQuAD F1 from 87.9 to 77.8. {#c-abl-ltr}
Larger models are more accurate on all four GLUE tasks of the size study, even on MRPC, which has only 3,600 labeled training examples. {#c-abl-size}

## Access and licence

Google released the code and the weights on GitHub, and the pre-trained models are under the same licence as the code, Apache 2.0. {#c-lic-models}
The licence file of the repository is the Apache License, Version 2.0. {#c-lic-file}
Google writes that the code runs out of the box on CPU, GPU and Cloud TPU. {#c-hw}
As of {fact:bert/access}, the repository offers the code and all of the pre-trained checkpoints for download.

### Licence history

- {fact:bert/released.bert}: BERT-Base and BERT-Large, under {fact:bert/licence}.
- Every later release in the repository, up to the smaller models of {fact:bert/released.bert_small}, under the same licence.

The training data were not released with the models: the README describes Wikipedia and the BooksCorpus but gives no copy of them.

## Safety and evaluations

The paper has no section on safety, and Google's release notes do not report any safety evaluation.
Later work tested BERT for bias.
May et al. extend the Word Embedding Association Test from words to sentences, test sentence encoders such as ELMo and BERT for social biases, and find mixed results, with patterns that suggest the test's assumptions may not hold in general. {#c-may}

The pre-training data have their own problems.
Bandy and Vincent find evidence that the BookCorpus likely violates copyright restrictions for many books, contains thousands of duplicated books and is skewed in genre. {#c-bandy}

## Reception and impact

In October 2019 Google announced that it applies BERT models to ranking and to featured snippets in Search, and that BERT would help Search better understand one in 10 searches in English in the U.S. {#c-search}
Google wrote that some BERT models were so complex that, for the first time, it served search results with Cloud TPUs. {#c-search-tpu}

BERT became an object of study in its own right.
Rogers et al. survey over 150 studies of BERT, and call it a must-have baseline. {#c-rogers}
Tenney et al. find that BERT's layers represent the steps of the traditional NLP pipeline in order: part-of-speech tagging, parsing, named entities, semantic roles, then coreference. {#c-tenney}
Clark et al. find attention heads that attend to delimiter tokens or to fixed offsets, and others that find direct objects, determiners, objects of prepositions and coreferent mentions with high accuracy. {#c-clark-heads}
Pires et al. show that Multilingual BERT, pre-trained on 104 languages with no parallel data, transfers zero-shot across languages surprisingly well, best between languages of similar type. {#c-pires}

Many later models change one part of BERT's recipe.
RoBERTa's authors find that BERT was significantly undertrained, and that with better training it matches or beats every model published after it. {#c-roberta}
RoBERTa trains longer, with bigger batches and more data, on longer sequences, with dynamic masking and without next sentence prediction. {#c-roberta-mods}
ALBERT reduces the parameters of BERT with two techniques and replaces next sentence prediction with a loss for inter-sentence coherence. {#c-albert}
Lan et al. report that an ALBERT configuration like BERT-Large has 18x fewer parameters and trains about 1.7x faster. {#c-albert-18x}
DistilBERT distils BERT during pre-training into a model 40% smaller and 60% faster that keeps 97% of its language understanding. {#c-distilbert}
ELECTRA trains the encoder to detect replaced tokens instead of predicting masked ones, and its authors report that a model trained on one GPU for 4 days beats GPT on GLUE, a model trained with 30x more compute. {#c-electra}
In December 2024 the authors of ModernBERT wrote that encoder pipelines still often use the original BERT as their backbone, with sequences limited to 512 tokens. {#c-modernbert}

## Known issues

**Shortcuts in benchmarks.** McCoy et al. find that models trained on MNLI, BERT among them, do very poorly on the HANS evaluation set, which suggests that they rely on fallible syntactic heuristics. {#c-mccoy}
Niven and Kao show that BERT's peak of 77% on the Argument Reasoning Comprehension Task comes entirely from spurious statistical cues in the dataset, and that all models reach only random accuracy on an adversarial version. {#c-niven}

**Negation.** Ettinger finds that BERT struggles with hard inferences and role-based event prediction, and is clearly insensitive to the effect of negation in context. {#c-ettinger}

**Fine-tuning instability.** Fine-tuning BERT-Large was sometimes unstable on small datasets, so the authors ran several random restarts and chose among the runs by their scores on the development set. {#c-res-unstable}

**Sequence length.** The models were pre-trained on inputs of at most {fact:bert/context_length}, so longer texts must be cut or split.

**The [CLS] vector.** Without fine-tuning, the final hidden vector of [CLS] is not a meaningful sentence representation (see Training).

## Lineage

Before BERT, Devlin et al. saw two ways to use pre-trained language representations: as features in a task-specific model, as in ELMo, or by fine-tuning all the parameters, as in OpenAI GPT. {#c-strategies}
ELMo's word vectors are learned functions of the internal states of a deep bidirectional language model pre-trained on a large corpus. {#c-elmo}
BERT differs from GPT, which pre-trains a left-to-right language model, and from ELMo, which concatenates separately trained left-to-right and right-to-left models. {#c-contrast}
BERT fine-tunes like GPT, and it is bidirectional in every layer, which ELMo is only at its output.
Its main benchmark, GLUE, is a collection of tools for evaluating models across a diverse set of existing language understanding tasks. {#c-glue}
RoBERTa, ALBERT, DistilBERT, ELECTRA and ModernBERT are encoders in BERT's line, described under Reception and impact.

## References
