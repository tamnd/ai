---
tag: NEW-llama
kind: model
slug: llama
name: Llama
aliases: [LLaMA, Llama 2, Meta Llama 3, Llama 3.1, Llama 3.2, Llama 3.3, Llama 4]
area: foundation-models/open-weight-llms
level: core
summary: >-
  A family of large language models made by Meta, whose weights Meta has shared since 2023 under licences of its own, from the dense LLaMA transformers to the mixture-of-experts Llama 4.
history:
  year: 2023
  note: "LLaMA was announced on 24 February 2023 (Meta AI, 2023); the later generations are Llama 2, Llama 3 and its point releases, and Llama 4."
sources:
  - {paper: "Touvron et al. (2023a)", title: "LLaMA: Open and Efficient Foundation Language Models", locator: "Sections 2.1 to 2.4, Tables 1, 2 and 9", note: "the architecture, the data, the sizes and the training of LLaMA"}
  - {paper: "Touvron et al. (2023b)", title: "Llama 2: Open Foundation and Fine-Tuned Chat Models", locator: "pp. 4 to 13, Tables 1 and 3, p. 47", note: "the changes from LLaMA, the data, the fine-tuning and the results of Llama 2"}
  - {paper: "Grattafiori et al. (2024)", title: "The Llama 3 Herd of Models", locator: "pp. 1 to 9, Section 3.2, Table 13 on p. 31", note: "the architecture, the data, the compute and the results of the Llama 3.1 models"}
  - {paper: "Meta (2023 to 2025)", title: "the announcements of LLaMA, Llama 2, Llama 3, Llama 3.1, Llama 3.2 and Llama 4", locator: "p. 1 of each"}
  - {paper: "Meta (2023 to 2025)", title: "the model cards of LLaMA, Llama 3.3 and Llama 4, and the community license agreements of Llama 2 to Llama 4", locator: "p. 1 of each"}
  - {paper: "Open Source Initiative (2023, 2024)", title: "Meta's LLaMa license is not Open Source; The Open Source AI Definition 1.0", locator: "p. 1 of each"}
  - {paper: "Zuckerberg (2024)", title: "Open Source AI Is the Path Forward", locator: "p. 1"}
role: family
---

Llama is a family of large language models (LLMs) made by Meta, first announced on {fact:llama/released.llama}.
Meta has shared the weights of each generation with outside users: on request for research at first, and for download from Llama 2 on.
The early models are dense transformers, and the Llama 4 models use a mixture of experts.
The licences are Meta's own, and whether they make Llama open source is disputed.

## Overview

Touvron et al. introduce LLaMA as a collection of foundation language models from 7B to 65B parameters, trained on publicly available datasets only. {#c-llama-collection}
They report that LLaMA-13B outperforms GPT-3 (175B) on most benchmarks. {#c-llama-gpt3}
Meta released LLaMA under a noncommercial license focused on research, and granted access case by case to researchers in academia, government, civil society and industry laboratories. {#c-llama-noncommercial}

Llama 2 followed on {fact:llama/released.llama_2}, and Meta announced it as free for research and commercial use. {#c-llama2-free}
Meta Llama 3, Llama 3.1, Llama 3.2 and Llama 3.3 came out in 2024, and Llama 4 on {fact:llama/released.llama_4}.
Meta describes the Llama 4 models as its first models that use a mixture-of-experts (MoE) architecture. {#c-llama4-moe}
Each release is listed with its date, sizes and licence under Releases and versions.

## Architecture

LLaMA is a transformer with three changes from the original design. {#c-arch-llama}
It normalizes the input of each sub-layer with root mean square layer normalization (RMSNorm), a pre-normalization taken from GPT-3.
It replaces the ReLU non-linearity with the SwiGLU activation function, after PaLM.
It drops absolute positional embeddings and adds rotary positional embeddings (RoPE) at each layer, after GPTNeo. {#c-arch-sources}

Llama 2 keeps this design, and its main architectural changes are a longer context and grouped-query attention (GQA). {#c-arch-llama2}
Only the 34B and 70B models of Llama 2 use GQA. {#c-llama2-gqa}
The context window grew from {fact:llama/context_length.llama} for Llama 1 to {fact:llama/context_length.llama_2} for Llama 2. {#c-context-2048-4096}

For Llama 3, Meta chose a relatively standard decoder-only transformer, with a tokenizer of 128K tokens and GQA in both the 8B and 70B sizes. {#c-arch-llama3}
Grattafiori et al. write that it does not deviate significantly from Llama and Llama 2 in its architecture. {#c-arch-llama3-same}
They chose a dense model over a mixture of experts to maximize training stability. {#c-arch-dense}
They use GQA with 8 key-value heads, and a mask that stops self-attention between different documents in the same sequence. {#c-arch-llama3-gqa8}
They raise the base frequency of RoPE to 500,000 to better support longer contexts. {#c-arch-rope-base}

Meta writes that the 11B and 90B models of Llama 3.2, the first Llama models to support vision tasks, needed a different architecture that supports image reasoning. {#c-arch-32-vision}

Llama 4 changes the design more.
The models are natively multimodal, with early fusion of text and vision tokens into one model backbone. {#c-arch-llama4-fusion}
Llama 4 Maverick alternates dense and MoE layers, and each token goes to a shared expert and to one of 128 routed experts. {#c-arch-llama4-moe}
Meta calls the design iRoPE: some attention layers have no positional embeddings, and most layers use RoPE. {#c-arch-irope}

## Training data

LLaMA uses only publicly available data. {#c-data-llama-public}
English CommonCrawl is 67.0% of the mixture, C4 15.0%, GitHub, Wikipedia and Books 4.5% each, ArXiv 2.5% and StackExchange 2.0%. {#c-data-llama-mix}
LLaMA 7B and 13B saw {fact:llama/training.tokens.llama_7b}, and LLaMA 33B and 65B saw {fact:llama/training.tokens.llama_65b}. {#c-tokens-llama}

Llama 2 uses data from publicly available sources and no data from Meta's products or services. {#c-data-llama2}
Meta tried to remove data from sites with a high volume of personal information about private individuals. {#c-data-llama2-pii}
The Llama 2 paper gives {fact:llama/training.tokens.llama_2} of training data, but not the share of each source. {#c-tokens-llama2}

Meta writes that Llama 3 saw over 15T tokens from publicly available sources, seven times the data of Llama 2, with four times more code. {#c-data-llama3}
Its sources hold knowledge until the end of 2023. {#c-data-llama3-cutoff}
The 405B model of Llama 3.1 saw {fact:llama/training.tokens.llama_3_1_405b}. {#c-tokens-405b}

For Llama 4, the model card lists publicly available and licensed data and information from Meta's products and services, including public Instagram and Facebook posts and interactions with Meta AI. {#c-data-llama4}
It gives {fact:llama/training.tokens.llama_4_scout} for Llama 4 Scout and {fact:llama/training.tokens.llama_4_maverick} for Llama 4 Maverick. {#c-tokens-llama4-card}
The announcement gives one figure for the whole mixture: more than 30 trillion tokens of text, image and video data. {#c-tokens-llama4-blog}
The sources do not list the share of each source for any generation after LLaMA.

## Training

The LLaMA authors report about 380 tokens per second per GPU on {fact:llama/training.gpus.llama_65b} A100 GPUs, so training the 65B model on 1.4T tokens takes about 21 days. {#c-train-llama-gpus}

Llama 2-Chat is the dialogue version of Llama 2. {#c-llama2-chat}
Its authors explored two main algorithms for fine-tuning it with reinforcement learning from human feedback (RLHF): Proximal Policy Optimization (PPO) and rejection sampling fine-tuning. {#c-train-llama2-rlhf}

The flagship model of Llama 3.1 took {fact:llama/training.compute_flop.llama_3_1_405b} of pre-training, which its authors put at almost 50 times more than the largest version of Llama 2. {#c-train-compute}
It was trained on up to {fact:llama/training.gpus.llama_3_1_405b} H100 GPUs. {#c-train-405b-gpus}
Its post-training is based on supervised finetuning (SFT), rejection sampling and direct preference optimization (DPO), rather than more complex reinforcement learning algorithms. {#c-train-llama3-post}

Meta reports that it codistilled Llama 4 Maverick from Llama 4 Behemoth as a teacher model. {#c-train-codistill}

## Releases and versions

| Release | Date | Sizes | Licence |
|---|---|---|---|
| LLaMA | {fact:llama/released.llama} | 7B, 13B, 33B, 65B | {fact:llama/licence.llama} |
| Llama 2 | {fact:llama/released.llama_2} | 7B, 13B, 70B | {fact:llama/licence.llama_2} |
| Meta Llama 3 | {fact:llama/released.llama_3} | 8B, 70B | {fact:llama/licence.llama_3} |
| Llama 3.1 | {fact:llama/released.llama_3_1} | 8B, 70B, 405B | {fact:llama/licence.llama_3_1} |
| Llama 3.2 | {fact:llama/released.llama_3_2} | 1B, 3B, 11B, 90B | {fact:llama/licence.llama_3_2} |
| Llama 3.3 | {fact:llama/released.llama_3_3} | 70B | {fact:llama/licence.llama_3_3} |
| Llama 4 | {fact:llama/released.llama_4} | Scout, Maverick | {fact:llama/licence.llama_4} |

Meta released Llama 2 at 7B, 13B and 70B, and also trained 34B models that the paper reports on but that were not released. {#c-llama2-sizes}
Meta Llama 3 came with pretrained and instruction-fine-tuned models of 8B and 70B parameters. {#c-llama3-sizes}
In April 2024, Meta wrote that models of over 400B parameters were still training. {#c-llama3-400b}
Llama 3.1 expanded the context to 128K tokens, added support for eight languages, and included Llama 3.1 405B. {#c-llama31}
Llama 3.2 added vision models of 11B and 90B and text-only models of 1B and 3B for edge and mobile devices. {#c-llama32}
Llama 3.3 is a single 70B text model, released as 70B Instruct, with a context of 128k. {#c-llama33}
Llama 4 Scout has 17 billion active parameters and 16 experts, and Llama 4 Maverick has 17 billion active parameters and 128 experts. {#c-llama4-models}
Their total sizes are {fact:llama/params.total.llama_4_scout} and {fact:llama/params.total.llama_4_maverick}. {#c-llama4-total}
Meta did not release Llama 4 Behemoth, a larger MoE model that was still training at the announcement. {#c-behemoth}

## Results

The table gives the five-shot accuracy of pretrained base models on Massive Multitask Language Understanding (MMLU), as Meta reports it.
Each source uses its own evaluation setup, so the rows are not one controlled comparison.

| Model | MMLU accuracy | Shots | Source |
|---|---|---|---|
| LLaMA 7B | 35.1 | 5 | Touvron et al. (2023a), Table 9 |
| LLaMA 65B | 63.4 | 5 | Touvron et al. (2023a), Table 9 |
| Llama 2 7B | 45.3 | 5 | Touvron et al. (2023b), Table 3 |
| Llama 2 70B | 68.9 | 5 | Touvron et al. (2023b), Table 3 |
| Llama 3.1 70B | 79.3 | 5 | Llama 4 model card; Grattafiori et al. (2024), Table 13 (p. 31) |
| Llama 3.1 405B | 85.2 | 5 | Llama 4 model card; Grattafiori et al. (2024), Table 13 (p. 31) |
| Llama 4 Scout | 79.6 | 5 | Llama 4 model card |
| Llama 4 Maverick | 85.5 | 5 | Llama 4 model card |

The LLaMA paper reports five-shot MMLU accuracy of 35.1 for LLaMA 7B and 63.4 for LLaMA 65B. {#c-mmlu-llama}
The Llama 2 paper reports MMLU (5 shot) of 45.3 for Llama 2 7B and 68.9 for Llama 2 70B, and gives 63.4 again for Llama 1 65B. {#c-mmlu-llama2}
Grattafiori et al. report MMLU scores of 66.7, 79.3 and 85.2 for the pre-trained Llama 3 8B, 70B and 405B models. {#c-mmlu-herd}
The Llama 4 model card reports 5-shot MMLU of 79.3 for Llama 3.1 70B, 85.2 for Llama 3.1 405B, 79.6 for Llama 4 Scout and 85.5 for Llama 4 Maverick. {#c-mmlu-l4card}

## Access and licence

Meta released LLaMA for research on request. {#c-llama-access}
From Llama 2 on, the weights can be downloaded under a community license agreement and an Acceptable Use Policy that prohibits certain use cases. {#c-aup}
For Llama 4, Meta made Scout and Maverick available for download on llama.com and Hugging Face. {#c-llama4-download}

### Licence history

- {fact:llama/released.llama}: LLaMA, under a "Non-commercial bespoke license". {#c-lic-llama}
- {fact:llama/released.llama_2}: the "Llama 2 Community License Agreement". A licensee with more than 700 million monthly active users must request a license from Meta, which Meta may grant in its sole discretion. {#c-lic-llama2-mau}
  The licence also forbids using Llama 2 or its outputs to improve any other large language model, apart from Llama 2 and its derivative works. {#c-lic-llama2-improve}
- {fact:llama/released.llama_3}: the "Meta Llama 3 Community License Agreement". It keeps the 700 million users clause and the bar on improving other models, and asks for "Built with Meta Llama 3" to be displayed. {#c-lic-llama3}
- {fact:llama/released.llama_3_1}: the "Llama 3.1 Community License Agreement". Meta writes that it changed the licence to let developers use the outputs of Llama models to improve other models. {#c-lic-31-outputs}
  The display text becomes "Built with Llama", and the 700 million users clause stays. {#c-lic-31-built}
- {fact:llama/released.llama_3_2}: the "Llama 3.2 Community License Agreement", which keeps the 700 million users clause. {#c-lic-32}
- {fact:llama/released.llama_3_3}: the "Llama 3.3 Community License Agreement", which keeps the 700 million users clause. {#c-lic-33}
- {fact:llama/released.llama_4}: the "Llama 4 Community License Agreement", effective from that date, which keeps the 700 million users clause. {#c-lic-4}

### Open weights and open source

The question is whether Llama is open source, or open weights under restrictive terms.
Open weights means here that anyone who accepts the licence can download the trained parameters.
Open source has a formal meaning for software, the Open Source Definition (OSD) of the Open Source Initiative (OSI), and since 2024 one for AI systems.

**Meta.**
The Llama 2 authors write that they believe the open release of LLMs, when done safely, will be a net benefit to society. {#c-meta-open-release}
Mark Zuckerberg writes that Meta is committed to open source AI, and argues that open source AI is good for developers, for Meta and for the world. {#c-zuck-open-source}
He argues that open source AI will be safer than the alternatives, because the systems are more transparent and can be widely scrutinized. {#c-zuck-safety}
Meta announced Llama 4 as part of its commitment to open source. {#c-llama4-open-source}

**The Open Source Initiative.**
OSI states that Meta's license for the LLaMa models and code is not Open Source. {#c-osi-not-os}
It argues that an Open Source license may not discriminate against persons, groups or fields of endeavor. {#c-osi-osd}
In its view, the Llama terms restrict commercial use for some users, and the Acceptable Use Policy restricts some uses. {#c-osi-restrict}
The Open Source AI Definition 1.0 asks for the freedoms to use a system for any purpose without asking for permission, and to study, modify and share it. {#c-osaid}
It also asks for data information detailed enough that a skilled person can build a substantially equivalent system, together with the code and the parameters. {#c-osaid-data}

This page does not decide between the positions.

## Safety and evaluations

The Llama 2 paper describes its approach to fine-tuning and safety improvements of Llama 2-Chat in detail, so that others can build on it. {#c-llama2-safety}
With the Llama 3.1 models, Meta publicly released Llama Guard 3, a model for input and output safety. {#c-llamaguard3}
The Llama 3.3 model card names three critical risk areas of its mitigation work: CBRNE helpfulness, child safety and cyber attack enablement. {#c-33-risks}
It also reports recurring red teaming exercises with experts. {#c-33-redteam}

## Reception and impact

Meta reports more than {fact:llama/requests.llama} requests for access to Llama 1 from researchers. {#c-requests}
It named Microsoft as its preferred partner for Llama 2. {#c-microsoft}
By the release of Llama 3.1, Meta reported more than {fact:llama/downloads.all_versions} downloads of all Llama versions. {#c-downloads}
Zuckerberg writes that Amazon, Databricks and NVIDIA launched services to help developers fine-tune and distill their own Llama models. {#c-ecosystem}

## Known issues

**Training tokens of Llama 2.** The Llama 2 paper reports training on 2 trillion tokens of data. {#c-tokens-llama2-known}
The Llama 3 paper compares about 15T multilingual tokens for Llama 3 with 1.8T tokens for Llama 2. {#c-tokens-llama2-18}
The fact on this page shows both values.

**Training tokens of Llama 4.** The model card gives a figure for each model and the announcement one figure for the mixture, so the two cannot be compared directly.

**Outputs.** The Llama 3.3 model card says that the model may in some instances produce inaccurate, biased or other objectionable responses. {#c-33-limits}

## Lineage

LLaMA starts from the transformer of Vaswani et al. (2017), with RMSNorm, SwiGLU and RoPE. {#c-lineage-l1}
The Llama 2 authors call it an updated version of Llama 1, trained on a different mix of publicly available data. {#c-lineage-l2}
Llama 3 uses GQA in both its 8B and 70B sizes and a tokenizer of 128K tokens. {#c-lineage-l3}
Llama 4 moves to a mixture of experts and early fusion, and Llama 4 Maverick is codistilled from Llama 4 Behemoth. {#c-lineage-l4}

## References
