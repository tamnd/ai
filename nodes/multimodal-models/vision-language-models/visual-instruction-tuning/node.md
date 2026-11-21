---
tag: NEW-visual-instruction-tuning
kind: method
slug: visual-instruction-tuning
name: Visual instruction tuning
aliases: [LLaVA, LLaVA-1.5, large multimodal model, LMM, large vision-language model, LVLM, vision-language model, VLM, Flamingo, BLIP-2, Q-Former, Perceiver Resampler, gated cross-attention, visual connector, multimodal instruction tuning, POPE]
area: multimodal-models/vision-language-models
level: core
summary: >-
  Connecting a pretrained image encoder to a pretrained language model with a small trained connector, then fine-tuning on image-grounded instructions so the model answers questions about images.
purpose: building a model that reads images and text together and answers in free text, by reusing a pretrained vision encoder and a pretrained language model instead of training a multimodal network from scratch
uses: [contrastive-language-image-pretraining, vision-transformer, decoder-only-transformer, instruction-tuning, fine-tuning, multilayer-perceptron, cross-entropy-loss@al, embedding@al]
related_to: [hallucination, few-shot-in-context-learning, llama, lora, multi-head-attention]
hyperparameters:
  - {name: connector, meaning: "the trained module that turns image features into inputs the language model can read", typical: "a linear projection in LLaVA; a two-layer MLP in LLaVA-1.5; a 188M-parameter Q-Former with 32 queries in BLIP-2; a Perceiver Resampler with 64 outputs feeding gated cross-attention in Flamingo"}
  - {name: visual tokens per image, meaning: "how many vectors one image adds to the language model's input or cross-attention memory", typical: "256 for a ViT-L/14 at 224 pixels, 576 at 336 pixels; 32 for BLIP-2; 64 for Flamingo"}
  - {name: what is frozen, meaning: "which pretrained parts keep their weights in each stage", typical: "encoder and LLM frozen in LLaVA stage 1; encoder frozen and LLM trained in stage 2; both frozen throughout in Flamingo and BLIP-2"}
  - {name: alignment data, meaning: "the image-caption pairs used to teach the connector before instruction tuning", typical: "595K filtered CC3M pairs for LLaVA; 129M images with synthetic captions for BLIP-2; M3W, ALIGN, LTIP and VTP for Flamingo"}
  - {name: instruction data, meaning: "the image-grounded conversations, descriptions and question-answer pairs of the tuning stage", typical: "LLaVA-Instruct-158K written by text-only GPT-4; 1.2M public samples with academic VQA and formatting prompts for LLaVA-1.5"}
  - {name: learning rate and epochs, meaning: "the optimizer settings of the two stages", typical: "1 epoch at 2e-3, batch 128 for alignment and 3 epochs at 2e-5, batch 32 for instruction tuning in LLaVA"}
  - {name: image resolution, meaning: "the input size of the vision encoder, which sets the number of patches and the detail the model can see", typical: "224 pixels in LLaVA; 336 pixels in LLaVA-1.5, and grids of 336-pixel crops in LLaVA-1.5-HD"}
cost:
  train: "about 4 hours of alignment and 10 hours of instruction tuning on 8 A100s for LLaVA, about 6 and 20 hours for LLaVA-1.5; under 9 days on one 16-A100 machine for BLIP-2 with ViT-g and FlanT5-XXL; 1536 TPU chips for 15 days for Flamingo-80B, whose frozen parts must still run forward and backward at every step"
  inference: "one image-encoder pass per image, then a language-model forward pass whose context grows by the number of visual tokens, 576 per image at 336 pixels for LLaVA-1.5, against 32 or 64 for query-based connectors"
history:
  year: 2023
  people: [Haotian Liu, Chunyuan Li, Yong Jae Lee, Jean-Baptiste Alayrac, Karen Simonyan, Junnan Li, Steven Hoi, Yifan Li, Wayne Xin Zhao]
  note: "Alayrac et al. posted Flamingo (April 2022, NeurIPS 2022), joining frozen vision and language models with gated cross-attention; Li et al. posted BLIP-2 (January 2023) with the Q-Former; Liu et al. posted LLaVA (April 2023, NeurIPS 2023), which introduced GPT-4-written visual instruction data; Li et al. posted POPE (May 2023) to measure object hallucination; Liu et al. posted LLaVA-1.5 (October 2023)."
prereqs: [contrastive-language-image-pretraining, decoder-only-transformer, instruction-tuning]
same_as:
  wikidata: null
  wikipedia: null
  arxiv: ["2304.08485", "2310.03744", "2204.14198", "2301.12597", "2305.10355"]
sources:
  - {paper: "Liu, Li, Wu and Lee (2023)", title: "Visual Instruction Tuning", locator: "NeurIPS 2023, arXiv 2304.08485v2, pp. 1 to 9 and 20", note: "GPT-4-generated instruction data, the linear connector, the two training stages, LLaVA-Bench and ScienceQA results"}
  - {paper: "Liu, Li, Li and Lee (2023)", title: "Improved Baselines with Visual Instruction Tuning", locator: "arXiv 2310.03744v2, pp. 1, 3, 4, 6 and 8", note: "the MLP connector, academic VQA data, response formatting prompts, compute and the resolution finding on hallucination"}
  - {paper: "Alayrac, Donahue, Luc, Miech and others (2022)", title: "Flamingo: a Visual Language Model for Few-Shot Learning", locator: "NeurIPS 2022, arXiv 2204.14198v2, pp. 3 to 10 and 21", note: "the Perceiver Resampler, tanh-gated cross-attention, interleaved web data, ablations and limitations"}
  - {paper: "Li, Li, Savarese and Hoi (2023)", title: "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models", locator: "arXiv 2301.12597v3, pp. 1, 3, 4 and 6 to 8", note: "the Q-Former, its two-stage pretraining, cost, results and limitations"}
  - {paper: "Li, Du, Zhou, Wang, Zhao and Wen (2023)", title: "Evaluating Object Hallucination in Large Vision-Language Models", locator: "arXiv 2305.10355v3, pp. 1 and 4 to 7", note: "object hallucination, the instability of CHAIR, frequency effects and the POPE benchmark"}
---

Visual instruction tuning builds a model that answers questions about images out of two networks that were trained separately: an image encoder, usually the vision tower of a CLIP model, and a large language model.
A small connector turns the encoder's feature vectors into vectors the language model can read as if they were words, and the whole system is then fine-tuned on examples of an image, an instruction about it and a good answer.
LLaVA, from 2023, made the recipe simple and cheap: a linear map as the connector, instruction data written by a text-only GPT-4 from captions and boxes, and two short training stages.
Flamingo and BLIP-2 came before it with heavier connectors, and LLaVA-1.5 tuned the recipe into a common baseline for open vision-language models.

## In plain words

A language model can only read token embeddings.
An image encoder turns a picture into a grid of vectors that mean something about the picture, but they are in a different space from word embeddings.
The connector is a translator between the two: it is trained so that, when its output is placed in front of a question, the language model behaves as if it had read a description of the picture.

Training goes in two steps.
First, teach the translator alone, using pictures and their captions, with both big networks frozen.
Then teach the system to follow instructions about pictures, such as "what is unusual about this image?", using a set of example conversations.
The language model already knows how to reason and talk; the steps only have to teach it to look.

## Setting

The input is one or more images with text, and the output is free text.
Three pretrained or trained parts are involved:

- an image encoder g, such as a CLIP Vision Transformer from contrastive language-image pretraining, which maps an image X_v to features Z_v = g(X_v), one vector per patch;
- a connector, which maps Z_v to vectors H_v of the language model's embedding width;
- a decoder-only Transformer language model, which reads [H_v, word embeddings of the instruction] and generates the answer token by token.

Training uses the next-token cross-entropy loss on the answer tokens only, as in instruction tuning of text models.
The connector puts image information into the language model's embedding space, so the visual tokens are a learned soft prompt.

## The method

**LLaVA's architecture.**
LLaVA connects the open-set visual encoder of CLIP with the language decoder Vicuna and fine-tunes end-to-end on the generated instructional vision-language data. {#c-lv-vicuna}
LLaVA uses a simple linear layer to connect image features into the word embedding space: a trainable projection matrix W converts the visual feature Zv = g(Xv) of the CLIP encoder into language embedding tokens Hv = W · Zv, which have the same dimensionality as the word embedding space in the language model. {#c-lv-proj}
Vicuna is a chat model fine-tuned from Llama, so LLaVA starts from a language model that already follows text instructions.

**Making the instruction data.**
No large set of image conversations existed, so LLaVA wrote one with a model that cannot see.
To create instruction-following data involving visual content, LLaVA uses language-only GPT-4 or ChatGPT as the teacher, both of which accept only text as input, and encodes an image for the text-only GPT through two symbolic representations: captions that describe the visual scene from various perspectives, and bounding boxes that localize the objects in the scene. {#c-lv-data}
GPT-4 is shown the captions and boxes of a COCO image and a few hand-written examples, and asked to write a conversation about the image, a detailed description, or a question that needs reasoning.
LLaVA collects 158K unique language-image instruction-following samples in total, including 58K in conversations, 23K in detailed description, and 77k in complex reasoning. {#c-lv-158}

**Stage 1: feature alignment.**
In stage 1, pre-training for feature alignment, CC3M is filtered to 595K image-text pairs converted to single-turn instructions, and both the visual encoder and LLM weights are kept frozen while the likelihood is maximized with the projection matrix W as the only trainable parameters, which can be understood as training a compatible visual tokenizer for the frozen LLM. {#c-lv-s1}

**Stage 2: end-to-end fine-tuning.**
In stage 2 the visual encoder weights stay frozen and both the pre-trained weights of the projection layer and the LLM are updated, so the trainable parameters are θ = {W, ϕ}. {#c-lv-s2}
All LLaVA models are trained with 8× A100s, pre-training on the filtered CC-595K subset for 1 epoch with a learning rate of 2e-3 and a batch size of 128, and fine-tuning on LLaVA-Instruct-158K for 3 epochs with a learning rate of 2e-5 and a batch size of 32. {#c-lv-hp}
With 8× A100s, pretraining on CC-595K completes within 4 hours and finetuning on Instruct-158K completes within 10 hours. {#c-lv-hours}

**LLaVA-1.5.**
LLaVA-1.5 makes simple modifications to LLaVA, using CLIP-ViT-L-336px with an MLP projection and adding academic-task-oriented VQA data with response formatting prompts; the final 13B checkpoint uses 1.2M publicly available data and finishes full training in about 1 day on a single 8-A100 node. {#c-l15-recipe}
The formatting prompt fixes a conflict between long chat answers and the short answers of academic benchmarks:
Liu et al. append a single response formatting prompt, Answer the question using a single word or phrase, at the end of VQA questions when short answers are wanted, because LLaVA falls short on academic benchmarks that require short-form answers and tends to answer yes for yes/no questions due to the lack of such data in the training distribution. {#c-l15-fmt}
Because of the increased input resolution of 336², training LLaVA-1.5 takes about twice as long as LLaVA: about 6 hours of pretraining and about 20 hours of visual instruction tuning, using 8× A100s. {#c-l15-hours}

**Flamingo: gated cross-attention.**
Flamingo, the first of these systems at scale, keeps the language model's input sequence for text and lets it look at the image through new cross-attention layers instead. These are multi-head attention layers whose queries come from the text and whose keys and values come from the visual tokens, where the self-attention of the language model takes all three from the text.
Flamingo models leverage two complementary pre-trained and frozen models, a vision model which can perceive visual scenes and a large language model which performs a basic form of reasoning, and novel architecture components are added in between these models to connect them. {#c-fl-frozen}
Flamingo builds on the 1.4B, 7B, and 70B parameter Chinchilla models, giving Flamingo-3B, Flamingo-9B and Flamingo-80B, and uses a pretrained and frozen NFNet-F6 vision encoder trained with a contrastive objective on image and text pairs. {#c-fl-sizes}
The Perceiver Resampler takes as input a variable number of image or video features from the vision encoder and produces a fixed number of visual outputs (64), reducing the computational complexity of the vision-text cross-attention, by learning a predefined number of latent input queries which are fed to a Transformer and cross-attend to the visual features. {#c-fl-resampler}
Flamingo freezes the pretrained LM blocks and inserts gated cross-attention dense blocks between the original layers, trained from scratch; each new layer's output is multiplied by tanh(α) before it is added to the residual, where α is a layer-specific learnable scalar initialized to 0, so at initialization the model output matches that of the pretrained LM. {#c-fl-tanh}
At a given text token, Flamingo attends to the visual tokens of the image that appeared just before it in the interleaved sequence, rather than to all previous images; it uses only up to 5 images per sequence when training on interleaved data yet benefits from sequences of up to 32 shots of images or videos and texts during evaluation. {#c-fl-mask}
Being able to read many images in one sequence is what gives Flamingo few-shot in-context learning: a prompt can hold several (image, answer) examples followed by a new image.
The MultiModal MassiveWeb (M3W) dataset extracts text and images from the HTML of approximately 43 million webpages, and from each document a random subsequence of L = 256 tokens is sampled with up to the first N = 5 images; Flamingo also trains on ALIGN with 1.8 billion images paired with alt-text, LTIP with 312 million image and text pairs, and VTP with 27 million short videos. {#c-fl-m3w}
Flamingo is trained by minimizing a weighted sum of per-dataset expected negative log-likelihoods of text given the visual inputs; tuning the per-dataset weights is key to performance, and gradients are accumulated over all datasets, which outperforms a round-robin approach. {#c-fl-loss}

**BLIP-2: the Q-Former.**
BLIP-2 puts a learned bottleneck between the two frozen networks and trains it in two stages.
The Q-Former of BLIP-2 is initialized with the pre-trained weights of BERT-base with randomly initialized cross-attention layers inserted every other transformer block, contains 188M parameters, and uses 32 learnable queries of dimension 768, so the output Z (32 × 768) is much smaller than the frozen image features, for example 257 × 1024 for ViT-L/14. {#c-b2-qf}
In the representation learning stage BLIP-2 jointly optimizes three objectives that share the same input format and model parameters, image-text contrastive learning, image-grounded text generation and image-text matching, each with a different attention mask between queries and text; for contrastive learning it takes the highest similarity between any query output and the text embedding as the image-text similarity. {#c-b2-obj}
In the generative stage BLIP-2 uses a fully-connected layer to linearly project the output query embeddings Z into the dimension of the LLM's text embedding, and the projected query embeddings are prepended to the input text embeddings, where they function as soft visual prompts that condition the frozen LLM. {#c-b2-fc}
BLIP-2 pre-trains for 250k steps in the first stage and 80k steps in the second on 129M images, and on a single 16-A100(40G) machine the model with ViT-g and FlanT5-XXL requires less than 6 days for the first stage and less than 3 days for the second stage. {#c-b2-cost}

## Derivation

The full derivation is in `derivations/vision-language-arithmetic.md`, and `vision-language-arithmetic.check.py` checks every number.

**An always-yes model on POPE.**
On a POPE split with as many yes questions as no questions, a model that always answers yes has accuracy 0.5, precision 0.5, recall 1 and F1 2/3, which is the 50.00 accuracy and 66.67 F1 reported for MultiModal-GPT on the popular and adversarial splits. {#c-d-pope}
So a POPE F1 near 66.67 together with a yes rate near 100% means the model is not looking at the question.

**The token cost of an image.**
A ViT-L/14 at 224 pixels makes (224/14)² = 256 patch tokens, 257 with the class token, and at 336 pixels 24² = 576; feeding five images through a 336-pixel linear projection costs 2,880 visual tokens in the LLM context, against 160 with 32 Q-Former queries and 320 with 64 Perceiver Resampler outputs. {#c-d-tokens}
Query-based connectors buy a short, fixed context at the price of a bottleneck. LLaVA-style linear connectors keep every patch and pay for it in context length.

**Why the tanh gate starts safe.**
With y' = y + tanh(α) f(y, x) and α = 0 the gated block is the identity, and since tanh'(0) = 1 the first gradient reaches α as the inner product of the upstream gradient with f(y, x), while every weight inside f gets a gradient multiplied by tanh(0) = 0. {#c-d-gate}
The first steps can only open gates, and the frozen model's behaviour is disturbed only as fast as the new layers prove useful.

**Trainable parameters.**
188 million Q-Former parameters against 10.2 billion trainable parameters for Flamingo-80B is a ratio of 54.3. {#c-d-ratio}
From a 1024-wide CLIP feature to a 5120-wide LLM embedding, a linear projection has 1024 × 5120 = 5,242,880 weights, and a two-layer MLP with a 5120-wide hidden layer has 1024 × 5120 + 5120 × 5120 = 31,457,280 weights, under 0.25% of a 13-billion-parameter LLM. {#c-d-params}

**Data and compute.**
58K conversations, 23K detailed descriptions and 77K complex reasoning samples add up to 158K. {#c-d-sum}
About 26 hours on 8 A100s is about 208 A100-hours for LLaVA-1.5, while 1536 TPU chips for 15 days is 552,960 chip-hours for Flamingo, about 2,658 times more device-hours. {#c-d-compute}

## Worked example

Take a toy version of stage 1, as in the NumPy implementation.
The frozen language model has ten object words; its next-token logits are its tied embedding matrix E times the mean of the input embeddings.
Each image is four noisy copies of a 16-dimensional prototype vector for its object.
The prompt is three fixed tokens, "what is this", and the target is the object word.

Before training, W is near zero, the four visual tokens are near zero, and the mean input is the mean of the three prompt embeddings.
The model gives the same prediction for every image. With the seed used in the tests that prediction is not an object word, so test accuracy is 0.

Gradient descent on W alone, with the cross-entropy loss of the object word, moves W so that W z lands near the embedding of the right word for each prototype z.
After 300 steps the training loss has fallen from 4.00 nats to 0.0013, and all 300 new noisy test images get the right word, while E has not changed at all.
This is LLaVA's "compatible visual tokenizer for the frozen LLM": W has learned to write each image in the language model's own vocabulary space.

The PyTorch implementation repeats the experiment with a small causal Transformer pretrained on text.
It then shows the yes bias that POPE measures.
Fine-tuning on balanced yes-or-no questions gives F1 above 0.9 with a yes rate near 50%.
Fine-tuning on the same images with only yes answers gives a yes rate above 95%, accuracy near 0.5 and F1 near 2/3, the always-yes numbers of the derivation.

## Hyperparameters

The connector and the number of visual tokens set the trade-off between detail and context length.
A linear or MLP connector at 336 pixels gives 576 tokens per image and keeps every patch. A Q-Former or resampler gives 32 or 64.
What is frozen decides the cost and the risk of forgetting: frozen language models keep their text skills, and unfrozen ones adapt more in stage 2.
The data of stage 2 decide the behaviour more than the architecture does.
Mixing in short-answer academic data with a formatting prompt changes benchmark scores and the yes bias, and raising the image resolution reduces hallucinated details.

## Cost

Training cost is dominated by forward and backward passes through the language model, even when it is frozen, because the gradient for the connector has to flow back through it.
The connector itself is small: a few million weights for a linear map, about 31 million for a two-layer MLP into a 5120-wide model.
At inference, each image adds its visual tokens to the context, so a linear connector at high resolution makes multi-image prompts long.
Flamingo pays instead with extra cross-attention layers in every few blocks, but its context grows only by 64 vectors per image.

## Evidence

**LLaVA.**
LLaVA-Bench is scored by feeding the question, the ground-truth textual descriptions and the responses of both assistants to a text-only GPT-4 judge that rates helpfulness, relevance, accuracy and level of detail on a scale of 1 to 10. {#c-lv-judge}
LLaVA yields an 85.1% relative score compared with GPT-4 on a synthetic multimodal instruction-following dataset, and when fine-tuned on Science QA the synergy of LLaVA and GPT-4 reaches an accuracy of 92.53%. {#c-lv-85}
On LLaVA-Bench (COCO), with instruction tuning the model's ability of following user instructions improves by over 50 points, and adding a small amount of detailed description and complex reasoning questions improves the overall capability by 7 points. {#c-lv-50}
On ScienceQA, LLaVA using the visual features before the last layer, predicting reasons first and trained for 12 epochs yields 90.92% accuracy, close to the previous 91.68%. {#c-lv-sqa}
The LLaVA-Bench numbers are judged by GPT-4, not by humans or exact-match metrics, so they measure agreement with GPT-4's preferences.

**LLaVA-1.5.**
Merely including VQAv2 in training improves LLaVA's score on MME from 809.6 to 1323.8, and replacing the linear projection with a two-layer MLP connector improves the multimodal capabilities further. {#c-l15-mme}
LLaVA-1.5 outperforms 80B IDEFICS, a Flamingo-like model with billions of trainable parameters for cross-modal connection, which leads the authors to rethink the benefits of the vision samplers and the necessity of additional large-scale pretraining for multimodal instruction-following. {#c-l15-rethink}

**Flamingo.**
On 6 of 16 tasks, Flamingo also outperforms fine-tuned models despite using only 32 task-specific examples, around 1000 times less task-specific training data. {#c-fl-1000}
Removing the interleaved image-text dataset M3W leads to a decrease of more than 17% in performance while removing the conventional paired image-text pairs also decreases performance by 9.8%. {#c-fl-abl-data}
Without the 0-initialized tanh gating the overall score drops by 4.2%, and disabling the gating leads to training instabilities. {#c-fl-abl-gate}
Inserting GATED XATTN-DENSE blocks every fourth block accelerates training by 66% while only decreasing the overall score by 1.9%, and the blocks are added every fourth layer for Flamingo-9B and every seventh for Flamingo-80B. {#c-fl-abl-freq}
The NFNet-F6 vision encoder has a +5.8% advantage over the CLIP ViT-L/14 and +8.0% over a smaller NFNet-F0 encoder. {#c-fl-abl-vis}
Training the language model from scratch decreases performance by 12.9%, and fine-tuning the pretrained language model also leads to a drop of 8.0%, an instance of catastrophic forgetting. {#c-fl-abl-lm}
Training Flamingo-80B took a data-centre run.
In the compute details, the main run was trained on 1536 TPU chips for 15 days. {#c-fl-compute}

**BLIP-2.**
BLIP-2 outperforms Flamingo80B by 8.7% on zero-shot VQAv2 with 54x fewer trainable parameters. {#c-b2-87}
On OK-VQA BLIP-2 comes second to Flamingo80B, which the authors attribute to OK-VQA focusing more on open-world knowledge and to the 70B Chinchilla language model holding more knowledge than the 11B FlanT5XXL; a stronger image encoder or a stronger LLM both lead to better performance. {#c-b2-okvqa}
Without representation learning, both types of LLMs give substantially lower performance on zero-shot VQA, and OPT suffers from catastrophic forgetting where performance drastically degrades as training proceeds. {#c-b2-noreps}

## Limitations and failure modes

**Object hallucination.**
The language model's prior can speak louder than the image.
This is a visual form of hallucination in language models.
Object hallucination in large vision-language models refers to generating contents that are inconsistent with ground-truth objects in the given image, and objects that frequently appear in the visual instructions or co-occur with the image objects are prone to be hallucinated. {#c-pope-def}
Although two captioning instructions have similar meanings, models prompted by the second can give doubled values of the CHAIR metrics compared with the first, and the performance order of some models also changes. {#c-pope-chair}
The HR_A@10 and HR_C@10 (dining table) of all evaluated models are near 0.5 and 0.6, so on average about half of the hallucinated objects in each image belong to the top 10 most frequently appearing objects. {#c-pope-hr}
POPE replaces free captions with yes-or-no polling:
POPE formulates object hallucination evaluation as a binary classification task that prompts the model to output Yes or No to questions of the form Is there a <object> in the image?, with the ratio between ground-truth and nonexistent objects set at 1:1, and samples the nonexistent objects at random, from the most frequent objects, or from the objects that co-occur most with the ground-truth objects. {#c-pope-method}
On 500 MSCOCO images with 6 questions each, LLaVA, MultiModal-GPT and mPLUG-Owl have F1 scores below 70 and answer Yes near 99% of the time, and the performance of every model decreases from the random setting to the popular and adversarial settings. {#c-pope-yes}
LLaVA answers yes when asked if strawberry-flavored yogurt is present although the fridge contains only yogurt and strawberries, which indicates that at times it perceives the image as a bag of patches. {#c-lv-bag}
LLaVA-1.5 points at a cause on the data side:
Hallucination on detailed descriptions is significantly reduced when the model's inputs are scaled to higher resolutions like 448², which suggests that models learn to hallucinate when the input resolution is not sufficient to discern the details in the training data. {#c-l15-res}

**Inherited weaknesses.**
Flamingo inherits the weaknesses of pretrained language models, whose priors may play a role in occasional hallucinations and ungrounded guesses; its classification performance lags behind that of contrastive models; and in-context learning is highly sensitive to the demonstrations, with inference cost and performance that scale poorly with the number of shots. {#c-fl-lim}

**No in-context learning from single-image data.**
Experiments with BLIP-2 do not observe an improved VQA performance when the LLM is given in-context VQA examples, which the authors attribute to a pretraining dataset that contains only a single image-text pair per sample. {#c-b2-icl}
Flamingo's interleaved web pages are what taught it to use several examples in one prompt.

**Evaluation by a language model.**
Benchmarks scored by GPT-4 inherit the judge's biases, and a model trained on GPT-4-written data may be rewarded for sounding like GPT-4.

## Variants

- **Linear connector (LLaVA).** One matrix; every patch becomes a token.
- **MLP connector (LLaVA-1.5).** Two layers with a GELU; more capacity at little cost.
- **Q-Former (BLIP-2, InstructBLIP).** Learned queries pretrained with contrastive, matching and generation losses, then projected into a frozen language model.
- **Perceiver Resampler with gated cross-attention (Flamingo, IDEFICS, OpenFlamingo).** The language model attends to a fixed set of visual vectors through new layers instead of through its input.
- **High-resolution tiling (LLaVA-1.5-HD).** The image is split into crops encoded separately, plus a downsampled global view.
- **Parameter-efficient tuning.** LoRA adapters on the language model in stage 2, instead of full fine-tuning, cut the memory needed to train it.

## When to use it

Use it to build an assistant that answers open questions about images, describes them or follows instructions that refer to them, when a pretrained language model and a pretrained image encoder are available.
For image classification or retrieval, a contrastive model is cheaper and usually more accurate.
For tasks that need exact counting, small text or fine spatial relations, test the model directly: those are where these systems fail first.

## Pitfalls

- Reporting accuracy or F1 on yes-or-no probes without the yes rate. An always-yes model scores 66.67 F1 on a balanced set.
- Training stage 2 only on long, descriptive answers and then scoring on short-answer benchmarks, or the reverse.
- Comparing models judged by an LLM with models scored by exact match.
- Feeding images at a lower resolution than the details the training data describe, which teaches the model to invent them.
- Unfreezing a language model in stage 1 with a randomly initialized connector, which can damage its text skills before the connector has learned anything.

## History

Earlier vision-language models were trained end to end, often from BERT-style encoders, and had to be fine-tuned for each task.
In 2022, Flamingo found that a frozen language model could be taught to look through new gated layers, and that training on web pages with interleaved images gave few-shot in-context learning.
In early 2023, BLIP-2 beat Flamingo-80B on zero-shot VQAv2 with a small Q-Former between frozen networks.
In April 2023, LLaVA used a text-only GPT-4 to write image-grounded instruction data and trained a linear connector and an open language model in under a day on one 8-GPU machine, and many open models followed the recipe.
POPE, in May 2023, measured how often such models claim objects that are not there.
LLaVA-1.5, in October 2023, made the recipe a common baseline with an MLP connector, higher resolution and public academic data.

## References
